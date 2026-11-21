"""Train the LightGBM model on weekly snapshots, validate, and/or write the submission.

    # development: train on label weeks before 11/14, validate on 11/14 - 11/21 (TW)
    python src/train.py dev --db ../data_final/db

    # ablation: drop feature groups (T temporal, G global, R relational)
    python src/train.py dev --db ../data_final/db --groups T G

    # final: data up to 11/21 00:00 TW, predict 11/22 - 11/29 TW
    python src/train.py final --db ../data_final/db --out output_314551179.csv

How it works
  1. Snapshots: weekly cutoffs c; X = features at c, y = activity in [c + gap, c + gap + 7d).
     Only cohort users with at least one activity before c are training rows; users with no
     history are predicted 0 (they are almost never active, see the README).
  2. One LightGBM binary classifier per activity on all snapshots stacked.
  3. Threshold per activity: fit on all but the newest training week, choose the probability
     threshold that maximises F1 on the newest week, then refit on all weeks.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DEV, FINAL, snapshot, train_snapshots  # noqa: E402
from data import ACTS, build_events, events_before, labels, load_db  # noqa: E402
from features import GROUPS, build_features  # noqa: E402
from metrics import scores  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEST_INPUT = ROOT / "BigData2026-Assignment2-main" / "submission" / "test_input.csv"
SEED = 2026

# small trees: chosen on the mean score of 3 validation weeks (results/experiments.csv, hp_*)
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_child_samples=80,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=150, verbose=-1, random_state=SEED, deterministic=True,
              force_col_wise=True, n_jobs=4)


class Data:
    def __init__(self, db_dir):
        self.db = load_db(db_dir)
        self.ev = build_events(self.db)
        U = self.db["Users"]
        self.cohort = U.loc[U["is_cohort"], "user_id"].to_numpy()
        self.gid2uid = dict(zip(U["github_user_id"], U["user_id"]))

    def active_before(self, c):
        past = events_before(self.ev, c)
        seen = past.loc[past["uid"].isin(self.cohort), "uid"].unique()
        return np.sort(seen)

    def last7(self, users, c):
        """0/1 per user and activity: did it in the 7 days before c (rows known before c)."""
        return labels(events_before(self.ev, c), users, c - pd.Timedelta(days=7), c)

    drop = ()      # feature-name prefixes removed for ablations

    def features(self, users, c, groups):
        X = build_features(self.db, self.ev, users, c, self.cohort, groups)
        return X.loc[:, [col for col in X.columns if not col.startswith(tuple(self.drop))]]

    def snapshot_xy(self, s, groups):
        users = self.active_before(s["cutoff"])
        X = self.features(users, s["cutoff"], groups)
        y = labels(self.ev, users, s["label_start"], s["label_end"])
        return X, y


def best_threshold(p, y):
    best_t, best_f = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.999, 300))):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        f = 2 * tp / (pred.sum() + (y == 1).sum() + 1e-9)
        if f > best_f:
            best_t, best_f = t, f
    return float(best_t), float(best_f)


def best_ratio(p, y, last7):
    """Best number of predicted positives, as a multiple of the users who did the activity
    in the 7 days before the cutoff (adapts to how complete the data is)."""
    n_prev = max(int((last7 > 0).sum()), 1)
    order = np.argsort(-p)
    hits = np.cumsum(y[order] == 1)
    k = np.arange(1, len(p) + 1)
    f = 2 * hits / (k + (y == 1).sum())
    return float((np.argmax(f) + 1) / n_prev)


def fit(X, y, weight=None):
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(X, y, sample_weight=weight)
    return m


def train_models(data, snaps, groups, recency=1.0, label_mask=None, refit=True):
    """snaps newest first. Returns {act: (model, threshold)} and training info."""
    xy = [data.snapshot_xy(s, groups) for s in snaps]
    info = {"snapshots": [{"cutoff": str(s["cutoff"]), "label_start": str(s["label_start"]),
                           "rows": len(x), "pos_rate": y.mean().round(4).to_dict()}
                          for s, (x, y) in zip(snaps, xy)]}
    models = {}
    for a in ACTS:
        keep = [i for i in range(len(snaps)) if not (label_mask and label_mask(snaps[i], a))]
        if len(keep) < 2:
            keep = list(range(len(snaps)))
        # threshold: train on older weeks, tune on the newest kept week
        newest, older = keep[0], keep[1:]
        Xo = pd.concat([xy[i][0] for i in older]); yo = pd.concat([xy[i][1][a] for i in older])
        wo = np.concatenate([np.full(len(xy[i][0]), recency ** k) for k, i in enumerate(older)])
        m_inner = fit(Xo, yo, wo)
        p = m_inner.predict_proba(xy[newest][0])[:, 1]
        thr, f_inner = best_threshold(p, xy[newest][1][a].to_numpy())
        ratio = best_ratio(p, xy[newest][1][a].to_numpy(), data.last7(xy[newest][0].index.to_numpy(), snaps[newest]["cutoff"])[a].to_numpy())
        # refit on all kept weeks
        Xa = pd.concat([xy[i][0] for i in keep]); ya = pd.concat([xy[i][1][a] for i in keep])
        wa = np.concatenate([np.full(len(xy[i][0]), recency ** k) for k, i in enumerate(keep)])
        models[a] = (fit(Xa, ya, wa) if refit else m_inner, thr, ratio)
        info.setdefault("inner_f1", {})[a] = round(f_inner, 4)
    return models, info


def predict(data, models, cutoff, gids, groups, thr_mode="prob", push_rule="model"):
    """0/1 predictions for the github ids in `gids` (users with no history -> 0)."""
    users = data.active_before(cutoff)
    X = data.features(users, cutoff, groups)
    out = pd.DataFrame(0, index=pd.Index(gids, name="UserID"), columns=ACTS, dtype="int64")
    prob = pd.DataFrame(0.0, index=out.index, columns=ACTS)
    uid = pd.Series(gids, index=gids).map(data.gid2uid)
    has = uid.notna() & uid.isin(users)
    idx = uid[has].astype("int64").to_numpy()
    prev7 = data.last7(X.index.to_numpy(), cutoff)
    for a, (m, thr, ratio) in models.items():
        p_all = pd.Series(m.predict_proba(X)[:, 1], index=X.index)
        if thr_mode == "rate":
            k = max(1, int(round(ratio * prev7[a].sum())))
            thr = p_all.nlargest(k).iloc[-1]
        p = p_all.loc[idx].to_numpy()
        prob.loc[has[has].index, a] = p
        out.loc[has[has].index, a] = (p >= thr).astype("int64")
    if push_rule == "copy14":
        # push labels depend on GH Archive gaps more than on users: copy the last 14 days
        recent = labels(events_before(data.ev, cutoff), X.index.to_numpy(),
                        cutoff - pd.Timedelta(days=14), cutoff)["push"]
        out.loc[has[has].index, "push"] = recent.loc[idx].to_numpy().astype("int64")
    return out, prob, X


def truth_for(data, gids, s):
    users = data.cohort
    y = labels(data.ev, users, s["label_start"], s["label_end"])
    uid = pd.Series(gids, index=gids).map(data.gid2uid)
    t = pd.DataFrame(0, index=pd.Index(gids, name="UserID"), columns=ACTS, dtype="int64")
    has = uid.notna() & uid.isin(y.index)
    t.loc[has[has].index] = y.loc[uid[has].astype("int64").to_numpy()].to_numpy()
    return t


def push_gap_mask(data, threshold=0.25):
    """Mask a training label (snapshot, activity) when the cohort-wide count of that activity
    in the label week is below `threshold` x its median weekly count (GH Archive gap)."""
    def weekly(s, a):
        e = data.ev[(data.ev["act"] == a) & data.ev["uid"].isin(data.cohort)]
        return ((e["time"] >= s["label_start"]) & (e["time"] < s["label_end"])).sum()
    cache = {}

    def mask(s, a):
        key = (s["label_start"], a)
        if key not in cache:
            ref = data._ref.setdefault(a, np.median([weekly(x, a) for x in data._all_snaps]))
            cache[key] = weekly(s, a) < threshold * ref
        return cache[key]
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dev", "final"])
    ap.add_argument("--db", required=True)
    ap.add_argument("--groups", nargs="+", default=["T", "R"], choices=GROUPS,
                    help="feature groups; G (cohort-wide counts) overfits with few weeks, so off")
    ap.add_argument("--weeks", type=int, default=5, help="number of training weeks")
    ap.add_argument("--gap_days", type=float, default=None, help="override the gap (days)")
    ap.add_argument("--recency", type=float, default=1.0, help="weight of week k is recency**k")
    ap.add_argument("--mask_gaps", action="store_true", help="mask labels of weeks with data gaps")
    ap.add_argument("--thr_mode", choices=["prob", "rate"], default="prob",
                    help="prob: fixed probability threshold; rate: top-k, k = ratio x last-week actives")
    ap.add_argument("--drop", nargs="*", default=["R_nb_", "R_repo_share_"],
                    help="feature-name prefixes to drop (default: neighbour shares, no gain)")
    ap.add_argument("--param", nargs="*", default=[], help="LightGBM params, e.g. num_leaves=15")
    ap.add_argument("--refit", type=int, default=1,
                    help="1: refit on all weeks after tuning thresholds; 0: keep the tuned model")
    ap.add_argument("--baseline", choices=["copy7", "copy14", "copy28"], default=None)
    ap.add_argument("--push_rule", choices=["model", "copy14"], default="model")
    ap.add_argument("--valid_offset", type=int, default=0,
                    help="dev only: move the validation week this many weeks earlier (backtest)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    cfg = DEV if a.mode == "dev" else FINAL
    gap = cfg["gap"] if a.gap_days is None else pd.Timedelta(days=a.gap_days)
    data = Data(a.db)
    data.drop = tuple(a.drop)
    for kv in a.param:
        k, v = kv.split("=")
        PARAMS[k] = type(PARAMS[k])(v) if k in PARAMS else float(v)
    gids = pd.read_csv(TEST_INPUT)["UserID"].to_numpy()

    if a.mode == "dev":
        vstart = cfg["valid_label_start"] - a.valid_offset * pd.Timedelta(days=7)
        anchor, target = vstart, snapshot(vstart, gap)
    else:
        anchor, target = cfg["data_end"], snapshot(cfg["test_label_start"], gap)
    snaps = train_snapshots(anchor, gap, a.weeks)
    if a.baseline:
        days = int(a.baseline.removeprefix("copy"))
        c = target["cutoff"]
        recent = labels(events_before(data.ev, c), data.cohort, c - pd.Timedelta(days=days), c)
        uid = pd.Series(gids, index=gids).map(data.gid2uid)
        pred = pd.DataFrame(0, index=pd.Index(gids, name="UserID"), columns=ACTS, dtype="int64")
        has = uid.notna() & uid.isin(recent.index)
        pred.loc[has[has].index] = recent.loc[uid[has].astype("int64").to_numpy()].to_numpy()
        truth = truth_for(data, gids, target)
        r = scores(pred.to_numpy(), truth.to_numpy())
        r_np = scores(pred.drop(columns="push").to_numpy(), truth.drop(columns="push").to_numpy(),
                      [x for x in ACTS if x != "push"])
        out_dir = ROOT / "solution" / "results"; out_dir.mkdir(exist_ok=True)
        (out_dir / f"{a.tag}.json").write_text(json.dumps({"valid": r, "valid_without_push": r_np}, indent=1))
        print(f"[{a.tag}] VALID score {r['score']:.4f}")
        return
    data._all_snaps, data._ref = snaps, {}
    mask = push_gap_mask(data) if a.mask_gaps else None
    models, info = train_models(data, snaps, a.groups, a.recency, mask, refit=bool(a.refit))
    if mask:
        info["masked"] = [(str(s["label_start"].date()), act) for s in snaps for act in ACTS if mask(s, act)]
    pred, prob, X = predict(data, models, target["cutoff"], gids, a.groups, a.thr_mode, a.push_rule)

    res = {"mode": a.mode, "groups": a.groups, "weeks": len(snaps), "gap_days": gap / pd.Timedelta(days=1),
           "recency": a.recency, "mask_gaps": a.mask_gaps, "n_features": X.shape[1], "drop": a.drop, "params": PARAMS,
           "thr_mode": a.thr_mode, "push_rule": a.push_rule, "valid_offset": a.valid_offset,
           "thresholds": {k: round(v[1], 4) for k, v in models.items()},
           "ratios": {k: round(v[2], 3) for k, v in models.items()}, **info}
    if a.mode == "dev":
        truth = truth_for(data, gids, target)
        r = scores(pred.to_numpy(), truth.to_numpy())
        r_np = scores(pred.drop(columns="push").to_numpy(), truth.drop(columns="push").to_numpy(),
                      [x for x in ACTS if x != "push"])
        res.update(valid=r, valid_without_push=r_np)
        print(f"[{a.tag}] VALID score {r['score']:.4f} macro {r['macro_f1']:.4f} micro {r['micro_f1']:.4f} | "
              + " ".join(f"{k}={v:.3f}" for k, v in r["per_act"].items())
              + f" | w/o push {r_np['score']:.4f}")
    out_dir = ROOT / "solution" / "results"
    out_dir.mkdir(exist_ok=True)
    tag = a.tag or "_".join([a.mode, "".join(a.groups)])
    (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    prob.to_csv(out_dir / f"{tag}_prob.csv")
    if a.mode == "dev":
        imp = pd.DataFrame({k: pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
                            for k, (m, *_) in models.items()})
        imp.to_csv(out_dir / f"{tag}_importance.csv")
    if a.out:
        pred.reset_index().to_csv(a.out, index=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
