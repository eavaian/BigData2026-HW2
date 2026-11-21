"""Extra analyses on cached features: model comparison, finer ablations, threshold study,
error analysis by user segment, and paired-bootstrap confidence intervals.

    python src/study.py --db ../data_final/db            # all parts
    python src/study.py --db ../data_final/db --parts models ablation

Every configuration uses exactly the protocol of train.py (5 weekly snapshots, threshold tuned
on the newest training week, refit on all weeks, push predicted by the model) and the same
3 validation weeks (offsets 0, 1, 2). Features are built once per cutoff and cached in
../study_cache, so a configuration only re-trains the classifiers.

Outputs (results/):
  study.csv            one row per (config, validation week)
  study_summary.csv    mean over the 3 weeks
  study_bootstrap.csv  paired bootstrap of the score difference vs the final model
  study_segments.csv   F1 by user segment (days since last activity at the cutoff)
  study_errors.csv     precision / recall / F1 per activity
  figs/study_models.png, figs/study_ablation.png, figs/study_segments.png
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from config import DEV, snapshot, train_snapshots  # noqa: E402
from data import ACTS  # noqa: E402
from metrics import scores  # noqa: E402
import train as T  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
FIGS = RES / "figs"
OFFSETS = [0, 1, 2]
SEED = 2026
NP_ACTS = [a for a in ACTS if a != "push"]
DROP = ("R_nb_", "R_repo_share_")          # same default as train.py

# ---------------------------------------------------------------- feature groups
GROUPS = {
    "T: window counts": lambda c: c.startswith(("T_cnt1d", "T_cnt3d", "T_cnt7d", "T_cnt14d", "T_cnt28d")),
    "T: all-time counts": lambda c: c.startswith("T_cntall"),
    "T: trend": lambda c: c.startswith("T_trend"),
    "T: decayed counts": lambda c: c.startswith("T_decay"),
    "T: days since": lambda c: c.startswith("T_days_since"),
    "T: activity profile": lambda c: c.startswith(("T_active", "T_mean_gap", "T_weekend", "T_night")),
    "R: own repositories": lambda c: c in {"R_repos28", "R_repos7", "R_repos_all", "R_top_repo_share",
                                           "R_owned_repos", "R_own_repo_frac28", "R_org_repo_frac28"},
    "R: repository activity": lambda c: c.startswith(("R_repo_heat", "R_repo_contrib", "R_repo_open_pr",
                                                      "R_repo_issue_comment", "R_repo_pr_review",
                                                      "R_repo_push")),
    "R: collaborators": lambda c: c.startswith(("R_collab", "R_collaborators")),
    "R: PR / issue state": lambda c: c in {"R_open_prs", "R_open_prs_recent", "R_prs_merged28",
                                           "R_prs_cross_repo_frac", "R_merges_done28", "R_open_issues",
                                           "R_issues_assigned"},
    "R: interactions": lambda c: c in {"R_reviews_received7", "R_changes_requested7",
                                       "R_review_comments_received7", "R_comments_received7",
                                       "R_reviewees28", "R_reviewers28", "R_review_backlog",
                                       "R_comment_on_pr_frac"},
    "R: is bot": lambda c: c == "R_is_bot",
}


def own_activity_only(a, cols):
    """Drop temporal features of the *other* activities (keep totals and relational ones)."""
    others = [x for x in ACTS if x != a]
    return [c for c in cols if not (c.startswith("T_") and any(c.endswith("_" + o) for o in others))]


SHORT_MEMORY = ("T_cnt1d", "T_cnt3d", "T_cnt7d", "T_decay2", "T_active_days7")   # <= 7 days only


# ---------------------------------------------------------------- models
def signed_log(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


class Score:
    """'Model' that only ranks users by one feature (a tuned heuristic, nothing learned)."""
    def __init__(self, col, fixed=None):
        self.col, self.fixed = col, fixed

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        s = X[self.col].to_numpy(dtype=float)
        if self.fixed is not None:          # copy rule: predicted iff the count is > 0
            s = (s > 0).astype(float)
        return np.c_[1 - s, s]


class Average:
    def __init__(self, makers):
        self.makers = makers

    def fit(self, X, y):
        self.ms = [m().fit(X, y) for m in self.makers]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def lgbm(**kw):
    return lambda: lgb.LGBMClassifier(**{**T.PARAMS, **kw})


# scikit-learn's compiled modules are blocked by Windows application control on our machine,
# so logistic regression and the MLP are written in numpy, and the random forest is
# LightGBM's random-forest mode (bagging + feature subsampling, no boosting).
class Scaled:
    """signed log1p, then standardise with the training mean / std."""
    def _prep(self, X, fit=False):
        Z = signed_log(X)
        if fit:
            self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        return (Z - self.mu) / self.sd


class LogReg(Scaled):
    """L2-regularised logistic regression fitted with Newton's method (IRLS)."""
    def __init__(self, l2=10.0, iters=25):
        self.l2, self.iters = l2, iters

    def fit(self, X, y):
        Z = self._prep(X, fit=True); Z = np.c_[np.ones(len(Z)), Z]; y = np.asarray(y, float)
        w = np.zeros(Z.shape[1]); R = self.l2 * np.eye(Z.shape[1]); R[0, 0] = 0
        for _ in range(self.iters):
            p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
            g = Z.T @ (p - y) + R @ w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + R
            step = np.linalg.solve(H, g); w -= step
            if np.abs(step).max() < 1e-6:
                break
        self.w = w
        return self

    def predict_proba(self, X):
        Z = np.c_[np.ones(len(X)), self._prep(X)]
        p = 1 / (1 + np.exp(-np.clip(Z @ self.w, -30, 30)))
        return np.c_[1 - p, p]


class MLP(Scaled):
    """2 hidden ReLU layers, Adam, mini-batches, early stopping on a 10% holdout."""
    def __init__(self, hidden=(64, 32), lr=1e-3, l2=1e-4, epochs=60, batch=256, patience=6):
        self.h, self.lr, self.l2, self.epochs, self.batch, self.patience = hidden, lr, l2, epochs, batch, patience

    def _forward(self, Z, W):
        acts = [Z]
        for k, (A, b) in enumerate(W):
            Z = Z @ A + b
            if k < len(W) - 1:
                Z = np.maximum(Z, 0)
            acts.append(Z)
        return acts

    def fit(self, X, y):
        rng = np.random.default_rng(SEED)
        Z = self._prep(X, fit=True); y = np.asarray(y, float)
        idx = rng.permutation(len(Z)); nv = len(Z) // 10
        va, tr = idx[:nv], idx[nv:]
        sizes = [Z.shape[1], *self.h, 1]
        W = [[rng.normal(0, np.sqrt(2 / i), (i, o)), np.zeros(o)] for i, o in zip(sizes[:-1], sizes[1:])]
        m = [[np.zeros_like(A), np.zeros_like(b)] for A, b in W]; v = [[np.zeros_like(A), np.zeros_like(b)] for A, b in W]
        best, best_W, bad, t = np.inf, None, 0, 0

        def loss(ix):
            o = self._forward(Z[ix], W)[-1][:, 0]
            return np.mean(np.logaddexp(0, o) - y[ix] * o)
        for _ in range(self.epochs):
            rng.shuffle(tr)
            for s in range(0, len(tr), self.batch):
                ix = tr[s:s + self.batch]; acts = self._forward(Z[ix], W)
                d = (1 / (1 + np.exp(-acts[-1][:, 0])) - y[ix])[:, None] / len(ix)
                for k in range(len(W) - 1, -1, -1):
                    gA = acts[k].T @ d + self.l2 * W[k][0]; gb = d.sum(0)
                    if k:
                        d = (d @ W[k][0].T) * (acts[k] > 0)
                    t += 1
                    for q, g in enumerate([gA, gb]):
                        m[k][q] = 0.9 * m[k][q] + 0.1 * g; v[k][q] = 0.999 * v[k][q] + 0.001 * g * g
                        W[k][q] -= self.lr * (m[k][q] / (1 - 0.9 ** (t / len(W)))) / (
                            np.sqrt(v[k][q] / (1 - 0.999 ** (t / len(W)))) + 1e-8)
            lv = loss(va)
            if lv < best - 1e-5:
                best, best_W, bad = lv, [[A.copy(), b.copy()] for A, b in W], 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.W = best_W or W
        return self

    def predict_proba(self, X):
        o = self._forward(self._prep(X), self.W)[-1][:, 0]
        p = 1 / (1 + np.exp(-np.clip(o, -30, 30)))
        return np.c_[1 - p, p]


def logreg():
    return LogReg()


def rf():
    return lgb.LGBMClassifier(boosting_type="rf", n_estimators=300, num_leaves=63, min_child_samples=20,
                              bagging_fraction=0.632, bagging_freq=1, feature_fraction=0.33,
                              verbose=-1, random_state=SEED, deterministic=True, n_jobs=4)


def mlp():
    return MLP()


# ---------------------------------------------------------------- data cache
def key(s):
    return str(s["cutoff"])


def weeks_for(offset, n_weeks=5):
    vstart = DEV["valid_label_start"] - offset * pd.Timedelta(days=7)
    return train_snapshots(vstart, DEV["gap"], n_weeks), snapshot(vstart, DEV["gap"])


def load(db, cache_dir):
    cache_dir = Path(cache_dir); cache_dir.mkdir(exist_ok=True)
    data = T.Data(db); data.drop = DROP
    gids = pd.read_csv(T.TEST_INPUT)["UserID"].to_numpy()
    snaps = {}
    for off in OFFSETS:
        tr, tg = weeks_for(off)
        for s in tr + [tg]:
            snaps[key(s)] = s
    xy = {}
    for k, s in sorted(snaps.items()):
        f = cache_dir / (k.replace(":", "").replace(" ", "_") + ".pkl")
        if f.exists():
            xy[k] = pickle.loads(f.read_bytes())
        else:
            t0 = time.time()
            xy[k] = data.snapshot_xy(s, ["T", "R"])
            f.write_bytes(pickle.dumps(xy[k]))
            print(f"  features at {k}: {xy[k][0].shape} ({time.time() - t0:.0f}s)", flush=True)
    weeks = {}
    for off in OFFSETS:
        tr, tg = weeks_for(off)
        X_t = xy[key(tg)][0]
        uid = pd.Series(gids).map(data.gid2uid)
        pos = X_t.index.get_indexer(uid.fillna(-1).astype("int64"))
        truth = T.truth_for(data, gids, tg).to_numpy()
        weeks[off] = dict(train=[xy[key(s)] for s in tr], X_t=X_t, pos=pos, truth=truth)
    return data, gids, weeks


# ---------------------------------------------------------------- one configuration
def threshold(p, y, mode):
    if mode == "tuned":
        return T.best_threshold(p, y)[0]
    if mode == "fixed0.5":
        return 0.5
    if mode == "prior":                 # predict as many positives as the newest week had
        k = max(int(y.sum()), 1)
        return float(np.sort(p)[::-1][k - 1])
    raise ValueError(mode)


def run(cfg, wk, n_weeks=5):
    """cfg: dict(make=act->factory, cols=(act, cols)->cols, thr='tuned', refit=True).
    Returns 0/1 predictions and probabilities for the gids of the validation week."""
    xy = wk["train"][:n_weeks]
    all_cols = list(wk["X_t"].columns)
    pred = np.zeros((len(wk["pos"]), len(ACTS)), dtype=int)
    prob = np.zeros(pred.shape)
    has = wk["pos"] >= 0
    for j, a in enumerate(ACTS):
        cols = cfg.get("cols", lambda a, c: c)(a, all_cols)
        make = cfg["make"](a)
        Xo = pd.concat([x[cols] for x, _ in xy[1:]]); yo = pd.concat([y[a] for _, y in xy[1:]])
        m = make().fit(Xo, yo)
        p = m.predict_proba(xy[0][0][cols])[:, 1]
        thr = threshold(p, xy[0][1][a].to_numpy(), cfg.get("thr", "tuned"))
        if cfg.get("refit", True):
            Xa = pd.concat([x[cols] for x, _ in xy]); ya = pd.concat([y[a] for _, y in xy])
            m = make().fit(Xa, ya)
        pt = m.predict_proba(wk["X_t"][cols])[:, 1]
        prob[has, j] = pt[wk["pos"][has]]
        pred[has, j] = (prob[has, j] >= thr).astype(int)
    return pred, prob


def evaluate(pred, truth):
    r = scores(pred, truth)
    keep = [ACTS.index(a) for a in NP_ACTS]
    r_np = scores(pred[:, keep], truth[:, keep], NP_ACTS)
    return {"score": r["score"], "macro": r["macro_f1"], "micro": r["micro_f1"],
            "score_wo_push": r_np["score"], **r["per_act"]}


# ---------------------------------------------------------------- configurations
def drop_group(g):
    return lambda a, cols: [c for c in cols if not GROUPS[g](c)]


def only(prefixes):
    return lambda a, cols: [c for c in cols if c.startswith(prefixes)]


FINAL = dict(make=lambda a: lgbm())


def model_configs():
    return {
        "Copy last 7 days": dict(make=lambda a: (lambda: Score(f"T_cnt7d_{a}", fixed=True)), thr="fixed0.5"),
        "Copy last 14 days": dict(make=lambda a: (lambda: Score(f"T_cnt14d_{a}", fixed=True)), thr="fixed0.5"),
        "Decayed count, tuned threshold": dict(make=lambda a: (lambda: Score(f"T_decay7_{a}"))),
        "Logistic regression": dict(make=lambda a: logreg),
        "MLP (64-32)": dict(make=lambda a: mlp),
        "Random forest": dict(make=lambda a: rf),
        "LightGBM (final)": FINAL,
        "LightGBM, 31 leaves x 500 trees": dict(make=lambda a: lgbm(num_leaves=31, n_estimators=500,
                                                                   min_child_samples=20)),
        "LightGBM, balanced class weights": dict(make=lambda a: lgbm(class_weight="balanced")),
        "Average of LightGBM + LR + RF": dict(make=lambda a: (lambda: Average([lgbm(), logreg, rf]))),
    }


def ablation_configs():
    c = {"Full (T + R)": FINAL}
    for g in GROUPS:
        c[f"- {g}"] = dict(make=lambda a: lgbm(), cols=drop_group(g))
    c["- all T (R only)"] = dict(make=lambda a: lgbm(), cols=only(("R_",)))
    c["- all R (T only)"] = dict(make=lambda a: lgbm(), cols=only(("T_",)))
    c["Own-activity temporal features only"] = dict(make=lambda a: lgbm(), cols=own_activity_only)
    c["Short memory: last 7 days only (no R)"] = dict(make=lambda a: lgbm(), cols=only(SHORT_MEMORY))
    c["Short memory: last 7 days + R"] = dict(make=lambda a: lgbm(), cols=only(SHORT_MEMORY + ("R_",)))
    return c


def threshold_configs():
    return {"Threshold tuned on newest week (final)": FINAL,
            "Fixed threshold 0.5": dict(make=lambda a: lgbm(), thr="fixed0.5"),
            "Top-k, k = positives in newest week": dict(make=lambda a: lgbm(), thr="prior"),
            "Balanced weights + fixed 0.5": dict(make=lambda a: lgbm(class_weight="balanced"), thr="fixed0.5")}


def history_configs():
    return {f"{n} training week{'s' if n > 1 else ''}": dict(make=lambda a: lgbm(), n_weeks=n)
            for n in [2, 3, 4, 5]}


# ---------------------------------------------------------------- analyses
def _sc(p, t, wo_push):
    if not wo_push:
        return scores(p, t)["score"]
    keep = [ACTS.index(a) for a in NP_ACTS]
    return scores(p[:, keep], t[:, keep], NP_ACTS)["score"]


def bootstrap(preds, weeks, ref, n=1000):
    """Paired bootstrap over test users (same resample for both configs): 95% CI of the
    3-week mean of score(config) - score(ref), for the full score and the score w/o push."""
    rows = []
    for name in preds:
        if name == ref:
            continue
        row = {"config": name, "vs": ref}
        for wo, tag in [(False, ""), (True, "_wo_push")]:
            rng = np.random.default_rng(SEED)
            diffs = np.zeros((n, len(OFFSETS)))
            for j, off in enumerate(OFFSETS):
                t, p0, p1 = weeks[off]["truth"], preds[ref][off], preds[name][off]
                for b in range(n):
                    i = rng.integers(0, len(t), len(t))
                    diffs[b, j] = _sc(p1[i], t[i], wo) - _sc(p0[i], t[i], wo)
            mean = diffs.mean(1)
            row.update({f"diff{tag}": float(np.mean([_sc(preds[name][o], weeks[o]["truth"], wo)
                                                     - _sc(preds[ref][o], weeks[o]["truth"], wo)
                                                     for o in OFFSETS])),
                        f"ci_low{tag}": float(np.quantile(mean, 0.025)),
                        f"ci_high{tag}": float(np.quantile(mean, 0.975)),
                        f"p_not_better{tag}": float((mean <= 0).mean())})
        rows.append(row)
    return pd.DataFrame(rows)


SEGMENTS = [("no history", None), ("active < 1 day ago", (0, 1)), ("1-7 days ago", (1, 7)),
            ("7-28 days ago", (7, 28)), ("> 28 days ago", (28, 1e9))]


def segments(preds, weeks, names):
    rows = []
    for off in OFFSETS:
        wk = weeks[off]
        ds = np.full(len(wk["pos"]), np.nan)
        has = wk["pos"] >= 0
        ds[has] = wk["X_t"]["T_days_since_any"].to_numpy()[wk["pos"][has]]
        for seg, rng_ in SEGMENTS:
            m = ~has if rng_ is None else (ds >= rng_[0]) & (ds < rng_[1])
            t = wk["truth"][m]
            for name in names:
                p = preds[name][off][m]
                tp = ((p == 1) & (t == 1)).sum(); fp = ((p == 1) & (t == 0)).sum(); fn = ((p == 0) & (t == 1)).sum()
                rows.append({"week": off, "segment": seg, "config": name, "users": int(m.sum()),
                             "positives": int(t.sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn)})
    d = pd.DataFrame(rows).groupby(["segment", "config"], sort=False)[["users", "positives", "tp", "fp", "fn"]].sum()
    d["users"] //= len(OFFSETS)
    d["share_of_positives"] = d["positives"] / d.groupby(level="config")["positives"].transform("sum")
    d["precision"] = d["tp"] / (d["tp"] + d["fp"]).replace(0, np.nan)
    d["recall"] = d["tp"] / d["positives"].replace(0, np.nan)
    d["f1"] = 2 * d["tp"] / (2 * d["tp"] + d["fp"] + d["fn"]).replace(0, np.nan)
    return d.reset_index()


def errors(preds, weeks, names):
    rows = []
    for name in names:
        for j, a in enumerate(ACTS):
            tp = fp = fn = 0
            for off in OFFSETS:
                p, t = preds[name][off][:, j], weeks[off]["truth"][:, j]
                tp += ((p == 1) & (t == 1)).sum(); fp += ((p == 1) & (t == 0)).sum(); fn += ((p == 0) & (t == 1)).sum()
            rows.append({"config": name, "activity": a, "predicted": int(tp + fp), "actual": int(tp + fn),
                         "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                         "f1": 2 * tp / max(2 * tp + fp + fn, 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2 = "#0b0b0b", "#52514e"


def _style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c9c8c2"); ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)


def fig_models(summary, path, highlight="LightGBM (final)"):
    d = summary.sort_values("score_wo_push")
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(d) + 1.3), sharey=True)
    for ax, col, title in [(axes[0], "score", "Score, 6 activities"),
                           (axes[1], "score_wo_push", "Score without push")]:
        colors = [ORANGE if n == highlight else BLUE for n in d["config"]]
        ax.barh(d["config"], d[col], color=colors, height=0.62, edgecolor="white", linewidth=2)
        for y, v in enumerate(d[col]):
            ax.text(v + 0.3, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(max(0, d[[ "score", "score_wo_push"]].min().min() - 6), d[col].max() + 3)
        ax.set_title(title, fontsize=11, color=INK, loc="left"); _style(ax)
    fig.suptitle("Model comparison (mean of 3 validation weeks, x100)", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def fig_ablation(summary, path, ref="Full (T + R)"):
    base = summary.set_index("config").loc[ref, "score_wo_push"]
    d = summary[summary["config"] != ref].assign(delta=lambda x: x["score_wo_push"] - base).sort_values("delta")
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(d) + 1.3))
    colors = [BLUE if v < 0 else GRAY for v in d["delta"]]
    ax.barh(d["config"], d["delta"], color=colors, height=0.62, edgecolor="white", linewidth=2)
    for y, v in enumerate(d["delta"]):
        ax.text(v + (0.08 if v >= 0 else -0.08), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax.axvline(0, color=INK2, lw=1)
    lo, hi = d["delta"].min(), d["delta"].max()
    ax.set_xlim(lo - 1.2, max(hi, 0) + 1.2)
    ax.set_xlabel(f"Change in score without push vs full model ({base:.2f})", color=INK2, fontsize=9)
    ax.set_title("Feature ablation (LightGBM, mean of 3 validation weeks)", fontsize=12, color=INK, loc="left")
    _style(ax); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


LABEL = {"- all R (T only)": "LightGBM, temporal only", "LightGBM (final)": "LightGBM (features)",
         "RDL (time enc.)": "RDL, 8 tables + time encoding (main)"}

# saved by rdl.py (dev mode): results/rdl_pred/<run>_o<week>.csv, one row per test user
RDL_RUNS = {"rdl_time1": "RDL (time enc.)", "rdl_time0": "RDL, no time encoding",
            "rdl_layers1": "RDL, 1 layer", "rdl_no_pushes": "RDL, without Pushes",
            "rdl_no_reviews": "RDL, without PRReviews + PRReviewComments",
            "rdl_no_issues": "RDL, without Issues + IssueComments"}
RDL_MAIN = "RDL (time enc.)"


def load_rdl_preds(gids):
    out = {}
    for run_, name in RDL_RUNS.items():
        files = [RES / "rdl_pred" / f"{run_}_o{o}.csv" for o in OFFSETS]
        if all(f.exists() for f in files):
            out[name] = {o: pd.read_csv(f).set_index("UserID").loc[gids, ACTS].to_numpy()
                         for o, f in zip(OFFSETS, files)}
    return out


def fig_segments(seg, path, names):
    segs = [s for s, _ in SEGMENTS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    s0 = seg[seg["config"] == names[0]].set_index("segment").loc[segs]
    axes[0].bar(segs, s0["share_of_positives"] * 100, color=BLUE, width=0.6, edgecolor="white", linewidth=2)
    for x, v in enumerate(s0["share_of_positives"] * 100):
        axes[0].text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK)
    axes[0].set_title("Share of all true (user, activity) positives", fontsize=11, color=INK, loc="left")
    w = 0.8 / len(names)
    for k, (name, col) in enumerate(zip(names, [GRAY, ORANGE, BLUE])):
        s = seg[seg["config"] == name].set_index("segment").loc[segs]
        xs = np.arange(len(segs)) + (k - (len(names) - 1) / 2) * w
        axes[1].bar(xs, s["f1"].fillna(0) * 100, width=w, color=col, label=LABEL.get(name, name), edgecolor="white", linewidth=2)
    short = ["no history", "< 1 day", "1-7 days", "7-28 days", "> 28 days"]
    axes[0].set_xticks(range(len(segs)), short)
    axes[1].set_xticks(range(len(segs)), short)
    axes[1].set_title("F1 (pooled over activities) by segment", fontsize=11, color=INK, loc="left")
    axes[1].legend(frameon=False, fontsize=8.5)
    for ax in axes:
        _style(ax); ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.suptitle("Users grouped by days since their last activity before the cutoff (3 validation weeks)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--cache", default=str(Path(__file__).resolve().parents[2] / "study_cache"))
    ap.add_argument("--parts", nargs="+", default=["models", "ablation", "threshold", "history", "analysis"])
    a = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    print("loading / building features ...", flush=True)
    data, gids, weeks = load(a.db, a.cache)

    path = RES / "study.csv"
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["part", "config"])
    preds = {}
    parts = {"models": model_configs, "ablation": ablation_configs,
             "threshold": threshold_configs, "history": history_configs}
    for part in [p for p in a.parts if p in parts]:
        rows = []
        for name, cfg in parts[part]().items():
            t0 = time.time()
            preds[name] = {}
            for off in OFFSETS:
                p, _ = run(cfg, weeks[off], cfg.get("n_weeks", 5))
                preds[name][off] = p
                rows.append({"part": part, "config": name, "valid_week": off, **evaluate(p, weeks[off]["truth"])})
            m = pd.DataFrame(rows[-3:])[["score", "score_wo_push"]].mean() * 100
            print(f"[{part}] {name:45s} score {m['score']:.2f}  w/o push {m['score_wo_push']:.2f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        old = pd.concat([old[old["part"] != part], pd.DataFrame(rows)])
        old.to_csv(path, index=False)
    if preds:
        pp = Path(a.cache) / "preds.pkl"
        allp = pickle.loads(pp.read_bytes()) if pp.exists() else {}
        allp.update(preds)
        pp.write_bytes(pickle.dumps(allp))

    allres = pd.read_csv(path)
    summ = (allres.groupby(["part", "config"], sort=False)
            .mean(numeric_only=True).drop(columns="valid_week").mul(100).round(2).reset_index())
    summ.to_csv(RES / "study_summary.csv", index=False)
    if "models" in summ["part"].values:
        fig_models(summ[summ["part"] == "models"], FIGS / "study_models.png")
    if "ablation" in summ["part"].values:
        fig_ablation(summ[summ["part"] == "ablation"], FIGS / "study_ablation.png")

    if "analysis" in a.parts:
        pp = Path(a.cache) / "preds.pkl"
        allp = pickle.loads(pp.read_bytes()) if pp.exists() else {}
        allp.update(preds)
        need = {"Copy last 7 days": model_configs()["Copy last 7 days"], "LightGBM (final)": FINAL,
                "- all R (T only)": ablation_configs()["- all R (T only)"],
                "Logistic regression": model_configs()["Logistic regression"]}
        for name, cfg in need.items():
            if name not in allp:
                allp[name] = {off: run(cfg, weeks[off])[0] for off in OFFSETS}
        pickle.dump(allp, open(pp, "wb"))
        cmp = [n for n in model_configs() if n in allp] + \
              [n for n in ["- all R (T only)", "- all T (R only)", "Fixed threshold 0.5"] if n in allp]
        rdl = load_rdl_preds(gids)
        allp.update(rdl)
        cmp += [n for n in rdl if n != RDL_MAIN]
        if RDL_MAIN in rdl:
            cmp.append(RDL_MAIN)
        bs = bootstrap({n: allp[n] for n in cmp}, weeks, "LightGBM (final)")
        bs.to_csv(RES / "study_bootstrap.csv", index=False)
        print(bs.round(4).to_string(index=False))
        if RDL_MAIN in rdl:      # the RDL variants and LightGBM against the main RDL model
            bs2 = bootstrap({n: allp[n] for n in [RDL_MAIN, *[n for n in rdl if n != RDL_MAIN],
                                                     "LightGBM (final)", "Copy last 7 days"]},
                            weeks, RDL_MAIN)
            bs2.to_csv(RES / "study_bootstrap_rdl.csv", index=False)
            print(bs2.round(4).to_string(index=False))
        seg_names = ["Copy last 7 days", "LightGBM (final)"] + ([RDL_MAIN] if RDL_MAIN in rdl else [])
        seg = segments(allp, weeks, seg_names)
        seg.to_csv(RES / "study_segments.csv", index=False)
        print(seg.round(3).to_string(index=False))
        fig_segments(seg, FIGS / "study_segments.png", seg_names)
        err = errors(allp, weeks, seg_names)
        err.to_csv(RES / "study_errors.csv", index=False)
        print(err.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
