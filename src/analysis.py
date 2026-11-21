"""Figures and extra analyses for the report.

    python src/analysis.py --db ../data_final/db

Writes results/figs/*.png and results/leakage.json:
  daily_activity.png   cohort events per day and activity (GH Archive gaps are visible)
  weekly_pos_rate.png  share of active users who do each activity, per label week
  importance.png       top LightGBM features (gain), coloured by feature group
  results_bar.png      validation score of baselines, ablations, GNNs and the final model
  leakage.json         random split vs. time split (why the split must be by time)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from config import DEV, DATA_START, snapshot, train_snapshots  # noqa: E402
from data import ACTS, labels  # noqa: E402
from metrics import scores  # noqa: E402
import train as T  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
FIG = RES / "figs"
COLORS = dict(zip(ACTS, ["#2a6fdb", "#e07b24", "#3a9e5f", "#c43c5c", "#7d5ab5", "#7a7a7a"]))


def daily_activity(data):
    ev = data.ev[(data.ev["time"] >= DATA_START) & data.ev["uid"].isin(data.cohort)]
    d = ev.assign(day=(ev["time"] + pd.Timedelta(hours=8)).dt.floor("D")) \
        .pivot_table(index="day", columns="act", values="uid", aggfunc="size", fill_value=0, observed=False)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(d.index, d["push"], color=COLORS["push"])
    axes[0].set_ylabel("push / day")
    axes[0].set_title("Cohort activity per day (Taiwan time): push falls from ~2,000-3,000 to 0-300 per day after Oct 27 (GH Archive gaps)")
    for a in ACTS[:-1]:
        axes[1].plot(d.index, d[a], label=a, color=COLORS[a])
    axes[1].set_ylabel("events / day")
    axes[1].legend(ncol=5, fontsize=8)
    v = DEV["valid_label_start"] + pd.Timedelta(hours=8)
    for ax in axes:
        ax.axvspan(v, v + pd.Timedelta(days=7), color="#f2d16b", alpha=0.3)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "daily_activity.png", dpi=150)
    d.to_csv(RES / "daily_activity.csv")


def weekly_pos_rate(data):
    snaps = train_snapshots(DEV["valid_label_start"] + pd.Timedelta(days=7), DEV["gap"], 6)[::-1]
    rows = []
    for s in snaps:
        users = data.active_before(s["cutoff"])
        y = labels(data.ev, users, s["label_start"], s["label_end"]).mean()
        rows.append({"week": (s["label_start"] + pd.Timedelta(hours=8)).strftime("%m/%d"), **y.to_dict()})
    df = pd.DataFrame(rows).set_index("week")
    df.to_csv(RES / "weekly_pos_rate.csv")
    ax = df.plot(marker="o", figsize=(8, 4), color=[COLORS[a] for a in ACTS])
    ax.set_ylabel("share of users with history")
    ax.set_title("Positive rate per label week (week starts, TW)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "weekly_pos_rate.png", dpi=150)


def importance():
    imp = pd.read_csv(RES / "exp_final_config_o0_importance.csv", index_col=0)
    tot = (imp / imp.sum()).mean(1).sort_values(ascending=False).head(20)[::-1]
    col = ["#2a6fdb" if n.startswith("T_") else "#e07b24" for n in tot.index]
    plt.figure(figsize=(8, 6))
    plt.barh(tot.index, tot.values, color=col)
    plt.title("Top 20 features (mean normalised gain over 6 activities)\nblue = temporal, orange = relational")
    plt.tight_layout()
    plt.savefig(FIG / "importance.png", dpi=150)
    grp = (imp / imp.sum()).groupby(lambda n: n.split("_")[0]).sum()
    grp.to_csv(RES / "importance_by_group.csv")


def results_bar():
    d = pd.read_csv(RES / "experiments.csv")
    names = {"base_copy7": "Copy last week", "base_copy14": "Copy last 14 days",
             "gnn_time0": "GNN user-repo, no time enc.", "gnn_time1": "GNN user-repo + time enc.",
             "abl_RG_noT": "LGBM no temporal feats", "abl_T": "LGBM temporal only",
             "final_config": "LGBM temporal + relational (supporting)",
             "rdl_time0": "RDL 8 tables, no time enc.", "rdl_time1": "RDL 8 tables + time enc. (main)"}
    m = d[d["config"].isin(names)].groupby("config")[["score", "score_wo_push"]].mean() \
        .reindex([k for k in names if k in set(d["config"])])
    m.index = [names[k] for k in m.index]
    m.to_csv(RES / "results_summary.csv")
    ax = m.plot.barh(figsize=(9, 4.5), color=["#7a7a7a", "#2a6fdb"])
    ax.set_xlabel("Score = (macro-F1 + micro-F1)/2, mean of 3 validation weeks")
    ax.legend(["all 6 activities", "without push (GH Archive gaps)"], fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(FIG / "results_bar.png", dpi=150)


RDL_RUNS = {"rdl_time1": "RDL main (8 tables, 2 layers, time enc.)",
            "rdl_time0": "no time encoding", "rdl_layers1": "1 layer (1 hop)",
            "rdl_no_pushes": "without Pushes",
            "rdl_no_reviews": "without PRReviews + PRReviewComments",
            "rdl_no_issues": "without Issues + IssueComments"}


def rdl_ablation():
    """RDL variants (rdl.py dev runs), mean of the 3 validation weeks."""
    d = pd.read_csv(RES / "experiments.csv")
    d = d[d["config"].isin(RDL_RUNS)].groupby("config")[["score", "score_wo_push"]].mean() * 100
    d = d.reindex([k for k in RDL_RUNS if k in d.index])
    d.index = [RDL_RUNS[k] for k in d.index]
    d.to_csv(RES / "rdl_ablation.csv")
    d = d.iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, col, title in [(axes[0], "score", "Score, 6 activities"),
                           (axes[1], "score_wo_push", "Score without push")]:
        colors = ["#eb6834" if n.startswith("RDL main") else "#2a78d6" for n in d.index]
        ax.barh(d.index, d[col], color=colors, height=0.62, edgecolor="white", linewidth=2)
        for y, v in enumerate(d[col]):
            ax.text(v + 0.2, y, f"{v:.2f}", va="center", fontsize=8.5)
        ax.set_xlim(25, d[col].max() + 3)
        ax.set_title(title, fontsize=11, loc="left")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.grid(axis="x", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("RDL ablation (mean of 3 validation weeks, x100)", fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "rdl_ablation.png", dpi=160)
    plt.close(fig)


def leakage(data):
    """Same features and model, two ways of splitting train / validation rows."""
    snaps = train_snapshots(DEV["valid_label_start"], DEV["gap"], 5)
    target = snapshot(DEV["valid_label_start"], DEV["gap"])
    xy = [data.snapshot_xy(s, ["T", "R"]) for s in [target] + snaps]
    X = pd.concat([x for x, _ in xy]); Y = pd.concat([y for _, y in xy])
    week = np.concatenate([np.full(len(x), i) for i, (x, _) in enumerate(xy)])
    rng = np.random.default_rng(0)
    out = {}
    for name, test_mask in (("time_split", week == 0),
                            ("random_split", rng.random(len(X)) < (week == 0).mean())):
        pred = np.zeros((test_mask.sum(), len(ACTS)), dtype=int)
        for k, a in enumerate(ACTS):
            m = T.fit(X[~test_mask], Y[a][~test_mask])
            p_tr = m.predict_proba(X[~test_mask])[:, 1]
            thr, _ = T.best_threshold(p_tr, Y[a][~test_mask].to_numpy())
            pred[:, k] = m.predict_proba(X[test_mask])[:, 1] >= thr
        r = scores(pred, Y[test_mask].to_numpy())
        out[name] = {"score": r["score"], "per_act": r["per_act"]}
        print(name, round(r["score"], 4))
    (RES / "leakage.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    data = T.Data(a.db)
    data.drop = ("R_nb_", "R_repo_share_")
    jobs = {"daily": lambda: daily_activity(data), "weekly": lambda: weekly_pos_rate(data),
            "importance": importance, "bar": results_bar, "rdl": rdl_ablation, "leakage": lambda: leakage(data)}
    for k, f in jobs.items():
        if a.only is None or k in a.only:
            f()
