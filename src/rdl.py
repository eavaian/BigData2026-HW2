"""Relational Deep Learning (Lecture 5 / RelBench) on all 8 tables.

    # validation on 3 weeks (needs the RDL environment, see README)
    python src/rdl.py --db ../data_final/db --time_enc 1 --epochs 25
    # ablations: 1 GNN layer, graph without some tables
    python src/rdl.py --db ../data_final/db --layers 1
    python src/rdl.py --db ../data_final/db --drop_tables Pushes
    # submission: data up to 11/21 00:00 TW, predict 11/22 - 11/29 TW
    python src/rdl.py --mode final --db ../data_final/db --out output_314551179.csv

Pipeline (as in the RelBench GNN tutorial):
  1. Database      the 8 tables as relbench Tables (primary key, foreign keys, row-time column).
                   Columns that are known only later (merged_at, merged_by_id, closed_at,
                   state_reason, dismissed_at) are dropped, because temporal sampling filters
                   rows by their row time but not single columns. Coded ids / hashes are dropped;
                   masked text becomes its length.
  2. Graph         make_pkey_fkey_graph: one node per row, one edge type per foreign key (and
                   its reverse). Columns are encoded by PyTorch Frame (HeteroEncoder).
  3. Training table one row per (user, weekly cutoff) with 6 labels, the same weekly snapshots as
                   the LightGBM model.
  4. Temporal graph for each cutoff c only rows whose row time < c are kept (edges between
                   two such rows), i.e. temporal neighbour sampling with *all* neighbours. (PyG's
                   NeighborLoader needs pyg_lib, whose DLL is blocked by Windows application
                   control on this machine; with ~100k rows full-batch message passing is cheap.)
  5. Model         HeteroEncoder -> (+ HeteroTemporalEncoder: time between each sampled row and
                   the cutoff, switched off with --time_enc 0) -> HeteroGraphSAGE (2 layers)
                   -> MLP head with 6 outputs (one per activity), BCE loss.
  6. Thresholds    trained on all weeks but the newest; the epoch and the per-activity
                   thresholds are chosen on the newest week. By default this early-stopped model
                   is kept (--refit 0): refitting on all weeks made the thresholds miscalibrated.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from config import DEV, FINAL, snapshot, train_snapshots  # noqa: E402
from data import ACTS, build_events, events_before, labels, load_db  # noqa: E402
from metrics import scores  # noqa: E402
from results_io import merge_experiments  # noqa: E402

from relbench.base import Database, Table  # noqa: E402
from relbench.modeling.graph import make_pkey_fkey_graph  # noqa: E402
from relbench.modeling.nn import HeteroEncoder, HeteroGraphSAGE, HeteroTemporalEncoder  # noqa: E402
from relbench.modeling.utils import get_stype_proposal  # noqa: E402
from torch_frame import stype  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEST_INPUT = ROOT / "BigData2026-Assignment2-main" / "submission" / "test_input.csv"
RES = ROOT / "solution" / "results"
SEED = 2026

PK = {"Users": "user_id", "Repositories": "repo_id", "PullRequests": "pr_id", "Issues": "issue_id",
      "IssueComments": "comment_id", "PRReviews": "review_id", "PRReviewComments": "comment_id",
      "Pushes": "push_id"}
TIME = {"Users": "first_seen_at", "Repositories": "first_seen_at", "PullRequests": "first_seen_at",
        "Issues": "first_seen_at", "IssueComments": "created_at", "PRReviews": "first_seen_at",
        "PRReviewComments": "created_at", "Pushes": "created_at"}
FK = {
    "PullRequests": {"repo_id": "Repositories", "author_id": "Users"},
    "Issues": {"repo_id": "Repositories", "author_id": "Users", "pull_request_id": "PullRequests",
               "assignee_id": "Users"},
    "IssueComments": {"issue_id": "Issues", "repo_id": "Repositories", "author_id": "Users"},
    "PRReviews": {"pr_id": "PullRequests", "repo_id": "Repositories", "reviewer_id": "Users"},
    "PRReviewComments": {"pr_id": "PullRequests", "repo_id": "Repositories", "author_id": "Users",
                         "pull_request_review_id": "PRReviews", "in_reply_to_id": "PRReviewComments"},
    "Pushes": {"user_id": "Users", "repo_id": "Repositories"},
}
KNOWN_LATER = {"PullRequests": ["merged_at", "merged_by_id", "closed_at"],
               "Issues": ["closed_at", "state_reason"], "PRReviews": ["dismissed_at"]}
CODED = ["github_user_id", "login", "github_repo_id", "name", "owner_login", "org_login",
         "github_pr_id", "head_sha", "base_sha", "head_repo_id", "github_issue_id",
         "github_comment_id", "github_review_id", "commit_id", "event_id", "org_id"]
MASKED_TEXT = ["title", "body", "head_ref", "base_ref", "labels", "milestone_title", "diff_hunk",
               "path", "ref", "branch"]
CATEGORICAL = ["state", "ref_type", "subject_type", "file_extension", "issue_type", "created_via_app",
               "via_app", "is_bot", "is_cohort", "draft", "is_cross_repo", "is_pull_request", "locked",
               "is_new_ref", "is_org"]


def to_database(db, drop_tables=()):
    """relbench Database of the tables not in drop_tables (FKs to dropped tables are removed)."""
    tables = {}
    keep = [t for t in db if t not in drop_tables]
    for name in keep:
        df = db[name]
        df = df.copy()
        if name == "Repositories":
            df["is_org"] = df["org_id"].notna()
        df = df.drop(columns=[c for c in KNOWN_LATER.get(name, []) + CODED if c in df.columns])
        for c in MASKED_TEXT:
            if c in df.columns:
                df[c + "_len"] = df[c].str.len().astype("float64")
                df = df.drop(columns=c)
        for c in df.columns:
            if c in CATEGORICAL:
                df[c] = df[c].astype("object").where(df[c].notna(), None).astype(str)
            elif str(df[c].dtype) in ("Int64", "boolean"):
                df[c] = df[c].astype("float64")
        for c in df.columns:   # other times (created_at, submitted_at): seconds before row time
            if c != TIME[name] and str(df[c].dtype).startswith("datetime64"):
                df[c + "_age_h"] = (df[TIME[name]] - df[c]).dt.total_seconds() / 3600.0
                df = df.drop(columns=c)
        fk = {c: t for c, t in FK.get(name, {}).items() if t in keep}
        df = df.drop(columns=[c for c in FK.get(name, {}) if c not in fk])
        tables[name] = Table(df=df, fkey_col_to_pkey_table=fk, pkey_col=PK[name],
                             time_col=TIME[name])
    return Database(tables)


class Model(nn.Module):
    def __init__(self, data, col_stats, time_enc, channels=64, layers=2):
        super().__init__()
        self.encoder = HeteroEncoder(channels=channels,
                                     node_to_col_names_dict={nt: data[nt].tf.col_names_dict
                                                             for nt in data.node_types},
                                     node_to_col_stats=col_stats)
        self.time_enc = time_enc
        self.temporal = HeteroTemporalEncoder(node_types=[nt for nt in data.node_types if "time" in data[nt]],
                                              channels=channels)
        self.gnn = HeteroGraphSAGE(node_types=data.node_types, edge_types=data.edge_types,
                                   channels=channels, aggr="sum", num_layers=layers)
        self.head = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, len(ACTS)))

    def forward(self, tf_dict, g, seeds, entity="Users"):
        """g: temporal graph of one cutoff (edge_index_dict, time_dict, cutoff)."""
        x = self.encoder(tf_dict)
        if self.time_enc:
            seed_time = torch.tensor([g["cutoff"]])
            batch = {nt: torch.zeros(len(t), dtype=torch.long) for nt, t in g["time"].items()}
            rel = self.temporal(seed_time, g["time"], batch)
            for nt, r in rel.items():
                x[nt] = x[nt] + r
        x = self.gnn(x, g["edges"])
        return self.head(x[entity][seeds])


def unix(ts):
    return int(pd.Timestamp(ts).value // 10**9)


class RDL:
    def __init__(self, db_dir, cache, drop_tables=(), layers=2):
        self.raw = load_db(db_dir)
        self.ev = build_events(self.raw)       # labels always come from all 8 tables
        self.layers = layers
        U = self.raw["Users"]
        self.cohort = U.loc[U["is_cohort"], "user_id"].to_numpy()
        self.gid2uid = dict(zip(U["github_user_id"], U["user_id"]))
        database = to_database(self.raw, drop_tables)
        col_to_stype = get_stype_proposal(database)
        for t, cols in col_to_stype.items():
            for c in list(cols):
                if c in CATEGORICAL:
                    cols[c] = stype.categorical
                elif cols[c] not in (stype.categorical, stype.numerical, stype.timestamp):
                    cols[c] = stype.numerical
        self.data, self.col_stats = make_pkey_fkey_graph(database, col_to_stype_dict=col_to_stype,
                                                         text_embedder_cfg=None, cache_dir=cache)

    def active_before(self, c):
        past = events_before(self.ev, c)
        return np.sort(past.loc[past["uid"].isin(self.cohort), "uid"].unique())

    def _unused_table(self, snaps):
        """Training table: users, unix cutoff times, labels (N x 6)."""
        u, t, y = [], [], []
        for s in snaps:
            users = self.active_before(s["cutoff"])
            u.append(users)
            t.append(np.full(len(users), unix(s["cutoff"])))
            y.append(labels(self.ev, users, s["label_start"], s["label_end"]).to_numpy())
        return (torch.as_tensor(np.concatenate(u)), torch.as_tensor(np.concatenate(t)),
                torch.as_tensor(np.concatenate(y), dtype=torch.float32))

    def tf_dict(self):
        return {nt: self.data[nt].tf for nt in self.data.node_types}

    def graph(self, c):
        """Edges between rows whose row time is before c (no row from the future is reachable)."""
        cu = unix(c)
        alive = {nt: self.data[nt].time < cu for nt in self.data.node_types}
        edges = {}
        for et in self.data.edge_types:
            src, _, dst = et
            ei = self.data[et].edge_index
            keep = alive[src][ei[0]] & alive[dst][ei[1]]
            edges[et] = ei[:, keep]
        return {"edges": edges, "cutoff": cu,
                "time": {nt: self.data[nt].time for nt in self.data.node_types}}


def snapshot_rows(rdl, snaps):
    """[(graph, seed users, labels)] for each snapshot."""
    out = []
    for s in snaps:
        users = rdl.active_before(s["cutoff"])
        y = labels(rdl.ev, users, s["label_start"], s["label_end"]).to_numpy()
        out.append((rdl.graph(s["cutoff"]), torch.as_tensor(users), torch.as_tensor(y, dtype=torch.float32)))
    return out


def predict_proba(model, rdl, g, seeds):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(rdl.tf_dict(), g, seeds)).numpy()


BATCH = 2048


def train(rdl, rows, epochs, time_enc, val=None, log=print, batch_size=None):
    batch_size = batch_size or BATCH
    """rows: snapshot_rows. Each step: one snapshot graph, a mini-batch of its seed users."""
    torch.manual_seed(SEED)
    model = Model(rdl.data, rdl.col_stats, time_enc, layers=rdl.layers)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    pos = torch.cat([y for _, _, y in rows]).mean(0)
    lossf = nn.BCEWithLogitsLoss(pos_weight=((1 - pos) / pos.clamp(min=1e-3)).sqrt())
    tf = rdl.tf_dict()
    best = (-1.0, 0, None)
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        steps = [(i, b) for i, (_, u, _) in enumerate(rows) for b in range(0, len(u), batch_size)]
        order = torch.randperm(len(steps)).tolist()
        for k in order:
            i, b = steps[k]
            g, u, y = rows[i]
            opt.zero_grad()
            loss = lossf(model(tf, g, u[b:b + batch_size]), y[b:b + batch_size])
            loss.backward()
            opt.step()
            tot += float(loss) * len(u[b:b + batch_size])
            n += len(u[b:b + batch_size])
        msg = f"  epoch {ep} loss {tot / n:.4f} ({time.time() - t0:.0f}s)"
        if val is not None:
            g, u, y = val
            p = predict_proba(model, rdl, g, u)
            f = np.mean([best_threshold(p[:, k], y[:, k].numpy())[1] for k in range(len(ACTS))])
            msg += f" val mean F1 {f:.4f}"
            if f > best[0]:
                best = (f, ep, copy.deepcopy(model.state_dict()))
        log(msg)
    if val is not None:
        model.load_state_dict(best[2])
    return model, best[1]


def best_threshold(p, y):
    best_t, best_f = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.999, 300))):
        pred = p >= t
        tp = (pred & (y == 1)).sum()
        f = 2 * tp / (pred.sum() + (y == 1).sum() + 1e-9)
        if f > best_f:
            best_t, best_f = t, f
    return float(best_t), float(best_f)


def fit_predict(rdl, snaps, cutoff, gids, time_enc, epochs, log, refit=False):
    """Train on snapshots (newest first; the newest one picks the epoch and thresholds) and
    predict 0/1 for the github ids `gids` at `cutoff` (users with no history -> 0)."""
    rows = snapshot_rows(rdl, snaps)
    model, best_ep = train(rdl, rows[1:], epochs, time_enc, val=rows[0], log=log)
    g, u, y = rows[0]
    p = predict_proba(model, rdl, g, u)
    thr = [best_threshold(p[:, k], y[:, k].numpy())[0] for k in range(len(ACTS))]
    if refit:   # note: the thresholds were chosen on the model before the refit
        log(f"  refit on all weeks for {best_ep} epochs")
        model, _ = train(rdl, rows, best_ep, time_enc, log=log)
    users = rdl.active_before(cutoff)
    p = predict_proba(model, rdl, rdl.graph(cutoff), torch.as_tensor(users))
    uid = pd.Series(gids, index=gids).map(rdl.gid2uid)
    has = uid.notna() & uid.isin(users)
    pos = pd.Series(np.arange(len(users)), index=users).loc[uid[has].astype("int64").to_numpy()].to_numpy()
    pred = pd.DataFrame(0, index=pd.Index(gids, name="UserID"), columns=ACTS, dtype="int64")
    prob = pd.DataFrame(0.0, index=pred.index, columns=ACTS)
    for k, a in enumerate(ACTS):
        prob.loc[has[has].index, a] = p[pos, k]
        pred.loc[has[has].index, a] = (p[pos, k] >= thr[k]).astype("int64")
    return pred, prob, thr, best_ep


def run_week(rdl, gids, offset, time_enc, epochs, log, refit=False):
    vstart = DEV["valid_label_start"] - offset * pd.Timedelta(days=7)
    target = snapshot(vstart, DEV["gap"])
    snaps = train_snapshots(vstart, DEV["gap"], 5)
    pred, prob, thr, best_ep = fit_predict(rdl, snaps, target["cutoff"], gids, time_enc, epochs, log, refit)
    uid = pd.Series(gids, index=gids).map(rdl.gid2uid)
    y = labels(rdl.ev, rdl.cohort, target["label_start"], target["label_end"])
    truth = pd.DataFrame(0, index=pred.index, columns=ACTS, dtype="int64")
    ht = uid.notna() & uid.isin(y.index)
    truth.loc[ht[ht].index] = y.loc[uid[ht].astype("int64").to_numpy()].to_numpy()
    r = scores(pred.to_numpy(), truth.to_numpy())
    r_np = scores(pred.drop(columns="push").to_numpy(), truth.drop(columns="push").to_numpy(),
                  [x for x in ACTS if x != "push"])
    return r, r_np, best_ep, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "final"], default="dev")
    ap.add_argument("--db", required=True)
    ap.add_argument("--time_enc", type=int, default=1)
    ap.add_argument("--layers", type=int, default=2, help="GraphSAGE layers (= hops)")
    ap.add_argument("--drop_tables", nargs="*", default=[], help="tables left out of the graph")
    ap.add_argument("--valid_offset", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--refit", type=int, default=0,
                    help="1: refit on all weeks after choosing epochs/thresholds; 0: keep the "
                         "early-stopped model, whose thresholds were tuned on it")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--cache", default=None, help="graph cache; default ../rdl_cache/<data folder>")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None, help="final mode: submission file")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    log = lambda m: print(m, flush=True)  # noqa: E731
    t0 = time.time()
    # one cache per database and table set (the cache stores the encoded tables)
    tag = Path(a.db).resolve().parent.name + ("_no_" + "_".join(sorted(a.drop_tables)) if a.drop_tables else "")
    cache = a.cache or str(ROOT / "rdl_cache" / tag)
    rdl = RDL(a.db, cache, a.drop_tables, a.layers)
    log(f"graph built in {time.time() - t0:.0f}s: " +
        ", ".join(f"{nt} {rdl.data[nt].num_nodes}" for nt in rdl.data.node_types) +
        f"; {len(rdl.data.edge_types)} edge types")
    gids = pd.read_csv(TEST_INPUT)["UserID"].to_numpy()

    if a.mode == "final":
        snaps = train_snapshots(FINAL["data_end"], FINAL["gap"], 5)
        target = snapshot(FINAL["test_label_start"], FINAL["gap"])
        log("training label weeks (UTC): " + ", ".join(str(s["label_start"]) for s in snaps)
            + f"; predict from cutoff {target['cutoff']}")
        pred, prob, thr, ep = fit_predict(rdl, snaps, target["cutoff"], gids, bool(a.time_enc),
                                          a.epochs, log, bool(a.refit))
        out = a.out or "output_314551179.csv"
        pred.reset_index().to_csv(out, index=False)
        prob.to_csv(RES / "rdl_final_prob.csv")
        log(f"wrote {out}: epochs {ep}, thresholds " + ", ".join(f"{k}={t:.3f}" for k, t in zip(ACTS, thr))
            + " | predicted positives " + ", ".join(f"{k}={pred[k].sum()}" for k in ACTS))
        return

    name = a.name or (f"rdl_time{a.time_enc}" + ("_refit" if a.refit else ""))
    pred_dir = RES / "rdl_pred"
    pred_dir.mkdir(exist_ok=True)
    rows = []
    for off in a.valid_offset:
        log(f"[{name}] validation week -{off}")
        r, r_np, ep, pred = run_week(rdl, gids, off, bool(a.time_enc), a.epochs, log, bool(a.refit))
        pred.reset_index().to_csv(pred_dir / f"{name}_o{off}.csv", index=False)
        log(f"[{name}] week -{off}: score {r['score']:.4f} w/o push {r_np['score']:.4f} epochs {ep} | "
            + " ".join(f"{k}={v:.3f}" for k, v in r["per_act"].items()))
        rows.append({"config": name, "valid_week": off, "score": r["score"], "macro": r["macro_f1"],
                     "micro": r["micro_f1"], "score_wo_push": r_np["score"], **r["per_act"]})
    df = pd.DataFrame(rows)
    df.to_csv(RES / f"{name}.csv", index=False)
    merge_experiments(df, name)
    m = df.drop(columns=["config", "valid_week"]).mean()
    log(f"{name} mean: score {m['score']:.4f} w/o push {m['score_wo_push']:.4f} "
        f"(total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
