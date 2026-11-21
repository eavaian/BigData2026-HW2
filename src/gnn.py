"""Heterogeneous GNN on the user-repository graph (comparison model, plain PyTorch).

    python src/gnn.py --db ../data_final/db --time_enc 1 --valid_offset 0
    python src/gnn.py --db ../data_final/db --time_enc 0 --valid_offset 0

Graph at cutoff c (one graph per weekly snapshot, rows known before c only):
  nodes  users (all users in Users), repositories
  edges  one edge per activity row before c: user --(activity type)--> repository
         (pushes, PRs, issues, comments, reviews, review comments: the 6 relation types)
         plus "authored PR/issue reviewed/commented by" user -> user edges are implied through
         the shared repository (2 hops).
Message passing (2 layers, mean aggregation, one weight matrix per relation type):
  repo  <- users   :  h_r = ReLU(W0 x_r + mean_{e=(u,r)} W_type(e) [h_u ; phi(c - t_e)])
  user  <- repos   :  h_u = ReLU(W1 h_u + mean_{e=(u,r)} W'_type(e) [h_r ; phi(c - t_e)])
phi is the time encoding of how long before c the row happened (log-days + sinusoids);
with --time_enc 0 it is zero, so the model sees the graph but not when edges happened.
Head: 6 logits (one per activity), trained with BCE on stacked weekly snapshots.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from config import DEV, snapshot, train_snapshots  # noqa: E402
from results_io import merge_experiments  # noqa: E402
from data import ACTS, events_before, labels  # noqa: E402
from metrics import scores  # noqa: E402
from train import TEST_INPUT, Data, best_threshold, truth_for  # noqa: E402

DAY_S = 86400.0
ROOT = Path(__file__).resolve().parents[2]


def time_enc(dt_days, on):
    """[log1p(days), sin/cos at periods 1, 7, 30 days]; zeros when time encoding is off."""
    d = torch.as_tensor(dt_days, dtype=torch.float32)
    feats = [torch.log1p(d)]
    for period in (1.0, 7.0, 30.0):
        w = 2 * np.pi / period
        feats += [torch.sin(w * d), torch.cos(w * d)]
    out = torch.stack(feats, 1)
    return out if on else torch.zeros_like(out)


class Graph:
    """Tensors of one snapshot."""

    def __init__(self, data, c, n_users, n_repos, time_on):
        past = events_before(data.ev, c).dropna(subset=["repo_id"])
        self.u = torch.as_tensor(past["uid"].to_numpy().copy(), dtype=torch.long)
        self.r = torch.as_tensor(past["repo_id"].astype("int64").to_numpy(), dtype=torch.long)
        self.t = torch.as_tensor(past["act"].cat.codes.to_numpy(), dtype=torch.long)
        dt = (c - past["time"]).dt.total_seconds().to_numpy() / DAY_S
        self.phi = time_enc(dt, time_on)
        U = data.db["Users"]
        R = data.db["Repositories"]
        # node features without time: bot / cohort flags; repo is owned by an organisation
        xu = np.zeros((n_users, 3), dtype=np.float32)
        xu[:, 0] = U["is_bot"].fillna(False).to_numpy()
        xu[:, 1] = U["is_cohort"].fillna(False).to_numpy()
        xu[:, 2] = (U["first_seen_at"] < c).to_numpy()
        xr = np.zeros((n_repos, 2), dtype=np.float32)
        xr[:, 0] = R["org_id"].notna().to_numpy()
        xr[:, 1] = (R["first_seen_at"] < c).to_numpy()
        self.xu, self.xr = torch.as_tensor(xu), torch.as_tensor(xr)
        self.n_users, self.n_repos = n_users, n_repos


class RelLayer(nn.Module):
    """mean over edges of W_type [h_src ; phi]; one linear map per relation type."""

    def __init__(self, d_in, d_out, n_types, d_phi):
        super().__init__()
        self.lin = nn.ModuleList(nn.Linear(d_in + d_phi, d_out) for _ in range(n_types))
        self.n_types = n_types

    def forward(self, h_src, src, dst, etype, phi, n_dst):
        msg = torch.zeros(len(src), self.lin[0].out_features)
        x = torch.cat([h_src[src], phi], 1)
        for k in range(self.n_types):
            m = etype == k
            if m.any():
                msg[m] = self.lin[k](x[m])
        out = torch.zeros(n_dst, msg.shape[1]).index_add_(0, dst, msg)
        deg = torch.zeros(n_dst).index_add_(0, dst, torch.ones(len(dst))).clamp(min=1)
        return out / deg[:, None], torch.log1p(deg)[:, None]


class HeteroGNN(nn.Module):
    def __init__(self, d=32, d_phi=7, n_types=len(ACTS)):
        super().__init__()
        self.emb_u, self.emb_r = nn.Linear(3, d), nn.Linear(2, d)
        self.r1 = RelLayer(d, d, n_types, d_phi)   # users -> repos
        self.u1 = RelLayer(d, d, n_types, d_phi)   # repos -> users
        self.r2 = RelLayer(d, d, n_types, d_phi)
        self.u2 = RelLayer(d, d, n_types, d_phi)
        self.self_r = nn.ModuleList([nn.Linear(d + 1, d), nn.Linear(d + 1, d)])
        self.self_u = nn.ModuleList([nn.Linear(d + 1, d), nn.Linear(d + 1, d)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, len(ACTS)))

    def forward(self, g):
        hu, hr = torch.relu(self.emb_u(g.xu)), torch.relu(self.emb_r(g.xr))
        for i, (lr, lu) in enumerate(((self.r1, self.u1), (self.r2, self.u2))):
            mr, degr = lr(hu, g.u, g.r, g.t, g.phi, g.n_repos)
            mu, degu = lu(hr, g.r, g.u, g.t, g.phi, g.n_users)
            hr = torch.relu(self.self_r[i](torch.cat([hr, degr], 1)) + mr)
            hu = torch.relu(self.self_u[i](torch.cat([hu, degu], 1)) + mu)
        return self.head(hu)


def train_eval(data, snaps, target, gids, time_on, epochs=60, seed=0, log=print):
    torch.manual_seed(seed)
    n_users, n_repos = len(data.db["Users"]), len(data.db["Repositories"])
    graphs, ys, idxs = [], [], []
    for s in snaps:
        users = data.active_before(s["cutoff"])
        graphs.append(Graph(data, s["cutoff"], n_users, n_repos, time_on))
        ys.append(torch.as_tensor(labels(data.ev, users, s["label_start"], s["label_end"]).to_numpy(),
                                  dtype=torch.float32))
        idxs.append(torch.as_tensor(users, dtype=torch.long))
    pos = torch.cat(ys).mean(0)
    pos_weight = ((1 - pos) / pos.clamp(min=1e-3)).sqrt()

    def run(train_ids, n_ep):
        torch.manual_seed(seed)
        model = HeteroGNN()
        opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for ep in range(n_ep):
            model.train()
            for i in train_ids:
                opt.zero_grad()
                loss = lossf(model(graphs[i])[idxs[i]], ys[i])
                loss.backward()
                opt.step()
        return model

    # threshold + epoch count: train on older weeks, evaluate on the newest week
    newest, older = 0, list(range(1, len(snaps)))
    model = run(older, epochs)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(graphs[newest])[idxs[newest]]).numpy()
    thr = {a: best_threshold(p[:, k], ys[newest][:, k].numpy())[0] for k, a in enumerate(ACTS)}
    model = run(list(range(len(snaps))), epochs)
    model.eval()
    # predict the target week
    users = data.active_before(target["cutoff"])
    g = Graph(data, target["cutoff"], n_users, n_repos, time_on)
    with torch.no_grad():
        p = torch.sigmoid(model(g)[torch.as_tensor(users)]).numpy()
    uid = pd.Series(gids, index=gids).map(data.gid2uid)
    pred = pd.DataFrame(0, index=pd.Index(gids, name="UserID"), columns=ACTS, dtype="int64")
    has = uid.notna() & uid.isin(users)
    pos_of = pd.Series(np.arange(len(users)), index=users)
    rows = pos_of.loc[uid[has].astype("int64").to_numpy()].to_numpy()
    for k, a in enumerate(ACTS):
        pred.loc[has[has].index, a] = (p[rows, k] >= thr[a]).astype("int64")
    # push: same rule as the main model (labels follow GH Archive gaps)
    recent = labels(events_before(data.ev, target["cutoff"]), users,
                    target["cutoff"] - pd.Timedelta(days=14), target["cutoff"])["push"]
    pred.loc[has[has].index, "push"] = recent.loc[uid[has].astype("int64").to_numpy()].to_numpy()
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--time_enc", type=int, default=1)
    ap.add_argument("--valid_offset", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--weeks", type=int, default=5)
    a = ap.parse_args()
    data = Data(a.db)
    gids = pd.read_csv(TEST_INPUT)["UserID"].to_numpy()
    rows = []
    for off in a.valid_offset:
        vstart = DEV["valid_label_start"] - off * pd.Timedelta(days=7)
        target = snapshot(vstart, DEV["gap"])
        snaps = train_snapshots(vstart, DEV["gap"], a.weeks)
        pred = train_eval(data, snaps, target, gids, bool(a.time_enc), a.epochs)
        truth = truth_for(data, gids, target)
        r = scores(pred.to_numpy(), truth.to_numpy())
        r_np = scores(pred.drop(columns="push").to_numpy(), truth.drop(columns="push").to_numpy(),
                      [x for x in ACTS if x != "push"])
        print(f"GNN time_enc={a.time_enc} week -{off}: score {r['score']:.4f} w/o push {r_np['score']:.4f} "
              + " ".join(f"{k}={v:.3f}" for k, v in r["per_act"].items()), flush=True)
        rows.append({"config": f"gnn_time{a.time_enc}", "valid_week": off, "score": r["score"],
                     "macro": r["macro_f1"], "micro": r["micro_f1"], "score_wo_push": r_np["score"],
                     **r["per_act"]})
    df = pd.DataFrame(rows)
    merge_experiments(df, f"gnn_time{a.time_enc}")
    m = df.drop(columns=["config", "valid_week"]).mean()
    print(f"gnn_time{a.time_enc} mean: score {m['score']:.4f} w/o push {m['score_wo_push']:.4f}")


if __name__ == "__main__":
    main()
