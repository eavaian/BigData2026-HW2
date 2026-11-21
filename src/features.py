"""Features of a user at a cutoff time c, built from the 8 tables.

Every row is used only if its row time is before c, and the "known only later" columns
(merged_at, merged_by_id, closed_at, state_reason, dismissed_at) only if their value is
before c. Features are grouped so that groups can be switched off for ablations:

  T_*  temporal    : counts in several windows, time since last / first activity, decayed
                     counts, active days, week-over-week trend, weekday/hour profile
  G_*  global      : cohort-wide activity before c (shows GH Archive gaps), per activity
  R_*  relational  : repositories, owners/orgs, collaborators, replies the user receives,
                     the user's open PRs / issues, review links between users
"""
import numpy as np
import pandas as pd

from data import ACTS, events_before

DAY = pd.Timedelta(days=1)
WINDOWS = [1, 3, 7, 14, 28]
HALF_LIVES = [2, 7, 21]


# ---------------------------------------------------------------------------- helpers

def _per_user(series_by_uid, users, name):
    return series_by_uid.reindex(users).fillna(0).rename(name)


def _act_counts(ev, users, prefix):
    if ev.empty:
        return pd.DataFrame(0, index=users, columns=[f"{prefix}{a}" for a in ACTS] + [f"{prefix}all"])
    m = pd.crosstab(ev["uid"], ev["act"]).reindex(index=users, columns=ACTS, fill_value=0)
    m.columns = [f"{prefix}{a}" for a in ACTS]
    m[f"{prefix}all"] = m.sum(1)
    return m


# ---------------------------------------------------------------------------- temporal

def temporal_features(past, users, c):
    own = past[past["uid"].isin(users)]
    out = []
    for w in WINDOWS:
        out.append(_act_counts(own[own["time"] >= c - w * DAY], users, f"T_cnt{w}d_"))
    # previous week, for a week-over-week trend
    prev = _act_counts(own[(own["time"] >= c - 14 * DAY) & (own["time"] < c - 7 * DAY)], users, "p_")
    trend = out[2].to_numpy() - prev.to_numpy()
    out.append(pd.DataFrame(trend, index=users, columns=[f"T_trend_{a}" for a in ACTS + ["all"]]))
    out.append(_act_counts(own, users, "T_cntall_"))

    age = (c - own["time"]).dt.total_seconds() / 86400.0
    for h in HALF_LIVES:
        wgt = np.power(0.5, age / h)
        d = own.assign(w=wgt).pivot_table(index="uid", columns="act", values="w", aggfunc="sum",
                                          fill_value=0, observed=False)
        d = d.reindex(index=users, columns=ACTS, fill_value=0)
        d.columns = [f"T_decay{h}_{a}" for a in ACTS]
        out.append(d)

    last = own.groupby(["uid", "act"], observed=True)["time"].max().unstack()
    last = last.reindex(index=users, columns=ACTS)
    since = ((c - last).apply(lambda s: s.dt.total_seconds()) / 86400.0).fillna(999.0)
    since.columns = [f"T_days_since_{a}" for a in ACTS]
    out.append(since)
    first = own.groupby("uid")["time"].min().reindex(users)
    lastall = own.groupby("uid")["time"].max().reindex(users)
    f = pd.DataFrame(index=users)
    f["T_days_since_any"] = ((c - lastall).dt.total_seconds() / 86400.0).fillna(999.0)
    f["T_days_since_first"] = ((c - first).dt.total_seconds() / 86400.0).fillna(0.0)
    day = own["time"].dt.floor("D")
    for w in (7, 14, 28):
        f[f"T_active_days{w}"] = _per_user(own[own["time"] >= c - w * DAY].assign(d=day)
                                           .groupby("uid")["d"].nunique(), users, "x")
    weeks = ((c - own["time"]) // (7 * DAY)).rename("wk")
    f["T_active_weeks4"] = _per_user(own[own["time"] >= c - 28 * DAY].assign(wk=weeks)
                                     .groupby("uid")["wk"].nunique(), users, "x")
    # regularity: mean gap between active days (days)
    ad = own.assign(d=day).drop_duplicates(["uid", "d"]).sort_values("d")
    gaps = ad.groupby("uid")["d"].diff().dt.total_seconds() / 86400.0
    f["T_mean_gap_days"] = gaps.groupby(ad["uid"]).mean().reindex(users).fillna(99.0)
    # weekday / hour profile (Taiwan time)
    tw = own["time"] + pd.Timedelta(hours=8)
    f["T_weekend_frac"] = (tw.dt.dayofweek >= 5).groupby(own["uid"]).mean().reindex(users).fillna(0)
    f["T_night_frac"] = ((tw.dt.hour < 7) | (tw.dt.hour >= 22)).groupby(own["uid"]).mean() \
        .reindex(users).fillna(0)
    out.append(f)
    return pd.concat(out, axis=1)


# ---------------------------------------------------------------------------- global

def global_features(past, cohort_uids, c, users):
    """Cohort-wide counts: shows the model how complete the data is around c."""
    co = past[past["uid"].isin(cohort_uids)]
    g = {}
    for a in ACTS:
        e = co[co["act"] == a]
        last7 = (e["time"] >= c - 7 * DAY).sum()
        prev21 = ((e["time"] >= c - 28 * DAY) & (e["time"] < c - 7 * DAY)).sum() / 3.0
        g[f"G_cnt7d_{a}"] = last7
        g[f"G_ratio_{a}"] = last7 / (prev21 + 1.0)
        # hours in the last 7 days with no event of this activity at all (data gaps)
        hrs = e.loc[e["time"] >= c - 7 * DAY, "time"].dt.floor("h").nunique()
        g[f"G_hours_covered_{a}"] = hrs
    return pd.DataFrame([g] * len(users), index=users)


# ---------------------------------------------------------------------------- relational

def relational_features(db, past, users, c, cohort_uids):
    f = pd.DataFrame(index=users)
    U = db["Users"]
    R = db["Repositories"]
    PR = db["PullRequests"]
    IS = db["Issues"]
    IC = db["IssueComments"]
    RV = db["PRReviews"]
    RC = db["PRReviewComments"]

    R = R[R["first_seen_at"] < c]
    PR = PR[PR["first_seen_at"] < c].copy()
    IS = IS[IS["first_seen_at"] < c].copy()
    IC = IC[IC["created_at"] < c]
    RV = RV[RV["first_seen_at"] < c]
    RC = RC[RC["created_at"] < c]
    # columns known only later: keep only values that are already known at c
    for df, cols in ((PR, ["merged_at", "closed_at"]), (IS, ["closed_at"])):
        for col in cols:
            df.loc[df[col] >= c, col] = pd.NaT
    PR.loc[PR["merged_at"].isna(), "merged_by_id"] = pd.NA
    IS.loc[IS["closed_at"].isna(), "state_reason"] = None

    own = past[past["uid"].isin(users)]
    recent = own[own["time"] >= c - 28 * DAY]

    # --- user <-> repository --------------------------------------------------
    f["R_repos28"] = _per_user(recent.groupby("uid")["repo_id"].nunique(), users, "x")
    f["R_repos7"] = _per_user(own[own["time"] >= c - 7 * DAY].groupby("uid")["repo_id"].nunique(),
                              users, "x")
    f["R_repos_all"] = _per_user(own.groupby("uid")["repo_id"].nunique(), users, "x")
    # share of activity in the user's single main repository (focus vs. many repos)
    rc = recent.groupby(["uid", "repo_id"]).size()
    f["R_top_repo_share"] = (rc.groupby(level=0).max() / rc.groupby(level=0).sum()).reindex(users).fillna(0)

    login = U.set_index("user_id")["login"]
    owned = R.groupby("owner_login").size()
    f["R_owned_repos"] = login.reindex(users).map(owned).fillna(0).to_numpy()
    rinfo = R.set_index("repo_id")
    rec_repo = recent.join(rinfo[["owner_login", "org_id"]], on="repo_id")
    rec_repo["own_repo"] = rec_repo["owner_login"].to_numpy() == login.reindex(rec_repo["uid"]).to_numpy()
    f["R_own_repo_frac28"] = rec_repo.groupby("uid")["own_repo"].mean().reindex(users).fillna(0)
    f["R_org_repo_frac28"] = rec_repo["org_id"].notna().groupby(rec_repo["uid"]).mean() \
        .reindex(users).fillna(0)

    # --- repository heat: how busy are the user's repositories right now -------
    # all events (any user) on each repo in the last 7 days, from every table
    allrep = past[past["time"] >= c - 7 * DAY].groupby("repo_id").size()
    allrep_users = past[past["time"] >= c - 28 * DAY].groupby("repo_id")["uid"].nunique()
    ur = recent[["uid", "repo_id"]].drop_duplicates()
    ur["heat7"] = ur["repo_id"].map(allrep).fillna(0)
    ur["contrib28"] = ur["repo_id"].map(allrep_users).fillna(0)
    f["R_repo_heat7_sum"] = _per_user(ur.groupby("uid")["heat7"].sum(), users, "x")
    f["R_repo_heat7_max"] = _per_user(ur.groupby("uid")["heat7"].max(), users, "x")
    f["R_repo_contrib_max"] = _per_user(ur.groupby("uid")["contrib28"].max(), users, "x")
    # per activity: activity of each type in the user's repos (7d), by anyone
    for a in ("open_pr", "issue_comment", "pr_review", "push"):
        ar = past[(past["time"] >= c - 7 * DAY) & (past["act"] == a)].groupby("repo_id").size()
        f[f"R_repo_{a}7"] = _per_user(ur.assign(v=ur["repo_id"].map(ar).fillna(0))
                                      .groupby("uid")["v"].sum(), users, "x")

    # --- collaborators: users sharing a repository in the last 28 days --------
    pairs = past[past["time"] >= c - 28 * DAY][["uid", "repo_id"]].drop_duplicates()
    co = ur[["uid", "repo_id"]].merge(pairs, on="repo_id", suffixes=("", "_o"))
    co = co[co["uid"] != co["uid_o"]]
    f["R_collaborators28"] = _per_user(co.groupby("uid")["uid_o"].nunique(), users, "x")
    # how active the collaborators were in the last 7 days (neighbour aggregation)
    act7 = past[past["time"] >= c - 7 * DAY].groupby("uid").size()
    co["o_act7"] = co["uid_o"].map(act7).fillna(0)
    cu = co.drop_duplicates(["uid", "uid_o"])
    f["R_collab_act7_mean"] = _per_user(cu.groupby("uid")["o_act7"].mean(), users, "x")
    f["R_collab_act7_sum"] = _per_user(cu.groupby("uid")["o_act7"].sum(), users, "x")
    f["R_collab_active_frac"] = _per_user((cu["o_act7"] > 0).groupby(cu["uid"]).mean(), users, "x")
    # per activity: share of collaborators who did that activity in the last 7 days
    # (a one-hop neighbour mean, as one message-passing step of a GNN would compute)
    did7 = past[past["time"] >= c - 7 * DAY].groupby(["uid", "act"], observed=True).size().unstack(fill_value=0)
    did7 = (did7.reindex(columns=ACTS, fill_value=0) > 0).astype("float32")
    nb = cu[["uid", "uid_o"]].join(did7, on="uid_o").fillna(0)
    for a in ACTS:
        f[f"R_nb_{a}_frac"] = _per_user(nb.groupby("uid")[a].mean(), users, "x")
    # two hops: share of users active in the user's repositories (weighted by the user's
    # own activity there) who did each activity in the last 7 days
    rp = past[past["time"] >= c - 7 * DAY][["uid", "repo_id", "act"]].drop_duplicates()
    repo_users = rp.groupby("repo_id")["uid"].nunique()
    for a in ("open_pr", "pr_review", "issue_comment"):
        ra = rp[rp["act"] == a].groupby("repo_id")["uid"].nunique() / repo_users
        f[f"R_repo_share_{a}"] = _per_user(ur.assign(v=ur["repo_id"].map(ra).fillna(0))
                                           .groupby("uid")["v"].mean(), users, "x")

    # --- the user's own pull requests and issues and what others do on them ---
    mine_pr = PR[PR["author_id"].isin(users)]
    open_pr = mine_pr[mine_pr["closed_at"].isna() & mine_pr["merged_at"].isna()]
    f["R_open_prs"] = _per_user(open_pr.groupby("author_id").size(), users, "x")
    f["R_open_prs_recent"] = _per_user(open_pr[open_pr["first_seen_at"] >= c - 14 * DAY]
                                       .groupby("author_id").size(), users, "x")
    f["R_prs_merged28"] = _per_user(mine_pr[mine_pr["merged_at"] >= c - 28 * DAY]
                                    .groupby("author_id").size(), users, "x")
    f["R_prs_cross_repo_frac"] = _per_user(mine_pr["is_cross_repo"].astype("float")
                                           .groupby(mine_pr["author_id"]).mean(), users, "x")
    f["R_merges_done28"] = _per_user(PR[PR["merged_at"] >= c - 28 * DAY].groupby("merged_by_id").size(),
                                     users, "x")
    pr_author = PR.set_index("pr_id")["author_id"]
    # reviews / review comments received on the user's PRs in the last 7 days
    rv7 = RV[RV["first_seen_at"] >= c - 7 * DAY].assign(author=lambda d: d["pr_id"].map(pr_author))
    rv7 = rv7[rv7["author"] != rv7["reviewer_id"]]
    f["R_reviews_received7"] = _per_user(rv7.groupby("author").size(), users, "x")
    f["R_changes_requested7"] = _per_user(rv7[rv7["state"] == "changes_requested"]
                                          .groupby("author").size(), users, "x")
    rc7 = RC[RC["created_at"] >= c - 7 * DAY].assign(author=lambda d: d["pr_id"].map(pr_author))
    rc7 = rc7[rc7["author"] != rc7["author_id"]]
    f["R_review_comments_received7"] = _per_user(rc7.groupby("author").size(), users, "x")
    # comments by others on issues / PR conversations the user opened
    iss_author = IS.set_index("issue_id")["author_id"]
    ic7 = IC[IC["created_at"] >= c - 7 * DAY].assign(author=lambda d: d["issue_id"].map(iss_author))
    ic7 = ic7[ic7["author"] != ic7["author_id"]]
    f["R_comments_received7"] = _per_user(ic7.groupby("author").size(), users, "x")
    mine_is = IS[(IS["author_id"].isin(users)) & (IS["is_pull_request"] == False)]  # noqa: E712
    f["R_open_issues"] = _per_user(mine_is[mine_is["closed_at"].isna()].groupby("author_id").size(),
                                   users, "x")
    f["R_issues_assigned"] = _per_user(IS[IS["closed_at"].isna()].groupby("assignee_id").size(), users, "x")
    # review relationships: distinct people the user reviewed / was reviewed by (28 days)
    rv28 = RV[RV["first_seen_at"] >= c - 28 * DAY].assign(author=lambda d: d["pr_id"].map(pr_author))
    rv28 = rv28[rv28["author"] != rv28["reviewer_id"]]
    f["R_reviewees28"] = _per_user(rv28.groupby("reviewer_id")["author"].nunique(), users, "x")
    f["R_reviewers28"] = _per_user(rv28.groupby("author")["reviewer_id"].nunique(), users, "x")
    # open PRs by others in repos where the user reviewed recently (review backlog)
    rev_repos = RV[(RV["first_seen_at"] >= c - 28 * DAY) & RV["reviewer_id"].isin(users)][
        ["reviewer_id", "repo_id"]].drop_duplicates()
    open_all = PR[PR["closed_at"].isna() & PR["merged_at"].isna() &
                  (PR["first_seen_at"] >= c - 14 * DAY)].groupby("repo_id").size()
    rev_repos["backlog"] = rev_repos["repo_id"].map(open_all).fillna(0)
    f["R_review_backlog"] = _per_user(rev_repos.groupby("reviewer_id")["backlog"].sum(), users, "x")
    # comment style
    myic = IC[IC["author_id"].isin(users)]
    f["R_comment_on_pr_frac"] = _per_user(myic["is_pull_request"].astype("float")
                                          .groupby(myic["author_id"]).mean(), users, "x")
    f["R_is_bot"] = U.set_index("user_id")["is_bot"].reindex(users).fillna(False).astype(int).to_numpy()
    return f


# ---------------------------------------------------------------------------- all

GROUPS = ("T", "G", "R")


def build_features(db, ev, users, c, cohort_uids, groups=GROUPS):
    past = events_before(ev, c)
    parts = []
    if "T" in groups:
        parts.append(temporal_features(past, users, c))
    if "G" in groups:
        parts.append(global_features(past, cohort_uids, c, users))
    if "R" in groups:
        parts.append(relational_features(db, past, users, c, cohort_uids))
    X = pd.concat(parts, axis=1).astype("float32")
    X.index.name = "uid"
    return X
