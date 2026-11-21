"""Load the 8 tables written by preprocess.py and turn them into one activity log.

Every target activity is one row of `events` (uid, act, time), where uid is Users.user_id
(a row number) and time is UTC:

    open_pr            PullRequests.author_id,     PullRequests.created_at
    open_issue         Issues.author_id,           Issues.created_at   (is_pull_request = False)
    issue_comment      IssueComments.author_id,    IssueComments.created_at
    pr_review          PRReviews.reviewer_id,      PRReviews.first_seen_at
    pr_review_comment  PRReviewComments.author_id, PRReviewComments.created_at
    push               Pushes.user_id,             Pushes.created_at

Leakage rule: a row may be used at cutoff c only if its row time (first_seen_at /
created_at, see the README of the assignment) is before c. `events_before` applies it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ACTS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
          "PRReviews", "PRReviewComments", "Pushes"]


def load_db(db_dir):
    db_dir = Path(db_dir)
    return {t: pd.read_parquet(db_dir / f"{t}.parquet") for t in TABLES}


def build_events(db):
    """One row per target activity: uid, act, time, row_time, repo_id."""
    pr, iss = db["PullRequests"], db["Issues"]
    iss = iss[iss["is_pull_request"] == False]  # noqa: E712  (nullable boolean)
    parts = [
        ("open_pr", pr, "author_id", "created_at", "first_seen_at"),
        ("open_issue", iss, "author_id", "created_at", "first_seen_at"),
        ("issue_comment", db["IssueComments"], "author_id", "created_at", "created_at"),
        ("pr_review", db["PRReviews"], "reviewer_id", "first_seen_at", "first_seen_at"),
        ("pr_review_comment", db["PRReviewComments"], "author_id", "created_at", "created_at"),
        ("push", db["Pushes"], "user_id", "created_at", "created_at"),
    ]
    frames = []
    for act, df, ucol, tcol, rcol in parts:
        f = pd.DataFrame({"uid": df[ucol], "act": act, "time": df[tcol],
                          "row_time": df[rcol], "repo_id": df["repo_id"]})
        frames.append(f.dropna(subset=["uid", "time"]))
    ev = pd.concat(frames, ignore_index=True)
    ev["uid"] = ev["uid"].astype("int64")
    ev["act"] = pd.Categorical(ev["act"], categories=ACTS)
    return ev.sort_values("time", ignore_index=True)


def events_before(ev, cutoff):
    """Events usable as features at `cutoff` (row known before cutoff and happened before it)."""
    return ev[(ev["row_time"] < cutoff) & (ev["time"] < cutoff)]


def labels(ev, users, start, end):
    """0/1 matrix (index = uid of `users`, columns = ACTS): did the user do it in [start, end)."""
    w = ev[(ev["time"] >= start) & (ev["time"] < end)]
    y = pd.crosstab(w["uid"], w["act"]).reindex(index=users, columns=ACTS, fill_value=0)
    return (y > 0).astype("int8")


def hourly_coverage(ev):
    """Number of events of each activity per UTC hour; zero hours reveal GH Archive gaps."""
    return ev.assign(hour=ev["time"].dt.floor("h")).pivot_table(
        index="hour", columns="act", values="uid", aggfunc="size", fill_value=0, observed=False)


if __name__ == "__main__":
    import sys
    db = load_db(sys.argv[1])
    ev = build_events(db)
    users = db["Users"]
    cohort = users.loc[users["is_cohort"], "user_id"].to_numpy()
    print(f"{len(ev):,} activity rows, {ev['time'].min()} .. {ev['time'].max()}")
    print(ev["act"].value_counts().to_string())
    test = pd.read_csv(Path(__file__).resolve().parents[2] /
                       "BigData2026-Assignment2-main/submission/test_input.csv")
    found = test["UserID"].isin(users["github_user_id"]).mean()
    print(f"cohort users seen: {len(cohort)}; test users found in Users: {found:.1%}")
    print("PR rows whose created_at is before first_seen_at:",
          int((db["PullRequests"]["created_at"] < db["PullRequests"]["first_seen_at"]).sum()))
