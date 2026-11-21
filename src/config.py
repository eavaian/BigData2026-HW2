"""Time settings. All times are UTC (Taiwan time = UTC+8), as in the tables.

A *snapshot* is (cutoff, label_start, label_end): features use only rows known before
`cutoff`; the label is whether the user does the activity in [label_start, label_end).
"""
import pandas as pd

TW = pd.Timedelta(hours=8)
WEEK = pd.Timedelta(days=7)


def tw(s):
    """Taiwan local time string -> UTC Timestamp."""
    return pd.Timestamp(s) - TW


DATA_START = tw("2026-10-01 08:00")      # = 2026-10-01 00:00 UTC, first GH Archive hour used

# Development: the same data as the final run (up to 11/20 23:59 TW). The newest validation
# week is 11/14 00:00 - 11/21 00:00 TW; backtests move it 1 and 2 weeks earlier (11/07, 10/31).
DEV = dict(
    data_end=tw("2026-11-21 00:00"),
    valid_label_start=tw("2026-11-14 00:00"),
    gap=pd.Timedelta(0),                 # features stop right where the label week starts
)

# Final submission: data up to 11/21 00:00 TW; predict 11/22 00:00 - 11/29 00:00 TW.
FINAL = dict(
    data_end=tw("2026-11-21 00:00"),
    test_label_start=tw("2026-11-22 00:00"),
    gap=pd.Timedelta(days=1),
)

N_TRAIN_WEEKS = 5    # training snapshots: label weeks 1..5 weeks before the validation week
MIN_HISTORY = pd.Timedelta(days=5)   # a snapshot needs at least this much history


def snapshot(label_start, gap):
    return dict(cutoff=label_start - gap, label_start=label_start, label_end=label_start + WEEK)


def train_snapshots(anchor, gap, n=N_TRAIN_WEEKS):
    """Weekly snapshots whose label weeks end at anchor, anchor - 1 week, ...; newest first."""
    out = []
    for k in range(1, n + 1):
        s = snapshot(anchor - k * WEEK, gap)
        if s["cutoff"] - DATA_START >= MIN_HISTORY:
            out.append(s)
    return out
