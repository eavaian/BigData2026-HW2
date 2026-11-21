"""Reference baselines on the validation week (11/14 - 11/21 TW).

    python src/baselines.py --db ../data_final/db

copy_last_week: 1 if the user did the activity in the 7 days before the cutoff.
copy_last_k:    1 if the user did it in the last k days (k = 14, 28).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DEV, snapshot  # noqa: E402
from data import ACTS, build_events, events_before, labels, load_db  # noqa: E402
from metrics import scores  # noqa: E402


def copy_last(ev, users, cutoff, days):
    past = events_before(ev, cutoff)
    return labels(past, users, cutoff - pd.Timedelta(days=days), cutoff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    a = ap.parse_args()
    db = load_db(a.db)
    ev = build_events(db)
    u = db["Users"]
    users = u.loc[u["is_cohort"], "user_id"].to_numpy()
    s = snapshot(DEV["valid_label_start"], DEV["gap"])
    y = labels(ev, users, s["label_start"], s["label_end"])
    print("validation positive rate:", (y.mean() * 100).round(2).to_dict())
    for days in (7, 14, 28):
        p = copy_last(ev, users, s["cutoff"], days)
        r = scores(p.to_numpy(), y.to_numpy())
        print(f"copy last {days:2d}d  score {r['score']:.4f}  macro {r['macro_f1']:.4f}  "
              f"micro {r['micro_f1']:.4f}  " +
              " ".join(f"{a}={r['per_act'][a]:.3f}" for a in ACTS))


if __name__ == "__main__":
    main()
