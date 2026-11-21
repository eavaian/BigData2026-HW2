"""Run a configuration on several validation weeks and summarise (mean over weeks).

    python src/experiments.py --db ../data_final/db --name full -- --thr_mode rate
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from results_io import merge_experiments

HERE = Path(__file__).parent
RES = HERE.parent / "results"


def run(db, name, extra, offsets):
    rows = []
    for off in offsets:
        tag = f"exp_{name}_o{off}"
        subprocess.run([sys.executable, str(HERE / "train.py"), "dev", "--db", db,
                        "--valid_offset", str(off), "--tag", tag, *extra], check=True,
                        stdout=subprocess.DEVNULL)
        r = json.loads((RES / f"{tag}.json").read_text())
        row = {"config": name, "valid_week": off, "score": r["valid"]["score"],
               "macro": r["valid"]["macro_f1"], "micro": r["valid"]["micro_f1"],
               "score_wo_push": r["valid_without_push"]["score"], **r["valid"]["per_act"]}
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"]
    df = run(a.db, a.name, extra, a.offsets)
    merge_experiments(df, a.name)
    m = df.drop(columns=["config", "valid_week"]).mean()
    print(f"{a.name:28s} mean over weeks {a.offsets}: score {m['score']:.4f}  w/o push {m['score_wo_push']:.4f}  "
          + " ".join(f"{k}={m[k]:.3f}" for k in ["open_pr", "open_issue", "issue_comment", "pr_review",
                                                  "pr_review_comment", "push"]))
