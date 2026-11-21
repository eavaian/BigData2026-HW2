"""Shared writer for results/experiments.csv (several runs may finish at the same time)."""
import os
import time
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"


def merge_experiments(df, name):
    """Replace the rows of config `name` in results/experiments.csv, under a lock file."""
    RES.mkdir(exist_ok=True)
    lock = RES / "experiments.csv.lock"
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            time.sleep(1)
    else:
        raise RuntimeError(f"{lock} is stuck; delete it and rerun")
    try:
        path = RES / "experiments.csv"
        old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["config"])
        pd.concat([old[old["config"] != name], df]).to_csv(path, index=False)
    finally:
        os.close(fd)
        os.remove(lock)
