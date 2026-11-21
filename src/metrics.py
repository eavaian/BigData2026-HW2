"""Same metric as scripts/evaluate.py: Score = (macro-F1 + micro-F1) / 2 over the activities."""
import numpy as np

from data import ACTS


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def scores(pred, truth, acts=ACTS):
    """pred, truth: (n_users, len(acts)) 0/1 arrays in the column order of `acts`."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    tp = ((pred == 1) & (truth == 1)).sum(0)
    fp = ((pred == 1) & (truth == 0)).sum(0)
    fn = ((pred == 0) & (truth == 1)).sum(0)
    per = {a: float(f1(*c)) for a, c in zip(acts, zip(tp, fp, fn))}
    macro = sum(per.values()) / len(acts)
    micro = float(f1(tp.sum(), fp.sum(), fn.sum()))
    return {"per_act": per, "macro_f1": macro, "micro_f1": micro, "score": (macro + micro) / 2}
