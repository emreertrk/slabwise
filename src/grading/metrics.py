import numpy as np


def qwk(y, p, lo=1, hi=10):
    n = hi - lo + 1
    o = np.zeros((n, n))
    for a, b in zip(y, p):
        o[a - lo, b - lo] += 1
    w = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)]) / (n - 1) ** 2
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    return 1 - (w * o).sum() / (w * e).sum() if (w * e).sum() else 0.0


def balanced_mae(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return float(np.mean([np.abs(y[y == g] - p[y == g]).mean() for g in np.unique(y)]))


def within_one(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) <= 1))


def report(y, p, eras=None):
    y = np.asarray(y)
    p = np.asarray(p)
    out = {
        'qwk': float(qwk(y, p)),
        'within1': within_one(y, p),
        'balanced_mae': balanced_mae(y, p),
        'baseline_within1': within_one(y, np.full_like(y, 9)),
        'baseline_balanced_mae': balanced_mae(y, np.full_like(y, 9)),
    }
    if eras is not None:
        eras = np.asarray(eras)
        out['per_era'] = {e: {'within1': within_one(y[eras == e], p[eras == e]), 'n': int((eras == e).sum())} for e in np.unique(eras)}
    return out
