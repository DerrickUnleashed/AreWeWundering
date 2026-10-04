"""Spearman rank correlation without a scipy dependency.

`scipy` is NOT installed in this venv (AGENTS.md sec 14: it is not among the packages
guaranteed in the scorer), so `scipy.stats.spearmanr` is unavailable. Rank correlation
needs only an average-rank of each vector, which is two sorts.

Ties are handled by the standard mid-rank method rather than ignored: several of these
features are saturating or floor-valued (a5/a6/a7 and i1_dv0/i1_dv3 sit at a floor
sentinel of -5.199 per AGENTS.md sec 14), so ties are not a corner case here and ignoring
them would bias exactly the columns this check exists to examine.
"""

from __future__ import annotations

import numpy as np


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties receiving their mid-rank."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    r = np.empty(len(x), dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _avg_rank(np.asarray(a, np.float64)), _avg_rank(np.asarray(b, np.float64))
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0