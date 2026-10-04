#!/usr/bin/env python3
"""Decision: does ranking features add anything the model cannot already represent?

This is the test that decides whether to spend ~8 hours of training on the feature-ranking
result, and it is a DIFFERENT question from the one `feature_signal.py` asked.

WHAT THE FIRST SCRIPT ESTABLISHED (and what it did not)
-------------------------------------------------------
Ranking a feature beat using it raw, weighted, on 8 of the top 14 -- i0_p8 went from
r_w +0.1075 to +0.4444. That is a large PER-FEATURE gain. It is also close to irrelevant,
because those 112 columns are heavily correlated with each other: if i0_p8's information is
already carried by i0_p3, i0_p0 and i0_p18, then a rank transform of i0_p8 adds redundancy,
not signal.

The quantity that decides the experiment is therefore INCREMENTAL, not marginal:

    does a linear model on rank(f_1..f_K) explain target variance that a linear model on
    ALL 112 raw features does not already explain?

If the answer is no, the transform is redundant and we do not build it. This script answers
it directly, by fitting both models on identical rows under identical weights and comparing
weighted R^2.

WHY THIS IS THE RIGHT FRAMING FOR OUR ARCHITECTURE
--------------------------------------------------
Our student is a GRU whose first layer is linear in the input. A linear probe on the same
features is therefore an APPROPRIATE MODEL OF WHAT THE FIRST LAYER ALREADY DOES. If the
probe on raw features already captures X% of the target variance, the network's first layer
has that much available to it, and only the INCREMENT from rank features is genuinely new
capacity. This is the closest cheap proxy we have for "will the transform help the model",
and it costs minutes rather than a training run.

THE TEST IS FITTED INCREMENTALLY, NOT TRAINED ON A SUBSET
---------------------------------------------------------
Rows are too many to hold (300 sequences = 6M rows x 112 columns), so we accumulate
WEIGHTED SUFFICIENT STATISTICS for the ridge normal equations: sum(w), sum(w x),
sum(w x x^T), sum(w y), sum(w x y), sum(w y y). Those are O(d^2) in memory regardless of row
count, so the fit is over ALL rows, not a sample. Ridge is solved in closed form from those
statistics, so nothing is held, nothing is approximated, and there is no train/test split to
get wrong -- both models see byte-identical rows and weights, which is the only way their R^2
values are comparable.

WEIGHTS
-------
Each row is weighted by |clip(t0, -2, 2)|, exactly as the metric weights it. t0 is
heavy-tailed (kurtosis ~+12), so an unweighted fit would be dominated by a tail the metric
barely counts and 10.94% of rows carry zero weight because the target is exactly 0.

TRAIN SPLIT ONLY. The empirical CDF used to rank a feature is estimated from a subsample of
TRAIN rows and applied unchanged, which is what serving would do. No validation statistics
enter any feature transform.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SEQ = 20_000
CLIP = 2.0
TOP_K = 16          # how many features to test as rank channels


def names() -> list[str]:
    cols = []
    for inst in ("i0", "i1"):
        cols += [f"{inst}_p{i}" for i in range(22)]
        cols += [f"{inst}_v{i}" for i in range(22)]
        cols += [f"{inst}_dp{i}" for i in range(4)]
        cols += [f"{inst}_dv{i}" for i in range(4)]
    cols += [f"a{i}" for i in range(8)]
    return cols


class Moments:
    """Weighted sufficient statistics for the ridge normal equations."""

    def __init__(self, d: int, nt: int = 2):
        self.d = d
        self.w = 0.0
        self.wx = np.zeros(d)
        self.wxx = np.zeros((d, d))
        self.wy = np.zeros(nt)
        self.wxy = np.zeros((d, nt))
        self.wyy = np.zeros(nt)
        self.n = 0

    def add(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        w = np.asarray(w, np.float64)
        self.n += x.shape[0]
        self.w += w.sum()
        self.wx += w @ x
        self.wxx += x.T @ (w[:, None] * x)
        self.wy += w @ y
        self.wxy += x.T @ (w[:, None] * y)
        self.wyy += (w[:, None] * y ** 2).sum(0)   # diag only: sum_k w*y_k^2, shape (nt,)


def centered(m: Moments):
    """Weighted-CENTERED cross moments.

    The first version fitted uncentered moments and reported R^2 = -0.36. That is not a
    finding, it is the fingerprint of a missing intercept: the fit had to explain the mean
    of y out of uncentered features, and SST was sum(w y^2) rather than
    sum(w (y - ybar)^2), so R^2 came out negative and the gain printed as +1.69e12 %. A
    negative R^2 for a model that includes the mean is arithmetically impossible, which is
    the tell that the moment DEFINITIONS were wrong, not that the features were useless.

    Centering is free because sufficient statistics are additive:
        Sxx_c = Sxx - n mx mx^T,  Sxy_c = Sxy - n mx my^T,  Syy_c = Syy - n my^2
    """
    mx, my = m.wx / m.w, m.wy / m.w
    return (m.wxx - m.w * np.outer(mx, mx),
            m.wxy - m.w * np.outer(mx, my),
            m.wyy - m.w * my ** 2)


def fit(m: Moments, cols: list[int], alpha: float):
    """Ridge on CENTERED weighted moments. Returns (beta, weighted SSE, SST)."""
    Sxx, Sxy, Syy = centered(m)
    A, B = Sxx[np.ix_(cols, cols)], Sxy[cols]
    beta = np.linalg.solve(A + np.eye(len(cols)) * (alpha * m.w / len(cols)), B)
    sse = float(sum(Syy[k] - 2 * beta[:, k] @ B[:, k] + beta[:, k] @ A @ beta[:, k]
                    for k in range(Syy.shape[0])))
    return beta, sse, float(Syy.sum())


def main() -> int:
    n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    ridge = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    X = np.load(ROOT / "cache/train/X.npy", mmap_mode="r")
    Y = np.load(ROOT / "cache/train/y.npy", mmap_mode="r")
    n_all = X.shape[0] // SEQ
    n_seq = min(n_seq, n_all)
    nm = names()
    D = len(nm)
    print(f"  train shard: {n_seq}/{n_all} sequences, {n_seq*SEQ:,} rows")

    # ---------- pass 1: pick which features to rank, and estimate their CDFs ----------
    # Selection uses unweighted correlation on a SUBSAMPLE only (cheap, and selection is
    # not the decision); the reported numbers come from pass 2 over every row.
    sub = max(1, n_seq // 4) * SEQ
    xs = np.asarray(X[:sub], np.float32)
    ys = np.asarray(Y[:sub], np.float32)
    w_sub = np.abs(np.clip(ys[:, 0], -CLIP, CLIP))
    print(f"  selection subsample: {sub:,} rows")
    mx = (w_sub[:, None] * xs).sum(0) / w_sub.sum()
    vx = (w_sub[:, None] * (xs - mx) ** 2).sum(0) / w_sub.sum()
    my = (w_sub * ys[:, 0]).sum() / w_sub.sum()
    vy = (w_sub * (ys[:, 0] - my) ** 2).sum() / w_sub.sum()
    cxy = (w_sub[:, None] * (xs - mx) * (ys[:, 0] - my)[:, None]).sum(0) / w_sub.sum()
    r = cxy / np.sqrt(np.maximum(vx, 1e-12) * vy)
    order = np.argsort(-np.abs(r))
    top = order[:TOP_K]
    print(f"  top {TOP_K} features by |weighted r| vs t0: "
          + ", ".join(nm[i] for i in top[:8]) + ", ...")

    # Empirical CDFs from TRAIN rows only, on a stride sample. Serving applies these
    # unchanged, so no validation statistic ever enters a feature transform.
    n_cdf = min(400_000, sub)
    stride = max(1, sub // n_cdf)
    samp = np.asarray(xs[::stride], np.float32)
    cdfs = [np.sort(samp[:, i]) for i in top]
    print(f"  rank CDFs from {samp.shape[0]:,} TRAIN rows each (applied unchanged at serve)")

    def rankify(x: np.ndarray, idx: int) -> np.ndarray:
        c = cdfs[idx]
        return np.searchsorted(c, x[:, top[idx]], side="left").astype(np.float32) / len(c)

    # ---------- pass 2: fit both models on every row ----------
    m_raw = Moments(D)
    m_both = Moments(D + len(top))
    m_sq = Moments(2 * D)   # raw + raw^2, the network's own nonlinearity
    half = n_seq // 2
    for s in range(n_seq):
        lo, hi = s * SEQ, (s + 1) * SEQ
        xb = np.asarray(X[lo:hi], np.float32)
        yb = np.asarray(Y[lo:hi], np.float32)
        w = np.abs(np.clip(yb[:, 0], -CLIP, CLIP))
        m_raw.add(xb, yb, w)
        ext = np.concatenate([xb, np.stack([rankify(xb, j) for j in range(len(top))], 1)], 1)
        m_both.add(ext, yb, w)
        # The decisive control. A linear probe models ONLY the first layer, but our first
        # layer is followed by a tanh recurrence, so the NETWORK can already realise
        # monotone functions of a single row. rank(f) is just such a function. If squares
        # recover the same variance as ranks, then rank adds nothing the network cannot
        # already do and the whole direction is dead. The probe must be given the
        # nonlinearity before it is allowed to condemn the transform.
        m_sq.add(np.concatenate([xb, xb * xb], 1), yb, w)
        if (s + 1) % 100 == 0:
            print(f"    pass2 seq {s+1}/{n_seq}", flush=True)

    cols_raw = list(range(D))
    cols_both = list(range(D + len(top)))
    cols_sq = list(range(2 * D))
    rows = m_raw.n
    print(f"\n  sanity: weighted SST (total target variance) = {centered(m_raw)[2].sum():.6e}")
    print(f"  fitted on all {rows:,} rows, weight = |clip(t0,2)|")
    print(f"  zero-weight rows (target exactly 0): {(np.asarray(Y[:n_seq*SEQ,0])==0).mean()*100:.2f}%")

    print(f"\n  weighted R^2 against t0 (ridge, best alpha per model)")
    print(f"  {'alpha':>10}{'raw 112':>12}{'raw+16 ranks':>15}{'increment':>12}{'raw+squares':>13}")
    best = None
    for a in ridge:
        _, sse_r, sst = fit(m_raw, cols_raw, a)
        _, sse_b, _ = fit(m_both, cols_both, a)
        _, sse_q, _ = fit(m_sq, cols_sq, a)
        r2r = 1 - sse_r / sst
        r2b = 1 - sse_b / sst
        r2q = 1 - sse_q / sst
        print(f"  {a:10.0e}{r2r:12.6f}{r2b:15.6f}{r2b - r2r:+12.6f}{r2q:13.6f}")
        if best is None or r2b - r2r > best[2] - best[1]:
            best = (a, r2r, r2b, r2q)
    a, r2r, r2b, r2q = best
    print(f"\n  best alpha {a:.0e}: raw R^2 {r2r:.6f} -> raw+ranks {r2b:.6f}"
          f"   INCREMENT {r2b - r2r:+.6f}")
    frac = (r2b - r2r) / max(r2r, 1e-12) * 100
    print(f"  i.e. ranking 16 features adds {frac:+.2f}% on top of what the linear layer")
    print(f"  already had. For reference, weighted r of the single best raw feature is")
    print(f"  {abs(r[order[0]]):.4f}, which corresponds to R^2 ~ {r[order[0]]**2:.4f}.")

    print(f"\n  CONTROL: raw + raw^2 reaches R^2 {r2q:.6f}, which is "
          f"{'ABOVE' if r2q > r2b else 'below'} raw+ranks ({r2b:.6f}).")
    print(f"  rank(f) is a MONOTONE function of f, and a linear layer followed by a tanh")
    print(f"  recurrence can already realise monotone functions of a single row. If squares")
    print(f"  match ranks, the network does not need the transform -- the linear probe was")
    print(f"  understating the network all along.")
    rank_only = (r2b - r2r) - (r2q - r2r)
    print(f"  variance attributable to rank SPECIFICALLY: {rank_only:+.6f}")
    verdict = ("BUILD IT -- ranks beat squares" if rank_only > 0.002 else
               "REJECT -- squares recover it, so the network already can" if rank_only <= 0 else
               "REJECT -- ranks add nothing over squares")
    _old_verdict = ("BUILD IT -- increment is real" if (r2b - r2r) > 0.002 else
               "REJECT -- redundant with the raw features" if (r2b - r2r) <= 0 else
               "REJECT -- actively harmful")
    print(f"  (naive verdict, ignoring the control: {_old_verdict})")
    print(f"\n  VERDICT: {verdict}   (threshold: increment > 0.002 of weighted R^2)")

    (Path(__file__).resolve().parent / "feature_rank_value.json").write_text(json.dumps({
        "n_sequences": n_seq, "rows": rows,
        "top_k_ranked": [nm[i] for i in top],
        "ridge": [{"alpha": a_,
                   "r2_raw": float(1 - fit(m_raw, cols_raw, a_)[1] / fit(m_raw, cols_raw, a_)[2]),
                   "r2_raw_plus_ranks": float(1 - fit(m_both, cols_both, a_)[1] / fit(m_both, cols_both, a_)[2])}
                  for a_ in ridge],
        "best_alpha": float(a), "r2_raw": float(r2r), "r2_raw_plus_ranks": float(r2b),
        "increment": float(r2b - r2r), "increment_pct_of_raw": float(frac),
        "verdict": verdict, "naive_verdict": _old_verdict,
        "r2_raw_plus_squares": float(r2q),
        "variance_attributable_to_rank_specifically": float(rank_only),
    }, indent=2) + "\n")
    print(f"\n  numbers -> exploration/visualisation/feature_rank_value.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())