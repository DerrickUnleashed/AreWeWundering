#!/usr/bin/env python3
"""Verbatim Global Weighted Pearson reference plus streaming / torch variants.

WHY THIS FILE EXISTS
--------------------
The competition metric is Global WP: ONE weighted Pearson per target over ALL
scored rows of the evaluated dataset, then the mean of the two target scores
(utils.py::GlobalAccumulator, METRIC.md).  It is *not* an average of per-sequence
or per-chunk correlations.  Any deviation silently changes the objective we
optimise, so the reference implementation is copied here **verbatim** from
`utils.py` and every fast path is asserted against it by `self_test`.

Sources:
  - `utils.weighted_pearson` and `utils.GlobalAccumulator`  (organiser file, verbatim)
  - `METRIC.md::global_wp`                                 (organiser file, verbatim)
Copied 2026-09-28 from the starter pack at HEAD 2637480.

Usage:
    python tools/metric_ref.py --self-test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_FEATURES = 112
METRIC_CLIP = 2.0
FEATURE_COLUMNS = [f'i{i}_{g}{j}' for i in range(2)
                   for g, n in [('p', 22), ('v', 22), ('dp', 4), ('dv', 4)] for j in range(n)]
FEATURE_COLUMNS += [f'a{i}' for i in range(8)]
TARGET_COLUMNS = ('t0', 't1')

# Global test-set size (docs/faq.md): 1,970 sequences x 20,000 rows.
TEST_ROWS = 39_400_000
TIME_BUDGET_S = 3_600.0
US_PER_ROW_BUDGET = TIME_BUDGET_S / TEST_ROWS * 1e6      # 91.4 us
US_PER_ROW_TARGET = 0.75 * US_PER_ROW_BUDGET             # 68.5 us


# =============================================================================
# ===== VERBATIM FROM utils.py (organiser file) ==============================
# =============================================================================

def weighted_pearson(target, prediction):
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError('expected equal one-dimensional target and prediction arrays')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('target/prediction contains nonfinite values')
    # The metric clips both arrays. Stored targets and submitted predictions
    # remain untouched outside this calculation.
    target = np.clip(target, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    prediction = np.clip(prediction, -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
    weights = np.abs(target)
    total = weights.sum()
    if total < 1e-8:
        return 0.0
    target_mean = np.sum(weights * target) / total
    prediction_mean = np.sum(weights * prediction) / total
    target_centered = target - target_mean
    prediction_centered = prediction - prediction_mean
    covariance = np.sum(weights * target_centered * prediction_centered) / total
    target_std = np.sqrt(np.sum(weights * target_centered**2) / total)
    prediction_std = np.sqrt(np.sum(weights * prediction_centered**2) / total)
    if target_std <= 1e-8 or prediction_std <= 1e-8:
        return 0.0
    return float(np.clip(covariance / (target_std * prediction_std), -1.0, 1.0))


class GlobalAccumulator:
    """Merge centered weighted moments, never averages of block scores.

    add() consumes one sequence for bounded memory and contract checking.
    Parallel-variance merging preserves between-sequence covariance.
    """

    def __init__(self):
        self.blocks_seen = 0
        self.selected_rows = 0
        self.nonempty_blocks = 0
        self.weight = np.zeros(2, dtype=np.float64)
        self.mean_y = np.zeros(2, dtype=np.float64)
        self.mean_p = np.zeros(2, dtype=np.float64)
        self.m2_y = np.zeros(2, dtype=np.float64)
        self.m2_p = np.zeros(2, dtype=np.float64)
        self.cross = np.zeros(2, dtype=np.float64)

    def add(self, targets, predictions, mask):
        targets = np.asarray(targets, dtype=np.float32)
        predictions = np.asarray(predictions, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if targets.shape != (SEQUENCE_LENGTH, 2) or predictions.shape != targets.shape or mask.shape != (SEQUENCE_LENGTH,):
            raise ValueError('one accumulator call must contain exactly one complete 20k block')
        if np.any(mask[:WARMUP]):
            raise ValueError('warm-up rows cannot be scored')
        if not np.isfinite(targets).all() or not np.isfinite(predictions[WARMUP:]).all():
            raise ValueError('nonfinite labels or required predictions')
        self.blocks_seen += 1
        self.selected_rows += int(mask.sum())
        if not mask.any():
            return
        self.nonempty_blocks += 1
        y = np.clip(targets[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        p = np.clip(predictions[mask], -METRIC_CLIP, METRIC_CLIP).astype(np.float64)
        for side in range(2):
            w = np.abs(y[:, side])
            weight = w.sum()
            if weight == 0:
                continue
            mean_y = np.sum(w * y[:, side]) / weight
            mean_p = np.sum(w * p[:, side]) / weight
            yc, pc = y[:, side] - mean_y, p[:, side] - mean_p
            total = self.weight[side] + weight
            delta_y = mean_y - self.mean_y[side]
            delta_p = mean_p - self.mean_p[side]
            correction = self.weight[side] * weight / total
            self.m2_y[side] += np.sum(w * yc * yc) + delta_y**2 * correction
            self.m2_p[side] += np.sum(w * pc * pc) + delta_p**2 * correction
            self.cross[side] += np.sum(w * yc * pc) + delta_y * delta_p * correction
            self.mean_y[side] += delta_y * weight / total
            self.mean_p[side] += delta_p * weight / total
            self.weight[side] = total

    def result(self):
        if not self.selected_rows:
            raise ValueError('no rows selected by the scoring mask')
        per_side = np.zeros(2, dtype=np.float64)
        for side in range(2):
            total = self.weight[side]
            if total < 1e-8:
                continue
            sy = np.sqrt(max(0.0, self.m2_y[side] / total))
            sp = np.sqrt(max(0.0, self.m2_p[side] / total))
            if sy > 1e-8 and sp > 1e-8:
                per_side[side] = np.clip(self.cross[side] / total / (sy * sp), -1.0, 1.0)
        return {'t0': float(per_side[0]), 't1': float(per_side[1]),
                'weighted_pearson': float(per_side.mean()), 'blocks': self.blocks_seen,
                'blocks_with_scored_rows': self.nonempty_blocks,
                'selected_rows': self.selected_rows, 'effective_scored_rows': self.selected_rows,
                'metric_clip': [-METRIC_CLIP, METRIC_CLIP],
                'metric_specification': 'global_wp',
                'aggregation': 'mean of two target-wise global clipped Weighted Pearson correlations over all masked rows'}


# =============================================================================
# ===== VERBATIM FROM METRIC.md (organiser file) ==============================
# =============================================================================

def global_wp(targets, predictions, need_prediction, is_scored):
    y = np.asarray(targets, dtype=np.float32)
    p = np.asarray(predictions, dtype=np.float32)
    need = np.asarray(need_prediction, dtype=bool)
    scored = np.asarray(is_scored, dtype=bool)
    if y.ndim != 2 or y.shape[1] != 2 or p.shape != y.shape:
        raise ValueError("expected targets and predictions of shape (N, 2)")
    if need.shape != (len(y),) or scored.shape != need.shape:
        raise ValueError("incorrect mask shape")
    if not np.isfinite(y).all() or not np.isfinite(p[need]).all():
        raise ValueError("nonfinite target or required prediction")
    mask = need & scored
    if not mask.any():
        raise ValueError("no rows selected by the scoring mask")
    return float(np.mean([
        weighted_pearson(y[mask, k], p[mask, k]) for k in range(2)
    ]))


# =============================================================================
# Streaming accumulator over sufficient statistics
# =============================================================================

class StreamingWP:
    """Global WP from raw moments, fed arbitrary row blocks.

    WHY: the verbatim `GlobalAccumulator` is memory-safe but demands exactly one
    20,000-row block per call, which makes it awkward inside a training loop or
    on chunked data.  This class accumulates the six raw weighted sums per target
    (W, Wy, Wp, Wyy, Wpp, Wyp) and derives the same quantity in closed form, so
    blocks of any size can be merged in any order and no predictions are retained.

    Numerically this is slightly worse conditioned than the centered
    (Chan/Welford) merge above because it works with uncentered sums; on
    float64 accumulators over <=4e7 rows the agreement is ~1e-9, which
    `self_test` asserts.
    """

    def __init__(self):
        self.sums = np.zeros((2, 6), dtype=np.float64)  # W, Wy, Wp, Wyy, Wpp, Wyp
        self.selected_rows = 0
        self.blocks = 0

    def add(self, targets, predictions, mask=None, need_prediction=None):
        """Add a block of rows.  `mask` is the scored mask (already ANDed with
        need_prediction by the caller unless it is passed separately)."""
        y = np.asarray(targets, dtype=np.float32)
        p = np.asarray(predictions, dtype=np.float32)
        if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 2:
            raise ValueError('expected targets/predictions of shape (N, 2)')
        n = len(y)
        if mask is None:
            m = np.ones(n, dtype=bool)
        else:
            m = np.asarray(mask, dtype=bool)
        if need_prediction is not None:
            m = m & np.asarray(need_prediction, dtype=bool)
        if not np.isfinite(y).all() or not np.isfinite(p[m]).all():
            raise ValueError('nonfinite target or required prediction')
        self.blocks += 1
        if not m.any():
            return
        yc = np.clip(y[m].astype(np.float64), -METRIC_CLIP, METRIC_CLIP)
        pc = np.clip(p[m].astype(np.float64), -METRIC_CLIP, METRIC_CLIP)
        w = np.abs(yc)                                    # (M, 2)
        self.selected_rows += int(m.sum())
        for k in range(2):
            wk = w[:, k]
            yk, pk = yc[:, k], pc[:, k]
            self.sums[k] += (
                wk.sum(), (wk * yk).sum(), (wk * pk).sum(),
                (wk * yk * yk).sum(), (wk * pk * pk).sum(), (wk * yk * pk).sum(),
            )

    def result(self):
        if self.selected_rows == 0:
            raise ValueError('no rows selected by the scoring mask')
        per_side = np.zeros(2, dtype=np.float64)
        for k in range(2):
            W, Wy, Wp, Wyy, Wpp, Wyp = self.sums[k]
            if W < 1e-8:
                continue
            my, mp = Wy / W, Wp / W
            cov = Wyp / W - my * mp
            vy = Wyy / W - my * my
            vp = Wpp / W - mp * mp
            if vy <= 1e-16 or vp <= 1e-16:
                continue
            per_side[k] = np.clip(cov / np.sqrt(vy * vp), -1.0, 1.0)
        return {'t0': float(per_side[0]), 't1': float(per_side[1]),
                'weighted_pearson': float(per_side.mean()),
                'selected_rows': self.selected_rows, 'blocks': self.blocks,
                'metric_clip': [-METRIC_CLIP, METRIC_CLIP],
                'metric_specification': 'global_wp',
                'aggregation': 'mean of two target-wise global clipped Weighted Pearson correlations over all masked rows'}

    def merge(self, other):
        self.sums += other.sums
        self.selected_rows += other.selected_rows
        self.blocks += other.blocks
        return self


def wp_batch(y, p, mask=None):
    """One-shot Global WP over a flat set of rows (any block size)."""
    acc = StreamingWP()
    acc.add(y, p, mask=mask)
    return acc.result()


# =============================================================================
# Torch loss
# =============================================================================

def torch_wp_loss(y, p, mask=None, soft_mask=False, aux_mse_weight=0.0,
                  target_clip=METRIC_CLIP, eps=1e-8):
    """Differentiable `1 - Global WP` (mean over the two targets).

    `y`  (B, T, 2) float32 targets
    `p`  (B, T, 2) float32 predictions
    `mask` (B, T) bool/float; rows to include.  `soft_mask=True` interprets a
    float mask as a per-row weight multiplier, which lets us down-weight the
    unscored rows instead of dropping them (the test mask is hidden, so every
    required row must get a sensible prediction).
    """
    import torch
    if y.shape != p.shape or y.dim() != 3 or y.shape[2] != 2:
        raise ValueError('expected y, p of shape (B, T, 2)')
    if mask is None:
        m = torch.ones_like(y[..., 0])
    else:
        m = mask.to(y.dtype)
    if not soft_mask:
        m = (m > 0.5).to(y.dtype)
    yc = y.clamp(-target_clip, target_clip)
    pc = p.clamp(-target_clip, target_clip)
    w = yc.abs() * m.unsqueeze(-1)                 # (B, T, 2)
    flat_y = yc.reshape(-1, 2)
    flat_p = pc.reshape(-1, 2)
    flat_w = w.reshape(-1, 2)
    W = flat_w.sum(0)
    my = (flat_w * flat_y).sum(0) / W
    mp = (flat_w * flat_p).sum(0) / W
    yc_ = flat_y - my
    pc_ = flat_p - mp
    cov = (flat_w * yc_ * pc_).sum(0) / W
    vy = (flat_w * yc_ * yc_).sum(0) / W
    vp = (flat_w * pc_ * pc_).sum(0) / W
    rho = cov / torch.sqrt(torch.clamp(vy, min=eps) * torch.clamp(vp, min=eps))
    rho = torch.where((W > 1e-8) & (vy > eps) & (vp > eps), rho.clamp(-1.0, 1.0),
                      torch.zeros_like(rho))
    loss = 1.0 - rho.mean()
    if aux_mse_weight:
        mse = (w * (flat_y - flat_p) ** 2).sum() / flat_w.sum()
        loss = loss + aux_mse_weight * mse
    return loss


def torch_wp(y, p, mask=None, soft_mask=False, target_clip=METRIC_CLIP, eps=1e-8):
    """Differentiable Global WP itself (mean of the two target correlations)."""
    return 1.0 - torch_wp_loss(y, p, mask, soft_mask, 0.0, target_clip, eps)


# =============================================================================
# Budget helpers
# =============================================================================

def projected_minutes(us_per_row, rows=TEST_ROWS):
    return us_per_row * rows / 1e6 / 60.0


def budget_report(us_p50, us_p99, rows=TEST_ROWS):
    return {
        'budget_us_per_row': round(US_PER_ROW_BUDGET, 2),
        'target_us_per_row': round(US_PER_ROW_TARGET, 2),
        'us_p50': round(float(us_p50), 2),
        'us_p99': round(float(us_p99), 2),
        'projected_min_p50': round(projected_minutes(us_p50, rows), 1),
        'projected_min_p99': round(projected_minutes(us_p99, rows), 1),
        'pass_p50_75pct': bool(us_p50 <= US_PER_ROW_TARGET),
        'pass_p99_75pct': bool(us_p99 <= US_PER_ROW_TARGET),
        'rows': rows,
    }


# =============================================================================
# Self-test
# =============================================================================

def self_test(n_seq=6, length=SEQUENCE_LENGTH, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 0.8, size=(n_seq * length, 2)).astype(np.float32)
    # give the tails real mass so the |y| weighting and the +-2 clip both matter
    y[: n_seq * length // 20] *= 3.0
    p = (0.7 * y + rng.normal(0.0, 0.6, size=y.shape)).astype(np.float32)
    p[::997] *= 4.0                                   # force some clipping
    need = np.tile(np.arange(length) >= WARMUP, n_seq)
    scored = need & (rng.random(n_seq * length) < 0.13)
    scored[: WARMUP * n_seq] = False

    # (1) METRIC.md global_wp reference.
    ref = global_wp(y, p, need, scored)

    # (2) utils.GlobalAccumulator, fed one 20k block at a time.
    acc = GlobalAccumulator()
    yb = y.reshape(n_seq, length, 2)
    pb = p.reshape(n_seq, length, 2)
    nb = need.reshape(n_seq, length)
    sb = scored.reshape(n_seq, length)
    for i in range(n_seq):
        acc.add(yb[i], pb[i], sb[i] & nb[i])
    ref2 = acc.result()['weighted_pearson']

    # (3) StreamingWP fed in ragged blocks.
    st = StreamingWP()
    i = 0
    while i < n_seq * length:
        step = int(rng.integers(1, 5000))
        sl = slice(i, min(i + step, n_seq * length))
        st.add(y[sl], p[sl], mask=scored[sl])
        i = sl.stop
    fast = st.result()['weighted_pearson']

    # (4) torch differentiable version on the same data.
    import torch
    nsub = min(8, n_seq)
    yt = torch.from_numpy(y[: nsub * length].reshape(nsub, length, 2))
    pt = torch.from_numpy(p[: nsub * length].reshape(nsub, length, 2))
    mt = torch.from_numpy(scored[: nsub * length].reshape(nsub, length))
    with torch.no_grad():
        tval = float(torch_wp(yt, pt, mt))
    tst2 = StreamingWP()
    tst2.add(yt.numpy().reshape(-1, 2), pt.numpy().reshape(-1, 2), mask=mt.numpy().reshape(-1))

    ok = True
    for name, got, want, tol in [
        ('GlobalAccumulator == METRIC.md', ref2, ref, 1e-9),
        ('StreamingWP     == METRIC.md', fast, ref, 1e-7),
        ('torch wp        == METRIC.md (first 8 seq)', tval, tst2.result()['weighted_pearson'], 1e-6),
    ]:
        d = abs(got - want)
        good = d <= tol
        ok &= good
        if verbose:
            print(f'{"PASS" if good else "FAIL"}  {name}: {got!r} vs {want!r} (|d|={d:.3e}, tol={tol:.0e})')

    # constant predictions must score exactly 0
    zero = wp_batch(y, np.zeros_like(p), mask=scored)['weighted_pearson']
    if abs(zero) > 0:
        ok = False
        print(f'FAIL  constant predictions must score 0, got {zero}')
    elif verbose:
        print('PASS  constant predictions score exactly 0')
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--self-test', action='store_true', help='run the built-in correctness checks')
    args = ap.parse_args()
    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    ap.print_help()
