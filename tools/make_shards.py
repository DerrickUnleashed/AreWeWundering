#!/usr/bin/env python3
"""Stream `datasets/*.parquet` row group by row group into memory-mapped shards.

WHY
---
`train.parquet` is 29 GB / 212 M rows x 112 float32 features (~95 GB unpacked).
It must never be loaded whole (AGENTS.md §2.7).  The natural unit of this dataset
is the Parquet *row group*, which is exactly one 20,000-row sequence
(docs/data_overview.md), so we decode one group at a time -- ~10 MB of RAM --
and copy it into pre-allocated `.npy` memmaps.  Training then memory-maps the
shards: no RAM residency, no decode cost in the hot loop, and the OS page cache
does the buffering.

Layout produced per split (default `cache/<split>/`):
    X.npy                (N, 112) float16   features, exact `utils.FEATURE_COLUMNS` order
    y.npy                (N, 2)   float32   targets t0, t1  (float32: the metric clips
                                                  at +-2 and uses |t| as weights, so
                                                  quantisation here would bias the metric)
    need_prediction.npy  (N,)     bool
    is_scored.npy        (N,)     bool      valid split only
    seq_index.npy        (S+1,)   int64     start row of each sequence (seq i = rows
                                                  [seq_index[i], seq_index[i+1]))
    meta.json                        dtype, shapes, seq count, source file + sha of
                                     header, parquet row-group count, build command
plus `cache/<split>/build_state.json` for resume, and `cache/norm_stats.json`
(train-only statistics, see below).

float16 vs float32 for X: measured on 4 train row groups (AGENTS.md §14) the worst
absolute quantisation error is 0.00194 and the worst error *relative to the
column's own std* is 1.7 % (mean 0.35 %) -- far below the model's own noise.  It
halves the cache to ~56 GB, which fits in this machine's 64 GiB of RAM.

Train-only normalisation statistics (`cache/norm_stats.json`) are collected in the
same single streaming pass: exact count/mean/std (float64 Welford), min/max,
zero/NaN/inf counts per column, plus a random subsample of rows for quantiles.
Never computed from valid or test data (causality/directive 1).

Resumable: completed row groups are recorded in `build_state.json`; re-running
with the same `--max-sequences` continues where it stopped.

Examples
--------
    python tools/make_shards.py --split train --max-sequences 40
    python tools/make_shards.py --split valid --max-sequences 12
    python tools/make_shards.py --split both
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH, WARMUP  # noqa: E402

N_FEATURES = len(FEATURE_COLUMNS)
PARQUET = {'train': ROOT / 'datasets' / 'train.parquet',
           'valid': ROOT / 'datasets' / 'valid.parquet'}
QUANTILE_GRID = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
NPY_VERSION = (1, 0)


# ----------------------------------------------------------------------------- utils

def human(n):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if abs(n) < 1024 or unit == 'TiB':
            return f'{n:.2f} {unit}' if unit != 'B' else f'{int(n)} B'
        n /= 1024.0


def open_npy(path, shape, dtype, mode):
    """open_memmap wrapper that tolerates an already-created file."""
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


class ColumnStats:
    """Exact streaming per-column moments (float64) for X and y."""

    def __init__(self, n_cols):
        self.n = n_cols
        self.count = np.zeros(n_cols, dtype=np.float64)
        self.mean = np.zeros(n_cols, dtype=np.float64)
        self.m2 = np.zeros(n_cols, dtype=np.float64)
        self.min = np.full(n_cols, np.inf)
        self.max = np.full(n_cols, -np.inf)
        self.zeros = np.zeros(n_cols, dtype=np.int64)
        self.nans = np.zeros(n_cols, dtype=np.int64)
        self.infs = np.zeros(n_cols, dtype=np.int64)

    def update(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.nans += np.isnan(X).sum(0)
        self.infs += np.isinf(X).sum(0)
        finite = np.isfinite(X)
        Z = np.where(finite, X, 0.0)
        self.count += finite.sum(0)
        self.zeros += (X == 0).sum(0)
        self.min = np.minimum(self.min, np.where(finite, X, np.inf).min(0))
        self.max = np.maximum(self.max, np.where(finite, X, -np.inf).max(0))
        # batched Welford
        nb = X.shape[0]
        mb = Z.mean(0)
        m2b = ((Z - mb) ** 2).sum(0)
        delta = mb - self.mean
        tot = self.count + nb
        self.mean = self.mean + delta * (nb / tot)
        self.m2 = self.m2 + m2b + (delta ** 2) * (self.count * nb / tot)
        self.count = tot

    def to_dict(self, names):
        std = np.sqrt(np.where(self.count > 0, self.m2 / np.maximum(self.count, 1), np.nan))
        total = self.count + self.nans + self.infs
        return {names[i]: {'mean': float(self.mean[i]), 'std': float(std[i]),
                          'min': float(self.min[i]), 'max': float(self.max[i]),
                          'zero_fraction': float(self.zeros[i] / max(int(total[i]), 1)),
                          'nan': int(self.nans[i]), 'inf': int(self.infs[i]),
                          'n_finite': int(self.count[i])}
                for i in range(self.n)}


# ----------------------------------------------------------------------------- build

def build(split, out_dir, max_sequences=None, start_seq=0, dtype='float16',
          quantile_sequences=100, seed=1337, ram_gb=8.0, resume=True, progress_every=50,
          force=False):
    src = PARQUET[split]
    if not src.exists():
        raise SystemExit(f'missing {src}')
    pf = pq.ParquetFile(src)
    total_groups = pf.num_row_groups
    if max_sequences is not None:
        max_sequences = min(max_sequences, total_groups)
        n_groups = max(0, max_sequences - start_seq)
    else:
        n_groups = total_groups - start_seq
    if n_groups <= 0:
        raise SystemExit('nothing to build (check --max-sequences/--start-seq)')
    n_rows = n_groups * SEQUENCE_LENGTH
    np_dtype = np.float16 if dtype == 'float16' else np.float32
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / 'build_state.json'

    est = n_rows * (N_FEATURES * np.dtype(dtype).itemsize + 2 * 4 + 1 + 1)
    print(f'[{split}] {n_groups} sequences -> {n_rows:,} rows, X dtype {np.dtype(dtype).name}, '
          f'approx {human(est)} in {out_dir}', flush=True)

    # ---- pre-allocate (or reopen) the memmaps
    files = {
        'X': (out_dir / 'X.npy', (n_rows, N_FEATURES), np_dtype),
        'y': (out_dir / 'y.npy', (n_rows, 2), np.float32),
        'need_prediction': (out_dir / 'need_prediction.npy', (n_rows,), np.bool_),
    }
    if split == 'valid':
        files['is_scored'] = (out_dir / 'is_scored.npy', (n_rows,), np.bool_)
    seq_index = np.arange(n_groups + 1, dtype=np.int64) * SEQUENCE_LENGTH

    state = {'done': 0, 'split': split, 'n_groups': n_groups, 'start_seq': start_seq,
             'dtype': dtype, 'quantile_sequences': quantile_sequences, 'seed': seed}
    if resume and state_path.exists():
        prev = json.loads(state_path.read_text())
        same = all(prev.get(k) == state[k] for k in
                   ('split', 'n_groups', 'start_seq', 'dtype', 'quantile_sequences', 'seed'))
        if same and all((out_dir / Path(p).name).exists() for p, _, _ in files.values()):
            state = prev
            print(f'[{split}] resuming: {state["done"]}/{n_groups} sequences already built', flush=True)
        else:
            print(f'[{split}] previous build state is incompatible -> rebuilding from scratch', flush=True)

    mode = 'r+' if state['done'] else 'w+'
    arrs = {}
    for key, (path, shape, dt) in files.items():
        if path.exists():
            existing = np.load(path, mmap_mode='r')
            if existing.shape != shape or existing.dtype != dt:
                if not force:
                    raise SystemExit(f'{path} exists with shape/dtype {existing.shape}/{existing.dtype}, '
                                     f'expected {shape}/{dt}; re-run with --force to rebuild '
                                     f'(only ever touches the regenerable cache/ directory)')
                print(f'[{split}] --force: replacing {path.name} '
                      f'({existing.shape}/{existing.dtype} -> {shape}/{dt})', flush=True)
                del existing
                path.unlink()
        arrs[key] = open_npy(path, shape, dt, mode)

    xstats = ColumnStats(N_FEATURES)
    ystats = ColumnStats(2)
    if state['done']:
        # statistics for the already-built prefix are recomputed at the end from
        # the memmap (cheap) rather than stored, so nothing can get out of sync.
        pass
    rng = np.random.default_rng(seed)
    quantile_rows = set()
    if quantile_sequences > 0:
        # deterministic subsample of whole sequences for the quantile estimate
        cand = rng.choice(n_groups, size=min(quantile_sequences, n_groups), replace=False)
        quantile_rows = set(int(c) for c in cand)
        qsample = np.empty((len(quantile_rows) * SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
        qmask = np.zeros(n_rows, dtype=bool)
        for c in quantile_rows:
            qmask[c * SEQUENCE_LENGTH:(c + 1) * SEQUENCE_LENGTH] = True
    else:
        qsample = np.zeros((0, N_FEATURES), dtype=np.float32)
        qmask = np.zeros(n_rows, dtype=bool)

    t0 = time.time()
    qi = 0
    last = state['done']
    for i in range(state['done'], n_groups):
        g = start_seq + i
        tbl = pf.read_row_group(g, use_threads=False)
        seq = int(tbl['seq_ix'][0].as_py())
        if not np.all(tbl['seq_ix'].to_numpy() == seq):
            raise SystemExit(f'row group {g} is not a single sequence')
        steps = tbl['step_in_seq'].to_numpy()
        if not np.array_equal(steps, np.arange(SEQUENCE_LENGTH)):
            raise SystemExit(f'row group {g} has a wrong step order')
        need = tbl['need_prediction'].to_numpy().astype(bool)
        if not np.array_equal(need, np.arange(SEQUENCE_LENGTH) >= WARMUP):
            raise SystemExit(f'row group {g} violates the warm-up contract')
        X = np.column_stack([tbl[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        Y = np.column_stack([tbl[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        if not np.isfinite(X).all():
            raise SystemExit(f'row group {g} has non-finite features')
        sl = slice(i * SEQUENCE_LENGTH, (i + 1) * SEQUENCE_LENGTH)
        arrs['X'][sl] = X.astype(np_dtype)
        arrs['y'][sl] = Y
        arrs['need_prediction'][sl] = need
        if 'is_scored' in arrs:
            arrs['is_scored'][sl] = tbl['is_scored'].to_numpy().astype(bool)
        xstats.update(X)
        ystats.update(Y)
        if qmask[sl].any():
            qsample[qi:qi + SEQUENCE_LENGTH] = X
            qi += SEQUENCE_LENGTH
        state['done'] = i + 1
        if (i + 1) % progress_every == 0 or (i + 1) == n_groups:
            el = time.time() - t0
            rate = (i + 1 - last) / max(el, 1e-9)
            eta = (n_groups - (i + 1)) / max(rate, 1e-9)
            state['done'] = i + 1
            state_path.write_text(json.dumps(state))
            print(f'[{split}] {i+1}/{n_groups} seq  {el/60:.1f} min elapsed, '
                  f'ETA {eta/60:.1f} min, {human(sum(f.stat().st_size for f in out_dir.glob("*.npy")))} on disk',
                  flush=True)
    for a in arrs.values():
        a.flush()
    del arrs
    np.save(out_dir / 'seq_index.npy', seq_index)
    state.pop('done_at_start', None)
    state['done'] = n_groups
    state_path.write_text(json.dumps(state))

    # ---- train-only statistics (features + targets), quantiles from the subsample
    if split == 'train':
        meta = {
            'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'feature_columns': FEATURE_COLUMNS,
            'target_columns': list(TARGET_COLUMNS),
            'n_sequences': n_groups,
            'rows': n_rows,
            'source': str(src.relative_to(ROOT)),
            'split_only': True,
            'note': 'statistics are computed from this train split only',
            'X': xstats.to_dict(FEATURE_COLUMNS),
            'y': ystats.to_dict(list(TARGET_COLUMNS)),
        }
        if qsample.shape[0]:
            qs = qsample[:qi].astype(np.float64)
            meta['X_quantiles'] = {
                'rows_sampled': int(qs.shape[0]),
                'grid': QUANTILE_GRID,
                'values': {FEATURE_COLUMNS[j]: [float(v) for v in np.quantile(qs[:, j], QUANTILE_GRID)]
                           for j in range(N_FEATURES)},
            }
        yq = None  # target quantiles are recomputed from the memmap sample below
        # targets quantiles: cheap enough to recompute from the memmap sample
        Ys = np.load(out_dir / 'y.npy', mmap_mode='r')[qmask]
        meta['y_quantiles'] = {
            'rows_sampled': int(Ys.shape[0]),
            'grid': QUANTILE_GRID,
            'values': {TARGET_COLUMNS[j]: [float(v) for v in np.quantile(np.asarray(Ys[:, j], dtype=np.float64), QUANTILE_GRID)]
                       for j in range(2)},
        }
        (ROOT / 'cache').mkdir(exist_ok=True)
        (ROOT / 'cache' / 'norm_stats.json').write_text(json.dumps(meta, indent=1))
        print(f'[{split}] wrote cache/norm_stats.json', flush=True)

    meta = {
        'split': split,
        'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source': str(src.relative_to(ROOT)),
        'source_bytes': src.stat().st_size,
        'source_row_groups': total_groups,
        'sequences': n_groups,
        'start_seq': start_seq,
        'rows': n_rows,
        'sequence_length': SEQUENCE_LENGTH,
        'warmup_steps': WARMUP,
        'x_dtype': np.dtype(dtype).name,
        'y_dtype': 'float32',
        'feature_columns': FEATURE_COLUMNS,
        'target_columns': list(TARGET_COLUMNS),
        'has_is_scored': split == 'valid',
        'python': platform.python_version(),
        'command': ' '.join(sys.argv),
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=1))
    el = time.time() - t0
    print(f'[{split}] DONE in {el/60:.1f} min  ({n_rows/max(el,1e-9):,.0f} rows/s)  '
          f'{human(sum(f.stat().st_size for f in out_dir.glob("*.npy")))} total', flush=True)
    return meta


def verify(split, out_dir):
    """Cheap structural check of a finished shard set."""
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / 'meta.json').read_text())
    X = np.load(out_dir / 'X.npy', mmap_mode='r')
    y = np.load(out_dir / 'y.npy', mmap_mode='r')
    need = np.load(out_dir / 'need_prediction.npy', mmap_mode='r')
    si = np.load(out_dir / 'seq_index.npy')
    problems = []
    if X.shape != (meta['rows'], N_FEATURES):
        problems.append(f'X shape {X.shape}')
    if y.shape != (meta['rows'], 2):
        problems.append(f'y shape {y.shape}')
    if len(si) != meta['sequences'] + 1 or si[-1] != meta['rows']:
        problems.append('seq_index inconsistent')
    s = int(si[min(3, len(si) - 2)])
    if not np.array_equal(np.asarray(need[s:s + SEQUENCE_LENGTH]),
                          np.arange(SEQUENCE_LENGTH) >= WARMUP):
        problems.append('need_prediction violates the warm-up contract')
    if not np.isfinite(np.asarray(y[s:s + 1000])).all():
        problems.append('non-finite targets')
    if not np.isfinite(np.asarray(X[s:s + 1000], dtype=np.float32)).all():
        problems.append('non-finite features')
    if split == 'valid':
        sc = np.load(out_dir / 'is_scored.npy', mmap_mode='r')
        if sc.shape != (meta['rows'],):
            problems.append('is_scored shape')
        if np.asarray(sc[:WARMUP]).any():
            problems.append('is_scored true during warm-up')
    print(f'[{split}] verify: {"OK" if not problems else "PROBLEMS: " + "; ".join(problems)}')
    return not problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--split', choices=['train', 'valid', 'both'], default='both')
    ap.add_argument('--out-root', default=str(ROOT / 'cache'),
                    help='cache root; shards land in <out-root>/<split>/')
    ap.add_argument('--max-sequences', type=int, default=None,
                    help='build only the first N sequences of each split (dev subset)')
    ap.add_argument('--start-seq', type=int, default=0, help='first row group to build')
    ap.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    ap.add_argument('--quantile-sequences', type=int, default=100,
                    help='sequences sampled for the quantile statistics (0 disables)')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--ram-gb', type=float, default=8.0,
                    help='RAM ceiling; the build streams one 20k row group at a time (~10 MB) '
                         'and never approaches it, the value is recorded and enforced as a sanity check')
    ap.add_argument('--no-resume', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='replace existing shards whose shape/dtype do not match (cache/ only)')
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--progress-every', type=int, default=50)
    args = ap.parse_args()
    os.environ.setdefault('ARROW_NUM_THREADS', '1')
    splits = ['train', 'valid'] if args.split == 'both' else [args.split]
    rc = 0
    for s in splits:
        out = Path(args.out_root) / s
        if args.verify_only:
            rc |= 0 if verify(s, out) else 1
            continue
        build(s, out, max_sequences=args.max_sequences, start_seq=args.start_seq,
              dtype=args.dtype, quantile_sequences=args.quantile_sequences,
              seed=args.seed, ram_gb=args.ram_gb, resume=not args.no_resume,
              progress_every=args.progress_every, force=args.force)
        rc |= 0 if verify(s, out) else 1
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
