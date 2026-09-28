#!/usr/bin/env python3
"""Measure µs/row of a solution or of a training script's inference section, 1 thread.

WHY
---
The scorer gives 60 minutes for the whole test set on 1 vCPU (AGENTS.md §1), i.e.
91.4 µs/row for the 39.4 M-row test set.  Architecture choice is therefore a
latency problem first (AGENTS.md §10), and nothing should be trained before its
per-row cost is known.  Random weights are fine for timing; the *data* must be
real, because a feature transform's cost depends on which branches are enabled.

Two modes:
  --solution path/to/solution.py                    time the packaged solution
  --script S --config C --untrained                 export a random-weight ONNX model
                                                   from the script's own export code
                                                   and time that (measures the real
                                                   architecture of that version)

The breakdown (`--breakdown`) attributes time to the feature transform vs the
model step, which is what tells you whether to shrink the model or the transform.

Examples
--------
    python tools/bench_latency.py --solution baseline/solution.py
    python tools/bench_latency.py --script exploration/models/train_gru_v000.py \
        --config exploration/version/version_gru_v000.yaml --untrained
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Pin the thread pools BEFORE numpy/onnxruntime are imported (AGENTS.md §10.1).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np  # noqa: E402
import metric_ref  # noqa: E402

WARMUP = 99


def pin_single_cpu():
    """Best-effort CPU affinity. Linux: sched_setaffinity; macOS: no-op."""
    if sys.platform.startswith('linux'):
        try:
            os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})
            return 'linux sched_setaffinity'
        except Exception as e:
            return f'linux affinity unavailable ({e})'
    return 'macOS: no per-process affinity control, onnxruntime threads pinned instead'


def load_real_rows(n_rows, source='cache'):
    """Real validation rows from the shard cache; falls back to the parquet file.

    Latency must be measured on realistic data: a zero row takes a different code
    path in some kernels and hides the transform's cost (AGENTS.md §13).
    """
    cache = ROOT / source / 'valid' / 'X.npy'
    if cache.exists():
        X = np.load(cache, mmap_mode='r')
        take = min(n_rows, X.shape[0])
        return np.ascontiguousarray(np.asarray(X[:take], dtype=np.float32))
    import pyarrow.parquet as pq
    from utils import FEATURE_COLUMNS
    pf = pq.ParquetFile(ROOT / 'datasets' / 'valid.parquet')
    rows, out = 0, []
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g)
        out.append(np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32))
        rows += 20000
        if rows >= n_rows:
            break
    return np.concatenate(out)[:n_rows]


class _Point:
    """Minimal stand-in for utils.DataPoint (avoids importing pyarrow here)."""
    __slots__ = ('seq_ix', 'step_in_seq', 'need_prediction', 'state')

    def __init__(self, seq_ix, step, need, state):
        self.seq_ix = seq_ix
        self.step_in_seq = step
        self.need_prediction = need
        self.state = state


def time_model(model, X, need_rows=None):
    """Feed rows in order, sequence by sequence, exactly like the scorer."""
    n_seq, seq_len = X.shape[0] // 20000, 20000
    lat = np.empty(n_seq * seq_len, dtype=np.float64)
    k = 0
    for s in range(n_seq):
        block = X[s * seq_len:(s + 1) * seq_len]
        for i in range(seq_len):
            p = _Point(7 * s + 1, i, i >= WARMUP, block[i])
            t0 = time.perf_counter()
            model.predict(p)
            lat[k] = (time.perf_counter() - t0) * 1e6
            k += 1
    return lat


def time_breakdown(model, X, transform=None):
    """Attribute time to the transform vs the ORT call, on the first sequences."""
    seq_len = 20000
    n_seq = min(4, X.shape[0] // seq_len)
    t_tr = t_ort = 0.0
    rows = 0
    for s in range(n_seq):
        block = X[s * seq_len:(s + 1) * seq_len]
        for i in range(seq_len):
            a = time.perf_counter()
            if transform is not None:
                x = transform.step(block[i])
            b = time.perf_counter()
            model.session.run(None, {'features': model._x, 'hidden_0': model._h0, 'hidden_1': model._h1})
            c = time.perf_counter()
            if transform is not None:
                np.copyto(model._x[0, 0], x)
            t_tr += b - a
            t_ort += c - b
            rows += 1
    return {'transform_us': t_tr / rows * 1e6, 'onnx_us': t_ort / rows * 1e6,
            'rows_measured': rows}


def report(name, lat, rows_measured, extra=None):
    p = np.percentile(lat, [50, 90, 99])
    rep = metric_ref.budget_report(p[0], p[2])
    print(f'\n=== {name} ===')
    print(f'rows measured        : {rows_measured:,} (warm-up calls discarded: {WARMUP * 4})')
    print(f'µs/row  mean         : {lat.mean():.2f}')
    print(f'µs/row  p50/p90/p99  : {p[0]:.2f} / {p[1]:.2f} / {p[2]:.2f}')
    print(f'budget (60 min / {rep["rows"]:,} rows): {rep["budget_us_per_row"]:.2f} µs/row '
          f'| 75% target: {rep["target_us_per_row"]:.2f} µs/row')
    print(f'projected total      : {rep["projected_min_p50"]:.1f} min at p50, '
          f'{rep["projected_min_p99"]:.1f} min at p99  (limit 60 min)')
    print(f'GATE  p50 <= 75% : {"PASS" if rep["pass_p50_75pct"] else "FAIL"}   '
          f'p99 <= 75% : {"PASS" if rep["pass_p99_75pct"] else "FAIL"}')
    if extra:
        for k, v in extra.items():
            print(f'{k:20}: {v}')
    return rep


def bench_solution(path, rows, data_source, breakdown, quiet=False):
    spec = importlib.util.spec_from_file_location('bench_solution_mod', path)
    mod = importlib.util.module_from_spec(spec)
    t0 = time.time()
    spec.loader.exec_module(mod)
    model = mod.PredictionModel()
    setup_s = time.time() - t0
    X = load_real_rows(rows, data_source)
    lat = time_model(model, X)
    lat = lat[WARMUP * 4:]                      # drop the first sequence's warm-up
    extra = {'model setup s': round(setup_s, 2)}
    if breakdown and hasattr(model, 'transform'):
        bd = time_breakdown(model, X[:80000], model.transform)
        extra['transform µs/row'] = round(bd['transform_us'], 2)
        extra['onnx µs/row'] = round(bd['onnx_us'], 2)
    return report(str(path), lat, len(lat), extra)


def bench_script(script, config, rows, data_source, outdir, breakdown, quiet=False):
    """Export a random-weight model from the training script's own code, then time it."""
    import yaml
    out = Path(outdir)
    if out.exists():
        subprocess.run(['rm', '-rf', str(out)], check=True)
    code = f'''
import sys, json, yaml
sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(ROOT / 'tools')!r})
spec_path = {str(script)!r}
import importlib.util
spec = importlib.util.spec_from_file_location("trainscript", spec_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cfg = yaml.safe_load(open({str(config)!r}))
stats = yaml.safe_load(open({str(ROOT / 'cache' / 'norm_stats.json')!r}))
mu = [stats["X"][c]["mean"] for c in stats["feature_columns"]]
sd = [stats["X"][c]["std"] for c in stats["feature_columns"]]
import numpy as np
if cfg["features"].get("normalisation", "none") == "none":
    mu = [0.0]*112; sd = [1.0]*112
tr = m.build_transform_from_stats({{"mean": np.array(mu, np.float32), "std": np.array(sd, np.float32)}},
                                 cfg["features"])
import torch
torch.manual_seed(0)
model = m.GRUModel(tr.dim(), hidden=cfg["model"]["hidden"], layers=cfg["model"]["layers"],
                   input_proj=cfg["model"].get("input_proj") or 0,
                   dropout=cfg["model"].get("dropout", 0.0))
m.export_inference(model, tr, {str(out)!r}, opset=cfg["inference"]["opset"], device="cpu", verify=True)
print(json.dumps({{"n_params": m.count_params(model), "feature_dim": tr.dim()}}))
'''
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:])
        raise SystemExit('failed to export a random-weight model for timing')
    info = json.loads([l for l in r.stdout.splitlines() if l.startswith('{')][-1])

    sol = out / 'solution_bench.py'
    src = Path(script).read_text()
    body = src.split('# ===== BEGIN SOLUTION SECTION =====')[1].split('# ===== END SOLUTION SECTION =====')[0]
    sol.write_text('# generated by tools/bench_latency.py -- do not edit\n' + body)
    return bench_solution(sol, rows, data_source, breakdown,
                          extra={'n_params': info['n_params'], 'feature_dim': info['feature_dim'],
                                 'export dir': str(out.relative_to(ROOT))})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--solution', help='path to a solution.py')
    g.add_argument('--script', help='path to a training script (pair with --config --untrained)')
    ap.add_argument('--config', help='version yaml for --script')
    ap.add_argument('--untrained', action='store_true',
                    help='required with --script: random weights are fine for timing')
    ap.add_argument('--rows', type=int, default=200_000, help='rows to time (>= 200k recommended)')
    ap.add_argument('--data-source', default='cache', choices=['cache', 'parquet'],
                    help='where the real rows come from')
    ap.add_argument('--export-dir', default='cache/latency_probe',
                    help='scratch dir for the random-weight export (gitignored, regenerable)')
    ap.add_argument('--breakdown', action='store_true', help='also time transform vs onnx')
    ap.add_argument('--json', action='store_true', help='emit the report as JSON')
    args = ap.parse_args()

    print(f'affinity: {pin_single_cpu()}')
    if args.script:
        if not args.config or not args.untrained:
            raise SystemExit('--script requires --config and --untrained')
        rep = bench_script(args.script, args.config, args.rows, args.data_source,
                           args.export_dir, args.breakdown)
    else:
        rep = bench_solution(Path(args.solution), args.rows, args.data_source, args.breakdown)
    if args.json:
        print(json.dumps(rep))
    return 0 if rep['pass_p50_75pct'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
