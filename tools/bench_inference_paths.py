"""Measure where the ~24 us/row of fixed inference cost actually goes.

WHY THIS EXISTS
---------------
Fitting our six packager measurements gave

    us/row  =  23.4  +  0.0731 * (params/1000)

i.e. roughly 23 us/row that does not depend on model size, about half of v022's
46.83 us. ORT's own profiler says v022 executes **24 ops per row**. 23 us over 24 ops
is ~1 us per operator dispatch, which is the right order for small tensors at batch 1
where the kernel work is trivial and the dispatch is not.

If that is right, the cost is dominated by OPERATOR COUNT, not by parameter count and
not by the single Python call. v022's 24 ops break down as:

    Reshape 7, Gather 3, Gemm 3, GELU 2, LayerNorm 2, Transpose 2, GRU 2, Squeeze 2, Concat 1

Ten of those fourteen Reshape/Gather/Transpose/Squeeze ops are torch-exporter plumbing
for the batch-first GRU and exist to move one row between shapes. dozken's graph is 18
nodes and, more importantly, arranged so that none of that plumbing is needed.

WHAT THIS SCRIPT ESTABLISHES
----------------------------
Two candidate fixes, measured on the SAME weights, SAME graph, SAME process, SAME
box, interleaved and repeated. That control is the whole point: the historical
numbers were taken hours apart in different machine states and cannot be compared --
v011 (192,258 params, 18 nodes) recorded 28.00 us while v017 (116,546 params, the
SAME 18 nodes) recorded 29.04 us, which is impossible for compute and is pure
measurement noise. Nothing here repeats that mistake.

    A. baseline     -- what we ship today: session.run with a string-keyed dict
    B. io_binding   -- preallocated OrtValue buffers, double-buffered A/B so the
                       next step's inputs are already where the previous step left
                       them, no per-call allocation and no dict
    C. fused        -- a hand-built single graph with the reshape/gather/squeeze/
                       transpose plumbing removed, if the build succeeds

READING THE RESULT
------------------
`ops/row` and the fixed-cost intercept are what matter. If B alone recovers most of
the overhead, the dict and the per-call allocation were the problem and we can stop.
If C is much faster than B, the operator count is the problem and the fix is a
graph, not a binding. If neither moves it, the ~23 us is something else entirely and
the whole line of attack is wrong.

Nothing here changes any shipped artifact. It is a measurement harness.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent


def _session(path, threads=1):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry('session.intra_op.allow_spinning', '0')
    o.add_session_config_entry('session.inter_op.allow_spinning', '0')
    return ort.InferenceSession(str(path), sess_options=o,
                                providers=['CPUExecutionProvider'])


class Baseline:
    """What we ship today: session.run with a string-keyed dict.

    Allocates a fresh output array for the prediction and both hidden states on
    every call, and rebuilds the input dict every call.
    """

    def __init__(self, path, hidden):
        self.s = _session(path)
        self.h = hidden
        self.x = np.zeros((1, 1, 224), np.float32)
        self.h0 = np.zeros((1, 1, hidden), np.float32)
        self.h1 = np.zeros((1, 1, hidden), np.float32)
        self.p = np.zeros((1, 1, 2), np.float32)

    def step(self, row):
        self.x[0, 0] = row
        self.p, self.h0, self.h1 = self.s.run(
            None, {'features': self.x, 'hidden_0': self.h0, 'hidden_1': self.h1})
        return self.p[0, 0]


class IoBinding:
    """Preallocated OrtValue buffers, double-buffered A/B, no per-call allocation.

    Binding 0 reads state set A and writes state set B; binding 1 reads B and writes
    A. Alternating them each step means the state the previous step produced is
    already where the next step reads it, so nothing is copied. The input row goes
    in via update_inplace, which writes into the bound buffer rather than building a
    new array for ORT to read.

    This is dozken's pattern, and it is the one thing in this experiment we did not
    write ourselves.
    """

    def __init__(self, path, hidden, fused=False):
        self.s = _session(path)
        self.h = hidden
        self.fused = fused
        self.sn = ('state_0', 'state_1') if fused else ('hidden_0', 'hidden_1')
        # the hand-built graph names its recurrent outputs h1/h2
        self.on = ('h1', 'h2') if fused else ('next_hidden_0', 'next_hidden_1')
        self.x = np.zeros((1, 224), np.float32) if fused else np.zeros((1, 1, 224), np.float32)
        self.x_val = ort.OrtValue.ortvalue_from_numpy(self.x)
        self.p = np.zeros((1, 2), np.float32) if fused else np.zeros((1, 1, 2), np.float32)
        self.p_val = ort.OrtValue.ortvalue_from_numpy(self.p)
        shape = (1, 1, hidden)
        self.zero = np.zeros(shape, np.float32)
        self.bufs = [[ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, np.float32))
                      for _ in range(2)] for _ in range(2)]
        self.bindings = []
        for k in (0, 1):
            b = self.s.io_binding()
            b.bind_ortvalue_input('features', self.x_val)
            b.bind_ortvalue_input(self.sn[0], self.bufs[k][0])
            b.bind_ortvalue_input(self.sn[1], self.bufs[k][1])
            b.bind_ortvalue_output('prediction', self.p_val)
            b.bind_ortvalue_output(self.on[0], self.bufs[1 - k][0])
            b.bind_ortvalue_output(self.on[1], self.bufs[1 - k][1])
            self.bindings.append(b)
        if fused:
            sh = ort.OrtValue.ortvalue_from_numpy(np.array([1, hidden], np.int64))
            for b in self.bindings:
                b.bind_ortvalue_input('shape_h', sh)
        self.k = 0
        self.ro = ort.RunOptions()

    def reset(self):
        for v in self.bufs[self.k]:
            v.update_inplace(self.zero)

    def step(self, row):
        self.x_val.update_inplace(row.reshape(self.x.shape))
        self.s.run_with_iobinding(self.bindings[self.k], self.ro)
        self.k = 1 - self.k
        return self.p


def profile_ops(path, hidden, feat_dim=224, iters=30):
    """Count the operators ORT actually executes per row, after graph optimisation."""
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.enable_profiling = True
    o.profile_file_prefix = tempfile.mktemp()
    s = ort.InferenceSession(str(path), sess_options=o,
                             providers=['CPUExecutionProvider'])
    feed = {'features': np.zeros((1, 1, feat_dim), np.float32),
            'hidden_0': np.zeros((1, 1, hidden), np.float32),
            'hidden_1': np.zeros((1, 1, hidden), np.float32)}
    for _ in range(iters):
        s.run(None, feed)
    prof = json.load(open(s.end_profiling()))
    c = collections.Counter(e.get('args', {}).get('op_name')
                            for e in prof if e.get('cat') == 'Node')
    return c, sum(c.values()) / iters


def bench(runner, rows, repeats=5):
    """Median-of-repeats microseconds per row, on pre-generated data.

    Rows are generated once up front so data generation never lands inside a timed
    region, and the same array is replayed every repeat so cache state is comparable.
    """
    out = []
    for _ in range(repeats):
        if hasattr(runner, 'reset'):
            runner.reset()
        t0 = time.perf_counter()
        for r in rows:
            runner.step(r)
        out.append((time.perf_counter() - t0) / len(rows) * 1e6)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--onnx', default='submissions/submission_gru_v022/model.onnx')
    ap.add_argument('--hidden', type=int, default=160)
    ap.add_argument('--rows', type=int, default=20000)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--fused', default=None,
                    help='optional hand-built graph to compare against')
    args = ap.parse_args()

    path = ROOT / args.onnx
    rng = np.random.default_rng(0)
    rows = [rng.normal(0, 1, 224).astype(np.float32) for _ in range(args.rows)]

    print(f'target        : {args.onnx}')
    print(f'rows/repeat   : {args.rows:,}   repeats: {args.repeats}')
    print()

    counts, ops = profile_ops(path, args.hidden)
    print(f'ORT executes  : {ops:.0f} ops/row  {dict(counts)}')
    print()

    runners = [('A baseline (session.run + dict)', Baseline(path, args.hidden)),
               ('B io_binding (prealloc A/B)', IoBinding(path, args.hidden))]
    if args.fused:
        runners.append(('C fused graph + io_binding',
                        IoBinding(ROOT / args.fused, args.hidden, fused=True)))

    print(f'  {"variant":<34} {"median us/row":>13} {"min":>8} {"max":>8}')
    results = {}
    for name, r in runners:
        samples = bench(r, rows, args.repeats)
        med = statistics.median(samples)
        results[name] = med
        print(f'  {name:<34} {med:>13.2f} {min(samples):>8.2f} {max(samples):>8.2f}')

    base = results['A baseline (session.run + dict)']
    print()
    for name, med in results.items():
        if name.startswith('A'):
            continue
        d = base - med
        print(f'  {name} vs baseline: {d:+.2f} us/row  ({d / base * 100:+.1f}%)')
        if d > 0:
            # what the freed time is worth in parameters, at the fitted 0.0731 us/1000
            print(f'    that buys roughly {d / 0.0731 * 1000:,.0f} extra parameters')

    print()
    print('interleaving note: variants run back to back in one process on one idle')
    print('box, so these numbers ARE comparable to each other, unlike the historical')
    print('packager figures taken hours apart in different machine states.')


if __name__ == '__main__':
    raise SystemExit(main())