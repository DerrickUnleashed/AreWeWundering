"""Benchmark a fused single model against a fused N-member ensemble, apples to apples.

WHY THIS EXISTS
---------------
dozken's 2-graph ensemble teacher TIMED OUT on the grader (69m59s vs a 60m limit) and
was killed. The reason was per-graph cost paid twice: two ORT sessions means two
rounds of fixed overhead, two dict/buffer sets and two ORT spin-ups, on a 1-vCPU box.

We can fuse the members into ONE graph instead (`tools/build_ensemble_graph.py`,
25 nodes for v023+v022), which pays the fixed overhead once. This script measures
what that actually buys, in the same process on the same rows, with the same session
options -- the only comparison that is worth anything, because ORT spin-up alone is
enough to move these numbers by several microseconds.

WHAT IS AND IS NOT SETTLED HERE
-------------------------------
This measures COST only. Whether averaging two of our students is worth even 1 point
is a separate question, and dozken's own notes say member averaging saturates around
+0.0009 -- so the honest prior on the score gain is LOW. What this script changes is
that the ensemble becomes *affordable to test* at all, which his version could not
manage even to the point of a valid timing.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ensemble_graph as B  # noqa: E402

GRADER_FACTOR = 1.62       # measured: grader is 1.62x this machine (see AGENTS.md sec 1)
TEST_ROWS = 39.4e6
BUDGET_MIN = 60.0


def sess_options():
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry('session.use_per_session_threads', '1')
    return o


def load_transform(version):
    spec = importlib.util.spec_from_file_location(
        'sol', ROOT / 'submissions' / f'submission_gru_{version}' / 'solution.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stats = np.load(ROOT / 'submissions' / f'submission_gru_{version}' / 'feature_stats.npz')
    return mod.FeatureTransform(stats)


def bench(path, members, outputs, xb, warm=400):
    """members: [(version, hidden), ...]; outputs ordered as documented below."""
    shapes = {f'{v}_shape_h': np.array([1, h], np.int64) for v, h in members}
    state = {}
    for v, h in members:
        state[f'{v}_h0_0'] = np.zeros((1, 1, h), np.float32)
        state[f'{v}_h0_1'] = np.zeros((1, 1, h), np.float32)
    n = xb.shape[0]

    def feed(i):
        return {'features': xb[i:i + 1], **state, **shapes}

    s = ort.InferenceSession(str(path), sess_options=sess_options(),
                             providers=['CPUExecutionProvider'])
    for i in range(warm):
        r = s.run(outputs, feed(0))
        state.update(_advance(outputs, r, members))
    ts = np.empty(n, np.float64)
    for i in range(n):
        t0 = time.perf_counter_ns()
        r = s.run(outputs, feed(i))
        ts[i] = time.perf_counter_ns() - t0
        state.update(_advance(outputs, r, members))
    return ts / 1000.0


def _advance(outputs, r, members):
    """Map the requested outputs back onto the state feed for the next step.

    For one branch the requested outputs are [.., v_g0, v_h2] and the two layer
    states are DIFFERENT tensors: layer 0's Y_h feeds h0_0, layer 1's Y_h feeds h0_1.
    Collapsing them onto one tensor silently computes a different model, so the
    mapping is explicit here rather than positional.
    """
    out = {}
    for j, name in enumerate(outputs):
        if name.endswith('_g0'):
            out[name[:-3] + '_h0_0'] = r[j]
        elif name.endswith('_h2'):
            out[name[:-3] + '_h0_1'] = r[j]
    return out


def report(label, a):
    p50, p90, p99 = (np.percentile(a, q) for q in (50, 90, 99))
    mins = p50 * GRADER_FACTOR * TEST_ROWS / 1e6 / 60
    print(f'  {label:26s} p50 {p50:6.2f}  p90 {p90:6.2f}  p99 {p99:6.2f} us/row')
    print(f'  {"":26s} grader {p50 * GRADER_FACTOR:6.2f} us/row -> {mins:5.1f} min '
          f'({mins / BUDGET_MIN * 100:4.1f}% of 60)')
    return p50


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--members', default='v023,v022')
    ap.add_argument('--rows', type=int, default=2000)
    ap.add_argument('--probe', default='cache/latency_probe')
    args = ap.parse_args()

    vers = [v.strip() for v in args.members.split(',') if v.strip()]
    members, feat_dim = [], None
    for v in vers:
        sd, fd, _ = B.load_sd(v)
        members.append((v, int(sd['grus.1.weight_hh_l0'].shape[1])))
        feat_dim = fd or feat_dim
    probe = ROOT / args.probe
    probe.mkdir(parents=True, exist_ok=True)

    print(f'  members: {members}')

    # single fused model
    v0, h0 = members[0]
    solo0 = probe / f'solo_{v0}.onnx'
    B.build_solo(B.load_sd(v0)[0], solo0, feat_dim, 17, v0)
    ens = probe / ('ensemble_' + '_'.join(vers) + '.onnx')

    tr = load_transform(v0)
    X = np.load(ROOT / 'cache/valid/X.npy', mmap_mode='r')
    xb = np.asarray(tr.batch(np.asarray(X[400:400 + args.rows], dtype=np.float32)),
                    dtype=np.float32)

    a1 = bench(solo0, members[:1], [f'{v0}_pred', f'{v0}_g0', f'{v0}_h2'], xb)
    a2 = bench(ens, members,
               ['prediction'] + [f'{v}_g0' for v, _ in members] + [f'{v}_h2' for v, _ in members],
               xb)

    p1 = report(f'fused {v0} alone', a1)
    p2 = report(f'fused {len(members)}-member ensemble', a2)
    print(f'\n  marginal cost of the 2nd member: {p2 - p1:+.2f} us/row '
          f'({(p2 - p1) / p1 * 100:+.1f}%)')
    print('  NOTE: cost only. Score gain is unmeasured -- run the full-validation '
          'rescore before believing any leaderboard move.')


if __name__ == '__main__':
    raise SystemExit(main())