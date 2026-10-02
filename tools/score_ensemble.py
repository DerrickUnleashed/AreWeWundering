"""Score a fused ensemble of our own students on the FULL validation set.

WHY THIS IS THE EXPERIMENT
--------------------------
dozken measured that ensembling two models trained with DIFFERENT losses is worth
+0.0056 global WP at P(>0)=1.000 (95% CI [+0.0019, +0.0100]), and that same-loss
pairs add nothing. His two-graph version of it TIMED OUT on the grader at 69m59s,
which is why he concluded "2-graph ensembles are out" and abandoned the idea.

That conclusion is specific to two ORT sessions, not to two models. We fuse the
members into ONE graph (tools/build_ensemble_graph.py), so the fixed per-session
cost is paid once. This script answers the question his timeout prevented: does
the gain survive when the ensemble is actually shippable?

THE MEMBERS ARE CHOSEN FOR LOSS DIVERSITY, NOT SEED DIVERSITY
--------------------------------------------------------------
  v028  distill_weight 1.0                          full_valid_wp 0.686510
  v029  distill_weight 0.5, tail-weighted (gamma 2) full_valid_wp 0.684683

Same architecture, same 225,922 params, same hidden 128, same state_reset 4000 --
so the fusion is cheap and the comparison is clean -- but genuinely different loss
functions. That is the axis dozken found carries the gain.

We average RAW outputs. dozken's own notes record that z-scoring members before
averaging inflated output std 1.55x and pushed 14.30% of scored rows past the
metric's +/-2 clip, and that this artifact is what produced his earlier
"ensembling is dead" conclusion. The metric clips, so scale is load-bearing.

HONEST PRIOR
------------
Mixed, and it should be said plainly. dozken's distil_ens (0.6585 public) did NOT
beat his distil160 (0.6588), so on HIS pair the ensemble's advantage failed to
transfer through distillation. Against that, his measured ensemble gain was real
and strongly significant; it just did not survive compression. The fused-inference
route keeps both members intact, so it avoids the failure mode his distillation
had to overcome. Unresolved either way -- which is why this is measured rather than
assumed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
from metric_ref import global_wp, weighted_pearson  # noqa: E402

CLIP = 2.0


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=o,
                                providers=['CPUExecutionProvider'])


def load_transform(version):
    import importlib.util
    p = ROOT / 'submissions' / f'submission_gru_{version}' / 'solution.py'
    spec = importlib.util.spec_from_file_location(f'sol_{version}', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stats = np.load(ROOT / 'submissions' / f'submission_gru_{version}' /
                    'feature_stats.npz')
    return mod.FeatureTransform(stats)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--members', default='v028,v029')
    ap.add_argument('--n-sequences', type=int, default=None,
                    help='default: all 1,873 (the official full-validation number)')
    ap.add_argument('--chunk', type=int, default=1000)
    ap.add_argument('--out', default=None, help='optional .npy of the mean prediction')
    args = ap.parse_args()

    vers = [v.strip() for v in args.members.split(',') if v.strip()]
    X = np.load(ROOT / 'cache/valid/X.npy', mmap_mode='r')
    Y = np.load(ROOT / 'cache/valid/y.npy', mmap_mode='r')
    S = np.load(ROOT / 'cache/valid/is_scored.npy', mmap_mode='r')
    N = np.load(ROOT / 'cache/valid/need_prediction.npy', mmap_mode='r')
    n_seq_all = X.shape[0] // 20_000
    n_seq = args.n_sequences or n_seq_all
    rows = n_seq * 20_000
    print(f'  members      : {vers}')
    print(f'  sequences    : {n_seq} of {n_seq_all}'
          f'{"  (OFFICIAL full-validation number)" if n_seq == n_seq_all else "  (SUBSET)"}')

    tr = load_transform(vers[0])
    sess = [session(ROOT / 'submissions' / f'submission_gru_{v}' / 'model.onnx')
            for v in vers]
    sr = int(np.load(ROOT / 'submissions' / f'submission_gru_{vers[0]}' /
                     'feature_stats.npz')['state_reset'])
    print(f'  state_reset  : {sr}')

    P = np.empty((len(sess), rows, 2), np.float32)
    t0 = time.time()
    for s in range(n_seq):
        a, b = s * 20_000, (s + 1) * 20_000
        raw = np.asarray(X[a:b], np.float32)
        xb = np.asarray(tr.batch(raw), np.float32)[:, None, :]   # (T,1,224)
        for k, ss in enumerate(sess):
            h0 = np.zeros((1, 1, 128), np.float32)
            h1 = np.zeros((1, 1, 128), np.float32)
            buf = np.empty((xb.shape[0], 1, 2), np.float32)
            for c in range(0, xb.shape[0], args.chunk):
                e = min(c + args.chunk, xb.shape[0])
                # replicate serving exactly: state is cleared every `sr` rows, and
                # every row updates state, including warm-up
                for i in range(c, e):
                    if i % sr == 0:
                        h0[:] = 0.0
                        h1[:] = 0.0
                    # graph wants rank-3 [1,1,224]; xb[i] is [1,224]
                    o = ss.run(['prediction', 'next_hidden_0', 'next_hidden_1'],
                               {'features': xb[i][None], 'hidden_0': h0,
                                'hidden_1': h1})
                    buf[i, 0, :] = o[0][0, 0]
                    h0, h1 = o[1], o[2]
            P[k, a:b] = buf[:, 0, :]
        if (s + 1) % 200 == 0:
            el = time.time() - t0
            print(f'    seq {s+1}/{n_seq}  {el:6.1f}s  eta {el/(s+1)*(n_seq-s-1):6.1f}s',
                  flush=True)

    y = np.asarray(Y[:rows], np.float32)
    m = np.asarray(S[:rows], bool)
    print(f'\n  scored rows  : {int(m.sum()):,}')

    def rep(name, p):
        # exactly the reference metric, on exactly the mask the scorer uses
        mean = global_wp(y, p.astype(np.float32), N[:rows], m)
        sel = m & N[:rows].astype(bool)
        per = [weighted_pearson(y[sel, k], p[sel, k]) for k in range(2)]
        print(f'  {name:22s} t0 {per[0]:.6f}  t1 {per[1]:.6f}  MEAN {mean:.6f}')
        return mean

    print()
    solo = []
    for k, v in enumerate(vers):
        solo.append(rep(f'{v} alone', P[k]))
    mean = P.mean(0)
    ens = rep(f'MEAN of {len(vers)}', mean)

    print(f'\n  ensemble gain over best member: {ens - max(solo):+.6f}')
    print(f'  dozken measured +0.0056 at P(>0)=1.000 for a loss-diverse pair.')
    print(f'  our 2-sigma noise floor is  +/-0.000317')
    if args.out:
        np.save(ROOT / args.out, mean)
        print(f'  wrote {args.out}')


if __name__ == '__main__':
    raise SystemExit(main())
