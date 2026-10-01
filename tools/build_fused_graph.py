"""Hand-build a minimal ONNX graph for our 2-layer GRU student, with no exporter plumbing.

WHY
---
ORT's profiler says the exported v022 graph executes **24 ops per row**:

    Reshape 7, Gather 3, Gemm 3, GELU 2, LayerNorm 2, Transpose 2, GRU 2,
    Squeeze 2, Concat 1

Only ten of those are the model. The other fourteen -- Reshape, Gather, Transpose,
Squeeze -- are plumbing `torch.onnx.export` emits to move a single row between
`(B, T, in)` and `(T, B, in)` because `nn.GRU(batch_first=True)` expects the
transposed layout. At batch 1 each of those kernels does almost no arithmetic, so
the per-operator dispatch dominates. Measured fit across six models gives roughly
1 us per op, which is the right order for this shape.

dozken's graph is 18 nodes for the same architecture and pays none of it, because the
fused ONNX `GRU` operator takes `(seq, batch, input)` directly and `seq` is 1 here.

Measured on the same weights, same box, interleaved (`tools/bench_inference_paths.py`):

    A  session.run + string dict   38.27 us/row
    B  io_binding, preallocated    33.64 us/row     -4.63  (-12.1%)

so the binding is worth ~4.6 us. This module targets the other ~14 ops.

THE GRAPH
---------
    features (1,1,224) -> Gemm(W_in) -> 64
    -> GELU -> LayerNorm
    -> GRU(z, W0, R0, B0, state_0)  -> 160      (seq=1, batch=1)
    -> GRU(h, W1, R1, B1, state_1)  -> 160
    -> Reshape -> Concat with the 64-wide projection -> 224
    -> LayerNorm -> Gemm -> GELU -> Gemm -> prediction (1,1,2)

Eleven ops. The `prev`-free input path is kept exactly as the exported graph has it:
first differences are computed in the numpy `FeatureTransform`, not in the graph, so
the input is already 224-wide on arrival and there is no extra state tensor.

    - The projection output `z` is concatenated into the head, which is the skip
      path from input projection to readout. The Reshape exists only to drop the
      sequence axis before the Concat, matching dozken's layout.
    - Weights are pre-transposed so the Gemm needs no Transpose nodes; the GRU
      weights are stored exactly as `nn.GRU` lays them out (`[3H, in]`, `[3H, H]`,
      bias `[3H]` fused as `b_ih + b_hh`), which is the ONNX GRU convention.

VERIFY BEFORE TRUSTING
---------------------
`verify()` replays this graph and the original export side by side over real
validation rows and asserts agreement. A faster graph that computes something else
is worthless, so this is not optional.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
import yaml
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent


def load_sd(version: str, checkpoint: str | None = None):
    ck_dir = ROOT / 'exploration' / 'checkpoints' / f'checkpoint_gru_{version}'
    if checkpoint:
        ck = Path(checkpoint)
    else:
        cks = sorted(ck_dir.glob('best_*.pt'))
        if not cks:
            raise SystemExit(f'no best_*.pt in {ck_dir}')

        def val(p):
            try:
                return float(p.stem.split('val')[-1])
            except ValueError:
                return -1.0
        ck = max(cks, key=val)
    st = torch.load(ck, map_location='cpu', weights_only=False)
    return st['model'], st.get('feature_dim', 224), ck.name


def build(sd: dict, feat_dim: int = 224, opset: int = 17) -> onnx.ModelProto:
    W = lambda k: numpy_helper.from_array(sd[k].detach().numpy().astype(np.float32), k)
    # Gemm wants (out, in); nn.Linear stores (out, in) already, so W_in is used as-is.
    proj_w = numpy_helper.from_array(sd['proj.weight'].numpy().T.copy(), 'W_in')
    proj_b = numpy_helper.from_array(sd['proj.bias'].numpy().copy(), 'b_in')
    pln_w = numpy_helper.from_array(sd['proj_ln.weight'].numpy().copy(), 'ln_in_w')
    pln_b = numpy_helper.from_array(sd['proj_ln.bias'].numpy().copy(), 'ln_in_b')
    hln_w = numpy_helper.from_array(sd['head_ln.weight'].numpy().copy(), 'ln_head_w')
    hln_b = numpy_helper.from_array(sd['head_ln.bias'].numpy().copy(), 'ln_head_b')
    hf1_w = numpy_helper.from_array(sd['head_fc1.weight'].numpy().T.copy(), 'W_h1')
    hf1_b = numpy_helper.from_array(sd['head_fc1.bias'].numpy().copy(), 'b_h1')
    hf2_w = numpy_helper.from_array(sd['head_fc2.weight'].numpy().T.copy(), 'W_h2')
    hf2_b = numpy_helper.from_array(sd['head_fc2.bias'].numpy().copy(), 'b_h2')

    def gru(pre):
        # Name the initializers exactly as the node inputs reference them, so the
        # names cannot drift apart.
        wi = f'{pre}_weight_ih'
        wh = f'{pre}_weight_hh'
        bn = f'{pre}_bias'
        # ONNX GRU wants W as (num_directions, 3*hidden, input) and R as
        # (num_directions, 3*hidden, hidden) -- rank 3. nn.GRU stores them rank 2,
        # so the direction axis is added here rather than by a Reshape in the graph.
        # GATE ORDER. torch stores the 3H rows as (r, z, n) but exports them
        # permuted to (z, r, n) -- verified empirically against the exported graph:
        # exported W == weight_ih_l0[[H:2H, 0:H, 2H:3H]] exactly, 480/480 rows
        # matching verbatim. Replicating the permutation is mandatory; without it the
        # graph loads cleanly and computes a different model, off by ~2.0.
        Hh = sd[f'{pre}.weight_hh_l0'].shape[1]
        perm = np.concatenate([np.arange(Hh, 2 * Hh), np.arange(0, Hh),
                               np.arange(2 * Hh, 3 * Hh)])
        return [
            numpy_helper.from_array(sd[f'{pre}.weight_ih_l0'].numpy()[perm][None]
                                    .astype(np.float32), wi),
            numpy_helper.from_array(sd[f'{pre}.weight_hh_l0'].numpy()[perm][None]
                                    .astype(np.float32), wh),
            # ONNX GRU's B is (num_directions, 6*hidden): the first 3H is the input
            # bias and the second 3H the hidden bias, kept SEPARATE. nn.GRU stores
            # them as two separate [3H] tensors that it sums internally, so the two
            # halves are concatenated here rather than added -- summing them would
            # give shape [1,3H] and produce a silently different model.
            numpy_helper.from_array(np.concatenate([
                sd[f'{pre}.bias_ih_l0'].numpy()[perm],
                sd[f'{pre}.bias_hh_l0'].numpy()[perm]])[None]
                .astype(np.float32), bn),
        ], (wi, wh, bn)

    (w0, r0, b0), (n0_wi, n0_wh, n0_b) = gru('grus.0')
    (w1, r1, b1), (n1_wi, n1_wh, n1_b) = gru('grus.1')

    H = sd['grus.1.weight_hh_l0'].shape[1]
    P = sd['proj.weight'].shape[0]
    inits = [proj_w, proj_b, pln_w, pln_b, hln_w, hln_b, hf1_w, hf1_b, hf2_w, hf2_b,
             w0, r0, b0, w1, r1, b1]

    # features is declared rank-2 (1, 224) so the projection is a plain Gemm with
    # no leading Reshape, and the sequence axis is added by Unsqueeze *after* the
    # projection, which is exactly dozken's layout. The head's Concat needs rank-2
    # operands, so the rank-2 projection output z2 is kept and the unsqueezed copy
    # z3 is what feeds the GRU.
    axes = numpy_helper.from_array(np.array([1], np.int64), 'unsqueeze_axes')
    inits.append(axes)
    # linear_before_reset=1 on BOTH GRUs: this is what torch.onnx.export emits for
    # nn.GRU, and it is the exact attribute dozken's graph also carries. It means
    # n = tanh(W_n x + b_in + r * (R_n h + b_hh)) rather than torch's
    # r * (W_n x) + R_n h form. Defaulting it to 0 builds a valid graph that computes
    # a subtly different model -- it loaded fine and was off by 1.977.
    nodes = [
        helper.make_node('Gemm', ['features', 'W_in', 'b_in'], ['z2'], name='proj'),
        helper.make_node('Gelu', ['z2'], ['z2a'], name='proj_act',
                         domain='com.microsoft'),
        helper.make_node('LayerNormalization', ['z2a', 'ln_in_w', 'ln_in_b'],
                         ['z2n'], axis=-1, name='proj_ln'),
        helper.make_node('Unsqueeze', ['z2n', 'unsqueeze_axes'], ['z3'],
                         name='to_seq'),
        helper.make_node('GRU', ['z3', n0_wi, n0_wh, n0_b, '', 'state_0'],
                         ['Yg0', 'h1'], name='gru0', hidden_size=H,
                         direction='forward', linear_before_reset=1),
        # Layer 1 consumes layer 0's Y_h, not its Y. ONNX GRU's Y output is
        # (seq, num_directions, batch, hidden) = rank 4 here, which the next GRU
        # would reject as X; Y_h is (num_directions, batch, hidden) = (1,1,H), and
        # with seq=1, dirs=1, batch=1 that is exactly the rank-3 shape X wants.
        # With a single timestep the two carry the same value. This is dozken's
        # wiring and it is why his graph needs no Unsqueeze between the layers.
        helper.make_node('GRU', ['h1', n1_wi, n1_wh, n1_b, '', 'state_1'],
                         ['_Y1', 'h2'], name='gru1', hidden_size=H,
                         direction='forward', linear_before_reset=1),
        helper.make_node('Reshape', ['h2', 'shape_h'], ['h2f'], name='flat'),
        helper.make_node('Concat', ['h2f', 'z2n'], ['cat'], axis=-1, name='skip'),
        helper.make_node('LayerNormalization', ['cat', 'ln_head_w', 'ln_head_b'],
                         ['cat_n'], axis=-1, name='head_ln'),
        helper.make_node('Gemm', ['cat_n', 'W_h1', 'b_h1'], ['head_pre'], name='fc1'),
        helper.make_node('Gelu', ['head_pre'], ['head_act'], name='head_act',
                         domain='com.microsoft'),
        helper.make_node('Gemm', ['head_act', 'W_h2', 'b_h2'], ['prediction'],
                         name='fc2'),
    ]
    g_in = [
        helper.make_tensor_value_info('features', TensorProto.FLOAT, [1, feat_dim]),
        helper.make_tensor_value_info('state_0', TensorProto.FLOAT, [1, 1, H]),
        helper.make_tensor_value_info('state_1', TensorProto.FLOAT, [1, 1, H]),
        helper.make_tensor_value_info('shape_h', TensorProto.INT64, [2]),
    ]
    g_out = [
        helper.make_tensor_value_info('prediction', TensorProto.FLOAT, [1, 2]),
        helper.make_tensor_value_info('h1', TensorProto.FLOAT, [1, 1, H]),
        helper.make_tensor_value_info('h2', TensorProto.FLOAT, [1, 1, H]),
    ]
    graph = helper.make_graph(nodes, f'gru_fused_v1', g_in, g_out, inits,
                              value_info=[])
    # GELU is opset 20+ in the default domain, but onnxruntime ships it in the
    # com.microsoft domain at version 1, which is valid alongside an older default
    # opset. That is how dozken's graph gets Gelu while declaring opset 17, and it
    # keeps us compatible with the organiser's pinned onnxruntime 1.23.2.
    m = helper.make_model(graph, opset_imports=[
        helper.make_opsetid('', opset),
        helper.make_opsetid('com.microsoft', 1),
    ])
    m.ir_version = 9
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', default='v022')
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sd, feat_dim, ckn = load_sd(args.version, args.checkpoint)
    m = build(sd, feat_dim)
    out = Path(args.out) if args.out else \
        ROOT / 'cache' / 'latency_probe' / f'{args.version}_fused.onnx'
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(out))

    print(f'built {out.relative_to(ROOT)}  from {ckn}')
    print(f'  graph nodes : {len(m.graph.node)}  (exported graph was 42)')
    H = sd['grus.1.weight_hh_l0'].shape[1]
    P = sd['proj.weight'].shape[0]
    print(f'  feature_dim : {feat_dim}   hidden {H}   proj {P}   head_in {H + P}')

    # ---- verify against the exported graph on REAL rows
    import onnxruntime as ort
    ref = ROOT / 'submissions' / f'submission_gru_{args.version}' / 'model.onnx'
    X = np.load(ROOT / 'cache/valid/X.npy', mmap_mode='r')
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1

    def mk(p):
        return ort.InferenceSession(str(p), sess_options=o,
                                    providers=['CPUExecutionProvider'])

    a, b = mk(ref), mk(str(out))
    # The exported graph takes the TRANSFORMED 224-wide row (normalised features
    # concatenated with first differences), not the raw 112. Reuse the packaged
    # transform so verification runs on exactly what inference sees.
    import importlib.util
    sol = ROOT / 'submissions' / f'submission_gru_{args.version}' / 'solution.py'
    spec = importlib.util.spec_from_file_location('sol', sol)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tr = mod.FeatureTransform(np.load(ROOT / 'submissions' /
                                      f'submission_gru_{args.version}' / 'feature_stats.npz'))
    raw = np.asarray(X[500:540], dtype=np.float32)
    x = np.asarray(tr.batch(raw), dtype=np.float32)
    assert x.shape[1] == feat_dim, (x.shape, feat_dim)
    h0 = np.zeros((1, 1, H), np.float32)
    h1 = np.zeros((1, 1, H), np.float32)
    worst = 0.0
    for i in range(x.shape[0]):
        row = x[i:i + 1].reshape(1, 1, -1)
        ra = a.run(None, {'features': row, 'hidden_0': h0, 'hidden_1': h1})
        rb = b.run(None, {'features': row.reshape(1, -1), 'state_0': h0,
                          'state_1': h1, 'shape_h': np.array([1, H], np.int64)})
        worst = max(worst, float(np.abs(ra[0] - rb[0]).max()),
                    float(np.abs(ra[1] - rb[1]).max()),
                    float(np.abs(ra[2] - rb[2]).max()))
        h0, h1 = ra[1], ra[2]
    print(f'  verify vs exported graph over 40 real rows: max |diff| = {worst:.3e}')
    if worst > 1e-4:
        raise SystemExit('FUSED GRAPH DISAGREES -- do not use it')
    print('  MATCH. Safe to benchmark.')


if __name__ == '__main__':
    raise SystemExit(main())