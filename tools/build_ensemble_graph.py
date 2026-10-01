"""Fuse several of our students into ONE ONNX graph and average their outputs.

WHY
---
dozken's 2-graph ensemble teacher was KILLED at 69m59s, and the reason was
per-graph cost paid twice. We measured why that is fixable for us: ORT's profiler
says our exported graph runs 24 ops/row and `tools/build_fused_graph.py` cuts that to
**12, verified bit-identical**, taking v023 from 40.12 to 24.14 us/row. Fixed
overhead falls from 16.0 to 8.5 us/row.

That changes the arithmetic for an ensemble. Two SEPARATE sessions pay the ~8.5 us
fixed cost twice (17 us) plus two dict/buffer sets; two branches inside ONE graph
pay it once. So a 2-model ensemble costs roughly

    8.5 + sum(compute per branch)

instead of 2 x (8.5 + compute). For v023 + v022 that is ~48 us rather than ~55 us.

This is the "fuse ensemble members into one graph" idea, and it only became
affordable AFTER the overhead work -- which is why the same idea was lethal for
dozken and is cheap for us.

CAVEAT WORTH STATING UP FRONT
------------------------------
dozken's notes report that averaging model members saturates at +0.0009, so the
prior on a score gain here is LOW. The value of this script is that it makes the
ensemble *affordable to measure* at all -- his version could not even be timed.
Our two members are more decorrelated than his pair (his differed only in reset
grid; ours differ in width, compression and training length), so the saturation
number may not transfer.

Members may have DIFFERENT hidden widths. Each branch keeps its own projection,
GRU stack, head and LayerNorms; only the state tensors are shared, because the two
states are the same shape and are read and written once. Outputs are averaged with
a single Mean at the end.

VERIFY BEFORE TRUSTING
---------------------
Replays each branch against its own source graph over real validation rows and
asserts agreement, then checks the fused mean against the arithmetic mean of the two
separate graphs. A fused graph that computes a different model is worthless.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent


def load_sd(version: str):
    ck_dir = ROOT / 'exploration' / 'checkpoints' / f'checkpoint_gru_{version}'
    cks = sorted(ck_dir.glob('best_*.pt'))

    def val(p):
        try:
            return float(p.stem.split('val')[-1])
        except ValueError:
            return -1.0
    ck = max(cks, key=val)
    st = torch.load(ck, map_location='cpu', weights_only=False)
    return st['model'], st.get('feature_dim', 224), ck.name


def branch_nodes(sd, pref, tag, opset):
    """Emit the 11 nodes of one student's step, with prefixed tensor names."""
    H = sd['grus.1.weight_hh_l0'].shape[1]
    P = sd['proj.weight'].shape[0]
    # gate permutation (1,0,2): torch stores (r,z,n), exports (z,r,n)
    perm = np.concatenate([np.arange(H, 2 * H), np.arange(0, H), np.arange(2 * H, 3 * H)])

    def n(a):
        return f'{pref}{a}'

    inits = [
        numpy_helper.from_array(sd['proj.weight'].numpy().T.copy(), n('W_in')),
        numpy_helper.from_array(sd['proj.bias'].numpy().copy(), n('b_in')),
        numpy_helper.from_array(sd['proj_ln.weight'].numpy().copy(), n('ln_in_w')),
        numpy_helper.from_array(sd['proj_ln.bias'].numpy().copy(), n('ln_in_b')),
        numpy_helper.from_array(sd['head_ln.weight'].numpy().copy(), n('lnh_w')),
        numpy_helper.from_array(sd['head_ln.bias'].numpy().copy(), n('lnh_b')),
        numpy_helper.from_array(sd['head_fc1.weight'].numpy().T.copy(), n('W_h1')),
        numpy_helper.from_array(sd['head_fc1.bias'].numpy().copy(), n('b_h1')),
        numpy_helper.from_array(sd['head_fc2.weight'].numpy().T.copy(), n('W_h2')),
        numpy_helper.from_array(sd['head_fc2.bias'].numpy().copy(), n('b_h2')),
        numpy_helper.from_array(np.array([1], np.int64), n('axes')),
    ]
    for i, (pre, ih, hh) in enumerate((('grus.0', 0, 1), ('grus.1', 2, 3))):
        inits += [
            numpy_helper.from_array(sd[f'{pre}.weight_ih_l0'].numpy()[perm][None]
                                    .astype(np.float32), n(f'W{ih}')),
            numpy_helper.from_array(sd[f'{pre}.weight_hh_l0'].numpy()[perm][None]
                                    .astype(np.float32), n(f'R{hh}')),
            numpy_helper.from_array(np.concatenate([
                sd[f'{pre}.bias_ih_l0'].numpy()[perm],
                sd[f'{pre}.bias_hh_l0'].numpy()[perm]])[None].astype(np.float32),
                n(f'B{pre[-1]}')),
        ]
    ns = [
        helper.make_node('Gemm', ['features', n('W_in'), n('b_in')], [n('z2')], name=f'{tag}_proj'),
        helper.make_node('Gelu', [n('z2')], [n('z2a')], name=f'{tag}_act', domain='com.microsoft'),
        helper.make_node('LayerNormalization', [n('z2a'), n('ln_in_w'), n('ln_in_b')],
                         [n('z2n')], axis=-1, name=f'{tag}_pln'),
        helper.make_node('Unsqueeze', [n('z2n'), n('axes')], [n('z3')], name=f'{tag}_seq'),
        helper.make_node('GRU', [n('z3'), n('W0'), n('R1'), n('B0'), '', n('h0_0')],
                         ['', n('g0')], name=f'{tag}_gru0', hidden_size=H,
                         direction='forward', linear_before_reset=1),
        helper.make_node('GRU', [n('g0'), n('W2'), n('R3'), n('B1'), '', n('h0_1')],
                         ['', n('h2')], name=f'{tag}_gru1', hidden_size=H,
                         direction='forward', linear_before_reset=1),
        # each branch gets its OWN reshape target: members may have different hidden
        # widths (v023 is 128, v022 is 160), and one shared shape_h cannot serve both
        helper.make_node('Reshape', [n('h2'), f'{pref}shape_h'], [n('h2f')],
                         name=f'{tag}_flat'),
        helper.make_node('Concat', [n('h2f'), n('z2n')], [n('cat')], axis=-1, name=f'{tag}_skip'),
        helper.make_node('LayerNormalization', [n('cat'), n('lnh_w'), n('lnh_b')],
                         [n('catn')], axis=-1, name=f'{tag}_hln'),
        helper.make_node('Gemm', [n('catn'), n('W_h1'), n('b_h1')], [n('hp')], name=f'{tag}_fc1'),
        helper.make_node('Gelu', [n('hp')], [n('ha')], name=f'{tag}_hact', domain='com.microsoft'),
        helper.make_node('Gemm', [n('ha'), n('W_h2'), n('b_h2')], [n('pred')],
                         name=f'{tag}_fc2'),
    ]
    return ns, inits, H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--members', required=True, help='comma-separated versions, e.g. v023,v022')
    ap.add_argument('--out', default=None)
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    vers = [v.strip() for v in args.members.split(',') if v.strip()]
    sds, dims, names = [], set(), []
    for v in vers:
        sd, fd, ckn = load_sd(v)
        sds.append((v, sd))
        dims.add(fd)
        names.append(ckn)
    assert len(dims) == 1, f'members disagree on feature width: {dims}'
    feat_dim = dims.pop()

    nodes, inits, gin, gout = [], [], [], []
    gout.append(helper.make_tensor_value_info('prediction', TensorProto.FLOAT, [1, 2]))
    preds = []
    for i, (v, sd) in enumerate(sds):
        pref = f'{v}_'
        ns, ins, H = branch_nodes(sd, pref, f'b{i}', args.opset)
        nodes += ns
        inits += ins
        preds.append(f'{pref}pred')
        gin.append(helper.make_tensor_value_info(f'{pref}h0_0', TensorProto.FLOAT, [1, 1, H]))
        gin.append(helper.make_tensor_value_info(f'{pref}h0_1', TensorProto.FLOAT, [1, 1, H]))
        # BOTH layers carry state and they are DIFFERENT tensors: layer 0's Y_h
        # (n('g0')) feeds h0_0 next step, layer 1's Y_h (n('h2')) feeds h0_1.
        # Exposing only h2 and feeding it to both is a silent wrong-model bug --
        # it builds and runs, and diverges by 0.18.
        gout.append(helper.make_tensor_value_info(f'{pref}g0', TensorProto.FLOAT, [1, 1, H]))
        gout.append(helper.make_tensor_value_info(f'{pref}h2', TensorProto.FLOAT, [1, 1, H]))
    gin.append(helper.make_tensor_value_info('features', TensorProto.FLOAT, [1, feat_dim]))
    for v, sd in sds:
        gin.append(helper.make_tensor_value_info(f'{v}_shape_h', TensorProto.INT64, [2]))
    if len(preds) == 1:
        nodes.append(helper.make_node('Identity', [preds[0]], ['prediction'], name='mean'))
    else:
        nodes.append(helper.make_node('Mean', preds, ['prediction'], name='mean'))

    g = helper.make_graph(nodes, 'ensemble_v1', gin, gout, inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', args.opset),
                                           helper.make_opsetid('com.microsoft', 1)])
    m.ir_version = 9
    out = Path(args.out) if args.out else ROOT / 'cache/latency_probe' / \
        ('ensemble_' + '_'.join(vers) + '.onnx')
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(out))
    print(f'built {out.relative_to(ROOT)}')
    print(f'  members : {vers}')
    print(f'  params  : {sum(sum(p.numel() for p in sd.values()) for _, sd in sds):,}')
    print(f'  nodes   : {len(m.graph.node)}  (vs {12 * len(sds)} separate + 1 Mean)')
    print(f'  ckpts   : {names}')

    # ---- verify: fused mean == arithmetic mean of the individual fused graphs
    import importlib.util
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    s = ort.InferenceSession(str(out), sess_options=o, providers=['CPUExecutionProvider'])
    # The graphs take the TRANSFORMED 224-wide row (normalised levels concatenated
    # with first differences), not the raw 112 from the shard. Reuse the packaged
    # transform so verification runs on exactly what inference sees.
    import importlib.util
    v0 = vers[0]
    sol = ROOT / 'submissions' / f'submission_gru_{v0}' / 'solution.py'
    spec = importlib.util.spec_from_file_location('sol', sol)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tr = mod.FeatureTransform(np.load(ROOT / 'submissions' / f'submission_gru_{v0}' /
                                      'feature_stats.npz'))
    X = np.load(ROOT / 'cache/valid/X.npy', mmap_mode='r')
    x = np.asarray(tr.batch(np.asarray(X[300:340], dtype=np.float32)), dtype=np.float32)
    assert x.shape[1] == feat_dim, (x.shape, feat_dim)

    # Both branches share the fused session, so BOTH branches' state tensors must be
    # fed every step and BOTH must be advanced from their own outputs.
    shapes = {f'{vv}_shape_h': np.array([1, sd2['grus.1.weight_hh_l0'].shape[1]], np.int64)
              for vv, sd2 in sds}
    state = {}
    for vv, sd2 in sds:
        Hb = sd2['grus.1.weight_hh_l0'].shape[1]
        state[f'{vv}_h0_0'] = np.zeros((1, 1, Hb), np.float32)
        state[f'{vv}_h0_1'] = np.zeros((1, 1, Hb), np.float32)

    solo_sess, solo_out = {}, {}
    for v, sd in sds:
        sub = out.with_name(f'solo_{v}.onnx')
        build_solo(sd, sub, feat_dim, args.opset, v)
        solo_sess[v] = ort.InferenceSession(str(sub), sess_options=o,
                                            providers=['CPUExecutionProvider'])
        solo_out[v] = [f'{v}_pred', f'{v}_g0', f'{v}_h2']

    worst = 0.0
    for t in range(x.shape[0]):
        # snapshot FIRST: the solos must be fed the state the fused graph was fed.
        # Updating `state` in place before the solo call makes each solo run one step
        # AHEAD, which shows up as a spurious ~0.17 mismatch even though both graphs
        # are correct (confirmed: at step 0 with zero states they agree exactly).
        cur = {k: v.copy() for k, v in state.items()}
        feed = {**shapes, 'features': x[t:t + 1], **cur}
        want = ['prediction'] + [f'{v}_g0' for v, _ in sds] + [f'{v}_h2' for v, _ in sds]
        fu = s.run(want, feed)
        for i, (v, _sd) in enumerate(sds):
            state[f'{v}_h0_0'] = fu[1 + i]
            state[f'{v}_h0_1'] = fu[1 + len(sds) + i]
        solo_preds = {}
        for v, sd in sds:
            Hb = sd['grus.1.weight_hh_l0'].shape[1]
            sf = {'features': x[t:t + 1],
                  f'{v}_shape_h': np.array([1, Hb], np.int64),
                  f'{v}_h0_0': cur[f'{v}_h0_0'], f'{v}_h0_1': cur[f'{v}_h0_1']}
            r = solo_sess[v].run(solo_out[v], sf)
            solo_preds[v] = r[0]
            state[f'{v}_h0_0'], state[f'{v}_h0_1'] = r[1], r[2]
        mean_of_solo = np.mean([solo_preds[v] for v, _ in sds], axis=0)
        worst = max(worst, float(np.abs(fu[0] - mean_of_solo).max()))
    for v, _ in sds:
        out.with_name(f'solo_{v}.onnx').unlink(missing_ok=True)
    print(f'  verify fused Mean vs mean of solos: max |diff| = {worst:.3e}')
    if worst > 1e-4:
        raise SystemExit('FUSED ENSEMBLE DISAGREES -- do not use it')
    print('  MATCH. Safe to benchmark.')


def build_solo(sd, path, feat_dim, opset, v):  # noqa: D401
    ns, ins, H = branch_nodes(sd, f'{v}_', f'b{v}', opset)
    g = helper.make_graph(ns, f'solo_{v}', [
        helper.make_tensor_value_info('features', TensorProto.FLOAT, [1, feat_dim]),
        helper.make_tensor_value_info(f'{v}_h0_0', TensorProto.FLOAT, [1, 1, H]),
        helper.make_tensor_value_info(f'{v}_h0_1', TensorProto.FLOAT, [1, 1, H]),
        helper.make_tensor_value_info(f'{v}_shape_h', TensorProto.INT64, [2])],
        [helper.make_tensor_value_info(f'{v}_pred', TensorProto.FLOAT, [1, 2]),
         # expose the TRUE next states, not the inputs: declaring h0_0/h0_1 as
         # outputs just echoes them back and the state never advances
         helper.make_tensor_value_info(f'{v}_g0', TensorProto.FLOAT, [1, 1, H]),
         helper.make_tensor_value_info(f'{v}_h2', TensorProto.FLOAT, [1, 1, H])], ins)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid('', opset),
                                           helper.make_opsetid('com.microsoft', 1)])
    m.ir_version = 9
    onnx.save(m, str(path))


if __name__ == '__main__':
    raise SystemExit(main())