#!/usr/bin/env python3
"""Framework template / smoke-test experiment for the Alfa Connectome repo.

This file is a fully working, minimal, *stateful* GRU experiment.  It exists to
prove every code path of the framework (AGENTS.md §5/§7) end to end and to serve
as the parent of the first real experiments.  `gru_v000` is a framework smoke
test, not a result.

Nothing below is clever: the point is that it is correct, reproducible and that
the part between the SOLUTION markers is byte-identical to what ships in
`solution.py`.

    python tools/new_experiment.py --model gru --version v000
    python tools/bench_latency.py --script exploration/models/train_gru_v000.py \
        --config exploration/version/version_gru_v000.yaml --untrained
    python exploration/models/train_gru_v000.py --config exploration/version/version_gru_v000.yaml --smoke
"""
from __future__ import annotations

# ===== SECTION: TRAINING IMPORTS & UTILITIES =====
import argparse
import json
import math
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import metric_ref  # noqa: E402  (tools/metric_ref.py: verbatim organiser metric + fast paths)

MODEL_NAME = 'gru'
VERSION = 'v011'
WARMUP = 99
SEQ_LEN = 20_000
N_FEATURES = 112
METRIC_CLIP = 2.0
TEST_ROWS = metric_ref.TEST_ROWS


class Tee:
    """Duplicate stdout+stderr into the run log, line buffered, flushing each write.

    WHY: AGENTS.md §8 requires the log to be the *exact* stdout+stderr of the run
    including tracebacks, so we install an excepthook rather than hoping nobody
    raises.
    """

    def __init__(self, path, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(self.path, 'a' if resume else 'w', buffering=1, encoding='utf-8')
        self.stdout, self.stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self
        if not resume:
            self.stream.write(f'### new run, log opened {time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')
        else:
            self.stream.write(f'\n### resume, log reopened {time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')
        sys.excepthook = self.excepthook

    def write(self, s):
        self.stdout.write(s)
        self.stream.write(s)
        self.stream.flush()

    def flush(self):
        self.stdout.flush()
        self.stream.flush()

    def excepthook(self, etype, value, tb):
        import traceback
        text = ''.join(traceback.format_exception(etype, value, tb))
        self.write(text)
        self.stream.flush()
        self.stdout.flush()

    def close(self):
        # restore the hooks BEFORE closing the file, otherwise the interpreter's
        # final traceback print hits a closed stream ("I/O operation on closed file")
        sys.excepthook = sys.__excepthook__
        sys.stdout, sys.stderr = self.stdout, self.stderr
        try:
            self.stream.close()
        except Exception:
            pass


def reproducibility_header(cfg, command):
    try:
        import onnxruntime as ort
        ort_v = ort.__version__
    except Exception:
        ort_v = 'not installed'
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                                text=True, timeout=10).stdout.strip() or 'n/a'
    except Exception:
        commit = 'n/a'
    print('=' * 78, flush=True)
    print(f'command            : {command}', flush=True)
    print(f'timestamp          : {time.strftime("%Y-%m-%dT%H:%M:%S%z")}', flush=True)
    print(f'host               : {platform.platform()}  python {platform.python_version()}', flush=True)
    print(f'packages           : torch {torch.__version__}  numpy {np.__version__}  onnxruntime {ort_v}', flush=True)
    print(f'cuda               : {torch.cuda.is_available()}   mps: {getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()}', flush=True)
    print(f'git commit         : {commit}', flush=True)
    print(f'seed               : {cfg["training"]["seed"]}', flush=True)
    print(f'log                : exploration/logs/log_{MODEL_NAME}_{VERSION}.log', flush=True)
    print('config:', flush=True)
    print(yaml.safe_dump(cfg, sort_keys=False), flush=True)
    print('=' * 78, flush=True)


def write_yaml_results(path, cfg, status=None, model_updates=None, results=None, artifacts=None,
                       clear_results=()):
    """Only identity.status / model.param_count / results.* / artifacts.* are written back.

    `results` entries whose value is None are IGNORED, so a value written by an
    earlier run survives -- that is deliberate for a resumed run, but it means a
    field can only be *revoked* through `clear_results`, which is why a subset
    rescore explicitly clears `full_valid_wp` (AGENTS.md §6).
    """
    data = yaml.safe_load(Path(path).read_text())
    if status is not None:
        data['identity']['status'] = status
    if model_updates:
        data['model'].update(model_updates)
    if results:
        data.setdefault('results', {}).update({k: v for k, v in results.items() if v is not None})
    for k in clear_results:
        data.setdefault('results', {})[k] = None
    if artifacts:
        data.setdefault('artifacts', {}).update({k: v for k, v in artifacts.items() if v is not None})
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def pick_device(requested):
    if requested and requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ===== BEGIN SOLUTION SECTION =====
# Everything between these markers is copied verbatim into the packaged
# solution.py.  It may import ONLY numpy, onnxruntime and the standard library:
# the scoring container is built from requirements.txt, which does not contain
# torch, yaml, pyarrow, pandas, scipy, matplotlib or tqdm.
import os as _os

import numpy as _np
import onnxruntime as _ort

MODEL_NAME = 'gru'
VERSION = 'v011'
WARMUP = 99
SEQ_LEN = 20000
METRIC_CLIP = 2.0
N_FEATURES = 112
ARTIFACT_DIR = _os.path.dirname(_os.path.abspath(__file__))


class FeatureTransform:
    """Row -> model input, with a streaming path and a vectorised path that agree.

    Configured from `feature_stats.npz` (written at export time), never from the
    yaml: yaml is not guaranteed to exist in the scoring container.

    Output layout, in order:
        1. base block   : raw features, n_in
        2. signed-log   : sign(x) * log1p(|x|) on the base block, n_in, optional
        3. cross-instr. : x[a] - x[b] for each configured pair, n_cross, optional
        4. differences  : first differences of the *normalised* base block at the
                         configured indices, n_diff, optional (causal: the first
                         row of a sequence has no predecessor and yields 0)
        5. step feature : step_in_seq / (SEQ_LEN - 1), optional
    Blocks 1-3 are jointly normalised per column when `normalisation != 'none'`.

    Causal by construction: `.step` touches only the current row and the previous
    row of the same sequence, and every accumulator is cleared by `reset()`, which
    the caller invokes whenever `seq_ix` changes.
    """

    def __init__(self, stats):
        self.base_mean = _np.asarray(stats['base_mean'], dtype=_np.float32).reshape(-1)
        self.base_std = _np.asarray(stats['base_std'], dtype=_np.float32).reshape(-1)
        self.base_std = _np.where(self.base_std > 1e-6, self.base_std, _np.float32(1.0))
        self.clip_sigma = float(stats['clip_sigma'])
        self.norm = str(stats['normalisation'])
        self.signed_log = bool(_np.asarray(stats['signed_log_volume']).reshape(-1).any())
        self.diff = _np.asarray(stats['first_differences'], dtype=_np.int32).reshape(-1)
        cross = _np.asarray(stats['cross_instrument'], dtype=_np.float32).reshape(-1, 2)
        self.cross = cross
        self.cross_a = cross[:, 0].astype(_np.int32) if cross.shape[0] else cross.astype(_np.int32)
        self.cross_b = cross[:, 1].astype(_np.int32) if cross.shape[0] else cross.astype(_np.int32)
        self.step_feature = int(stats['step_feature'])
        self.n_in = int(_np.asarray(stats['n_in']).reshape(-1)[0])
        self.n_cross = int(self.cross.shape[0])
        self.n_diff = int(self.diff.shape[0])
        self.n_base = self.n_in * (2 if self.signed_log else 1) + self.n_cross
        assert self.base_mean.shape[0] == self.n_base, (self.base_mean.shape, self.n_base)
        self.n_out = self.n_base + self.n_diff + self.step_feature
        # ---- preallocated: the hot path must not allocate
        self._buf = _np.zeros(self.n_out, dtype=_np.float32)
        self._base = _np.zeros(self.n_base, dtype=_np.float32)
        # holds the previous row's base block AFTER normalisation, because both
        # .step() and .batch() difference the normalised block. Storing the RAW row
        # here made .step() differ from .batch() by up to 7.2 absolute whenever
        # first_differences and a non-'none' normalisation were combined
        # (measured 2026-09-29) -- and .batch() is what trains while .step() is
        # what serves, i.e. a silent train/serve skew (AGENTS.md §2.2).
        self._prev = _np.zeros(self.n_base, dtype=_np.float32)
        self._diff_idx = self.diff.astype(_np.int32) if self.n_diff else self.diff
        self._step_i = 0
        self._have_prev = False
        # If every optional block is off and there is no normalisation, the
        # transform is the identity.  The organiser already ships standardised
        # features (AGENTS.md §14), so this is the common case and it lets
        # predict() skip two copies per row -- ~1.5 us of a 30 us budget.
        self.identity = (self.norm == 'none' and not self.signed_log
                         and self.n_cross == 0 and self.n_diff == 0
                         and self.step_feature == 0)

    # ------------------------------------------------------------------ helpers
    def reset(self):
        self._step_i = 0
        self._have_prev = False
        self._prev.fill(0.0)

    def _base_rows(self, X):
        """Raw (un-normalised) base block for a (n, n_in) array."""
        parts = [X]
        if self.signed_log:
            parts.append(_np.sign(X) * _np.log1p(_np.abs(X)))
        if self.n_cross:
            parts.append(X[:, self.cross_a] - X[:, self.cross_b])
        if len(parts) > 1:
            return _np.concatenate(parts, axis=1).astype(_np.float32)
        return _np.asarray(X, dtype=_np.float32)

    def _normalise(self, Z):
        if self.norm == 'none':
            return Z
        Y = (Z - self.base_mean) / self.base_std
        if self.clip_sigma > 0:
            _np.clip(Y, -self.clip_sigma, self.clip_sigma, out=Y)
        return Y

    def _diffs_from(self, Z, prev=None, have_prev=False):
        """First differences of the normalised base block.

        Z    : (n, n_base) already normalised, rows of ONE sequence in order
        prev : (n_base,) the previous row's normalised base block, or None
        Row 0 has no predecessor inside the sequence and yields 0, which is what
        `.step()` does on the first row after a reset -- that is what makes the two
        paths agree.

        NOTE: the within-sequence differences (rows 1..n-1) must ALWAYS be computed.
        An earlier version returned early when `have_prev` was False, and since
        `.batch()` passes have_prev=False, the whole first-difference block came out
        as zeros during training while `.step()` produced real differences at serve
        time -- a silent train/serve skew of up to 4.4 absolute (measured
        2026-09-29). See AGENTS.md §2.2.
        """
        n = Z.shape[0]
        D = _np.zeros((n, self.n_diff), dtype=_np.float32)
        if self.n_diff == 0 or n == 0:
            return D
        idx = self._diff_idx
        if n > 1:
            _np.subtract(Z[1:][:, idx], Z[:-1][:, idx], out=D[1:])
        if have_prev and prev is not None:
            D[0] = Z[0, idx] - prev[idx]
        return D

    def _step_rows(self, n):
        if not self.step_feature:
            return None
        return (_np.arange(n, dtype=_np.float32) / _np.float32(SEQ_LEN - 1))[:, None]

    # ------------------------------------------------------------------ public
    def batch(self, X):
        """Vectorised transform of a whole (n, 112) block of ONE sequence."""
        X = _np.asarray(X, dtype=_np.float32)
        Z = self._normalise(self._base_rows(X))
        D = self._diffs_from(Z)
        self._step_i = len(X)
        self._have_prev = bool(len(X))
        if self._have_prev:
            self._prev[:] = Z[-1]
        pieces = [Z, D]
        s = self._step_rows(len(X))
        if s is not None:
            pieces.append(s)
        out = _np.concatenate(pieces, axis=1) if len(pieces) > 1 else Z
        assert out.shape[1] == self.n_out, (out.shape, self.n_out)
        return _np.ascontiguousarray(out, dtype=_np.float32)

    def step(self, x_row):
        """Streaming transform of one row. Returns a view of a preallocated buffer.

        The returned array is REUSED on the next call; copy it if you need to keep
        it.  `PredictionModel.predict` feeds it straight into the ONNX session.
        """
        if self.identity:
            return _np.asarray(x_row, dtype=_np.float32).reshape(-1)
        x = _np.asarray(x_row, dtype=_np.float32).reshape(-1)
        base = self._base
        _np.copyto(base[:self.n_in], x)
        o = self.n_in
        if self.signed_log:
            _np.copyto(base[o:o + self.n_in], _np.sign(x) * _np.log1p(_np.abs(x)))
            o += self.n_in
        if self.n_cross:
            _np.subtract(x[self.cross_a], x[self.cross_b], out=base[o:o + self.n_cross])
            o += self.n_cross
        Z = self._normalise(base)
        buf = self._buf
        _np.copyto(buf[:self.n_base], Z)
        o = self.n_base
        if self.n_diff:
            if self._have_prev:
                _np.subtract(Z[self._diff_idx], self._prev[self._diff_idx],
                             out=buf[o:o + self.n_diff])
            else:
                buf[o:o + self.n_diff] = 0.0
            o += self.n_diff
        if self.step_feature:
            buf[o] = self._step_i / _np.float32(SEQ_LEN - 1)
        self._prev[:] = Z
        self._have_prev = True
        self._step_i += 1
        return buf

    def dim(self):
        return self.n_out


class PredictionModel:
    """Stateful single-step inference, exactly as the scorer calls it."""

    def __init__(self):
        stats = _np.load(_os.path.join(ARTIFACT_DIR, 'feature_stats.npz'))
        self.transform = FeatureTransform(stats)
        # Periodic state reset: 0 = never reset (one contiguous 20k context).
        # A positive N zeroes the recurrent state every N rows of a sequence, so the
        # model must emit a consistent level from a FRESH state at any position.
        # The training loop must use the same period or the model is scored far out
        # of regime.
        self.state_reset = int(_np.asarray(stats['state_reset']).reshape(-1)[0]) \
            if 'state_reset' in stats.files else 0
        options = _ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.use_per_session_threads = True
        options.add_session_config_entry('session.intra_op.allow_spinning', '0')
        options.add_session_config_entry('session.inter_op.allow_spinning', '0')
        self.session = _ort.InferenceSession(
            _os.path.join(ARTIFACT_DIR, 'model.onnx'), sess_options=options,
            providers=['CPUExecutionProvider'])
        # input signature is features[1, 1, F] / hidden_0[1, 1, H] (the baseline's
        # shape); the feature dimension is therefore the LAST axis, not index 1.
        meta = self.session.get_inputs()[0].shape
        self.feat_dim = int(meta[-1])
        hmeta = self.session.get_inputs()[1].shape
        self.hidden = int(hmeta[-1])
        self._x = _np.zeros((1, 1, self.feat_dim), dtype=_np.float32)
        self._p = _np.zeros(2, dtype=_np.float32)
        self._h0 = _np.zeros((1, 1, self.hidden), dtype=_np.float32)
        self._h1 = _np.zeros((1, 1, self.hidden), dtype=_np.float32)
        self.current_seq_ix = None
        self.previous_step = None

    def reset(self):
        self._h0.fill(0.0)
        self._h1.fill(0.0)
        self.transform.reset()
        self.previous_step = None

    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        if self.current_seq_ix != seq_ix:
            if step != 0:
                raise ValueError('a new sequence must start at step zero')
            self.current_seq_ix = seq_ix
            self.reset()
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError('rows must arrive in sequence order')
        self.previous_step = step
        # amortised cost: two buffer fills every `state_reset` rows, i.e. free
        if self.state_reset and step and not step % self.state_reset:
            self._h0.fill(0.0)
            self._h1.fill(0.0)
        if self.transform.identity:
            _np.copyto(self._x[0, 0], data_point.state)
        else:
            _np.copyto(self._x[0, 0], self.transform.step(data_point.state))
        pred, self._h0, self._h1 = self.session.run(
            None, {'features': self._x, 'hidden_0': self._h0, 'hidden_1': self._h1})
        if not data_point.need_prediction:
            return None
        # _np.isfinite(...).all() allocates a bool array per call (~1 us); scalar
        # comparisons are exact (NaN and inf both fail `abs(v) < 1e30`) and ~10x cheaper.
        p0 = pred[0, 0, 0]
        p1 = pred[0, 0, 1]
        if not (abs(p0) < 1e30 and abs(p1) < 1e30):
            raise ValueError('non-finite prediction')
        _np.clip(pred[0, 0], -METRIC_CLIP, METRIC_CLIP, out=self._p)
        return self._p
# ===== END SOLUTION SECTION =====


# ===== SECTION: MODEL (PyTorch) =====
class GRUModel(nn.Module):
    """input_proj (optional) -> stack of single-layer GRUs -> linear head(2).

    WHY A STACK OF `nn.GRU(num_layers=1)` INSTEAD OF ONE `nn.GRU(num_layers=L)`:
    at inference the graph must carry one ONNX `GRU` op per layer whose initial
    state is passed straight through, which is exactly what the organiser's
    `baseline.onnx` does (10 nodes total, hidden_0/hidden_1 in, next_hidden_0/1
    out).  Handing `nn.GRU(num_layers=2)` a packed (2, 1, H) state makes the
    exporter emit Split/Concat/Slice/Gather plumbing (50 nodes) and cost ~8 us/row
    more per call at batch 1 -- 30 % of this model's entire budget.  Measured, not
    assumed (AGENTS.md §10.4).

    The feature transform is applied OUTSIDE the model (in numpy, inside the
    solution section) so the same code runs at train and at serve time and so a
    linear part of the transform could later be folded into `input_proj` at export
    time.  `forward_chunk` returns a prediction for every step of the chunk.
    """

    def __init__(self, in_dim, hidden=128, layers=2, input_proj=0, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.input_proj = input_proj
        width = input_proj or in_dim
        self.proj = nn.Linear(in_dim, input_proj) if input_proj else nn.Identity()
        self.grus = nn.ModuleList([nn.GRU(width if i == 0 else hidden, hidden, 1,
                                         batch_first=True) for i in range(layers)])
        self.drop = nn.Dropout(dropout) if (dropout and layers > 1) else None
        self.head = nn.Linear(hidden, 2)

    def _run(self, z, h):
        """z (1, T, width); h = list of (1, B, hidden). Returns (1, T, width), h."""
        h = list(h)
        for i, gru in enumerate(self.grus):
            z, h[i] = gru(z, h[i])
            if self.drop is not None:
                z = self.drop(z)
        return z, h

    def forward_chunk(self, x, h):
        """x (B, T, in_dim) already transformed; h = list of (1, B, hidden)."""
        y, hn = self._run(self.proj(x), h)
        return self.head(y), hn

    def forward_step(self, x, h):
        """x (in_dim,) single row; h = list of (1, 1, hidden)."""
        y, hn = self._run(self.proj(x).view(1, 1, -1), h)
        return self.head(y[:, -1]), hn


class StepGraph(nn.Module):
    """Wrapper matching the ONNX I/O signature used by the baseline.

    inputs : features (1, 1, F), hidden_0 (1, 1, H), hidden_1 (1, 1, H)
    outputs: prediction (1, 1, 2), next_hidden_0, next_hidden_1
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, hidden_0, hidden_1):
        # the states are already (1, 1, H) -- the exact layout a single-layer GRU
        # wants -- so they pass through the graph untouched, as in baseline.onnx
        p, hn = self.model.forward_step(features[0, 0], [hidden_0, hidden_1])
        return p.reshape(1, 1, 2), hn[0], hn[1]


class IdentityTransformNn(nn.Module):
    """Placeholder kept out of the graph: the transform is applied in numpy on both
    the training and the serving side, so that the same code runs in both paths
    (AGENTS.md §2.2 train/serve parity).  Not used by the template; do not delete
    without checking that a future version does not want the transform fused."""


def export_inference(model, transform, out_dir, opset=17, verify=True, device='cpu',
                     state_reset=0):
    """Write model.onnx + feature_stats.npz and verify ONNX == PyTorch within 1e-4."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    fdim = transform.dim()
    # feature constants travel with the model: no yaml is read at inference time
    np.savez(out / 'feature_stats.npz',
             base_mean=transform.base_mean, base_std=transform.base_std,
             n_in=np.int32(transform.n_in),
             clip_sigma=np.float32(transform.clip_sigma),
             normalisation=np.str_(transform.norm),
             signed_log_volume=np.int32(1 if transform.signed_log else 0),
             first_differences=transform.diff,
             cross_instrument=transform.cross,
             step_feature=np.int32(transform.step_feature),
             state_reset=np.int32(state_reset))
    graph = StepGraph(model).to(device).eval()
    h = model.hidden
    dummy = (torch.zeros(1, 1, fdim, device=device),
             torch.zeros(1, 1, h, device=device),
             torch.zeros(1, 1, h, device=device))
    path = out / 'model.onnx'
    with torch.no_grad():
        # dynamo=False selects the TorchScript exporter.  The dynamo exporter is
        # the default in torch>=2.9 but needs the extra `onnxscript` package, which
        # is NOT in requirements.txt, so we keep the dependency-free path (it is
        # also what produced baseline.onnx with producer pytorch 2.5.1).
        torch.onnx.export(graph, dummy, str(path), opset_version=opset,
                          input_names=['features', 'hidden_0', 'hidden_1'],
                          output_names=['prediction', 'next_hidden_0', 'next_hidden_1'],
                          dynamic_axes=None, do_constant_folding=True, dynamo=False)
    if verify:
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(path), sess_options=o, providers=['CPUExecutionProvider'])
        rng = np.random.default_rng(0)
        worst = 0.0
        hh = [np.zeros((1, 1, h), np.float32), np.zeros((1, 1, h), np.float32)]
        with torch.no_grad():
            for _ in range(20):
                x = rng.normal(0, 1, (1, 1, fdim)).astype(np.float32)
                pt, th0, th1 = graph(torch.from_numpy(x),
                                     torch.from_numpy(hh[0]), torch.from_numpy(hh[1]))
                pr, hh[0], hh[1] = sess.run(None, {'features': x, 'hidden_0': hh[0], 'hidden_1': hh[1]})
                worst = max(worst, float(np.abs(pr[0, 0] - pt[0, 0].numpy()).max()),
                            float(np.abs(hh[0] - th0.numpy()).max()),
                            float(np.abs(hh[1] - th1.numpy()).max()))
        if worst > 1e-4:
            raise SystemExit(f'ONNX export verification FAILED: max |diff| = {worst:.3e} > 1e-4')
        print(f'export verified: ONNX == PyTorch, max |diff| = {worst:.3e}', flush=True)
    return path


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# ===== SECTION: DATA =====
class ShardStore:
    """Read-only view over the memmap shards produced by tools/make_shards.py."""

    def __init__(self, root):
        root = Path(root)
        self.root = root
        meta = json.loads((root / 'meta.json').read_text())
        self.meta = meta
        self.n_seq = meta['sequences']
        self.seq_len = meta['sequence_length']
        self.X = np.load(root / 'X.npy', mmap_mode='r')
        self.y = np.load(root / 'y.npy', mmap_mode='r')
        self.need = np.load(root / 'need_prediction.npy', mmap_mode='r')
        self.is_scored = (np.load(root / 'is_scored.npy', mmap_mode='r')
                          if (root / 'is_scored.npy').exists() else None)
        self.seq_index = np.load(root / 'seq_index.npy')

    def __len__(self):
        return self.n_seq

    def sequence(self, i, soft_mask=None):
        a, b = int(self.seq_index[i]), int(self.seq_index[i + 1])
        return (np.asarray(self.X[a:b], dtype=np.float32),
                np.asarray(self.y[a:b], dtype=np.float32),
                np.asarray(self.need[a:b], dtype=bool),
                None if soft_mask is None else soft_mask(a, b))


class SoftMaskStore:
    """P(is_scored) per row, produced by tools/train_mask_model.py.

    float16 and memory-mapped: 212 M rows x 2 B = 424 MB, the same order as the
    feature cache. Read-only and never touched at inference.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.p = np.load(self.path, mmap_mode='r')
        self.n = self.p.shape[0]

    def __call__(self, lo, hi):
        return np.asarray(self.p[lo:hi], dtype=np.float32)


class StatefulBatchStream:
    """B lanes, each walking one sequence chunk by chunk with its hidden state.

    WHY THIS EXISTS (AGENTS.md §12 Phase 3): inference sees one 20,000-step
    causal stream, so training must too.  A lane owns one sequence, keeps its
    hidden state across chunks, is reset only when it starts a NEW sequence, and
    produces no loss on its burn-in steps.

    `chunk_len` must divide `seq_len` (20,000) so that every chunk of every lane
    ends exactly on a sequence boundary.  Otherwise a lane would have to be cut
    mid-sequence and its state would silently wrap.  Divisors include
    100, 125, 200, 250, 400, 500, 625, 1000, 2000, 4000.
    """

    def __init__(self, store, batch_sequences, chunk_len, burn_in, seed=0, device='cpu',
                 n_sequences=None, thread_prefetch=True, state_reset=0, soft_mask=None):
        if store.seq_len % chunk_len:
            raise ValueError(f'chunk_len ({chunk_len}) must divide the sequence length '
                             f'({store.seq_len}) so lanes end on sequence boundaries')
        self.store = store
        self.B = batch_sequences
        self.T = chunk_len
        self.burn_in = burn_in
        self.chunks_per_seq = store.seq_len // chunk_len
        # A reset must land on a chunk boundary, otherwise the lane would be cut
        # mid-chunk. state_reset must therefore be a whole number of chunks.
        if state_reset:
            if store.seq_len % state_reset:
                raise ValueError(f'state_reset ({state_reset}) must divide the sequence '
                                 f'length ({store.seq_len})')
            if state_reset % chunk_len:
                raise ValueError(f'state_reset ({state_reset}) must be a multiple of '
                                 f'chunk_len ({chunk_len}) so resets land on chunk boundaries')
            if state_reset <= chunk_len:
                raise ValueError(
                    f'state_reset ({state_reset}) must be STRICTLY GREATER than chunk_len '
                    f'({chunk_len}). A reset at every chunk boundary discards all carried '
                    f'state, leaving a model with no memory beyond one chunk -- silently, '
                    f'and the error message would not otherwise say so. Use 2*chunk_len or '
                    f'more (e.g. 4000 with chunk_len 2000) to keep one chunk of carry.')
        self.device = device
        self.state_reset = state_reset
        self.soft_mask = soft_mask
        self.n = n_sequences if n_sequences is not None else len(store)
        self.rng = np.random.default_rng(seed)
        self.thread_prefetch = thread_prefetch
        self.lane_seq = [-1] * batch_sequences
        self.lane_chunk = [0] * batch_sequences
        self._new_lane()

    def _new_lane(self):
        for i in range(self.B):
            if self.lane_seq[i] < 0:
                self.lane_seq[i] = int(self.rng.integers(0, self.n))
                self.lane_chunk[i] = 0

    def _fetch(self):
        X = np.empty((self.B, self.T, N_FEATURES), dtype=np.float32)
        Y = np.empty((self.B, self.T, 2), dtype=np.float32)
        M = np.zeros((self.B, self.T), dtype=np.float32)
        S = None if self.soft_mask is None else np.empty((self.B, self.T), dtype=np.float32)
        R = np.zeros(self.B, dtype=bool)
        for i in range(self.B):
            seq = self.lane_seq[i]
            chunk = self.lane_chunk[i]
            x, y, need, sm = self.store.sequence(seq, self.soft_mask)
            a = chunk * self.T
            b = a + self.T
            X[i] = x[a:b]
            Y[i] = y[a:b]
            step = np.arange(a, b)
            M[i] = (step >= self.burn_in).astype(np.float32) * need[a:b].astype(np.float32)
            if S is not None:
                S[i] = sm[a:b]
            R[i] = (chunk == 0) or (self.state_reset
                                 and (chunk * self.T) % self.state_reset == 0)
            self.lane_chunk[i] = chunk + 1
            if self.lane_chunk[i] >= self.chunks_per_seq:
                self.lane_seq[i] = -1
        self._new_lane()
        return X, Y, M, R, S

    def __iter__(self):
        while True:
            yield self._fetch()

    def __next__(self):
        return self._fetch()


def prefetch(iterator, depth=3):
    """Background thread that runs the (I/O bound) shard fetch ahead of the GPU."""
    q = queue.Queue(maxsize=depth)

    def work():
        for item in iterator:
            q.put(item)
        q.put(None)

    t = threading.Thread(target=work, daemon=True)
    t.start()

    def gen():
        while True:
            item = q.get()
            if item is None:
                break
            yield item
    return gen()


class ValidationStream:
    """Whole validation sequences, deterministic order, no shuffling."""

    def __init__(self, store, batch_sequences, n_sequences=None):
        self.store = store
        self.B = batch_sequences
        n = n_sequences if n_sequences is not None else len(store)
        self.order = list(range(min(n, len(store))))

    def __iter__(self):
        for s in range(0, len(self.order), self.B):
            idx = self.order[s:s + self.B]
            X = np.empty((len(idx), self.store.seq_len, N_FEATURES), dtype=np.float32)
            Y = np.empty((len(idx), self.store.seq_len, 2), dtype=np.float32)
            M = np.empty((len(idx), self.store.seq_len), dtype=np.float32)
            S = np.empty((len(idx), self.store.seq_len), dtype=bool)
            for j, i in enumerate(idx):
                x, y, need, _ = self.store.sequence(i)
                X[j] = x
                Y[j] = y
                M[j] = need.astype(np.float32)
                S[j] = self.store.is_scored[i * self.store.seq_len:(i + 1) * self.store.seq_len] \
                    if self.store.is_scored is not None else need
                S[j] = S[j] & need
            yield torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(M), torch.from_numpy(S)


# ===== SECTION: LOSS & METRIC =====
def build_transform_from_stats(stats_dict, transform_cfg, n_features=N_FEATURES):
    """Turn plain train-only constants into the numpy FeatureTransform.

    The `base` block is the raw features; the optional signed-log and
    cross-instrument columns are appended and start out UN-normalised (identity
    mean 0 / std 1) until a version measures proper statistics for them.  That is
    deliberate: inventing statistics for a nonlinear column would silently change
    the model, and a follow-up version can add them from `cache/norm_stats.json`.
    """
    mu = np.asarray(stats_dict['mean'], dtype=np.float32)
    sd = np.asarray(stats_dict['std'], dtype=np.float32)
    signed_log = bool(transform_cfg.get('signed_log_volumes'))
    cross = np.asarray(transform_cfg.get('cross_instrument') or [], dtype=np.float32).reshape(-1, 2)
    extra = 0
    if signed_log:
        extra += mu.shape[0]
    extra += cross.shape[0]
    base_mean = np.concatenate([mu, np.zeros(extra, np.float32)])
    base_std = np.concatenate([sd, np.ones(extra, np.float32)])
    return FeatureTransform({
        'base_mean': base_mean,
        'base_std': base_std,
        'n_in': np.int32(mu.shape[0]),
        'clip_sigma': transform_cfg.get('clip_sigma') or 0.0,
        'normalisation': transform_cfg.get('normalisation', 'none'),
        'signed_log_volume': np.int32(1 if signed_log else 0),
        'first_differences': np.asarray(transform_cfg.get('first_differences') or [], dtype=np.int32),
        'cross_instrument': cross,
        'step_feature': 1 if transform_cfg.get('step_feature', False) else 0,
    })


def chunk_loss(pred, target, mask, loss_cfg, state=None, soft_mask=None):
    """1 - Global WP over the whole chunk-batch, plus optional weighted MSE.

    WHY THIS SHAPE (AGENTS.md §12 Phase 2): the metric is ONE weighted Pearson
    per target over all scored rows, so the loss is one weighted Pearson per
    target over the whole batch (B x T rows, typically >10^5) -- not a mean of
    per-sequence or per-chunk correlations.  `state` (a StreamingWP) carries the
    moments across chunks so the statistic is cumulative rather than restarted.
    """
    m = mask if str(loss_cfg.get('mask', 'soft')) == 'soft' else (mask > 0.5).to(mask.dtype)
    # Soft mask: weight each row by the probability that it will be scored. Only
    # ~10% of required rows are scored and they are a DISTINCT regime (measured
    # 0.58 WP on them vs 0.30 on all rows), so the loss should concentrate there.
    # `floor` keeps a non-zero weight on every row so no gradient is ever lost
    # outright. This is a TRAINING-ONLY change: at inference we still predict every
    # required row, so it costs nothing in latency and needs no extra artifacts.
    if soft_mask is not None:
        floor = float(loss_cfg.get('mask_floor', 0.1))
        m = m * (floor + (1.0 - floor) * soft_mask)
    y = target.clamp(-METRIC_CLIP, METRIC_CLIP)
    w = y.abs() * m.unsqueeze(-1)          # m is (B, T): broadcast over the 2 targets
    B, T, K = w.shape
    fw = w.reshape(-1, K)
    fy = y.reshape(-1, K)
    fp = pred.clamp(-METRIC_CLIP, METRIC_CLIP).reshape(-1, K)
    W = fw.sum(0) + 1e-8
    my = (fw * fy).sum(0) / W
    mp = (fw * fp).sum(0) / W
    dy, dp = fy - my, fp - mp
    cov = (fw * dy * dp).sum(0) / W
    vy = (fw * dy * dy).sum(0) / W
    vp = (fw * dp * dp).sum(0) / W
    valid = (W > 1e-8) & (vy > 1e-16) & (vp > 1e-16)
    rho = torch.where(valid, (cov / torch.sqrt(vy.clamp_min(1e-16) * vp.clamp_min(1e-16))).clamp(-1.0, 1.0),
                      torch.zeros_like(cov))
    loss = 1.0 - rho[valid].mean() if valid.any() else rho.sum() * 0.0
    aux = loss_cfg.get('aux_mse_weight', 0.0) or 0.0
    if aux:
        loss = loss + aux * (fw * (fy - fp) ** 2).sum() / fw.sum().clamp_min(1e-8)
    return loss, rho.detach()


@torch.no_grad()
def evaluate(model, transform, store, batch_sequences=8, n_sequences=None, device='cpu',
             max_rows=None, progress=True, state_reset=0):
    """Offline Global WP over the validation shard, streamed, no predictions kept."""
    model.eval()
    acc = metric_ref.StreamingWP()
    stream = ValidationStream(store, batch_sequences, n_sequences)
    it = iter(stream)
    total_seq = len(stream.order)
    seen_rows = 0
    pbar = tqdm(total=total_seq, desc='eval', leave=False) if progress else None
    for X, Y, M, S in it:
        # hidden state per lane, reset at the start of every sequence
        B = X.shape[0]
        h = [torch.zeros(1, B, model.hidden, device=device) for _ in range(model.layers)]
        Xt = torch.from_numpy(np.stack([transform.batch(X[b].numpy()) for b in range(B)]))
        Xt = Xt.to(device)
        preds = np.empty((B, store.seq_len, 2), dtype=np.float32)
        sub = 2000
        for a in range(0, store.seq_len, sub):
            b = min(a + sub, store.seq_len)
            # a reset-trained model MUST be evaluated with the same period it was
            # trained on, otherwise it is scored out of regime
            if state_reset and a and a % state_reset == 0:
                for layer in h:
                    layer.zero_()
            p, h = model.forward_chunk(Xt[:, a:b], h)
            preds[:, a:b] = p.float().cpu().numpy()
        for b_ in range(B):
            acc.add(Y[b_].numpy(), preds[b_], mask=S[b_].numpy())
            seen_rows += int(S[b_].numpy().sum())
        if pbar:
            pbar.update(B)
        if max_rows and seen_rows >= max_rows:
            break
    if pbar:
        pbar.close()
    model.train()
    return acc.result()


# ===== SECTION: TRAIN LOOP =====
class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def state_dict(self):
        return self.shadow


def lr_at(step, total, warmup, base_lr):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + np.cos(np.pi * min(t, 1.0)))


def run_training(cfg, cfg_path, device, smoke, resume, checkpoint_override=None):
    key = f'{MODEL_NAME}_{VERSION}'
    tcfg = cfg['training']
    seed = tcfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_store = ShardStore(cfg['data']['train_shards'])
    valid_store = ShardStore(cfg['data']['valid_shards'])
    if cfg['data'].get('train_sequences'):
        train_store.n_seq = min(train_store.n_seq, cfg['data']['train_sequences'])
    if smoke:
        train_store.n_seq = min(train_store.n_seq, 20)
    n_train_seq = train_store.n_seq
    n_valid_seq = min(len(valid_store), tcfg.get('eval_subset_sequences') or len(valid_store))
    if smoke:
        n_valid_seq = min(n_valid_seq, 10)

    fcfg = cfg['features']
    stats = json.loads((ROOT / 'cache' / 'norm_stats.json').read_text())
    stats_dict = {'mean': np.array([stats['X'][c]['mean'] for c in stats['feature_columns']], np.float32),
                  'std': np.array([stats['X'][c]['std'] for c in stats['feature_columns']], np.float32)}
    if fcfg.get('normalisation', 'none') == 'none':
        stats_dict['mean'] = np.zeros(N_FEATURES, np.float32)
        stats_dict['std'] = np.ones(N_FEATURES, np.float32)
    transform = build_transform_from_stats(stats_dict, fcfg)
    fdim = transform.dim()

    mcfg = cfg['model']
    model = GRUModel(fdim, hidden=mcfg['hidden'], layers=mcfg['layers'],
                     input_proj=mcfg.get('input_proj') or 0, dropout=mcfg.get('dropout', 0.0)).to(device)
    nparams = count_params(model)
    print(f'model: {mcfg["arch"]} hidden={mcfg["hidden"]} layers={mcfg["layers"]} '
          f'input_proj={mcfg.get("input_proj")} in_dim={fdim} params={nparams:,}', flush=True)
    write_yaml_results(cfg_path, cfg, model_updates={'param_count': nparams})

    ocfg = cfg['optim']
    opt = torch.optim.AdamW(model.parameters(), lr=ocfg['lr'], weight_decay=ocfg['weight_decay'],
                            betas=tuple(ocfg.get('betas', [0.9, 0.99])))
    ema = EMA(model, ocfg['ema_decay']) if ocfg.get('ema_decay') else None
    if smoke:
        total_steps = int(tcfg.get('smoke_steps', 120))
    else:
        total_steps = tcfg.get('max_steps')
        if not total_steps:
            epochs = tcfg.get('epochs')
            if not epochs:
                raise SystemExit('set training.max_steps or training.epochs for a real run')
            # One epoch = every training sequence consumed exactly once. A step
            # consumes batch_sequences lanes x chunk_len rows and a sequence is
            # seq_len/chunk_len chunks long, so this is exact rather than an
            # estimate. data.train_sequences: null means ALL sequences, so
            # "epochs" and "the entire training set" cannot drift apart.
            steps_per_epoch = math.ceil(
                n_train_seq * train_store.seq_len
                / (tcfg['batch_sequences'] * tcfg['chunk_len']))
            total_steps = int(steps_per_epoch * epochs)
            print(f'max_steps derived: {steps_per_epoch:,} steps/epoch x {epochs} epoch(s) '
                  f'= {total_steps:,} steps over all {n_train_seq:,} training sequences',
                  flush=True)
    state_reset = int(tcfg.get('state_reset') or 0)
    mask_path = cfg['loss'].get('mask_model')
    soft = SoftMaskStore(mask_path) if mask_path else None
    if soft is not None:
        if soft.n != train_store.n_seq * train_store.seq_len:
            raise SystemExit(f'soft mask has {soft.n:,} rows but the train shard has '
                             f'{train_store.n_seq * train_store.seq_len:,}; rebuild it with '
                             f'tools/train_mask_model.py')
        print(f'soft mask: {soft.path} ({soft.n:,} rows, floor {cfg["loss"].get("mask_floor", 0.1)})',
              flush=True)
    stream = StatefulBatchStream(train_store, tcfg['batch_sequences'], tcfg['chunk_len'],
                                 tcfg['burn_in'], seed=seed, device=device, n_sequences=n_train_seq,
                                 state_reset=state_reset, soft_mask=soft)
    it = prefetch(iter(stream)) if tcfg.get('stateful', True) else iter(stream)
    print(f'train sequences={n_train_seq}  valid sequences={n_valid_seq}  steps={total_steps}', flush=True)

    ck_dir = ROOT / cfg['artifacts']['checkpoint_dir']
    ck_dir.mkdir(parents=True, exist_ok=True)
    history_path = ck_dir / 'history.json'
    history = json.loads(history_path.read_text()) if (resume and history_path.exists()) else []
    best_wp = history[-1]['wp'] if history else -1.0
    best_file = None
    start_step = 0
    last_path = ck_dir / 'last.pt'
    if resume and last_path.exists():
        st = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(st['model'])
        opt.load_state_dict(st['optim'])
        if ema and st.get('ema'):
            ema.shadow = st['ema']
        start_step = st['step']
        torch.set_rng_state(st['rng'].cpu() if hasattr(st['rng'], 'cpu') else st['rng'])
        print(f'resumed from step {start_step} (best so far {best_wp:.6f})', flush=True)

    # transform is applied to whole chunks here so that .batch() is the code path
    # exercised in training (train/serve parity, AGENTS.md §2.2).
    def prepare(Xb):
        return np.stack([transform.batch(Xb[b]) for b in range(Xb.shape[0])])

    t_start = time.time()
    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, ocfg.get('warmup_steps', 0), ocfg['lr'])
        for g in opt.param_groups:
            g['lr'] = lr
        Xb, Yb, Mb, Rb, Sb = next(it)
        Xt = torch.from_numpy(prepare(Xb)).to(device)
        Yt = torch.from_numpy(Yb).to(device)
        Mt = torch.from_numpy(Mb).to(device)
        h = [torch.zeros(1, Xb.shape[0], model.hidden, device=device) for _ in range(model.layers)]
        with torch.no_grad():
            # Rb marks lanes that must start this chunk from a ZERO state: either a
            # new sequence, or a periodic state_reset boundary. Inference does the
            # same at the same positions (see PredictionModel.predict).
            for i, flag in enumerate(Rb):
                if flag:
                    for layer in h:
                        layer[:, i, :].zero_()
        pred, _ = model.forward_chunk(Xt, h)
        loss, _ = chunk_loss(pred, Yt, Mt, cfg['loss'],
                             soft_mask=None if Sb is None else torch.from_numpy(Sb).to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg.get('grad_clip', 1.0))
        opt.step()
        if ema:
            ema.update(model)
        if step % 25 == 0 or step == total_steps - 1:
            print(f'step {step}/{total_steps} lr {lr:.2e} loss {loss.detach().item():.6f} '
                  f'grad_norm {float(gn):.3f} ({time.time()-t_start:.0f}s)', flush=True)
        # ---- periodic evaluation
        if (step + 1) % tcfg['eval_every_steps'] == 0 or step == total_steps - 1:
            bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if ema:
                model.load_state_dict(ema.state_dict())
            res = evaluate(model, transform, valid_store, batch_sequences=4,
                           n_sequences=n_valid_seq, device=device, state_reset=state_reset)
            model.load_state_dict(bak)
            wp = res['weighted_pearson']
            print(f'EVAL step={step} val_wp={wp:.6f} (t0={res["t0"]:.6f} t1={res["t1"]:.6f} '
                  f'rows={res["selected_rows"]}, {"subset" if n_valid_seq < len(valid_store) else "full"})',
                  flush=True)
            if wp > best_wp:
                best_wp = wp
                n_best = sum(1 for h_ in history if h_['wp'] == best_wp)
                name = f'best_{n_best+1:03d}_step{step}_val{wp:.6f}.pt'
                torch.save({'model': model.state_dict(),
                            'ema': ema.state_dict() if ema else None,
                            'config': cfg, 'step': step, 'metrics': res,
                            'feature_stats': stats_dict, 'feature_dim': fdim,
                            'n_params': nparams}, ck_dir / name)
                best_file = name
                print(f'NEW BEST saved {name}', flush=True)
            history.append({'step': step, 'wp': wp, 't0': res['t0'], 't1': res['t1'],
                            'rows': res['selected_rows'], 'wall_s': round(time.time() - t_start, 1),
                            'file': best_file})
            history_path.write_text(json.dumps(history, indent=1))
            torch.save({'model': model.state_dict(),
                        'optim': opt.state_dict(),
                        'ema': ema.state_dict() if ema else None,
                        'rng': torch.get_rng_state(), 'step': step + 1,
                        'config': cfg}, last_path)

    # ---- finish: full-validation rescore of the best checkpoint, export, write back
    if best_file is None:
        best_file = sorted(ck_dir.glob('best_*.pt'))[-1].name if list(ck_dir.glob('best_*.pt')) else None
    full_wp = None
    if best_file:
        st = torch.load(ck_dir / best_file, map_location=device, weights_only=False)
        model.load_state_dict(st['model'])
        model.eval()
        t0 = time.time()
        # A non-smoke run always rescores the COMPLETE validation set: n_sequences
        # below is None, which means "every sequence". Judge the scope by what is
        # actually passed in, NOT by n_valid_seq -- that is the subset used for the
        # periodic EVAL calls, and conflating the two made a genuine full-validation
        # score get discarded and mislabelled "subset over 200 sequences"
        # (measured 2026-09-29 on gru_v001..v004).
        rescore_n = n_valid_seq if smoke else None
        res_full = evaluate(model, transform, valid_store, batch_sequences=4,
                            n_sequences=rescore_n, device=device, state_reset=state_reset)
        is_full = not smoke
        scope = 'FULL validation' if is_full else f'subset of {n_valid_seq} sequences'
        n_seq_scored = len(valid_store) if is_full else n_valid_seq
        print(f'final rescore, {scope} ({res_full["selected_rows"]} scored rows over '
              f'{n_seq_scored} sequences, {time.time()-t0:.0f}s): '
              f't0={res_full["t0"]:.6f} t1={res_full["t1"]:.6f} '
              f'mean={res_full["weighted_pearson"]:.6f}', flush=True)
        full_wp = res_full['weighted_pearson']
        if not is_full:
            full_wp = None
            print('NOTE: this rescore covered only a subset of validation, so '
                  'results.full_valid_wp is left null. It is NOT the official number.', flush=True)
        export_inference(model, transform, ck_dir, opset=cfg['inference']['opset'],
                         device='cpu', verify=True, state_reset=state_reset)
        write_yaml_results(cfg_path, cfg, status='smoke' if smoke else 'finished',
                           results={'best_step': st['step'],
                                    'best_val_wp': st['metrics']['weighted_pearson'],
                                    'best_val_t0': st['metrics']['t0'],
                                    'best_val_t1': st['metrics']['t1'],
                                    'full_valid_wp': full_wp,
                                    'train_hours': round((time.time() - t_start) / 3600, 4),
                                    'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                           clear_results=() if full_wp is not None else ('full_valid_wp',),
                           artifacts={'submission_dir': None})
    return {'best_wp': best_wp, 'full_wp': full_wp, 'params': nparams}


def feature_parity_test(transform, seq_x, tol=1e-6):
    """.step() applied row by row must equal .batch() on the same sequence.

    The `.copy()` is essential, not defensive: `FeatureTransform.step` returns a
    REUSED preallocated buffer, so collecting the results in a list and stacking
    them afterwards would alias one buffer N times and compare the last row N
    times. That silently produced a parity of 11.0 for the robust_zscore config
    on a transform that was in fact exact (measured 2026-09-29). It went unnoticed
    on the identity fast path only because that path returns a fresh array.
    """
    transform.reset()
    steps = np.stack([transform.step(seq_x[i]).copy() for i in range(len(seq_x))])
    batched = transform.batch(seq_x)
    d = float(np.abs(steps - batched).max())
    print(f'feature transform parity: max |step - batch| = {d:.3e} (tol {tol:.0e})', flush=True)
    if d > tol:
        raise SystemExit(f'FEATURE PARITY FAILED: {d:.3e} > {tol:.0e}')
    return d


# ===== SECTION: MAIN =====
def main():
    ap = argparse.ArgumentParser(description=f'train {MODEL_NAME}_{VERSION}')
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke', action='store_true', help='tiny end-to-end run exercising every path')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--export-only', action='store_true')
    ap.add_argument('--checkpoint', default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    ident = cfg['identity']
    if ident['model'] != MODEL_NAME or ident['version'] != VERSION:
        raise SystemExit(f'config identity {ident["model"]}_{ident["version"]} does not match '
                         f'{MODEL_NAME}_{VERSION} in the script')
    if cfg_path.stem != f'version_{MODEL_NAME}_{VERSION}':
        raise SystemExit(f'config filename must be version_{MODEL_NAME}_{VERSION}.yaml')
    log_path = ROOT / cfg['artifacts']['log']
    if not args.resume and not args.export_only and log_path.exists():
        raise SystemExit(f'{log_path} already exists; a version is immutable. '
                         f'Use --resume, or create a new version with tools/new_experiment.py')
    command = 'python ' + ' '.join([str(Path(__file__).relative_to(ROOT))] + sys.argv[1:])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(log_path, resume=args.resume)
    try:
        reproducibility_header(cfg, command)
        if not args.export_only:
            # an --export-only run does not train, so it must not leave the version
            # looking like a live run to tools/summarize.py
            write_yaml_results(cfg_path, cfg, status='running')
        device = pick_device(args.device or cfg['training'].get('device', 'auto'))
        print(f'device: {device}', flush=True)
        if args.export_only:
            ck = Path(args.checkpoint) if args.checkpoint else \
                sorted((ROOT / cfg['artifacts']['checkpoint_dir']).glob('best_*.pt'))[-1]
            st = torch.load(ck, map_location='cpu', weights_only=False)
            stats_dict = st['feature_stats']
            transform = build_transform_from_stats(stats_dict, cfg['features'])
            model = GRUModel(st['feature_dim'], hidden=cfg['model']['hidden'],
                             layers=cfg['model']['layers'],
                             input_proj=cfg['model'].get('input_proj') or 0)
            model.load_state_dict(st['model'])
            export_inference(model, transform, ck.parent, opset=cfg['inference']['opset'],
                             state_reset=int(st.get('state_reset', 0)))
            print(f'exported inference artifacts from {ck.name} into {ck.parent}', flush=True)
            return 0
        if args.smoke:
            fcfg = cfg['features']
            stats = json.loads((ROOT / 'cache' / 'norm_stats.json').read_text())
            sd = {'mean': np.array([stats['X'][c]['mean'] for c in stats['feature_columns']], np.float32),
                  'std': np.array([stats['X'][c]['std'] for c in stats['feature_columns']], np.float32)}
            if fcfg.get('normalisation', 'none') == 'none':
                sd = {'mean': np.zeros(N_FEATURES, np.float32), 'std': np.ones(N_FEATURES, np.float32)}
            tr = build_transform_from_stats(sd, fcfg)
            store = ShardStore(cfg['data']['valid_shards'])
            x, _, _, _ = store.sequence(0)
            feature_parity_test(tr, x[:2000])
        out = run_training(cfg, cfg_path, device, args.smoke, args.resume)
        print('done:', out, flush=True)
        return 0
    except SystemExit as e:
        write_yaml_results(cfg_path, cfg, status='failed')
        raise
    except Exception:
        write_yaml_results(cfg_path, cfg, status='failed')
        raise
    finally:
        tee.close()


if __name__ == '__main__':
    raise SystemExit(main())
