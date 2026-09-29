# Derived from train_gru_v005.py by tools/package_submission.py.
# Only the code between the BEGIN/END SOLUTION SECTION markers is kept, unchanged,
# so the inference code is byte-identical to the code used in training.
# source sha256: 57820074215820654fee0d8455dccccfc0ef60ff4e48f13df772c434561492f8
# packaged (UTC): 2026-09-29T16:41:36Z
# Inference dependencies: numpy + onnxruntime + stdlib only.
# Everything between these markers is copied verbatim into the packaged
# solution.py.  It may import ONLY numpy, onnxruntime and the standard library:
# the scoring container is built from requirements.txt, which does not contain
# torch, yaml, pyarrow, pandas, scipy, matplotlib or tqdm.
import os as _os

import numpy as _np
import onnxruntime as _ort

MODEL_NAME = 'gru'
VERSION = 'v005'
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
        self._prev = _np.zeros(self.n_in, dtype=_np.float32)
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

    def _diffs_from(self, Z, prev, have_prev):
        """Z: (n, n_base) already normalised; prev: (n_in,) or None."""
        D = _np.zeros((Z.shape[0], self.n_diff), dtype=_np.float32)
        if self.n_diff == 0:
            return D
        if not have_prev:
            return D                      # first row of a sequence has no predecessor
        D[0] = Z[0, self._diff_idx] - prev[self._diff_idx]
        if Z.shape[0] > 1:
            _np.subtract(Z[1:][:, self._diff_idx], Z[:-1][:, self._diff_idx], out=D[1:])
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
        D = self._diffs_from(Z, None, False)
        self._step_i = len(X)
        self._have_prev = bool(len(X))
        if self._have_prev:
            self._prev[:] = X[-1]
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
        self._prev[:] = x
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
