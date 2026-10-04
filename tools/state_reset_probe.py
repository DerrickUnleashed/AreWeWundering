#!/usr/bin/env python3
"""How much is the training/serving `state_reset` skew worth?

THE BUG THIS MEASURES
---------------------
Every version from v021 onward has `training.state_reset: 2000` and
`inference.state_reset: 4000`. Those are not the same knob:

  * the training loop reads `cfg['training']['state_reset']` and clears the recurrent
    state every 2000 steps, so the model is TRAINED to expect a reset twice as often as
  * the exporter writes `feature_stats.npz` from `cfg['inference']['state_reset']`, and
    `solution.py` reads that back at serving time and clears every 4000.

So every model we have submitted was trained under one reset schedule and served under
another. It survived because the promotion gate's parity test passes by construction: it
compares the STREAMING path against the OFFLINE path, and both read the same
feature_stats.npz, so both use 4000 and both agree. Parity cannot see this. Only comparing
serving against TRAINING can.

This is exactly the class of bug AGENTS.md sec 2 prime directive 2 exists to prevent
(train/serve parity), and the gate built for it has a blind spot precisely here.

THE MEASUREMENT
---------------
Streams the same validation rows twice through the same exported weights, changing only the
serving state_reset, and scores each with the reference metric. Identical rows, identical
weights, identical mask -- so any difference is attributable to the reset schedule alone.

If 2000 (the value training actually used) scores materially higher, the fix is one line in
the yaml and zero latency: the state-clear is an `if` on the step index either way, so
halving the interval costs nothing per row.
"""

from __future__ import annotations

import argparse
import collections
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from metric_ref import global_wp  # noqa: E402

SEQ, WARMUP = 20_000, 99
Row = collections.namedtuple("Row", "seq_ix step_in_seq state need_prediction")


def load_bundle(src: Path, dst: Path, state_reset: int) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in ("solution.py", "model.onnx"):
        shutil.copy(src / f, dst / f)
    st = dict(np.load(src / "feature_stats.npz", allow_pickle=True))
    st["state_reset"] = np.int32(state_reset)
    np.savez(dst / "feature_stats.npz", **st)
    return dst


def score(bundle: Path, X, Y, N, S, n_seq: int) -> tuple[float, int]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("sol", bundle / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.PredictionModel()
    rows = n_seq * SEQ
    P = np.empty((rows, 2), np.float32)
    for s in range(n_seq):
        lo, hi = s * SEQ, (s + 1) * SEQ
        raw = np.asarray(X[lo:hi], np.float32)
        for i in range(hi - lo):
            P[lo + i] = m.predict(Row(s, i, raw[i], bool(N[lo + i])))
    y = np.asarray(Y[:rows], np.float32)
    need = np.asarray(N[:rows], bool)
    sc = np.asarray(S[:rows], bool)
    return global_wp(y, P, need, sc), rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v044")
    ap.add_argument("--n-sequences", type=int, default=300)
    ap.add_argument("--resets", default="2000,4000")
    a = ap.parse_args()

    src = ROOT / "submissions" / f"submission_gru_{a.version}"
    X = np.load(ROOT / "cache/valid/X.npy", mmap_mode="r")
    Y = np.load(ROOT / "cache/valid/y.npy", mmap_mode="r")
    N = np.load(ROOT / "cache/valid/need_prediction.npy", mmap_mode="r")
    S = np.load(ROOT / "cache/valid/is_scored.npy", mmap_mode="r")
    resets = [int(x) for x in a.resets.split(",")]
    n_seq = min(a.n_sequences, X.shape[0] // SEQ)

    print(f"  {a.version}: {n_seq} validation sequences, {n_seq*SEQ:,} rows")
    print(f"  identical weights, identical rows; ONLY the serving state_reset varies\n")
    print(f"  {'serving state_reset':>20}{'global WP':>14}{'delta vs 4000':>16}{'seconds':>9}")
    out = {}
    for sr in resets:
        with tempfile.TemporaryDirectory() as td:
            b = load_bundle(src, Path(td) / "b", sr)
            t0 = time.time()
            wp, rows = score(b, X, Y, N, S, n_seq)
            el = time.time() - t0
        out[sr] = wp
        ref = out.get(4000, wp)
        d = wp - ref
        flag = "  <- TRAINED with this" if sr == 2000 else ""
        print(f"  {sr:>20}{wp:14.6f}{d:+16.6f}{el:9.1f}{flag}")
    if len(resets) > 1:
        best = max(out, key=out.get)
        print(f"\n  best serving state_reset = {best} at {out[best]:.6f}")
        print(f"  training used 2000. If 2000 wins, every model we have ever submitted was")
        print(f"  served at the wrong reset interval and the fix is a yaml value, not a")
        print(f"  retrain. Per-row cost is unchanged either way: the clear is an `if` on the")
        print(f"  step index, so this is free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())