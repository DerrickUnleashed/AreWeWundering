#!/usr/bin/env python3
"""Prove the fused dozken ensemble equals the 2-session mean, bit for bit, then time it.

A fused graph is only worth anything if it computes the SAME function. Averaging is
associative in exact arithmetic and is NOT in float32, so "obviously equivalent" is not
a safe assumption -- it has to be measured, over rows that exercise the state carry and
the in-graph first-differencing, because that is where a wiring mistake would hide.

Checks, in order:
  1. predictions match the 2-session mean EXACTLY (max abs diff == 0.0)
  2. next_prev matches, which is what lets one shared `prev` wire replace two
  3. both hidden states evolve identically to the separate sessions'
  4. latency, 1 thread, against the separate-session cost it is meant to beat
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent


def make_sess(path: Path, threads: int = 1):
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = threads
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--feat", type=int, default=112)
    ap.add_argument("--seq", type=int, default=3000)
    ap.add_argument("--rows", type=int, default=120)
    ap.add_argument("--bench-rows", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=0.0)
    a = ap.parse_args()

    H, F = a.hidden, a.feat
    sa, sb, sf = make_sess(ROOT / a.a), make_sess(ROOT / a.b), make_sess(ROOT / a.fused)

    X = np.load(ROOT / "cache/valid/X.npy", mmap_mode="r")
    rows = np.asarray(X[a.seq:a.seq + a.rows], np.float32)

    # Separate sessions: one state pair + one prev each.
    st = [[np.zeros((1, 1, H), np.float32) for _ in range(2)] for _ in (sa, sb)]
    ps = [np.zeros((1, F), np.float32) for _ in (sa, sb)]
    # Fused: one prev (shared), two state pairs.
    fs = [np.zeros((1, 1, H), np.float32) for _ in range(4)]
    pf = np.zeros((1, F), np.float32)

    # Discover the fused graph's state input names rather than assuming the prefix.
    fused_states = [i.name for i in sf.get_inputs() if "state" in i.name]
    assert len(fused_states) == 4, f"expected 4 state inputs, got {fused_states}"
    # Index outputs BY NAME. The fused graph's output order is an artefact of how the
    # branches were appended, so positional indexing silently reads a hidden state as
    # the prediction -- which fails loudly here but would not in a looser harness.
    fo_names = [o.name for o in sf.get_outputs()]
    IDX = {n: i for i, n in enumerate(fo_names)}
    P_PRED, P_PREV = IDX["prediction"], IDX["next_prev"]
    P_ST = [IDX[n] for n in fused_states and
            [f"b{i}_next_state_{j}" for i in (0, 1) for j in (0, 1)]]

    dpred = dprev = dstate = 0.0
    for i in range(rows.shape[0]):
        r = rows[i:i + 1]
        outs = []
        for k, s in enumerate((sa, sb)):
            o = s.run(None, {"features": r, "state_0": st[k][0], "state_1": st[k][1],
                             "prev": ps[k]})
            st[k] = [o[1], o[2]]
            ps[k] = o[3]
            outs.append(o[0])
        sep_mean = (outs[0] + outs[1]) / 2.0

        feed = {"features": r, "prev": pf,
                fused_states[0]: fs[0], fused_states[1]: fs[1],
                fused_states[2]: fs[2], fused_states[3]: fs[3]}
        fo = sf.run(None, feed)
        fs = [fo[k] for k in P_ST]
        pf = fo[P_PREV]

        dpred = max(dpred, float(np.abs(fo[P_PRED] - sep_mean).max()))
        dprev = max(dprev, float(np.abs(pf - ps[0]).max()))
        dstate = max(dstate, float(np.abs(fs[0] - st[0][0]).max()),
                     float(np.abs(fs[2] - st[1][0]).max()))

    print(f"  rows compared                       : {rows.shape[0]}")
    print(f"  max |fused pred - 2-session mean|   : {dpred:.3e}")
    print(f"  max |fused next_prev - sep next_prev|: {dprev:.3e}")
    print(f"  max |fused state_0 - sep state_0|   : {dstate:.3e}")
    ok = max(dpred, dprev, dstate) <= a.tol
    print(f"  BIT-EXACT (tol {a.tol:g})              : {ok}")
    if not ok:
        print("  FAIL -- the fused graph is NOT the same function; do not ship it.")

    # ---- latency, 1 thread, real rows ----
    bench = np.asarray(X[a.seq:a.seq + a.bench_rows], np.float32)
    n = bench.shape[0]

    def timeit(kind):
        if kind == "sep":
            s = [[np.zeros((1, 1, H), np.float32) for _ in range(2)] for _ in (sa, sb)]
            p = [np.zeros((1, F), np.float32) for _ in (sa, sb)]
            t0 = time.perf_counter()
            for i in range(n):
                r = bench[i:i + 1]
                for k, ss in enumerate((sa, sb)):
                    o = ss.run(None, {"features": r, "state_0": s[k][0],
                                      "state_1": s[k][1], "prev": p[k]})
                    s[k] = [o[1], o[2]]
                    p[k] = o[3]
            return (time.perf_counter() - t0) / n * 1e6
        s = [np.zeros((1, 1, H), np.float32) for _ in range(4)]
        p = np.zeros((1, F), np.float32)
        t0 = time.perf_counter()
        for i in range(n):
            r = bench[i:i + 1]
            o = sf.run(None, {"features": r, "prev": p, fused_states[0]: s[0],
                              fused_states[1]: s[1], fused_states[2]: s[2],
                              fused_states[3]: s[3]})
            s = [o[k] for k in P_ST]
            p = o[P_PREV]
        return (time.perf_counter() - t0) / n * 1e6

    # warm both, then take the best of three to blunt scheduler noise
    timeit("sep")
    timeit("fused")
    sep_us = min(timeit("sep") for _ in range(3))
    fus_us = min(timeit("fused") for _ in range(3))
    ROWS, MIN = 39.4e6, 60 * 60
    print()
    print(f"  2 separate sessions : {sep_us:7.2f} us/row   -> "
          f"{sep_us * 1.62 * ROWS / 1e6 / 60:5.1f} min on grader  "
          f"({sep_us * 1.62 * ROWS / 1e6 / 60 / MIN * 100:.0f}% of budget)")
    print(f"  1 fused graph       : {fus_us:7.2f} us/row   -> "
          f"{fus_us * 1.62 * ROWS / 1e6 / 60:5.1f} min on grader  "
          f"({fus_us * 1.62 * ROWS / 1e6 / 60 / MIN * 100:.0f}% of budget)")
    print(f"  saved              : {sep_us - fus_us:7.2f} us/row "
          f"({(sep_us / fus_us - 1) * 100:.0f}% faster)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())