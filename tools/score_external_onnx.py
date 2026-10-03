#!/usr/bin/env python3
"""Score ANY set of ONNX models on validation, including FOREIGN graphs, on identical rows.

WHY THIS TOOL EXISTS
--------------------
`tools/score_ensemble.py` can only score OUR students: it hardcodes a rank-3 precomputed
`features` input, `hidden_0`/`hidden_1` state names, and reads `state_reset` out of
`feature_stats.npz`. dozken's graphs have a different contract entirely -- raw rank-2
`features`, `state_0`/`state_1`, an explicit `prev` input because they compute first
differences INSIDE the graph, and `state_reset` in an ONNX metadata prop. Feeding one to
the other tool fails at session creation.

So the question "how good is his teacher on OUR metric, on OUR rows?" had no tool that
could answer it. That question decides the entire v04x batch: if the teacher scores near
0.688 there is headroom above the student and buying better teachers is the play; if it
scores near 0.66 the student has already extracted everything distillation can give and
the direction is exhausted.

ONE PASS, MANY MODELS, IDENTICAL ROWS
-------------------------------------
The comparison that matters is a DIFFERENCE between two models, so both must see exactly
the same rows under exactly the same mask. This streams the sequences once and evaluates
every model plus every requested mean on each row, so no model can be advantaged by a
different row selection. It also scores the mean of a set, which is the only way to
compare a foreign pair against a single local model without a second pass.

Subsets vs full validation are DIFFERENT SCALES and must never be mixed: our own runs read
~0.012 higher on the 200-sequence eval subset than on full validation (v002 subset 0.6859
vs full 0.6742). Because every model here is scored on the same rows, a subset run answers
the comparison question exactly; it just does not answer "what is the official number",
and the log says so explicitly.

Usage
-----
    python tools/score_external_onnx.py \\
        --graph submissions/submission_dozken_ens_6879/reset2k.onnx \\
        --graph submissions/submission_dozken_ens_6879/reset4k.onnx \\
        --student v039 \\
        --mean-of dozken:reset2k,reset4k \\
        --n-sequences 500
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from metric_ref import global_wp, weighted_pearson  # noqa: E402

SEQ = 20_000


class Graph:
    """One ONNX model, with its input contract detected rather than assumed."""

    def __init__(self, name: str, path: Path, transform=None):
        self.name = name
        self.path = path
        m = onnx.load(str(path))
        self.raw_mode = any(i.name == "prev" for i in m.graph.input)
        self.state_names = [i.name for i in m.graph.input
                            if "state" in i.name or "hidden" in i.name]
        self.state_reset = int(dict((p.key, p.value)
                                    for p in m.metadata_props).get("state_reset", 0) or 0)
        fv = next(i for i in m.graph.input if i.name == "features")
        self.feat_rank = len(fv.type.tensor_type.shape.dim)
        self.feat_dim = fv.type.tensor_type.shape.dim[-1].dim_value
        H = next(i.type.tensor_type.shape.dim[-1].dim_value
                 for i in m.graph.input if i.name in self.state_names)
        self.H = H
        self.transform = transform
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(path), sess_options=o,
                                         providers=["CPUExecutionProvider"])
        outs = [x.name for x in self.sess.get_outputs()]
        self.o_pred = outs.index("prediction")
        self.o_states = [outs.index(n) for n in outs
                         if n.startswith(("next_state", "next_hidden"))]
        self.o_prev = outs.index("next_prev") if "next_prev" in outs else None

    def reset(self):
        self.st = [np.zeros((1, 1, self.H), np.float32) for _ in self.state_names]
        self.prev = np.zeros((1, self.feat_dim), np.float32)

    def step(self, raw_row: np.ndarray, xrow: np.ndarray, row_ix: int):
        """One row. State is cleared every `state_reset` rows, exactly as at serving."""
        if self.state_reset and row_ix and row_ix % self.state_reset == 0:
            self.st = [np.zeros((1, 1, self.H), np.float32) for _ in self.state_names]
        feed = {}
        if self.raw_mode:
            feed["features"] = raw_row
            feed["prev"] = self.prev
        else:
            # our students take a precomputed rank-3 matrix, already transformed
            feed["features"] = xrow
        for n, v in zip(self.state_names, self.st):
            feed[n] = v
        out = self.sess.run(None, feed)
        self.st = [out[k] for k in self.o_states]
        if self.o_prev is not None:
            self.prev = out[self.o_prev]
        return out[self.o_pred]


def load_transform(version: str):
    p = ROOT / "submissions" / f"submission_gru_{version}" / "solution.py"
    spec = importlib.util.spec_from_file_location(f"sol_{version}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stats = np.load(ROOT / "submissions" / f"submission_gru_{version}" /
                    "feature_stats.npz")
    return mod.FeatureTransform(stats)


def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              cwd=str(ROOT)).stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", action="append", default=[], metavar="NAME=PATH",
                    help="foreign or local graph; NAME is how it is labelled")
    ap.add_argument("--student", action="append", default=[],
                    help="our student version, e.g. v039 (uses its FeatureTransform)")
    ap.add_argument("--mean-of", default="",
                    help="comma-separated NAMES to also score as a mean, "
                         "e.g. dozken_pair:reset2k,reset4k")
    ap.add_argument("--n-sequences", type=int, default=500)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()

    log = Path(a.log) if a.log else ROOT / "exploration" / "logs" / "log_external_score.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    _real = sys.stdout
    fh = open(log, "w")

    class T:
        def write(self, t):
            _real.write(t)
            fh.write(t)

        def flush(self):
            _real.flush()
            fh.flush()

    sys.stdout = T()

    X = np.load(ROOT / "cache/valid/X.npy", mmap_mode="r")
    Y = np.load(ROOT / "cache/valid/y.npy", mmap_mode="r")
    N = np.load(ROOT / "cache/valid/need_prediction.npy", mmap_mode="r")
    S = np.load(ROOT / "cache/valid/is_scored.npy", mmap_mode="r")
    n_all = X.shape[0] // SEQ
    n_seq = min(a.n_sequences, n_all)

    print(f"### external-model scoring, log opened "
          f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}")
    print("=" * 78)
    print(f"command     : {' '.join(sys.argv)}")
    print(f"host        : {platform.platform()}  python {platform.python_version()}")
    print(f"git commit  : {git('rev-parse', 'HEAD')}")
    print(f"numpy / ort : {np.__version__} / {ort.__version__}")
    print(f"sequences   : {n_seq} of {n_all}"
          f"{'   (OFFICIAL full-validation number)' if n_seq == n_all else '   (SUBSET -- not the official scale)'}")
    print("NOTE        : every model below sees IDENTICAL rows and mask, so the")
    print("              DIFFERENCES between them are exact regardless of subset size.")
    print("              Absolute values are NOT comparable to full_valid_wp.")

    graphs: dict[str, Graph] = {}
    for spec in a.graph:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(name).stem, name
        graphs[name] = Graph(name, ROOT / path)
    for v in a.student:
        graphs[v] = Graph(v, ROOT / "submissions" / f"submission_gru_{v}" / "model.onnx",
                          transform=load_transform(v))

    print()
    for n, g in graphs.items():
        print(f"  {n:16} raw_mode={str(g.raw_mode):5} feat_rank={g.feat_rank} "
              f"dim={g.feat_dim} H={g.H} state_reset={g.state_reset} "
              f"states={g.state_names}")

    means: dict[str, list[str]] = {}
    for spec in a.mean_of.split(","):
        if not spec.strip():
            continue
        nm, _, members = spec.partition(":")
        means[nm.strip()] = [m.strip() for m in members.split(",") if m.strip()]

    rows = n_seq * SEQ
    P = {n: np.empty((rows, 2), np.float32) for n in graphs}
    for nm in means:
        P[nm] = np.empty((rows, 2), np.float32)

    t0 = time.time()
    for s in range(n_seq):
        lo, hi = s * SEQ, (s + 1) * SEQ
        raw = np.asarray(X[lo:hi], np.float32)
        xb = None
        if any(not g.raw_mode for g in graphs.values()):
            tr = next(g.transform for g in graphs.values() if not g.raw_mode)
            xb = np.asarray(tr.batch(raw), np.float32)[:, None, :]
        for g in graphs.values():
            g.reset()
            for i in range(SEQ):
                rr = raw[i][None] if g.raw_mode else None
                xr = xb[i][None] if not g.raw_mode else None
                o = g.step(rr if rr is not None else raw[i:i + 1],
                           xr if xr is not None else raw[i:i + 1], i)
                P[g.name][lo + i] = np.asarray(o).reshape(-1)[:2]
        for nm, mem in means.items():
            P[nm][lo:hi] = np.mean([P[m][lo:hi] for m in mem], axis=0)
        if (s + 1) % 50 == 0:
            el = time.time() - t0
            print(f"    seq {s+1}/{n_seq}  {el:7.1f}s  eta {el/(s+1)*(n_seq-s-1):7.1f}s",
                  flush=True)

    y = np.asarray(Y[:rows], np.float32)
    need = np.asarray(N[:rows], np.float32).astype(bool)
    sc = np.asarray(S[:rows], bool)
    print(f"\n  scored rows: {int((need & sc).sum()):,} of {rows:,}")
    print()
    res = {}
    order = list(graphs) + list(means)
    for nm in order:
        mean = global_wp(y, P[nm], need, sc)
        sel = need & sc
        per = [weighted_pearson(y[sel, k], P[nm][sel, k]) for k in range(2)]
        res[nm] = mean
        print(f"  {nm:22} t0 {per[0]:.6f}  t1 {per[1]:.6f}  MEAN {mean:.6f}")

    print()
    print("  PAIRWISE (same rows, so these differences are the real signal)")
    solo = [n for n in graphs]
    for i in range(len(solo)):
        for j in range(i + 1, len(solo)):
            d = res[solo[i]] - res[solo[j]]
            print(f"    {solo[i]:20} - {solo[j]:20} = {d:+.6f}")
    for nm, mem in means.items():
        for m in mem:
            if m in res:
                print(f"    {nm:20} - {m:20} = {res[nm] - res[m]:+.6f}")

    np.save(ROOT / "cache" / "external_scores.npy",
            np.array([[res[n]] for n in order], np.float64), allow_pickle=True)
    try:
        shown = log.relative_to(ROOT)
    except ValueError:
        shown = log          # log may live outside the repo (e.g. /tmp for a smoke test)
    print(f"\n  elapsed {time.time()-t0:.1f}s   log: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
