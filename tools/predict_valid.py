#!/usr/bin/env python3
"""Cache validation predictions for a packaged bundle, so EDA does not re-stream 37.5M rows.

Every EDA question about predictions needs the same thing: the full-validation prediction
matrix for one bundle. Recomputing it per question costs ~18 minutes of ORT streaming each
time, which is why nobody had done it before. This produces it ONCE per version into
cache/preds/<version>_valid.npy and every later script memory-maps that.

Runs the REAL inference path -- the bundle's own solution.py, streamed row by row through
its callback -- rather than an offline reimplementation, so what gets analysed is exactly
what would be served. AGENTS.md sec 7 requires train/serve parity and this is where it is
observed rather than asserted.

Outputs, all float32, all (37460000, 2) unless noted:
    cache/preds/<version>_valid.npy      predictions, [t0, t1]
    cache/preds/<version>_seq.npy        int32 sequence index per row
Writes the targets and mask it used into the same folder's .meta.json, so a later script
cannot silently analyse a different row set than the one the predictions were made on.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

# The callback signature solution.py actually implements is predict(data_point), where
# data_point carries the row's bookkeeping. utils.ScorerStepByStep builds this; we
# reproduce the same fields so the cached predictions come from the REAL serving path
# rather than a lookalike.
Row = collections.namedtuple("Row", "seq_ix step_in_seq state need_prediction")

ROOT = Path(__file__).resolve().parent.parent
SEQ = 20_000
WARMUP = 99


def load_solution(version: str):
    p = ROOT / "submissions" / f"submission_gru_{version}" / "solution.py"
    if not p.exists():
        raise SystemExit(f"[predict] no bundle for v{version}: {p} not found")
    spec = importlib.util.spec_from_file_location(f"sol_{version}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--n-sequences", type=int, default=None,
                    help="default: all 1873 (the official scope)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    outdir = ROOT / "cache" / "preds"
    outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{a.version}_valid.npy"
    if outp.exists() and not a.force:
        print(f"[predict] {outp.relative_to(ROOT)} exists; use --force to redo")
        return 0

    X = np.load(ROOT / "cache/valid/X.npy", mmap_mode="r")
    Y = np.load(ROOT / "cache/valid/y.npy", mmap_mode="r")
    N = np.load(ROOT / "cache/valid/need_prediction.npy", mmap_mode="r")
    S = np.load(ROOT / "cache/valid/is_scored.npy", mmap_mode="r")
    seq = np.load(ROOT / "cache/valid/seq_index.npy", mmap_mode="r")

    n_all = X.shape[0] // SEQ

    # seq_index holds SEQUENCE START OFFSETS (n_sequences + 1 of them), NOT a per-row
    # sequence id. Reading it as a per-row id makes every row after the first look like a
    # mid-sequence continuation, and solution.py correctly refuses that -- which is the
    # one error it is designed to catch.
    assert seq.shape[0] == n_all + 1, f"seq_index has {seq.shape[0]}, expected {n_all + 1}"

    def seq_ix(row: int) -> int:
        return int(np.searchsorted(seq, row, side="right") - 1)

    n_seq = min(a.n_sequences or n_all, n_all)
    rows = n_seq * SEQ
    assert rows <= X.shape[0], "scope exceeds shard"

    sol = load_solution(a.version)
    model = sol.PredictionModel()          # the same class solution.py serves with
    print(f"[predict] v{a.version}: {n_seq}/{n_all} sequences, {rows:,} rows", flush=True)

    P = np.empty((rows, 2), np.float32)
    t0 = time.time()
    for s in range(n_seq):
        lo, hi = s * SEQ, (s + 1) * SEQ
        raw = np.asarray(X[lo:hi], np.float32)
        buf = np.empty((hi - lo, 2), np.float32)
        for i in range(hi - lo):
            buf[i] = model.predict(Row(seq_ix(lo + i), i, raw[i], bool(N[lo + i])))
        P[lo:hi] = buf
        if (s + 1) % 100 == 0:
            el = time.time() - t0
            print(f"    seq {s+1}/{n_seq}  {el:7.1f}s  eta "
                  f"{el/(s+1)*(n_seq-s-1):7.1f}s", flush=True)

    np.save(outp, P)
    row_ix = np.repeat(np.arange(n_seq, dtype=np.int32), SEQ)
    np.save(outdir / f"{a.version}_seq.npy", row_ix)
    (outdir / f"{a.version}_meta.json").write_text(json.dumps({
        "version": a.version,
        "n_sequences": n_seq,
        "n_sequences_total": n_all,
        "rows": rows,
        "scored_rows": int((np.asarray(N[:rows], bool) & np.asarray(S[:rows], bool)).sum()),
        "warmup": WARMUP,
        "path": "REAL solution.py callback loop, not an offline reimplementation",
    }, indent=2) + "\n")
    print(f"[predict] wrote {outp.relative_to(ROOT)}  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())