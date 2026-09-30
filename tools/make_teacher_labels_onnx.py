"""Generate distillation teacher labels from dozken's reset2k ONNX, exactly.

WHY AN ONNX REPLAY AND NOT A TORCH REBUILD
-------------------------------------------
`tools/dozken_model.py` reconstructs the architecture in torch so we can inspect
and retrain it, but its GRU gate wiring does not yet reproduce the ONNX graph
exactly: the input path (normalisation, first differences, input projection,
LayerNorm, GELU) matches to 1.5e-06, while the recurrent state diverges by ~0.06
under the best of six gate-order / bias-layout / linear_before_reset
combinations. A near-match is not good enough to distil from.

Since the graph is the ground truth, labels come from the graph itself. That is
exact by construction and needs no reverse-engineering.

THE BATCHING PROBLEM
--------------------
The graph declares its inputs as `[1, 112]` and `[1, 1, 192]`, and onnxruntime
rejects any other batch size -- so it cannot be batched along the time axis
inside one session. Measured single-row cost is ~43 us, which is 2.5 hours for
the 212,140,000-row train shard on one core.

The fix is to batch along SEQUENCES instead: each worker owns whole sequences and
runs them strictly one row at a time, which is exactly how the served model runs.
With one process per core the shard is replayed in ~20 minutes instead of 2.5
hours, with no change to the numerics.

CAUSALITY
---------
Strictly causal and identical to serving: one sequence per worker, hidden state
carried row to row, `prev` carrying the previous normalised row, state zeroed at
each sequence start and at every multiple of `state_reset` (2000, from the
graph's own metadata). Nothing is averaged across sequences and nothing from the
future is used, so the labels are deployable knowledge, not hindsight.

The `prev` state is what makes this easy to get right: the graph exposes
`next_prev` precisely so the caller carries the normalised row forward. Getting
that wrong would silently corrupt every first difference after step 0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
SEQ_LEN = 20_000
N_FEATURES = 112


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ===== SECTION: WORKER =====


def _worker(job):
    """Replay a contiguous block of sequences into a slice of the output file."""
    (onnx_path, shard, out_path, lo, hi, state_reset, threads) = job
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=['CPUExecutionProvider'])

    meta = sess.get_modelmeta().custom_metadata_map or {}
    reset = int(meta.get('state_reset', state_reset) or 0)

    X = np.load(Path(shard) / 'X.npy', mmap_mode='r')
    idx = np.load(Path(shard) / 'seq_index.npy')
    out = np.load(out_path, mmap_mode='r+')

    n_rows_done = 0
    for i in range(lo, hi):
        a, b = int(idx[i]), int(idx[i + 1])
        buf = np.empty((b - a, 2), dtype=np.float16)
        h0 = np.zeros((1, 1, 192), np.float32)
        h1 = np.zeros((1, 1, 192), np.float32)
        prev = np.zeros((1, N_FEATURES), np.float32)
        for r in range(b - a):
            if reset and r and r % reset == 0:
                h0 = np.zeros((1, 1, 192), np.float32)
                h1 = np.zeros((1, 1, 192), np.float32)
            row = np.asarray(X[a + r], dtype=np.float32).reshape(1, N_FEATURES)
            pred, h0, h1, prev = sess.run(
                None, {'features': row, 'state_0': h0, 'state_1': h1, 'prev': prev})
            buf[r] = pred[0]
        out[a:b] = buf
        n_rows_done += b - a
    out.flush()
    return lo, hi, n_rows_done


# ===== SECTION: MAIN =====


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--onnx', default='submissions/submission_dozken_6569/reset2k.onnx')
    ap.add_argument('--shard', default='cache/train')
    ap.add_argument('--out', default='cache/teacher')
    ap.add_argument('--tag', default='dozken_reset2k')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--threads', type=int, default=1,
                    help='ORT intra-op threads per worker (1 is fastest at batch 1)')
    ap.add_argument('--limit', type=int, default=None,
                    help='replay only the first N sequences (for a smoke test)')
    args = ap.parse_args()

    onnx_path = str((ROOT / args.onnx).resolve())
    shard = ROOT / args.shard
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f'teacher_{args.tag}.npy'

    X = np.load(shard / 'X.npy', mmap_mode='r')
    idx = np.load(shard / 'seq_index.npy')
    n_seq_total = len(idx) - 1
    n_seq = min(args.limit, n_seq_total) if args.limit else n_seq_total
    total_rows = int(idx[n_seq])

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    meta = sess.get_modelmeta().custom_metadata_map or {}
    state_reset = int(meta.get('state_reset', 0) or 0)
    import onnx
    n_params = sum(int(np.prod(t.dims)) for t in onnx.load(onnx_path).graph.initializer
                   if t.dims)
    print(f'teacher: {args.tag}', flush=True)
    print(f'  graph     : {args.onnx}  ({n_params:,} params, state_reset={state_reset})',
          flush=True)
    print(f'  shard     : {args.shard}  {n_seq:,} sequences, {total_rows:,} rows', flush=True)
    print(f'  workers   : {args.workers} x {args.threads} ORT thread(s)', flush=True)

    if out_npy.exists():
        print(f'  {out_npy.name} exists, re-created fresh', flush=True)
    out = np.lib.format.open_memmap(out_npy, mode='w+', dtype=np.float16,
                                    shape=(total_rows, 2))
    del out

    # Contiguous blocks per worker: each sequence stays whole inside one worker,
    # so no two processes ever touch the same rows and no locking is needed.
    n_workers = max(1, min(args.workers, n_seq))
    edges = np.linspace(0, n_seq, n_workers + 1).astype(int)
    jobs = [(onnx_path, str(shard), str(out_npy), int(edges[i]), int(edges[i + 1]),
             state_reset, args.threads) for i in range(n_workers) if edges[i] < edges[i + 1]]

    t0 = time.time()
    done_rows = 0
    with mp.get_context('spawn').Pool(len(jobs)) as pool:
        for lo, hi, nrows in pool.imap_unordered(_worker, jobs):
            done_rows += nrows
            el = time.time() - t0
            rate = done_rows / el
            print(f'  seqs {lo}-{hi} done  {done_rows:,}/{total_rows:,} rows  '
                  f'{el / 60:.1f} min  {rate:,.0f} rows/s  '
                  f'ETA {(total_rows - done_rows) / rate / 60:.0f} min', flush=True)
    secs = time.time() - t0

    manifest = {
        'tag': args.tag,
        'source': 'dozken reset2k ONNX, replayed through onnxruntime',
        'onnx': args.onnx,
        'onnx_sha256': sha256(Path(onnx_path)),
        'params': n_params,
        'state_reset': state_reset,
        'shard': args.shard,
        'rows': int(total_rows),
        'sequences': int(n_seq),
        'seconds': round(secs, 1),
        'dtype': 'float16',
        'causal': True,
        'exact': True,
        'note': 'Replayed through the original graph rather than a torch rebuild, '
                'so the labels are exact. Hard-batched to batch 1, so parallelism '
                'is across sequences, never along time. The prev state carries the '
                'normalised row forward; zeroed at sequence start and every '
                'multiple of state_reset, exactly as the served model does.',
    }
    (out_dir / f'teacher_{args.tag}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'\nwrote {out_npy.relative_to(ROOT)}  {total_rows:,} x 2  in {secs / 60:.1f} min')

    arr = np.load(out_npy, mmap_mode='r')
    n_probe = min(32, n_seq)
    probe = np.concatenate([np.asarray(arr[int(idx[s]):int(idx[s + 1])])
                            for s in range(n_probe)])
    finite = bool(np.isfinite(probe).all())
    zeros = int((probe == 0).all(axis=1).sum())
    print(f'finite check over {len(probe):,} rows of {n_probe} sequences: '
          f'{"OK" if finite else "FAILED"}')
    print(f'all-zero rows in probe: {zeros:,} of {len(probe):,} '
          f'({zeros / len(probe) * 100:.1f}%) -- a few are legitimate, the model '
          f'can output near-zero')
    if not finite:
        raise SystemExit('teacher labels contain non-finite values; refusing to continue')

    firsts = np.asarray([arr[int(idx[s])] for s in range(n_probe)])
    ok = bool(np.isfinite(firsts).all() and not np.all(firsts == 0))
    print(f'all {n_probe} sequence-leading rows written and non-zero: {ok}')
    if not ok:
        raise SystemExit('sequence-leading rows are zero; the replay skipped them')


if __name__ == '__main__':
    raise SystemExit(main())
