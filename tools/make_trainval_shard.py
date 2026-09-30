"""Build a combined train + held-out-validation shard.

WHY
---
dozken's recipe (`challenge-3/README.md`) trains on

    train.parquet + the non-held-out validation sequences   (`--add-valid`)

which is 10,607 + 1,681 = 12,288 sequences, i.e. +15.8% more data than we use.
`docs/faq.md` permits it explicitly: "All solutions must be trained using only the
provided train.parquet and valid.parquet". We never did it. Every one of our
~19 runs trained on the train shard alone.

The eval split stays clean. Our historical eval subset is validation sequences
0..199, so training adds sequences 200..1872 only -- the SAME split the mask model
already fits on, so the soft mask and the training data agree on what is held out.

Writes a shard directory with the same layout `tools/make_shards.py` produces, so
`ShardStore` reads it unchanged and no training script needs to know it happened.
Sequences are copied one at a time (20,000 x 112 float16 = 4.5 MB) so peak RAM is
a few MB regardless of shard size.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np


def copy_shard(src: Path, dst: Path, seq_from: int, seq_to: int, tag: str,
               y_dtype=np.float32):
    """Append sequences [seq_from, seq_to) of `src` into the memmaps already in `dst`."""
    meta = json.loads((src / 'meta.json').read_text())
    seq_len = meta['sequence_length']
    idx = np.load(src / 'seq_index.npy')
    out_idx = np.load(dst / 'seq_index.npy', mmap_mode='r')
    n_before = len(out_idx) - 1

    src_X = np.load(src / 'X.npy', mmap_mode='r')
    src_y = np.load(src / 'y.npy', mmap_mode='r')
    src_nd = np.load(src / 'need_prediction.npy', mmap_mode='r')
    src_sc = (np.load(src / 'is_scored.npy', mmap_mode='r')
              if (src / 'is_scored.npy').exists() else None)

    dst_X = np.load(dst / 'X.npy', mmap_mode='r+')
    dst_y = np.load(dst / 'y.npy', mmap_mode='r+')
    dst_nd = np.load(dst / 'need_prediction.npy', mmap_mode='r+')
    dst_sc = (np.load(dst / 'is_scored.npy', mmap_mode='r+')
              if (dst / 'is_scored.npy').exists() else None)

    for j, s in enumerate(range(seq_from, seq_to)):
        a, b = int(idx[s]), int(idx[s + 1])
        o = (n_before + j) * seq_len
        n = b - a
        dst_X[o:o + n] = src_X[a:b]
        dst_y[o:o + n] = src_y[a:b].astype(y_dtype)
        dst_nd[o:o + n] = src_nd[a:b]
        if dst_sc is not None:
            # The train shard has no is_scored column. Those rows are simply never
            # scored, so fill False -- it preserves dtype and shape without
            # inventing labels, and ShardStore treats a missing column the same way.
            dst_sc[o:o + n] = (src_sc[a:b] if src_sc is not None
                               else np.zeros(n, dtype=bool))
        if (j + 1) % 500 == 0:
            print(f'  {tag} {j + 1}/{seq_to - seq_from} sequences', flush=True)
    for arr in (dst_X, dst_y, dst_nd) + ((dst_sc,) if dst_sc is not None else ()):
        arr.flush()

    # extend seq_index last, so a partial build is never loadable as complete
    full_idx = np.arange(0, (n_before + (seq_to - seq_from) + 1) * seq_len, seq_len, dtype=np.int64)
    np.save(dst / 'seq_index.npy', full_idx)
    return n_before + (seq_to - seq_from)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', default='cache/train')
    ap.add_argument('--valid', default='cache/valid')
    ap.add_argument('--out', default='cache/trainval')
    ap.add_argument('--valid-start', type=int, default=200,
                    help='first validation sequence to ADD to training (default 200; '
                         'sequences 0..199 are our eval subset and must stay out)')
    ap.add_argument('--force', action='store_true', help='rebuild if --out already exists')
    args = ap.parse_args()

    tmeta = json.loads((Path(args.train) / 'meta.json').read_text())
    vmeta = json.loads((Path(args.valid) / 'meta.json').read_text())
    seq_len = tmeta['sequence_length']
    assert vmeta['sequence_length'] == seq_len, 'sequence lengths differ'
    n_train = tmeta['sequences']
    n_valid_total = vmeta['sequences']
    vidx = np.load(Path(args.valid) / 'seq_index.npy')
    n_add = n_valid_total - args.valid_start
    if n_add <= 0:
        raise SystemExit(f'--valid-start {args.valid_start} leaves nothing to add')

    out = Path(args.out)
    if out.exists():
        if not args.force:
            raise SystemExit(f'{out} exists; pass --force to rebuild')
        shutil.rmtree(out)
    out.mkdir(parents=True)

    total_rows = (n_train + n_add) * seq_len
    total_seq = n_train + n_add
    print(f'building {out}: {n_train} train + {n_add} valid[{args.valid_start}:{n_valid_total}]'
          f' = {total_seq} sequences, {total_rows:,} rows')

    t0 = time.time()
    # allocate the output memmaps up front, then fill them
    np.lib.format.open_memmap(out / 'X.npy', mode='w+', dtype=np.float16,
                              shape=(total_rows, tmeta['feature_columns'].__len__()))
    np.lib.format.open_memmap(out / 'y.npy', mode='w+', dtype=np.float32,
                              shape=(total_rows, 2))
    np.lib.format.open_memmap(out / 'need_prediction.npy', mode='w+', dtype=bool,
                              shape=(total_rows,))
    np.lib.format.open_memmap(out / 'is_scored.npy', mode='w+', dtype=bool,
                              shape=(total_rows,))
    # Seed with ONE entry so copy_shard sees n_before == 0 for the first shard and
    # extends the index itself. Seeding the full range made n_before == total_seq
    # and every write landed past the end of the arrays.
    np.save(out / 'seq_index.npy', np.array([0], dtype=np.int64))

    copy_shard(Path(args.train), out, 0, n_train, 'train')
    copy_shard(Path(args.valid), out, args.valid_start, n_valid_total, 'valid')

    meta = dict(tmeta)
    meta.update(sequences=total_seq, rows=total_rows, start_seq=0,
                generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                source='datasets/train.parquet + datasets/valid.parquet',
                source_row_groups=n_train + n_add,
                source_bytes=int(tmeta['source_bytes']) + int(vmeta['source_bytes']),
                train_sequences=n_train,
                added_valid_sequences=n_add,
                valid_added_from=args.valid_start,
                eval_sequences_held_out=f'0..{args.valid_start - 1}')
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(f'\ndone: {total_seq} sequences, {total_rows:,} rows, '
          f'{(time.time() - t0) / 60:.1f} min -> {out}')


if __name__ == '__main__':
    raise SystemExit(main())
