"""Generate distillation teacher labels by replaying trained models over a shard.

WHY THIS EXISTS
---------------
dozken's NOTES (`2026-09-17_distil_ens`) describe the only lever that decouples
teacher cost from serving cost: the teacher may be arbitrarily slow because it
never runs at inference. They distilled the raw-mean predictions of reset2k +
reset4k, replaying each on its OWN state_reset grid, into a single student that
matched the 2-graph teacher exactly (0.6879 vs 0.6879) after the 2-graph version
itself was killed at 69m59s.

Their teacher carries no soft mask. Ours can: the P(is_scored) loss weighting was
worth +0.0058 (36 sigma) locally at zero inference cost, so a teacher built from
masked members should start from a stronger teacher, and that strength propagates
into the student. This script is the "Phase B" that produces those labels.

WHAT IT WRITES
--------------
`<out>/teacher_<tag>.npy`  float16 (n_rows, 2), the raw prediction of one member
                           over every row of the shard, row-major, in shard order.
`<out>/teacher_<tag>.json` provenance: checkpoint, config, param count, sha256,
                           and the shard it was replayed over.

Members are written SEPARATELY and averaged by the training script, not averaged
here, because each member must be replayed on its own state_reset grid -- a 4,000
reset and a 2,000 reset have to see the state trajectory they were trained with,
and averaging first would destroy that.

CAUSALITY
---------
Replay is strictly causal and mirrors inference exactly: one lane per sequence,
hidden state carried chunk to chunk, zeroed at the start of each sequence and at
every multiple of state_reset. No lookahead, no statistic computed across
sequences, nothing from the future. The teacher sees the same inputs a served
model would see, so its labels are deployable knowledge and not hindsight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
SEQ_LEN = 20_000
WARMUP = 99
N_FEATURES = 112


# ===== SECTION: TRAINING IMPORTS & UTILITIES =====


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def build_model(cfg: dict):
    """Reconstruct the torch module from the version yaml (no checkpoint needed)."""
    import importlib.util
    script = ROOT / 'exploration' / 'models' / f"train_{cfg['identity']['model']}_{cfg['identity']['version']}.py"
    spec = importlib.util.spec_from_file_location(f"mod_{cfg['identity']['version']}", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.GRUModel(N_FEATURES, hidden=cfg['model']['hidden'],
                     layers=cfg['model']['layers'],
                     input_proj=cfg['model']['input_proj'],
                     dropout=cfg['model'].get('dropout', 0.0))
    return m, mod


class Transform:
    """The version's feature transform, applied exactly as at train and serve.

    The teacher must consume the SAME features the served model would, or it
    distils knowledge that does not exist at inference. Configured from
    feature_stats.npz only, never the yaml, because FeatureTransform itself
    documents that the yaml is not guaranteed to exist where a model runs.

    `.batch` and `.step` agree within 1e-6 by construction (train_template.py
    feature_parity_test), so using the vectorised path here is safe.
    """

    def __init__(self, mod, stats_path: Path):
        with np.load(stats_path) as z:
            self.t = mod.FeatureTransform({k: z[k] for k in z.files})

    def batch(self, X: np.ndarray) -> np.ndarray:
        return self.t.batch(X)


# ===== SECTION: DATA =====


def replay(model, transform, store_root: Path, out_path: Path, state_reset: int,
           chunk: int, lanes: int, device: str, cfg_hidden: int, cfg_layers: int,
           progress_every: int = 200):
    """Walk every sequence of the shard, writing that model's prediction per row."""
    X = np.load(store_root / 'X.npy', mmap_mode='r')
    idx = np.load(store_root / 'seq_index.npy')
    n_seq = int(idx[-1]) if idx[0] != 0 else int(idx[-1])
    seq_len = int(X.shape[0] // len(idx)) if len(idx) > 1 else SEQ_LEN
    total_rows = int(X.shape[0])
    print(f'replay: {len(idx) - 1} sequences, {total_rows:,} rows, '
          f'state_reset={state_reset}, chunk={chunk}, lanes={lanes}', flush=True)

    # (n_rows, 2) float16 = ~849 MB for the full train shard. Written once, read
    # by the distillation script; float16 is well below the model's own noise.
    out = np.lib.format.open_memmap(out_path, mode='w+', dtype=np.float16,
                                    shape=(total_rows, 2))
    model.eval()
    hidden = int(cfg_hidden)
    t0 = time.time()
    written = 0
    with torch.no_grad():
        for i in range(len(idx) - 1):
            a, b = int(idx[i]), int(idx[i + 1])
            # Fresh zero state per sequence, matching train_template.py:842 and
            # the served model, which zeros at every new seq_ix. Batch is 1.
            h = [torch.zeros(1, 1, hidden, device=device) for _ in range(cfg_layers)]
            for c0 in range(a, b, chunk):
                c1 = min(c0 + chunk, b)
                xb = np.asarray(X[c0:c1], dtype=np.float32)
                xb = transform.batch(xb)
                xt = torch.from_numpy(xb).unsqueeze(0).to(device)   # (1, T, F)
                step0 = c0 - a
                # zero the state at a sequence start and on the reset grid, the
                # same instants the served model zeroes it
                if state_reset and step0 and step0 % state_reset == 0:
                    h = [torch.zeros(1, 1, hidden, device=device)
                         for _ in range(cfg_layers)]
                p, h = model.forward_chunk(xt, h)
                out[c0:c1] = p.squeeze(0).cpu().numpy().astype(np.float16)
                written += c1 - c0
            if (i + 1) % progress_every == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                print(f'  seq {i + 1}/{len(idx) - 1}  {written:,} rows  '
                      f'{el:.0f}s  {rate:.2f} seq/s  '
                      f'ETA {(len(idx) - 1 - (i + 1)) / rate / 60:.0f} min', flush=True)
    out.flush()
    del out
    return total_rows, written, time.time() - t0


# ===== SECTION: MAIN =====


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', required=True,
                    help='version whose best checkpoint becomes a teacher member, e.g. v011')
    ap.add_argument('--shard', default='cache/train',
                    help='shard root to replay over (default cache/train)')
    ap.add_argument('--out', default='cache/teacher',
                    help='output directory (default cache/teacher)')
    ap.add_argument('--tag', default=None,
                    help='output filename stem (default: the version)')
    ap.add_argument('--checkpoint', default=None,
                    help='best_*.pt to replay (default: highest val in the checkpoint dir)')
    ap.add_argument('--chunk', type=int, default=2000,
                    help='rows per forward (must divide chunk_len of the member)')
    ap.add_argument('--lanes', type=int, default=1,
                    help='kept for interface compatibility; replay is strictly sequential '
                         'so lanes=1 is the only correct value')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    if args.lanes != 1:
        ap.error('--lanes must be 1: replay must be strictly sequential and causal')

    tag = args.tag or args.version
    cfg_path = ROOT / 'exploration' / 'version' / f'version_gru_{args.version}.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    ck_dir = ROOT / cfg['artifacts']['checkpoint_dir']

    if args.checkpoint:
        ck = Path(args.checkpoint)
    else:
        cks = sorted(ck_dir.glob('best_*.pt'))
        if not cks:
            sys.exit(f'no best_*.pt in {ck_dir}')

        def _val(p):
            try:
                return float(p.stem.split('val')[-1])
            except ValueError:
                return -1.0
        ck = max(cks, key=_val)
    print(f'member {args.version}: {ck.name}', flush=True)

    st = torch.load(ck, map_location=args.device, weights_only=False)
    model, mod = build_model(cfg)
    model.load_state_dict(st['model'])
    model.to(args.device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    # The replay grid MUST match the member's own training/serving reset, or the
    # state trajectory differs from what it was trained under.
    state_reset = int(cfg['training'].get('state_reset') or 0)
    chunk = args.chunk
    if cfg['training']['chunk_len'] % chunk:
        sys.exit(f'--chunk {chunk} must divide the member chunk_len '
                 f'{cfg["training"]["chunk_len"]}')

    stats = ck_dir / 'feature_stats.npz'
    if not stats.exists():
        sys.exit(f'missing {stats}; run the trainer with --export-only first')
    transform = Transform(mod, stats)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = out_dir / f'teacher_{tag}.npy'

    n_rows, written, secs = replay(model, transform, ROOT / args.shard, out_npy,
                                   state_reset, chunk, args.lanes, args.device,
                                   cfg_hidden=int(cfg['model']['hidden']),
                                   cfg_layers=int(cfg['model']['layers']))

    manifest = {
        'tag': tag,
        'version': args.version,
        'checkpoint': str(ck.relative_to(ROOT)),
        'checkpoint_sha256': sha256(ck),
        'config': str(cfg_path.relative_to(ROOT)),
        'config_sha256': sha256(cfg_path),
        'params': n_params,
        'state_reset': state_reset,
        'chunk': chunk,
        'shard': args.shard,
        'rows': int(n_rows),
        'rows_written': int(written),
        'seconds': round(secs, 1),
        'dtype': 'float16',
        'causal': True,
        'note': 'raw prediction of one teacher member, replayed on its own '
                'state_reset grid. Members are averaged downstream, not here.',
    }
    (out_dir / f'teacher_{tag}.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'\nwrote {out_npy.relative_to(ROOT)}  {n_rows:,} rows x 2  in {secs / 60:.1f} min')
    print(f'wrote {out_dir.relative_to(ROOT)}/teacher_{tag}.json')

    # Finite check: a label file containing NaN or inf would poison distillation
    # silently, because the student just regresses onto whatever is there. Note
    # rows 0..98 of each sequence are warm-up and legitimately never written.
    arr = np.load(out_npy, mmap_mode='r')
    idx = np.load((ROOT / args.shard) / 'seq_index.npy')
    seq_len = int(cfg['data']['seq_len'])
    n_probe = min(64, len(idx) - 1)
    probe = np.concatenate([np.asarray(arr[int(idx[s]):int(idx[s + 1]) - 1])
                            for s in range(n_probe)])
    finite = bool(np.isfinite(probe).all())
    print(f'finite check over {len(probe):,} rows of {n_probe} sequences: '
          f'{"OK" if finite else "FAILED - do not use"}')
    if not finite:
        raise SystemExit('teacher labels contain non-finite values; refusing to continue')

    # The replay deliberately labels EVERY row of every sequence, warm-up
    # included: the teacher is a forward pass of a fully-trained model and has no
    # warm-up semantics of its own. The distillation loss masks the burn-in rows
    # downstream exactly as training does, so keeping them is harmless and avoids
    # a special case. What matters is that the file is row-aligned to the shard and
    # that nothing is unwritten, so assert every row of a probe sequence is finite
    # INCLUDING its first, which would be a zero fill if the replay had skipped it.
    firsts = np.asarray([arr[int(idx[s])] for s in range(n_probe)])
    firsts_ok = bool(np.isfinite(firsts).all() and not np.all(firsts == 0))
    print(f'all {n_probe} sequence-leading rows written and finite: {firsts_ok}')
    if not firsts_ok:
        raise SystemExit('sequence-leading rows are zero or non-finite, so the replay '
                         'skipped them and the file is not what we expect')


if __name__ == '__main__':
    raise SystemExit(main())
