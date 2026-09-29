#!/usr/bin/env python3
"""Train a P(is_scored) predictor and label the training rows with it.

WHY THIS EXISTS
---------------
Only ~10 % of required rows carry `is_scored`, and they are a *different regime*,
not a random sample: a model trained without any mask scores 0.58 on scored rows
and 0.30 on all rows. The metric only ever sees the scored ones, so the training
loss should concentrate on rows that look scored. That is this script's output.

It is legal: the FAQ explicitly allows "your own validation splits and masks using
the supplied labelled data", and everything here is derived from `valid.parquet`
(which ships `is_scored`) and `train.parquet` features. No external data.

It is TRAINING-ONLY. Nothing here is needed at inference, so it adds no latency and
no bytes to the submission.

Method
------
Small stateful GRU (1 layer, 64 hidden) over the 112 raw features -> logit. The
GRU is the right shape because the mask is sticky: P(scored | previous scored) is
~0.77, so the label depends on short-range history.

SPLIT DISCIPLINE (the important part)
-------------------------------------
`--mask-start` / `--mask-end` bound the sequences used to FIT the mask. The default
leaves sequences 0-199 untouched, because every version in this repo evaluates on
exactly those. A mask fitted on rows we also score on would leak.

    fit    : sequences [mask_start, mask_holdout)   train the predictor
    holdout: sequences [mask_holdout, mask_end)     early stopping + reported AUC
    eval   : sequences [0, mask_start)              NEVER touched here

Examples
--------
    # fit + label (writes cache/train_scored.npy, float16, ~424 MB)
    venv/bin/python tools/train_mask_model.py --epochs 4

    # smoke the plumbing on a few sequences
    venv/bin/python tools/train_mask_model.py --epochs 1 --max-sequences 60
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

SEQUENCE_LENGTH = 20_000
WARMUP = 99
N_FEATURES = 112
WARMUP_STATES = 200_000   # flat state used to un-warm the recurrent state


class MaskGRU(nn.Module):
    """logit(is_scored) = w.x + GRU(x), i.e. a per-row linear path PLUS a context path.

    WHY THE LINEAR SKIP (measured 2026-09-29): plain logistic regression on the
    CURRENT ROW, using no history at all, already reaches holdout AUC 0.807, with
    `a3` (column 107) carrying most of the weight. The mask is therefore mostly a
    per-row function. A pure GRU has to route that signal through a recurrent state
    carried across 2,000-row chunks, and it does not: its loss falls (it learns the
    base rate) while its AUC sits at 0.507, i.e. chance. The skip gives the model
    the easy per-row path directly and lets the recurrence add only whatever genuine
    history-dependent part exists on top.
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(N_FEATURES, hidden, 1, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.skip = nn.Linear(N_FEATURES, 1)      # per-row path, no recurrence

    def forward_chunk(self, x, h):
        y, hn = self.gru(x, h)
        return (self.head(y) + self.skip(x)).squeeze(-1), hn


def auc(scores, labels):
    """Rank-based AUC; ties averaged. Pure numpy so it matches no sklearn assumption."""
    order = np.argsort(scores, kind='mergesort')
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    lab = labels[order]
    n_pos = float(lab.sum())
    n_neg = float(len(lab) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[lab == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_split(root, seq_ids):
    """Return (X_memmap, ids, labels) for a set of sequences.

    X is NOT copied or stacked. Stacking the default fit range as float32 costs
    15 GB (1400 x 20,000 x 112 x 4) and would thrash alongside the concurrent
    training runs; the valid shard is already a float16 memmap, so we index into
    it per sequence and cast the one chunk we are using. Only the bool labels are
    materialised, at 1400 x 20,000 = 28 MB.
    """
    X = np.load(root / 'valid' / 'X.npy', mmap_mode='r')
    sc = np.load(root / 'valid' / 'is_scored.npy', mmap_mode='r')
    idx = np.load(root / 'valid' / 'seq_index.npy')
    ids = np.asarray(seq_ids, dtype=np.int64)
    S = np.stack([np.asarray(sc[int(idx[i]):int(idx[i + 1])], dtype=bool) for i in ids])
    return X, ids, S, idx


def chunks(X, ids, S, idx, lanes, chunk, hidden):
    """Yield (Xb, yb, valid, h) walking `lanes` sequences `chunk` rows at a time.

    A lane that exhausts its sequence starts a new one: its hidden state is zeroed
    and its positions past the end are masked out of the loss. Zeroing the WHOLE
    batch state when any lane wraps would corrupt the other lanes, so the reset is
    per-lane.
    """
    pos = np.zeros(lanes, dtype=np.int64)
    h = torch.zeros(1, lanes, hidden)
    while True:
        Xb = np.zeros((lanes, chunk, N_FEATURES), dtype=np.float32)
        yb = np.zeros((lanes, chunk), dtype=np.float32)
        vb = np.zeros((lanes, chunk), dtype=np.float32)
        wrapped = []
        for i in range(lanes):
            if pos[i] >= SEQUENCE_LENGTH:          # this lane begins a new sequence
                pos[i] = 0
                wrapped.append(i)
            a = pos[i]
            b = min(a + chunk, SEQUENCE_LENGTH)
            n = b - a
            off = int(idx[ids[i]])
            Xb[i, :n] = np.asarray(X[off + a:off + b], dtype=np.float32)
            yb[i, :n] = S[i, a:b].astype(np.float32)
            vb[i, :n] = 1.0
            pos[i] = b
        for i in wrapped:
            h[0, i, :].zero_()
        yield torch.from_numpy(Xb), torch.from_numpy(yb), torch.from_numpy(vb), h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--lanes', type=int, default=32)
    ap.add_argument('--chunk', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--steps', type=int, default=600, help='optimiser steps per epoch')
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--pos-weight', type=float, default=1.0,
                    help='class weight for is_scored=True (~10%% of rows). Leave at 1.0: '
                         'the output is used as a calibrated probability, and reweighting '
                         'the classes would break that calibration.')
    ap.add_argument('--mask-start', type=int, default=200,
                    help='sequences below this are NEVER used to fit the mask')
    ap.add_argument('--mask-holdout', type=int, default=1600,
                    help='sequences [mask_start, mask_holdout) fit; [mask_holdout, end) holdout')
    ap.add_argument('--max-sequences', type=int, default=None)
    ap.add_argument('--out', default=str(ROOT / 'cache' / 'train_scored.npy'))
    ap.add_argument('--ckpt', default=str(ROOT / 'cache' / 'mask_model.pt'))
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--no-label', action='store_true', help='train and report, skip writing labels')
    ap.add_argument('--reuse-ckpt', action='store_true',
                    help='skip training and reuse an existing --ckpt, going straight to '
                         'labelling. Use when training already succeeded and only the '
                         'labelling pass failed.')
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    root = ROOT / 'cache'
    n_valid = len(np.load(root / 'valid' / 'seq_index.npy')) - 1
    holdout_end = n_valid if args.max_sequences is None else min(args.max_sequences, n_valid)
    fit_ids = list(range(args.mask_start, min(args.mask_holdout, holdout_end)))
    hold_ids = list(range(min(args.mask_holdout, holdout_end), holdout_end))
    assert fit_ids and hold_ids, 'no sequences in the fit or holdout range'
    print(f'fit     : {len(fit_ids)} sequences [{fit_ids[0]}..{fit_ids[-1]}]', flush=True)
    print(f'holdout : {len(hold_ids)} sequences [{hold_ids[0]}..{hold_ids[-1]}]', flush=True)
    print(f'EVAL    : sequences [0..{args.mask_start - 1}] are never touched here '
          f'(that is what every version in this repo scores on)', flush=True)

    X_all, ids_fit, Sf, vidx = load_split(root, fit_ids)
    _, ids_hold, Sh, _ = load_split(root, hold_ids)
    print(f'positive rate: fit {Sf.mean():.4f}   holdout {Sh.mean():.4f}', flush=True)

    m = MaskGRU(args.hidden).to(dev)
    if args.reuse_ckpt:
        from pathlib import Path as _P
        if not _P(args.ckpt).exists():
            raise SystemExit(f'--reuse-ckpt but {args.ckpt} does not exist')
        st = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        m.load_state_dict(st['model'])
        best_auc = st.get('auc', float('nan'))
        print(f'reusing {args.ckpt} (holdout AUC {best_auc:.4f}); skipping training',
              flush=True)
        if best_auc < 0.85:
            print('ABORT: the reused checkpoint is below 0.85 AUC; the soft mask would be a '
                  'no-op and v011 would quietly become a state-reset-only run.', flush=True)
            return 3
        return finish(m, args, root, dev, best_auc)

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    pw = torch.tensor(args.pos_weight, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')
    best, best_auc, best_state = -1.0, -1.0, None
    rng = np.random.default_rng(1337)

    for ep in range(args.epochs):
        m.train()
        order = rng.permutation(len(fit_ids))
        gen = chunks(X_all, ids_fit[order], Sf[order], vidx, args.lanes,
                     args.chunk, args.hidden)
        running = 0.0
        for step in range(args.steps):
            Xb, yb, vb, h = next(gen)
            Xb, yb, vb = Xb.to(dev), yb.to(dev), vb.to(dev)
            logit, h = m.forward_chunk(Xb, h)
            # padded positions are excluded from the loss
            loss = (lossf(logit, yb) * vb).sum() / vb.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            h = h.detach()
            running += float(loss.detach())
            if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
                a = evaluate_mask(m, X_all, ids_hold, Sh, vidx, args.lanes,
                                 args.chunk, dev)
                if a > best_auc:
                    best_auc = a
                    best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
                print(f'ep {ep} step {step+1}/{args.steps} loss {running/(step+1):.5f} '
                      f'holdout AUC {a:.4f} (best {best_auc:.4f})', flush=True)
        best = running / args.steps

    m.load_state_dict(best_state)
    m.eval()
    print(f'\nbest holdout AUC: {best_auc:.4f}', flush=True)
    if best_auc < 0.85:
        print('WARNING: AUC below 0.85 means the mask is barely predictable; the soft '
              'mask will be close to a no-op. Do not use it.')
    torch.save({'model': m.state_dict(), 'hidden': args.hidden, 'auc': best_auc,
                'fit_ids': [fit_ids[0], fit_ids[-1]], 'holdout_ids': [hold_ids[0], hold_ids[-1]]},
               args.ckpt)
    print(f'saved predictor -> {args.ckpt}', flush=True)

    if args.no_label:
        return 0
    return finish(m, args, root, dev, best_auc)


def finish(m, args, root, dev, best_auc=float('nan')):
    """Label every training row with P(is_scored) and write the manifest."""
    n_train = len(np.load(root / 'train' / 'seq_index.npy')) - 1
    Xtr = np.load(root / 'train' / 'X.npy', mmap_mode='r')
    idx = np.load(root / 'train' / 'seq_index.npy')
    out = np.lib.format.open_memmap(args.out, mode='w+', dtype=np.float16,
                                    shape=(n_train * SEQUENCE_LENGTH,))
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(n_train), desc='label train'):
            a, b = int(idx[i]), int(idx[i + 1])
            x = torch.from_numpy(np.asarray(Xtr[a:b], dtype=np.float32)).unsqueeze(0)
            h = torch.zeros(1, 1, args.hidden)
            p = np.empty(SEQUENCE_LENGTH, dtype=np.float32)
            for c in range(0, SEQUENCE_LENGTH, args.chunk):
                d = min(c + chunk_len(args), SEQUENCE_LENGTH)
                logit, h = m.forward_chunk(x[:, c:d], h)
                p[c:d] = torch.sigmoid(logit[0]).numpy()
            out[a:b] = p.astype(np.float16)
    out.flush()
    sample = np.asarray(out[::1003], dtype=np.float32)
    print(f'wrote {args.out}  ({n_train:,} rows, {(time.time()-t0)/60:.1f} min, '
          f'mean P(scored) {sample.mean():.4f}, frac>0.5 {float((sample>0.5).mean()):.4f})',
          flush=True)
    meta = {'auc': best_auc, 'hidden': args.hidden, 'rows': int(n_train * SEQUENCE_LENGTH),
            'mask_start': args.mask_start, 'mask_holdout': args.mask_holdout,
            'note': 'fitted on validation sequences [mask_start, mask_holdout), '
                    'early-stopped on [mask_holdout, end); sequences [0, mask_start) are '
                    'never touched and remain the repo-wide evaluation set',
            'args': vars(args)}
    Path(args.out).with_suffix('.json').write_text(json.dumps(meta, indent=1))
    return 0


def chunk_len(args):
    return args.chunk


@torch.no_grad()
def evaluate_mask(m, X, ids, S, idx, lanes, chunk, dev):
    """AUC over whole sequences, state carried, reset at each sequence start.

    The per-batch score matrix is filled across the time axis and then batches are
    joined, so scores and labels stay row-aligned. Concatenating the per-CHUNK
    outputs instead silently produced 640,000 scores against 5,460,000 labels and
    reported AUC 0.506 (chance) for a model whose real holdout AUC is 0.836.
    """
    scores, labels = [], []
    for s0 in range(0, len(ids), lanes):
        sel = ids[s0:s0 + lanes]
        xb = torch.from_numpy(np.stack([
            np.asarray(X[int(idx[j]):int(idx[j]) + SEQUENCE_LENGTH], dtype=np.float32)
            for j in sel])).to(dev)
        sb = S[s0:s0 + lanes]
        B = xb.shape[0]
        h = torch.zeros(1, B, m.gru.hidden_size, device=dev)
        sc = np.empty((B, SEQUENCE_LENGTH), dtype=np.float32)
        for c in range(0, SEQUENCE_LENGTH, chunk):
            d = min(c + chunk, SEQUENCE_LENGTH)
            logit, h = m.forward_chunk(xb[:, c:d], h)
            sc[:, c:d] = torch.sigmoid(logit).cpu().numpy()
        scores.append(sc)
        labels.append(sb)
    return auc(np.concatenate(scores, axis=0).reshape(-1),
               np.concatenate(labels, axis=0).reshape(-1))


if __name__ == '__main__':
    raise SystemExit(main())
