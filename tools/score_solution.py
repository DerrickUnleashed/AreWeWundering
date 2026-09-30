#!/usr/bin/env python3
"""Run a `solution.py` through the REAL callback loop and score it with the reference metric.

WHY
---
`utils.ScorerStepByStep` is the organiser's own loop: it constructs the model once,
builds a `DataPoint` per row, checks the contract (None in warm-up, finite `(2,)`
float32 on every required row) and feeds a 20,000-row block at a time to
`GlobalAccumulator`.  Anything that scores well offline but fails here is broken,
so this is the tool the promotion gate uses (AGENTS.md §9).

    python tools/score_solution.py --solution submissions/submission_gru_v000/solution.py \
        --sequences 5
    python tools/score_solution.py --solution baseline/solution.py --full      # ~30 min
    python tools/score_solution.py --solution S.py --sequences 20 \
        --parity-against-offline --offline-script exploration/models/train_gru_v000.py \
        --offline-config exploration/version/version_gru_v000.yaml \
        --offline-checkpoint .../best_001_....pt

`--sequences` (no `--full`) is the documented fast protocol used during
development: a fixed, seeded subset of validation sequences.  A subset score is
NOT the official full-validation number and must never be quoted as one.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np  # noqa: E402
import metric_ref  # noqa: E402
from utils import (DataPoint, GlobalAccumulator, FEATURE_COLUMNS, TARGET_COLUMNS,  # noqa: E402
                   SEQUENCE_LENGTH, WARMUP)


def load_solution(path):
    spec = importlib.util.spec_from_file_location('scored_solution', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PredictionModel


def read_sequence(parquet_path, group):
    """Read one row group (== one sequence) straight from parquet, as the scorer does."""
    import pyarrow.parquet as pq
    tbl = pq.ParquetFile(parquet_path).read_row_group(group, use_threads=False)
    d = {c: tbl[c].to_numpy() for c in tbl.column_names}
    seq_ix = int(d['seq_ix'][0])
    if not np.all(d['seq_ix'] == seq_ix):
        raise SystemExit(f'row group {group} is not a single sequence')
    if not np.array_equal(d['step_in_seq'], np.arange(SEQUENCE_LENGTH)):
        raise SystemExit(f'row group {group} has a wrong step order')
    need = d['need_prediction'].astype(bool)
    if not np.array_equal(need, np.arange(SEQUENCE_LENGTH) >= WARMUP):
        raise SystemExit(f'row group {group} violates the warm-up contract')
    features = np.column_stack([d[c] for c in FEATURE_COLUMNS]).astype(np.float32)
    targets = np.column_stack([d[c] for c in TARGET_COLUMNS]).astype(np.float32)
    return seq_ix, features, targets, need, d['is_scored'].astype(bool)


def run_loop(PredictionModel, parquet_path, groups, label='', deterministic_repeat=False):
    model = PredictionModel()
    acc = GlobalAccumulator()
    t0 = time.time()
    rows = 0
    per_call = []
    for g in groups:
        seq_ix, feats, targets, need, scored = read_sequence(parquet_path, g)
        preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
        for step in range(SEQUENCE_LENGTH):
            point = DataPoint(seq_ix, step, bool(need[step]), feats[step])
            ts = time.perf_counter()
            value = model.predict(point)
            per_call.append((time.perf_counter() - ts) * 1e6)
            if not need[step]:
                if value is not None:
                    raise SystemExit('contract violation: predict returned a value during warm-up')
            else:
                value = np.asarray(value, dtype=np.float32)
                if value.shape != (2,) or not np.isfinite(value).all():
                    raise SystemExit('contract violation: required rows need two finite scores')
                preds[step] = value
        acc.add(targets, preds, scored & need)
        rows += SEQUENCE_LENGTH
        print(f'  seq {g} (id {seq_ix}) done, {rows:,} rows, {time.time()-t0:.1f}s', flush=True)
    res = acc.result()
    res['rows_seen'] = rows
    res['elapsed_s'] = round(time.time() - t0, 2)
    res['us_per_row'] = round(res['elapsed_s'] / rows * 1e6, 2)
    lat = np.array(per_call[WARMUP:])
    res['call_us_p50'] = round(float(np.percentile(lat, 50)), 2)
    res['call_us_p99'] = round(float(np.percentile(lat, 99)), 2)
    res['sequences'] = list(groups)
    res['protocol'] = 'full validation' if label == '__full__' else f'{len(groups)} validation sequences'
    res['label'] = label
    if deterministic_repeat:
        acc2 = GlobalAccumulator()
        model2 = PredictionModel()
        for g in groups:
            seq_ix, feats, targets, need, scored = read_sequence(parquet_path, g)
            preds = np.full((SEQUENCE_LENGTH, 2), np.nan, dtype=np.float32)
            for step in range(SEQUENCE_LENGTH):
                v = model2.predict(DataPoint(seq_ix, step, bool(need[step]), feats[step]))
                if need[step]:
                    preds[step] = np.asarray(v, dtype=np.float32)
            acc2.add(targets, preds, scored & need)
        res['repeat_identical'] = bool(acc2.result()['weighted_pearson'] == res['weighted_pearson'])
    return res


def offline_predict(offline_script, offline_config, offline_checkpoint, parquet_path, groups):
    """Batched, whole-sequence evaluation of the same checkpoint: the parity reference."""
    import torch
    import yaml
    spec = importlib.util.spec_from_file_location('offline_train', offline_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = yaml.safe_load(Path(offline_config).read_text())
    st = torch.load(offline_checkpoint, map_location='cpu', weights_only=False)
    transform = mod.build_transform_from_stats(st['feature_stats'], cfg['features'])
    model = mod.GRUModel(st['feature_dim'], hidden=cfg['model']['hidden'],
                         layers=cfg['model']['layers'],
                         input_proj=cfg['model'].get('input_proj') or 0)
    model.load_state_dict(st['model'])
    model.eval()
    out = {}
    with torch.no_grad():
        # The parity reference must be driven through the SAME periodic state reset
        # the served model uses, or it is not the same model. A version with
        # state_reset=4000 that is scored here with an unreseted state diverges from
        # solution.py from the first reset onward -- which is exactly how the v011
        # export bug hid: parity read 1.466 instead of failing loudly, because the
        # reference had no reset and the bundle had none either, so the two agreed
        # while both were wrong.
        state_reset = int(cfg['inference'].get('state_reset')
                          or cfg['training'].get('state_reset') or 0)
        seq_len = int(cfg['data']['seq_len'])
        chunk = int(cfg['training']['chunk_len'])
        if state_reset and (seq_len % state_reset or state_reset % chunk):
            raise SystemExit(f'offline parity cannot honour state_reset={state_reset} '
                             f'with seq_len={seq_len} chunk_len={chunk}')
        for g in groups:
            seq_ix, feats, _, _, _ = read_sequence(parquet_path, g)
            xb = transform.batch(feats)
            preds = []
            h = [torch.zeros(1, 1, model.hidden) for _ in range(model.layers)]
            with torch.no_grad():
                for c0 in range(0, xb.shape[0], chunk):
                    c1 = min(c0 + chunk, xb.shape[0])
                    if state_reset and c0 and c0 % state_reset == 0:
                        h = [torch.zeros(1, 1, model.hidden) for _ in range(model.layers)]
                    p, h = model.forward_chunk(
                        torch.from_numpy(xb[c0:c1]).unsqueeze(0), h)
                    preds.append(p[0].numpy())
            p = np.concatenate(preds, axis=0)
            # clip exactly as PredictionModel.predict does, so parity compares like with like
            out[g] = np.clip(np.asarray(p), -2.0, 2.0).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--solution', required=True)
    ap.add_argument('--parquet', default=str(ROOT / 'datasets' / 'valid.parquet'))
    ap.add_argument('--sequences', type=int, default=5, help='number of validation sequences')
    ap.add_argument('--random', type=int, default=None, metavar='N',
                    help='use a seeded random subset of N sequences instead of the first N')
    ap.add_argument('--full', action='store_true', help='score every validation sequence (~30 min)')
    ap.add_argument('--seed', type=int, default=1337)
    ap.add_argument('--deterministic-repeat', action='store_true',
                    help='run the whole loop twice and check the scores are bit-identical')
    ap.add_argument('--parity-against-offline', action='store_true')
    ap.add_argument('--offline-script')
    ap.add_argument('--offline-config')
    ap.add_argument('--offline-checkpoint')
    ap.add_argument('--parity-tol', type=float, default=1e-3,
                    help='raw prediction bound: a bug tripwire, not an accuracy claim '
                         '(float32 drifts ~3e-4 over 20k recurrent steps; a real bug is O(1e-1))')
    ap.add_argument('--parity-score-tol', type=float, default=1e-4,
                    help='the bound that actually matters: agreement in the WP score')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    import pyarrow.parquet as pq
    n_groups = pq.ParquetFile(args.parquet).num_row_groups
    if args.full:
        groups = list(range(n_groups))
    elif args.random:
        groups = sorted(np.random.default_rng(args.seed).choice(n_groups, args.random, replace=False).tolist())
    else:
        groups = list(range(min(args.sequences, n_groups)))

    PM = load_solution(args.solution)
    res = run_loop(PM, args.parquet, groups,
                   label='__full__' if args.full else str(args.solution),
                   deterministic_repeat=args.deterministic_repeat)
    print('\n' + '=' * 70)
    print(f'solution            : {args.solution}')
    print(f'protocol            : {res["protocol"]} (groups {groups if len(groups) <= 12 else groups[:6] + ["..."]})')
    print(f'rows seen           : {res["rows_seen"]:,}   scored rows: {res["selected_rows"]:,}')
    print(f'WP t0 / t1 / mean   : {res["t0"]:.6f} / {res["t1"]:.6f} / {res["weighted_pearson"]:.6f}')
    print(f'elapsed             : {res["elapsed_s"]}s  ({res["us_per_row"]} µs/row incl. loop overhead)')
    print(f'per-call µs         : p50 {res["call_us_p50"]}  p99 {res["call_us_p99"]}')
    if args.deterministic_repeat:
        print(f'deterministic       : {"PASS (identical scores)" if res["repeat_identical"] else "FAIL"}')
    print('=' * 70)

    if args.parity_against_offline:
        missing = [a for a in (args.offline_script, args.offline_config, args.offline_checkpoint)
                   if not a]
        if missing:
            raise SystemExit('--parity-against-offline needs --offline-script, --offline-config '
                             'and --offline-checkpoint')
        off = offline_predict(args.offline_script, args.offline_config, args.offline_checkpoint,
                              args.parquet, groups)
        PM2 = load_solution(args.solution)
        model = PM2()
        # Compare the SCORE, not just the raw predictions. A fully trained model run
        # row-by-row in float32 through 20,000 recurrent steps drifts from the batched
        # PyTorch path by ~3e-4 in the raw predictions (measured 2026-09-29 on
        # gru_v001, and it grows with step index: 1.6e-7 at step 0 -> 6e-5 at step
        # 10,000, with per-step agreement 2.8e-6). That drift moves the metric by
        # 3.3e-7, i.e. not at all. A REAL implementation bug -- wrong axis, wrong
        # state, a broken transform -- shows up as O(1e-1) or larger and fails both
        # bounds, so the raw bound stays as a bug tripwire, just at a defensible
        # tolerance for an accumulated recurrent state.
        acc_stream = metric_ref.StreamingWP()
        acc_offline = metric_ref.StreamingWP()
        worst = 0.0
        for g in groups:
            seq_ix, feats, targets, need, scored = read_sequence(args.parquet, g)
            stream_preds = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
            for step in range(SEQUENCE_LENGTH):
                v = model.predict(DataPoint(seq_ix, step, bool(need[step]), feats[step]))
                if need[step]:
                    v = np.asarray(v, dtype=np.float32)
                    stream_preds[step] = v
                    worst = max(worst, float(np.abs(v - off[g][step]).max()))
            m = scored & need
            acc_stream.add(targets, stream_preds, mask=m)
            acc_offline.add(targets, off[g].astype(np.float32), mask=m)
        wp_stream = acc_stream.result()['weighted_pearson']
        wp_offline = acc_offline.result()['weighted_pearson']
        score_drift = abs(wp_stream - wp_offline)
        raw_ok = worst <= args.parity_tol
        score_ok = score_drift <= args.parity_score_tol
        verdict = 'PASS' if (raw_ok and score_ok) else 'FAIL'
        print(f'parity raw predictions : max |diff| = {worst:.3e} '
              f'(tol {args.parity_tol:.0e}, float32 drift over 20k recurrent steps)  '
              f'{"PASS" if raw_ok else "FAIL"}')
        print(f'parity SCORE           : WP streaming {wp_stream:.6f} vs offline {wp_offline:.6f}, '
              f'|diff| = {score_drift:.3e} (tol {args.parity_score_tol:.0e})  '
              f'{"PASS" if score_ok else "FAIL"}')
        print(f'parity overall         : {verdict}')
        res['parity_max_abs_diff'] = worst
        res['parity_score_drift'] = score_drift
        res['parity_wp_streaming'] = wp_stream
        res['parity_wp_offline'] = wp_offline
        res['parity_pass'] = verdict == 'PASS'

    if args.json:
        import json
        print(json.dumps(res, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
