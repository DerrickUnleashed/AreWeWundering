#!/usr/bin/env python3
"""Print a leaderboard table of every version yaml, sorted by validation WP.

WHY: the repo holds one yaml per experiment (AGENTS.md §6) and the yaml is the
single source of truth for hyper-parameters, status and results.  Reading a dozen
of them by hand before deciding what to run next is the slowest part of the
workflow, and stale `running` statuses hide runs that died silently.

It also flags integrity problems, because a missing log or checkpoint folder
means a result cannot be reproduced or promoted:
  * a version with no log or no checkpoint folder
  * a version whose status is `running` but whose log has not grown for > 1 h
    (compare the log mtime with the process list rather than guessing)

Examples
--------
    python tools/summarize.py
    python tools/summarize.py --csv
    python tools/summarize.py --key gru_v001
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

STALE_HOURS = 1.0


def running_pids():
    out = subprocess.run(['ps', '-eo', 'pid,command'], capture_output=True, text=True).stdout
    pids = set()
    for line in out.splitlines():
        if 'train_' in line and 'summarize' not in line:
            parts = line.strip().split(None, 1)
            if parts and parts[0].isdigit():
                pids.add(int(parts[0]))
    return pids


def collect():
    rows, problems = [], []
    vdir = ROOT / 'exploration' / 'version'
    for p in sorted(vdir.glob('version_*.yaml')):
        key = p.stem.replace('version_', '')
        try:
            cfg = yaml.safe_load(p.read_text()) or {}
        except Exception as e:
            problems.append((key, f'yaml unreadable: {e}'))
            continue
        ident = cfg.get('identity', {})
        res = cfg.get('results') or {}
        art = cfg.get('artifacts') or {}
        log = ROOT / art.get('log', f'exploration/logs/log_{key}.log') if art.get('log') else None
        ck = ROOT / art.get('checkpoint_dir', f'exploration/checkpoints/checkpoint_{key}') \
            if art.get('checkpoint_dir') else None
        n_ck = len(list(ck.glob('best_*.pt'))) if ck and ck.exists() else 0
        stale = None
        if ident.get('status') == 'running':
            pids = running_pids()
            mine = log and any(f'train_{key}' in
                               subprocess.run(['ps', '-p', str(pid), '-o', 'command='],
                                              capture_output=True, text=True).stdout
                               for pid in pids)
            if log and log.exists() and not mine:
                age_h = (time.time() - log.stat().st_mtime) / 3600
                if age_h > STALE_HOURS:
                    stale = round(age_h, 2)
        rows.append({
            'key': key, 'parent': ident.get('parent'), 'status': ident.get('status'),
            'params': (cfg.get('model') or {}).get('param_count'),
            'hidden': (cfg.get('model') or {}).get('hidden'),
            'layers': (cfg.get('model') or {}).get('layers'),
            'wp': res.get('best_val_wp'), 't0': res.get('best_val_t0'),
            't1': res.get('best_val_t1'), 'full': res.get('full_valid_wp'),
            'p50': res.get('latency_us_p50'), 'p99': res.get('latency_us_p99'),
            'step': res.get('best_step'), 'hours': res.get('train_hours'),
            'change': (cfg.get('change_vs_parent') or '').strip().splitlines()[0][:60]
            if cfg.get('change_vs_parent') else '',
            'has_log': bool(log and log.exists()),
            'n_ck': n_ck, 'submission': art.get('submission_dir'),
            'stale_h': stale,
        })
        if not rows[-1]['has_log']:
            problems.append((key, 'no log file (AGENTS.md §8)'))
        if n_ck == 0:
            problems.append((key, 'no checkpoint in the checkpoint folder'))
        if stale:
            problems.append((key, f'status=running but the log has not grown for {stale} h'))
    return rows, problems


def fmt(v, spec, dash='-'):
    return dash if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', action='store_true')
    ap.add_argument('--key', default=None, help='show full detail for one experiment key')
    args = ap.parse_args()
    rows, problems = collect()

    if args.key:
        r = [x for x in rows if x['key'] == args.key]
        if not r:
            raise SystemExit(f'unknown key {args.key}')
        r = r[0]
        print(f'{r["key"]}  status={r["status"]}  parent={r["parent"]}')
        for k in ('params', 'wp', 't0', 't1', 'full', 'p50', 'p99', 'step', 'hours',
                  'n_ck', 'has_log', 'submission', 'change'):
            print(f'  {k:12}: {r[k]}')
        for key, msg in problems:
            if key == args.key:
                print(f'  PROBLEM     : {msg}')
        return 0

    rows.sort(key=lambda r: (-1e9 if r['wp'] is None else r['wp']))
    if args.csv:
        import csv
        w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()) if rows else ['key'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
        return 0

    hdr = (f'{"key":<12} {"parent":<12} {"status":<10} {"hid":>4} {"L":>2} {"params":>9} '
           f'{"val_wp":>8} {"t0":>7} {"t1":>7} {"full":>8} {"us_p50":>7} {"step":>6} {"h":>6} '
           f'{"ck":>3} {"sub":<4} change')
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f'{r["key"]:<12} {str(r["parent"]):<12} {str(r["status"]):<10} '
              f'{fmt(r["hidden"], ">4")} {fmt(r["layers"], ">2")} {fmt(r["params"], ">9")} '
              f'{fmt(r["wp"], "8.4f")} {fmt(r["t0"], "7.4f")} {fmt(r["t1"], "7.4f")} '
              f'{fmt(r["full"], "8.4f")} {fmt(r["p50"], "7.1f")} {fmt(r["step"], ">6")} '
              f'{fmt(r["hours"], "6.2f")} {r["n_ck"]:>3} {("yes" if r["submission"] else "-"):<4} '
              f'{r["change"]}')
    if problems:
        print('\nproblems:')
        for key, msg in problems:
            print(f'  {key}: {msg}')
    else:
        print('\nno integrity problems found')
    print('\nnote: `val_wp` is the best subset score recorded during training; `full` is the '
          'full-validation rescore. Neither is the official test score.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
