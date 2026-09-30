#!/usr/bin/env python3
"""Rebuild a training log from checkpoints when the canonical log was orphaned.

WHY THIS EXISTS
---------------
On 2026-09-30 the canonical log of gru_v018 was destroyed by an accident in this
repo, not by the training run. The run itself never failed: it completed all
19,890 steps and wrote every checkpoint.

The training process held an open write descriptor on the log file's inode. A
`git add` on that same path replaced the file (git writes by creating a new
file and renaming it over the old one, rather than editing in place). The
running process kept writing to the now-unlinked inode. So the bytes kept
accumulating -- 85,747 of them -- but nothing could reach them by name, and the
file at the path was a different inode frozen at 6,870 bytes.

Concretely, at 23:39:

    process fd  -> inode 254438603, 85,747 bytes, growing, unreachable by name
    path on disk-> inode 254443425,  6,870 bytes, stale since 18:50

`lsof` does not flag the orphan as "(deleted)" because the inode is still
allocated with an open descriptor; it is simply unreachable by pathname.

LESSON: never `git add` a file that a live process holds open. Commit live logs
only after the run exits, or copy them to a scratch path first.

WHAT THIS SCRIPT RECOVERS, AND WHAT IT DOES NOT
-----------------------------------------------
Recoverable, from durable checkpoint state:
  * the config the run used, and the parameter count
  * every periodic evaluation: step, WP, per-target WP, scored row count
  * which checkpoint each evaluation set as the new best
  * the final `full_valid_wp` / per-target scores, once the yaml records them
  * wall-clock and seed, from the checkpoints and yaml

NOT recoverable, because it only ever existed in the orphaned bytes:
  * the per-step loss and grad_norm trace
  * the exact lr at every step
  * the reproducibility header as originally printed (command line, host,
    package versions, git commit at launch)

So the output is explicitly a RECONSTRUCTION. It is written to a separate
`log_<model>_<version>.reconstructed.log` and never overwrites the canonical
log, which stays as-is: a truncated artifact that is itself evidence of the
incident. Anyone reading the reconstruction is told in its first lines that it
is derived, not original.

Usage:
    python tools/rebuild_log.py --model gru --version v018 [--out PATH]

Exit codes: 0 rebuilt, 1 refused (run unfinished / no source data), 2 error.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def human(seconds: float | None) -> str:
    if seconds is None:
        return '--'
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f'{h}h{m:02d}m{s:02d}s' if h else f'{m}m{s:02d}s'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='gru')
    ap.add_argument('--version', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    m, v = args.model, args.version
    cfg_path = ROOT / 'exploration' / 'version' / f'version_{m}_{v}.yaml'
    ckpt_dir = ROOT / 'exploration' / 'checkpoints' / f'checkpoint_{m}_{v}'
    hist_path = ckpt_dir / 'history.json'
    canon = ROOT / 'exploration' / 'logs' / f'log_{m}_{v}.log'
    out = Path(args.out) if args.out else canon.with_suffix('.reconstructed.log')

    if not cfg_path.exists():
        print(f'error: no config at {cfg_path}', file=sys.stderr)
        return 2
    if not hist_path.exists():
        print(f'error: no history.json at {hist_path}', file=sys.stderr)
        return 1

    cfg = yaml.safe_load(cfg_path.read_text())
    res = (cfg or {}).get('results') or {}
    tr = (cfg or {}).get('training') or {}
    model = (cfg or {}).get('model') or {}

    history = json.loads(hist_path.read_text())
    if isinstance(history, dict):
        history = history.get('history') or history.get('evals') or []

    # A run is only reconstructable once it has actually finished; otherwise the
    # tail of the evaluation series is simply missing, not merely unlogged.
    status = ((cfg or {}).get('identity') or {}).get('status')
    # Trust `status` only. `finished_utc` is written once at the end of a run, so
    # it can survive from an EARLIER run of the same version -- relaunching v021
    # after a smoke test left the smoke run's finished_utc in place, and an
    # `or finished_utc is not None` clause then waved a running job straight
    # through this guard. `status` is rewritten on every state change, so it is
    # the only field that reliably describes the CURRENT run.
    finished = status in ('finished', 'smoke')
    if not finished:
        print(f'error: run status is {status!r}, not finished. '
              f'The reconstruction would be silently incomplete. '
              f'Wait for the run to exit, then re-run.', file=sys.stderr)
        return 1

    total = (tr.get('max_steps') or tr.get('epochs') and None)
    lines: list[str] = []
    A = lines.append
    A('=' * 78)
    A('RECONSTRUCTED TRAINING LOG -- NOT THE ORIGINAL CANONICAL LOG')
    A('=' * 78)
    A('')
    A('This file was rebuilt from durable checkpoint state by')
    A('tools/rebuild_log.py after the canonical log was orphaned by an')
    A('accidental in-place file replacement while the run was live. The training')
    A('run itself did not fail and lost no results.')
    A('')
    A(f'model            : {m}')
    A(f'version          : {v}')
    A(f'built (UTC)      : {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    A(f'built by         : {platform.platform()}  python {platform.python_version()}')
    A(f'source config    : exploration/version/version_{m}_{v}.yaml')
    A(f'source history   : {hist_path.relative_to(ROOT)}')
    A(f'source checkpoints: {ckpt_dir.relative_to(ROOT)}/')
    A(f'canonical log    : {canon.relative_to(ROOT)}  (TRUNCATED -- left untouched '
      f'as incident evidence)')
    A('')
    A('RECOVERED: config, parameter count, every periodic evaluation with its')
    A('           step / WP / per-target WP / scored rows, which evaluation set')
    A('           each new best, and the final full-validation scores.')
    A('MISSING  : the per-step loss and grad_norm trace, the per-step learning')
    A('           rate, and the launch-time reproducibility header (command,')
    A('           host, package versions, git commit). Those existed only in the')
    A('           orphaned bytes and are NOT reproduced here.')
    A('')
    A('Treat every number below as recovered-from-checkpoints, not as a verbatim')
    A('record of what was printed during the run.')
    A('')
    A('=' * 78)
    A('CONFIG AS LOADED')
    A('=' * 78)
    for section in ('data', 'features', 'model', 'loss', 'optim', 'training', 'inference'):
        if cfg.get(section):
            A(f'{section}:')
            A(yaml.safe_dump(cfg[section], sort_keys=False, default_flow_style=False).rstrip())
    A('')
    A(f'param_count      : {model.get("param_count")}')
    A(f'seed             : {tr.get("seed")}')
    A(f'status           : {status}')
    A('')
    A('=' * 78)
    A('PERIODIC EVALUATIONS  (recovered from history.json)')
    A('=' * 78)
    A(f'{"step":>8}  {"wp":>10}  {"t0":>10}  {"t1":>10}  {"rows":>10}  {"wall":>10}  new best')
    A('-' * 78)
    best = None
    for e in history:
        wp = e.get('wp', e.get('val_wp'))
        if wp is None:
            continue
        is_best = best is None or wp > best
        if is_best:
            best = wp
        A(f'{e.get("step", 0):>8}  {wp:>10.6f}  '
          f'{(e.get("t0") or float("nan")):>10.6f}  '
          f'{(e.get("t1") or float("nan")):>10.6f}  '
          f'{(e.get("rows") or 0):>10,}  '
          f'{human(e.get("wall_s")):>10}  '
          f'{"yes" if is_best else ""}')
    A('-' * 78)
    A('')
    A('=' * 78)
    A('FINAL RESULTS  (recovered from the version yaml)')
    A('=' * 78)
    for key in ('best_step', 'best_val_wp', 'best_val_t0', 'best_val_t1',
                'full_valid_wp', 'train_hours', 'finished_utc'):
        val = res.get(key)
        A(f'{key:20s}: {f"{val:.6f}" if isinstance(val, float) else (val if val is not None else "--")}')
    A('')
    A('=' * 78)
    A('END OF RECONSTRUCTION')
    A('=' * 78)

    out.write_text('\n'.join(lines) + '\n')
    print(f'wrote {out.relative_to(ROOT)}  ({out.stat().st_size} bytes, '
          f'{len(history)} evaluations recovered)')
    if res.get('full_valid_wp') is None:
        print('note: full_valid_wp is still null in the yaml -- the final rescore had '
              'not written back yet when this was built. Re-run once it lands.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())