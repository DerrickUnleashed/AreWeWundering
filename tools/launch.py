#!/usr/bin/env python3
"""Launch a command in its OWN session so it survives the parent shell dying.

WHY THIS EXISTS
---------------
Launching a long training run with `nohup ... &` from a tool-invoked shell does
NOT work here: the shell is killed as a process *group*, and `nohup` only
ignores SIGHUP -- it does not escape the group. Every run started that way died
within a minute of the parent command being aborted (measured 2026-09-29, four
runs, all killed at ~30 s, no traceback in their logs).

`os.setsid()` puts the child in a new session with no controlling terminal and a
new process group, so neither a SIGHUP nor a group kill aimed at the parent can
reach it. That is the difference between a 5-hour run that survives and one that
does not.

Thread limits are set BEFORE the exec, because OpenMP/MKL read them at load time
and this is what keeps 4 concurrent runs from oversubscribing the 12 cores.

    python tools/launch.py --log-dir /tmp/runs -- python train.py --config c.yaml
    python tools/launch.py --threads 3 -- python train.py --config c.yaml
    python tools/launch.py --dry-run -- python train.py     # print, do not launch

Every launch appends one line to <log-dir>/launches.log with the pid, the command
and the UTC time, so there is a record of what was started even if the shell dies.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def launch(cmd, threads=None, log_dir=None, cwd=None, env_extra=None):
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if threads:
        for v in THREAD_VARS:
            env[v] = str(threads)
    env.update(env_extra or {})

    pid = os.fork()
    if pid > 0:
        # parent: record the launch and return immediately
        if log_dir:
            with open(Path(log_dir) / 'launches.log', 'a') as f:
                f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} pid={pid} '
                        f'threads={threads} cwd={cwd or os.getcwd()} cmd={" ".join(cmd)}\n')
        return pid

    # ---- child: become a session leader, then exec
    try:
        os.setsid()
    except OSError as e:                                  # already a group leader
        print(f'warning: setsid failed ({e}); the run may not survive a group kill',
              file=sys.stderr)
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    if log_dir:
        out = open(Path(log_dir) / 'child_stdout.log', 'ab', buffering=0)
        os.dup2(out.fileno(), 1)
        os.dup2(out.fileno(), 2)
    else:
        os.dup2(null, 1)
        os.dup2(null, 2)
    if cwd:
        os.chdir(cwd)
    os.execvpe(cmd[0], cmd, env)                           # never returns


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--threads', type=int, default=None,
                    help='pin BLAS/OpenMP to this many threads (set before exec)')
    ap.add_argument('--log-dir', default=None, help='where to record the launch and child stdio')
    ap.add_argument('--cwd', default=str(ROOT))
    ap.add_argument('--env', action='append', default=[], metavar='K=V')
    ap.add_argument('--dry-run', action='store_true', help='print the command and exit')
    ap.add_argument('command', nargs=argparse.REMAINDER,
                   help='the command to launch, after --')
    args = ap.parse_args()

    cmd = args.command
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        ap.error('give a command after --, e.g. -- python train.py --config c.yaml')
    if args.dry_run:
        print('would launch:', ' '.join(cmd))
        print(f'  threads={args.threads}  cwd={args.cwd}  log_dir={args.log_dir}')
        return 0
    env_extra = dict(kv.split('=', 1) for kv in args.env)
    pid = launch(cmd, args.threads, args.log_dir, args.cwd, env_extra)
    print(f'launched pid {pid}: {" ".join(cmd)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
