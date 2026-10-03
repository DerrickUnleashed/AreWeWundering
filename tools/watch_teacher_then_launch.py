#!/usr/bin/env python3
"""Wait for a teacher-label replay to actually FINISH, then launch a training run.

Why this exists
---------------
`tools/make_teacher_labels.py` pre-allocates its whole output `.npy` before it starts
writing. So the destination file exists, at full size, within seconds of launch, and
`np.load(..., mmap_mode='r')` opens it without complaint. Everything past the write
frontier reads as ZERO.

That makes the obvious gate -- "wait until the file exists" -- actively dangerous: it
passes almost immediately and the run then trains for hours against a teacher whose tail
is all zeros. No crash, no NaN, no warning. Just a silently broken run whose only symptom
is a loss curve that never learns.

So the gate here is the replay's own completion line, which asserts the full row count:

    wrote cache/teacher/teacher_x.npy  212,140,000 rows x 2  in NN.N min

and then a direct read of the array to confirm the tail is real data and not zeros.
Both checks must pass before anything is launched.

Usage
-----
    python tools/watch_teacher_then_launch.py \
        --replay-log /tmp/teacher_v037.log \
        --teacher    cache/teacher/teacher_member_v037.npy \
        --script     exploration/models/train_gru_v042.py \
        --config     exploration/version/version_gru_v042.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[watch] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def replay_finished(replay_log: Path) -> tuple[bool, int | None]:
    """True once the log contains the completion line. Returns (done, rows_found)."""
    if not replay_log.exists():
        return False, None
    try:
        text = replay_log.read_text(errors="replace")
    except OSError:
        return False, None
    m = re.search(r"wrote\s+\S*teacher\S*\.npy\s+([\d,]+)\s+rows", text)
    if not m:
        return False, None
    return True, int(m.group(1).replace(",", ""))


def replay_alive(pattern: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return r.returncode == 0


def verify_teacher(path: Path, expect_rows: int) -> None:
    """Hard gate. Raises if the array is the wrong size or its tail is still zeros."""
    import numpy as np

    a = np.load(path, mmap_mode="r")
    if a.shape[0] != expect_rows:
        raise SystemExit(f"[watch] FAIL {path} has {a.shape[0]:,} rows, expected {expect_rows:,}")
    if a.shape[1] != 2:
        raise SystemExit(f"[watch] FAIL {path} second dim is {a.shape[1]}, expected 2")

    # The whole point: confirm the far end of the file was really written.
    tail = np.asarray(a[-100_000:], dtype=np.float32)
    zeros = int((tail == 0).all(axis=1).sum())
    if zeros > 0:
        raise SystemExit(
            f"[watch] FAIL tail of {path} still has {zeros:,}/100,000 all-zero rows. "
            "The .npy is pre-allocated, so this file exists before it is filled -- "
            "this is exactly the failure this watcher exists to prevent."
        )
    log(f"verified {path.name}: {a.shape[0]:,} rows x {a.shape[1]}, tail is real data "
        f"(range {tail.min():.4f}..{tail.max():.4f})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--replay-log", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--replay-pattern", default="make_teacher_labels.py",
                    help="pgrep pattern used to notice the replay died")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--timeout-hours", type=float, default=6.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    replay_log = ROOT / a.replay_log
    teacher = ROOT / a.teacher
    script = ROOT / a.script
    config = ROOT / a.config
    deadline = time.time() + a.timeout_hours * 3600

    log(f"watching {a.replay_log} for {a.teacher} -> will launch {a.script}")

    while True:
        done, rows = replay_finished(replay_log)
        if done and rows:
            log(f"replay reported completion: {rows:,} rows")
            break
        if time.time() > deadline:
            log(f"FAIL timed out after {a.timeout_hours} h without a completion line")
            return 1
        if not replay_alive(a.replay_pattern):
            # The process may have exited between polls; give the log one more read.
            time.sleep(5)
            done, rows = replay_finished(replay_log)
            if not (done and rows):
                log("FAIL replay process is gone and no completion line was ever written. "
                    "Refusing to launch on a partial teacher.")
                return 1
        time.sleep(a.poll)

    verify_teacher(teacher, rows)

    if a.dry_run:
        log("dry run: teacher verified, NOT launching")
        return 0

    log(f"launching {a.script}")
    logf = open(ROOT / "exploration/logs" / f"watch_{config.stem.split('_')[-1]}.log", "a")
    proc = subprocess.Popen(
        [sys.executable, str(script), "--config", str(config)],
        cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"launched, pid {proc.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())