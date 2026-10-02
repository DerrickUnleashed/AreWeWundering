#!/bin/bash
# Wait for dozken's reset4k teacher replay to finish, build the averaged ensemble
# teacher, verify it, then launch v039 automatically.
#
# WHY A WATCHER RATHER THAN WAITING BY HAND
# ------------------------------------------
# v039 is the only one of the five experiments that is blocked on data. Its teacher is
# dozken's two-graph loss-diverse ensemble (0.6879), which we have as two .onnx files but
# not as labels -- and labels cannot be derived on the fly, because the replay must be
# strictly causal and stateful across all 212M train rows (tools/make_teacher_labels_onnx.py).
# That takes a couple of hours. Sitting on it manually means polling; a watcher means v039
# starts the moment the data is valid.
#
# WHY TWO FILES AND NOT A PRE-AVERAGED ONE
# -----------------------------------------
# teacher_dozken_reset2k.npy already exists and was produced by the same tool from the same
# graph, so re-replaying it would burn an hour to reproduce a file we trust. The ensemble is
# a RAW MEAN of the two members' outputs -- confirmed from his solution.py, which does
# `total * (1.0/len(graphs))` with no z-scoring. That matters: the metric clips at +/-2, and
# his notes record that z-scoring members before averaging inflated output std 1.55x and
# pushed 14.30% of scored rows past the clip.
#
# v039 does not need an averaged file at all -- TeacherStore already averages multiple label
# files on read, and its yaml lists both members. So this script's real job is to VERIFY the
# second member exists, is complete, and is finite, and only then to launch.

set -euo pipefail
ROOT=/Users/derricksamuel/Desktop/Projects/wnn_connectome_starterpack
PY="$ROOT/venv/bin/python"
LOG=/tmp/teacher_reset4k.log
OUT="$ROOT/cache/teacher/teacher_dozken_reset4k.npy"
EXPECT_ROWS=212140000

echo "[watcher] waiting for $OUT"
while true; do
    if [ -f "$OUT" ]; then
        # the tool writes the file only on success, but confirm the log agrees
        if grep -q "wrote cache/teacher/teacher_dozken_reset4k.npy" "$LOG" 2>/dev/null; then
            break
        fi
    fi
    if ! pgrep -f "make_teacher_labels_onnx.py" >/dev/null; then
        if [ ! -f "$OUT" ]; then
            echo "[watcher] replay process is gone and no output file -- ABORTING"
            exit 1
        fi
    fi
    sleep 60
done
echo "[watcher] replay finished, verifying"

"$PY" - <<'EOF'
import numpy as np
from pathlib import Path
R = Path('/Users/derricksamuel/Desktop/Projects/wnn_connectome_starterpack')
a = np.load(R/'cache/teacher/teacher_dozken_reset4k.npy', mmap_mode='r')
b = np.load(R/'cache/teacher/teacher_dozken_reset2k.npy', mmap_mode='r')
assert a.shape == (212140000, 2), f'reset4k shape {a.shape}'
assert b.shape == (212140000, 2), f'reset2k shape {b.shape}'
# sequence-leading rows are the canary: an interrupted replay leaves the tail unwritten
for nm, arr in (('reset4k', a), ('reset2k', b)):
    s = np.asarray(arr[[0, 20000, 5_000_000, 100_000_000, 212_100_000]])
    assert np.isfinite(s).all(), f'{nm}: non-finite in probe'
    assert np.abs(s).sum() > 0, f'{nm}: all-zero probe -- looks unwritten'
    print(f'  {nm}: finite, non-zero, {arr.shape}')
s = np.asarray(a[300:400])
print(f'  reset4k probe row mean {s.mean():+.6f}  absmax {np.abs(s).max():.4f}')
EOF

echo "[watcher] launching v039"
cd "$ROOT"
caffeinate -i nohup "$PY" exploration/models/train_gru_v039.py \
    --config exploration/version/version_gru_v039.yaml > /tmp/v039.out 2>&1 &
sleep 90
tail -3 exploration/logs/log_gru_v039.log 2>/dev/null | tr -s ' ' || tail -5 /tmp/v039.out
echo "[watcher] done"
