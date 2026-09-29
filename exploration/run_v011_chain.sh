#!/bin/bash
# Train the soft-mask predictor, label the training rows, then launch gru_v011.
#
# WHY A CHAIN: gru_v011 needs cache/train_scored.npy and the training script
# deliberately refuses to start without it. Chaining here means the mask is built
# and v011 starts without anyone watching, and the run is launched through
# tools/launch.py so it survives this shell being killed.
#
# The AUC gate is the point: if the mask is not predictable, the soft mask is a
# no-op and v011 would silently become a state-reset-only run -- i.e. a different
# experiment from the one its yaml claims. Better to not run it at all.
set -u
cd /Users/derricksamuel/Desktop/Projects/wnn_connectome_starterpack

LOGDIR=/tmp/wnn_mask
mkdir -p "$LOGDIR"
exec > "$LOGDIR/chain.log" 2>&1
echo "=== mask + v011 chain started $(date -u +%FT%TZ) ==="

# low thread count: six training runs already hold all 12 cores
THREADS=${MASK_THREADS:-2}

venv/bin/python tools/train_mask_model.py \
    --epochs 4 --steps 600 --lanes 32 --chunk 2000 --threads "$THREADS" \
    --mask-start 200 --mask-holdout 1600 \
    --out cache/train_scored.npy --ckpt cache/mask_model.pt
rc=$?
echo "=== train_mask_model.py exit $rc ==="
if [ $rc -ne 0 ] || [ ! -f cache/train_scored.npy ]; then
    echo "ABORT: mask training failed or produced no labels; not launching v011"
    exit 1
fi

AUC=$(venv/bin/python -c "import json;print(json.load(open('cache/train_scored.json'))['auc'])" 2>/dev/null || echo 0)
echo "=== mask holdout AUC: $AUC ==="
awk -v a="$AUC" 'BEGIN{exit !(a>=0.85)}' || {
    echo "ABORT: holdout AUC $AUC is below 0.85. The soft mask would be a no-op and"
    echo "       v011 would quietly become a state-reset-only run. Not launching."
    exit 2
}

venv/bin/python -c "
import numpy as np
p = np.load('cache/train_scored.npy', mmap_mode='r')
s = np.asarray(p[::1003], dtype=np.float32)
print(f'labels: {p.shape[0]:,} rows  mean P(scored) {s.mean():.4f}  '
      f'min {s.min():.4f}  max {s.max():.4f}  frac>0.5 {float((s>0.5).mean()):.4f}')
"

echo "=== launching gru_v011 ==="
venv/bin/python tools/launch.py --threads 2 --log-dir /tmp/wnn_runs -- \
    venv/bin/python exploration/models/train_gru_v011.py \
    --config exploration/version/version_gru_v011.yaml --device cpu
echo "=== chain finished $(date -u +%FT%TZ) ==="
