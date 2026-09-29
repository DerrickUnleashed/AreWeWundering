# v011 — dozken/wunder's recipe: soft mask + periodic state reset

**Status: implemented, validated, NOT launched** (per instruction, 2026-09-29).

## Where the idea came from

`github.com/dozken/wunder`, `challenge-3/`. He reports public **0.6569** (rank #11
when written) with a GRU 192×2 recipe. His README is unusually complete, and two of
its findings are things we had never tried.

**His stated recipe:** GRU 192×2 + input projection, first-difference inputs,
sequence-level weighted-Pearson loss, soft-masked rows, periodic state reset,
AdamW + cosine, EMA/SWA, trained on train+valid.

| Technique | Ours before v011 | His |
|---|---|---|
| GRU 2×192 | v004 ✓ | ✓ |
| first differences | v007 (running) | ✓ |
| sequence-level WP loss | ✓ | ✓ |
| **soft mask on scored rows** | **no** | ✓ (+0.036) |
| **periodic state reset** | **no** | ✓ (his best run) |
| EMA | v003 ✓ | ✓ + SWA |
| input projection | v004 ✓ | ✓ (proj 64 costs ~0.014) |
| train+valid | no | ✓ (final model only) |

## The two mechanisms

### 1. Soft mask — scored rows are a different regime

Only ~10 % of required rows are scored, and they are **not a random sample**: the
same model scores **0.58 WP on them and 0.30 on all rows**. His oracle test —
training with the loss on all rows vs on `is_scored` rows — measured **0.569 vs
0.605**, a +0.036 swing. The metric only ever sees the scored rows, so the training
loss should concentrate there.

`tools/train_mask_model.py` trains a 1×64 GRU to predict `is_scored` from the 112
features (his AUC is 0.9416) and writes `P(scored)` for every training row to
`cache/train_scored.npy` (float16, 424 MB). The loss weight becomes
`|clip(y)| · (floor + (1−floor) · P(scored))` with `floor = 0.1` so no row is ever
dropped outright.

**Legal:** the FAQ allows "your own validation splits and masks using the supplied
labelled data"; `valid.parquet` ships `is_scored` and no external data is involved.
**Training-only:** nothing is needed at inference, so it adds no latency and no
bytes to the submission.

**Split discipline (the part that matters).** The mask is fitted on validation
sequences **200–1600** and early-stopped on **1600–1873**. Sequences **0–199** are
never touched, because that is exactly what every version in this repo evaluates
on. A mask fitted on rows we also score would leak. The script prints this
boundary on every run.

### 2. Periodic state reset — the decay is the recurrent state

Per-quarter WP falls 0.70 → 0.638 across a sequence. He ruled out the obvious
explanations: scored density, target std and the strongest single-feature |r| are
all flat across quarters, and oracle per-quarter recalibration moves WP by ±0.0001.
It is the **state**: hard-resetting to zero at row 10,000 lifts Q3 from 0.6536 to
0.6881 — a *zero* state beats the state's own 10,000 rows.

Resetting the state every 2,000 rows at *inference only* fails, because the two
segments then sit at different prediction levels and the metric is one Pearson over
the whole sequence. The fix is to reset during **training as well**, so the model
learns to emit a consistent level from a fresh state at any position.

He notes a reset-trained model scores **~0.07 low** if the evaluation path ignores
the period. That is why `state_reset` is wired through *four* places in the
template, and verified to reset at identical positions in training and inference:

```
train resets at     : [0, 4000, 8000, 12000, 16000]
inference resets at : [0, 4000, 8000, 12000, 16000]   IDENTICAL
```

The period is written into `feature_stats.npz` at export, so the packaged
`solution.py` resets in-regime without any config file.

`state_reset` must divide 20,000, must be a multiple of `chunk_len`, and must be
**strictly greater** than `chunk_len` — a reset equal to `chunk_len` would silently
discard all carried state and leave a model with no memory beyond one chunk. The
template raises on all three.

## Why 4000, not 2000

He uses chunk 1000 with reset 2000 — two chunks of carry. Ours is
`chunk_len: 2000`, so the equivalent that preserves one chunk of carry is
`state_reset: 4000`. With `state_reset: 2000` every chunk would start from zero.

## This run is confounded, on purpose

Two levers change together. If v011 wins we will not know which half did it; if it
loses we will not know which half broke it. That is the accepted cost of
reproducing his recipe in one run, and it is recorded in the yaml
`hypothesis` field. **v012 must be a single-variable follow-up** — soft mask only,
or reset only — to attribute any gain.

## Validated before launch

- transform `.step()` vs `.batch()` parity **0.0** (tol 1e-6)
- loss differs with and without the soft mask (mask is active)
- ONNX export verified against PyTorch to **3.9e-07**
- `state_reset=4000` present in the exported `feature_stats.npz`
- `train_mask_model.py` runs end to end: loss falls, AUC computed, checkpoint and
  labels written, and it refuses to be used if holdout AUC < 0.85

## Still to do before v011 can run

1. Train the mask: `venv/bin/python tools/train_mask_model.py --epochs 4`
   (fits 200–1600, early-stops on 1600–1873, writes `cache/train_scored.npy`).
   Expect ~30–60 min plus ~20 min to label 212 M training rows.
2. Check the reported AUC is comfortably above 0.85.
3. `venv/bin/python tools/bench_latency.py --script exploration/models/train_gru_v011.py
   --config exploration/version/version_gru_v011.yaml --untrained` **on an idle
   machine** — the state reset must not change inference latency materially.
4. Launch.

## One thing NOT copied

His **2-graph ensemble** scored best on his held-out (0.6879 vs 0.6864) but was
**killed at 69m59s — a timeout**. Ensembling is off the table; the only route to
that gain inside one graph is distillation, which is a later idea.
