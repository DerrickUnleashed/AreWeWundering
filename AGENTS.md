# AGENTS.md — Alfa Connectome (Wunder Fund) working agreement

This file is the project memory for any coding agent or human working in this repository. Read it fully before doing anything. If it conflicts with `docs/` from the organisers, the organisers' docs win and this file must be corrected.

## 1. Project snapshot

- **Goal:** predict `t0`, `t1` (future price-movement indicators of instrument `i0`) for every row with `need_prediction == True`, from 112 streaming features per row (i0 book/trade group, i1 book/trade group, a0–a7).
- **Metric:** Global Weighted Pearson (WP), computed per target over all rows where `need_prediction AND is_scored`, weights = `abs(clip(target, -2, 2))`, predictions and targets clipped to [-2, 2], final = mean over the two targets. Reference code: `METRIC.md` + `utils.weighted_pearson` / `utils.GlobalAccumulator` (copied verbatim into `tools/metric_ref.py`).
- **Data:** train 10,607 seq / 212,140,000 rows; valid 1,873 seq / 37,460,000 rows; every sequence 20,000 rows; warm-up steps 0–98; predictions required at steps 99–19,999; reset state per sequence.
- **Baseline:** vanilla stateful GRU (ONNX, 1 thread) = 0.617052 WP on full validation. Graph verified: `nn.GRU(112 -> 128, 2 layers) -> Linear(128 -> 2)`, **no input projection**, 192,258 parameters, ONNX opset 17, ir_version 8, producer `pytorch 2.5.1`, inputs `features[1,1,112] / hidden_0[1,1,128] / hidden_1[1,1,128]`, output `prediction[1,1,2]`. Its graph contains **no feature normalisation** — the organisers ship pre-standardised features.
- **Deadline:** submissions close **2026-11-15 23:59 UTC** (verified: `docs/timeline.md`; start SEP 11 2026, winners announced DEC 01 2026).
- **Scorer limits:** 1 vCPU, CPU only, 16 GB RAM, 60 min for the whole test set, ZIP ≤ 20 MB, Python 3.11 (`python:3.11-slim-bookworm`), no internet.
- **Test-set size (verified, `docs/faq.md`):** **1,970 sequences = 39,400,000 rows.** Test targets and the scoring mask are hidden.
- **Derived per-row budget:** 3,600 s / 39.4 M rows = **91.4 µs per row, all-in**. Engineering target: **≤ 68.5 µs/row (75 % of budget) on the dev machine** (headroom for a slower/noisier grader). Measured reference point: the official baseline runs at **27.9 µs/row mean (p50 26.9, p99 45.7)** on this M2 Max — i.e. roughly 2.5x of headroom before the budget is consumed.
- **Rules digest (verified: `docs/rules.md`, `docs/faq.md`, `docs/submission_guide.md`):**
  - **No external data.** "Your solution must only use the training data we provide" / "All solutions must be trained using only the provided `train.parquet` and `valid.parquet`". Retraining on train+valid is explicitly in scope.
  - **Pre-trained models are allowed** if publicly available and containing **no external market data**. Distillation from a self-trained teacher is therefore not an external-data problem; a pretrained general model is fine; a pretrained *market* model is not.
  - **Custom validation splits are allowed** ("You may define your own validation splits and masks using the supplied labeled data"). The hidden test mask is unaffected.
  - **5 submissions per day** maximum.
  - Teams allowed in principle, but "Currently there's no way to team-up in the interface" → in practice an individual entry.
  - Two leaderboards: Public (a sample of the test data) and Private (separate hidden dataset). **Final ranking is decided solely by the Private Leaderboard**; near the end you are asked to **select previously submitted and scored solutions to be re-scored on a "Final Dataset"**. Tie-break: earlier submission wins.
  - **Determinism required:** "Repeated runs on the same input must produce identical predictions."
  - **Prize eligibility:** top 8 must pass verification — full inference code + weights, full training code with fixed seeds, a 2–8 page technical report, and a 30–60 min code-review call. Pool: $13,600 USDT, 1st = $5,000.
  - **ZIP layout:** `solution.py` at the archive root plus model artifacts, loaded via paths relative to `solution.py`. Anything else may live in subfolders.
  - **Guaranteed packages in the scorer:** exactly `requirements.txt` = `numpy==2.2.6`, `onnxruntime==1.23.2`, `pyarrow==19.0.1`. The Dockerfile additionally references a PyTorch CPU extra-index and installs `orbax-checkpoint`, but `pip install -r requirements.txt` is the binding command, so **torch / pyyaml / pandas / scipy / matplotlib / tqdm are NOT guaranteed to exist at scoring time.** Inference code must import only `numpy` + `onnxruntime` (+ stdlib). Bake every constant into `.npz`/`.onnx` artifacts, never read a yaml at inference.

## 2. Prime directives (non-negotiable)

1. **Causality.** Inference sees one row at a time. No lookahead, no bidirectional layers, no normalisation statistics computed from a full sequence or from test data. Any per-sequence statistic must be a running (causal) statistic.
2. **Train/serve parity.** The feature transform used in training and the one used in `solution.py` are the same code path (see §7). A parity test must pass before any submission.
3. **Latency first.** Before training any new architecture, benchmark its inference step (`tools/bench_latency.py`). Never spend GPU hours on a model that cannot meet the budget.
4. **Metric fidelity.** Every reported score comes from the reference metric code with the correct mask. Never report a Pearson computed some other way as "WP".
5. **Immutability.** Never overwrite an existing version's script, yaml, log, checkpoint folder or submission folder. A change of any kind = a new version. Never delete anything unless the user explicitly asks.
6. **No silent uploads. Agents never push.** Two separate prohibitions:
   (a) agents never upload anything to the competition site — the user does that;
   (b) **agents never run `git push` (or `git push --force*`, or `git lfs push`), under any circumstances, even when the user asks about pushing.** The user pushes. Committing locally is fine and expected; the moment a change would cross the network, stop and report. Asking "should I push?" is still not enough — the answer is always no; instead tell the user what is ready and let them run it. **A mistake already made on 2026-09-29:** four pushes and one history rewrite were performed after being asked to "help push" the datasets, and the first push had already been authorised for that one task. Do not generalise from one approval to later, unasked-for pushes.
7. **No data in RAM.** Stream or memory-map; never `pd.read_parquet` the train file whole.
8. **Reproducibility.** Every run records seed, config (yaml), git commit if available, package versions, hardware, and the exact command in its log header.
9. **Don't touch organiser files:** `README.md`, `METRIC.md`, `utils.py`, `requirements.txt`, `baseline/`, `datasets/`, `docs/`.

## 3. Repository map

```text
.
├── AGENTS.md                      # this file
├── METRIC.md README.md utils.py requirements.txt   # organiser files (read-only)
├── baseline/  datasets/  docs/  venv/              # organiser files / environment (read-only)
├── cache/                         # gitignored: sharded memmap data, normalisation stats (regenerable)
├── tools/                         # shared, versioned-by-git utilities (not experiments)
│   ├── metric_ref.py              # verbatim METRIC.md functions + fast vectorised/torch equivalents
│   ├── make_shards.py             # parquet -> memmap shards + normalisation stats
│   ├── new_experiment.py          # scaffolds a new version (script + yaml) from template or parent
│   ├── bench_latency.py           # µs/row benchmark of a solution.py / ONNX model, 1 thread
│   ├── score_solution.py          # runs a solution.py through the real callback loop on N valid sequences
│   ├── package_submission.py      # builds submissions/submission_<model>_<version>/
│   ├── summarize.py               # prints a leaderboard table of all version yamls
│   └── templates/train_template.py
├── exploration/
│   ├── models/                    # train_<model>_<version>.py
│   ├── version/                   # version_<model>_<version>.yaml
│   ├── checkpoints/               # checkpoint_<model>_<version>/   (one folder per version)
│   ├── visualisation/             # EDA scripts (only when they answer a decision)
│   │   └── images/                # EDA figures
│   └── logs/                      # log_<model>_<version>.log
└── submissions/                   # submission_<model>_<version>/
```

## 4. Naming and versioning

`<model>` = lowercase slug of the architecture family, `[a-z0-9]+` (e.g. `gru`, `mingru`, `lru`, `ssm`). `<version>` = `v` + zero-padded 3-digit integer (`v001`, `v002`, …); `v000` is reserved for the framework smoke test. The pair `<model>_<version>` is the experiment key, identical across all artifacts:

| Artifact | Path |
|---|---|
| Training script | `exploration/models/train_<model>_<version>.py` |
| Parameters / description | `exploration/version/version_<model>_<version>.yaml` |
| Checkpoints (one folder) | `exploration/checkpoints/checkpoint_<model>_<version>/` |
| Exact training output | `exploration/logs/log_<model>_<version>.log` |
| Submission bundle | `submissions/submission_<model>_<version>/` |

A new version is created only through `python tools/new_experiment.py --model <m> --version <v> [--from <parent_model>_<parent_version>]`. The parent relationship is recorded in the yaml. One hypothesis per version where possible; confounded changes make results uninterpretable.

## 5. How to run an experiment

```bash
# Scaffold:
python tools/new_experiment.py --model gru --version v001 --from gru_v000

# Edit the yaml (single source of truth for hyper-parameters) and, if the change is structural, the new script.

# Latency gate:
python tools/bench_latency.py --script exploration/models/train_gru_v001.py \
    --config exploration/version/version_gru_v001.yaml --untrained
# Must be within budget.

# Smoke:
python exploration/models/train_gru_v001.py --config exploration/version/version_gru_v001.yaml --smoke
# (tiny subset, a few hundred steps; must produce a log, a checkpoint folder and a yaml results block.)

# Full run (long-running, detached):
nohup python exploration/models/train_gru_v001.py --config exploration/version/version_gru_v001.yaml > /dev/null 2>&1 &
# the script writes its own log; do not rely on shell redirection for the canonical log.

# Post-run:
python tools/summarize.py
# and compare with the parent on identical rows.

# Promote (only if it beats the current best by more than the noise floor and passes the gate in §9):
python tools/package_submission.py --model gru --version v001
```

## 6. The version yaml (schema)

`exploration/version/version_<model>_<version>.yaml` is read by the training script and is the only place hyper-parameters live. Sections (all present, unused ones may be null):

```yaml
schema_version: 1
identity: {model: gru, version: v001, parent: gru_v000, status: planned}   # planned|smoke|running|finished|failed|abandoned
description: >            # what this model IS, in plain words
hypothesis: >             # what we expect to learn / improve, and why
change_vs_parent: >       # exact delta vs parent
data: {train_shards: cache/train, valid_shards: cache/valid, seq_len: 20000, warmup_steps: 99, target_clip: 2.0, train_sequences: null}
features: {normalisation: robust_zscore, signed_log_volumes: true, first_differences: [..], cross_instrument: [..], step_feature: false, clip_sigma: 8}
model: {arch: gru, input_proj: 128, hidden: 192, layers: 2, dropout: 0.0, layernorm: false, head: linear, param_count: null}
loss: {type: seq_weighted_pearson, mask: soft, aux_mse_weight: 0.0, target_clip: 2.0}
optim: {name: adamw, lr: 2.0e-3, weight_decay: 1.0e-2, betas: [0.9, 0.99], schedule: cosine, warmup_steps: 500, grad_clip: 1.0, ema_decay: 0.999, precision: fp32}
training: {chunk_len: 2048, burn_in: 128, batch_sequences: 64, max_steps: null, epochs: 3, seed: 1337, eval_every_steps: 2000, eval_subset_sequences: null, device: auto, stateful: true}
inference: {export: onnx, opset: 17, ort_threads: 1, target_us_per_row: 68.5}
results: {best_step: null, best_val_wp: null, best_val_t0: null, best_val_t1: null, full_valid_wp: null, latency_us_p50: null, latency_us_p99: null, train_hours: null, finished_utc: null}
artifacts: {log: ..., checkpoint_dir: ..., submission_dir: null}
```

Only `identity.status`, `model.param_count`, `results.*` and `artifacts.*` may be written after creation; everything else is frozen once a run starts. Two writers, and no others: the **training script** writes `identity.status`, `model.param_count`, `results.*` (except latency) and `artifacts.log`/`artifacts.checkpoint_dir`; **tools/package_submission.py** writes `results.latency_us_p50`, `results.latency_us_p99` and `artifacts.submission_dir`. `results.full_valid_wp` is only ever set from a rescore covering **every** validation sequence; a subset number is logged but leaves the field null so it can never be mistaken for the official one.

`training.chunk_len` **must divide 20,000**. Each stateful lane consumes one sequence chunk by chunk with its hidden state carried, so a chunk that does not end on a sequence boundary would have to be cut mid-sequence and the state would silently wrap into the next sequence. The template raises on a non-divisor; valid choices include 100, 125, 200, 250, 400, 500, 625, 1000, 2000, 4000 (2000 = 10 chunks, the default).

## 7. Anatomy of a training script

Each `train_<model>_<version>.py` is self-contained (one file = one reproducible experiment) and is organised in fixed, marked sections so the packager can derive `solution.py` from it mechanically:

```python
# ===== SECTION: TRAINING IMPORTS & UTILITIES ===== (torch, yaml, tqdm, Tee logger…) — training only
# ===== BEGIN SOLUTION SECTION ===== …
# ===== END SOLUTION SECTION =====     — code that is copied verbatim into solution.py:
    # constants (MODEL_NAME, VERSION), the numpy FeatureTransform (streaming .step(x) and
    # vectorised .batch(X) that must agree within 1e-6), the PredictionModel class
    # (ONNX Runtime or hand-rolled numpy inference, preallocated buffers, state reset on new seq_ix).
# ===== SECTION: MODEL (PyTorch) ===== — the trainable module; exports to the inference format.
# ===== SECTION: DATA ===== — memmap shard reader; stateful streaming batches (each batch lane walks
    # through one sequence chunk after chunk, hidden state carried, reset when a lane starts a new sequence).
# ===== SECTION: LOSS & METRIC ===== — sequence-level soft-masked weighted Pearson; evaluation via tools/metric_ref.py.
# ===== SECTION: TRAIN LOOP ===== — AMP/EMA/grad-clip, periodic evaluation, checkpoint on every new best (§8), yaml results write-back.
# ===== SECTION: MAIN ===== — CLI (--config, --smoke, --resume, --device, --export-only).
```

`solution.py` = the file with all non-solution sections removed and nothing else altered, so the code that ran in training is the code that runs in inference. **The solution section may not import torch, yaml, pyarrow, tqdm or pandas** — only `numpy`, `onnxruntime` and stdlib.

## 8. Checkpoint and log policy

**Checkpoints:** `exploration/checkpoints/checkpoint_<model>_<version>/` holds `best_<NNN>_step<S>_val<WP>.pt` for every new best validation score during training (never overwritten, never pruned automatically), `last.pt` (resume state: model + optimiser + EMA + RNG + step), and `history.json` (step, val WP per target, wall-clock, which file). Each `.pt` contains model weights (and EMA weights), the config, feature-normalisation statistics, and metric values.

**Logs:** `exploration/logs/log_<model>_<version>.log` contains the exact stdout+stderr of the run (tee both streams, line-buffered, include tracebacks). First lines are a reproducibility header (command, timestamp, host, Python/torch/numpy/onnxruntime versions, git commit or n/a, seed, yaml contents). A resumed run appends after a clearly marked separator line. A new (non-resume) run refuses to start if the log already exists.

## 9. Submission bundle and promotion gate

`submissions/submission_<model>_<version>/` contains:

- `metadata.json` — model params (the yaml content), scores (t0, t1, mean; rows and mask used), checkpoint used (file, step, sha256), latency (µs/row p50/p99 and projected total minutes), package versions, sha256 of the train script and of `solution.py`, timestamp, git commit, notes.
- the trained model in inference format (ONNX and/or `.npz`; plus the best `.pt` for provenance if it fits comfortably),
- `solution.py` (derived from the training script per §7),
- `submission_<model>_<version>.zip` with `solution.py` at the ZIP root and all runtime model files (≤ 20 MB).

**Promotion gate** — all must pass or the bundle is marked NOT-READY in `metadata.json`:

1. **Contract test:** correct return types, `None` in warm-up, finite `[t0, t1]` otherwise, state reset on `seq_ix` change, deterministic across two runs, no NaN/inf on any validation sequence tested.
2. **Parity:** streaming `solution.py` output vs the offline vectorised evaluation agree (max abs diff ≤ 1e-4 on ≥ 20 sequences).
3. **Score:** full-validation WP from `tools/score_solution.py` (real callback loop, or a documented sampled protocol if the full loop is too slow) matches the training-time number within tolerance.
4. **Latency:** single-thread µs/row × 39.4 M rows ≤ 75 % of 60 min on the dev machine.
5. Size ≤ 20 MB, imports limited to packages guaranteed in the scoring environment, Python-3.11-compatible syntax and libraries.

## 10. Inference engineering rules

1. One thread. Configure ONNX Runtime with `intra_op_num_threads=1`, `inter_op_num_threads=1`, sequential execution, full graph optimisation; benchmark spinning on/off. The baseline additionally sets `allow_spinning=0` and `use_per_session_threads=True` — copy that.
2. Preallocate every buffer; use `io_binding` or in-place `np.copyto`; no `np.concatenate`, no list/dict creation per call, no Python-level loops over features.
3. Fold every linear feature operation (scaling, centring, linear combinations) into the first layer's weights at export time. Only genuinely non-linear or stateful features (signed log, first differences, EMAs) are computed per row.
4. **Prefer few large fused operators over many small ones.** A 2-layer ONNX GRU (2 ops) beats a hand-built cell graph unless proven otherwise; also benchmark a hand-rolled numpy cell with pre-transposed fused weights. Measure, don't assume. **Measured 2026-09-28:** exporting a single `nn.GRU(num_layers=2)` (state handed over as one packed `(2,1,H)` tensor) makes the exporter emit Split/Concat/Slice/Gather plumbing — **18–50 graph nodes instead of the baseline's 10** — and cost **+8 µs/row at batch 1** (34.8 vs 26.9 µs/row p50, 30 % of the budget). The fix is a `ModuleList` of single-layer `nn.GRU`s, so each ONNX `GRU` op receives its `(1,1,H)` state untouched, exactly as `baseline.onnx` does. Always check the exported node count and the µs/row, never assume the exporter gives you the baseline's graph.
5. All rows go through `predict` — including warm-up and unscored rows — so the hidden state must be updated on every call; only the output is skipped in warm-up.
6. **Per-row Python overhead is a real cost.** At batch 1 the callback is ~30 µs of which only ~25 µs is ONNX Runtime. Measured, avoidable costs: `np.isfinite(p).all()` allocates a bool array every call (~1 µs) — use scalar comparisons instead, which are equally exact; `np.clip` without `out=` allocates; a non-identity `FeatureTransform` costs 1.5 µs/row, so give it an identity fast path when no block is enabled (the organiser's features are already standardised, so identity is the common case). `int()` on the `seq_ix`/`step` fields and the `previous_step` continuity check are deliberate: they turn a silently mis-ordered stream into a loud error.
7. Ensembling multiplies latency. Prefer weight-averaging (EMA/SWA), distillation, or a single wider model to N separate models, unless the benchmark shows spare budget.
8. **Export with `dynamo=False`.** torch >= 2.9 defaults `torch.onnx.export` to the dynamo exporter, which needs the extra `onnxscript` package — not in `requirements.txt`, so it is not available in the scoring container and the local environment would diverge from the scorer. The TorchScript exporter (`dynamo=False`) has no extra dependency and is what produced `baseline.onnx`.

## 11. Evaluation protocol

1. The only score that counts is global WP from `tools/metric_ref.py` on `need_prediction AND is_scored` rows.
2. Fast offline evaluation (batched, whole sequences, GPU/CPU) is used during development; the slow row-by-row callback loop is used for gate checks and for a fixed parity subset.
3. Compare candidates on identical validation rows, judge at the end of the LR schedule, and treat short proxy runs as unable to resolve differences below ~0.01 <<VERIFY empirically: run the parent with 3 seeds and record the seed-to-seed standard deviation here>>.
4. Record the noise floor once measured: seed std = <<...>>. A change is "real" only if it beats the parent by > 2x that std on identical rows.
5. Expect a public-vs-local gap (a third party reports the public score ~0.04 below their held-out estimate). Do not over-fit to any single number; prefer changes that are consistent across seeds and across validation halves.

## 12. Strategy and roadmap

**Guiding thesis:** this is a latency-constrained stateful sequence-regression problem with a tail-weighted global correlation metric. The gains come from (a) matching the training objective to the metric, (b) training on the real inference regime (long, stateful, 20 k-step sequences), (c) good input representation, and (d) squeezing the best model into ~91 µs/row. Roughly in this order of expected value per unit of effort:

**Phase 0 — Harness (days 1–3).** Sharded data; verbatim metric; fast offline evaluator; reproduce the baseline's 0.617052 (offline evaluator on the baseline ONNX should match the official number to ~1e-5); latency harness. Nothing else starts until this matches.

**Phase 1 — Minimal EDA (day 3–4; only what changes a decision).** Questions: feature distributions and heavy tails (per-column quantiles), which columns are near-constant or redundant, whether volumes need signed-log, whether price-like columns are already mid-relative, the `is_scored` pattern in validation (fraction, position in sequence, contiguous blocks vs random), target distributions and autocorrelation, correlation between t0 and t1, how concentrated the metric weight is (share of Σ|target| held by the top 1 % of rows), train-vs-valid drift, baseline error broken down by step position and by |target| decile, and information in i1/a0–a7. Figures go to `exploration/visualisation/images/`; scripts to `exploration/visualisation/`. Skip any plot that does not change a decision.

**Phase 2 — Objective (week 1).** Replace any per-chunk loss with a sequence-level / large-batch weighted Pearson that mirrors the metric (clipped targets, weights |clip(t)|, both targets averaged). Ablate: hard mask vs soft mask on unscored rows (the hidden mask means every predicted row matters, but the scored subset may have structure), auxiliary weighted-MSE term (0, 0.05, 0.2), Huber-style clipping of the aux term. Keep the architecture fixed at the baseline size while doing this so the effect is isolated.

**Phase 3 — Training regime (week 1–2).** Stateful truncated BPTT: each batch lane streams through a whole 20 k-step sequence in chunks (1–4 k steps), hidden state carried across chunks, burn-in steps excluded from the loss, reset at sequence boundaries. This removes the train/inference mismatch of training on short independent windows. Then: AdamW + cosine or one-cycle, gradient clipping, EMA weights (free at inference), modest input noise or dropout, long schedules (the third-party note suggests short runs are misleading). Use TF32/bf16 for matmuls if a GPU exists but keep recurrent state in fp32.

**Phase 4 — Inputs (week 2).** Robust z-score with train-only statistics; explicit first differences of the top-of-book price/volume features and of trade features; cross-instrument features (i0 vs i1 spreads/imbalances); optionally a few multi-timescale EMAs of key signals. Each addition is checked for per-row cost — linear ones are folded into layer 1 and cost nothing. **Note: the organisers already ship standardised features (the baseline graph applies no normalisation), so `signed_log_volumes` and `robust_zscore` are hypotheses to test, not givens.**

**Phase 5 — Architecture under budget (weeks 2–4).** Sweep GRU width/depth against measured µs/row: (1x256, 2x128 [= baseline], 2x192, 3x128…), input projection size, residual skip around the recurrence, output MLP head. Then test cheaper-to-serve recurrences that may buy width at equal latency: minGRU/minLSTM (parallel-scan trainable, simple step), diagonal linear RNNs (LRU/S4D-style) with a small nonlinear readout, and a hybrid (linear recurrent memory + small GRU). Discard anything whose operator count blows the budget. Multi-task shared trunk with separate t0/t1 heads by default; test separate small heads.

**Phase 6 — Capacity beyond the budget (weeks 4–6).** Train a teacher that ignores the latency limit (wider/deeper GRU, or a causal Transformer/SSM) and distil into the budget-sized student using labels + teacher outputs. Note: rules forbid *external data* but permit self-trained teachers; pretrained general models are permitted if they contain no external market data. Also: EMA/SWA over the last checkpoints; small seed ensembles only if the benchmark shows headroom; INT8 only if measured faster at batch 1 (often it is not).

**Phase 7 — Endgame (weeks 6–7).** Freeze the recipe. Optionally retrain on train + validation (explicitly permitted by the FAQ) with a fixed step count decided from the train-only run — keep the train-only model as fallback, because validation is then no longer a clean estimate. Note the final ranking comes from a *re-scored previously submitted solution*, so the thing you upload must already be your best artifact. Run the full promotion gate. Keep the final 3–4 days for gate failures, not for new ideas.

Budget of attempts: prefer many cheap, single-variable comparisons early; reserve GPU time for a small number of long runs once the recipe is stable.

## 13. Pitfalls checklist (review before every promotion)

- Any statistic (mean/std/quantile) computed over test-time or full-sequence data?
- Bidirectional layer, future-shifted feature, or a target-derived column leaking into inputs?
- Hidden state reset on every new `seq_ix` and at the start of every training lane sequence?
- `chunk_len` still divides 20,000 (a non-divisor silently carries a lane's state across a sequence boundary)?
- Metric mask applied identically (`need_prediction AND is_scored`) and clipping order matches METRIC.md?
- Prediction dtype/shape exactly what the scorer expects (finite, length 2, `[t0, t1]` order)?
- float64 creeping into inference (slower, and can change results)?
- Python 3.13 syntax/library used locally but scorer is 3.11?
- Import of torch / yaml / pyarrow / pandas in `solution.py` — none of these are guaranteed in the scorer image?
- ONNX opset / onnxruntime version compatible with the scorer's onnxruntime?
- Latency measured with one thread and cold caches, on realistic data (not zeros)?
- ZIP has `solution.py` at the root, ≤ 20 MB, no absolute paths, model files loaded relative to `__file__`?

## 14. Environment facts (measured 2026-09-28)

**Machine.** macOS 27.0.0 (Darwin Kernel 27.0.0, arm64), Apple M2 Max, 12 physical cores, 64 GiB RAM (68719476736 B), 280 GiB free of 926 GiB on `/dev/disk3s5` (project volume). **No CUDA GPU; `torch.backends.mps.is_available() == True`.** This is a development machine — the grader is 1 x86 Linux vCPU, plausibly 2–3x slower per core, so the 75 % rule (§1) is the gate.

**Python.** `venv/bin/python` = **3.13.3** and is the *training* environment (torch 2.14, PyYAML, onnx, matplotlib, tqdm). The scorer is **3.11**. `venv311/` is a local, faithful reproduction of the scorer: Python **3.11.9** with **exactly** `pip install -r requirements.txt` and nothing else (numpy 2.2.6, onnxruntime 1.23.2, pyarrow 19.0.1 + their deps); `torch`, `yaml`, `pandas`, `scipy`, `matplotlib`, `tqdm` are all confirmed **absent** there. Rebuild it with:
```bash
~/.pyenv/versions/3.11.9/bin/python3.11 -m venv venv311 && venv311/bin/pip install -r requirements.txt
```
**The 3.11 gate is PASSED** (measured 2026-09-29): the packaged `gru_v000/solution.py` scores **t0 0.579720 / t1 0.643295 / mean 0.611508** on validation sequences 0–4 under `venv311/` — byte-identical to the 3.13.3 result — and is deterministic across two runs. Latency under 3.11: **27.92 µs/row p50, 41.75 p99** (0.46 transform + 25.65 ONNX), gate PASS. Any future bundle must be re-checked with `venv311/bin/python tools/score_solution.py ...` and `venv311/bin/python tools/bench_latency.py --solution ...`; do not use `venv/` for those.

**Packages.** Pre-existing and *guaranteed in the scorer*: `numpy 2.2.6`, `onnxruntime 1.23.2`, `pyarrow 19.0.1`, `protobuf 7.36.2`, `sympy 1.14.0`, `flatbuffers`, `packaging`, `mpmath`, plus `humanfriendly`/`coloredlogs` (coloredlogs is an onnxruntime dep). Installed by us into `venv/` for **training/EDA only, NOT guaranteed at scoring time**: `torch 2.14.0` (MPS available, 8 torch threads), `PyYAML 6.0.3`, `onnx 1.23.0`, `matplotlib 3.11.2`, `tqdm 4.70.1`, `pillow 12.3.0`.

**Git.** The folder **is** a git repository, remote `origin` = `https://github.com/DerrickUnleashed/AreWeWundering.git` (branch `main`, **private**). **Datasets are deliberately NOT in git** (see below). `git log` mixes our framework commits with the owner's own (`Adding gitattributes`, `Add Valid_mask.parquet`), so check `git log --format='%h %an %s'` before assuming who changed what. `.gitignore` covers `cache/`, `exploration/checkpoints/`, `exploration/logs/`, `submissions/*/*.zip`, `venv/`, `__pycache__/`, `.DS_Store`, and the two oversized parquets. Do not `git init`; do not commit without being asked.

**Why the parquets are not in git — the GitHub LFS limits, measured 2026-09-29:**
- **A single LFS object may not exceed 2 GiB.** `train.parquet` (29.2 GB) and `valid.parquet` (5.6 GB) are rejected by the LFS API with `422 Size must be less than or equal to 2147483648`. This is *not* the familiar 100 MB per-blob limit (LFS bypasses that entirely); it is a separate per-object ceiling that **splitting the repository does not avoid**.
- Worse, the LFS **pre-push hook then aborts the whole push**, and GitHub's **server-side** pre-receive hook independently refuses any push whose history references objects missing upstream — so `--no-verify` does not help.
- Even split into <2 GiB parts, 33 GB exceeds the free Git LFS tier (10 GiB storage / 50 GiB bandwidth per month), so it would need LFS *billing*.
- **Resolution (2026-09-29):** both files were removed from git history with `git filter-repo --invert-paths --path datasets/train.parquet --path datasets/valid.parquet`, then the branch was force-pushed with `--force-with-lease`. This dropped exactly 2 commits (50 → 48) and changed every SHA after them. `valid_mask.parquet` (92 MB) is under the limit and **is** LFS-tracked and pushed. `datasets/MANIFEST.md` records the sha256 of all three files and the cache-regeneration command.
- **Caveats:** (a) the local `.git/lfs` still holds ~33 GB of now-unreferenced objects; `git lfs prune --dry-run` reports 2 prunable files (35 GB) — reclaimable, but only ask first. (b) with the LFS filter active, `git status` on a changed `.parquet` reports the *pointer* size (133 B), not the real file; a "modified datasets/…" line is not evidence of data loss — check `shasum -a 256` and `pq.ParquetFile(...).metadata`. (c) the working copies of all three parquets were verified intact after every step.

**Verified end state:** `git lfs fsck` OK; a fresh `git clone` of the remote yields working `tools/`, `exploration/`, `submissions/`, and a real 37,460,000-row `datasets/valid_mask.parquet`. The only thing a clone cannot get is the two big parquets — re-download the starter pack for those, or rebuild `cache/` with `tools/make_shards.py` in ~7 min.

**The shard cache was BUILT (2026-09-29), not estimated.** `cache/train/` (46.0 GiB, 212,140,000 rows) and `cache/valid/` (8.2 GiB, 37,460,000 rows), float16 X + float32 y, both `--verify OK`. Measured build time: **train 5.9 min, valid 1.1 min** (≈ 596 k rows/s, parquet-decode bound, single-threaded) — far faster than the 25–35 min originally estimated. `cache/norm_stats.json` holds train-only statistics over all 10,607 sequences. Peak RAM stays flat (one 20 k row group at a time, ~10 MB), so the 8 GB ceiling is never approached; the `--ram-gb` flag is recorded but not actively enforced. `tools/make_shards.py --verify-only` re-checks a build without touching it.

**Parquet layout (metadata only, nothing loaded whole).**

| File | Size | Rows | Row groups | Cols | Rows / row group |
|---|---:|---:|---:|---:|---:|
| `datasets/train.parquet` | 29,240,219,011 B | 212,140,000 | 10,607 | 117 | 20,000 |
| `datasets/valid.parquet` | 5,615,509,128 B | 37,460,000 | 1,873 | 118 | 20,000 |
| `datasets/valid_mask.parquet` | 91,972,846 B | 37,460,000 | 1,873 | 3 | 20,000 |

- **One row group == one sequence** (confirmed: `seq_ix` constant and `step_in_seq == arange(20000)` inside each group). This is the unit of streaming.
- `train.parquet` columns: `seq_ix(int32), step_in_seq(int32), need_prediction(bool), i0_p0..i0_dv3, i1_p0..i1_dv3, a0..a7, t0, t1`. **There is no `is_scored` in train** (confirmed). 117 columns.
- `valid.parquet` = same + `is_scored(bool)` after `need_prediction`. 118 columns. `ScorerStepByStep` requires exactly this order.
- `valid_mask.parquet` = `seq_ix, step_in_seq, is_scored`. Verified `is_scored` matches `valid.parquet` row group 0 element-wise.
- Feature column order (from `utils.FEATURE_COLUMNS`, identical to `docs/data_overview.md`): for i in {0,1}: `p0..p21` (22, price-like, 0–10 bid / 11–21 ask), `v0..v21` (22, volume-like), `dp0..dp3` (trade price), `dv0..dv3` (trade volume) → 52 each; then `a0..a7`. **Never sort these names alphabetically** (`i0_p10` would land next to `i0_p1`). The index is an identifier, not a depth level.

**Data sanity (row groups 0 / 1 / 5000 / 10606 of train, 0 of valid).** No NaN, no ±inf anywhere in features or targets. Targets: `t0` min/max −2.383/3.010, mean 0.108, std 0.738 (train rg 0); `t1` min/max −2.132/2.383, mean −0.129, std 0.716. **1.3–1.7 % of target values fall outside ±2**, so the metric's clipping is not a formality. `corr(t0, t1) = −0.796` on scored rows of valid rg 0 — the two targets are strongly (negatively) related, so a shared trunk is a sensible default.

**Features are already standardised by the organisers** (global mean ≈ −0.005, std ≈ 1.01 on train rg 0; per-column means range from −3.06 (`a7`) to +1.96 (`i0_p19`), stds from 0.04 (`i1_p12`) to 2.91 (`a7`)). Many columns saturate near ±2.33, and a floor sentinel of exactly **−5.199** appears in `a5`, `a6`, `a7` and `i1_dv0`/`i1_dv3` (a "no trade / missing" marker — treat as a categorical flag, not a number). Zero fractions are ~0 everywhere except `i0_dp2` (52 % zeros in train rg 0, 29 % in valid) and `i1_dp2` (23 %/48 %) — the `dp2`/`dp1` trade features are sparse event indicators.

**`is_scored` pattern in validation (row group 0).** 2,599 of 19,901 required rows = **13.0 %**. Never inside warm-up. Scored rows appear in **short contiguous runs of 1–5 steps scattered across the whole sequence** (most common gap = 1, then 2, 3, 4, 5) — i.e. it looks like a sampled subset of "moments", exactly as the docs say, not a contiguous block. It is *not* a prefix or a suffix, so there is no way to detect scored rows at test time and none should be attempted.

**Metric weight concentration (valid row group 0, scored rows).** The top **0.5 %** of scored rows hold **11.8 %** of Σ|y|; the top **5 %** hold **69.6 %**. This is the single most important structural fact about the objective: bulk accuracy is nearly irrelevant, tail accuracy is everything.

**float16 vs float32 for the shard cache — decision: float16.** Measured on 4 train row groups: max absolute quantisation error 0.00194 (median 0.00097), i.e. the fp16 step near |x|≈1 is 2^-10 = 0.00098; relative to each column's own std the worst column is `i1_p13` at **1.7 %**, mean **0.35 %**. That is far below the noise floor of the model itself. Benefit: 47.5 GB train + 8.4 GB valid ≈ 56 GB total, which fits inside 64 GiB of RAM (so shards stay in page cache and training I/O is free) and well inside 280 GiB of disk. float32 would be 95 + 17 = 112 GB, i.e. guaranteed page-cache thrashing. `y` is kept in float32 (it must be, because clipping to ±2 and |t| weights are exact) and the bool masks are stored as `np.bool_`.

**Target ranges over the FULL train set** (`cache/norm_stats.json`, all 212,140,000 rows — supersedes the row-group-0 sample above, which badly understated them): `t0` mean **−0.271**, std **0.711**, min **−125.6**, max **+74.5**; `t1` mean **−0.268**, std **0.715**, min **−179.9**, max **+77.3**. **10.94 % of target values are exactly 0** and therefore carry zero metric weight. The extreme tails matter: the metric clips to ±2, so any tail mass beyond that is invisible to the score, and a loss that is not clip-aware will spend capacity fitting values the metric never sees. Always keep the clip in the loss *and* in the model output.

**Feature statistics over the FULL train set:** per-column means span **−1.81 … +1.98** and stds **0.026 … 2.03**; zero NaN and zero inf across all 212 M × 112 feature values. The organisers' standardisation is real but imperfect (it is a global affine map, not per-window), so `robust_zscore` remains a hypothesis to ablate.

**Baseline latency on this machine** (5 validation sequences, 100,000 rows, real callback loop, 1 ORT thread, spinning disabled): **27.9 µs/row mean, p50 26.9, p90 28.6, p99 45.7**. Projected for the 39.4 M-row test set: **18.3 min at the mean, 30.0 min at p99** — versus a 60 min budget. So the baseline alone would pass; roughly 2.5x of per-row headroom exists for a bigger model or a heavier feature transform.

**Baseline score on the first 5 validation sequences** (row groups 0–4, i.e. *not* the official full-set number): t0 = 0.6015, t1 = 0.7011, **mean = 0.6513**, 9,110 selected rows. Recorded as the anchor for subset comparisons only; the official 0.617052 refers to all 1,873 sequences. A 5-sequence subset is therefore ~0.034 optimistic relative to the full set and must never be quoted as a full-validation score.

**Our scorer is verified against the organiser's own loop.** `tools/score_solution.py --solution baseline/solution.py --sequences 5` returns **t0 0.601470 / t1 0.701132 / mean 0.651301**, and a hand-built 5-row-group parquet run through `utils.ScorerStepByStep` itself returns **0.6014696434521138 / 0.7011323559886197 / 0.6513009997203667** — the same numbers to the last printed digit, same 9,110 selected rows. So the metric path in this repo is the organiser's, not a re-derivation.

**Framework smoke test `gru_v000`** (status `smoke`, 60 steps, 20 train / 10 validation sequences, random-init, CPU): feature-transform parity `.step` vs `.batch` = **0.0**; params 192,258 (identical to the baseline by construction); best WP **0.570341** on 10 validation sequences (t0 0.557097, t1 0.583585, 15,615 scored rows); ONNX == PyTorch to **2.4e-07**. Its packaged bundle `submissions/submission_gru_v000/` passes the whole promotion gate: contract PASS, determinism PASS, parity streaming-vs-offline **6.6e-07** (tol 1e-4), ZIP 1.42 MB, score through the real loop **0.611508** on 5 validation sequences. Its latency: **29.2 µs/row mean, p50 27.7, p99 46.0**, of which **0.5 µs transform + 25.5 µs ONNX** → 18.2 min projected at p50, 30.2 min at p99. That is the reference point for "a 2×128 GRU with an identity transform" and it is within 2 µs of the organiser's baseline. `gru_v000` is a framework example, **not** a result — never cite its score as a modelling reference. Two fields in that bundle are stale by construction and must be read from the numbers above instead: `metadata.json.gate.latency` is `null` and `results.latency_us_*` in the yaml are `null`, because the bundle was produced by the pre-fix tools (the packager only started recording latency after it was packaged, and bundles are immutable). Re-running the packager for a *new* version populates both.

## 15. Known discrepancies between this file and the organiser docs

1. **Prompt vs `docs/faq.md` — test-set size.** The briefing assumed ≈37.46 M rows. The FAQ states **1,970 sequences / 39,400,000 rows**. The docs win: the budget is **91.4 µs/row**, not 96 µs/row. All budget numbers in this file use 39.4 M.
2. **Prompt vs `utils.py` — "prediction may be a list/array/tuple".** `utils.ScorerStepByStep` coerces the return value with `np.asarray(value, dtype=np.float32)` and only checks `shape == (2,)` and finiteness, so a list would technically pass the *local* scorer. `docs/submission_guide.md` however specifies "two finite float32-compatible values with shape `(2,)`" and the baseline returns `np.ndarray`. **We return `np.ndarray` of shape (2,), dtype float32** — no ambiguity at the real scorer.
3. **Prompt's yaml default `features.signed_log_volumes: true` / `normalisation: robust_zscore` vs the shipped baseline.** The baseline graph contains no normalisation at all, and per-column statistics show the features are already standardised. Those two flags are therefore *hypotheses to ablate*, not defaults to inherit. Recorded in §12 Phase 4.
4. **Prompt's third-party note "public ≈ held-out − ~0.04" and "a GRU of 2 layers × 192 with a 128-unit input projection".** Unverified; the actual baseline is 2 × 128 with **no** input projection. Treat all third-party numbers as hypotheses.
5. **`docs/submission_guide.md` Dockerfile vs `requirements.txt`.** The Dockerfile shows a PyTorch CPU extra-index and `orbax-checkpoint`, but the install line is `pip install -r requirements.txt`, and `requirements.txt` lists only numpy/onnxruntime/pyarrow. **We therefore treat torch as absent from the scorer** and forbid it in `solution.py`. VERIFY by asking the organisers if a 3.11 environment with torch can be provided.
6. **`docs/quick_start.md` names the competition "Alpha Connectome"** while the README and the task title say "Alfa". Cosmetic only; the starter pack is the same. Noted, no action.
7. **`baseline/README.md` and `quick_start.md` disagree on how to zip** (`cd baseline; zip -r submission.zip solution.py baseline.onnx` vs `zip -r ../submission.zip .` from the submission dir). Both put `solution.py` at the root. We follow `tools/package_submission.py` and always build the ZIP from inside the bundle directory.
8. **torch 2.14 vs the exporter.** `torch.onnx.export` defaults to the dynamo exporter from torch 2.9 on, which imports `onnxscript` — not installed, not in `requirements.txt`. The template passes `dynamo=False` to keep the dependency-free TorchScript path (the one that produced `baseline.onnx`, producer `pytorch 2.5.1`). It emits a `DeprecationWarning`; that is expected, not a problem.
9. **nn.GRU hidden-state layout.** `nn.GRU` wants one packed tensor `(num_layers, B, H)`, whereas `baseline.onnx` carries `hidden_0`/`hidden_1` as two separate `(1,1,H)` inputs. Exporting a 2-layer `nn.GRU` therefore produces extra Split/Slice/Gather/Concat nodes and a ~30 % latency penalty (see §10.4). The template works around it with a `ModuleList` of single-layer `nn.GRU`s.
10. **The 3.11 gate — CLOSED 2026-09-29, see §14.** `venv311/` reproduces the scorer exactly and the `gru_v000` bundle scores identically under it. The earlier VERIFY is resolved; §15.5 remains a question for the organisers only in the weaker sense that we have *proved* torch is unnecessary rather than *asked* whether it happens to be installed.

## 16. Definition of done for any task

State what was run, the exact command, the resulting numbers, the files created/changed, and the next best action. Never claim a score you did not compute with `tools/metric_ref.py`. Never claim a file exists that you did not create and verify.
