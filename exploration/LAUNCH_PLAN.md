# Launch plan — 4 parallel training runs (NOT YET LAUNCHED)

**Status: awaiting explicit confirmation. Do not launch without it.**

Prepared 2026-09-29. The four runs below are fully configured, validated and
latency-gated. The only thing missing is a human saying "go".

## Why this file exists

The first attempt to launch these four runs **silently died**: they were started
with `nohup ... &` from a tool-invoked shell, and when that shell was aborted the
whole *process group* was killed — `nohup` only ignores SIGHUP, it does not escape
the group. All four logs stop mid-step with no traceback. Their partial logs are
archived in `exploration/logs/_aborted_first_launch_2026-09-29/`.

`tools/launch.py` fixes this by calling `os.setsid()` in the forked child, which
puts it in a new session with no controlling terminal. **Verified**: a detached
child kept running (6 -> 14 ticks) after its supervisor's entire process group was
`SIGKILL`ed.

## The four runs

All four: CPU (`device: cpu`), `train_sequences: null` (**all 10,607 sequences**),
`epochs` x 6,630 steps/epoch, `eval_subset_sequences: 200` (**the same fixed first
200 validation sequences, so the four are directly comparable**), seed 1337,
3 threads each.

| Version | Single variable under test | Shape | Params | Steps | Rows |
|---|---|---|---:|---:|---:|
| `gru_v001` | width beats depth | 1 x 256 | 284,674 | 19,890 | 636 M |
| `gru_v002` | re-standardise inputs | 2 x 128 + z-score | 192,258 | 19,890 | 636 M |
| `gru_v003` | aux MSE + EMA | 2 x 128 | 192,258 | 19,890 | 636 M |
| `gru_v004` | third-party shape | 2 x 192 + proj 128 | 422,658 | 13,260 | 424 M |

`v004` is cut to 2 epochs because it is ~1.65x slower per step; this keeps the row
budget comparable rather than giving it 1.5x the compute of the others.

## Latency gate (already passed, 1 thread, real validation rows)

| Version | p50 us/row | p99 us/row | params | gate (68.5 us) |
|---|---:|---:|---:|---|
| `gru_v001` | 32.37 | 51.71 | 284,674 | PASS / PASS |
| `gru_v002` | 31.58 | 50.88 | 192,258 | PASS / PASS |
| `gru_v003` | 27.67 | 42.71 | 192,258 | PASS / PASS |
| `gru_v004` | 44.33 | **71.21** | 422,658 | PASS / **FAIL** |

`v004` exceeds the budget at p99. It may still be trained (training cost is
unrelated to inference cost) but it is **promotion-risky**: on a grader slower per
core it could approach the 60-minute limit. Do not ship it without a re-measured
latency on an idle machine.

## Validated (no training run needed)

For each of the four: feature transform `.step()` vs `.batch()` parity **0.0**
(tol 1e-6), finite loss and finite gradient norm after one forward+backward on
real cached rows, `chunk_len` divides 20,000, and `max_steps` correctly derived
from `epochs` over all 10,607 sequences.

## Expected wall-clock (3 threads each, 12 physical cores)

Aggregate 4-way throughput measured at 178,475 rows/s = **3.57x** a single run.

| Version | s/step | train | +8 evals | total |
|---|---:|---:|---:|---:|
| `gru_v001` | ~0.69 | 3.8 h | 0.3 h | ~4.1 h |
| `gru_v002` | ~0.78 | 4.3 h | 0.3 h | ~4.6 h |
| `gru_v003` | ~0.72 | 4.0 h | 0.3 h | ~4.3 h |
| `gru_v004` | ~1.20 | 4.4 h | 0.3 h | ~4.7 h |

## The command (do not run unconfirmed)

```bash
for v in v001 v002 v003 v004; do
  venv/bin/python tools/launch.py --threads 3 --log-dir /tmp/wnn_runs \
    -- venv/bin/python exploration/models/train_gru_$v.py \
       --config exploration/version/version_gru_$v.yaml --device cpu
done
```

Then, **sequentially and only when the machine is idle** (latency measured under
4-way contention is meaningless):

```bash
for v in v001 v002 v003 v004; do
  venv/bin/python tools/package_submission.py --model gru --version $v --sequences 20
done
```

## After these four

The seed noise floor is **still unmeasured**, so the four results above are
exploratory: a 0.005 difference between them may be pure seed noise. The next run
should be a same-recipe, different-seed replicate to fix that threshold
(AGENTS.md §11). Do not over-interpret the leaderboard of these four.
