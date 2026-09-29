# Experiment ranking — v001 to v009

All scores are `full_valid_wp`: global weighted Pearson over every one of the
1,873 validation sequences (3,528,273 scored rows), computed with the
reference metric in `tools/metric_ref.py`.

| Rank | Ver | What it changed | full_valid_wp | sigma vs seed mean | Verdict | Status | Public score | Duration | Memory |
|---:|---|---|---:|---:|---|---|---:|---|---:|
| 1 | **v009** | signed-log volumes | 0.674456 | +1.8 | within noise | Yet to package | -- | -- | -- |
| 2 | **v008** | cross-instrument (i0 vs i1) | 0.674445 | +1.7 | within noise | Yet to package | -- | -- | -- |
| 3 | **v002** | baseline config, seed 1337 | 0.674350 | +1.1 | within noise | Submitted | 0.6435 | 35m02s (58%) | 628 MB |
| 4 | **v001** | 1x256 wider GRU | 0.674315 | +0.9 | within noise | Submitted | 0.6416 | 34m02s (57%) | 626 MB |
| 5 | **v007** | first differences | 0.674258 | +0.5 | within noise | Yet to package | -- | -- | -- |
| 6 | **v006** | baseline config, seed 1339 | 0.674149 | -0.2 | within noise | Yet to package | -- | -- | -- |
| 7 | **v005** | baseline config, seed 1338 | 0.674037 | -0.9 | within noise | Yet to package | -- | -- | -- |
| 8 | **v003** | aux MSE + EMA | 0.673711 | **-2.9** | **worse, real** | Submitted | **0.6457** | 30m57s (52%) | 627 MB |
| 9 | **v004** | 2x192 + input projection | 0.672206 | **-12.4** | **worse, real** | Submitted | 0.6446 | 43m16s (72%) | 630 MB |

All nine beat the organiser baseline of 0.617052 by roughly +0.055 to +0.057.

Duration and memory are the grader's own readings for the 39.4 M-row test set,
as a fraction of the 60-minute / 16 GB limits.

### Reference points on the leaderboard

| Entry | Public score | Duration | Memory |
|---|---:|---|---:|
| organiser `baseline_submission.zip` | 0.5935 | 26m21s (44%) | 624 MB |
| our `gru_v000` (60-step smoke test) | 0.4986 | 29m47s (50%) | 626 MB |
| **our best** (`gru_v003`) | **0.6457** | 30m57s (52%) | 627 MB |

The organiser's baseline scores 0.5935 and every one of our completed runs scores
0.64-0.65, so the training programme is worth roughly **+0.05 public**, consistent
with the +0.057 measured locally. `gru_v000` is the 60-step framework smoke test
and is not a modelling result: it is below the organiser's own baseline because it
has barely trained, and it was uploaded to validate the submission path only.

### v004 was shippable after all

This corrects a verdict in `AGENTS.md`. v004 was marked **DO NOT SHIP** on a local
p50 of 44.33 us/row, projected to 79 % of budget at p50 and ~76 min at p99. The
grader ran it in **43m16s, 72.1 % of the limit**, and it scored 0.6446 -- the
second-best public score of our entries.

So the p99 projection was too pessimistic: the latency tail is thinner on the
grader than a contended local p99 suggests, and the 1.62x slowdown factor measured
from `gru_v000` does not apply cleanly to a wider model with more per-row work.
v004 remains the worst model locally (-12.4 sigma), so it is still not the one to
ship, but the "must not be uploaded" language was wrong and the latency gate is
calibrated too conservatively.

### Still in flight

Not ranked yet, since neither has a final `full_valid_wp`.

| Ver | What it changed | best so far (subset) | Status | Public score |
|---|---|---:|---|---:|
| **v010** | 6 epochs instead of 3 | 0.684870 (step 24,999) | Training | -- |
| **v011** | soft mask on `P(is_scored)` + state reset | 0.689067 (step 14,999) | Training | -- |

v011 is the only hypothesis in this batch with a mechanism that could explain a
gain: it changes the objective and the state handling rather than the inputs. At
0.689067 on the subset metric it sits about +4.9 sigma above the seed mean
(0.685955), which is the first result here to clear the noise bar. Its two levers
are confounded, so a single-variable follow-up would be needed to attribute it.

### Status column

One of **Training**, **Yet to Train**, **Ready to Submit**, **Submitted**, or
**Do not ship**. The public score is populated only for Submitted rows, from the
competition leaderboard.

- **Submitted** -- a bundle exists in `submissions/` and has been uploaded and
  scored. v001, v002, v003, and v004 are all submitted.
- **Do not ship** -- reserved for a bundle that has been measured as unable to
  finish inside the grader's limits. **Nothing currently holds this status.**
- **Yet to package** -- trained and scored, but no bundle built yet. v005 through
  v009 are the same 2x128 architecture as v002, which the grader ran in 35m02s
  (58 % of budget), so they have ample headroom and are eligible to package on
  demand.
- **Training** -- v010 (6 epochs) and v011 (soft mask + state reset) are still
  running.

v007 is a legal candidate but has a truncated training log (stdout buffering lost
roughly 12,000 lines), so it is the weakest of the seven on the reproducibility
criterion even though it ranks 5th.

## Why ranks 1-7 are one statistical blob

The noise floor comes from `gru_v002` / `gru_v005` / `gru_v006`, which are the
*identical* configuration at seeds 1337 / 1338 / 1339. Their spread is the
smallest difference that means nothing:

- mean 0.674179, std 0.000159, range 0.000313
- so the detection threshold is **2 sigma = +/-0.0003**

Seven configurations -- spanning a wider GRU, three separate feature-engineering
ideas, and three seeds of the same config -- span 0.0004 in total. **That is
smaller than the 0.0003 gap between three seeds of a single config.** Ranks 1
through 7 are therefore statistically indistinguishable, and their ordering is
noise, not quality. Note that 1.8 sigma is only a ~4% one-sided chance of
beating the seed mean by luck, so v009's rank-1 position is not a result.

## The only two real results are negative

- **v003 (aux MSE + EMA) is worse at -2.9 sigma.** The auxiliary weighted-MSE
  term actively hurt the correlation objective.
- **v004 (2x192 + input projection) is much worse at -12.4 sigma.** Doubling the
  width and adding an input projection is a clear regression. It is also
  independently latency-blocked, so it could not ship regardless.

## The uncomfortable conclusion

**None of the input-representation ideas worked.** First differences (v007),
cross-instrument features (v008), and signed-log volumes (v009) all landed inside
the noise band. Nor did extra capacity (v001) or a longer schedule test
(v010, still running at the time of writing).

The baseline configuration is at a local optimum, and the remaining headroom is
not in feature engineering. The one hypothesis with a plausible mechanism still
open is `v011` (soft mask on `P(is_scored)` plus periodic state reset), which
changes the objective and state handling rather than the inputs.

## Choosing a ship candidate

Because ranks 1-7 are tied, the choice should be made on secondary criteria
rather than on the score:

- **Latency** -- v002 / v005 / v006 / v007 are all 2x128 at 27.96 us/row p50
- **Reproducibility** -- v007's training log is truncated by stdout buffering, so
  it should be avoided despite ranking 5th
- **Simplicity** -- v002 is the unmodified baseline configuration

That points to **v002**, with v009 as the alternative if its +1.8 sigma is worth
gambling on.

## Caveat: the public leaderboard disagrees

The public scores ran the opposite way to the local ranking:
v003 0.6457 > v004 0.6446 > v002 0.6435 > v001 0.6416. Since the public set is a
sample of the test data, this is mild evidence that the local metric does not
transfer perfectly. v002 and v003 tie locally while v003 wins publicly, which is a
caution against over-trusting any of ranks 1-7.
