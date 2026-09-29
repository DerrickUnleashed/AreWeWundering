# Experiment ranking — v001 to v009

All scores are `full_valid_wp`: global weighted Pearson over every one of the
1,873 validation sequences (3,528,273 scored rows), computed with the
reference metric in `tools/metric_ref.py`.

| Rank | Ver | What it changed | full_valid_wp | sigma vs seed mean | Verdict |
|---:|---|---|---:|---:|---|
| 1 | **v009** | signed-log volumes | 0.674456 | +1.8 | within noise |
| 2 | **v008** | cross-instrument (i0 vs i1) | 0.674445 | +1.7 | within noise |
| 3 | **v002** | baseline config, seed 1337 | 0.674350 | +1.1 | within noise |
| 4 | **v001** | 1x256 wider GRU | 0.674315 | +0.9 | within noise |
| 5 | **v007** | first differences | 0.674258 | +0.5 | within noise |
| 6 | **v006** | baseline config, seed 1339 | 0.674149 | -0.2 | within noise |
| 7 | **v005** | baseline config, seed 1338 | 0.674037 | -0.9 | within noise |
| 8 | **v003** | aux MSE + EMA | 0.673711 | **-2.9** | **worse, real** |
| 9 | **v004** | 2x192 + input projection | 0.672206 | **-12.4** | **worse, real** |

All nine beat the organiser baseline of 0.617052 by roughly +0.055 to +0.057.

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
