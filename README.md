# Experiment ranking — v001 to v011

All scores are `full_valid_wp`: global weighted Pearson over every one of the
1,873 validation sequences (3,528,273 scored rows), computed with the
reference metric in `tools/metric_ref.py`.

<!-- BEGIN GENERATED: leaderboard -->

| Rank | Ver | What it changed | full_valid_wp | Status | Public score | Duration | Memory |
|---:|---|---|---:|---|---|---:|---|
| 1 | **v011** | soft mask on P(is_scored) + state reset | 0.679986 | Yet to package | -- | -- | -- |
| 2 | **v009** | signed-log volumes | 0.674456 | Ready to Submit (latency unverified) | -- | -- | -- |
| 3 | **v008** | cross-instrument (i0 vs i1) | 0.674445 | Ready to Submit (latency unverified) | -- | -- | -- |
| 4 | **v002** | baseline config, seed 1337 | 0.674350 | Submitted | 0.6435 | 35m02s (58%) | 628 MB |
| 5 | **v001** | 1x256 wider GRU | 0.674315 | Submitted | 0.6416 | 34m02s (57%) | 626 MB |
| 6 | **v007** | first differences | 0.674258 | Ready to Submit (latency unverified) | -- | -- | -- |
| 7 | **v006** | baseline config, seed 1339 | 0.674149 | Ready to Submit (latency unverified) | -- | -- | -- |
| 8 | **v005** | baseline config, seed 1338 | 0.674037 | Ready to Submit (latency unverified) | -- | -- | -- |
| 9 | **v003** | aux MSE + EMA | 0.673711 | Submitted | 0.6457 | 30m57s (52%) | 627 MB |
| 10 | **v004** | 2x192 + input projection | 0.672206 | Submitted | 0.6446 | 43m16s (72%) | 630 MB |

Ranked by `full_valid_wp`, the global weighted Pearson over all 1,873
validation sequences (3,528,273 scored rows).

### Still in flight

No final `full_valid_wp` yet, so not ranked.

| Ver | What it changed | best so far (subset) | Status | Public score |
|---|---|---:|---|---|
| **v010** | 6 epochs instead of 3 | -- | Training | -- |

<!-- END GENERATED: leaderboard -->

## Why most of this table is one statistical blob

The noise floor comes from `gru_v002` / `gru_v005` / `gru_v006`, which are the
*identical* configuration at seeds 1337 / 1338 / 1339. Their spread is the
smallest difference that means nothing:

- mean 0.674179, std 0.000159, range 0.000313
- so the detection threshold is **2 sigma = +/-0.0003**

Seven configurations -- spanning a wider GRU, three separate feature-engineering
ideas, and three seeds of the same config -- span 0.0004 in total. **That is
smaller than the 0.0003 gap between three seeds of a single config.** Those seven
are therefore statistically indistinguishable, and their ordering is noise, not
quality. Note that 1.8 sigma is only a ~4% one-sided chance of beating the seed
mean by luck, so v009's position above the baseline configs is not a result.

## v011 is the one real win

**`v011` scores 0.679986, which is +0.0058 above the seed mean -- 36 sigma.** It
is the only version in this batch that clears the noise floor by a wide margin,
and the largest single gain since the baseline config itself.

The recipe is the third-party one: a soft mask weighting the training loss by
`P(is_scored)` for each row, plus a periodic reset of the recurrent state every
4,000 steps. Both were trained together, so the run is deliberately confounded --
it does not yet say which lever did the work.

What makes it interesting is *where* it works. Every input-representation idea
failed (below), and the winning change touches the objective and the state
handling rather than the features. A soft mask on likely-scored rows is a better
match for the metric than the previous weighting, because the metric only scores
`need_prediction AND is_scored` rows while the loss was spending capacity on
scored-looking rows that never count. It is a training-only change and costs
nothing at inference.

The two failures below are also informative: they are the *objective* and
*capacity* directions, which the soft-mask result now brackets.

## The failures

- **v003 (aux MSE + EMA) is worse at -2.9 sigma.** The auxiliary weighted-MSE
  term actively hurt the correlation objective.
- **v004 (2x192 + input projection) is much worse at -12.4 sigma.** Doubling the
  width and adding an input projection is a clear regression.

## The input-representation dead end

**None of the input-representation ideas worked.** First differences (v007),
cross-instrument features (v008), and signed-log volumes (v009) all landed inside
the noise band of the baseline config, as did extra capacity (v001). The
organisers ship pre-standardised features, and the baseline already sits at a
local optimum with respect to them. The remaining headroom was not in feature
engineering -- which is why the soft-mask result landing outside that band is
meaningful rather than lucky.

Note that the feature transforms are also expensive at inference (6.7-12.9 us/row
for v005-v009 against ~0.5 for the baseline's identity path), so those ideas
could not have been cheap wins even if they had scored.

## Choosing a ship candidate

**`v011` is the candidate**, and the only version here justified on score. It
beats the best submitted model (v003, public 0.6457) by +0.0055 local while
keeping the same 2x128 architecture and 192,258 parameters, so its inference cost
should be close to the baseline's.

Before shipping it, the latency gate must be re-run on an idle box. The gates on
v005-v009 were measured while v010 and v011 held the cores and are inflated by
about 1.7x, which is why they are marked "latency unverified" rather than
"Ready to Submit".

## Caveat: the public leaderboard disagrees

The public scores ran the opposite way to the local ranking:
v003 0.6457 > v004 0.6446 > v002 0.6435 > v001 0.6416. Locally, v003 ranks
*last* of the seven statistically-tied configs at -2.9 sigma, and v004 last
overall at -12.4 sigma. Since the public set is a sample of the test data, the
local metric clearly does not transfer perfectly to it.

Two consequences worth keeping in mind:

1. Treat the public board as the only real signal for final selection. Local
   `full_valid_wp` is still the right tool for detecting a change of the size v011
   made (+0.0058, 36 sigma), but it is not a reliable ranker inside the noise band.
2. It also means v011's local win needs public confirmation before it is trusted
   as the ship candidate. It should be the next submission for that reason alone.

Every one of our completed runs beats the organiser's 0.5935 baseline, and the
programme as a whole is worth roughly +0.05 public.
