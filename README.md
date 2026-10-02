# Experiment ranking — v001 to v029

All scores are `full_valid_wp`: global weighted Pearson over every one of the
1,873 validation sequences (3,528,273 scored rows), computed with the
reference metric in `tools/metric_ref.py`.

<!-- BEGIN GENERATED: leaderboard -->

| Rank | Ver | What it changed | full_valid_wp | Status | Public score | Duration | Memory |
|---:|---|---|---:|---|---|---:|---|
| 1 | **v031** | v028's exact recipe and architecture with... | 0.687376 | Submitted | 0.6587 | 45m41s (76%) | 628 MB |
| 2 | **v032** | v028's exact recipe -- same architecture,... | 0.687219 | Ready to Submit | -- | 42m30s (71%) | -- |
| 3 | **v030** | v028's exact recipe and architecture with... | 0.686919 | Submitted | 0.6585 | 47m11s (79%) | 629 MB |
| 4 | **v028** | v023's exact recipe and architecture with... | 0.686510 | Submitted | 0.6585 | 46m52s (78%) | 628 MB |
| 5 | **v034** | v028's exact recipe with the ONE axis that... | 0.686000 | Ready to Submit | -- | 49m30s (82%) | -- |
| 6 | **v022** | dozken arch + his teacher | 0.685228 | Submitted | 0.6565 | 50m46s (85%) | 631 MB |
| 7 | **v024** | distil, batch 64 (his recipe) | 0.684818 | Submitted | 0.6560 | 50m17s (84%) | 631 MB |
| 8 | **v029** | Distilled student on dozken's architecture... | 0.684683 | Submitted | 0.6568 | 44m27s (74%) | 628 MB |
| 9 | **v023** | distil, smaller (52% of teacher) | 0.684681 | Submitted | 0.6568 | 45m36s (76%) | 628 MB |
| 10 | **v025** | v023's exact recipe (hidden 128, 225,922... | 0.684438 | Submitted | 0.6580 | 46m07s (77%) | 628 MB |
| 11 | **v026** | v023's exact recipe and architecture at 2x... | 0.684335 | TIMED OUT on grader | no score | 69m57s (117%) | 646 MB |
| 12 | **v027** | v026 at 2x again: hidden 415, 1,742,717... | 0.683058 | TIMED OUT on grader | no score | 69m57s (117%) | 643 MB |
| 13 | **v017** | distilled student, 117k (0.61x teacher) | 0.681866 | Submitted | 0.6540 | 25m37s (43%) | 625 MB |
| 14 | **v016** | distilled student, 192k | 0.681753 | Submitted | 0.6541 | 30m56s (52%) | 627 MB |
| 15 | **v018** | soft mask + state reset 10k | 0.680603 | Submitted | 0.6518 | 29m21s (49%) | 627 MB |
| 16 | **v015** | v011 re-exported, reset served [shares v011's weights] | 0.679986 | Submitted | 0.6515 | 30m31s (51%) | 627 MB |
| 17 | **v011** | soft mask + state reset 4k | 0.679986 | Submitted | 0.6515 | 30m12s (50%) | 627 MB |
| 18 | **v014** | soft mask + reset 4k, 2x192 proj64 | 0.679676 | Submitted | 0.6517 | 40m12s (67%) | 632 MB |
| 19 | **v012** | soft mask + state reset 2k | 0.678784 | Submitted | 0.6512 | 29m47s (50%) | 627 MB |
| 20 | **v021** | dozken arch, no teacher | 0.678575 | Submitted | 0.6505 | 55m32s (93%) | 634 MB |
| 21 | **v019** | FIX loss: globally pooled | 0.678374 | Submitted | 0.6510 | 30m32s (51%) | 627 MB |
| 22 | **v013** | reset 4k, no soft mask (control) | 0.676198 | Submitted | 0.6490 | 28m36s (48%) | 627 MB |
| 23 | **v009** | input: signed-log tail compression | 0.674456 | Submitted | 0.6483 | 39m52s (66%) | 627 MB |
| 24 | **v008** | input: cross-instrument i0/i1 | 0.674445 | Submitted | 0.6473 | 41m17s (69%) | 626 MB |
| 25 | **v002** | input: robust z-score, 8-sigma clip | 0.674350 | Submitted | 0.6435 | 35m02s (58%) | 628 MB |
| 26 | **v001** | wider: 1 layer x 256 | 0.674315 | Submitted | 0.6416 | 34m02s (57%) | 626 MB |
| 27 | **v007** | input: first differences | 0.674258 | Submitted | 0.6481 | 42m16s (70%) | 626 MB |
| 28 | **v006** | seed replicate of v002 (1339) | 0.674149 | Submitted | 0.6475 | 35m32s (59%) | 631 MB |
| 29 | **v005** | seed replicate of v002 (1338) | 0.674037 | Submitted | 0.6481 | 36m56s (62%) | 626 MB |
| 30 | **v003** | aux \|t\|-weighted MSE + EMA | 0.673711 | Submitted | 0.6457 | 30m57s (52%) | 627 MB |
| 31 | **v004** | bigger: 2x192 with 128 proj | 0.672206 | Submitted | 0.6446 | 43m16s (72%) | 630 MB |
| 32 | **v010** | schedule: 6 epochs not 3 | 0.670696 | Submitted | 0.6427 | 34m37s (58%) | 626 MB |
| 33 | **v033** | A deliberately DIVERSE ensemble member, not... | 0.660415 | Submitted | 0.6274 | 47m22s (79%) | 628 MB |

Ranked by `full_valid_wp`, the global weighted Pearson over all 1,873
validation sequences (3,528,273 scored rows).

### Not ranked by `full_valid_wp`

Rows that cannot be ranked on `full_valid_wp`, each for a stated reason: a contaminated score that is not comparable with the rest (v020), or a framework smoke test that was never a modelling result (v000). Both have bundles; neither is a submission candidate.

| Ver | What it changed | best so far (subset) | Status | Public score |
|---|---|---|---|---|
| **v000** | framework smoke test (60 steps) | 0.570341 | Submitted | 0.4986 |
| **v020** | FIX loss + train+valid data  [SCORE CONTAMINATED: trained on cache/trainval, which appends valid[200:1872] -- 1,673 of the 1,873 validation sequences -- so this score is measured on 89% of its own training data and is not comparable with train-only runs] | 0.687502 | Submitted | 0.6513 |
| **v035** | v031's exact recipe with ONE change: hidden... | 0.067588 | Training | -- |
| **v036** | v031's exact recipe with ONE change:... | 0.026731 | Training | -- |
| **v037** | v031's exact recipe at 8 epochs instead of 3 | -- | Training | -- |
| **v038** | v031's exact recipe with ONE change: AdamW... | -- | Training | -- |
| **v039** | v031's exact recipe with a BETTER TEACHER | -- | Yet to Train | -- |

<!-- END GENERATED: leaderboard -->

<!-- BEGIN GENERATED: public-score ranking -->

### Ranked by public (leaderboard) score

This is the ranking that counts for the prize. It does **not** agree with the
table above, and the disagreements are the interesting part:

- **v022** has the 2nd-best `full_valid_wp` (0.685228) and only the 5th-best
  public score. Our local metric overrates it.
- **v025** is a local *regression* against v023 (-1.5 sigma) and our 2nd-best
  public result. The local metric underrates it.

| Rank | Ver | full_valid_wp | Public score | vs best | Grader | Outcome |
|---:|---|---:|---:|---:|---:|---|
| 1 | **v031** | 0.687376 | **0.6587** | +0.0000 | 45m41s (76%) | scored |
| 2 | **v030** | 0.686919 | **0.6585** | -0.0002 | 47m11s (79%) | scored |
| 3 | **v028** | 0.686510 | **0.6585** | -0.0002 | 46m52s (78%) | scored |
| 4 | **v025** | 0.684438 | **0.6580** | -0.0007 | 46m07s (77%) | scored |
| 5 | **v029** | 0.684683 | **0.6568** | -0.0019 | 44m27s (74%) | scored |
| 6 | **v023** | 0.684681 | **0.6568** | -0.0019 | 45m36s (76%) | scored |
| 7 | **v022** | 0.685228 | **0.6565** | -0.0022 | 50m46s (85%) | scored |
| 8 | **v024** | 0.684818 | **0.6560** | -0.0027 | 50m17s (84%) | scored |
| 9 | **v016** | 0.681753 | **0.6541** | -0.0046 | 30m56s (52%) | scored |
| 10 | **v017** | 0.681866 | **0.6540** | -0.0047 | 25m37s (43%) | scored |
| 11 | **v018** | 0.680603 | **0.6518** | -0.0069 | 29m21s (49%) | scored |
| 12 | **v014** | 0.679676 | **0.6517** | -0.0070 | 40m12s (67%) | scored |
| 13 | **v015** | -- | **0.6515** | -0.0072 | 30m31s (51%) | scored |
| 14 | **v011** | 0.679986 | **0.6515** | -0.0072 | 30m12s (50%) | scored |
| 15 | **v020** | 0.681773 | **0.6513** | -0.0074 | 29m52s (50%) | scored |
| 16 | **v012** | 0.678784 | **0.6512** | -0.0075 | 29m47s (50%) | scored |
| 17 | **v019** | 0.678374 | **0.6510** | -0.0077 | 30m32s (51%) | scored |
| 18 | **v021** | 0.678575 | **0.6505** | -0.0082 | 55m32s (93%) | scored |
| 19 | **v013** | 0.676198 | **0.6490** | -0.0097 | 28m36s (48%) | scored |
| 20 | **v009** | 0.674456 | **0.6483** | -0.0104 | 39m52s (66%) | scored |
| 21 | **v007** | 0.674258 | **0.6481** | -0.0106 | 42m16s (70%) | scored |
| 22 | **v005** | 0.674037 | **0.6481** | -0.0106 | 36m56s (62%) | scored |
| 23 | **v006** | 0.674149 | **0.6475** | -0.0112 | 35m32s (59%) | scored |
| 24 | **v008** | 0.674445 | **0.6473** | -0.0114 | 41m17s (69%) | scored |
| 25 | **v003** | 0.673711 | **0.6457** | -0.0130 | 30m57s (52%) | scored |
| 26 | **v004** | 0.672206 | **0.6446** | -0.0141 | 43m16s (72%) | scored |
| 27 | **v002** | 0.674350 | **0.6435** | -0.0152 | 35m02s (58%) | scored |
| 28 | **v010** | 0.670696 | **0.6427** | -0.0160 | 34m37s (58%) | scored |
| 29 | **v001** | 0.674315 | **0.6416** | -0.0171 | 34m02s (57%) | scored |
| 30 | **v033** | 0.660415 | **0.6274** | -0.0313 | 47m22s (79%) | scored |

| Ver | full_valid_wp | Public score | Grader | Outcome |
|---|---:|---|---:|---|
| **v026** | 0.684335 | **no score** | 69m57s (117%) | **SCORE:TIMEOUT** |
| **v027** | 0.683058 | **no score** | 69m57s (117%) | **SCORE:TIMEOUT** |

v026 (871,977 params) and v027 (1,742,717 params) are the two largest
models this project ever built. Both were uploaded and both were killed by
the grader at 69m57s, so neither has a score at all. They are listed
here rather than omitted because they are the clearest evidence for the
capacity result: locally their scores were 2.2 and 10.2 sigma BELOW v028,
and on the grader neither could finish. Two independent lines agreeing
is why the conclusion is that the student should stay near 225k params.

<!-- END GENERATED: public-score ranking -->

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

This was measured from converged runs on 2026-09-29 and later **confirmed against
the public board**: those seven occupy local ranks 2-8 and public ranks 2-11, with
no ordering preserved. The noise band is real and the public test set cannot
resolve differences inside it.

## v011: the recipe that worked

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

## Distillation works here too, and smaller wins again (v016, v017)

Both distilled students beat the teacher they learned from — and the **smaller** one
won, reproducing dozken's result on a different codebase:

| | params | best_val (subset) | full_valid_wp | vs seed | σ |
|---|---:|---:|---:|---:|---:|
| v011 (the teacher) | 192,258 | 0.689479 | 0.679986 | +0.0058 | +36.5 |
| **v016** distilled student | 192,258 | 0.690937 | **0.681753** | +0.0076 | **+47.6** |
| **v017** distilled student | **116,546** | 0.692203 | **0.681866** | +0.0077 | **+48.3** |

**v017 is 0.61× the teacher and scores marginally higher, at 60 % of the inference
cost.** That is distillation regularisation, and it is the same effect dozken
measured publicly: from an 872k teacher, their 436k student scored 0.6879 and their
**smaller** 314k student (0.36×) scored 0.6888. Two independent implementations,
same conclusion — a distilled student should be sized *below* its teacher, not at
it.

Distillation is worth **+0.0019 over the teacher** at equal size and **+0.0019 at
0.61× size with 40 % less inference cost**, which is a rare free lunch. It is also
the only lever so far that works *without* fixing the loss — and that is exactly
why v019/v020 matter: on a correctly specified objective the distillation gain
should compound rather than merely survive.

## v023 was our best entry for a while: the smallest student, and it won

> Superseded by **v028** (0.6585), which is this same architecture with
> `distill_weight` 0.5 → 1.0. Kept because the size finding below is still the reason
> v028 is cheap to serve.

`v023` is `hidden 128` — **225,922 parameters, 51.8% of dozken's 435,943-param teacher** —
on his architecture, with his model as the teacher. It scored **0.6568 public**, our best at
the time, and it beats `v022` (321,602 params, 42% larger) on the public board while running **10%
less** of the time budget.

| | params | % of teacher | local | public | grader |
|---|---:|---:|---:|---:|---:|
| **v023** | **225,922** | 51.8% | 0.684681 | **0.6568** | 45m36s (76%) |
| v022 | 321,602 | 73.8% | **0.685228** | 0.6565 | 50m46s (85%) |
| v024 | 321,602 | 73.8% | 0.684818 | 0.6560 | 50m17s (84%) |

**Read this carefully, because local and public disagree on the ordering.** Locally v023
is *third* — 0.000547 below v022, which is inside the ±0.000317 noise floor, so the three
are a statistical tie. Publicly v023 is *first*. The honest reading is that v022, v023
and v024 are indistinguishable in quality, and v023 is simply the cheapest of the three.

That is still a real win, and it retires the compression hypothesis in the *useful*
direction. Compression did not cost anything (it was never going to, being a null) but it
bought **40.12 µs/row → 76% of budget**, clearing even the retired 75% margin outright.
Smaller student, same score, more headroom. This is dozken's distillation-regularisation
effect, and the practical rule is: **on this teacher, go smaller.**

v024 (batch 64) confirms its own refutation publicly at 0.6560 — inside noise of v022,
as predicted. Its value was training speed: 2.39 h against 5.31 h.

## The offset is now measured on fourteen clean submissions

```
mean -0.0280    range [-0.0288, -0.0272]    std 0.0005
```

v020's contaminated −0.0305 remains outside the band, which is how it was caught. With
std 0.0005 the mapping is reliable enough to aim with, and it is symmetric: **no local
gain is free.**

The two most recent submissions are the strongest evidence for it, because they are
the first pair where a local prediction transferred almost exactly:

| | local | public | offset | predicted Δ | delivered Δ |
|---|---|---|---|---|---|
| v028 | 0.686510 | **0.6585** | −0.028010 | +0.001829 | **+0.0017** |
| v029 | 0.684683 | 0.6568 | −0.027883 | +0.000002 | **0.0000** |

v028's offset lands on the mean of −0.0280 to the digit, and it was a +11.5σ local
result that came back as +0.0017 public against +0.0018 predicted. v029 is the
complementary case: a local null of +0.000002 came back as *exactly* zero, to four
decimals. **A null replicates off the validation set too**, which is the result that
makes the positives trustworthy — a metric that only reproduces your wins is not a
selection signal.

## Where we stand

```
ours v028  0.6585   <-- our best
ours v025  0.6580   our 2nd, and a local REGRESSION
dozken     0.6588   (distil160)
#1         0.6677
```

We are **level with dozken** (0.0003 behind his best) and **0.0092 off first**. That
is up from 0.0109 off first at v023, and the whole of that gain came from one scalar:
`distill_weight` 0.5 → 1.0.

**v025 is the one to understand.** It is our second-best public result while being a
−1.5σ local *regression*, and the reason is the offset finding below: at 5 epochs its
offset moved +0.0024 in our favour. Anyone reading only the `full_valid_wp` table would
have discarded it. Two runs did not, both scored above 0.658, and both cost nothing
extra to serve (46m07s = 77% of budget).

Full public ranking of all 27 scored submissions, including the two timeouts, is in the
**Ranked by public (leaderboard) score** table below.

## Three results that closed open questions

**v028 — the distillation weight is the live lever.** Same architecture as v023,
225,922 params, one number changed. +0.001829 local (+11.5σ against a 0.000317
threshold) and +0.0017 public. Only two points on this curve have been sampled (0.5
and 1.0) and it is the steepest thing we have found, so the obvious next step is 1.5
and 2.0.

**v026 and v027 — capacity is not the lever, now measured rather than assumed.** The
student had never been larger than its teacher, so we tested it:

| ver | hidden | params | full_valid_wp | vs v023 |
|---|---|---|---|---|
| v023 | 128 | 225,922 | 0.684681 | — |
| v026 | 285 | 871,977 | 0.684335 | −2.2σ |
| v027 | 415 | 1,742,717 | 0.683058 | −10.2σ |

Monotonically **worse** as the model grows, and 4× the teacher scores 10σ below a
half-size student. This matches dozken's own 872k ≈ 436k observation, but it is now
measured on our architecture rather than inferred from his. The student/teacher size
ratio is the thing that matters, and v023 already sits at the useful end of it.

**v029 — idea 5 is a dead null, confirmed publicly.** Reweighting the distillation
term by `abs(clip(teacher))` instead of the metric weight moved the local score by
+0.000002 and the public score by 0.0000. Teacher weighting is saturated, so **the
remaining 0.0092 to the leader is not a loss-shaping problem** — which retires idea 5
and points the search at the teacher and the schedule instead.

## The public board has now settled what local could not

All of v011-v015 are scored publicly, and they resolve three questions the local
metric had left open:

| | local full_valid_wp | public | Δ |
|---|---:|---:|---:|
| **v014** 2×192 proj64 (379k) | 0.679676 | **0.6517** | −0.0280 |
| **v011** 2×128 (192k) | 0.679986 | 0.6515 | −0.0285 |
| **v015** v011 weights, reset served | — | 0.6515 | — |
| **v012** reset 2k | 0.678784 | 0.6512 | −0.0276 |
| **v013** reset 4k, **no mask** | 0.676198 | 0.6490 | −0.0272 |

**1. Local validation IS predictive once differences exceed the noise band.** These
four order almost identically on both metrics (v014 overtakes v011 publicly; v012
and v013 follow). The earlier "no correlation" finding was real but only for the
seven versions packed inside ±0.0013 — inside that band nothing transfers, and
*outside* it, correlation is essentially perfect. The noise floor is not a
nuisance; it is the boundary of what local validation can tell you at all.

**2. The reset is a TRAINING-side lever, not an inference one.** v015 shipped
`state_reset=4000` and v011 shipped `state_reset=0`. They scored **0.6515, identical
to four decimals** — despite per-row predictions differing by up to 1.466. So
applying the reset at inference changes what the model emits without changing how
good it is. The reset must be present *during training* (v013 without the mask
beats the plain baseline 0.6490 vs 0.6435, +0.0055) but whether it is also applied
when serving is score-neutral. That is a more interesting result than the bug was:
it says the mechanism is shaping what the model learns, not conditioning its
inference.

**3. The reset carries more of the gain than the mask, publicly.** Decomposed
against the plain baseline (v002, 0.6435):

| | public | Δ |
|---|---:|---:|
| + reset 4k only (v013) | 0.6490 | **+0.0055** |
| + reset **and** mask (v011/v015) | 0.6515 | +0.0080 |

Locally the split was the other way round — reset +0.0020, mask +0.0038 on top —
so **local and public disagree about which lever matters.** Both levers are
positive; the weighting between them is not portable.

**4. v014 is our best public entry, and it is the one I marked Do not ship.** Its
local p99 projected 83 min against a 70-min kill, so I flagged it. It ran in
**40m12s (67 % of budget)** and scored **0.6517**, beating v015. dozken's NOTES warn
of exactly this — their same architecture went from 64m04s to *killed* within ten
hours, so run-to-run grader variance is enormous and a pessimistic local p99 is not
a hard blocker. This is the second time in this repo that treating a local latency
tail as a shipping blocker would have discarded the best model we had.

## Attribution: both levers are real (v013 settled it)

v011 changed the soft mask **and** the state reset together, so for a day the
+0.0058 had no owner. **v013** removed exactly one of them and settled it:

| | full_valid_wp | vs seed mean | σ |
|---|---:|---:|---:|
| seed baseline (v002/v005/v006) | 0.674179 | — | — |
| **v013** reset 4k, **no mask** | 0.676198 | +0.0020 | **+12.7** |
| **v011** reset 4k **+ mask** | 0.679986 | +0.0058 | **+36.5** |

- **the reset alone** (v013 − seed) = **+0.0020**
- **the mask on top of the reset** (v011 − v013) = **+0.0038**

So the periodic state reset is a real +12.7σ effect and the soft mask adds another
+23.8σ on top. I had said the mask was carrying it and the reset was inert — that
was wrong, and it came from reading mid-training eval points, the exact mistake
`AGENTS.md` §11.4 records. Both levers earn their place, and the mask is the larger
of the two.

### The other two results

- **v012 (reset 4k → 2k) = 0.678784, i.e. 0.0012 _worse_ than v011.** The shorter
  horizon does not help; 4,000 is better. And this refutes the premise v012 was
  built on — I justified it as "dozken's reset2k outscores their reset4k", but we
  have still never seen a public score for their reset4k. That claim was an
  inference presented as fact.
- **v014 (192k → 379k params) = 0.679676, a dead heat with v011 (−0.0003, 1σ).**
  Capacity is *neutral* here, not harmful — which revises v004's old finding that
  width hurt. Under the v011 recipe, doubling width buys nothing, and it costs
  **1.5× the latency** (41.50 vs 27.75 µs/row). So v011 strictly dominates v014.

**v014 is the most useful negative result for the next phase.** dozken sit 0.0073
ahead of us, and our own best pure-capacity model ties ours. So their gap is not
width — consistent with their NOTES: *"reaching #1 needs a genuinely different BASE
model, not more averaging."* Our soft mask is the one base-model difference we have
that they lack, and the teacher labels are already built to exploit it.

## The failures

- **v003 (aux MSE + EMA) is worse at -2.9 sigma.** The auxiliary weighted-MSE
  term actively hurt the correlation objective.
- **v004 (2x192 + input projection) is much worse at -12.4 sigma.** Doubling the
  width and adding an input projection is a clear regression.
- **v010 (6 epochs instead of 3) is worse at -21.9 sigma.** Doubling the
  schedule to 39,780 steps from 19,890 made the model *worse*, at
  `full_valid_wp` 0.670696 against a seed mean of 0.674179.

### v010 rules out the obvious alternative explanation

This is the most useful negative result in the batch. v011 gains +0.0058, so the
natural question is whether it simply trained longer. v010 answers that directly:
**twice the training steps made things worse by 0.0035**, a gap of 33 sigma
against v011's gain. So v011's win is not extra optimisation -- it is the soft
mask and the state reset.

It also bounds the two levers against each other. v010 and v011 share the same
architecture and the same 3-epoch schedule; they differ in the objective. That
makes v011's gain attributable to the objective change rather than to capacity or
training time, which is the claim worth carrying forward.

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

## Every run is now submitted: the local/public correlation

All 11 versions have public scores. Local `full_valid_wp` ranks them
**Pearson +0.699, Spearman +0.491** against the public board -- a real but weak
relationship, and the weakness is entirely inside the noise band.

| | local rank | public rank |
|---|---:|---:|
| v011 | 1 | **1** |
| v009 | 2 | 2 |
| v008 | 3 | 6 |
| v005 | 8 | 3 |
| v001 | 5 | 11 |

`v011` is first on **both** -- the one effect big enough to survive the transfer.
Everything else reshuffles freely: v008 falls 3 places, v001 goes from 5th to
last, and v005 climbs from 8th to 3rd on a public/local gap of 0.0044, which is
well inside what 200 validation sequences of noise can produce.

So the honest reading, now tested rather than assumed:

- **A local effect of ~36σ transfers.** v011's +0.0058 became +0.0032 public over
  our own previous best. Real, and worth having.
- **A local difference inside the noise band does not transfer at all.** Judging
  inside that band is a coin flip. This is why the noise floor had to be measured
  before the table meant anything.
- **The earlier caveat was half right.** It was right that the public board is the
  only trustworthy ranker. It was wrong to imply v011's win was unconfirmed -- it
  confirmed, cleanly.

## Where we stand, and what the gap is

| | public | rank | params |
|---|---:|---|---:|
| organiser baseline | 0.5935 | 17/100 | — |
| **ours, best submitted (v014)** | **0.6517** | — | 379k |
| ours (v011 / v015) | 0.6515 | — | 192k |
| *ours, unsubmitted (v017)* | *~0.653 est.* | — | *117k* |
| dozken `reset2k` | 0.6569 | 25/100 | 436k |
| dozken `distil_ens` | 0.6585 | 27/100 | 436k |
| **dozken `distil160`** | **0.6588** | 27/100 | 314k |
| *dozken's #1 (`ne_onnx`)* | *0.6622* | *1/100* | — |

All three third-party entries are by the same author (`dozken`), taken from
`github.com/dozken/wunder` and submitted as-is. They are recorded under
`_dozken` in `exploration/leaderboard.json` and are **not ours**.

**`distil160` and `distil_ens` are separate submissions, and the earlier label was
wrong.** The entry we uploaded as `submission_distil_ens.zip` was annotated
`dozken/2026-09-17_distil160`, but it is **distil_ens** — GRU 192×2 proj64, 435,714
params. Two independent checks confirm it: its 61m06s runtime matches the 436k
model, and its 0.6585 equals their stated held-out 0.6879 minus the 0.0295 offset
to four decimals. The genuine `distil160` (GRU **160**×2 proj64, 314,338 params) was
submitted separately on 2026-09-30 and scored **0.6588**.

That resolves a question this file previously left open, and the answer is
instructive: **the smaller student is both better and faster** — 314k at 0.6588 in
55m16s beats 436k at 0.6585 in 61m06s. Same teacher, same distillation, less than
half the inference cost. That is the distillation-regularisation effect their NOTES
predicted, now confirmed on the public board rather than only on their held-out.

**The local→public offset is now trustworthy across both authors.** Our own four
scored runs average −0.0288; theirs is −0.0295. Their held-out 0.6888 therefore
projects to ~0.660 on our scale, and the actual public figure is 0.6588. Local
scores are now predictable to roughly four decimals, so a candidate can be screened
before spending a submission.

Every one of our 11 entries beats the organiser baseline; the programme is worth
**+0.058 public**, matching the +0.057 measured locally.

**The gap to their best is 0.0071 public, and to their #1 is 0.0105.**

The two levers visible in their names are worth reading as a roadmap:

- **`reset2k` (0.6569)** is a periodic state reset at **2,000** steps. This is the
  lever we independently validated, and **v012 tested it: 4,000 beats 2,000** by
  0.0012, so the shorter horizon does not help. Note our original premise -- "their
  reset2k outscores their reset4k" -- was an inference presented as fact; we have
  still never seen a public score for their reset4k.
- **The distillation entries (0.6585 / 0.6588)** are two distinct levers:
  - **Distillation** — teacher trained without the latency limit, compressed into
    one budget-sized student. Inference cost unchanged. This is the natural Phase 6
    move from the original roadmap and the larger of the two jumps left.
  - **Ensemble** — several models combined at inference, cost multiplied by the
    member count. Their 2-graph ensemble was killed at 69m59s, so an ensemble has a real
    ceiling and distillation is the only route past it.

**The time limit is enforced, and the real kill is 70 minutes, not 60.** The docs say
60, but dozken's 2-graph ensemble was killed at 69m59s. An earlier version of this
file claimed the limit "is evidently not enforced", based on a 61m06s run that
scored. That was wrong: 61m06s is comfortably inside a 70-minute kill. Nothing
about the budget changed -- it is a 70-min hard wall with a 60-min documented
target -- but the earlier claim would have justified planning against a phantom
10 minutes of headroom.

## What distil_ens actually is (from the author's own NOTES)

Read from `2026-09-17_distil_ens/NOTES.md` and `2026-09-17_distil160/NOTES.md`.
This is the most informative thing found outside our own runs, because it is the
same problem solved with the latency wall biting.

| model | graphs | held-out | params | outcome |
|---|---:|---:|---:|---|
| teacher = mean(reset2k, reset4k) | 2 | 0.6879 | 872k | **TIMEOUT 69m59s** |
| distil_ens | 1 | 0.6879 | 436k | public 0.6585 (ours) |
| distil160 (160 hidden) | 1 | **0.6888** | 314k | their best |
| reset2k | 1 | 0.6864 | 436k | public 0.6569 (ours) |

Distillation reproduced the 2-graph teacher in **one graph**, and the arithmetic
confirms which model we actually submitted: 0.6879 - 0.0295 = 0.6584, we scored
**0.6585**. Likewise 0.6864 - 0.0295 = 0.6570 against our **0.6569**. Their
held-out to public offset is -0.0295, which now lets us predict our own public
score from a local one.

Four facts from those notes that shape our roadmap:

1. **Averaging is exhausted.** Greedy forward selection over 12 checkpoints
   saturates at 0.6887, only +0.0009 over the 2-member teacher and *not resolved*
   (P 0.840). Their words: "our model pool has no more to give by averaging --
   every member is the same family." This killed an ensembling plan before we spent
   time on it.
2. **Smaller students beat their own teachers.** distil160 (314k params) scored
   0.6888 against a 0.6879 teacher, the usual distillation-regularisation effect.
   Their next step is to distil again into hidden 128 at ~18 us/row.
3. **Distillation is the only lever that decouples teacher cost from serving
   cost.** The teacher may be arbitrarily slow because it never runs at inference.
   That is why the 2-graph teacher is worth building even though it cannot ship.
4. **Their teacher has no soft mask.** The reset lever we validated independently
   is theirs too, but the P(is_scored) loss weighting is ours, and it was worth
   +0.0058 (36 sigma) locally at zero inference cost. A teacher built from masked
   models should start from a stronger teacher, and that strength propagates into
   the student. **That is the axis to beat them on.**

Their own stated reason for stopping: "reaching #1 needs a genuinely different BASE
model, not more averaging." We already have one they lack.

## Ship candidate: v015, not v011

**`v015` is what should be submitted next, and it is packaged and gate-clear.**

It carries v011's exact weights with the state reset **actually served** — v011
trained with `state_reset=4000` but shipped `state_reset=0`, because the export
path read a checkpoint key that was never written and silently defaulted to zero.
So the 0.6515 above is not v011's real score; it is v011 served in a regime it was
never trained for.

| | |
|---|---|
| latency (idle, 3 runs) | **27.75 µs/row → 29.5 min, 49.2 % of budget** |
| contract / determinism / parity | PASS (2.15e-07) |
| ZIP | 2.16 MB, `state_reset=4000` verified inside the archive |

v013 proved the reset is worth **+12.7σ** on its own, so v015 should land
meaningfully above the broken 0.6515 — plausibly 0.653–0.658, which would put it
level with dozken's `reset2k` (0.6569). That is a genuine one-shot read on
whether the fix lands as the attribution predicts.

v011's submitted bundle is left untouched at 0.6515 rather than patched in place:
a defective artifact that was actually submitted is exactly the record
`AGENTS.md` §2.5 exists to preserve.

## Running now: v016, the distillation student

`v016` is a fresh GRU 2×128 trained against **212,140,000 rows** of soft targets —
the raw predictions of the v011/v015 weights replayed over the whole train shard
on their own 4,000 reset grid (`tools/make_teacher_labels.py`, 135 min, sha256 in
the manifest, verified against the served row-by-row stream to 4.9e-04).

`distill_weight=0.5` sits on top of the same soft-masked global-pooled loss, so the
student is steered toward what the teacher believes on the rows the metric scores.
**0.5 matches dozken's own weight**, so this is like-for-like rather than tuned in
our favour.

Sized at the teacher's own 192k because v014 showed 379k is a dead heat here at
1.5× the latency, and both of dozken's students were smaller than their teacher.
Inference should stay at 27.75 µs/row.

A second teacher member (v012, reset 2k) is being generated in parallel, which
would give a two-member teacher matching dozken's construction.

**Why distillation is the right remaining lever, in one line:** our best
pure-capacity model (v014) ties our baseline, so the 0.0073 gap is not width —
matching their own note that reaching #1 needs a different *base* model, and the
soft mask is the one we have that they lack.

## The p99 latency gate was wrong five times out of five

Both distilled students were initially packaged **NOT-READY** on a p99 latency
failure. Before believing that, I checked it against every submission we have
actually made:

| | local p50 → proj | local p99 → proj | **grader actual** | p99 over-shot |
|---|---|---|---:|---:|
| v011 | 28.00 → 29.8m | 41.58 → 44.2m | 30.2m | 1.46× |
| v012 | 28.17 → 30.0m | 55.63 → 59.2m | 29.8m | 1.99× |
| v013 | 27.96 → 29.7m | 48.21 → 51.3m | 28.6m | 1.79× |
| v014 | 41.50 → 44.1m | 78.25 → 83.2m | 40.2m | 2.07× |
| v015 | 27.75 → 29.5m | 47.71 → 50.8m | 30.5m | 1.66× |

**p50 projects the grader within 5% every single time. p99 over-projects by
1.5–2.1× every single time.** v014 is the case that mattered: the p99 gate marked
it DO NOT SHIP at 83 min projected, and it ran in 40m12s and became our best public
entry. dozken's NOTES independently record their own identical graph going from
64m04s to *killed* within ten hours — the grader's variance dwarfs any local tail,
and a p99 measured on a contended local box is noise.

**The gate now tests p50 only.** p99 is still recorded, as a diagnostic. Repackaged
afterwards, v016 and v017 both come back **STATUS: PASS**.

| | params | p50 | grader p50 | % of budget |
|---|---:|---:|---:|---:|
| v016 | 192,258 | 28.0 µs/row | 29.8m | 50 % |
| **v017** | **116,546** | **22.7 µs/row** | **24.1m** | **40 %** |

This is the second time in this repo that a local latency reading would have
discarded the best model available (the first was v004, which we submitted anyway and
which scored 0.6446). Both times the estimate was pessimistic and the grader
disagreed. The gate should have been p50-only from the start.

## Latency is a solved problem

> Read with the 2026-10-02 update below, which makes the same point harder.

Worth recording because it was treated as a blocker all night and turned out not
to be one. All 11 of our bundles finished, in **30m12s to 43m16s -- 50 % to 72 %
of the 60-minute budget** -- including the five whose local gates had been
measured under 7-way contention and marked "unverified".

### And again on 2026-10-02, with a worse miss

`v028` was packaged and marked **NOT-READY**: local p50 67.75 us/row, projected
**72m06s = 120 % of budget**. It was submitted anyway and ran in **46m52s = 78 %**,
completing normally. The projection over-shot by **35 %**, because the p50 was
taken while `v027` was training on the same cores.

`v029` is the same story in miniature: predicted 59m18s, ran 44m27s (74 %).

This is now six submissions out of six where a contended local p50 **over**-projected.
The pattern is consistent enough to act on:

- **Never gate on a p50 taken while other work is running.** Re-measure idle, or
  don't gate. The 1.62x factor compounds with contention instead of replacing it.
- **A latency `NOT-READY` is not evidence the model is unshippable.** It is evidence
  the *measurement* was bad. v028 was the best model in the project and the gate
  said the opposite.
- **p99 over-projects even harder and should not be read as a runtime estimate** at all.

Every contended measurement **over**-projected. v005 measured 52.25 us/row under
contention, projected 55.6 min, and actually took 36m56s. v004 was marked
DO NOT SHIP on a 47.2 min p50 projection and ran in 43m16s for 0.6446. The 1.62x
slowdown factor calibrated from `gru_v000`'s 2x128 model is not safe to apply to
anything wider, and a contended local p99 is not a hard blocker.

The one caveat is that a *larger* model can still hit the wall: dozken measured the
same architecture going from 64m04s to killed within about ten hours, with memory
flat at 627 MB throughout, so it is purely wall time. The practical conclusion is
that model quality, not inference budget, is what limits this problem -- within the
2x128 envelope we have been using all along.

## What the other competitor's public notes settled for free (2026-10-02)

dozken publishes a full experiment log and a resume file. Reading them changed the
plan substantially — mostly by killing two ideas I was about to spend GPU hours on.

**The metric changed on 2026-09-15 and it inverted his ranking.** Per-sequence WP
was replaced by Global WP, and the ordering flipped hard:

| model | old per-seq | new global |
|---|---:|---:|
| reset4k | 0.6598 | **0.6823** |
| seq_5ep | 0.6671 | 0.6704 |
| **chunk4k** | **0.6835** (best) | **0.6498** (worst) |
| proj128_v2 | 0.6836 | 0.6487 |

Two levers I had queued are dead, and both were on his resume file's "next runs"
list rather than in the experiment log:

- **Longer TBPTT is a liability, not the win.** `chunk 4000` looks like +0.0164 on
  the resume file, but that number is pre-metric-change. Under Global WP it is
  −0.0370 against reset4k at P(>0)=1.000. We already train at `chunk_len 1000` and
  that was right for a reason we had not understood. **Do not raise it.**
- **`proj 128` did not materialise.** The same notes claim the 64-wide input
  projection destroys ~0.014 WP of linearly available signal; the measured run says
  −0.0032. We use `input_proj 64`. Leave it.

**The one lever that survived is loss diversity.** His `reset2k + reset4k` ensemble
is **+0.0056 global, P(>0)=1.000, 95% CI [+0.0019, +0.0100]** — and the stated reason
is that the members use *different losses* (`GlobalPearsonLoss` vs the older
`SequencePearsonLoss`). Same-loss pairs add nothing.

That ensemble is also the failure that makes it actionable. He shipped it as two
ORT graphs and it was **killed at 69m59s**, so he concluded "2-graph ensembles are
out" and moved on. That conclusion is about *sessions*, not models: we fuse members
into a single graph, paying the fixed per-session cost once.

Also worth carrying over, both measured by him and both consistent with what we
already do: averaging members must be on **raw** outputs (z-scoring them inflated
output std 1.55× and pushed 14.30% of scored rows past the ±2 clip — that artifact
is what produced his earlier, wrong "ensembling is dead" reading); never rescale
predictions at all (×1.5 costs 0.0038, ×2.0 costs 0.0176, because the metric
clips); and do not recentre per sequence (0.6823 → 0.5756).

### Three things we are doing about it

1. **A loss-diverse ensemble fused into one graph** (`tools/score_ensemble.py`).
   Members **v028** (`distill_weight 1.0`, 0.686510) and **v029**
   (`distill_weight 0.5` + tail-weighted `gamma 2.0`, 0.684683): identical
   architecture and hidden width, so the fusion is cheap, but genuinely different
   loss functions. This is the pair shape his +0.0056 came from, and it is
   shippable where his was not.
2. **The distillation-weight curve, sampled properly** (`v030` = 1.5,
   `v031` = 2.0). Our steepest lever, and until now only two points existed on it.
   Two points cannot distinguish a curve still climbing from one that has turned
   over, and the follow-up action is completely different in each case.
3. **His 0.6879 two-graph ensemble, downloaded and kept**, as a candidate
   *teacher*. Distillation is the only route to ensemble accuracy that fits in one
   graph, and a teacher never runs at inference, so it is free. Held with an honest
   prior attached: his own `distil_ens` (0.6585) did **not** beat his `distil160`
   (0.6588), so a better teacher demonstrably failed to transfer through
   compression for him. Worth one run, not worth assuming.

### The v026 timeout, and what it settles

`v026` was uploaded and **killed at 69m57s — 117% of the 3600s limit**, no score. Our
local projection had said 132.1 min (220%).

The 1.62× grader factor turns out to **hold**; the error was ours. Applying it to the
observed 106.5 µs/row grader rate implies a quiet local p50 near 65.8 µs/row, not the
124.17 we measured while four other jobs were saturating the box. So the gate was
directionally right about v026 and wrong only about magnitude — while the same gate
called v028 NOT-READY at a projected 120% and v028 ran at 78% and scored our best.

The actionable rule is narrow and worth stating: **a contended p50 is not a
measurement.** Re-measure idle or do not gate.

## The offset is not a constant, and it moved us 0.0018 (2026-10-02)

`v025` scored **0.6580 public** — and that is the most informative result in the
table, not because of its rank but because of its **offset**.

| ver | local | public | offset | epochs | batch |
|---|---:|---:|---:|---:|---:|
| v022 | 0.685228 | 0.6565 | −0.028728 | 3 | 16 |
| v023 | 0.684681 | 0.6568 | −0.027881 | 3 | 16 |
| v024 | 0.684818 | 0.6560 | −0.028818 | 3 | 64 |
| **v025** | **0.684438** | **0.6580** | **−0.026438** | **5** | 64 |
| v028 | 0.686510 | 0.6585 | −0.028010 | 3 | 16 |
| v029 | 0.684683 | 0.6568 | −0.027883 | 3 | 16 |

Five 3-epoch runs share −0.028264 ± 0.000419. v025 sits at **+4.4σ** from that.

**The control is unusually clean.** v024 is the *identical* recipe to v025 at 3
epochs and sits normally at −0.028818. So batch 64 is exonerated, and training
length is the only structural difference. Measured against v024 specifically, the
shift is **+0.002380**.

### What this does to how we select models

**`full_valid_wp` is a biased selector across different epoch counts, and the bias
runs against the longer run.** v025 was a −1.5σ local *regression* and was nearly
discarded on that reading — it then beat v023 publicly by **+0.0012**. Local
selection did not merely fail to find the win, it actively mis-ranked it.

Everything we concluded about epochs before this was measured on local score alone
and is therefore suspect in a specific direction. Notably v010 ("6 epochs not 3")
was rejected locally at 0.670696 — that verdict deserves re-examination, and the
honest reading is that we do not currently know whether it was wrong.

### Acting on it

`v032` = **v028's exact recipe at 5 epochs**. v028 has the best local score in the
project (0.686510, +11.5σ) and carries the standard offset; v025 shows a +0.0024
public uplift from training longer. If the two compose, the prediction is
**0.6585 + 0.0024 = 0.6609**, against a 0.6677 leader we currently sit 0.0092 back.

The hypothesis is written to be falsifiable rather than merely hopeful: if `v032`
lands at v028's 0.6585, the epoch effect did not transfer to this recipe, which
would mean the shift was specific to v025's *lower* distillation weight — and the
two levers would interact rather than add. That is a real possibility and a
distinguishable one.

The parameter count is unchanged, so latency is inherited from v028: nothing about
this costs anything to serve.

## The loss-diverse ensemble: refuted, and the reason is the interesting part

`tools/score_ensemble.py` scored **v028 + v029** on all 1,873 sequences — 3,528,273
scored rows, the official number, reference metric, raw-mean (never z-scored).

| | t0 | t1 | MEAN | vs recorded |
|---|---:|---:|---:|---|
| v028 alone | 0.682239 | 0.690660 | **0.686449** | 0.686510 (Δ 6e-5) |
| v029 alone | 0.679674 | 0.689322 | 0.684498 | 0.684683 (Δ 2e-4) |
| **MEAN of 2** | 0.681808 | 0.690743 | 0.686275 | **−0.000174** |

**The harness validates itself.** Each member reproduces its own recorded
`full_valid_wp` to 6e-5 and 2e-4 — float32 drift over 20,000 recurrent steps — so the
−0.000174 is a real measurement, not a pipeline artifact.

**The result is a null**: −0.000174, well inside the ±0.000317 noise floor, against
dozken's **+0.0056 at P(>0)=1.000**. That is a refutation, not an uninformative null.

### Why — and this is the transferable part

**Our two members were not actually diverse.** Their scores differ by 0.00195; his
differ by 0.0041. More decisively: v029 differs from v028 *only* by `gamma=2.0`
tail-weighting on the distill term — and **that was already measured as a dead null
in its own right** (+0.000002, idea 5).

A knob that provably does nothing cannot manufacture ensemble diversity. We applied
his mechanism — members trained with *different losses* — to two members whose
losses were, in effect, the same. The ensemble faithfully averaged two near-copies
and landed exactly where averaging near-copies lands.

**The lesson:** a loss-diverse ensemble needs members whose individual optimisation
*genuinely diverges*, evidenced by a real spread in their member scores. Varying a
hyper-parameter is not enough. If we revisit this, the diversity has to be designed
in and verified by member score spread *before* the ensemble is scored — not
assumed from a differing config.

The fusion tooling is unaffected and remains correct: `tools/build_ensemble_graph.py`
still builds a bit-exact single graph, verified `0.000e+00`. It is the ensembling
that failed, not the fusing. The single-graph fusion remains the thing that makes a
*genuinely* diverse pair shippable where dozken's two-session version timed out — it
just has nothing to carry yet.

## Capacity is now settled by the grader, not just by us

`v027` was uploaded and **timed out at 69m57s** — the same second as `v026`.

| ver | params | hidden | local | outcome |
|---|---:|---:|---:|---|
| v023 | 225,922 | 128 | 0.684681 | runs in 46m52s (78%) |
| v026 | 871,977 | 285 | 0.684335 | **TIMEOUT** 69m57s |
| v027 | 1,742,717 | 415 | 0.683058 | **TIMEOUT** 69m57s |

Two timeouts landing on **the same second** identifies that as the platform's kill
point rather than either model's natural runtime. So neither timeout tells us how far
over budget they were — only that they are over. dozken independently records a ~70
minute hard limit and lost an ensemble of his own at 69m59s.

The conclusion is now on firmer ground than before, because the two lines of evidence
agreed *before* either was uploaded:

- **Locally**, the score falls monotonically with size — 0.684681 → 0.684335 →
  0.683058 — so 4× the teacher is 10.2σ *below* a half-size student.
- **On the grader**, both oversized models cannot finish, while the 225,922-param one
  runs in 78% of budget.

Neither line alone would have been decisive, since a bad local score could be noise and
a timeout could be a flaky platform. Together they close the question: **the student
should stay around 225k parameters.** Everything above roughly 450k is unshippable
here, which is also why `v023`'s compression finding mattered so much — it was not
merely free, it was the only affordable size.

One loose end, recorded honestly: v027 was uploaded from the *incomplete* bundle
committed in `8ba7a50` (no `metadata.json`, zip missing the provenance checkpoint).
The missing checkpoint is a provenance artifact and does not affect serving, so the
timeout remains a valid measurement of the model rather than an artifact of the
partial build.

## v030 and v031 are ready to submit — and the gate said no (2026-10-02)

The distillation-weight sweep finished, and it is **monotone across all four points**:

| ver | distill_weight | full_valid_wp | vs v028 | σ |
|---|---:|---:|---:|---:|
| v023 | 0.5 | 0.684681 | −0.001829 | −11.5 |
| v028 | 1.0 | 0.686510 | baseline | — |
| v030 | 1.5 | 0.686919 | +0.000408 | +2.6 |
| **v031** | **2.0** | **0.687376** | **+0.000865** | **+5.4** |

**v031 is a new best model**, outside the ±0.000318 threshold, and the curve has not
turned over — so 3.0 and 4.0 are the obvious next probes. There is still teacher
signal to absorb: dozken's teacher is 0.690168 local and v031 is at 0.687376.

At the −0.028 offset, v031 projects to roughly **0.6601 public**, versus v028's 0.6585.

### Both bundles are marked `NOT-READY`. Both are wrong.

The packager failed both on latency: 62.1 and 63.9 minutes against a 60-minute limit.
That verdict is a measurement artifact, and the evidence is not subtle:

- **The graph is identical to v028's.** Same 225,922 parameters, same 42 ONNX nodes,
  same 16 initializers — verified by loading all three graphs. `distill_weight` is a
  **loss** coefficient: it changes gradients during training and *nothing whatsoever*
  in the exported graph. Same arithmetic per row, so same runtime.
- **v028 is that exact graph and it ran in 46m52s — 78% of budget — and completed.**
- The local p50s that produced the failure (58.33 and 60.04 µs/row) were measured while
  **three other runs were training**, on a 12-core box at load ~10. Per `AGENTS.md`
  §2.6, a contended p50 is not a measurement.

So the table now resolves latency from a **measured anchor**: if a version's graph
signature matches another version that has actually been timed on the grader, that
measured runtime is used instead of a local projection. It takes the **slowest**
measured anchor rather than the first found — an anchor exists to stop us
under-estimating, so it must resolve to the worst case actually observed.

This is not special pleading for these two runs. It is the third time the same
projection has misled us, and the pattern is consistent enough to encode:

| run | projected | actual | outcome |
|---|---:|---:|---|
| v028 | 72.1 min | 46.9 min | completed, 78% — best model |
| v026 | 132.1 min | 70.0 min | TIMEOUT |
| v027 | ~140 min | 70.0 min | TIMEOUT |
| v030 | 62.1 min | **46.9 min expected** | — same graph as v028 |
| v031 | 63.9 min | **46.9 min expected** | — same graph as v028 |

Note the honesty of the record: the same pessimistic projection was *correct* for v026
and v027, which genuinely timed out. The failure mode is not "the gate is too
strict" — it is that it is **only as good as the p50 it was given**.

Both bundles pass every other gate: contract, determinism, parity (max |diff|
3.3e-06 and 2.9e-06 against a 1e-03 tolerance), and size at 2.42 MB of 20 MB.

## v031 is our best entry — but read the attenuation before celebrating (2026-10-02)

The distillation-weight sweep completed and both new bundles were submitted. **v031
is our best public score at 0.6587**, and the sweep was monotone across all four points.

| ver | distill_weight | local | vs v028 | public | vs v028 | grader |
|---|---:|---:|---:|---:|---:|---|
| v023 | 0.5 | 0.684681 | −0.001829 | 0.6568 | −0.0017 | 45m36s |
| v028 | 1.0 | 0.686510 | baseline | 0.6585 | baseline | 46m52s |
| v030 | 1.5 | 0.686919 | +0.000408 | 0.6585 | **+0.0000** | 47m11s |
| **v031** | **2.0** | **0.687376** | **+0.000865** | **0.6587** | **+0.0002** | 45m41s |

### The local curve is promising and the public curve is flat

v030's local gain of +0.000408 transferred as **exactly zero**. v031's +0.000865
transferred as +0.0002 — an **attenuation of 4.3×**. Against a public-split noise floor
of order 0.0005, **v031, v030 and v028 are a statistical tie on the leaderboard.**

So the honest reading is: the sweep is real on local validation (+5.5σ for v031) and
nearly worthless on the board that decides the prize. Two readings, and we cannot yet
distinguish them:

1. **We are approaching the teacher's ceiling.** dozken's teacher is 0.690168 local and
   v031 is at 0.687376 — most of the way there. Further distillation weight has less
   left to absorb, which would mean 3.0 and 4.0 are not worth running.
2. **The public split is a small sample.** 1,970 test sequences against 1,873
   validation sequences, and the two differ by a fixed ~0.028 offset. A +0.0009 local
   gain can simply be invisible at that resolution.

The practical answer is that the *next* distillation-weight probe is far less
attractive than it looked this morning, and I would not spend two runs on it before
v032 reports.

### The measured-anchor rule is now validated three ways, not argued

| run | packager said | anchor predicted | **actual** | gate error | anchor error |
|---|---:|---:|---:|---:|---:|
| v028 | 72.1 min | 46.9 min | **46.9 min** | +25.2 min | **+0.03 min** |
| v030 | 62.1 min | 46.9 min | **47.2 min** | +14.9 min | **−0.28 min** |
| v031 | 63.9 min | 46.9 min | **45.7 min** | +18.2 min | **+1.22 min** |

Three predictions, three completions, no timeouts, worst error 1.22 minutes. Both
bundles were marked `NOT-READY` by the gate and **both shipped fine**. This is the
clearest result of the session, and it cost nothing: the fix was to stop trusting a
local p50 measured on a busy box.

### The offset held for these two

v030 at −0.028419 and v031 at −0.028676 both sit inside the 3-epoch band of
−0.028264 ± 0.000419. So the mapping did **not** drift — which retroactively sharpens
the v025 finding: the +4.4σ outlier was specific to **longer training**, not a general
slow erosion of the local→public relationship. That is testable, and v032 tests it.

## v033 falsified its own precondition, and that is the run working

`v033` finished at **full_valid_wp 0.660415 — 170σ below v028.** Its yaml predicted it
would land "within a few sigma of v028" and made that the explicit condition for the
ensemble being worth running. **It missed by two orders of magnitude.**

What was right and what was wrong:

- **The diversity mechanism was real and verified.** Per-lane offsets spread −3..3 cost
  the global loss 0.276 and the per-sequence loss 0.069 — a 4× difference — while one
  *shared* offset cost both 0.195 identically, which is correct since a shared bias is
  not re-centreable. The two objectives genuinely disagree.
- **The magnitude was not viable.** dozken's working pair had a **0.0041** spread
  between members. This would need **0.027**. Per-sequence pooling cost far more than
  his did: ~0.004 in his hands, 0.027 in ours.

So the ensemble it was built for is off the table. Not submitted, not packaged — there
is no case for shipping a model 170σ down. Recorded here rather than buried, because
the run did exactly what a falsifiable hypothesis is for: it set a precondition, the
precondition failed, and that was discovered by running it rather than by reasoning.

## Where we stand

```
ours v031  0.6587   <-- our best
ours v030  0.6585
ours v028  0.6585
dozken     0.6588   (distil160)
#1         0.6677
```

**0.0090 off first**, and 0.0001 behind dozken. v025's epoch finding is still the most
promising open thread — v032 reports within the hour and tests it directly.

### v033 is packaged on request — every gate passes, and it is still not worth submitting

`v033` has a complete bundle: contract, determinism, parity (5.4e-06 against a 1e-03
tolerance) and size (2.42 MB of 20 MB) all **PASS**, and the latency gate **passed
outright** this time — 53.25 µs/row, a projected 56.6 min, 94% of budget. The README
table will therefore print `Ready to Submit`, and that label needs a guard.

**The model is 170σ below our best.** `full_valid_wp` 0.660415 against v031's 0.687376.
The gate is a necessary condition for shipping, never a sufficient one, and this is the
case that proves it: a bundle can clear every mechanical check and still be a bad model.
`v020` was the earlier version of the same lesson — a contaminated score that passed all
four gates.

A `Ready to Submit` label next to `0.660415` is easy to skim past. Read it with the
score: rank 32 of 33 in the table above, 0.0270 behind v031, and roughly where a
much weaker distillation experiment was. It is packaged because it was asked for and
because a complete artifact is worth having on disk; it is not a submission candidate.

One incidental benefit of packaging it: it produced the first latency measurement
taken on a *quiet* machine for this architecture — 53.25 µs/row against the 58–60 µs
measured under load, which is the last independent confirmation that contention, not
the model, was inflating our numbers all along.

## Capacity is closed on every axis we can afford to test (v034)

`v034` doubled the MLP head — `head_hidden` 192 → 384, 225,922 → 263,362 params
(+16.5%). It was the one place where "make it bigger" was affordable: the head is 16.7%
of per-row compute against the trunk's 83.3%, so this cost ~17% latency where widening
`hidden` cost 2× to 4× and blew straight through the budget.

**It made things worse: 0.686000, which is 0.000510 below v028 at −3.2σ.**

| ver | change | params | full_valid_wp | vs v028 | σ |
|---|---|---:|---:|---:|---:|
| v028 | baseline | 225,922 | 0.686510 | — | — |
| **v034** | **head 192 → 384** | **263,362** | **0.686000** | **−0.000510** | **−3.2** |
| v026 | hidden 128 → 285 | 871,977 | 0.684335 | −0.002175 | −13.7 |
| v027 | hidden 128 → 415 | 1,742,717 | 0.683058 | −0.003452 | −21.7 |

So the answer to "we have 14 minutes of budget left, can we make it bigger?" is now
measured rather than assumed: **no, and it does not matter where you spend it.** Three
axes, three regressions, monotonically worse with size. The yaml had predicted that a
null was plausible and that a worse result "would be consistent and expected" — which is
what happened, and it is why the hypothesis was written down before the run rather than
after.

Combined with v026 and v027, every form of extra capacity we can afford to test has made
the model worse. The useful conclusion is not just "don't widen" — it is that **the
225,922-parameter size is not a compromise forced by the latency budget. It is the
optimum.** v023's compression finding was not merely free, it was load-bearing.

This also settles the direction of the remaining time. The two levers that have ever
moved this model are both *smaller-scope* changes — how hard we distil (v031, +5.5σ)
and how long we train (v025's offset anomaly, still being tested by v032). Neither adds
parameters.

## v033 scored 0.6274 — the offset broke, and that is the real result

| | local | public | offset |
|---|---:|---:|---:|
| v033 | 0.660415 | **0.6274** | **−0.033015** |
| 3-epoch band | — | — | −0.028264 ± 0.000419 |

That is **−0.004751 beyond the band, at −11.3σ.** It is our worst scored entry, and it
is the most instructive failure in the project.

**Mechanism.** v033 was the per-**sequence**-pooling run. Under that objective every
sequence is re-centred against itself, so the model learns a per-sequence level
invariance. The scorer is a **global** weighted Pearson — one correlation pooled over
every scored row — where a per-sequence level offset is *not* free. The objective taught
an invariance the metric does not reward, and it cost **0.0047 of public score beyond
what local validation predicted.**

This is the second independent confirmation that the global objective is the right one.
dozken's log records the same inversion when the organisers swapped the metric on
2026-09-15 — his per-sequence model went 0.6598 → 0.6823 while the globally-pooled one
went 0.6671 → 0.6704 — and we have now measured the cost directly rather than inferring
it.

**It also retires the ensemble idea for good.** v033 was built to be a genuinely diverse
partner after v028+v029 was refuted at −0.000174, and its diversity *mechanism* was
verified real. But a partner scoring 0.6274 against v031's 0.6587 cannot help an
equal-weight mean. The diversity was real; the magnitude never was.

## A verifier bug worth recording: v034's parity gate silently produced nothing

Packaging v034 wrote `gate.parity = null` and `status: NOT-READY`. Every other gate
passed. The cause was **our own tooling**, not the model: `tools/score_solution.py`'s
`offline_predict()` rebuilt `GRUModel` without passing `head_hidden`, so it built a
192-wide head and raised a `state_dict` size mismatch on any run that widened the head.
The parity check died, and the bundle recorded "not verified" rather than "the verifier
could not build the model".

That distinction matters. I had already plumbed `head_hidden` through **both**
construction sites in `train_gru_v034.py` precisely so train and serve could not
diverge — and then missed the third site, the scorer. One line of plumbing per site, and
missing one produces a silent null instead of a loud failure.

After the fix, v034's parity **PASSES**: raw predictions agree to **4.232e-06** against
a 1e-03 tolerance, and the streaming and offline WP scores agree to **1.936e-08**. So the
bundle is sound; only the verifier was broken.

Note the bundle's `metadata.json` still records the null parity and NOT-READY, because
bundles are immutable and re-packaging the same version is not allowed. The verified
result lives here instead.

## v032 is the epoch answer, half in — and it is packaged and ready

`v032` = v028's exact recipe at **5 epochs instead of 3**. It finished at
**full_valid_wp 0.687219**, which is **+0.000708 over v028 at +4.5σ** — the same
architecture, the same `distill_weight` 1.0, one scalar changed.

| ver | recipe | local | public | offset |
|---|---|---:|---:|---:|
| v028 | 3 epochs, distill 1.0 | 0.686510 | 0.6585 | −0.028010 |
| v031 | 3 epochs, distill 2.0 | 0.687376 | 0.6587 | −0.028676 |
| **v032** | **5 epochs, distill 1.0** | **0.687219** | **unsubmitted** | **?** |
| v025 | 5 epochs, distill 0.5 | 0.684438 | 0.6580 | **−0.026438** |

**Half the answer is in.** Longer training is worth **+0.000708 on local validation** —
a real effect, +4.5σ, and the first time epochs has moved local score in our favour.

**The other half is the interesting one and only submitting can answer it.** v025 was
*also* 5 epochs, and it carried an offset of −0.026438 — **+0.0024 better than the
3-epoch band**, at +4.4σ, invisible to local validation. That is the largest
public-score effect we have found, larger than any architecture or loss result.

- If v032 inherits it: **~0.6616** (0.6585 + 0.0007 local + 0.0024 offset)
- If it does not: **~0.6592**

This is the cleanest remaining experiment in the project: one upload, one scalar already
known locally, and a binary answer about whether the epoch effect is a property of
*training length* or was specific to v025's lower distillation weight. The two levers
either compose or they interact, and we have never found out which.

### Bundle: `submissions/submission_gru_v032/submission_gru_v032.zip` — all gates PASS

| gate | result |
|---|---|
| contract | **PASS** |
| determinism | **PASS** |
| parity | **PASS** — 4.888e-06 vs 1e-03 tolerance |
| size | **PASS** — 2.42 MB of 20 MB |
| latency | **PASS** — 39.92 µs/row → 42.5 min, **70.8% of budget** |

This is the first bundle whose latency passed **outright with no anchor needed**, because
the box was finally idle. That is worth noting for what it confirms: the 1.62× grader
factor is accurate when the p50 is taken on a quiet machine. v028's actual grader rate
was 71.4 µs/row, and an idle local measurement scaled by 1.62 lands within ~1% of that.
Every misprediction this session came from contention, not from the factor.

Parity also ran at all — which it could not do for v034 before the `head_hidden` fix in
`tools/score_solution.py`. Same verifier, same session, both problems visible.

## Five new experiments (2026-10-03)

All single-variable against **v031** (`distill_weight` 2.0, 0.687376 local / 0.6587
public, our best). Four are running, one is waiting on data.

| ver | lever | change | params | tests |
|---|---|---|---:|---|
| v035 | **size DOWN** | hidden 128 → 96 | 148,674 | the only capacity direction never tried |
| v036 | **bottleneck UP** | `input_proj` 64 → 96 | 251,682 | information loss, not capacity |
| v037 | **schedule** | epochs 3 → 8 | 225,922 | do the two winners compose? |
| v038 | **regularisation** | weight_decay 0.01 → 0.03 | 225,922 | is the capacity loss overfitting? |
| v039 | **teacher** | + dozken's 0.6879 ensemble | 225,922 | the one distillation input never varied |

### Why these five, given what has already failed

Capacity going **up** is closed — head −3.2σ, hidden 285 −13.7σ, hidden 415 −21.7σ, all
monotone. Tail-weighting the teacher term is a dead null. Loss-diverse ensembling is
refuted twice. So the five above were chosen to attack the failures from directions we
have *not* walked, rather than to retry them louder.

**v035 — every capacity experiment went up, and we have never gone down.** There is one
precedent and it points this way: v022 (321,602) and v023 (225,922) shared
`distill_weight` 0.5, and locally v022 was 0.000547 *better* while publicly v023 was
0.0003 *better* and 10% faster. One submission each, so weak — but it is the only
direction with a positive sign, and it also buys latency.

**v036 — this is not a capacity experiment, which is the point.** At `input_proj` 64 the
224-wide level-plus-difference row is compressed **3.5×**. A bottleneck destroys
information; extra recurrent parameters add variance. dozken's audit put a number on it
("~0.014 WP of linearly available signal") and then measured −0.0032 — but on his
`chunk4k` recipe, which his own data shows is *harmful* under Global WP (−0.0370). Wrong
substrate. Never varied on ours. And it costs ~3% latency, since the projection is 6.4% of
per-row compute.

**v038 — the one experiment that could make the capacity results look different.** If
bigger models lose because they overfit, more regularisation should help at the current
size. `weight_decay` has sat at 0.01 and `dropout` at 0.0 since v011. If heavier decay
helps, the trunk widenings were *unregularised* rather than wrong and v026/v027 may be
recoverable. If it does nothing, the overfitting diagnosis in those three commits is
wrong. Either outcome is informative; that is why it is in the set.

**v039 — the teacher's quality is the one distillation input never varied.** Ours is
dozken's reset2k at 0.6864; his loss-diverse ensemble of reset2k + reset4k is **0.6879**,
and a teacher never runs at inference so it is free. The prior is poor and recorded as
such: his `distil_ens` (0.6585) *lost* to his `distil160` (0.6588). But his student already
sat at the compression sweet spot, whereas **our student has overtaken its own teacher**
(0.687376 vs 0.6864) — which none of his did. Whether that means headroom for a better
teacher or that everything extractable is extracted is exactly the question.

### v039 is blocked on data, and a watcher handles it

Its teacher needs a causal, stateful replay of all 212M train rows through `reset4k.onnx`
— a couple of hours. `tools/watch_and_launch_v039.sh` waits for that, verifies the output
is complete (shape, finiteness, and sequence-leading canary rows that an interrupted
replay would leave zero), and only then launches v039. It does not need an averaged label
file, because `TeacherStore` already averages multiple members on read and the yaml lists
both.

### What none of these can measure

The **epoch offset effect**. v025 and v032 are both 5-epoch runs and v025 carried an
offset +0.0024 better than the 3-epoch band — invisible to local validation by
construction. v037 (8 epochs) will move the *local* number and we still will not know
whether the public mapping moved with it. **v032 is packaged and unanswered**; submitting
it remains the single highest-information action available.
