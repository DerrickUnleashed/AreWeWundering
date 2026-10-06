# RESUME.md — full project archive

**README.md is now the short one: one table ranked by public score, plus the handful of
extreme cases that change how you read that table.** Everything else lives here, and
nothing was deleted in the move.

This file is the working memory for the next session. It answers questions README
deliberately does not: what each experiment was testing, what the noise floor actually is,
which latency verdicts were measured under load and which were not, which of our own
conclusions were later refuted, and every bug we hit along the way. `AGENTS.md` remains the
rules; this is the narrative those rules produced.

**Layout.** One generated table — the local-metric ranking, which is what you consult while
choosing the next run — sits between the GENERATED markers below. Everything else is
hand-written and ordered chronologically, so reading top to bottom replays the project.

**What is NOT here any more.** The public-score ranking table, because it now lives only in
README.md. It was moved rather than copied, on purpose: two copies of the same table in two
files is two copies to keep in sync, and they would drift silently.


# Experiment ranking — v001 to v029

All scores are `full_valid_wp`: global weighted Pearson over every one of the
1,873 validation sequences (3,528,273 scored rows), computed with the
reference metric in `tools/metric_ref.py`.

<!-- BEGIN GENERATED: leaderboard -->

| Rank | Ver | What it changed | full_valid_wp | Status | Public score | Duration | Memory |
|---:|---|---|---:|---|---|---:|---|
| 1 | **v044** | v039 with epochs 3 -> 8 AND distill_weight... | 0.689846 | Submitted | 0.6599 | 47m41s (79%) | 628 MB |
| 2 | **v040** | v039 (our best at 0.6594) trained for 8... | 0.689146 | Submitted | 0.6596 | 44m57s (75%) | 628 MB |
| 3 | **v045** | v041 with hidden 128 -> 192, i.e | 0.689116 | Submitted | 0.6595 | 55m37s (93%) | 634 MB |
| 4 | **v041** | v039 with distill_weight 2.0 -> 3.0, i.e | 0.689036 | Submitted | 0.6593 | 46m51s (78%) | 628 MB |
| 5 | **v046** | v042 (our best PUBLIC score, 0.6600)... | 0.689030 | Submitted | 0.6598 | 48m32s (81%) | 628 MB |
| 6 | **v049** | v042's three-member teacher with... | 0.688784 | Submitted | 0.6598 | 46m07s (77%) | 628 MB |
| 7 | **v048** | v042 with training.state_reset 2000 ->... | 0.688708 | Submitted | 0.6600 | 45m27s (76%) | 628 MB |
| 8 | **v050** | v042's EXACT recipe at seed 1338 instead of... | 0.688599 | Submitted | 0.6596 | 45m57s (77%) | 628 MB |
| 9 | **v047** | v042 with training.state_reset 2000 ->... | 0.688458 | Submitted | 0.6600 | 50m32s (84%) | 628 MB |
| 10 | **v037** | v031's exact recipe at 8 epochs instead of 3 | 0.688370 | Submitted | 0.6592 | 48m06s (80%) | 628 MB |
| 11 | **v042** | v039 with a THREE-member teacher: dozken... | 0.688268 | Submitted | 0.6600 | 44m31s (74%) | 628 MB |
| 12 | **v039** | v031's exact recipe with a BETTER TEACHER | 0.688212 | Submitted | 0.6594 | 47m42s (80%) | 628 MB |
| 13 | **v054** | v042 recipe EXACTLY, on the patched... | 0.688151 | Ready to Submit | -- | 42m54s (72%) | -- |
| 14 | **v043** | v039 with ONLY dozken reset4k as teacher,... | 0.687944 | Submitted | 0.6569 | 45m52s (76%) | 628 MB |
| 15 | **v038** | v031's exact recipe with ONE change: AdamW... | 0.687577 | Submitted | 0.6588 | 48m06s (80%) | 628 MB |
| 16 | **v036** | v031's exact recipe with ONE change:... | 0.687535 | Submitted | 0.6586 | 48m06s (80%) | 629 MB |
| 17 | **v031** | v028's exact recipe and architecture with... | 0.687376 | Submitted | 0.6587 | 45m41s (76%) | 628 MB |
| 18 | **v032** | v028's exact recipe -- same architecture,... | 0.687219 | Submitted | 0.6587 | 45m32s (76%) | 628 MB |
| 19 | **v035** | v031's exact recipe with ONE change: hidden... | 0.687177 | Submitted | 0.6586 | 44m52s (75%) | 626 MB |
| 20 | **v052** | PROOF RUN | 0.687051 | Submitted | 0.6588 | 44m51s (75%) | 628 MB |
| 21 | **v030** | v028's exact recipe and architecture with... | 0.686919 | Submitted | 0.6585 | 47m11s (79%) | 629 MB |
| 22 | **v053** | v042 recipe on the PATCHED trainer with... | 0.686871 | Submitted | 0.6592 | 47m02s (78%) | 628 MB |
| 23 | **v051** | PROOF RUN | 0.686628 | Submitted | 0.6595 | 46m47s (78%) | 628 MB |
| 24 | **v028** | v023's exact recipe and architecture with... | 0.686510 | Submitted | 0.6585 | 46m52s (78%) | 628 MB |
| 25 | **v034** | v028's exact recipe with the ONE axis that... | 0.686000 | Submitted | 0.6579 | 47m17s (79%) | 629 MB |
| 26 | **v022** | dozken arch + his teacher | 0.685228 | Submitted | 0.6565 | 50m46s (85%) | 631 MB |
| 27 | **v024** | distil, batch 64 (his recipe) | 0.684818 | Submitted | 0.6560 | 50m17s (84%) | 631 MB |
| 28 | **v029** | Distilled student on dozken's architecture... | 0.684683 | Submitted | 0.6568 | 44m27s (74%) | 628 MB |
| 29 | **v023** | distil, smaller (52% of teacher) | 0.684681 | Submitted | 0.6568 | 45m36s (76%) | 628 MB |
| 30 | **v025** | v023's exact recipe (hidden 128, 225,922... | 0.684438 | Submitted | 0.6580 | 46m07s (77%) | 628 MB |
| 31 | **v026** | v023's exact recipe and architecture at 2x... | 0.684335 | TIMED OUT on grader | no score | 69m57s (117%) | 646 MB |
| 32 | **v027** | v026 at 2x again: hidden 415, 1,742,717... | 0.683058 | TIMED OUT on grader | no score | 69m57s (117%) | 643 MB |
| 33 | **v017** | distilled student, 117k (0.61x teacher) | 0.681866 | Submitted | 0.6540 | 25m37s (43%) | 625 MB |
| 34 | **v016** | distilled student, 192k | 0.681753 | Submitted | 0.6541 | 30m56s (52%) | 627 MB |
| 35 | **v018** | soft mask + state reset 10k | 0.680603 | Submitted | 0.6518 | 29m21s (49%) | 627 MB |
| 36 | **v015** | v011 re-exported, reset served [shares v011's weights] | 0.679986 | Submitted | 0.6515 | 30m31s (51%) | 627 MB |
| 37 | **v011** | soft mask + state reset 4k | 0.679986 | Submitted | 0.6515 | 30m12s (50%) | 627 MB |
| 38 | **v014** | soft mask + reset 4k, 2x192 proj64 | 0.679676 | Submitted | 0.6517 | 40m12s (67%) | 632 MB |
| 39 | **v012** | soft mask + state reset 2k | 0.678784 | Submitted | 0.6512 | 29m47s (50%) | 627 MB |
| 40 | **v021** | dozken arch, no teacher | 0.678575 | Submitted | 0.6505 | 55m32s (93%) | 634 MB |
| 41 | **v019** | FIX loss: globally pooled | 0.678374 | Submitted | 0.6510 | 30m32s (51%) | 627 MB |
| 42 | **v013** | reset 4k, no soft mask (control) | 0.676198 | Submitted | 0.6490 | 28m36s (48%) | 627 MB |
| 43 | **v009** | input: signed-log tail compression | 0.674456 | Submitted | 0.6483 | 39m52s (66%) | 627 MB |
| 44 | **v008** | input: cross-instrument i0/i1 | 0.674445 | Submitted | 0.6473 | 41m17s (69%) | 626 MB |
| 45 | **v002** | input: robust z-score, 8-sigma clip | 0.674350 | Submitted | 0.6435 | 35m02s (58%) | 628 MB |
| 46 | **v001** | wider: 1 layer x 256 | 0.674315 | Submitted | 0.6416 | 34m02s (57%) | 626 MB |
| 47 | **v007** | input: first differences | 0.674258 | Submitted | 0.6481 | 42m16s (70%) | 626 MB |
| 48 | **v006** | seed replicate of v002 (1339) | 0.674149 | Submitted | 0.6475 | 35m32s (59%) | 631 MB |
| 49 | **v005** | seed replicate of v002 (1338) | 0.674037 | Submitted | 0.6481 | 36m56s (62%) | 626 MB |
| 50 | **v003** | aux \|t\|-weighted MSE + EMA | 0.673711 | Submitted | 0.6457 | 30m57s (52%) | 627 MB |
| 51 | **v004** | bigger: 2x192 with 128 proj | 0.672206 | Submitted | 0.6446 | 43m16s (72%) | 630 MB |
| 52 | **v010** | schedule: 6 epochs not 3 | 0.670696 | Submitted | 0.6427 | 34m37s (58%) | 626 MB |
| 53 | **v033** | A deliberately DIVERSE ensemble member, not... | 0.660415 | Submitted | 0.6274 | 47m22s (79%) | 628 MB |

Ranked by `full_valid_wp`, the global weighted Pearson over all 1,873
validation sequences (3,528,273 scored rows).

### Not ranked by `full_valid_wp`

Rows that cannot be ranked on `full_valid_wp`, each for a stated reason: a contaminated score that is not comparable with the rest (v020), or a framework smoke test that was never a modelling result (v000). Both have bundles; neither is a submission candidate.

| Ver | What it changed | best so far (subset) | Status | Public score |
|---|---|---|---|---|
| **v000** | framework smoke test (60 steps) | 0.570341 | Submitted | 0.4986 |
| **v020** | FIX loss + train+valid data  [SCORE CONTAMINATED: trained on cache/trainval, which appends valid[200:1872] -- 1,673 of the 1,873 validation sequences -- so this score is measured on 89% of its own training data and is not comparable with train-only runs] | 0.687502 | Submitted | 0.6513 |
| **v055** | v054's exact recipe with ONE change:... | -0.125819 | Training | -- |
| **v056** | v054's recipe with the REGULARISATION/HEAD... | -0.024982 | Training | -- |

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
#1         0.6711
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
**0.6585 + 0.0024 = 0.6609**, against a 0.6711 leader we currently sit 0.0092 back.

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
#1         0.6711
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

**RESOLVED — v032 was submitted and scored 0.6587.** It was packaged as
`submissions/submission_gru_v032/submission_gru_v032.zip` and ran 45m32s = 76% of budget.

| ver | recipe | local | public | offset |
|---|---|---:|---:|---:|
| v028 | 3 epochs, distill 1.0 | 0.686510 | 0.6585 | −0.028010 |
| v031 | 3 epochs, distill 2.0 | 0.687376 | 0.6587 | −0.028676 |
| **v032** | **5 epochs, distill 1.0** | **0.687219** | **0.6587** | **−0.028519** |
| v025 | 5 epochs, distill 0.5 | 0.684438 | 0.6580 | **−0.026438** |

**Both halves are now in, and the answer is negative.** The pre-registered prediction was
~0.6616 if v025's +0.0024 offset bonus transferred and ~0.6592 if it did not. **It did
not** — v032's offset is −0.028519, dead normal at −0.4σ against the 8-run band, landing
squarely on the second branch. Longer training does **not** move the local→public mapping.

Full analysis, including what is left of the v025 anomaly and why batch size does not
explain it either, is in [v032 refuted the epoch hypothesis](#v032-refuted-the-epoch-hypothesis--the-most-promising-thread-we-had-is-dead).

The submission settled it, and the answer is no: the epoch effect was not a property of
training length. `v032`'s offset came back dead normal, so the two levers do not compose
in the way the offset anomaly suggested, and v025's +0.0024 remains a single unreproduced
observation we cannot attribute.

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
public, our best at the time). **All five finished and were submitted** — see
[Five new experiments (2026-10-03)](#five-new-experiments-2026-10-03) for the results and
[Six experiments upstream](#six-experiments-upstream-2026-10-03) for what they set up.

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

## v032 refuted the epoch hypothesis — the most promising thread we had is dead

`v032` was built to answer one binary question and it answered it, cleanly and negatively.

**Result: public 0.6587, offset −0.028519 — dead normal, at −0.4σ.**

The prediction written into its yaml *before* the run:

| | predicted | actual |
|---|---:|---:|
| if v025's +0.0024 offset bonus transferred | ~0.6616 | |
| if it did not | ~0.6592 | **0.6587** |

It landed squarely on the second branch. **Longer training does not move the
local→public mapping.** That was the thread three of our last five ideas were resting on —
v025's +4.4σ offset looked like the single largest public-score effect we had ever found,
bigger than any architecture or loss result, and it turns out not to be a property of
training length.

### What is left of v025, honestly

| ver | batch | epochs | offset | vs band |
|---|---:|---:|---:|---:|
| v024 | 64 | 3 | −0.028818 | −1.2σ |
| **v025** | **64** | **5** | **−0.026438** | **+5.3σ** |
| v032 | 16 | 5 | −0.028519 | −0.4σ |

The obvious next suspect was **batch size** — v025 is batch 64 and v032 is batch 16. But
v024 is *also* batch 64 and its offset is perfectly ordinary, so batch size does not explain
it either. **v025 is a real +5.3σ anomaly that we cannot attribute.** That is a loose end,
not a lever, and it should not be used to justify more long-schedule runs.

The disciplined reading is that one run at one setting produced one outlier we have not
reproduced, and the correct response to an unreproduced outlier is to stop building on it
rather than to construct an explanation after the fact.

### Where that leaves us

```
ours v032  0.6587   ties on public, 2nd locally
ours v031  0.6587   ties on public, BEST locally (0.687376)
dozken     0.6588
#1         0.6711   → 0.0090 back
```

**v031 and v032 are tied at 0.6587** and are within 0.0002 of dozken. Two independent
levers — distillation weight and training length — both moved *local* score (+5.5σ and
+4.5σ) and neither moved the *board* beyond v028's 0.6585.

That is now three separate confirmations of the same uncomfortable fact:

| lever | local | public |
|---|---|---|
| distill_weight 0.5→1.0→1.5→2.0 | +0.0027 | +0.0002 |
| epochs 3→5 | +0.0007 | +0.0002 |
| both | — | 0.6587, tied |

**Local gains in this regime are decaying into the public split.** We are close enough to
the teacher's ceiling (0.690168) and the teacher itself (0.6879) that further local
improvement is no longer buying board position. The remaining 0.0090 is probably not
reachable by tuning this student at all.

### Latency note

v032 ran 45m32s = 76% of budget. Its bundle had predicted 42.5 min from an
**idle-machine** measurement — accurate to within 8%. The packager's projections earlier
in the session, taken under 4-way contention, were off by up to 25 minutes and marked
shippable models NOT-READY. The only reliable latency number we have ever produced was
the one taken on a quiet box.

## All five new experiments reported — two wins, and two of my priors were wrong

| ver | lever | change | params | full_valid_wp | vs v031 | σ |
|---|---|---|---:|---:|---:|---:|
| v035 | size **down** | hidden 128→96 | 148,674 | 0.687177 | −0.000199 | −1.3 |
| v036 | bottleneck **up** | `input_proj` 64→96 | 251,682 | 0.687535 | +0.000159 | +1.0 |
| **v037** | **schedule** | **epochs 3→8** | 225,922 | **0.688370** | **+0.000994** | **+6.3** |
| v038 | regularisation | weight_decay 3× | 225,922 | 0.687577 | +0.000201 | +1.3 |
| **v039** | **teacher** | **+ dozken 0.6879 ensemble** | 225,922 | **0.688212** | **+0.000836** | **+5.3** |

**v037 is the best local score this project has produced**, v039 the second. Both clear
±0.000318 by a wide margin. Nothing got worse.

### v039 refuted the prior I recorded against it

I wrote a **poor** prior into v039's yaml: dozken's `distil_ens` (0.6585) lost to his
`distil160` (0.6588), so on his pair a better teacher did not beat a worse one through
compression. **That did not hold here — the better teacher gained +0.000836 (+5.3σ).**

The reasoning I offered for why our student might differ was the whole argument, and it
was right: his student already sat at the compression sweet spot, so there was no gap left
for the teacher's extra 0.0015 to close. **Ours had overtaken its own teacher** (0.687376 vs
0.6864), which none of his did — so there *was* a gap, and a better teacher filled it.
The transferable lesson: a prior carried over from another team's configuration is worth
less than the structural difference between the two cases, and the structural difference
is usually knowable before the run.

### v038 refuted my diagnosis of the capacity failures

Across three separate commit messages I argued that bigger models lose **because they
overfit**, and that heavier regularisation should therefore help at the current size.
**Tripling `weight_decay` gained +0.000201 — +1.3σ, inside the noise.**

So the capacity losses were **not** an overfitting signature and that explanation was
wrong. The three regressions (head −3.2σ, hidden 285 −13.7σ, hidden 415 −21.7σ) are real
and reproducible, but "too many parameters" is a description, not a mechanism. I wrote a
causal story into those commits from a correlation, and the honest position is that we do
not know why capacity hurts here.

### The three nulls are informative too

- **v035 (smaller, −34%): −1.3σ.** 128 is a genuine **interior** optimum — neither too
  big nor too small. That closes the capacity axis in *both* directions at once, which is a
  cleaner result than the one-sided closure I had.
- **v036 (widen the bottleneck, +11%): +1.0σ.** `input_proj` 64 was already past the knee.
  dozken's "~0.014 WP destroyed by the projection" does not reproduce on our recipe; the
  useful information is that the bottleneck is not the binding constraint.
- **v038: +1.3σ.** See above.

### The caveat that governs all five

v032 just established that local gains in this regime **decay into the public split**:
v031's +0.000865 local delivered +0.0002 public, a 4.3× attenuation. At the same factor,
v037's +0.000994 projects to roughly **+0.0002 public** and v039's +0.000836 to about
+0.0002.

So these are **real local gains of uncertain public value**, not a predicted scoreboard.
Two are worth submission slots; the three nulls are not.

## v034 scored 0.6579 — and it sharpens what the attenuation problem actually is

`v034` (doubled MLP head, 263,362 params) is now on the board: **0.6579**, our 5th best.

| | local | public | Δ local | Δ public | ratio |
|---|---:|---:|---:|---:|---:|
| v031 vs v028 | +0.000865 | +0.0002 | gain | gain | **0.23×** |
| **v034 vs v028** | **−0.000510** | **−0.0006** | loss | loss | **1.18×** |

**This is the useful part.** v031's local *gain* delivered only 23% of itself publicly,
while v034's local *loss* delivered 118%. So the attenuation is **not** the mapping being
noisy in both directions — **it is specifically that local gains are not converting.**

That is a sharper problem than "local and public disagree", and it changes what to do about
it. A noisy mapping would mean our local metric is uninformative and we should stop
trusting it. A one-sided gain attenuation means the local metric is measuring something
real that simply stops mattering near the ceiling — which is consistent with v031 at
0.687376 having already overtaken its 0.6864 teacher, and with the top of our table
(v028/v030/v031/v032 all within 0.0002 public) being a genuine plateau rather than four
coincidences.

Its offset is **−0.028100, +0.7σ — completely normal.** So widening the head perturbed the
local-to-public mapping no more than it perturbed anything else.

### v033 and v034, both now on the board

| ver | what it tested | local | public | outcome |
|---|---|---:|---:|---|
| v034 | double the MLP head | 0.686000 | **0.6579** | clean negative, 5th best |
| v033 | per-sequence pooling | 0.660415 | **0.6274** | offset broke, −11.3σ |

Both were worth submitting and both were worth the slots: v034 gives a clean 1:1
confirmation of the head regression, and v033's −0.0047 excess offset is the single
clearest measurement in the project of what per-sequence pooling costs against a global
metric. Neither could have been known locally.

## All seven bundles are packaged, every gate PASS on an idle machine

| ver | params | full_valid_wp | grader | % budget | zip | public |
|---|---:|---:|---:|---:|---:|---|
| v035 | 148,674 | 0.687177 | 36.8 min | 61% | 1.60 MB | — |
| v038 | 225,922 | 0.687577 | 42.4 min | 71% | 2.42 MB | — |
| v032 | 225,922 | 0.687219 | 42.5 min | 71% | 2.42 MB | **0.6587** |
| v039 | 225,922 | 0.688212 | 42.7 min | 71% | 2.42 MB | — |
| v037 | 225,922 | **0.688370** | 42.8 min | 71% | 2.42 MB | — |
| v036 | 251,682 | 0.687535 | 44.5 min | 74% | 2.69 MB | — |
| v031 | 225,922 | 0.687376 | *(45m41s actual)* | 76% | 2.42 MB | **0.6587** |

All seven: contract, determinism, parity and size **PASS**, and latency **PASS**. Parity
ranged 2.146e-06 to 4.888e-06 against a 1e-03 tolerance.

**Every one of these latency numbers was taken on an idle box**, which is the first time
that has been true for a full packaging session. It is why they cluster so tightly — the
four 225,922-param models all land at 71%, within 0.4 minutes of each other, which is what
identical graphs should do and is exactly what never happened under contention.

### v035 is the cheap fallback, and that is its real value

At **148,674 params it runs in 61% of budget and ships in a 1.60 MB zip** — 34% smaller
than any other bundle and 10 points cheaper to serve. Its score is 0.687177, which is
−1.3σ against v031: a null.

So the smallest model in the project is also, by a wide margin, the most efficient
artifact, and the capacity axis being closed costs us almost nothing. If latency ever
becomes the binding constraint rather than score, v035 is the obvious fallback and it is
already packaged and verified. That is a useful thing to have on disk for a null result.

### One stale number, deliberately left

`v031`'s `metadata.json` still records **NOT-READY at 63.9 min (106%)**, because it was
packaged under 4-way contention. It has since been submitted and **ran 45m41s = 76% of
budget**, so the recorded verdict is wrong by 40%.

It is not corrected, for the same reason v034's null parity was not: the bundle is an
immutable record of what the gate said at packaging time, and hand-editing a NOT-READY to
READY would destroy the evidence that this failure mode exists and keeps recurring. The
README resolves it through the measured-anchor rule and through the real grader reading,
which is where a reader should look.

## The recurrent state never carried across chunks (2026-10-05)

**The most consequential bug in the project, and it was invisible for 40+ versions.**

### What the code did

    h = [torch.zeros(1, Xb.shape[0], model.hidden) for _ in range(model.layers)]
    ... zero the lanes that Rb flags ...
    pred, _ = model.forward_chunk(Xt, h)          # returned state DISCARDED

`Rb` marks the lanes that must start a chunk from a zero state. It was being applied to a
tensor that was already entirely zeros, and the state the GRU produced was thrown away
rather than carried. So the recurrence saw **at most `chunk_len` = 1000 steps of history**,
every step, forever.

### Two silent consequences

**1. A 4x train/serve memory mismatch on every model ever shipped.** Serving clears state
every `inference.state_reset` = 4000 steps, so the deployed model runs on up to 4000 steps
of context. Training gave it 1000. `AGENTS.md` §12 Phase 3 specified "stateful truncated
BPTT: hidden state carried across chunks" from the very first day, and it was never
implemented.

**2. `training.state_reset` could not change the weights at all.** v047 (4000) and v048
(10000) came out **byte-identical** to v042 (2000) — all 18 tensors equal to the bit:

    param                       v042[0]         v047[0]         v048[0]   identical?
    grus.0.bias_hh_l0          -0.088002      -0.088002      -0.088002   True
    ... all 18 tensors ...

Their reported scores differed only because the **evaluation** path (line 947) does honour
`state_reset`, so the same weights were scored on three different reset grids. Roughly
**24 of the last ~35 GPU-hours produced one model**, presented as a three-point sweep.

### How it was found, and how nearly it was missed

The ensemble probe reported three models with identical local WP to six decimal places.
That is implausible for three different training runs, so it was worth checking. Two false
alarms came first — an "export bug" conclusion that was really my own test using
`ls | tail -1` (the *lowest*-val checkpoint), and then an `--out` flag the exporter does not
have, so the export silently never ran. Both were my errors, not the codebase's. The third
check — comparing actual tensor values across the three checkpoints — was the one that held.

**Lesson worth more than the bug: an identical number across supposedly-different runs is a
finding, not a coincidence to be explained away.** It was the only signal, and the two
attempts to explain it away were both more expensive than checking it.

### The fix, and the proof

`h` is initialised once before the step loop and carried; the returned state is detached at
the chunk boundary (truncated BPTT — without the detach the graph grows across the whole
sequence and backprop reaches step 0). Applied to the template and all 41 existing scripts.

The loader's `R` mask was always correct — it produced 30 / 15 / 6 resets per 60 chunks for
2000 / 4000 / 10000. **The mask was fine; the consumer was broken.** That is why no amount
of inspecting the data path would have found it.

Proof runs v051 (sr 2000) and v052 (sr 10000), one epoch, identical seed and data:
their losses now **diverge** from step 25 onward (2.331575 vs 2.331618). Before the patch
they were bit-identical, which is the definition of a dead axis.

### What this invalidates

Every result in this project was produced by a model trained on 1000 steps of context and
served on 4000. The plateau at ~0.660, the "preprocessing is closed" conclusions, the
"capacity is closed" conclusions and the state_reset results themselves all describe a
system that was not the one being shipped. They are not wrong, they are about a different
model than the one we thought we were measuring.

## EDA: the last preprocessing direction, refuted (2026-10-04)

The first EDA in the project. It ran against one rule from `AGENTS.md` §12 Phase 1 — *no
plot that does not change a decision* — and it changed two decisions, both of them to
"leave it alone".

### Output calibration: REJECTED, `solution.py` is already optimal

`solution.py` already applies `METRIC_CLIP = 2.0`, and 0.0000% of scored predictions reach
it, so clipping is a no-op. Pearson is invariant to positive **scaling**, so no rescaling can
ever help either. Only a nonlinear monotone transform could, and **every one of them lost**,
monotonically in how much they squashed: clip ±1.6 −3.0σ, clip ±1.2 −53σ, clip ±1.0 −117σ,
power 0.3 −151σ, tanh −74σ. Squashing only destroys tail information the target genuinely has.

The decile decomposition says there is nothing to recover: WP **rises** with |target|
(0.17 → 0.84) and the top decile carries **25.3%** of the metric weight. We are strongest
exactly where the metric weights most. The weak rows are low-|target| ones, and 9.7% of
targets are exactly 0.0 and carry **zero** weight — rows whose sign is not merely hard but
unweighted.

### Feature ranking: REJECTED, and a linear probe was the wrong instrument

`feature_signal.py` found that ranking a feature beats using it raw, weighted, on 8 of the
top 14. That is outside a linear first layer's hypothesis class, so it looked like the one
legitimate preprocessing direction left. Three checks later:

1. **At 30 sequences the gains were ~5× inflated.** i0_p8's weighted gain fell from +0.3369
   to +0.0630 at 300 sequences. The small sample was badly optimistic.

2. **The MARGINAL value is what decides, and the per-feature number is the wrong one.** The
   112 columns are heavily correlated, so a per-feature gain can be pure redundancy.
   `feature_rank_value.py` fits ridge on identical rows and weights, 300 sequences, and asks
   whether ranks explain variance the raw features do not:

   | model | weighted R² |
   |---|---:|
   | raw 112 | 0.088471 |
   | raw + 16 rank channels | 0.094709 (+0.0062) |
   | **raw + 112 raw squares** | **0.097749 (+0.0093)** |

   **Squares beat ranks.** rank(f) is a *monotone* function of f, and our first layer is
   followed by a tanh recurrence, so the network already realises monotone functions of a
   single row. Variance attributable to rank *specifically*: **−0.0030**.

3. **The control is what caught it.** The first version compared raw against raw+ranks and
   returned "BUILD IT". That was wrong: a *linear* probe models only the first layer, while
   the network is nonlinear immediately downstream. The probe was understating the network
   the whole time. Adding x² as a control arm turned a false positive into a correct one.

### What this closes

Linear preprocessing folds into layer 1 for free (`AGENTS.md` §10.3). This shows the
**nonlinear** transforms fold into the layer after it just as cheaply. Combined with the
capacity plateau, the loss-diverse-ensemble refutations, the third-teacher-member null, the
`distill_weight` sweep that turned over at 3.0, and the output-calibration null, there is no
preprocessing or architecture lever left with evidence behind it.

## Six experiments upstream (2026-10-03)

All six **finished**. All were launched against the finding below rather than against the
plateau, and the finding turned out to matter more than the plateau did.

### The plateau, and what it was pointing at

The five experiments above landed at **0.6586–0.6594 public** — a spread of 0.0008 across
five levers sharing nothing but the architecture: student size, bottleneck width, schedule
length, weight decay, and teacher identity. A spread that tight is a **ceiling, not noise**.

But the attenuation factor was also wrong. I had generalised **4.3×** from v031 alone —
one sample. Across the five the factors are 0.50, 0.84, 0.50, −0.63, 0.50, so local gains
convert at roughly **half**, not a fifth. Local validation is a materially better selection
signal than I had been claiming, and the ±0.0003 official floor is worth spending against.

Only one of the five levers won: **teacher identity**. So the six upstream runs (v040–v045)
all attack the teacher and the supervision signal, not the student.

### The probe: the teacher is BETTER than our student

Nothing had ever scored dozken's models on **our** metric. His 0.6879 comes from his own
192-sequence held-out set, a different scale from our `full_valid_wp`, so the entire
"teacher identity is the lever" conclusion rested on a number we could not compare to ours.

`tools/score_external_onnx.py` scores foreign graphs alongside ours — it detects the input
contract instead of assuming it, because dozken's take **raw** features plus an explicit
`prev` and difference inside the graph, while ours take a precomputed rank-3 matrix. One
pass, 500 sequences, every model on **identical rows** under the same mask:

| model | params | t0 | t1 | MEAN |
|---|---:|---:|---:|---:|
| dozken `reset4k` | 435,943 | 0.708644 | 0.710524 | **0.709584** |
| dozken `reset2k` | 435,943 | 0.703904 | 0.705128 | 0.704516 |
| **our v039** | 225,922 | 0.702736 | 0.705008 | 0.703872 |

```
reset4k - v039 = +0.005712
```

**The teacher is 0.0057 better than the student on our own rows and our own metric.** This
inverted the premise, and the error in v039's own config is legible once written down: it
argued the student "has already overtaken its own teacher", which it had established by
comparing our subset number against dozken's number from a *different* held-out set.
Different scales, so the comparison could not have held.

### The six runs, and two hypotheses that both died

| ver | lever | full_valid_wp | Δ vs parent | σ | verdict |
|---|---|---:|---:|---:|---|
| **v044** | epochs 8 **+** dw 3.0 | **0.689846** | +0.001634 | **+10.3** | **best in project** |
| v040 | epochs 3 → 8 | 0.689146 | +0.000934 | +5.9 | win |
| v045 | hidden 128 → 192 | 0.689116 | +0.000080 | +0.5 | **NULL** |
| v041 | dw 2.0 → 3.0 | 0.689036 | +0.000823 | +5.2 | win |
| v042 | third teacher member | 0.688268 | +0.000056 | +0.4 | **NULL** |
| v043 | single `reset4k` teacher | 0.687944 | −0.000268 | −1.7 | NULL |

2σ threshold is ±0.000318.

**v045 — capacity. Predicted right, measured null.** The teacher is 2× wider (192 vs 128),
so a student of the teacher's width should have been able to represent its function. It
exported at **435,714 params against the teacher's 435,943** — capacity parity confirmed by
construction — and gained **+0.000080, +0.5σ**. The teacher–student gap is **not** a
capacity problem. It also cost ~2× latency for nothing, and its latency gate failed; the
saving grace is that it was never packaged as a ship candidate.

**v042 — more members. Predicted right, measured null.** The pair beat a single teacher by
+0.001091 (**+6.9σ**), so a third *decorrelated* member should have kept paying. It gained
**+0.000056, +0.4σ**. **Teacher identity saturates at two members.** The third member was
deliberately diverse — our own v037, trained on a different schedule — so this is not the
v023 same-recipe mistake; genuinely different teachers simply stop helping.

### What actually worked

Not the teacher, and not the student. **Supervision weight and schedule length**:

```
distill_weight 2.0 -> 3.0   +0.000823  (+5.2σ)
epochs 3 -> 8               +0.000934  (+5.9σ)
both together (v044)        +0.001634 (+10.3σ)
```

The two stack almost additively, which is the opposite of the plateau: `dw` was monotone and
unturned at 2.0, and 2.0 was never retested against a *shorter* schedule until v044 put them
together.

### Bundles (all built, none uploaded)

| ver | full_valid_wp | status | duration |
|---|---:|---|---|
| **v044** | **0.689846** | **PASS** | 42m30s (71%) |
| v040 | 0.689146 | **PASS** | 42m12s (70%) |
| v045 | 0.689116 | NOT-READY | 55m32s (93%) — null **and** marginal |
| v041 | 0.689036 | NOT-READY\* | 48m06s (80%) |
| v042 | 0.688268 | **PASS** | 42m12s (70%) |
| v043 | 0.687944 | NOT-READY\* | 48m06s (80%) |

This was the **first idle-machine packaging session**, which is why the numbers cluster so
tightly: the four 225,922-param models land at 70–71%, within 0.4 minutes of each other —
what identical graphs should do, and exactly what never happened under contention. The same
tool read v041 at 57.46 µs/row with four runs training and v045 at 72.67; their true idle
figures are ~40 and 54.62.

\* v041/v043 say NOT-READY **only** because of contended measurements. Both are
byte-identical in graph to v037 — 16 initializers, `features (1,1,224)`,
`hidden_0/1 (1,1,128)` — which **actually ran 48m06s at 80%** on the grader. Substantively
they are submittable; the recorded verdict is a load artifact, left uncorrected for the same
reason v031's is (bundles are immutable records of what the gate said when it ran).

### Two bugs worth keeping

**A latency label that ignored its own number.** v045 rendered as `Ready to Submit (latency
anchored on v021)` at **93% of budget** while its own `metadata.json` said NOT-READY at
96.8%. The anchor branch returns *before* consulting the gate result, so it overrode a
failure and attached a green label to a number contradicting it — wrong in exactly the
direction that gets a model uploaded. Anchored branches now check the figure and return
`Marginal (93% of budget…) - review before upload` above 85%. 85% is where our evidence
stops: everything submitted landed between 56.7% and 80%, and v026 died at 117%.

**A parser that destroyed its own member lists.** `--mean-of` split on `,` before `:`,
so `dozken_pair:reset2k,reset4k` became two specs and the "pair" was averaged over **one**
member. Symptom: `dozken_pair` absent and `student_plus_pair` *exactly* equal to v039. The
three single-model scores were never affected, so the probe's conclusion stood. It is
repeatable now, validates its members, and rejects unknown names instead of silently
averaging the wrong set.

### v041 and v043 scored — and they measure the local metric's limits

| ver | local | σ vs parent | public | Δ public vs v039 | grader |
|---|---:|---:|---:|---:|---|
| v039 (reset2k teacher, dw 2.0) | 0.688212 | — | **0.6594** | — | 47m42s (80%) |
| **v041** (pair teacher, dw 3.0) | 0.689036 | **+5.2** | **0.6593** | **−0.0001** | 46m51s (78%) |
| **v043** (reset4k alone, dw 3.0) | 0.687944 | −1.7 | **0.6569** | **−0.0025** | 45m52s (76%) |

**Latency was never the problem.** 76% and 78% of budget, both comfortably inside 60
minutes. The `NOT-READY` verdicts on these two bundles were purely contended-measurement
artifacts, and the identical-graph anchor to v037's measured 48m06s predicted both correctly.
That is the third time the anchor has beaten the local projection, after v028 and v031.

### The asymmetry: local gains don't transfer, local losses do

v041 was the largest single-lever local gain of the batch — **+0.000824, +5.2σ**, which is
far outside the ±0.000318 detection threshold. It came out **0.0001 below** its own parent on
the board that decides the prize. The +5.2σ was real, correctly measured, on 1,873 sequences
and 3,528,273 rows. It simply was not worth anything off-board.

v043 went the other way. Locally a null at −1.7σ; publicly **−0.0025**, the worst score we
have submitted since v033. A −0.000268 local loss transferred at roughly full size.

So the two are not symmetric:

| | local Δ | public Δ | transfer |
|---|---:|---:|---|
| v041 (gain) | +0.000824 | −0.0001 | **none** |
| v043 (loss) | −0.000268 | −0.0025 | **amplified ~9×** |

**Practical consequence.** Small local gains cannot be trusted to transfer, so chasing them
is not a good use of remaining submissions. A small local *regression* is worth respecting,
because it tends to arrive larger than expected. This is the third distinct attenuation
measurement in this project and none of them agree: 4.3× from v031 alone, ~0.5 across the
five levers, and now 0 for v041 and 9× for v043. The consistent finding is not a factor — it
is that **local differences below roughly 0.001 have no reliable relationship to public
score**, in either direction.

### The teacher ladder survives, in the right order

The one place local and public agree is the arm v043 was packaged to test:

| teacher | local | public |
|---|---:|---:|
| reset4k alone (v043) | 0.687944 | 0.6569 |
| reset2k alone (v039) | 0.688212 | 0.6594 |
| pair (v041) | 0.689036 | 0.6593 |

`reset4k` alone is worst in **both**, by a similar margin. So the control did its job: "the
pair is better than a single teacher" is not the same claim as "the pair happened to pick the
better member", and now both metrics agree that it was the better member doing the work.
What does *not* hold is the size of the pair's advantage — +0.001091 locally, ±0.0001
publicly, i.e. a tie.

### v044 has not been submitted

**v044 (0.689846, +10.3σ over v039) is our best local model and is still unsubmitted.** If
the asymmetry above holds, its +0.001634 local gain may also arrive as nothing. It is
packaged, `STATUS: PASS`, and measured at 42m30s (71%) of budget on an idle machine, so it
is ready to upload whenever you want to spend one of the five daily submissions on it.
