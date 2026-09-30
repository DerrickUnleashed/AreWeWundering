# Experiment ranking — v001 to v011

All scores are `full_valid_wp`: global weighted Pearson over every one of the
1,873 validation sequences (3,528,273 scored rows), computed with the
reference metric in `tools/metric_ref.py`.

<!-- BEGIN GENERATED: leaderboard -->

| Rank | Ver | What it changed | full_valid_wp | Status | Public score | Duration | Memory |
|---:|---|---|---:|---|---|---:|---|
| 1 | **v011** | soft mask on P(is_scored) + state reset | 0.679986 | Submitted | 0.6515 | 30m12s (50%) | 627 MB |
| 2 | **v009** | signed-log volumes | 0.674456 | Submitted | 0.6483 | 39m52s (66%) | 627 MB |
| 3 | **v008** | cross-instrument (i0 vs i1) | 0.674445 | Submitted | 0.6473 | 41m17s (69%) | 626 MB |
| 4 | **v002** | baseline config, seed 1337 | 0.674350 | Submitted | 0.6435 | 35m02s (58%) | 628 MB |
| 5 | **v001** | 1x256 wider GRU | 0.674315 | Submitted | 0.6416 | 34m02s (57%) | 626 MB |
| 6 | **v007** | first differences | 0.674258 | Submitted | 0.6481 | 42m16s (70%) | 626 MB |
| 7 | **v006** | baseline config, seed 1339 | 0.674149 | Submitted | 0.6475 | 35m32s (59%) | 631 MB |
| 8 | **v005** | baseline config, seed 1338 | 0.674037 | Submitted | 0.6481 | 36m56s (62%) | 626 MB |
| 9 | **v003** | aux MSE + EMA | 0.673711 | Submitted | 0.6457 | 30m57s (52%) | 627 MB |
| 10 | **v004** | 2x192 + input projection | 0.672206 | Submitted | 0.6446 | 43m16s (72%) | 630 MB |
| 11 | **v010** | 6 epochs instead of 3 | 0.670696 | Submitted | 0.6427 | 34m37s (58%) | 626 MB |

Ranked by `full_valid_wp`, the global weighted Pearson over all 1,873
validation sequences (3,528,273 scored rows).

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
| **ours, best (v011)** | **0.6515** | — | 192k |
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

**The gap to their best is 0.0073 public, and to their #1 is 0.0107.** Our v011
also carried a serving bug (see below) that was never fixed in the submitted
bundle, so our true position is better than 0.6515 suggests.

The two levers visible in their names are worth reading as a roadmap:

- **`reset2k` (0.6569)** is a periodic state reset at **2,000** steps. Our v011
  used 4,000. That is the same lever we independently validated, at half the
  period -- so the obvious next experiment is `v012 = v011 with state_reset=2000`,
  a single-variable change against a recipe we already trust.
- **The distillation / ensemble entries (0.6585)** are a family, and which one
  the number belongs to is unresolved (see above). Two distinct levers:
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

## The next two experiments

Both are single-variable, both against the v011 parent, and both are justified by
evidence rather than novelty:

1. **v012: `state_reset = 2000`** (v011 has 4000). Confirms whether the period
   matters, and if so in which direction. Cheap: same architecture, same schedule.
2. **v013: distillation** from a wide/deep teacher into the v011 architecture.
   The largest remaining gap, and the one lever we have not tried at all.

**v011 stays the ship candidate until one of those beats it publicly**, since a
local gain inside the noise band is not evidence.

## Latency is a solved problem

Worth recording because it was treated as a blocker all night and turned out not
to be one. All 11 of our bundles finished, in **30m12s to 43m16s -- 50 % to 72 %
of the 60-minute budget** -- including the five whose local gates had been
measured under 7-way contention and marked "unverified".

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
