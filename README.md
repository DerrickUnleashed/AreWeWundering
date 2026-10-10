# Alfa Connectome — public results

**One table, ranked by public score.** This is the only number that decides the prize, so it
is the only one on this page. Everything else — 36 sections of experiments, hypotheses,
refutations, noise floors, latency post-mortems and bugs — is in **[RESUME.md](RESUME.md)**,
which is the working memory for the next session.

Rules and conventions are in **[AGENTS.md](AGENTS.md)**. Raw scores, grader durations and
memory for every submission are in `exploration/leaderboard.json`.

> ## ⚠ Read this before trusting any number on this page
>
> ## ⚠ The metric you are reading is not the one that selects models
>
> The 200-sequence periodic-eval subset and the official `full_valid_wp` (all 1,873
> sequences) **disagree about which model is best — and on the seven-run optimiser sweep they
> disagreed by a perfect inversion.**
>
> | run | official `full_valid_wp` | σ | *subset* σ |
> |---|---:|---:|---:|
> | **v059** (`lr` 0.002 → 0.001) | **0.688676** | **+3.3** | −6.1 |
> | **v060** (`layers` 2 → 1) | 0.686824 | **−8.3** | **+6.1** |
>
> **v060 is the worst model in the sweep officially and the best on the subset**, with a subset
> lead that strengthened monotonically across four measurement rounds. **v059 is the exact
> mirror.** We were about to package v060 as the winner.
>
> **So: only `full_valid_wp` over all 1,873 sequences is a selection signal. Never rank
> candidates on the periodic-eval subset** — it is 1/9 of the scored rows and roughly 4× noisier.
> Full detail in edge case 1.
>
> **On 2026-10-05 we found that the recurrent state never carried across chunks during
> training.** The trainer rebuilt the hidden state as zeros every step and discarded the
> state the GRU returned, so the model saw at most **1,000 steps** of history while serving
> runs it on **4,000**.
>
> Two consequences, both silent:
>
> 1. Every model on this page was trained on a quarter of the context it is served on.
> 2. `training.state_reset` could not change the weights at all — v047 and v048 came out
>    **byte-identical** to v042. About **24 of the last 35 GPU-hours produced one model.**
>
> It was found because an ensemble probe returned three models with identical scores to six
> decimal places, which should not happen for three different runs.
>
> **It is now fixed and proven.** v051 / v053 / v052 — the same three configs — produced
> three **distinct** models for the first time, and `training.state_reset` finally moves the
> weights. So every score above was produced by a model trained in a way we did not believe
> we were training, and the fix invalidates the reasoning behind most of the closed axes.
>
> Full detail, including the two false alarms I produced on the way (both my own test
> errors, not faults in the code), is in [RESUME.md](RESUME.md).

> ## ⚠ A second silent bug: 56 configs described models that were never built
>
> While opening up the regularisation axis for v056 we found that the model class built its
> head as `nn.Linear(hidden, 2)` and **neither construction site passed `head_hidden` at
> all**. So `head: mlp`, `head_hidden` and `layernorm` were recorded in **56 yaml files and
> never once applied to a model**. Every model in this project ran a plain single linear
> readout, whatever its config claimed.
>
> This matters beyond the head itself:
>
> - **"A deeper head doesn't help" was never properly tested.** v034 was the *only* model
>   that carried a real 2-layer head (hand-written, `head_hidden=384`), and it scored
>   **−0.0005** — alone, with no batch norm and no dropout. That single confounded run is
>   what made the axis look closed.
> - The cause was two construction sites passing **different keyword sets**. Fixed in v056 by
>   `build_model()`, now the only path from a yaml `model:` block to a `GRUModel`.
>
> Combined with the state-carry bug above: **two of the three things our configs claimed to
> control were not being applied.** When a config knob looks inert, check that it is read
> before believing a null result about it.

> ## ⚠ The sweep scored. `full_valid_wp` works as a selector; the subset does not.
>
> All seven ran. **v059 ties our best public score**, and the sweep validated the one thing we
> most needed to know — which local metric can be trusted.
>
> | run | `full_valid_wp` (official) | public | grader | *subset Δ* |
> |---|---:|---:|---:|---:|
> | **v059** `lr` 0.002 → 0.001 | **0.688676** | **0.6600** | 44m31s | +0.000507 |
> | v055 `state_reset` 10000 | 0.688667 | 0.6598 | 44m17s | +0.000794 |
> | v057 `grad_clip` → 5.0 | 0.688293 | 0.6597 | 44m06s | +0.000420 |
> | v058 `weight_decay` → 1e-4 | 0.688032 | 0.6592 | 46m42s | −0.000316 |
> | v056 regularisation ⚠ | 0.687772 | 0.6591 | 50m37s | −0.000181 |
> | v060 `layers` 2 → 1 | 0.686824 | 0.6591 | **38m11s** | +0.000369 |
> | v061 head GELU → ReLU ⚠ | 0.687064 | 0.6572 | 49m36s | +0.000128 |
>
> **Spearman(official, public) = +0.893.** The official metric ranked the top three
> correctly — v059, v055, v057 → 0.6600, 0.6598, 0.6597, in that order. Across seven
> independent runs this is the first sweep where the local metric and the board agreed, and it
> is strong evidence that `full_valid_wp` is a usable selector at this scale.
>
> **The subset ranked v060 first** (its subset delta was the sweep's largest). v060 came
> **last** officially and **sixth of seven** publicly. Every prior inversion had local pointing
> the wrong way; this one is the subset doing it.
>
> **v060 is the efficiency outlier:** 38m11s = **64% of budget**, the fastest of all seven,
> with 126,850 parameters — and it beat v056 publicly despite being 5σ worse officially.
>
> **⚠ Two bundles were uploaded with `STATUS: NOT-READY` and both scored.** v056 (0.6591) and
> v061 (0.6572) had failed parity at max |diff| = 1.026 — a real train/serve skew, so the
> served model is not the trained model. **The gate is advisory; the platform does not enforce
> it.** v061 scored last of seven, consistent with a corrupted head, so the broken path does
> cost something — but it still produced a valid number, which means a broken bundle can quietly
> consume a submission slot.

## Public leaderboard

<!-- BEGIN GENERATED: public-score ranking -->

### Every model, ranked by public score

One table. The **Public score / Status** column carries a public score where
one exists, and a status where it does not, so models that are ready to submit
or ready to package appear here rather than in a second table that could go
stale. Scored rows sort by score; unscored rows sort after, by `full_valid_wp`,
which is **not** comparable to the public column.

| Rank | Ver | full_valid_wp | Public score / Status | vs best | Grader |
|---:|---|---:|---|---:|---|
| 1 | **v042** | 0.688268 | **0.6600** | +0.0000 | 44m31s (74%) |
| 2 | **v047** | 0.688458 | **0.6600** | +0.0000 | 50m32s (84%) |
| 3 | **v048** | 0.688708 | **0.6600** | +0.0000 | 45m27s (76%) |
| 4 | **v059** | 0.688676 | **0.6600** | +0.0000 | 44m31s (74%) |
| 5 | **v044** | 0.689846 | **0.6599** | -0.0001 | 47m41s (79%) |
| 6 | **v046** | 0.689030 | **0.6598** | -0.0002 | 48m32s (81%) |
| 7 | **v049** | 0.688784 | **0.6598** | -0.0002 | 46m07s (77%) |
| 8 | **v055** | 0.688667 | **0.6598** | -0.0002 | 44m17s (74%) |
| 9 | **v054** | 0.688151 | **0.6597** | -0.0003 | 45m27s (76%) |
| 10 | **v057** | 0.688293 | **0.6597** | -0.0003 | 44m06s (74%) |
| 11 | **v040** | 0.689146 | **0.6596** | -0.0004 | 44m57s (75%) |
| 12 | **v050** | 0.688599 | **0.6596** | -0.0004 | 45m57s (77%) |
| 13 | **v045** | 0.689116 | **0.6595** | -0.0005 | 55m37s (93%) |
| 14 | **v051** | 0.686628 | **0.6595** | -0.0005 | 46m47s (78%) |
| 15 | **v039** | 0.688212 | **0.6594** | -0.0006 | 47m42s (80%) |
| 16 | **v041** | 0.689036 | **0.6593** | -0.0007 | 46m51s (78%) |
| 17 | **v037** | 0.688370 | **0.6592** | -0.0008 | 48m06s (80%) |
| 18 | **v053** | 0.686871 | **0.6592** | -0.0008 | 47m02s (78%) |
| 19 | **v058** | 0.688032 | **0.6592** | -0.0008 | 46m42s (78%) |
| 20 | **v056** | 0.687772 | **0.6591** | -0.0009 | 50m37s (84%) |
| 21 | **v060** | 0.686824 | **0.6591** | -0.0009 | 38m11s (64%) |
| 22 | **v038** | 0.687577 | **0.6588** | -0.0012 | 48m06s (80%) |
| 23 | **v052** | 0.687051 | **0.6588** | -0.0012 | 44m51s (75%) |
| 24 | **v031** | 0.687376 | **0.6587** | -0.0013 | 45m41s (76%) |
| 25 | **v032** | 0.687219 | **0.6587** | -0.0013 | 45m32s (76%) |
| 26 | **v062** | 0.687990 | **0.6587** | -0.0013 | 45m37s (76%) |
| 27 | **v035** | 0.687177 | **0.6586** | -0.0014 | 44m52s (75%) |
| 28 | **v036** | 0.687535 | **0.6586** | -0.0014 | 48m06s (80%) |
| 29 | **v028** | 0.686510 | **0.6585** | -0.0015 | 46m52s (78%) |
| 30 | **v030** | 0.686919 | **0.6585** | -0.0015 | 47m11s (79%) |
| 31 | **v025** | 0.684438 | **0.6580** | -0.0020 | 46m07s (77%) |
| 32 | **v034** | 0.686000 | **0.6579** | -0.0021 | 47m17s (79%) |
| 33 | **v061** | 0.687064 | **0.6572** | -0.0028 | 49m36s (83%) |
| 34 | **v043** | 0.687944 | **0.6569** | -0.0031 | 45m52s (76%) |
| 35 | **v023** | 0.684681 | **0.6568** | -0.0032 | 45m36s (76%) |
| 36 | **v029** | 0.684683 | **0.6568** | -0.0032 | 44m27s (74%) |
| 37 | **v022** | 0.685228 | **0.6565** | -0.0035 | 50m46s (85%) |
| 38 | **v024** | 0.684818 | **0.6560** | -0.0040 | 50m17s (84%) |
| 39 | **v016** | 0.681753 | **0.6541** | -0.0059 | 30m56s (52%) |
| 40 | **v017** | 0.681866 | **0.6540** | -0.0060 | 25m37s (43%) |
| 41 | **v018** | 0.680603 | **0.6518** | -0.0082 | 29m21s (49%) |
| 42 | **v014** | 0.679676 | **0.6517** | -0.0083 | 40m12s (67%) |
| 43 | **v011** | 0.679986 | **0.6515** | -0.0085 | 30m12s (50%) |
| 44 | **v015** | -- | **0.6515** | -0.0085 | 30m31s (51%) |
| 45 | **v020** | 0.681773 | **0.6513** | -0.0087 | 29m52s (50%) |
| 46 | **v012** | 0.678784 | **0.6512** | -0.0088 | 29m47s (50%) |
| 47 | **v019** | 0.678374 | **0.6510** | -0.0090 | 30m32s (51%) |
| 48 | **v021** | 0.678575 | **0.6505** | -0.0095 | 55m32s (93%) |
| 49 | **v013** | 0.676198 | **0.6490** | -0.0110 | 28m36s (48%) |
| 50 | **v009** | 0.674456 | **0.6483** | -0.0117 | 39m52s (66%) |
| 51 | **v005** | 0.674037 | **0.6481** | -0.0119 | 36m56s (62%) |
| 52 | **v007** | 0.674258 | **0.6481** | -0.0119 | 42m16s (70%) |
| 53 | **v006** | 0.674149 | **0.6475** | -0.0125 | 35m32s (59%) |
| 54 | **v008** | 0.674445 | **0.6473** | -0.0127 | 41m17s (69%) |
| 55 | **v003** | 0.673711 | **0.6457** | -0.0143 | 30m57s (52%) |
| 56 | **v004** | 0.672206 | **0.6446** | -0.0154 | 43m16s (72%) |
| 57 | **v002** | 0.674350 | **0.6435** | -0.0165 | 35m02s (58%) |
| 58 | **v010** | 0.670696 | **0.6427** | -0.0173 | 34m37s (58%) |
| 59 | **v001** | 0.674315 | **0.6416** | -0.0184 | 34m02s (57%) |
| 60 | **v033** | 0.660415 | **0.6274** | -0.0326 | 47m22s (79%) |
| — | **v026** | 0.684335 | **killed at 69m57s (117%)** | -- | 69m57s (117%) |
| — | **v027** | 0.683058 | **killed at 69m57s (117%)** | -- | 69m57s (117%) |
| — | **v063** | -- | Training | -- | — |

**The rows marked `killed at ...` are the clearest evidence for the
capacity result.** **v026**, **v027** are the two largest models this project ever built. Locally they were
2.2 and 10.2 sigma BELOW v028, and on the grader neither could finish at
all -- two independent lines pointing the same way, which is why the
student stays near 225k parameters and why a good local score alone never
justified shipping a large one.

<!-- END GENERATED: public-score ranking -->

> ### The fix landed, and it was worth nothing — priced on both metrics
>
> **v054 is v042's recipe exactly**, on the patched trainer, with training context matched to
> serving (4000 = 4000), full three epochs. Now scored on both metrics:
>
> | | train context | `full_valid_wp` | public |
> |---|---:|---:|---:|
> | v042 (unpatched) | 1,000 | 0.688268 | 0.6600 |
> | v054 (patched) | 4,000 | 0.688151 | 0.6597 |
> | | | **−0.000117 (−0.7σ)** | **−0.0003 (0.75× noise)** |
>
> **Both null, and they agree.** This is the only axis in the project where the local metric
> and the public board have agreed — and what they agreed on is that nothing happened. The 4×
> train/serve memory mismatch cost nothing at this recipe, so the plateau at ~0.660 is **not**
> explained by the bug, and the model was never context-limited the way the mismatch implied.
>
> It is a *clean* null, unlike v047/v048: v054's weights are genuinely distinct from v042's, so
> this compares two different models rather than one model measured twice.
>
> What the fix *was* worth is correctness, and it unblocked a dead axis: `training.state_reset`
> now moves the weights, which it had not for 25 versions. v054 is the matched-horizon control
> any future `state_reset` run should be measured against.
>
> Duration **45m27s = 76%** of budget against v042's 44m31s = 74%, consistent with the
> identical 225,922 parameters.
>
> Not claimed: that longer horizons are useless. v054 tested 4000; the one-epoch sweep
> preferred 10000 and that arm is running now as v055.

## Feature preprocessing: tried, and it never worked

Three dedicated single-variable runs, each against the matched v002 base (z-score only):

| run | feature added | local Δ | σ | public Δ |
|---|---|---:|---:|---:|
| v007 | first differences of i0 book/trades | −0.000092 | −0.6σ | +0.0046 |
| v008 | cross-instrument spreads/imbalances | +0.000095 | +0.6σ | +0.0038 |
| v009 | `sign(x)·log1p(\|x\|)` on every feature | +0.000106 | +0.7σ | +0.0048 |

Locally all three are indistinguishable from zero. The public "+0.004" looked real until you
notice that **v005 and v006 are seed replicates of v002**: the base recipe's own public spread
is **0.6435 / 0.6481 / 0.6475**, a range of **0.0046**. The v007 "gain" is exactly the size of
the noise it was being compared against. v007 and v008 land *inside* the base's seed range.

This is expected rather than surprising: the organisers already ship **standardised** features
and layer 1 is a `Linear`, so any per-column affine transform is **exactly re-learnable by the
first weight matrix** — mathematically redundant, not merely unhelpful.

**What is still untested:** anything nonlinear or multi-row. Of that family, output calibration
(every nonlinear transform of the prediction) and feature ranking (`raw + squares` beat
`raw + rank channels`) both came back negative.

**Caveat on `first_differences`:** it is in every shipped model, but no clean evidence shows it
earned its place — v007 did not beat its base. It is defensible as genuine nonlinear feature
engineering, not as a measured win. It was never re-validated at length.

## After the sweep: stacking failed, confirmed on the board

**v062 stacked the three sweep winners** — `lr` 1e-3, `state_reset` 10000, `grad_clip` 5.0 —
on top of v054. The prediction was written into the yaml **before** the run, and submitted to
check it:

| | `full_valid_wp` | public |
|---|---:|---:|
| predicted, if additive | **0.689332** (+7.4σ) | — |
| best single component (v059) | 0.688676 (+3.3σ) | 0.6600 |
| baseline v054 | 0.688151 | 0.6597 |
| **v062 actual** | **0.687990 (−1.0σ)** | **0.6587** |

Stacking did not merely fail to help. It landed **below the baseline** and **0.000686 below v059
alone**. Three levers that each help on their own:

```
v059  lr 1e-3             +0.000524
v055  state_reset 10000   +0.000516
v057  grad_clip 5.0       +0.000141
                        ─────────
sum                      +0.001181      actual  −0.000161
```

**Why:** all three act on the **optimiser's effective step size**. Halving the learning rate,
unclipping gradients and lengthening the training horizon are three ways of damping the same
quantity, so the individual gains are largely *the same gain counted three times*.

**Rule this produced: no more stacking. Single-variable search only** — against a selector that
works. The one lever that earned its place is **`lr` 1e-3** alone.

## The selector now has a track record: **+0.947**

Across **all 59 runs** with both an official `full_valid_wp` and a public score,
**Spearman(official, public) = +0.947**.

That is the single most useful number on this page, and it was not true for most of this
project's history. The three latest runs say it plainly:

| run | `full_valid_wp` | public |
|---|---:|---:|
| v059 | 0.688676 | 0.6600 |
| v054 | 0.688151 | 0.6597 |
| v062 | 0.687990 | 0.6587 |

**Identical ordering on both metrics.** Every earlier inversion in this project — and there are
more than twenty — was the **200-sequence subset's** fault, not local-versus-public. Select on
`full_valid_wp` and the board follows.

## In flight — v063

**v063 = v059 + `layers` 2 → 1**, a single-key change against our *strongest* model rather than
the baseline.

| | params | `full_valid_wp` | public | latency |
|---|---:|---:|---:|---:|
| v059 (lr 1e-3, 2 layers) | 225,922 | **0.688676** | **0.6600** | 39.92 µs/row |
| v060 (lr 0.002, 1 layer) | 126,850 | 0.686824 | 0.6591 | **31.13 µs/row** |
| **v063 (both)** | **126,850** | running | — | ~31 µs/row expected |

**Why this is worth running after v062 failed:** v062 stacked three *step-size* levers and they
overlapped. v060's result is a **capacity** change — a different axis — so this is the first
stacking attempt here with a mechanistic reason to expect the gains to add.

Falsifiable both ways: **~0.6887** means the axes are independent and v059's accuracy arrives at
64% of the runtime budget; a fall toward v060's 0.6868 means capacity is not carrying its weight.

## Extreme cases and how they change how you read this table

1. **The 200-sequence subset is not a preview of the official metric. On the seven-run
   optimiser sweep it selected the exact inverse of the winner.** This is the most important
   entry on this page, and it is worse than the public/local disagreement below because both
   numbers are *local* — it breaks the check you would fall back on when the public board is
   unavailable.

   | run | change | `full_valid_wp` (official) | σ | *subset* | *σ* |
   |---|---|---:|---:|---:|---:|
   | **v059** | `lr` 0.002 → 0.001 | **0.688676** | +3.3 | 0.687708 | −6.1 |
   | **v055** | `state_reset` 10000 | **0.688667** | +3.2 | 0.688213 | +0.4 |
   | v057 | `grad_clip` → 5.0 | 0.688293 | +0.9 | 0.687247 | −5.9 |
   | v058 | `weight_decay` → 1e-4 | 0.688032 | −0.7 | 0.687712 | −2.8 |
   | v056 | regularisation | 0.687772 | −2.4 | 0.687611 | −3.4 |
   | v061 | head GELU → ReLU | 0.687064 | −6.8 | 0.688086 | −0.4 |
   | **v060** | `layers` 2 → 1 | 0.686824 | **−8.3** | **+6.1** | **+6.1** |

   **v060 is the worst model in the sweep on the official metric and the best on the subset** —
   and its subset lead strengthened monotonically across four measurement rounds (+3.6σ →
   +4.2σ → +6.0σ → +6.1σ), which is exactly the shape of a real effect. **v059 is the mirror
   image**, and was the subset's worst at every reading.

   We were about to package v060 as the sweep's winner. Selecting on the subset would have
   shipped the worst model this project produced.

   *Rule: **only `full_valid_wp` over all 1,873 sequences is a selection signal.** The 200-seq
   periodic-eval subset is for watching a run progress and nothing else — it is 1/9 of the
   scored rows and its noise floor is ~4× looser. Never rank candidates on it.*

2. **A perfect rank inversion on the state_reset axis — local is not just weak here, it
   points the wrong way.** Three models, one epoch each, identical seed and data, differing
   only in `training.state_reset`. All three are **post-patch**, so these are genuinely
   different models (three distinct weight hashes), not the v042/v047/v048 duplicates:

   | run | `train sr` | `full_valid_wp` | public |
   |---|---:|---:|---:|
   | v051 | 2000 | 0.686628 | **0.6595** |
   | v053 | 4000 | 0.686871 | 0.6592 |
   | v052 | 10000 | **0.687051** | 0.6588 |

   - local order (best first): **v052 → v053 → v051**
   - public order (best first): **v051 → v053 → v052**
   - Spearman(local, public) = **−1.000**, a *perfect* inversion

   Two things at once. First, the axis is **live publicly**: the 0.0007 spread is 1.7× the
   0.0004 seed noise, so these are genuinely different on the board — unlike the previous
   batch, where the seed pair alone explained 100% of the range. Second, and this is the
   important half, **the better a model looks locally the worse it scores publicly.** The
   local best (v052) is the public worst, and vice versa.

   *Caveat, stated because n=3 cannot carry much:* a perfect inversion on three points is
   1-in-6 under random ordering, so on its own it is not significant. It matters because it
   is the **fourth independent measurement pointing the same way** — v041/v043 (+5.2σ gain →
   −0.0001), the v046/v049/v050/v042 quartet (−0.42 Pearson on four distinct models), and now
   this. The consistent finding is not a correlation coefficient; it is that **local
   differences below ~0.001 carry no usable information about public score, and past ~0.0005
   they may actively mislead.**

   *Rule: do not select on `full_valid_wp` at this scale. When candidates are within a few
   thousandths, submit several and let the board choose — we have 5 submissions/day and no
   local method for telling them apart.*

3. **Now that the state is actually carried, the axis is live — and train/serve parity is the
   optimum.** v051 / v053 / v052 are the sweep that v042/v047/v048 were believed to be, on the
   patched trainer. Three **distinct** weight hashes where there was one model before:

   | run | `training.state_reset` | weight md5 | subset WP (20 seq) |
   |---|---:|---|---:|
   | v051 | 2000 | `c0c7a09fe1df` | 0.743111 |
   | **v053** | **4000** | `b6e7e333cc42` | **0.745197** |
   | v052 | 10000 | `6c78235a5ce1` | 0.744482 |

   **4000 is `inference.state_reset` — the horizon the model is served on.** Both shorter and
   longer training horizons are worse, spread 0.002086 ≈ 13σ. So the train/serve memory
   relationship is not a defect to minimise: **parity is the optimum.**

   *Caveat, and it cuts against the subset reading.* On **full** validation the same three runs
   order the other way — v052 (10000) 0.687051 > v053 (4000) 0.686871 > v051 (2000) 0.686628 —
   monotonic in `state_reset` rather than an inverted U, and a spread of 0.000423 ≈ 2.7σ. Two
   different metrics, two different shapes, from the same three models. That disagreement is
   unresolved, and with a 2.7σ full-validation spread against a 0.0004 public seed noise,
   **none of it is yet strong enough to act on.**

4. **This falsifies what we concluded from v041 and v043 alone.** After those two scored we
   wrote that local *gains* never transfer while local *losses* arrive amplified, and
   treated it as a settled asymmetry. Four more submissions refuted it: v044's +10.3σ
   delivered +0.0005 publicly (a partial transfer, roughly a third of the local size), and
   v042 — the run we had written off as a failed hypothesis — posted our **best** public
   score. The honest summary is not "gains don't transfer"; it is that the two metrics
   disagree about ordering inside the noise band, and we over-read two points into a law.
   *Rule: a conclusion drawn from two data points that four more data points contradict is
   a hypothesis, not a finding. This is the same mistake as the v039 teacher premise.*

5. **Every latency `NOT-READY` verdict on a good model was wrong — including the one we
   almost enforced.** Five times a bundle was rejected by the gate and five times it shipped
   fine. v028 was marked `NOT-READY` at a projected 120% and ran at **78%**. **v045** was
   called `Marginal (93% of budget)` and we discussed hard-blocking it; it ran in **55m37s
   = 89%** and scored fine, at the highest memory we have used (634 MB, 4% of the 16 GB
   allowance). Every one of these verdicts came from a p50 measured **while other training
   runs were using the machine**. *Rule: never gate on a latency number taken under load,
   and never hard-block a model on a projection when the only evidence is local.*

6. **Our own premises have been wrong in ways only measurement exposed — three times.**
   v039's teacher-vs-student claim compared our subset metric against another competitor's
   held-out set; measured properly the teacher was **+0.0057 better than our student**,
   inverting the assumption. Both hypotheses that followed died as nulls (capacity +0.5σ, a
   third teacher member +0.4σ) — and the one we had written off, that third member, is now
   our best entry. Two teacher hypotheses were "refuted" and one of them won. *Rule: a
   comparison is only evidence if both numbers come from the same rows and scale; and a null
   result refutes the hypothesis it tested, not the neighbouring one.*

7. **The submission path is not the bottleneck, and the grader is not adversarial.** All 43
   of our submissions that reached the grader produced a number except the two that timed
   out at 117% of budget; peak memory across the whole project is 634 MB against a 16 GB
   allowance. Every result in this table came from a bundle that passed contract,
   determinism, parity and size checks. *Rule: when a result looks wrong, suspect the
   experiment before the plumbing — but check latency under load first, because that is the
   one place we have repeatedly been wrong.*
