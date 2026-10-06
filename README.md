# Alfa Connectome — public results

**One table, ranked by public score.** This is the only number that decides the prize, so it
is the only one on this page. Everything else — 36 sections of experiments, hypotheses,
refutations, noise floors, latency post-mortems and bugs — is in **[RESUME.md](RESUME.md)**,
which is the working memory for the next session.

Rules and conventions are in **[AGENTS.md](AGENTS.md)**. Raw scores, grader durations and
memory for every submission are in `exploration/leaderboard.json`.

> ## ⚠ Read this before trusting any number on this page
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
| 4 | **v044** | 0.689846 | **0.6599** | -0.0001 | 47m41s (79%) |
| 5 | **v046** | 0.689030 | **0.6598** | -0.0002 | 48m32s (81%) |
| 6 | **v049** | 0.688784 | **0.6598** | -0.0002 | 46m07s (77%) |
| 7 | **v040** | 0.689146 | **0.6596** | -0.0004 | 44m57s (75%) |
| 8 | **v050** | 0.688599 | **0.6596** | -0.0004 | 45m57s (77%) |
| 9 | **v045** | 0.689116 | **0.6595** | -0.0005 | 55m37s (93%) |
| 10 | **v051** | 0.686628 | **0.6595** | -0.0005 | 46m47s (78%) |
| 11 | **v039** | 0.688212 | **0.6594** | -0.0006 | 47m42s (80%) |
| 12 | **v041** | 0.689036 | **0.6593** | -0.0007 | 46m51s (78%) |
| 13 | **v037** | 0.688370 | **0.6592** | -0.0008 | 48m06s (80%) |
| 14 | **v053** | 0.686871 | **0.6592** | -0.0008 | 47m02s (78%) |
| 15 | **v038** | 0.687577 | **0.6588** | -0.0012 | 48m06s (80%) |
| 16 | **v052** | 0.687051 | **0.6588** | -0.0012 | 44m51s (75%) |
| 17 | **v031** | 0.687376 | **0.6587** | -0.0013 | 45m41s (76%) |
| 18 | **v032** | 0.687219 | **0.6587** | -0.0013 | 45m32s (76%) |
| 19 | **v035** | 0.687177 | **0.6586** | -0.0014 | 44m52s (75%) |
| 20 | **v036** | 0.687535 | **0.6586** | -0.0014 | 48m06s (80%) |
| 21 | **v028** | 0.686510 | **0.6585** | -0.0015 | 46m52s (78%) |
| 22 | **v030** | 0.686919 | **0.6585** | -0.0015 | 47m11s (79%) |
| 23 | **v025** | 0.684438 | **0.6580** | -0.0020 | 46m07s (77%) |
| 24 | **v034** | 0.686000 | **0.6579** | -0.0021 | 47m17s (79%) |
| 25 | **v043** | 0.687944 | **0.6569** | -0.0031 | 45m52s (76%) |
| 26 | **v023** | 0.684681 | **0.6568** | -0.0032 | 45m36s (76%) |
| 27 | **v029** | 0.684683 | **0.6568** | -0.0032 | 44m27s (74%) |
| 28 | **v022** | 0.685228 | **0.6565** | -0.0035 | 50m46s (85%) |
| 29 | **v024** | 0.684818 | **0.6560** | -0.0040 | 50m17s (84%) |
| 30 | **v016** | 0.681753 | **0.6541** | -0.0059 | 30m56s (52%) |
| 31 | **v017** | 0.681866 | **0.6540** | -0.0060 | 25m37s (43%) |
| 32 | **v018** | 0.680603 | **0.6518** | -0.0082 | 29m21s (49%) |
| 33 | **v014** | 0.679676 | **0.6517** | -0.0083 | 40m12s (67%) |
| 34 | **v011** | 0.679986 | **0.6515** | -0.0085 | 30m12s (50%) |
| 35 | **v015** | -- | **0.6515** | -0.0085 | 30m31s (51%) |
| 36 | **v020** | 0.681773 | **0.6513** | -0.0087 | 29m52s (50%) |
| 37 | **v012** | 0.678784 | **0.6512** | -0.0088 | 29m47s (50%) |
| 38 | **v019** | 0.678374 | **0.6510** | -0.0090 | 30m32s (51%) |
| 39 | **v021** | 0.678575 | **0.6505** | -0.0095 | 55m32s (93%) |
| 40 | **v013** | 0.676198 | **0.6490** | -0.0110 | 28m36s (48%) |
| 41 | **v009** | 0.674456 | **0.6483** | -0.0117 | 39m52s (66%) |
| 42 | **v005** | 0.674037 | **0.6481** | -0.0119 | 36m56s (62%) |
| 43 | **v007** | 0.674258 | **0.6481** | -0.0119 | 42m16s (70%) |
| 44 | **v006** | 0.674149 | **0.6475** | -0.0125 | 35m32s (59%) |
| 45 | **v008** | 0.674445 | **0.6473** | -0.0127 | 41m17s (69%) |
| 46 | **v003** | 0.673711 | **0.6457** | -0.0143 | 30m57s (52%) |
| 47 | **v004** | 0.672206 | **0.6446** | -0.0154 | 43m16s (72%) |
| 48 | **v002** | 0.674350 | **0.6435** | -0.0165 | 35m02s (58%) |
| 49 | **v010** | 0.670696 | **0.6427** | -0.0173 | 34m37s (58%) |
| 50 | **v001** | 0.674315 | **0.6416** | -0.0184 | 34m02s (57%) |
| 51 | **v033** | 0.660415 | **0.6274** | -0.0326 | 47m22s (79%) |
| — | **v054** | 0.688151 | Ready to Submit | -- | — |
| — | **v026** | 0.684335 | **killed at 69m57s (117%)** | -- | 69m57s (117%) |
| — | **v027** | 0.683058 | **killed at 69m57s (117%)** | -- | 69m57s (117%) |

**The rows marked `killed at ...` are the clearest evidence for the
capacity result.** **v026**, **v027** are the two largest models this project ever built. Locally they were
2.2 and 10.2 sigma BELOW v028, and on the grader neither could finish at
all -- two independent lines pointing the same way, which is why the
student stays near 225k parameters and why a good local score alone never
justified shipping a large one.

<!-- END GENERATED: public-score ranking -->

> ### The fix landed, and it was worth nothing — which is the useful part
>
> **v054 is v042's recipe exactly**, on the patched trainer, with training context matched
> to serving (4000 = 4000), full three epochs:
>
> | | train context | `full_valid_wp` |
> |---|---|---:|
> | v042 (unpatched) | 1,000 | 0.688268 |
> | v054 (patched) | 4,000 | 0.688151 |
> | | | **−0.000117 (−0.7σ)** |
>
> **A null.** The 4× train/serve memory mismatch cost nothing at this recipe. So the plateau
> at ~0.660 is **not** explained by the bug, and the model was never context-limited the way
> the mismatch implied — surprising, given the target's 0.977 lag-1 autocorrelation.
>
> What the fix *was* worth is correctness, and it unblocked a dead axis: `training.state_reset`
> now moves the weights, where before it was inert. v054 is also the first bundle whose model
> trained with the state actually carrying, and it is the matched-horizon control any future
> `state_reset` run should be measured against.
>
> Not claimed: that longer horizons are useless. v054 tested 4000; the one-epoch sweep put
> 10000 best on full validation, and that arm was never run at length.

## Extreme cases and how they change how you read this table

1. **A perfect rank inversion on the state_reset axis — local is not just weak here, it
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

2. **Now that the state is actually carried, the axis is live — and train/serve parity is the
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

2. **This falsifies what we concluded from v041 and v043 alone.** After those two scored we
   wrote that local *gains* never transfer while local *losses* arrive amplified, and
   treated it as a settled asymmetry. Four more submissions refuted it: v044's +10.3σ
   delivered +0.0005 publicly (a partial transfer, roughly a third of the local size), and
   v042 — the run we had written off as a failed hypothesis — posted our **best** public
   score. The honest summary is not "gains don't transfer"; it is that the two metrics
   disagree about ordering inside the noise band, and we over-read two points into a law.
   *Rule: a conclusion drawn from two data points that four more data points contradict is
   a hypothesis, not a finding. This is the same mistake as the v039 teacher premise.*

3. **Every latency `NOT-READY` verdict on a good model was wrong — including the one we
   almost enforced.** Five times a bundle was rejected by the gate and five times it shipped
   fine. v028 was marked `NOT-READY` at a projected 120% and ran at **78%**. **v045** was
   called `Marginal (93% of budget)` and we discussed hard-blocking it; it ran in **55m37s
   = 89%** and scored fine, at the highest memory we have used (634 MB, 4% of the 16 GB
   allowance). Every one of these verdicts came from a p50 measured **while other training
   runs were using the machine**. *Rule: never gate on a latency number taken under load,
   and never hard-block a model on a projection when the only evidence is local.*

4. **Our own premises have been wrong in ways only measurement exposed — three times.**
   v039's teacher-vs-student claim compared our subset metric against another competitor's
   held-out set; measured properly the teacher was **+0.0057 better than our student**,
   inverting the assumption. Both hypotheses that followed died as nulls (capacity +0.5σ, a
   third teacher member +0.4σ) — and the one we had written off, that third member, is now
   our best entry. Two teacher hypotheses were "refuted" and one of them won. *Rule: a
   comparison is only evidence if both numbers come from the same rows and scale; and a null
   result refutes the hypothesis it tested, not the neighbouring one.*

5. **The submission path is not the bottleneck, and the grader is not adversarial.** All 43
   of our submissions that reached the grader produced a number except the two that timed
   out at 117% of budget; peak memory across the whole project is 634 MB against a 16 GB
   allowance. Every result in this table came from a bundle that passed contract,
   determinism, parity and size checks. *Rule: when a result looks wrong, suspect the
   experiment before the plumbing — but check latency under load first, because that is the
   one place we have repeatedly been wrong.*
