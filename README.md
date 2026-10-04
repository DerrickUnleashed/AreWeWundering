# Alfa Connectome — public results

**One table, ranked by public score.** This is the only number that decides the prize, so it
is the only one on this page. Everything else — 36 sections of experiments, hypotheses,
refutations, noise floors, latency post-mortems and bugs — is in **[RESUME.md](RESUME.md)**,
which is the working memory for the next session.

Rules and conventions are in **[AGENTS.md](AGENTS.md)**. Raw scores, grader durations and
memory for every submission are in `exploration/leaderboard.json`.

## Public leaderboard

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
| 1 | **v039** | 0.688212 | **0.6594** | +0.0000 | 47m42s (80%) | scored |
| 2 | **v041** | 0.689036 | **0.6593** | -0.0001 | 46m51s (78%) | scored |
| 3 | **v037** | 0.688370 | **0.6592** | -0.0002 | 48m06s (80%) | scored |
| 4 | **v038** | 0.687577 | **0.6588** | -0.0006 | 48m06s (80%) | scored |
| 5 | **v032** | 0.687219 | **0.6587** | -0.0007 | 45m32s (76%) | scored |
| 6 | **v031** | 0.687376 | **0.6587** | -0.0007 | 45m41s (76%) | scored |
| 7 | **v036** | 0.687535 | **0.6586** | -0.0008 | 48m06s (80%) | scored |
| 8 | **v035** | 0.687177 | **0.6586** | -0.0008 | 44m52s (75%) | scored |
| 9 | **v030** | 0.686919 | **0.6585** | -0.0009 | 47m11s (79%) | scored |
| 10 | **v028** | 0.686510 | **0.6585** | -0.0009 | 46m52s (78%) | scored |
| 11 | **v025** | 0.684438 | **0.6580** | -0.0014 | 46m07s (77%) | scored |
| 12 | **v034** | 0.686000 | **0.6579** | -0.0015 | 47m17s (79%) | scored |
| 13 | **v043** | 0.687944 | **0.6569** | -0.0025 | 45m52s (76%) | scored |
| 14 | **v029** | 0.684683 | **0.6568** | -0.0026 | 44m27s (74%) | scored |
| 15 | **v023** | 0.684681 | **0.6568** | -0.0026 | 45m36s (76%) | scored |
| 16 | **v022** | 0.685228 | **0.6565** | -0.0029 | 50m46s (85%) | scored |
| 17 | **v024** | 0.684818 | **0.6560** | -0.0034 | 50m17s (84%) | scored |
| 18 | **v016** | 0.681753 | **0.6541** | -0.0053 | 30m56s (52%) | scored |
| 19 | **v017** | 0.681866 | **0.6540** | -0.0054 | 25m37s (43%) | scored |
| 20 | **v018** | 0.680603 | **0.6518** | -0.0076 | 29m21s (49%) | scored |
| 21 | **v014** | 0.679676 | **0.6517** | -0.0077 | 40m12s (67%) | scored |
| 22 | **v015** | -- | **0.6515** | -0.0079 | 30m31s (51%) | scored |
| 23 | **v011** | 0.679986 | **0.6515** | -0.0079 | 30m12s (50%) | scored |
| 24 | **v020** | 0.681773 | **0.6513** | -0.0081 | 29m52s (50%) | scored |
| 25 | **v012** | 0.678784 | **0.6512** | -0.0082 | 29m47s (50%) | scored |
| 26 | **v019** | 0.678374 | **0.6510** | -0.0084 | 30m32s (51%) | scored |
| 27 | **v021** | 0.678575 | **0.6505** | -0.0089 | 55m32s (93%) | scored |
| 28 | **v013** | 0.676198 | **0.6490** | -0.0104 | 28m36s (48%) | scored |
| 29 | **v009** | 0.674456 | **0.6483** | -0.0111 | 39m52s (66%) | scored |
| 30 | **v007** | 0.674258 | **0.6481** | -0.0113 | 42m16s (70%) | scored |
| 31 | **v005** | 0.674037 | **0.6481** | -0.0113 | 36m56s (62%) | scored |
| 32 | **v006** | 0.674149 | **0.6475** | -0.0119 | 35m32s (59%) | scored |
| 33 | **v008** | 0.674445 | **0.6473** | -0.0121 | 41m17s (69%) | scored |
| 34 | **v003** | 0.673711 | **0.6457** | -0.0137 | 30m57s (52%) | scored |
| 35 | **v004** | 0.672206 | **0.6446** | -0.0148 | 43m16s (72%) | scored |
| 36 | **v002** | 0.674350 | **0.6435** | -0.0159 | 35m02s (58%) | scored |
| 37 | **v010** | 0.670696 | **0.6427** | -0.0167 | 34m37s (58%) | scored |
| 38 | **v001** | 0.674315 | **0.6416** | -0.0178 | 34m02s (57%) | scored |
| 39 | **v033** | 0.660415 | **0.6274** | -0.0320 | 47m22s (79%) | scored |

**Killed by the grader, no score at all:** **v026** (0.684335 local, 69m57s (117%)), **v027** (0.683058 local, 69m57s (117%)).

These are the two largest models this project ever built, and they
are the clearest evidence for the capacity result: locally they were
2.2 and 10.2 sigma BELOW v028, and on the grader neither could finish
at all. Two independent lines agreeing is why the student stays near
225k parameters -- and why a local score alone never justified shipping
a big one.


<!-- END GENERATED: public-score ranking -->

## Extreme cases and how they change how you read this table

1. **A local gain of +5.2σ arrived publicly as −0.0001.** v041 was the largest single-lever
   local gain in the project (`full_valid_wp` 0.689036, +0.000824, +5.2σ against a ±0.000318
   detection threshold) and scored **0.6593** — *below* its own parent. Three separate
   attenuation measurements now disagree (4.3× from one sample, ~0.5 across five levers, 0
   for v041), so there is no usable conversion factor. **Rule: local differences below
   roughly 0.001 have no reliable relationship to public score, in either direction.** Judge
   runs locally, but never treat a small local win as a reason to spend a submission.

2. **A local null arrived publicly as −0.0025, the opposite way.** v043 measured −0.000268
   (−1.7σ) locally and scored **0.6569**, our worst submission since v033 — roughly 9×
   larger than the local signal. So the asymmetry is not "local is noise", it is that small
   local *regressions* arrive amplified. **Rule: a small local loss is worth respecting;
   a small local gain is not worth chasing.**

3. **Every latency `NOT-READY` verdict on a good model was wrong.** Four times a bundle was
   rejected by the gate and four times it shipped comfortably — v028 was marked
   `NOT-READY` at a projected 120% and ran at **78%**. The cause is always the same: latency
   measured **while other training runs were using the machine**. The numbers in this table
   that come from the grader are the only trustworthy ones. **Rule: never gate on a p50 taken
   while other work is running** — re-measure idle, and prefer a measured anchor from an
   identical graph over any local projection.

4. **Our own configs have been wrong in ways that only became visible when measured.** The
   teacher-vs-student premise behind v039 was established by comparing our subset metric
   against another competitor's held-out set — different scales, so the conclusion could not
   hold. Measuring it properly showed the **teacher was +0.0057 better than our student**,
   inverting the assumption. Two follow-up hypotheses from that finding then died as nulls
   (capacity, +0.5σ; a third teacher member, +0.4σ). **Rule: a comparison between two
   numbers is only evidence if both were computed on the same rows and the same scale.**
