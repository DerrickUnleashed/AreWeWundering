# Rejected bundles — parity gate measured the wrong thing

These four bundles were produced on 2026-09-29 and marked NOT-READY because
`tools/score_solution.py` compared raw streaming vs offline predictions with a
1e-4 tolerance. That tolerance was calibrated on `gru_v000`, a 60-step
undertrained model whose outputs are nearly constant.

A fully trained model run row-by-row in float32 through 20,000 recurrent steps
drifts from the batched PyTorch path by ~3.5e-4 in the raw predictions. That is
accumulation, not a bug: per-step ONNX-vs-PyTorch agreement is 2.8e-6 and the
divergence grows monotonically with step index (1.6e-7 at step 0, 6e-5 at step
10,000). Critically, it moves the score by **3.3e-7** — Global WP is identical to
7 significant figures between the two paths.

The gate was changed to test the score (tol 1e-4) with the raw bound retained
only as a bug tripwire (tol 1e-3, since a real implementation bug is O(1e-1) or
larger). The bundles here are kept for the record and are NOT valid submissions.
