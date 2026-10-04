"""Decision: should `solution.py` transform its output before returning it?

If a transform buys more than +/-0.0003 of WP we adopt it and ship it, because it costs
almost nothing at serving time (a multiply and a clip per row, no extra ONNX op, no extra
buffer). If it does not, we leave `solution.py` alone and stop considering the question.

WHY THIS IS THE FIRST QUESTION AND NOT A CURIOUSITY
--------------------------------------------------
Two facts make calibration the highest-value thing we have not yet checked:

1. The metric CLIPS both prediction and target to [-2, 2] before correlating. `solution.py`
   already applies the same METRIC_CLIP = 2.0, so clipping at 2.0 is a no-op -- but a
   TIGHTER clip is not, and nothing has ever tested one. Clipping is a monotone transform,
   and Pearson is invariant to positive *scaling*, so scaling our output cannot change the
   score at all. Only a NONLINEAR monotone transform can, which is exactly what clipping
   and the power family below are.

2. Targets are heavy-tailed and heavily zero-inflated (10.94% exactly zero over the full
   train split, so those rows carry no weight at all). If our model is systematically
   over-confident in the tails -- predicting 1.8 where the target is usually 0.5 -- then
   squashing the tails raises the global correlation without changing any ranking. That is
   a pure win and it is invisible to any metric we have computed so far, because every
   score we have is Pearson against a metric that only sees the correlation and not the
   shape of the prediction cloud.

The same script decomposes the error by |target| decile, because that says WHERE the
correlation is being lost. If the loss is concentrated in the top decile -- the rows the
metric weights most -- then calibration is worth attacking. If the loss is uniform, the
model is simply not discriminating and no output transform will help, and that is worth
knowing before we spend more compute on preprocessing.

VALIDATION ONLY. Nothing here feeds a model input. Uses the reference metric from
tools/metric_ref.py on the official `need_prediction AND is_scored` mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from metric_ref import weighted_pearson  # noqa: E402

FIGDIR = Path(__file__).resolve().parent / "images"
SEQ, WARMUP, CLIP = 20_000, 99, 2.0


def wp(y, p, sel, per_target=True):
    """Reference WP on an explicit mask. Mirrors METRIC.md exactly."""
    if not sel.any():
        return float("nan")
    if per_target:
        return float(np.mean([weighted_pearson(y[sel, k], p[sel, k]) for k in range(2)]))
    return float(weighted_pearson(y[:, 0], p[:, 0]))


def main() -> int:
    ver = sys.argv[1] if len(sys.argv) > 1 else "v044"
    outdir = ROOT / "cache" / "preds"
    P = np.load(outdir / f"{ver}_valid.npy", mmap_mode="r")
    Y = np.load(ROOT / "cache/valid/y.npy", mmap_mode="r")
    N = np.load(ROOT / "cache/valid/need_prediction.npy", mmap_mode="r")
    S = np.load(ROOT / "cache/valid/is_scored.npy", mmap_mode="r")
    meta = json.loads((outdir / f"{ver}_meta.json").read_text())
    rows = meta["rows"]
    print(f"  model {ver} | {meta['n_sequences']} sequences | {rows:,} rows")

    y = np.asarray(Y[:rows], np.float32)
    p = np.asarray(P[:rows], np.float32)
    need = np.asarray(N[:rows], bool)
    sc = np.asarray(S[:rows], bool)
    sel = need & sc

    base = wp(y, p, sel)
    print(f"\n  BASELINE (as shipped)  WP {base:.6f}   scored rows {int(sel.sum()):,}")

    # ---------------- 1. prediction cloud vs target cloud ----------------
    ps = p[sel]
    ys = y[sel]
    print(f"\n  distribution on scored rows")
    for k, nm in ((0, "t0"), (1, "t1")):
        print(f"    {nm}: pred mean {ps[:,k].mean():+.4f} std {ps[:,k].std():.4f} "
              f"max|p| {np.abs(ps[:,k]).max():.3f}   |   "
              f"target mean {ys[:,k].mean():+.4f} std {ys[:,k].std():.4f} "
              f"max|y| {np.abs(ys[:,k]).max():.3f}")
    at_clip = np.mean(np.abs(ps) >= CLIP - 1e-6)
    print(f"    fraction of predictions AT the +/-2 clip : {at_clip*100:.4f}%")
    print(f"    fraction of TARGETS beyond +/-2         : "
          f"{np.mean(np.abs(ys) > CLIP)*100:.4f}%")
    print(f"    fraction of targets exactly 0.0        : "
          f"{np.mean(ys == 0)*100:.4f}%   (these carry ZERO metric weight)")
    ratio = np.abs(ps).mean() / max(np.abs(ys).mean(), 1e-9)
    print(f"    mean|pred| / mean|target|               : {ratio:.4f} "
          f"({'OVER-confident in the tails' if ratio > 1.15 else 'under/ok' if ratio > 0.85 else 'UNDER-confident'})")

    # ---------------- 2. transform sweep ----------------
    print(f"\n  monotone output transforms (each costs ~0 at serving time)")
    print(f"  {'transform':34}{'WP':>11}{'delta':>11}{'sig':>8}")
    results = []

    def rec(name, q):
        v = wp(y, q, sel)
        d = v - base
        results.append((name, float(v), float(d)))
        print(f"  {name:34}{v:11.6f}{d:+11.6f}{d/0.000159:+8.1f}")

    for c in (2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6):
        rec(f"clip at +/-{c:.1f}", np.clip(p, -c, c))
    for g in (0.3, 0.5, 0.7, 0.85, 1.15, 1.4):
        rec(f"power p*|p|^{g}", np.sign(p) * np.abs(p) ** g)
    for k in (1.05, 1.15, 1.4):
        rec(f"tanh {k}*p (soft clip)", np.tanh(k * p))
    best = max(results, key=lambda r: r[1])
    worst = min(results, key=lambda r: r[1])
    print(f"\n  BEST  {best[0]:34} {best[1]:.6f}  ({best[2]:+.6f}, {best[2]/0.000159:+.1f} sigma)")
    print(f"  WORST {worst[0]:34} {worst[1]:.6f}  ({worst[2]:+.6f})")
    verdict = ("ADOPT" if best[2] > 2 * 0.000159 else
               "REJECT -- inside noise" if abs(best[2]) <= 2 * 0.000159 else
               "investigate sign")
    print(f"  VERDICT: {verdict}   (threshold is +/-{2*0.000159:.6f} = 2 sigma)")

    # ---------------- 3. where is the correlation lost? ----------------
    # Decile of |target| on scored rows. The metric weights each row by |clip(y)|, so a
    # per-decile score shows whether we are strong where the weight is or not.
    a = np.abs(np.clip(ys, -CLIP, CLIP)).mean(axis=1)
    order = np.argsort(a)
    nb = 10
    print(f"\n  WP by |target| decile (rows ordered by mean |clip(y)| across both targets)")
    print(f"  {'decile':>8}{'rows':>9}{'meanW':>8}{'share%':>8}{'WP t0':>9}{'WP t1':>9}{'WP mean':>9}")
    edges = np.linspace(0, len(a), nb + 1).astype(int)
    dec_rows = []
    for i in range(nb):
        idx = order[edges[i]:edges[i + 1]]
        if len(idx) < 50:
            continue
        w = np.abs(np.clip(ys[idx], -CLIP, CLIP)).sum()
        w0 = weighted_pearson(ys[idx, 0], ps[idx, 0])
        w1 = weighted_pearson(ys[idx, 1], ps[idx, 1])
        share = np.abs(np.clip(ys[idx], -CLIP, CLIP)).sum() / np.abs(np.clip(ys, -CLIP, CLIP)).sum() * 100
        dec_rows.append((i, len(idx), share, w0, w1))
        print(f"  {i:8d}{len(idx):9d}{w/np.abs(ys).sum():8.3f}{share:8.2f}{w0:9.4f}{w1:9.4f}{(w0+w1)/2:9.4f}")
    print(f"\n  If WP falls as |target| rises, the model is worse exactly where the metric")
    print(f"  weights most -- and a tail transform is the only cheap lever. If it is flat")
    print(f"  or rising, the model is uniformly mediocre and no output transform will help.")

    # ---------------- plot ----------------
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    a1 = ax[0, 0]
    names = [r[0] for r in results]
    vals = [r[1] for r in results]
    a1.axhline(base, color="k", ls="--", lw=1, label=f"as shipped {base:.6f}")
    a1.axhspan(base - 2 * 0.000159, base + 2 * 0.000159, color="grey", alpha=.25,
               label="+/-2 sigma (noise)")
    a1.barh(range(len(vals)), np.array(vals) - base, color=["g" if v > 2 * 0.000159 else "r" if v < -2 * 0.000159 else "b" for v in np.array(vals) - base])
    a1.set_yticks(range(len(names)))
    a1.set_yticklabels(names, fontsize=7)
    a1.set_xlabel("WP change vs as-shipped")
    a1.set_title(f"{ver}: monotone output transform sweep")
    a1.legend(fontsize=8, loc="lower right")

    a2 = ax[0, 1]
    qs = np.linspace(0, 1, 101)
    for k, c in ((0, "C0"), (1, "C3")):
        a2.plot(np.quantile(ps[:, k], qs), np.quantile(ys[:, k], qs), c, lw=2, label=f"t{k}")
    a2.plot([-3, 3], [-3, 3], "k--", lw=1, label="identity")
    a2.set_xlabel("prediction quantile"); a2.set_ylabel("target quantile")
    a2.set_title("Q-Q: prediction vs target (ideal = identity line)"); a2.legend(fontsize=8)

    a3 = ax[1, 0]
    a3.plot([d[0] for d in dec_rows], [d[3] for d in dec_rows], "o-", label="t0")
    a3.plot([d[0] for d in dec_rows], [d[4] for d in dec_rows], "o-", label="t1")
    a3.set_xlabel("|target| decile (low -> high)")
    a3.set_ylabel("WP within decile")
    a3.set_title("WP by |target| decile"); a3.legend(); a3.grid(alpha=.3)

    a4 = ax[1, 1]
    a4.bar([d[0] for d in dec_rows], [d[2] for d in dec_rows], color="purple")
    a4.set_xlabel("|target| decile"); a4.set_ylabel("% of total metric weight")
    a4.set_title("Share of SUM|clip(y)| by decile")
    a4.grid(alpha=.3)

    fig.tight_layout()
    out = FIGDIR / "prediction_calibration_by_transform.png"
    fig.savefig(out, dpi=110)
    print(f"\n  figure -> {out.relative_to(ROOT)}")

    (ROOT / "exploration" / "visualisation" /
     f"prediction_calibration_{ver}.json").write_text(json.dumps({
        "model": ver, "baseline_wp": base,
        "transforms": [{"name": n, "wp": v, "delta": d} for n, v, d in results],
        "best": {"name": best[0], "wp": best[1], "delta": best[2]},
        "verdict": verdict,
        "pred_mean_abs_over_target_mean_abs": float(ratio),
        "frac_pred_at_clip": float(at_clip),
        "deciles": [{"decile": int(d[0]), "rows": int(d[1]), "weight_share_pct": float(d[2]),
                     "wp_t0": float(d[3]), "wp_t1": float(d[4])} for d in dec_rows],
    }, indent=2) + "\n")
    print(f"  numbers -> exploration/visualisation/prediction_calibration_{ver}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())