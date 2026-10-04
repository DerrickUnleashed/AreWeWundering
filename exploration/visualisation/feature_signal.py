#!/usr/bin/env python3
"""Decision: is there input-side signal our architecture cannot represent?

If a NONLINEAR transform of the 112 features correlates with the targets materially better
than the raw features do, we build that transform and fold it into serving. If every strong
relationship is already linear -- or is already visible to the network through features we
already feed -- we stop, because a linear transform is provably redundant.

THE ARGUMENT THAT KILLS MOST PREPROCESSING, STATED UP FRONT
-----------------------------------------------------------
Our first layer is linear: 112 -> hidden. Therefore ANY linear preprocessing we could
invent -- robust z-score, per-column centring, cross-instrument linear combinations, feature
selection, PCA, whitening -- is already inside the model's hypothesis class. The network can
represent it whether or not we compute it. AGENTS.md sec 10.3 says to fold such transforms
into layer 1 at export, which is the same statement: they cost nothing because they change
nothing.

So this script is not asking "can we normalise the inputs" (yes, and it is pointless). It is
asking the only question that matters: **is any of the signal NONLINEAR, and is any of it
MISSING from what the model already sees?** Three concrete sub-questions, each of which
changes what we build:

  1. WHERE is the signal? Per-feature correlation, then aggregated by group (i0 book/trade,
     i1 book/trade, a0-a7). If one group carries almost everything, a preprocessing effort
     aimed elsewhere is wasted.
  2. Is the signal ALREADY REACHABLE? Our hidden state is 128 wide and carries state across
     the whole 20k-step sequence, so anything expressible as a function of the recent past
     is in principle reachable. What is NOT reachable is instantaneous nonlinearity of a
     single row if the first layer is linear and there are no such terms in the feature set.
     We look for feature NORMALITY as the proxy: a saturating or heavy-tailed feature whose
     linear correlation badly understates its relationship.
  3. Is there exploitable TEMPORAL structure the model might be missing? Target
     autocorrelation at lag 1..20 and target-vs-recent-history correlations. A GRU over the
     raw rows should already capture these, so this is a sanity check on the architecture
     rather than a search -- but it is cheap and it would be embarrassing to miss.

TRAIN SPLIT ONLY. Nothing here feeds a model input. Rows are streamed from the memmap
shards in chunks; no EDA script opens datasets/*.parquet.
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
FIGDIR = Path(__file__).resolve().parent / "images"

# Column order from utils.FEATURE_COLUMNS / docs/data_overview.md. The INDEX is an
# identifier, never a depth level, so these must never be sorted alphabetically --
# i0_p10 would land next to i0_p1 and silently reorder every per-column plot.
def names() -> list[str]:
    cols = []
    for inst in ("i0", "i1"):
        cols += [f"{inst}_p{i}" for i in range(22)]
        cols += [f"{inst}_v{i}" for i in range(22)]
        cols += [f"{inst}_dp{i}" for i in range(4)]
        cols += [f"{inst}_dv{i}" for i in range(4)]
    cols += [f"a{i}" for i in range(8)]
    return cols


def groups() -> dict[str, list[int]]:
    n = names()
    g = {k: [] for k in ("i0_price", "i0_vol", "i0_tradeP", "i0_tradeV",
                         "i1_price", "i1_vol", "i1_tradeP", "i1_tradeV", "a0-7")}
    for i, c in enumerate(n):
        if c.startswith("a"):
            g["a0-7"].append(i)
        else:
            key = ("i0" if c.startswith("i0") else "i1") + "_" + {
                "p": "price", "v": "vol", "dp": "tradeP", "dv": "tradeV"}[c.split("_")[1][0:2].rstrip("0123456789")] \
                if False else None
            base = c[:2]
            pre = c[3:]
            kind = ("price" if pre[0] == "p" else "vol" if pre[0] == "v"
                    else "tradeP" if pre[:2] == "dp" else "tradeV")
            g[f"{base}_{kind}"].append(i)
    return g


def main() -> int:
    n_seq = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    X = np.load(ROOT / "cache/train/X.npy", mmap_mode="r")
    Y = np.load(ROOT / "cache/train/y.npy", mmap_mode="r")
    SEQ = 20_000
    n_all = X.shape[0] // SEQ
    n_seq = min(n_seq, n_all)
    rows = n_seq * SEQ

    nm = names()
    G = groups()
    print(f"  train shard: {n_seq}/{n_all} sequences, {rows:,} rows")
    print(f"  {len(nm)} features, groups: " + ", ".join(f"{k}={len(v)}" for k, v in G.items()))

    # ---- streaming sufficient statistics for per-feature Pearson with each target ----
    # Welford-style accumulators so we never hold more than one chunk.
    sx = np.zeros(len(nm), np.float64); sx2 = np.zeros(len(nm), np.float64)
    sy = np.zeros(2, np.float64); sy2 = np.zeros(2, np.float64)
    sxy = np.zeros((len(nm), 2), np.float64)
    # quantiles need a sample; keep a strided reservoir of raw values
    samp = np.empty((min(rows, 4_000_000), len(nm)), np.float16)
    samp_t0 = np.empty(min(rows, 4_000_000), np.float32)
    k = 0
    for s in range(n_seq):
        lo, hi = s * SEQ, (s + 1) * SEQ
        xb = np.asarray(X[lo:hi], np.float32)
        yb = np.asarray(Y[lo:hi], np.float32)
        sx += xb.sum(0); sx2 += (xb.astype(np.float64) ** 2).sum(0)
        sy += yb.sum(0); sy2 += (yb.astype(np.float64) ** 2).sum(0)
        sxy += (xb.astype(np.float64).T @ yb.astype(np.float64))
        step = max(1, hi // 8)
        take = xb[::step]
        m = min(k + take.shape[0], samp.shape[0])
        samp[k:m] = take[:m - k].astype(np.float16)
        samp_t0[k:m] = yb[::step][:m - k, 0]
        k = m
        if (s + 1) % 100 == 0:
            print(f"    seq {s+1}/{n_seq}", flush=True)
    n = rows
    mx, my = sx / n, sy / n
    vx = np.maximum(sx2 / n - mx ** 2, 1e-12)
    vy = np.maximum(sy2 / n - my ** 2, 1e-12)
    corr = (sxy / n - np.outer(mx, my)) / np.sqrt(np.outer(vx, vy))

    print(f"\n  === 1. WHERE IS THE SIGNAL? (|Pearson| per feature vs each target) ===")
    print(f"  {'group':12}{'n':>5}{'max|r| t0':>11}{'mean|r| t0':>11}{'max|r| t1':>11}{'mean|r| t1':>11}")
    for gname, idx in G.items():
        idx = [i for i in idx if i < len(nm)]
        if not idx:
            continue
        c = corr[idx]
        print(f"  {gname:12}{len(idx):5d}{np.abs(c[:,0]).max():11.4f}"
              f"{np.abs(c[:,0]).mean():11.4f}{np.abs(c[:,1]).max():11.4f}{np.abs(c[:,1]).mean():11.4f}")
    order = np.argsort(-np.abs(corr[:, 0]))
    print(f"\n  strongest 12 features vs t0:")
    for i in order[:12]:
        print(f"    {nm[i]:10} r={corr[i,0]:+.4f}   std={np.sqrt(vx[i]):.3f}"
              f"   r/sd={corr[i,0]/np.sqrt(vx[i])*np.sqrt(vy[0]):+.5f}")

    # ---- 2. is anything nonlinear / missing? ----
    print(f"\n  === 2. NONLINEARITY CHECK ===")
    print(f"  A linear model cannot gain from transforming a feature unless the transform")
    print(f"  changes its relationship with the target. Heavier tails and saturation are the")
    print(f"  candidates, so we report tail mass and a rank correlation alongside Pearson.")
    sm = samp[:k]
    st0 = samp_t0[:k]
    from scipy_free_rank import spearman, _avg_rank  # local helper, no scipy dependency
    # Weights matter here: the metric weights each row by |clip(t0)|, and t0 is
    # heavy-tailed, so an UNWEIGHTED correlation is not the quantity the metric uses.
    wts = np.abs(np.clip(st0, -2, 2))
    tkurt = float(((st0 - st0.mean()) ** 4).mean() / st0.std() ** 4 - 3)
    print(f"  target t0 kurtosis {tkurt:+.2f} -- tails are heavy, which is exactly why a")
    print(f"  naive Pearson-vs-Spearman comparison lies (see note below).\n")
    print(f"  {'feature':10}{'r(f)':>8}{'r(rank f)':>11}{'gain':>8}{'r_w(f)':>9}"
          f"{'r_w(rank f)':>12}{'gain_w':>8}{'kurt f':>9}{'verdict':>14}")
    nl_candidates = []
    for i in order[:14]:
        col = sm[:, i].astype(np.float32)
        cc = np.corrcoef(col, st0)[0, 1]
        # rank ONLY the feature -- the target stays raw, so the delta is attributable
        # to the transform and not to de-fattening t0
        rk = _avg_rank(col)
        rr = np.corrcoef(rk, st0)[0, 1]
        def wcorr(a, b, w):
            w = w / w.mean()
            a = a - (w * a).mean(); b = b - (w * b).mean()
            return float((w * a * b).mean() / np.sqrt((w * a * a).mean() * (w * b * b).mean()))
        cw = wcorr(col, st0, wts)
        rw = wcorr(rk, st0, wts)
        g, gw = rr - cc, rw - cw
        sd = col.std()
        kurt = float(((col - col.mean()) ** 4).mean() / sd ** 4 - 3)
        verdict = "GAIN" if gw > 0.02 else "no gain"
        if gw > 0.02:
            nl_candidates.append((nm[i], cc, rr, cw, rw, gw))
        print(f"  {nm[i]:10}{cc:+8.4f}{rr:+11.4f}{g:+8.4f}{cw:+9.4f}{rw:+12.4f}{gw:+8.4f}"
              f"{kurt:+9.2f}{verdict:>14}")
    print(f"\n  features where RANKING THE FEATURE beats using it raw, weighted: {len(nl_candidates)}")
    if not nl_candidates:
        print(f"  -> A monotone transform of any single feature adds nothing the linear")
        print(f"     first layer cannot already represent. signed_log_volumes and rank")
        print(f"     transforms stay OFF, and we should not build them.")
    else:
        print(f"  -> Worth ONE experiment: append rank(f) for the top offenders as extra")
        print(f"     channels. Cheap (folded into serving), and unlike a linear transform")
        print(f"     it is genuinely outside the model's hypothesis class.")

    # ---- 3. temporal structure ----
    print(f"\n  === 3. TEMPORAL STRUCTURE (target autocorrelation) ===")
    yb = np.asarray(Y[0:2 * SEQ], np.float32)
    print(f"  {'lag':>5}{'autocorr t0':>14}{'autocorr t1':>14}")
    for lag in (1, 2, 3, 5, 10, 20, 50, 100, 500, 2000):
        a0 = np.corrcoef(yb[:-lag, 0], yb[lag:, 0])[0, 1]
        a1 = np.corrcoef(yb[:-lag, 1], yb[lag:, 1])[0, 1]
        print(f"  {lag:5d}{a0:14.4f}{a1:14.4f}")
    print(f"  A 128-wide stateful GRU over raw rows can represent any function of recent")
    print(f"  past, so lag structure above is NOT a reason to add features.")

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))

    a1_ = ax[0, 0]
    pos = np.arange(len(nm))
    for gi, (gname, idx) in enumerate(G.items()):
        idx = [i for i in idx if i < len(nm)]
        a1_.scatter(pos[idx], corr[idx, 0], s=9, label=gname, alpha=.85)
    a1_.axhline(0, color="k", lw=.8)
    a1_.set_xticks(pos); a1_.set_xticklabels(nm, rotation=90, fontsize=5)
    a1_.set_ylabel("Pearson r vs t0"); a1_.set_title("Per-feature correlation vs t0 (train)")
    a1_.legend(fontsize=7, ncol=2)

    a2_ = ax[0, 1]
    for gname, idx in G.items():
        idx = [i for i in idx if i < len(nm)]
        a2_.scatter(pos[idx], corr[idx, 1], s=9, label=gname, alpha=.85)
    a2_.axhline(0, color="k", lw=.8)
    a2_.set_xticks(pos); a2_.set_xticklabels(nm, rotation=90, fontsize=5)
    a2_.set_ylabel("Pearson r vs t1"); a2_.set_title("Per-feature correlation vs t1 (train)")
    a2_.legend(fontsize=7, ncol=2)

    a3_ = ax[1, 0]
    top = order[:20]
    a3_.barh(range(len(top)), corr[top, 0], color="C0")
    a3_.set_yticks(range(len(top))); a3_.set_yticklabels([nm[i] for i in top], fontsize=7)
    a3_.axvline(0, color="k", lw=.8); a3_.set_title("Top 20 features vs t0")
    a3_.set_xlabel("Pearson r")

    a4_ = ax[1, 1]
    sd = sm.astype(np.float32).std(0)
    a4_.scatter(sd, np.abs(corr[:, 0]), s=12, c="C3")
    for i in top[:10]:
        a4_.annotate(nm[i], (sd[i], abs(corr[i, 0])), fontsize=6,
                     xytext=(3, 3), textcoords="offset points")
    a4_.set_xlabel("feature std"); a4_.set_ylabel("|Pearson r| vs t0")
    a4_.set_title("Signal vs scale: is anything under-used?")
    a4_.grid(alpha=.3)

    fig.tight_layout()
    out = FIGDIR / "feature_signal_by_group.png"
    fig.savefig(out, dpi=110)
    print(f"\n  figure -> {out.relative_to(ROOT)}")

    (Path(__file__).resolve().parent / "feature_signal.json").write_text(json.dumps({
        "n_sequences": n_seq, "rows": rows,
        "per_group": {g: {"n": len(v),
                          "max_abs_r_t0": float(np.abs(corr[v, 0]).max()),
                          "mean_abs_r_t0": float(np.abs(corr[v, 0]).mean()),
                          "max_abs_r_t1": float(np.abs(corr[v, 1]).max())}
                      for g, v in G.items() if len(v)},
        "top_features_t0": [{"name": nm[i], "r_t0": float(corr[i, 0]),
                             "r_t1": float(corr[i, 1])} for i in order[:20]],
        # SIX fields are pushed (name, raw r, rank r, weighted raw r, weighted rank r,
        # weighted gain) -- the weighted pair is what the decision uses. Unpacking five
        # crashed the 300-sequence run at the JSON dump, AFTER the analysis completed,
        # which is why the 30-sequence smoke passed and only the long run failed.
        "nonlinear_candidates": [
            {"name": a, "r_raw": b, "r_rank": c,
             "r_weighted_raw": d, "r_weighted_rank": e, "gain_weighted": f}
            for a, b, c, d, e, f in nl_candidates],
    }, indent=2) + "\n")
    print(f"  numbers -> exploration/visualisation/feature_signal.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())