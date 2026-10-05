#!/usr/bin/env python3
"""Does averaging several of our own models beat the best single one?

THE QUESTION THIS ANSWERS
-------------------------
Public seed noise is 0.0004 -- larger than every effect measured in this project. Six runs of
one recipe family landed in 0.6596-0.6600 publicly, and the seed pair ALONE accounts for
100% of that range. If that is the whole story, then every model we train is a noisy draw
from a distribution centred somewhere we have never sampled, and the reason no lever ever
"works" is that we are measuring below the noise floor.

There is a cheap way to test that: average the predictions of several models and ask whether
the MEAN is better than its best MEMBER. If it is, the distribution's centre is above
everything we have submitted, and DISTILLING that average into a single shippable model is
worth 30 GPU-free hours. If the mean is no better than the best member, the members already
sit at the centre and ensembling is pointless.

This is the binary pre-test. It costs no training at all.

METHOD
------
One streaming pass over validation, every model evaluated on EVERY row under the SAME mask,
so no model can be advantaged by row selection. For every subset up to a cap we compute the
reference global WP of the mean prediction, then:

  * best single model
  * mean of all of them
  * GREEDY forward selection -- repeatedly add whichever model most improves the running
    mean, which is how an ensemble would actually be built if it were worth shipping

and we compare the best ensemble against the best single model against the 0.0004 seed
noise. The comparison is only meaningful if the gap exceeds the noise.

REFERENCE METRIC throughout (tools/metric_ref.global_wp), on the official
`need_prediction AND is_scored` mask. Validation only; nothing here feeds a model input.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from metric_ref import global_wp  # noqa: E402

SEQ = 20_000


def load_transform(version: str):
    p = ROOT / "submissions" / f"submission_gru_{version}" / "solution.py"
    spec = importlib.util.spec_from_file_location(f"sol_{version}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stats = np.load(ROOT / "submissions" / f"submission_gru_{version}" / "feature_stats.npz")
    return mod.FeatureTransform(stats)


def session(path: Path):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=o,
                               providers=["CPUExecutionProvider"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", required=True,
                    help="comma-separated versions, e.g. v042,v047,v048,v050")
    ap.add_argument("--n-sequences", type=int, default=400)
    ap.add_argument("--cap", type=int, default=4, help="max subset size to score exhaustively")
    ap.add_argument("--out", default="cache/preds/ensemble_probe.json")
    a = ap.parse_args()

    vers = [v.strip() for v in a.members.split(",") if v.strip()]
    n_all = None
    print(f"  {len(vers)} members: {', '.join(vers)}")

    tr = load_transform(vers[0])
    sess = {v: session(ROOT / "exploration" / "checkpoints" / f"checkpoint_gru_{v}" / "model.onnx")
            for v in vers}

    X = np.load(ROOT / "cache/valid/X.npy", mmap_mode="r")
    Y = np.load(ROOT / "cache/valid/Y.npy", mmap_mode="r")
    N = np.load(ROOT / "cache/valid/need_prediction.npy", mmap_mode="r")
    S = np.load(ROOT / "cache/valid/is_scored.npy", mmap_mode="r")
    n_all = X.shape[0] // SEQ
    n_seq = min(a.n_sequences, n_all)
    rows = n_seq * SEQ

    # state_reset comes from each version's own export, and it DIFFERS across the project
    # (training 2000/4000/10000, serving 4000). Honouring each bundle's own value is the
    # whole point -- a mismatch here would silently compare the wrong models.
    srs = {}
    for v in vers:
        st = np.load(ROOT / "submissions" / f"submission_gru_{v}" / "feature_stats.npz")
        srs[v] = int(st["state_reset"]) if "state_reset" in st.files else 0
    print(f"  serving state_reset per member: {srs}")

    P = {v: np.empty((rows, 2), np.float32) for v in vers}
    t0 = time.time()
    for s in range(n_seq):
        lo, hi = s * SEQ, (s + 1) * SEQ
        raw = np.asarray(X[lo:hi], np.float32)
        xb = np.asarray(tr.batch(raw), np.float32)[:, None, :]
        for v in vers:
            h = [np.zeros((1, 1, 128), np.float32) for _ in range(2)]
            sr = srs[v]
            for i in range(hi - lo):
                if sr and i and i % sr == 0:
                    h = [np.zeros((1, 1, 128), np.float32) for _ in range(2)]
                o = sess[v].run(["prediction", "next_hidden_0", "next_hidden_1"],
                                {"features": xb[i][None], "hidden_0": h[0], "hidden_1": h[1]})
                h = [o[1], o[2]]
                P[v][lo + i] = o[0][0, 0]
        if (s + 1) % 100 == 0:
            el = time.time() - t0
            print(f"    seq {s+1}/{n_seq}  {el:6.1f}s  eta "
                  f"{el/(s+1)*(n_seq-s-1):6.1f}s", flush=True)

    y = np.asarray(Y[:rows], np.float32)
    need = np.asarray(N[:rows], bool)
    sc = np.asarray(S[:rows], bool)
    print(f"\n  {n_seq}/{n_all} sequences, {rows:,} rows, "
          f"{int((need & sc).sum()):,} scored")

    def wp(p):
        return global_wp(y, p, need, sc)

    singles = {v: wp(P[v]) for v in vers}
    print(f"\n  INDIVIDUAL")
    for v, s in sorted(singles.items(), key=lambda kv: -kv[1]):
        print(f"    {v}  {s:.6f}")

    best_single = max(singles.values())
    best_name = max(singles, key=singles.get)
    allmean = np.mean([P[v] for v in vers], axis=0)
    wp_all = wp(allmean)
    print(f"\n  best single     : {best_name}  {best_single:.6f}")
    print(f"  mean of all {len(vers)}   : {wp_all:.6f}   "
          f"({wp_all - best_single:+.6f} vs best single)")

    # exhaustive small subsets
    best_sub, best_sub_wp = None, -1.0
    scored = {}
    for k in range(2, min(a.cap, len(vers)) + 1):
        for combo in itertools.combinations(vers, k):
            w = wp(np.mean([P[v] for v in combo], axis=0))
            scored[",".join(combo)] = w
            if w > best_sub_wp:
                best_sub, best_sub_wp = combo, w
    if best_sub:
        print(f"  best subset<={a.cap}  : {','.join(best_sub)}  {best_sub_wp:.6f}   "
              f"({best_sub_wp - best_single:+.6f} vs best single)")

    # greedy forward selection
    chosen, run = [], np.zeros((rows, 2), np.float32)
    greedy = []
    while len(chosen) < min(a.cap, len(vers)):
        bestv, bestw = None, -1.0
        for v in vers:
            if v in chosen:
                continue
            w = wp((run * len(chosen) + P[v]) / (len(chosen) + 1))
            if w > bestw:
                bestv, bestw = v, w
        if bestv is None:
            break
        chosen.append(bestv)
        run = (run * (len(chosen) - 1) + P[bestv]) / len(chosen)
        greedy.append((",".join(chosen), bestw))
        print(f"  greedy +{bestv:<5} -> {bestw:.6f}   "
              f"({bestw - best_single:+.6f} vs best single)")

    print()
    gain = max(wp_all, best_sub_wp) - best_single
    NOISE = 0.0004     # measured public seed spread, v042 vs v050
    print(f"  best ensemble gain over best single : {gain:+.6f}")
    print(f"  public seed noise                   : {NOISE:+.6f}")
    verdict = ("ENSEMBLING IS REAL -- distil the consensus" if gain > NOISE else
               "NOT WORTH IT -- members already sit at the centre")
    print(f"\n  VERDICT: {verdict}")
    print(f"  Method 2 {'PROCEEDS' if gain > NOISE else 'IS KILLED'}.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "members": vers, "n_sequences": n_seq, "rows": rows,
        "state_reset": srs,
        "singles": singles, "best_single": best_name,
        "best_single_wp": best_single,
        "mean_of_all_wp": wp_all,
        "best_subset": list(best_sub) if best_sub else None,
        "best_subset_wp": best_sub_wp,
        "greedy": greedy,
        "gain": gain, "seed_noise": NOISE, "verdict": verdict,
    }, indent=2) + "\n")
    print(f"  numbers -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())