#!/usr/bin/env python3
"""Find yaml keys that appear in more than one section and could silently disagree.

WHY THIS EXISTS
---------------
`state_reset` was in `training:` and `inference:` with DIFFERENT values for 25 consecutive
versions. Training read one, the exporter baked the other into feature_stats.npz, and
solution.py read that back -- so every model this project ever submitted was trained under
one reset schedule and served under another. It was worth +0.001221 (7.7 sigma), in our
favour, by accident.

That was found by accident while checking whether the lever was available. This tool makes
the search repeatable instead of lucky: any key that lives in two sections of the config is
a place where two halves of the system can drift apart without anything noticing.

WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------
Comparison is PER VERSION, not across the whole file population. Comparing pooled sets says
nothing about whether any single run was skewed -- "training used {2000,4000,10000} and
inference used {2000,4000,10000}" looks like agreement while most individual runs disagree.

It also does not just report mismatches, because a mismatch is not automatically a bug.
Two keys share a NAME across sections and mean unrelated things:
`data.warmup_steps` is the scorer's warm-up (99 steps: predictions are required from step
99) and `optim.warmup_steps` is the learning-rate warm-up. Reporting that as a conflict
would train us to ignore the tool. So each key is classified, and the classification is the
output, not a pass/fail.

Exit 0 always: this is an audit, not a gate. Nothing here should fail a build.
"""

from __future__ import annotations

import collections
import glob
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Keys whose two homes are genuinely different quantities that happen to share a name.
# Without this, the tool reports them as conflicts and gets ignored.
NAMING_COLLISIONS = {
    "warmup_steps": "data.warmup_steps is the SCORER's warm-up (predictions required from "
                    "step 99); optim.warmup_steps is the LEARNING-RATE warm-up. Different "
                    "quantities, unrelated values, not a bug.",
}

# Keys where the two sections MUST agree, and where disagreement is a genuine defect.
MUST_AGREE = {
    "state_reset": "the training loop clears recurrent state on this grid; the exporter "
                   "bakes the OTHER section's value into feature_stats.npz and solution.py "
                   "reads that back. Disagreement is a train/serve skew.",
}


def main() -> int:
    loc = collections.defaultdict(lambda: collections.defaultdict(set))
    files = sorted(glob.glob(str(ROOT / "exploration" / "version" / "version_gru_v*.yaml")))
    for p in files:
        d = yaml.safe_load(open(p)) or {}
        for sec, body in d.items():
            if isinstance(body, dict):
                for k, val in body.items():
                    loc[k][sec].add(repr(val))

    shared = {k: sorted(v) for k, v in loc.items() if len(v) > 1}
    print(f"  {len(files)} config files scanned")
    print(f"  keys appearing in more than one section: {len(shared)}")

    defects, collisions, benign = [], [], []
    for k in sorted(shared):
        secs = shared[k]
        # per-version agreement
        disagree = []
        for p in files:
            d = yaml.safe_load(open(p)) or {}
            v = Path(p).stem.replace("version_gru_", "")
            present = {s: d[s].get(k) for s in secs if isinstance(d.get(s), dict) and k in d[s]}
            if len(present) > 1 and len({repr(x) for x in present.values()}) > 1:
                disagree.append((v, present))
        cls = ("defect" if k in MUST_AGREE and disagree else
               "naming collision" if k in NAMING_COLLISIONS else
               "benign")
        (defects if cls == "defect" else
         collisions if cls == "naming collision" else benign).append(k)

        print(f"\n  {k}   [{cls}]")
        print(f"    sections: {', '.join(secs)}")
        for s in secs:
            print(f"      {s:12} {sorted(loc[k][s])}")
        if disagree:
            print(f"    DISAGREE IN {len(disagree)}/{len(files)} VERSIONS:")
            for v, present in disagree[:4]:
                print(f"      v{v}: " + ", ".join(f"{s}={x}" for s, x in present.items()))
            if len(disagree) > 4:
                print(f"      ... and {len(disagree)-4} more")
        if k in NAMING_COLLISIONS:
            print(f"    note: {NAMING_COLLISIONS[k]}")
        if k in MUST_AGREE:
            print(f"    rule : {MUST_AGREE[k]}")

    print(f"\n  defects          : {defects or 'none'}")
    print(f"  naming collisions: {collisions or 'none'}")
    print(f"  benign           : {benign or 'none'}")
    print("\n  No exit code: this is an audit, not a gate. Read the classification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())