#!/usr/bin/env python3
"""Check that every markdown table in the repo is structurally intact.

WHY THIS EXISTS
---------------
`tools/update_readme.py` emitted a table for one run with the header row DUPLICATED, the
`|---|` separator row MISSING entirely, and a blank line wedged between the header and the
data. The write reported success, `git status` was clean, and the markdown renderer showed
the block as plain text. Nothing in the pipeline noticed, because nothing was checking the
one property that matters.

That is not a near-miss, it is the recurring failure mode of this project: a generator
rewrites a block, the rewrite is wrong, and the wrongness is only visible to whoever happens
to look at the rendered page. AGENTS.md sec 16 says never claim a file is correct without
verifying it; this verifies the generated tables rather than trusting the generator.

WHAT IT CHECKS, per table
-------------------------
  1. a `|---|` separator exists
  2. the separator's cell count equals the header's
  3. EVERY data row has the header's cell count
  4. no blank line inside the table body
  5. the block is not accidentally duplicated (same header twice)

Run it with no arguments to check every *.md in the repo, or pass paths.
Exit code 0 = clean, 1 = at least one problem found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEP = re.compile(r"^\|[\s:-]*-[\s|:-]*\|$")
ROW = re.compile(r"^\|")


_SPLIT = re.compile(r"(?<!\\)\|")   # a pipe NOT preceded by a backslash


def cells(line: str) -> int:
    """Count table cells, ignoring ESCAPED pipes.

    A backslash-escaped pipe (\|) is a literal pipe inside a cell, not a delimiter, and
    RESUME.md contains at least one legitimately -- e.g. `aux \|t\|-weighted MSE`. A
    naive split("|") counts it as a boundary, reports a phantom cell-count mismatch, and
    trains you to ignore this checker. Guard the negative lookbehind instead."""
    return len(_SPLIT.split(line.strip().strip("|")))


def check(path: Path) -> list[str]:
    problems: list[str] = []
    lines = path.read_text().split("\n")
    i = 0
    seen_headers: dict[str, int] = {}
    while i < len(lines):
        if not (ROW.match(lines[i]) and i + 1 < len(lines) and SEP.match(lines[i + 1])):
            i += 1
            continue
        header = lines[i]
        header_cells = cells(header)
        start = i + 1
        seen_headers[header] = seen_headers.get(header, 0) + 1
        if seen_headers[header] > 1:
            problems.append(
                f"line {start}: header duplicated ({seen_headers[header]}x): {header[:70]}")
        if header_cells != cells(lines[i + 1]):
            problems.append(
                f"line {start}: separator has {cells(lines[i+1])} cells, header has "
                f"{header_cells}")
        j = start + 1
        body = 0
        while j < len(lines) and ROW.match(lines[j]):
            if cells(lines[j]) != header_cells:
                problems.append(
                    f"line {j+1}: row has {cells(lines[j])} cells, header has "
                    f"{header_cells}: {lines[j][:70]}")
            body += 1
            j += 1
        if body == 0:
            problems.append(f"line {start}: table has a header but no data rows")
        # a blank line inside the body terminates the table early; catch rows orphaned
        # under a second header, which is the shape the bug actually produced
        if j < len(lines) and lines[j].strip() == "" and j + 1 < len(lines) \
                and lines[j + 1] == header:
            problems.append(
                f"line {j+1}: blank line then a repeated header -- table split in two")
        i = j
    problems += orphan_runs(lines)
    return problems


def orphan_runs(lines: list[str]) -> list[str]:
    """Find runs of pipe-rows that are NOT a recognised table -- i.e. a table with no
    separator row.

    This is the blind spot that made the checker useless for the exact bug it was written
    for: the loop above identifies a table by its `|---|` separator, so when the generator
    emitted the header WITHOUT a separator, nothing matched, the table was skipped
    entirely, and the checker reported "ok". Verified by injecting that bug on purpose --
    the first version of this script passed it.

    A run of two or more consecutive pipe-lines with a consistent cell count, not already
    consumed by a real table, and with no separator immediately under its first line, is a
    broken table. Requires >= 2 lines so a stray single pipe line cannot trip it.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        if not ROW.match(lines[i]):
            i += 1
            continue
        j = i
        while j < len(lines) and ROW.match(lines[j]):
            j += 1
        run = j - i
        if run >= 2 and not SEP.match(lines[i + 1] if i + 1 < len(lines) else ""):
            counts = {cells(l) for l in lines[i:j]}
            if len(counts) == 1:
                out.append(
                    f"line {i+1}: {run} pipe-rows with no separator row "
                    f"({counts.pop()} cells each) -- this will NOT render as a table")
        i = j
    return out


def main() -> int:
    targets = ([Path(a).resolve() for a in sys.argv[1:]]
               or sorted(ROOT.glob("*.md")) + sorted(ROOT.glob("exploration/**/*.md")))
    bad = 0
    for p in targets:
        if not p.exists():
            print(f"  MISSING {p}")
            bad += 1
            continue
        probs = check(p)
        rel = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        if probs:
            bad += 1
            print(f"  FAIL {rel}")
            for x in probs:
                print(f"       {x}")
        else:
            print(f"  ok   {rel}")
    print(f"\n  {'FAILED' if bad else 'all markdown tables structurally valid'} "
          f"({len(targets)} files checked)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())