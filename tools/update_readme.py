"""Regenerate the ranking tables in README.md from the version yamls and bundles.

README.md holds hand-written prose (the noise-floor explanation, the ship-candidate
advice) around two generated tables. Only the region between the GENERATED markers
is rewritten, so the prose survives.

Sources of truth:
  exploration/version/version_gru_<v>.yaml   scores, status, submission_dir
  submissions/submission_gru_<v>/metadata.json   gate results, latency, zip size
  exploration/leaderboard.json                public scores + grader timings

Run after any run finishes, after packaging, and after a new submission scores:
    venv/bin/python tools/update_readme.py            # rewrite the tables
    venv/bin/python tools/update_readme.py --check    # non-zero exit if stale
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / 'README.md'
VERSIONS = ROOT / 'exploration' / 'version'
LEADERBOARD = ROOT / 'exploration' / 'leaderboard.json'
BUNDLES = ROOT / 'submissions'

BEGIN = '<!-- BEGIN GENERATED: leaderboard -->'
END = '<!-- END GENERATED: leaderboard -->'

# Ordered by the local full_valid_wp ranking. Anything not listed is appended
# as still-in-flight, so a new run shows up without editing this file.
DESCRIPTION = {
    'v001': '1x256 wider GRU',
    'v002': 'baseline config, seed 1337',
    'v003': 'aux MSE + EMA',
    'v004': '2x192 + input projection',
    'v005': 'baseline config, seed 1338',
    'v006': 'baseline config, seed 1339',
    'v007': 'first differences',
    'v008': 'cross-instrument (i0 vs i1)',
    'v009': 'signed-log volumes',
    'v010': '6 epochs instead of 3',
    'v011': 'soft mask on P(is_scored) + state reset',
}


def load_yaml(v: str) -> dict | None:
    p = VERSIONS / f'version_gru_{v}.yaml'
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text())


def load_leaderboard() -> dict:
    if not LEADERBOARD.exists():
        return {}
    return json.loads(LEADERBOARD.read_text())


def fmt_mmss(seconds) -> str:
    if seconds is None:
        return '--'
    return f'{int(seconds) // 60}m{int(seconds) % 60:02d}s'


def bundle_status(v: str, y: dict, lb: dict) -> tuple[str, str, str, str]:
    """Return (status, public score, duration, memory) for one version."""
    entry = lb.get(v)
    if entry and entry.get('public_score') is not None:
        dur = fmt_mmss(entry.get('grader_seconds'))
        pct = entry.get('grader_seconds')
        if pct:
            dur += f' ({pct / 3600 * 100:.0f}%)'
        return 'Submitted', f"{entry['public_score']:.4f}", dur, f"{entry.get('memory_mb', '--')} MB"

    sub = (y.get('artifacts') or {}).get('submission_dir')
    if sub and (ROOT / sub).exists():
        meta_path = ROOT / sub / 'metadata.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            lat = (meta.get('gate') or {}).get('latency') or {}
            if lat.get('grader_safe'):
                return 'Ready to Submit', '--', '--', '--'
            # A failed latency gate measured under contention is not evidence of a
            # real failure. flag_contended_latency() marks those; without the flag
            # a FAIL is taken at face value.
            if meta.get('latency_contaminated'):
                return 'Ready to Submit (latency unverified)', '--', '--', '--'
            return 'Do not ship', '--', '--', '--'
    return 'Yet to package', '--', '--', '--'


def flag_contended_latency() -> list[str]:
    """Mark bundles whose latency was measured while other runs held the cores.

    v003 measured 27.71 us/row packaged on an idle box and 48.12 us/row in the
    window where v010 and v011 were training -- 1.7x inflation. Bundles gated in
    that window carry a latency verdict that reflects contention, not the model,
    so 'Do not ship' would be a false reading. This sets the flag so
    bundle_status() reports them as unverified rather than unshippable.
    """
    marked = []
    submitted = set(load_leaderboard())
    for meta_path in sorted(BUNDLES.glob('submission_gru_v*/metadata.json')):
        # A version with a real grader reading has ground truth; never flag it.
        if meta_path.parent.name.replace('submission_gru_', '') in submitted:
            continue
        meta = json.loads(meta_path.read_text())
        if meta.get('latency_contaminated'):
            continue
        lat = (meta.get('gate') or {}).get('latency') or {}
        if lat.get('grader_safe') or not lat:
            continue
        meta['latency_contaminated'] = True
        meta['latency_contamination_note'] = (
            'Latency measured while gru_v010 and gru_v011 were training on the same '
            'cores. Reference: gru_v003 measured 27.71 us/row on an idle box and '
            '48.12 us/row in that window (1.7x). This verdict reflects contention, '
            'not the model. Re-run the gate on an idle box before shipping.'
        )
        meta_path.write_text(json.dumps(meta, indent=2) + '\n')
        marked.append(meta_path.parent.name)
    return marked


def build_tables() -> str:
    lb = load_leaderboard()
    rows, inflight = [], []
    for v, desc in DESCRIPTION.items():
        y = load_yaml(v)
        if y is None:
            continue
        r = y.get('results') or {}
        fv, bv = r.get('full_valid_wp'), r.get('best_val_wp')
        st = y.get('identity', {}).get('status')
        if st == 'finished' and fv is not None:
            rows.append((fv, v, desc))
        else:
            inflight.append((v, desc, bv, st))

    rows.sort(reverse=True)
    out = [BEGIN, '']
    if rows:
        vals = [r[0] for r in rows]
        mean = sum(vals) / len(vals)
        out += [
            '| Rank | Ver | What it changed | full_valid_wp | Status | Public score | Duration | Memory |',
            '|---:|---|---|---:|---|---|---:|---|',
        ]
        for i, (fv, v, desc) in enumerate(rows, 1):
            status, pub, dur, mem = bundle_status(v, load_yaml(v), lb)
            out.append(f'| {i} | **{v}** | {desc} | {fv:.6f} | {status} | {pub} | {dur} | {mem} |')
        out += ['',
                f'Ranked by `full_valid_wp`, the global weighted Pearson over all 1,873',
                'validation sequences (3,528,273 scored rows).',
                '']

    if inflight:
        out += ['### Still in flight', '',
                'No final `full_valid_wp` yet, so not ranked.', '',
                '| Ver | What it changed | best so far (subset) | Status | Public score |',
                '|---|---|---:|---|---|']
        for v, desc, bv, st in inflight:
            status = 'Training' if st == 'running' else 'Yet to Train'
            best = f'{bv:.6f}' if bv is not None else '--'
            out.append(f'| **{v}** | {desc} | {best} | {status} | -- |')
        out.append('')

    out.append(END)
    return '\n'.join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='exit non-zero if README.md is out of date, without writing')
    ap.add_argument('--no-flag-latency', action='store_true',
                    help='do not annotate contended latency verdicts in bundle metadata')
    args = ap.parse_args()

    if not args.no_flag_latency:
        for name in flag_contended_latency():
            print(f'flagged contended latency: {name}')

    if not README.exists():
        sys.exit(f'{README} not found')
    text = README.read_text()
    if BEGIN not in text or END not in text:
        sys.exit(f'{README} is missing the GENERATED markers; add them by hand once')

    new = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END),
                 lambda _: build_tables(), text, flags=re.S)

    if new == text:
        print('README.md is up to date')
        return 0
    if args.check:
        print('README.md is STALE -- run: venv/bin/python tools/update_readme.py')
        return 1
    README.write_text(new)
    print('README.md tables updated')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
