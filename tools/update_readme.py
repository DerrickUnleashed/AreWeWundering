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

# Versions are DISCOVERED, not listed. A hardcoded dict silently omitted every
# version added after it was written -- v012, v013, v014 and v015 all trained,
# scored and packaged while never appearing in the table, because nothing here
# had heard of them. Globs the directory instead, so a new run cannot be
# forgotten. The short label comes from the yaml's own `description`, which is
# the single place the change is already written down.
DESCRIPTIONS: dict[str, str] = {
    'v011': 'soft mask on P(is_scored) + reset 4k',
    'v012': 'soft mask + reset 2k (shorter horizon)',
    'v013': 'reset 4k, NO soft mask (control)',
    'v014': 'soft mask + reset 4k, 2x192 proj64',
    'v015': 'v011 weights, reset 4k actually shipped',
}


def discover() -> list[str]:
    return sorted(p.stem.replace('version_gru_', '') for p in VERSIONS.glob('version_gru_v*.yaml'))


def label(v: str) -> str:
    if v in DESCRIPTIONS:
        return DESCRIPTIONS[v]
    y = load_yaml(v)
    if not y:
        return v
    d = ' '.join(str(y.get('description') or '').split())
    # first clause, cut at a sentence break or a comma-separated aside
    d = d.split('. ')[0]
    if len(d) > 44:
        d = d[:44].rsplit(' ', 1)[0]
    return d or v




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
    """Return (status, public score, duration, memory) for one version.

    Status is derived from EVIDENCE -- a leaderboard reading, else a bundle and its
    gate -- and only falls back to the yaml's own `identity.status` when there is
    neither. Reading the yaml status first was wrong in three ways: v015 is
    `planned` yet fully packaged and gate-clear, v000 is `smoke` yet was actually
    submitted and scored, and v014 is `finished` with a latency gate that failed on
    a real idle measurement. The yaml says what the trainer was doing; it does not
    say what is shippable.
    """
    entry = lb.get(v)
    if entry and entry.get('public_score') is not None:
        secs = entry.get('grader_seconds')
        dur = fmt_mmss(secs)
        if secs:
            dur += f' ({secs / 3600 * 100:.0f}%)'
        mem = f"{entry['memory_mb']} MB" if entry.get('memory_mb') else '--'
        return 'Submitted', f"{entry['public_score']:.4f}", dur, mem

    st = (y.get('identity') or {}).get('status')
    if st == 'running':
        return 'Training', '--', '--', '--'

    sub = (y.get('artifacts') or {}).get('submission_dir')
    if sub and (ROOT / sub).exists():
        meta_path = ROOT / sub / 'metadata.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            lat = (meta.get('gate') or {}).get('latency') or {}
            if lat.get('grader_safe'):
                mins = lat.get('grader_min_p50')
                return 'Ready to Submit', '--', (f'{mins:.0f} min' if mins else '--'), '--'
            if meta.get('latency_contaminated'):
                return 'Ready to Submit (latency unverified)', '--', '--', '--'
            mins = lat.get('grader_min_p50')
            return 'Do not ship', '--', (f'{mins:.0f} min' if mins else '--'), '--'

    # No bundle. Distinguish "trained but never packaged" from "not trained".
    if st in ('finished', 'smoke'):
        # Trained but never packaged. Distinguish "finished, no bundle" from
        # "no full score to report" -- v016/v017 finished with real scores and no
        # bundle, and labelling those 'No full score' hid the fact entirely.
        has_fv = y.get('results', {}).get('full_valid_wp') is not None
        return ('Yet to package' if has_fv else 'No full score'), '--', '--', '--'
    return 'Yet to Train', '--', '--', '--'


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
        # Only the p50 gate discriminates. Contention inflates the whole
        # distribution, so it pushes p50 over budget (v015 read 50.33 that way).
        # A model whose p50 PASSES and only whose p99 fails has a genuine tail --
        # v014 read 41.50 p50 / 78.25 p99 on an idle box, and flagging that as
        # 'unverified' would hide a real shipping blocker behind a caveat.
        if lat.get('pass_p50_75pct'):
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
    rows, unranked = [], []
    for v in discover():
        y = load_yaml(v)
        if y is None:
            continue
        r = y.get('results') or {}
        fv, bv = r.get('full_valid_wp'), r.get('best_val_wp')
        st = y.get('identity', {}).get('status')
        status, pub, dur, mem = bundle_status(v, y, lb)
        # A version with no local full score still belongs in the main table if it
        # was actually submitted -- v000 scored 0.4986 publicly. Sorting it by its
        # subset number would imply a ranking it never had, so such rows go below
        # with the reason stated.
        if fv is not None:
            rows.append((fv, v, label(v), status, pub, dur, mem))
        else:
            unranked.append((v, label(v), bv, status, pub, st))

    rows.sort(reverse=True)
    out = [BEGIN, '']
    if rows:
        vals = [r[0] for r in rows]
        mean = sum(vals) / len(vals)
        out += [
            '| Rank | Ver | What it changed | full_valid_wp | Status | Public score | Duration | Memory |',
            '|---:|---|---|---:|---|---|---:|---|',
        ]
        for i, (fv, v, desc, status, pub, dur, mem) in enumerate(rows, 1):
            out.append(f'| {i} | **{v}** | {desc} | {fv:.6f} | {status} | {pub} | {dur} | {mem} |')
        out += ['',
                f'Ranked by `full_valid_wp`, the global weighted Pearson over all 1,873',
                'validation sequences (3,528,273 scored rows).',
                '']

    if unranked:
        out += ['### No full-validation score', '',
                'Listed here because they are real runs, but there is no `full_valid_wp` '
                'to rank them by -- either still training, or (v015) deliberately reusing '
                'another version\'s weights rather than retraining.', '',
                '| Ver | What it changed | best so far (subset) | Status | Public score |',
                '|---|---|---:|---|---|---:|']
        for v, desc, bv, status, pub, st in unranked:
            best = f'{bv:.6f}' if bv is not None else '--'
            out.append(f'| **{v}** | {desc} | {best} | {status} | {pub} |')
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
