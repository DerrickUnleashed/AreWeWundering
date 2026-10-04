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
# A SECOND generated block, ranked by the PUBLIC score rather than by full_valid_wp.
# The first table is the selection signal and has to stay ordered by local WP, but the
# two orderings genuinely disagree -- v022 has the 2nd-best local score and the 5th-best
# public one, and v025 is a local regression that is our 2nd-best public result -- so a
# reader who only ever sees the local ranking is being shown a misleading picture of
# where we actually stand on the board that decides the prize.
# These marker strings must NOT be prefixed by BEGIN/END. re.sub(BEGIN .*? END) is
# global, and if BEGIN_PUB merely STARTED with BEGIN then the public block looked like
# a second start marker, got paired with some much later END, and swallowed the region
# between -- re-inserting build_tables() output there and duplicating ~3,500 lines into
# README.md. The write "succeeded" and the file looked plausible. Hence the deliberately
# different wording rather than a suffix of the same phrase.
BEGIN_PUB = '<!-- BEGIN GENERATED: public-score ranking -->'
END_PUB = '<!-- END GENERATED: public-score ranking -->'

# Versions are DISCOVERED, not listed. A hardcoded dict silently omitted every
# version added after it was written -- v012, v013, v014 and v015 all trained,
# scored and packaged while never appearing in the table, because nothing here
# had heard of them. Globs the directory instead, so a new run cannot be
# forgotten. The short label comes from the yaml's own `description`, which is
# the single place the change is already written down.
DESCRIPTIONS: dict[str, str] = {
    'v000': 'framework smoke test (60 steps)',
    'v001': 'wider: 1 layer x 256',
    'v002': 'input: robust z-score, 8-sigma clip',
    'v003': 'aux |t|-weighted MSE + EMA',
    'v004': 'bigger: 2x192 with 128 proj',
    'v005': 'seed replicate of v002 (1338)',
    'v006': 'seed replicate of v002 (1339)',
    'v007': 'input: first differences',
    'v008': 'input: cross-instrument i0/i1',
    'v009': 'input: signed-log tail compression',
    'v010': 'schedule: 6 epochs not 3',
    'v011': 'soft mask + state reset 4k',
    'v012': 'soft mask + state reset 2k',
    'v013': 'reset 4k, no soft mask (control)',
    'v014': 'soft mask + reset 4k, 2x192 proj64',
    'v015': 'v011 re-exported, reset served',
    'v016': 'distilled student, 192k',
    'v017': 'distilled student, 117k (0.61x teacher)',
    'v018': 'soft mask + state reset 10k',
    'v019': 'FIX loss: globally pooled',
    'v020': 'FIX loss + train+valid data',
    'v021': "dozken arch, no teacher",
    'v022': 'dozken arch + his teacher',
    'v023': 'distil, smaller (52% of teacher)',
    'v024': 'distil, batch 64 (his recipe)',
}


# Versions that deliberately reuse another version's WEIGHTS, so they have no
# independent full_valid_wp but do share one. v015 is v011's checkpoint re-exported
# with the state reset actually baked in; the model, and therefore the score, is
# identical. It belongs in the ranking, flagged, rather than parked as unranked --
# it is a real submission at 0.6515, not a placeholder.
INHERITS_FULL_SCORE: dict[str, str] = {
    'v015': 'v011',
}

# Versions whose full_valid_wp is NOT a clean held-out score, and why. Kept here
# rather than only in metadata.json so the leaderboard generator cannot rank a
# contaminated number even if the bundle is regenerated.
SCORE_CONTAMINATED: dict[str, str] = {
    'v020': ('trained on cache/trainval, which appends valid[200:1872] -- 1,673 of '
             'the 1,873 validation sequences -- so this score is measured on 89% '
             'of its own training data and is not comparable with train-only runs'),
}


def discover() -> list[str]:
    return sorted(p.stem.replace('version_gru_', '') for p in VERSIONS.glob('version_gru_v*.yaml'))


def label(v: str) -> str:
    """Short human label for the table.

    The auto-derived fallback used to truncate at a fixed character count and cut
    mid-phrase, producing cells like 'INPUTS: append sign(x)*log1p(|x|) of every'
    and 'SEED REPLICATE of the gru_v002 recipe, seed'. Worse, a label containing a
    pipe -- v003's '0.05 |t|-weighted auxiliary MSE' -- SILENTLY BREAKS the markdown
    table, because an unescaped | is a cell separator. Pipes are escaped here so a
    label can never corrupt the table again.
    """
    if v in DESCRIPTIONS:
        return DESCRIPTIONS[v].replace('|', '\\|')
    y = load_yaml(v)
    if not y:
        return v
    d = ' '.join(str(y.get('description') or '').split())
    d = d.split('. ')[0]                 # first clause only
    if len(d) > 44:
        cut = d[:44].rsplit(' ', 1)[0]    # never mid-word
        d = cut + '...'
    return d.replace('|', '\\|')




def load_yaml(v: str) -> dict | None:
    p = VERSIONS / f'version_gru_{v}.yaml'
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text())


def graph_signature(v):
    """A structural fingerprint of a version's inference graph.

    Two versions with the same signature execute the same arithmetic per row, so they
    take the same time on the grader -- whatever their weights are. That makes a
    MEASURED grader runtime for one of them a valid measurement for the other, which is
    worth far more than any local projection.
    """
    y = VERSIONS / f'version_gru_{v}.yaml'
    sub = None
    if y.exists():
        d = yaml.safe_load(y.read_text()) or {}
        sub = (d.get('artifacts') or {}).get('submission_dir')
    if not sub:
        return None
    onnx_path = ROOT / sub / 'model.onnx'
    meta_path = ROOT / sub / 'metadata.json'
    if not onnx_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    nparams = (meta.get('checkpoint') or {}).get('params')
    if nparams is None:
        ck = (meta.get('checkpoint') or {}).get('path')
        nparams = None
    try:
        import onnx as _onnx
        g = _onnx.load(str(onnx_path)).graph
        nodes, inits = len(g.node), len(g.initializer)
    except Exception:
        return None
    sd = ((meta.get('config') or {}).get('model') or {}).get('param_count')
    return (sd, nodes, inits)


def measured_runtime_anchor(v):
    """If an identical-graph version has a MEASURED grader runtime, return (secs, ver).

    A local p50 taken on a busy box is not a measurement (AGENTS.md sec 2.6), and it has
    now demonstrably produced both false alarms -- v028 projected 120% and ran at 78% --
    and, before that, two models it correctly rejected. When the graph is provably the
    same shape as one we have already timed on the grader, that measurement is the
    better evidence and should win over the projection.
    """
    sig = graph_signature(v)
    if sig is None:
        return None
    lb = load_leaderboard()
    best = None
    for other, e in lb.items():
        if not other.startswith('v') or other == v:
            continue
        secs = e.get('grader_seconds')
        if not secs or e.get('outcome') == 'SCORE:TIMEOUT':
            continue          # a timeout is a kill point, not a natural runtime
        if graph_signature(other) == sig and (best is None or secs > best[0]):
            best = (secs, other)
    # Take the SLOWEST measured runtime among identical graphs, not the first one
    # found. Iterating in dict order happened to pick v023's 45m36s over v028's
    # 46m52s for the same graph; an anchor exists to stop us under-estimating, so it
    # must resolve to the worst case we have actually observed.
    return best


def load_leaderboard() -> dict:
    if not LEADERBOARD.exists():
        return {}
    return json.loads(LEADERBOARD.read_text())


def fmt_mmss(seconds) -> str:
    if seconds is None:
        return '--'
    return f'{int(seconds) // 60}m{int(seconds) % 60:02d}s'


def _proj_dur(mins, pct) -> str:
    """Projected grader runtime, formatted like a measured one (m:ss plus % of limit).

    A projected duration was rendered as '31 min' while every measured one read
    '30m12s (50%)', so two formats sat in one column. Same shape now.
    """
    if mins is None:
        return '--'
    total = int(round(float(mins) * 60))
    s = f'{total // 60}m{total % 60:02d}s'
    return f'{s} ({float(pct):.0f}%)' if pct else s


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
    if entry and entry.get('outcome') == 'SCORE:TIMEOUT':
        # A real grader TIMEOUT outranks every local verdict, including a passing
        # gate and including a margin flag. v026 sat at "Ready to Submit (latency
        # unverified)" in this table right up until it was uploaded and killed at
        # 69m57s. If the grader has spoken, the table must say so.
        secs = entry.get('grader_seconds')
        dur = fmt_mmss(secs)
        if secs:
            dur += f' ({secs / 3600 * 100:.0f}%)'
        mem = f"{entry['memory_mb']} MB" if entry.get('memory_mb') else '--'
        return 'TIMED OUT on grader', 'no score', dur, mem
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
            # A bundle may pass every mechanical gate and still be the wrong thing
            # to ship: v020 passes contract, determinism, parity and latency, and
            # its score is contaminated by training on the validation set. The
            # packager's verdict is a necessary condition, not a sufficient one,
            # so an explicit 'Do not ship' in metadata wins over 'Ready to Submit'.
            # v020 is contaminated AND unshippable; v021 is clean but a regression.
            # The label has to say which, or the table misattributes v021's problem.
            # gate on grader_safe, NOT pass_p50_75pct: those are different margins.
            # v031 clears the 75% target (60.04 < 68.53 us/row) and fails the 95% one
            # (63.9 min > 60), and it is the 95% margin the packager actually decides on.
            anchor = measured_runtime_anchor(v)
            if (not lat.get('grader_safe')) and anchor:
                secs, src = anchor
                pct = secs / 3600 * 100
                dur = fmt_mmss(secs) + f' ({pct:.0f}%)'
                mem = f"{meta.get('memory_mb')} MB" if meta.get('memory_mb') else '--'
                # A measured anchor is a better duration than a local projection, but it
                # is NOT a licence to say "Ready to Submit" at any speed. v045 was being
                # labelled "Ready to Submit" here on the strength of a v021 anchor while
                # its own bundle said NOT-READY at 96.8% of budget -- the anchor branch
                # returned before the failure was ever consulted. A label is a decision,
                # and a decision has to look at the number it is attached to.
                #
                # 85% is the threshold because that is where our own history actually
                # sits: everything we have submitted landed between 56.7% (v001) and
                # 80% (v037, v039), and the one model that was killed, v026, died at 117%.
                # So >85% is not a formatting detail, it is outside everything we have
                # evidence for and inside the range where a timeout is plausible.
                if pct <= 85:
                    return (f'Ready to Submit (latency anchored on {src})',
                            (meta.get('scores') or {}).get('weighted_pearson') and '--' or '--',
                            dur, mem)
                return (f'Marginal ({pct:.0f}% of budget, anchored on {src}) - review before upload',
                        '--', dur, mem)
            if meta.get('status') == 'PASS-BUT-DO-NOT-SUBMIT':
                why = ('score contaminated' if meta.get('full_valid_score_contaminated')
                       else 'clean but below our own base')
                return (f'Do not ship ({why})', '--',
                        _proj_dur(lat.get('grader_min_p50'), lat.get('grader_pct_budget_p50')), '--')
            if meta.get('status') == 'PASS-BUT-GATE-MARGINAL':
                return ('Ready to Submit (latency marginal, 83% of budget)', '--',
                        _proj_dur(lat.get('grader_min_p50'), lat.get('grader_pct_budget_p50')), '--')
            if lat.get('grader_safe'):
                return ('Ready to Submit', '--',
                        _proj_dur(lat.get('grader_min_p50'), lat.get('grader_pct_budget_p50')), '--')
            if meta.get('latency_contaminated'):
                return 'Ready to Submit (latency unverified)', '--', '--', '--'
            return ('Do not ship', '--',
                    _proj_dur(lat.get('grader_min_p50'), lat.get('grader_pct_budget_p50')), '--')

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

        # A score computed on rows the model was TRAINED on must never be ranked
        # beside clean ones. v020 trained on cache/trainval, which appends
        # valid[200:1872] -- 1,673 of the 1,873 validation sequences -- so its
        # full_valid_wp of 0.681773 is measured on 89% of its own training data.
        # Sorted naively it landed at rank 2, above v011 and v018, implying it
        # beat them. It did not; the number is inflated and not comparable.
        # Such versions are pushed below with the reason stated.
        contaminated = bool(SCORE_CONTAMINATED.get(v))
        if contaminated:
            fv = None
        inherits = INHERITS_FULL_SCORE.get(v)
        if fv is None and inherits:
            donor = load_yaml(inherits)
            fv = (donor or {}).get('results', {}).get('full_valid_wp')
            bv = bv if bv is not None else (donor or {}).get('results', {}).get('best_val_wp')
            # mark the inheritance in the label rather than a separate column
            label_txt = f"{label(v)} [shares {inherits}'s weights]"

        # A version with no local full score still belongs in the main table if it
        # was actually submitted -- v000 scored 0.4986 publicly. Sorting it by its
        # subset number would imply a ranking it never had, so such rows go below
        # with the reason stated.
        if fv is not None:
            rows.append((fv, v, label_txt if inherits else label(v),
                         status, pub, dur, mem))
        elif contaminated:
            unranked.append((v, f'{label(v)}  [SCORE CONTAMINATED: {SCORE_CONTAMINATED[v]}]',
                             bv, status, pub, st))
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
        out += ['### Not ranked by `full_valid_wp`', '',
                'Rows that cannot be ranked on `full_valid_wp`, each for a stated '
                'reason: a contaminated score that is not comparable with the rest '
                '(v020), or a framework smoke test that was never a modelling result '
                '(v000). Both have bundles; neither is a submission candidate.', '',
                '| Ver | What it changed | best so far (subset) | Status | Public score |',
                '|---|---|---|---|---|']
        for v, desc, bv, status, pub, st in unranked:
            best = f'{bv:.6f}' if bv is not None else '--'
            out.append(f'| **{v}** | {desc} | {best} | {status} | {pub} |')
        out.append('')

    out.append(END)
    return '\n'.join(out)


def render_public():
    """Rank every submission by its PUBLIC score, with the timeouts shown as failures.

    Kept separate from the main table because the two rankings disagree, and only one
    of them is the board that decides the prize. A timeout appears here rather than
    being dropped: v026 and v027 are the two largest models we ever built, they were
    uploaded, and the grader killed both at 69m57s. Listing them by what they would
    have scored locally would misrepresent them; listing what actually happened does
    not.
    """
    out = [BEGIN_PUB, '',
           '### Ranked by public (leaderboard) score', '',
           'This is the ranking that counts for the prize. It does **not** agree with the',
           'table above, and the disagreements are the interesting part:', '',
           '- **v022** has the 2nd-best `full_valid_wp` (0.685228) and only the 5th-best',
           '  public score. Our local metric overrates it.',
           '- **v025** is a local *regression* against v023 (-1.5 sigma) and our 2nd-best',
           '  public result. The local metric underrates it.',
           '',
           '| Rank | Ver | full_valid_wp | Public score | vs best | Grader | Outcome |',
           '|---:|---|---:|---:|---:|---:|---|']
    scored, failed = [], []
    lb = load_leaderboard()          # LEADERBOARD is a PATH; this is the parsed dict
    for v in sorted(k for k in lb if k.startswith('v')):
        e = lb[v]
        y = ROOT / 'exploration' / 'version' / f'version_gru_{v}.yaml'
        loc = None
        if y.exists():
            try:
                loc = (yaml.safe_load(y.read_text()).get('results') or {}).get('full_valid_wp')
            except Exception:
                loc = None
        secs = e.get('grader_seconds')
        dur = fmt_mmss(secs) + (f' ({secs / 3600 * 100:.0f}%)' if secs else '')
        pub = e.get('public_score')
        if e.get('outcome') == 'SCORE:TIMEOUT' or pub is None:
            failed.append((v, loc, dur, e.get('outcome') or 'not submitted',
                           e.get('grader_seconds')))
        elif v == 'v000':
            continue   # framework smoke test, never a modelling result
        else:
            scored.append((pub, v, loc, dur))
    scored.sort(reverse=True)
    if scored:
        best = scored[0][0]
        for i, (pub, v, loc, dur) in enumerate(scored, 1):
            l = f'{loc:.6f}' if loc else '--'
            out.append(f'| {i} | **{v}** | {l} | **{pub:.4f}** | {pub - best:+.4f} | {dur} | scored |')
        out.append('')
    if failed:
        out += ['| Ver | full_valid_wp | Public score | Grader | Outcome |',
                '|---|---:|---|---:|---|']
        for v, loc, dur, outcome, secs in failed:
            l = f'{loc:.6f}' if loc else '--'
            out.append(f'| **{v}** | {l} | **no score** | {dur} | **{outcome}** |')
        out += ['',
                'v026 (871,977 params) and v027 (1,742,717 params) are the two largest',
                'models this project ever built. Both were uploaded and both were killed by',
                'the grader at 69m57s, so neither has a score at all. They are listed',
                'here rather than omitted because they are the clearest evidence for the',
                'capacity result: locally their scores were 2.2 and 10.2 sigma BELOW v028,',
                'and on the grader neither could finish. Two independent lines agreeing',
                'is why the conclusion is that the student should stay near 225k params.']
    out.append('')
    out.append(END_PUB)
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

    # TWO independent substitutions, one per generated block. Doing this as a single
    # substitution whose replacement contained BOTH blocks meant the pre-existing public
    # block was never consumed, so each run appended another copy -- it grew by one
    # duplicate per invocation while the write still reported success.
    if BEGIN_PUB not in text or END_PUB not in text:
        print(f'{README} has no {BEGIN_PUB} marker; adding it after {END}')
        text = text.replace(END, END + '\n\n' + render_public(), 1)
    new = re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END),
                 lambda _: build_tables(), text, flags=re.S)
    new = re.sub(re.escape(BEGIN_PUB) + r'.*?' + re.escape(END_PUB),
                 lambda _: render_public(), new, flags=re.S)

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
