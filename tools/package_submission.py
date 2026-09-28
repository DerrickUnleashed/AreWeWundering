#!/usr/bin/env python3
"""Package an experiment into `submissions/submission_<model>_<version>/` and gate it.

NEVER UPLOADS.  It builds a local bundle and tells you whether the promotion gate
of AGENTS.md §9 passes:

  1. inputs exist (script, yaml, checkpoint folder, log)
  2. `--export-only` regenerates model.onnx + feature_stats.npz from a checkpoint
  3. solution.py is derived MECHANICALLY from the training script by deleting
     everything outside the BEGIN/END SOLUTION SECTION markers, so the code that
     trained is byte-for-byte the code that will run
  4. the bundle is created (refuses to overwrite an existing one) with
     metadata.json and a ZIP that has solution.py at its root
  5. the gate runs: contract test, determinism, parity vs the offline model,
     score on a documented subset, latency, size and import checks
  6. PASS / NOT-READY is written into metadata.json

The import check enforces AGENTS.md §1: only numpy, onnxruntime, pyarrow and the
standard library may be imported by solution.py, because the scoring container is
built from requirements.txt (numpy, onnxruntime, pyarrow) and nothing else.

Examples
--------
    python tools/package_submission.py --model gru --version v000
    python tools/package_submission.py --model gru --version v001 \
        --checkpoint best_003_step8000_val0.641234.pt --sequences 20
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import yaml  # noqa: E402

import metric_ref  # noqa: E402

BEGIN = '# ===== BEGIN SOLUTION SECTION ====='
END = '# ===== END SOLUTION SECTION ====='
ALLOWED_TOP_IMPORTS = {'numpy', 'onnxruntime', 'pyarrow', 'os', 'sys', 'pathlib', 'math',
                       'json', 'time', 'itertools', 'functools', 'collections', 'typing',
                       '__future__', 'array', 'struct', 'ctypes', 'io', 're'}
MAX_ZIP_BYTES = 20 * 1024 * 1024
STDLIB_HINT = 'stdlib'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or 'n/a'
    except Exception:
        return 'n/a'


def derive_solution(script_path):
    """Delete everything outside the solution markers; add a provenance header."""
    src = Path(script_path).read_text()
    if BEGIN not in src or END not in src:
        raise SystemExit(f'{script_path} has no BEGIN/END SOLUTION SECTION markers')
    body = src.split(BEGIN, 1)[1].split(END, 1)[0]
    header = (f'# Derived from {Path(script_path).name} by tools/package_submission.py.\n'
              f'# Only the code between the BEGIN/END SOLUTION SECTION markers is kept, unchanged,\n'
              f'# so the inference code is byte-identical to the code used in training.\n'
              f'# source sha256: {sha256(script_path)}\n'
              f'# packaged (UTC): {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}\n'
              f'# Inference dependencies: numpy + onnxruntime + stdlib only.\n')
    return header + body.lstrip('\n')


def check_imports(solution_path):
    tree = ast.parse(Path(solution_path).read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split('.')[0]
                if root not in ALLOWED_TOP_IMPORTS:
                    bad.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or '').split('.')[0]
            if node.level == 0 and root and root not in ALLOWED_TOP_IMPORTS:
                bad.append(node.module)
    return sorted(set(bad))


def run(cmd, **kw):
    print(f'$ {" ".join(str(c) for c in cmd)}', flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--checkpoint', default=None, help='default: the best_*.pt in the folder')
    ap.add_argument('--sequences', type=int, default=5,
                    help='validation sequences used by the gate score (documented subset)')
    ap.add_argument('--random-sequences', type=int, default=None,
                    help='use a seeded random subset of this size for the gate instead')
    ap.add_argument('--latency-rows', type=int, default=100_000)
    ap.add_argument('--skip-latency', action='store_true')
    ap.add_argument('--python', default=sys.executable,
                    help='interpreter for the gate subprocesses (use a 3.11 one to be faithful)')
    args = ap.parse_args()

    model, version = args.model, args.version
    key = f'{model}_{version}'
    script = ROOT / 'exploration' / 'models' / f'train_{key}.py'
    config = ROOT / 'exploration' / 'version' / f'version_{key}.yaml'
    bundle = ROOT / 'submissions' / f'submission_{key}'
    failures, warnings, notes = [], [], []

    # ---- 1. inputs
    for p in (script, config):
        if not p.exists():
            raise SystemExit(f'missing {p}; run tools/new_experiment.py first')
    cfg = yaml.safe_load(config.read_text())
    ck_dir = ROOT / cfg['artifacts']['checkpoint_dir']
    log_path = ROOT / cfg['artifacts']['log']
    if not ck_dir.exists():
        raise SystemExit(f'missing {ck_dir}; train the version first')
    cks = sorted(ck_dir.glob('best_*.pt'))
    if not cks:
        raise SystemExit(f'no best_*.pt in {ck_dir}')
    ck = Path(args.checkpoint) if args.checkpoint else cks[-1]
    if not ck.is_absolute():
        ck = ck_dir / ck
    if not ck.exists():
        raise SystemExit(f'checkpoint {ck} not found')
    if not log_path.exists():
        warnings.append(f'no log at {log_path} (AGENTS.md §8 expects one)')
    if bundle.exists():
        raise SystemExit(f'{bundle} already exists. Bundles are immutable (AGENTS.md §2.5); '
                         f'promote a new version instead.')

    # ---- 2. export inference artifacts
    r = run([args.python, script, '--config', config, '--export-only', '--checkpoint', ck])
    if r.returncode:
        failures.append(f'--export-only failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    for f in ('model.onnx', 'feature_stats.npz'):
        if not (ck_dir / f).exists():
            failures.append(f'export did not produce {f}')

    # ---- 3/4. bundle
    bundle.mkdir(parents=True)
    sol_text = derive_solution(script)
    (bundle / 'solution.py').write_text(sol_text)
    shutil.copy2(ck_dir / 'model.onnx', bundle / 'model.onnx')
    shutil.copy2(ck_dir / 'feature_stats.npz', bundle / 'feature_stats.npz')
    pt_size = ck.stat().st_size
    if pt_size <= MAX_ZIP_BYTES // 2:
        shutil.copy2(ck, bundle / ck.name)
        notes.append(f'included the {pt_size/1e6:.1f} MB checkpoint for provenance')
    else:
        notes.append(f'checkpoint is {pt_size/1e6:.1f} MB, left out of the bundle')
    zpath = bundle / f'submission_{key}.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(bundle.iterdir()):
            if f.suffix == '.zip':
                continue
            z.write(f, f.name)          # flat: solution.py must be at the ZIP root
    zip_bytes = zpath.stat().st_size
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    if 'solution.py' not in names:
        failures.append('solution.py is not at the ZIP root')
    if zip_bytes > MAX_ZIP_BYTES:
        failures.append(f'ZIP is {zip_bytes/1e6:.1f} MB, over the 20 MB limit')

    # ---- 5. gate
    bad_imports = check_imports(bundle / 'solution.py')
    if bad_imports:
        failures.append(f'solution.py imports packages that are not guaranteed in the scorer: '
                        f'{bad_imports}')
    gate = {'contract': None, 'determinism': None, 'parity': None, 'score': None,
            'latency': None, 'size': None, 'imports': bad_imports}
    if not bad_imports:
        sc_args = [args.python, ROOT / 'tools' / 'score_solution.py',
                   '--solution', bundle / 'solution.py', '--sequences', args.sequences,
                   '--deterministic-repeat', '--json']
        if args.random_sequences:
            sc_args += ['--random', args.random_sequences]
        r = run(sc_args)
        if r.returncode:
            failures.append('the real callback loop failed (contract violation or crash):\n'
                            f'{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
        else:
            import re
            blob = r.stdout
            m = re.search(r'^\{', blob, re.M)
            data = json.loads(blob[m.start():]) if m else {}
            gate['score'] = {k: data.get(k) for k in
                             ('t0', 't1', 'weighted_pearson', 'selected_rows', 'rows_seen',
                              'us_per_row', 'call_us_p50', 'call_us_p99', 'sequences', 'protocol')}
            gate['contract'] = 'PASS' if data.get('rows_seen') else 'FAIL'
            gate['determinism'] = 'PASS' if data.get('repeat_identical') else 'FAIL'
            if not data.get('repeat_identical'):
                failures.append('predictions are not identical across two runs (docs/faq.md '
                                'requires determinism)')
            notes.append(f'gate score measured on {data.get("protocol")} -- NOT the official '
                         f'full-validation number')
        # parity against the offline model of the same checkpoint
        r = run([args.python, ROOT / 'tools' / 'score_solution.py',
                 '--solution', bundle / 'solution.py', '--sequences', args.sequences,
                 '--parity-against-offline', '--offline-script', script,
                 '--offline-config', config, '--offline-checkpoint', ck, '--json'])
        if r.returncode or 'parity streaming vs offline' not in r.stdout:
            failures.append('parity check against the offline model could not be run')
        else:
            line = [l for l in r.stdout.splitlines() if l.startswith('parity streaming')][-1]
            gate['parity'] = line
            if 'PASS' not in line:
                failures.append(f'parity FAILED: {line}')
        if not args.skip_latency:
            r = run([args.python, ROOT / 'tools' / 'bench_latency.py',
                     '--solution', bundle / 'solution.py', '--rows', args.latency_rows,
                     '--breakdown', '--json'])
            if r.returncode is None:
                m = re.search(r'^\{', r.stdout, re.M)
                if m:
                    gate['latency'] = json.loads(r.stdout[m.start():])
            ok = r.returncode == 0
            gate['latency_pass'] = ok
            if not ok:
                failures.append('latency gate FAILED (> 75 % of the 60 min budget)')
            for l in r.stdout.splitlines():
                if 'projected total' in l or 'µs/row' in l or 'GATE' in l:
                    print('   ' + l)
    gate['size'] = {'zip_bytes': zip_bytes, 'limit_bytes': MAX_ZIP_BYTES,
                    'pass': zip_bytes <= MAX_ZIP_BYTES}

    # ---- 6. metadata
    try:
        import torch
        st = torch.load(ck, map_location='cpu', weights_only=False)
        step, metrics = st.get('step'), st.get('metrics')
    except Exception:
        step, metrics = None, None
    try:
        import platform
        import numpy
        import onnxruntime
        pkgs = {'python': platform.python_version(), 'numpy': numpy.__version__,
                'onnxruntime': onnxruntime.__version__}
        try:
            import torch
            pkgs['torch'] = torch.__version__
        except Exception:
            pass
    except Exception:
        pkgs = {}
    meta = {
        'key': key,
        'packaged_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'git_commit': git_commit(),
        'config': cfg,
        'checkpoint': {'file': ck.name, 'path': str(ck.relative_to(ROOT)), 'step': step,
                       'sha256': sha256(ck), 'train_time_metrics': metrics},
        'scores': gate.get('score'),
        'latency': gate.get('latency'),
        'gate': gate,
        'packages': pkgs,
        'gate_python': args.python,
        'sha256': {'train_script': sha256(script),
                   'solution_py': sha256(bundle / 'solution.py'),
                   'model_onnx': sha256(bundle / 'model.onnx'),
                   'feature_stats_npz': sha256(bundle / 'feature_stats.npz')},
        'zip': {'file': zpath.name, 'bytes': zip_bytes, 'root_entries': names},
        'notes': notes,
        'warnings': warnings,
        'failures': failures,
        'status': 'PASS' if not failures else 'NOT-READY',
    }
    (bundle / 'metadata.json').write_text(json.dumps(meta, indent=1))

    # Write the measured latency and the bundle location back into the yaml
    # (AGENTS.md §6: the packager owns results.latency_* and artifacts.submission_dir).
    if gate.get('latency'):
        cfg_on_disk = yaml.safe_load(config.read_text())
        cfg_on_disk.setdefault('results', {}).update({
            'latency_us_p50': gate['latency'].get('us_p50'),
            'latency_us_p99': gate['latency'].get('us_p99')})
        cfg_on_disk.setdefault('artifacts', {})['submission_dir'] = str(bundle.relative_to(ROOT))
        config.write_text(yaml.safe_dump(cfg_on_disk, sort_keys=False))

    print('\n' + '=' * 70)
    print(f'bundle              : {bundle.relative_to(ROOT)}')
    print(f'zip                 : {zpath.name}  {zip_bytes/1e6:.2f} MB  root={names}')
    print(f'gate contract       : {gate["contract"]}')
    print(f'gate determinism    : {gate["determinism"]}')
    print(f'gate parity         : {gate["parity"]}')
    print(f'gate score          : {gate["score"]}')
    print(f'gate latency        : {gate.get("latency_pass")}')
    for n in notes:
        print(f'note                : {n}')
    for w in warnings:
        print(f'WARNING             : {w}')
    for f in failures:
        print(f'FAILURE             : {f}')
    print(f'STATUS              : {meta["status"]}')
    print('=' * 70)
    print('NOT uploaded. Upload the ZIP yourself from '
          f'{zpath.relative_to(ROOT)} (5 submissions/day, docs/faq.md).')
    return 0 if not failures else 1


if __name__ == '__main__':
    raise SystemExit(main())
