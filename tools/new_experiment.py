#!/usr/bin/env python3
"""Scaffold a new experiment version: training script + version yaml.

WHY THIS TOOL
-------------
AGENTS.md §4 makes `<model>_<version>` an immutable experiment key that appears
verbatim in five different artifact paths.  Hand-creating them invites typos and
silent overwrites, so the naming rules live here as *validators* that raise a
clear error, and the only supported way to start a version is this script.

It creates exactly two files (AGENTS.md §4):
    exploration/models/train_<model>_<version>.py
    exploration/version/version_<model>_<version>.yaml
The log and the checkpoint folder appear when training starts, not now.

Examples
--------
    python tools/new_experiment.py --model gru --version v000 \
        --description "Framework smoke test. Not a real result."
    python tools/new_experiment.py --model gru --version v001 --from gru_v000 \
        --description "wider GRU" --set model.hidden=256 --set training.max_steps=4000
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'
TEMPLATE = TOOLS / 'templates' / 'train_template.py'
MODEL_RE = re.compile(r'^[a-z][a-z0-9]*$')
VERSION_RE = re.compile(r'^v\d{3}$')

DEFAULT_YAML = """schema_version: 1
identity: {model: {model}, version: {version}, parent: {parent}, status: planned}
description: >
  {description}
hypothesis: >
  {hypothesis}
change_vs_parent: >
  {change}
data:
  train_shards: cache/train
  valid_shards: cache/valid
  seq_len: 20000
  warmup_steps: 99
  target_clip: 2.0
  train_sequences: null
features:
  # The organisers already ship standardised features (the baseline ONNX graph
  # applies no normalisation), so `none` is the default to inherit and
  # `robust_zscore` is a hypothesis to ablate (AGENTS.md §15.3).
  normalisation: none
  signed_log_volumes: false
  first_differences: []
  cross_instrument: []
  step_feature: false
  clip_sigma: 0
model:
  arch: gru
  input_proj: 0
  hidden: 128
  layers: 2
  dropout: 0.0
  layernorm: false
  head: linear
  param_count: null
loss:
  type: seq_weighted_pearson
  mask: soft
  aux_mse_weight: 0.0
  target_clip: 2.0
optim:
  name: adamw
  lr: 2.0e-3
  weight_decay: 1.0e-2
  betas: [0.9, 0.99]
  schedule: cosine
  warmup_steps: 100
  grad_clip: 1.0
  ema_decay: 0.0
  precision: fp32
training:
  # chunk_len MUST divide 20000 so every lane ends its chunk on a sequence boundary.
  chunk_len: 2000
  burn_in: 99
  batch_sequences: 16
  max_steps: 2000
  smoke_steps: 60
  epochs: null
  seed: 1337
  eval_every_steps: 250
  eval_subset_sequences: 20
  device: auto
  stateful: true
inference:
  export: onnx
  opset: 17
  ort_threads: 1
  target_us_per_row: 68.5
results:
  best_step: null
  best_val_wp: null
  best_val_t0: null
  best_val_t1: null
  full_valid_wp: null
  latency_us_p50: null
  latency_us_p99: null
  train_hours: null
  finished_utc: null
artifacts:
  log: exploration/logs/log_{model}_{version}.log
  checkpoint_dir: exploration/checkpoints/checkpoint_{model}_{version}
  submission_dir: null
"""

NEXT_COMMANDS = """
Next (AGENTS.md §5):
  1. latency gate:
     python tools/bench_latency.py --script {script} --config {config} --untrained
  2. smoke (tiny end-to-end run; creates the log and the checkpoint folder):
     python {script} --config {config} --smoke
  3. full run (long, detached; the script writes its own log):
     nohup python {script} --config {config} > /dev/null 2>&1 &
  4. review + compare on identical rows:
     python tools/summarize.py
  5. promote (only if it beats the parent by > 2x the seed noise floor and passes §9):
     python tools/package_submission.py --model {model} --version {version}
"""


def validate(model, version):
    if not MODEL_RE.match(model):
        raise SystemExit(f'invalid model name {model!r}: must match ^[a-z][a-z0-9]*$ (AGENTS.md §4)')
    if not VERSION_RE.match(version):
        raise SystemExit(f'invalid version {version!r}: must match ^v\\d{{3}}$ (AGENTS.md §4)')
    if version == 'v000' and model != 'gru':
        raise SystemExit('v000 is reserved for the gru framework smoke test (AGENTS.md §4)')


def set_dotted(cfg, dotted, value):
    node = cfg
    parts = dotted.split('.')
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise SystemExit(f'--set {dotted}: {p} is not a section of the yaml')
        node = node[p]
    if parts[-1] not in node:
        raise SystemExit(f'--set {dotted}: key {parts[-1]!r} does not exist; use --set-new to add it')
    node[parts[-1]] = value


def coerce(v):
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--version', required=True)
    ap.add_argument('--from', dest='parent', default=None,
                    help='<model>_<version> of the parent experiment to copy from')
    ap.add_argument('--description', default=None)
    ap.add_argument('--hypothesis', default=None)
    ap.add_argument('--change', default=None, help='exact delta vs the parent')
    ap.add_argument('--set', action='append', default=[], metavar='dotted.key=value',
                    help='override a yaml key, e.g. --set model.hidden=256')
    args = ap.parse_args()

    validate(args.model, args.version)
    script = ROOT / 'exploration' / 'models' / f'train_{args.model}_{args.version}.py'
    config = ROOT / 'exploration' / 'version' / f'version_{args.model}_{args.version}.yaml'
    for p in (script, config):
        if p.exists():
            raise SystemExit(f'{p} already exists. Versions are immutable (AGENTS.md §2.5); '
                             f'bump --version or create a new model.')

    if args.parent:
        import re as _re
        m = _re.match(r'^([a-z][a-z0-9]*)_(v\d{3})$', args.parent)
        if not m:
            raise SystemExit(f'--from expects <model>_<version>, got {args.parent!r}')
        p_script = ROOT / 'exploration' / 'models' / f'train_{args.parent}.py'
        p_config = ROOT / 'exploration' / 'version' / f'version_{args.parent}.yaml'
        if not p_script.exists() or not p_config.exists():
            raise SystemExit(f'parent {args.parent} not found '
                             f'({p_script.name} / {p_config.name} missing)')
        src = p_script.read_text()
        # rewrite the identity constants so the solution section names the new version
        src, n1 = _re.subn(r"(?m)^MODEL_NAME = '[^']*'$", f"MODEL_NAME = '{args.model}'", src)
        src, n2 = _re.subn(r"(?m)^VERSION = '[^']*'$", f"VERSION = '{args.version}'", src)
        if not n1 or not n2:
            raise SystemExit(f'could not rewrite MODEL_NAME/VERSION in {p_script.name}; '
                             f'repair the parent first rather than shipping a mislabelled version')
        script.write_text(src)
        cfg = yaml.safe_load(p_config.read_text())
        cfg['identity'] = {'model': args.model, 'version': args.version,
                           'parent': args.parent, 'status': 'planned'}
        cfg.setdefault('results', {}).update({k: None for k in
                                              ('best_step', 'best_val_wp', 'best_val_t0', 'best_val_t1',
                                               'full_valid_wp', 'latency_us_p50', 'latency_us_p99',
                                               'train_hours', 'finished_utc')})
        cfg.setdefault('artifacts', {})['submission_dir'] = None
        cfg['artifacts']['log'] = f'exploration/logs/log_{args.model}_{args.version}.log'
        cfg['artifacts']['checkpoint_dir'] = \
            f'exploration/checkpoints/checkpoint_{args.model}_{args.version}'
        if 'param_count' in cfg.get('model', {}):
            cfg['model']['param_count'] = None
    else:
        if not TEMPLATE.exists():
            raise SystemExit(f'missing {TEMPLATE}')
        script.write_text(TEMPLATE.read_text())
        parent = None
        cfg = yaml.safe_load(DEFAULT_YAML.format(
            model=args.model, version=args.version, parent=parent,
            description=args.description or 'TODO: what this model IS, in plain words',
            hypothesis=args.hypothesis or 'TODO: what we expect to learn, and why',
            change=args.change or 'TODO: the exact delta vs the parent'))

    for kv in args.set:
        if '=' not in kv:
            raise SystemExit(f'--set expects dotted.key=value, got {kv!r}')
        k, v = kv.split('=', 1)
        set_dotted(cfg, k.strip(), coerce(v.strip()))

    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f'created {script.relative_to(ROOT)}')
    print(f'created {config.relative_to(ROOT)}  (parent: {args.parent or "none, from template"})')
    print(NEXT_COMMANDS.format(script=script.relative_to(ROOT), config=config.relative_to(ROOT),
                               model=args.model, version=args.version))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
