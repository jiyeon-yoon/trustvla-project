# TrustVLA Starter

TrustVLA is a lightweight starter scaffold for studying whether Vision-Language-Action
models remain stable under instruction edits and safety constraints.

The immediate goal is not to train a new VLA. It is to build a reproducible evaluation
loop:

1. Start from seed manipulation tasks.
2. Generate paired instruction variants.
3. Run a policy adapter in simulation or offline replay.
4. Score task success, unsafe success, constraint violations, and language sensitivity.

## Why This Narrow Angle

Recent 2026 work already covers broad VLA safety and counterfactual language grounding.
This project should therefore avoid a generic "VLA safety benchmark" claim. A stronger
position is:

> Paired, constraint-aware instruction edits reveal whether VLA policies preserve safety
> and language grounding when the desired task stays visually plausible but the action
> constraints change.

## Repository Layout

```text
src/trustvla/
  schema.py          Typed JSON-compatible task, variant, rollout, and metric records.
  perturbations.py   Deterministic instruction edit operators.
  metrics.py         Aggregate metric computation from rollout JSONL.
  detectors.py       Trace-based object/contact event detector.
  verifier.py        Runtime guard that blocks or clarifies unsafe proposals.
  report.py          Markdown report generation for before/after tables.
  adapters.py        Policy adapter protocol and a dummy smoke-test adapter.
  cli.py             Command line entry points.
data/
  seed_tasks.json    Small LIBERO-like seed task set for development.
docs/
  datasets.md
  instruction_variation_rules.md
  libero_openvla_rollout.md
  paper_mvp.md
  runpod_4090_quickstart.md
  runpod_hf_workflow.md
  what_to_run.md
tests/
  test_*.py          Fast unit tests for the benchmark generator and metrics.
notebooks/
  runpod_libero_openvla.ipynb
```

The instruction edits are governed by a written spec:

- `docs/instruction_variation_rules.md`

## Quick Start

Check the environment:

```bash
PYTHONPATH=src python -m trustvla.cli doctor
```

`libero: missing` is fine for local smoke tests. Real LIBERO/OpenVLA rollout commands
must run on a GPU/simulation environment with LIBERO installed.

Generate a benchmark JSONL from the seed tasks:

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/seed_tasks.json \
  --out runs/smoke/generated_benchmark.jsonl
```

Create dummy rollouts for a smoke test:

```bash
PYTHONPATH=src python -m trustvla.cli dummy-rollouts \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --out runs/smoke/dummy_rollouts.jsonl
```

Create guarded dummy rollouts:

```bash
PYTHONPATH=src python -m trustvla.cli guard-dummy-rollouts \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --out runs/smoke/guarded_dummy_rollouts.jsonl
```

Score the dummy rollouts:

```bash
PYTHONPATH=src python -m trustvla.cli score \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --rollouts runs/smoke/dummy_rollouts.jsonl
```

Generate a before/after report:

```bash
PYTHONPATH=src python -m trustvla.cli compare \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --rollout baseline=runs/smoke/dummy_rollouts.jsonl \
  --rollout guarded=runs/smoke/guarded_dummy_rollouts.jsonl \
  --out runs/smoke/comparison_report.md
```

Current smoke-test report:

```text
baseline: wrong-target 0.292, constraint-violation 0.542, unsafe-success 0.125
guarded:  wrong-target 0.000, constraint-violation 0.000, unsafe-success 0.000
```

## Next Implementation Steps

- Run `docs/libero_openvla_rollout.md` on a GPU/simulation machine.
- Annotate exported LIBERO seed tasks with targets, distractors, absent objects, and hazards.
- Save rollout traces with per-step safety events, not only final success.
- Add target/contact detectors so wrong-target and unsafe-success metrics are automatic.
- Add paired action-distance metrics once real actions are available.
- Write the first paper table from this pipeline before adding more models.
