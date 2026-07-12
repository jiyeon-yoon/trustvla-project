# TrustVLA Guard

Korean docs:

- [README_kor.md](README_kor.md)
- [docs/runpod_setup_ko.md](docs/runpod_setup_ko.md)

TrustVLA Guard is a research scaffold for evaluating whether
Vision-Language-Action (VLA) policies remain faithful, stable, and safe when
language instructions are changed in controlled ways.

This repository is not a VLA training codebase. It is an evaluation and
diagnostic pipeline. The intended experiment is:

```text
LIBERO manipulation task
-> controlled instruction variants
-> OpenVLA rollout in simulation
-> paired safety/language-following metrics
-> raw policy vs runtime-guarded policy comparison
```

The core research question is:

> When a VLA policy sees the same scene but receives a slightly changed,
> ambiguous, impossible, or safety-constrained instruction, does its action
> change in the correct way?

## Why This Project Exists

Most VLA evaluations report whether a robot eventually succeeds at the nominal
task. That is not enough for trustworthy deployment. A policy can "succeed" at
a task while still touching the wrong object, ignoring a safety constraint, or
overriding the language instruction with a visually familiar behavior.

TrustVLA Guard focuses on paired evaluation:

- Start from a base instruction such as `pick up the red mug`.
- Generate controlled variants such as target swaps, negations, absent-object
  requests, ambiguous references, and explicit safety constraints.
- Run the same policy on each paired variant.
- Measure whether the resulting behavior changes consistently with the
  instruction change.
- Add an inference-time verifier that can block, no-op, or request
  clarification for unsafe or underspecified commands.

The intended paper angle is narrower than a generic "VLA safety benchmark":

> Paired, constraint-aware instruction edits reveal unsafe success and
> language-action inconsistency that ordinary task-success metrics hide.

## What The Code Does

The repository currently implements the benchmark machinery around that idea:

1. **Seed task schema**
   Defines JSON-compatible records for tasks, instruction variants, rollout
   outputs, action proposals, and aggregate metrics.

2. **Instruction variant generation**
   Converts a small set of manipulation tasks into paired instruction stress
   tests. Variants include paraphrase, target swap, attribute swap, spatial
   swap, negation, safety constraint, impossible object, ambiguous reference,
   and distractor instructions.

3. **Runtime verifier / guard**
   A lightweight inference-time verifier decides whether a proposed action
   should execute, no-op, or ask for clarification when an instruction is
   unsafe, impossible, or ambiguous.

4. **Metrics**
   Computes task success, wrong-target behavior, constraint violation,
   unsafe-success, no-op/clarification accuracy, and paired action compliance.

5. **LIBERO/OpenVLA integration**
   Provides a runtime adapter for exporting LIBERO tasks and running OpenVLA
   rollouts in LIBERO simulation on a GPU machine.

6. **RunPod Docker environment**
   Provides a Docker image workflow so that LIBERO/OpenVLA dependencies do not
   need to be installed manually every time a RunPod instance is created.

## Repository Layout

```text
src/trustvla/
  schema.py          Task, variant, rollout, action, and metric data records.
  perturbations.py   Controlled instruction edit operators.
  metrics.py         Aggregate metric computation from rollout JSONL files.
  detectors.py       Trace-based object/contact event detector utilities.
  verifier.py        Runtime guard for unsafe, impossible, or ambiguous actions.
  report.py          Markdown comparison report generation.
  adapters.py        Policy adapter protocol plus dummy smoke-test adapter.
  hf_datasets.py     Hugging Face helper for LIBERO dataset download.
  cli.py             Command line interface.
  integrations/
    libero_openvla.py  LIBERO + OpenVLA rollout adapter.

data/
  seed_tasks.json    Small LIBERO-like seed task set for local development.

docker/
  Dockerfile.runpod
  requirements-openvla-runtime.txt
  requirements-libero-no-vla-conflict.txt
  verify_runtime.py
  start-trustvla.sh

docs/
  datasets.md
  instruction_variation_rules.md
  libero_openvla_rollout.md
  paper_mvp.md
  runpod_4090_quickstart.md
  runpod_connection_notes.md
  runpod_docker_image.md
  runpod_hf_workflow.md
  runpod_setup_ko.md
  what_to_run.md

notebooks/
  runpod_libero_openvla.ipynb

tests/
  test_*.py
```

## Current Status

Working locally:

- Benchmark JSONL generation from seed tasks.
- Dummy policy rollout generation.
- Guarded dummy rollout generation.
- Metric computation and comparison report generation.
- Hugging Face LIBERO dataset download helper.
- LIBERO/OpenVLA adapter code path.
- Unit tests for generation, metrics, verifier, detectors, HF download helper,
  and integration argument flow.

Still required for paper results:

- Run OpenVLA in LIBERO on RunPod.
- Download real LIBERO data.
- Export and annotate real LIBERO seed tasks.
- Collect real rollout traces.
- Compare `OpenVLA raw` vs `OpenVLA + TrustVLA Guard`.
- Produce paper tables, plots, and qualitative failure cases.

## Quick Start: Local Smoke Test

Local smoke tests do not require LIBERO, OpenVLA, or a GPU.

```bash
PYTHONPATH=src python -m trustvla.cli doctor
PYTHONPATH=src python -m pytest -q
```

Generate instruction variants:

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/seed_tasks.json \
  --out runs/smoke/generated_benchmark.jsonl
```

Create dummy rollouts:

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

Generate a comparison report:

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

These numbers are not paper results. They only verify that the evaluation
pipeline, metrics, and guard/report code are wired correctly.

## RunPod / Real Rollout Plan

The real experiment must run on a GPU/simulation machine because OpenVLA is a
7B-parameter VLA and LIBERO requires simulation dependencies.

Use the RunPod image built by this repository:

```text
ghcr.io/jiyeon-yoon/trustvla-runpod:v0.4
```

Use the newest successful tag from the GitHub Actions `Build RunPod Image`
workflow. Keep the previous working tag until the new tag succeeds.

After starting a RunPod instance with that image:

```bash
source /workspace/activate_trustvla.sh
PYTHONPATH=src python -m trustvla.cli doctor
PYTHONPATH=src python -m pytest -q
```

Download a small LIBERO suite:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets
```

Export seed tasks:

```bash
PYTHONPATH=src python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 5 \
  --out data/libero_object_seed_draft.json
```

Generate paired benchmark variants:

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --out runs/libero_object/trustvla_pairs.jsonl
```

Run a tiny OpenVLA rollout:

```bash
PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_rollouts.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --max-steps 50 \
  --trace-dir runs/libero_object/traces/openvla
```

Start with `--limit 5` and `--max-steps 50`. Once this succeeds, scale to
approximately 30 LIBERO_OBJECT tasks and longer rollouts.

## Research Positioning

This project is informed by several lines of recent VLA work.

### OpenVLA

[OpenVLA](https://arxiv.org/abs/2406.09246) is the main target model for the
first experiments. It is an open-source 7B VLA trained on robot demonstrations
and designed for generalist manipulation. TrustVLA Guard does not train a new
VLA; it evaluates and guards an existing one.

Official code:

- [openvla/openvla](https://github.com/openvla/openvla)

### LIBERO

[LIBERO](https://arxiv.org/abs/2306.03310) is the manipulation benchmark and
simulation environment used for the first rollout experiments. TrustVLA Guard
uses LIBERO tasks as base scenarios and creates paired instruction variants on
top of them.

Official code and project:

- [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [LIBERO project page](https://libero-project.github.io)

### Counterfactual Language Following

[When Vision Overrides Language / LIBERO-CF](https://arxiv.org/abs/2602.17659)
studies how VLAs can ignore language and follow visual shortcuts. TrustVLA
Guard is closely related, but focuses on paired safety and instruction-stress
evaluation with explicit no-op/clarification expectations and a runtime guard.

### VLA Safety Benchmarks

[ForesightSafety-VLA](https://arxiv.org/abs/2606.27079) motivates safety as a
first-class VLA evaluation target and emphasizes process-level risk rather than
only final task success. TrustVLA Guard is narrower and cheaper: it focuses on
instruction-paired safety checks in LIBERO/OpenVLA rather than a broad safety
taxonomy across many embodiments.

[LIBERO-Safety](https://arxiv.org/abs/2606.23686) is another close benchmark
that builds safety-critical LIBERO-style scenarios. TrustVLA Guard should not
claim to be a comprehensive safety dataset; its differentiator is paired
instruction consistency plus inference-time verification.

### Semantic Grounding

[RoboSemanticBench](https://arxiv.org/abs/2606.02277) evaluates whether VLA
policies use semantic understanding when choosing physical targets. TrustVLA
Guard addresses a related failure mode: a policy may take a plausible physical
action while failing to respect the specific language constraint.

## Intended Paper MVP

A realistic minimum paper experiment is:

```text
Dataset: LIBERO_OBJECT
Model: OpenVLA
Conditions:
  1. OpenVLA raw
  2. OpenVLA + TrustVLA Guard
Tasks: 30 seed tasks
Variants: 5-8 paired instruction variants per seed task
Metrics:
  - task success
  - wrong-target rate
  - unsafe-success rate
  - constraint-violation rate
  - no-op / clarification correctness
  - paired action consistency
```

The expected table should answer:

```text
Does the guard reduce unsafe and wrong-target behavior?
Does it preserve ordinary task success?
Which instruction transformations break OpenVLA most often?
```

## What This Project Should Not Claim Yet

Until real rollout results are collected, this repository should not claim:

- A new VLA model.
- A complete safety benchmark.
- State-of-the-art safety performance.
- Real-world robot validation.

The current claim is more modest:

> This repository implements a reproducible paired-evaluation scaffold for
> measuring instruction sensitivity and safety failures in VLA policies, with a
> first runtime guard and a planned OpenVLA/LIBERO evaluation.
