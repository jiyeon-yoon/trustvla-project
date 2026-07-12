# What To Run

## Local Mac: code and metric validation

```bash
cd /Users/yoon_jiyeon/Documents/Codex/2026-07-05/trustvla-project/trustvla-guard

PYTHONPATH=src python -m pytest -q

PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/seed_tasks.json \
  --out runs/smoke/benchmark.jsonl

PYTHONPATH=src python -m trustvla.cli validate-benchmark \
  --benchmark runs/smoke/benchmark.jsonl \
  --safety-policies data/safety_policies.json

PYTHONPATH=src python -m trustvla.cli tradeoff-dummy-rollouts \
  --benchmark runs/smoke/benchmark.jsonl \
  --safety-policies data/safety_policies.json \
  --out-dir runs/smoke

PYTHONPATH=src python -m trustvla.cli compare \
  --benchmark runs/smoke/benchmark.jsonl \
  --rollout visual=runs/smoke/dummy_visual_prior.jsonl \
  --rollout grounded=runs/smoke/dummy_grounded.jsonl \
  --rollout guarded=runs/smoke/dummy_grounded_guarded.jsonl \
  --out runs/smoke/selective_obedience_report.md
```

Open `runs/smoke/selective_obedience_report.md`. Every `dummy_*` value is synthetic
and only verifies that metrics and guard wiring work.

## RunPod: first real pilot

Check the image runtime:

```bash
source /workspace/activate_trustvla.sh
cd /workspace/trustvla-project
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m trustvla.cli doctor
nvidia-smi
```

Export five LIBERO tasks and annotate the generated JSON manually:

```bash
PYTHONPATH=src python -m trustvla.cli export-libero-seeds \
  --suite libero_object --limit 5 \
  --out data/libero_object_seed_draft.json
```

Required manual fields include `target_object`, `possible_objects`, compatible
`distractor_objects`, `absent_objects`, `ambiguous_targets`, and `safety_hazards`.

Create and independently review a trusted policy draft:

```bash
PYTHONPATH=src python -m trustvla.cli export-safety-policies \
  --seed-tasks data/libero_object_seed_draft.json \
  --out data/libero_object_safety_policies_draft.json
```

Generate and validate the benchmark:

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --init-states 3 \
  --out runs/libero_object/benchmark.jsonl

PYTHONPATH=src python -m trustvla.cli validate-benchmark \
  --benchmark runs/libero_object/benchmark.jsonl \
  --safety-policies data/libero_object_safety_policies_draft.json
```

Warnings named `native_success_not_valid` are expected for target-changing edits. They
mean a custom counterfactual success predicate is required before those cases can be
reported as task success.

Run raw OpenVLA on a small budget first:

```bash
PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/benchmark.jsonl \
  --out runs/libero_object/openvla_raw.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object --device cuda:0 --max-steps 50 \
  --trace-dir runs/libero_object/traces/raw
```

Run the cheap prompt-grounding pilot:

```bash
PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/benchmark.jsonl \
  --out runs/libero_object/openvla_language_emphasis.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object --device cuda:0 --max-steps 50 \
  --grounding-mode language_emphasis \
  --trace-dir runs/libero_object/traces/language_emphasis
```

This prompt-only condition is a pilot baseline, not a replacement for a published
grounding method such as CAG or IGAR.

Run language emphasis plus the non-oracle safety-policy gate:

```bash
PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/benchmark.jsonl \
  --out runs/libero_object/openvla_gated.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object --device cuda:0 --max-steps 50 \
  --grounding-mode language_emphasis \
  --guarded \
  --safety-policies data/libero_object_safety_policies_draft.json \
  --trace-dir runs/libero_object/traces/gated
```

Score language-safety trade-offs and pairwise action traces:

```bash
PYTHONPATH=src python -m trustvla.cli tradeoff-score \
  --benchmark runs/libero_object/benchmark.jsonl \
  --rollouts runs/libero_object/openvla_raw.jsonl

PYTHONPATH=src python -m trustvla.cli pair-score \
  --benchmark runs/libero_object/benchmark.jsonl \
  --rollouts runs/libero_object/openvla_raw.jsonl \
  --difference-threshold 0.05 --prefix-steps 10
```

The first real milestone is one completed base rollout plus its trace JSON. Do not launch
the full benchmark until that case has been visually inspected.

Each episode is appended immediately. After interruption, rerun the same rollout command
with `--resume`; existing `case_id` records in the output JSONL are skipped.
