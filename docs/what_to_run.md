# What To Run

## On This Local Machine

Use this to verify the TrustVLA pipeline without LIBERO/OpenVLA:

```bash
cd /Users/yoon_jiyeon/Documents/Codex/2026-07-05/trustvla-project/trustvla-guard

PYTHONPATH=src python -m pytest -q

PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/seed_tasks.json \
  --out runs/smoke/generated_benchmark.jsonl

PYTHONPATH=src python -m trustvla.cli dummy-rollouts \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --out runs/smoke/dummy_rollouts.jsonl

PYTHONPATH=src python -m trustvla.cli guard-dummy-rollouts \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --out runs/smoke/guarded_dummy_rollouts.jsonl

PYTHONPATH=src python -m trustvla.cli compare \
  --benchmark runs/smoke/generated_benchmark.jsonl \
  --rollout baseline=runs/smoke/dummy_rollouts.jsonl \
  --rollout guarded=runs/smoke/guarded_dummy_rollouts.jsonl \
  --out runs/smoke/comparison_report.md
```

Look at:

- `runs/smoke/generated_benchmark.jsonl`: generated paired instruction cases
- `runs/smoke/comparison_report.md`: before/after metric table

This is a pipeline check, not a paper result.

## On A GPU/SIM Machine Only

Do not run this section on the local Mac unless LIBERO is installed. Use it after
installing LIBERO, MuJoCo/robosuite, PyTorch, Transformers, and OpenVLA dependencies:

First read `docs/datasets.md`. For the pilot, download only `libero_object`, not the
full 100GB dataset mirror.

If using a notebook, open:

```text
notebooks/runpod_libero_openvla.ipynb
```

If using CLI, the selective Hugging Face downloader is:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets
```

```bash
cd /path/to/trustvla-guard

PYTHONPATH=src python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 30 \
  --out data/libero_object_seed_draft.json
```

Then manually fill `target_object`, `possible_objects`, `distractor_objects`,
`absent_objects`, `ambiguous_targets`, and `safety_hazards`.

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --out runs/libero_object/trustvla_pairs.jsonl

PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_rollouts.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --trace-dir runs/libero_object/traces/openvla

PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_guarded_rollouts.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --guarded \
  --trace-dir runs/libero_object/traces/openvla_guarded

PYTHONPATH=src python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_rollouts.jsonl \
  --out runs/libero_object/openvla_rollouts.detected.jsonl

PYTHONPATH=src python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_guarded_rollouts.jsonl \
  --out runs/libero_object/openvla_guarded_rollouts.detected.jsonl

PYTHONPATH=src python -m trustvla.cli compare \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollout baseline=runs/libero_object/openvla_rollouts.detected.jsonl \
  --rollout guarded=runs/libero_object/openvla_guarded_rollouts.detected.jsonl \
  --out runs/libero_object/openvla_comparison_report.md
```

Look at:

- `runs/libero_object/openvla_comparison_report.md`: first paper table candidate
- `runs/libero_object/traces/openvla/*.json`: per-case rollout traces
- cases where `selected_target` is empty: detector needs to be adapted to the actual
  LIBERO `info` keys

## Environment Check

Run this before real rollout commands:

```bash
PYTHONPATH=src python -m trustvla.cli doctor
```

If `libero: missing`, stay with the local smoke-test section or move to a GPU/SIM
environment.
