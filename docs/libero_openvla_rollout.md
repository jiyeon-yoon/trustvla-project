# LIBERO + OpenVLA Rollout Adapter

This document describes how to run real TrustVLA rollouts once a GPU/simulation
environment has LIBERO, MuJoCo/robosuite, PyTorch, Transformers, and an
OpenVLA-compatible checkpoint. These commands are not expected to work on a laptop
environment that does not have LIBERO installed.

Check the environment first:

```bash
PYTHONPATH=src python -m trustvla.cli doctor
```

## What The Adapter Does

The adapter reads a TrustVLA benchmark JSONL, maps each case to a LIBERO task id, sends
the edited instruction to OpenVLA, steps the LIBERO environment, and writes one rollout
record per case.

Current first version:

- records sparse task success from reward or `info["success"]`-style fields
- records step count and action traces
- records generic safety events if LIBERO info exposes collision/contact/unsafe keys
- does not yet infer `selected_target`; that requires an object/contact detector

## Step 1: Export LIBERO Tasks

On the GPU/simulation machine:

```bash
PYTHONPATH=src python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 30 \
  --out data/libero_object_seed_draft.json
```

This creates an annotation draft. Fill in:

- `target_object`
- `possible_objects`
- `distractor_objects`
- `absent_objects`
- `ambiguous_targets`
- `safety_hazards`

Keep `metadata.libero_task_id`; it is needed to map benchmark cases back to LIBERO.

## Step 2: Generate TrustVLA-Pairs

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --out runs/libero_object/trustvla_pairs.jsonl
```

## Step 3: Run OpenVLA Baseline

```bash
PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_rollouts.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --image-key agentview_image \
  --max-steps 300 \
  --trace-dir runs/libero_object/traces/openvla
```

Set `--unnorm-key` if the checkpoint requires a dataset-specific action normalization
key.

## Step 4: Run Guarded Rollouts

```bash
PYTHONPATH=src python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_guarded_rollouts.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --image-key agentview_image \
  --max-steps 300 \
  --trace-dir runs/libero_object/traces/openvla_guarded \
  --guarded
```

The current coarse guard blocks no-op/clarification-required variants before rollout.
Target/contact-level guarding needs a detector module and is the next development step.

## Step 5: Detect Object/Contact Events

The first OpenVLA rollout records traces. Run the detector to infer selected objects and
contact violations from simulator info records:

```bash
PYTHONPATH=src python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_rollouts.jsonl \
  --out runs/libero_object/openvla_rollouts.detected.jsonl
```

For guarded rollouts:

```bash
PYTHONPATH=src python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_guarded_rollouts.jsonl \
  --out runs/libero_object/openvla_guarded_rollouts.detected.jsonl
```

The detector is conservative. If the trace does not expose contact or grasp object names
in `info`, the output will keep `selected_target` empty. In that case, inspect a trace
JSON and add the relevant keys to `src/trustvla/detectors.py`.

## Step 6: Score And Compare

```bash
PYTHONPATH=src python -m trustvla.cli compare \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollout baseline=runs/libero_object/openvla_rollouts.detected.jsonl \
  --rollout guarded=runs/libero_object/openvla_guarded_rollouts.detected.jsonl \
  --out runs/libero_object/openvla_comparison_report.md
```

## Expected Failure Modes

- `ImportError: LIBERO is not installed`: install LIBERO and its simulator dependencies.
- observation key not found: inspect the keys printed in the error and pass the right
  `--image-key`.
- bad actions or low success: set the correct `--unnorm-key` for the checkpoint.
- target metrics stay empty: inspect trace `infos` and update the detector key rules.
