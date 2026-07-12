# LIBERO + OpenVLA Rollout Adapter

This document describes how to run real TrustVLA rollouts once a GPU/simulation
environment has LIBERO, MuJoCo/robosuite, PyTorch, Transformers, and an
OpenVLA-compatible checkpoint. These commands are not expected to work on a laptop
environment that does not have LIBERO installed.

Check the environment first:

```bash
source /workspace/activate_trustvla.sh
cd /workspace/trustvla-project
python -m trustvla.cli doctor
```

## What The Adapter Does

The adapter reads a TrustVLA benchmark JSONL, maps each case to a LIBERO task id, sends
the edited instruction to OpenVLA, steps the LIBERO environment, and writes one rollout
record per case.

Current adapter:

- records sparse task success from reward or `info["success"]`-style fields
- records step count and action traces
- records MuJoCo geom contact pairs when the wrapped simulator exposes `sim`
- keeps LIBERO's original `native_success` separate from edited `instruction_success`
- runs a conservative object/contact detector over trace records

## Step 1: Export LIBERO Tasks

On the GPU/simulation machine:

```bash
python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 30 \
  --out data/libero_object_seed_draft.json
```

The exporter uses LIBERO's structured `bddl_utils.robosuite_parse_problem` output to
prefill `possible_objects` and a target when `obj_of_interest` is unique. Review and fill:

- `target_object`
- `possible_objects`
- `distractor_objects`
- `absent_objects`
- `ambiguous_targets`
- `safety_hazards`

Keep `metadata.libero_task_id`; it is needed to map benchmark cases back to LIBERO.

## Step 2: Generate TrustVLA-Pairs

First create an independently reviewed safety-policy draft:

```bash
python -m trustvla.cli export-safety-policies \
  --seed-tasks data/libero_object_seed_draft.json \
  --out data/libero_object_safety_policies_draft.json
```

```bash
python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --init-states 3 \
  --out runs/libero_object/trustvla_pairs.jsonl
```

Validate before spending GPU time:

```bash
python -m trustvla.cli validate-benchmark \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --safety-policies data/libero_object_safety_policies_draft.json
```

## Step 3: Run OpenVLA Baseline

```bash
python -m trustvla.cli run-openvla-libero \
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

## Step 4: Run Language-Emphasis Pilot And Guarded Rollouts

Cheap grounding pilot:

```bash
python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_language_emphasis.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object --device cuda:0 \
  --grounding-mode language_emphasis \
  --trace-dir runs/libero_object/traces/language_emphasis
```

This prompt-only intervention is a pilot baseline, not a substitute for CAG/IGAR or
another published grounding method in the paper-scale evaluation.

```bash
python -m trustvla.cli run-openvla-libero \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --out runs/libero_object/openvla_guarded_rollouts.jsonl \
  --model-path openvla/openvla-7b \
  --suite libero_object \
  --device cuda:0 \
  --image-key agentview_image \
  --max-steps 300 \
  --grounding-mode language_emphasis \
  --trace-dir runs/libero_object/traces/openvla_guarded \
  --guarded \
  --safety-policies data/libero_object_safety_policies_draft.json
```

The gate reads the raw instruction and external safety policy. It does not read
benchmark expected labels. It is currently a pre-execution semantic gate; per-step
low-level motion shielding is not implemented.

## Step 5: Detect Object/Contact Events

The first OpenVLA rollout records traces. Run the detector to infer selected objects and
contact violations from simulator info records:

```bash
python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_rollouts.jsonl \
  --out runs/libero_object/openvla_rollouts.detected.jsonl
```

For guarded rollouts:

```bash
python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_guarded_rollouts.jsonl \
  --out runs/libero_object/openvla_guarded_rollouts.detected.jsonl
```

The detector is conservative. The adapter stores MuJoCo geom pairs under
`trustvla_contacts` where possible. If `selected_target` remains empty, inspect a trace
JSON and map the simulator's geom names before scoring the experiment.

## Step 6: Score And Compare

```bash
python -m trustvla.cli compare \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollout baseline=runs/libero_object/openvla_rollouts.detected.jsonl \
  --rollout guarded=runs/libero_object/openvla_guarded_rollouts.detected.jsonl \
  --out runs/libero_object/openvla_comparison_report.md
```

Also report the new metrics:

```bash
python -m trustvla.cli tradeoff-score \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_rollouts.detected.jsonl

python -m trustvla.cli pair-score \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_rollouts.detected.jsonl
```

## Expected Failure Modes

- `ImportError: LIBERO is not installed`: install LIBERO and its simulator dependencies.
- observation key not found: inspect the keys printed in the error and pass the right
  `--image-key`.
- bad actions or low success: set the correct `--unnorm-key` for the checkpoint.
- target metrics stay empty: inspect trace `infos` and update the detector key rules.
