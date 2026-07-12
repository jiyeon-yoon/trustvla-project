# Datasets

## What Is Real Data In This Project?

The current `data/seed_tasks.json` is only a smoke-test fixture. It is not paper data.

The real paper dataset should be built from:

1. LIBERO task definitions:
   - language instructions
   - BDDL task files
   - fixed initial states
   - sparse success evaluator

2. LIBERO human teleoperation demonstrations:
   - HDF5 demonstrations per task
   - useful for inspecting objects, trajectories, and possible hazards
   - optional for evaluation if we only run VLA rollouts, but useful for annotation

3. TrustVLA-Pairs generated from annotated LIBERO seed tasks:
   - base instruction
   - paraphrase
   - target swap
   - negation
   - spatial swap
   - safety constraint
   - impossible object
   - ambiguous reference

The final dataset produced by this project is therefore not "raw LIBERO". It is:

```text
LIBERO tasks + our annotation + deterministic instruction variants = TrustVLA-Pairs
```

## Where To Get LIBERO

Official code:

- https://github.com/Lifelong-Robot-Learning/LIBERO

Official project:

- https://libero-project.github.io

Official Hugging Face dataset mirror:

- https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets

The Hugging Face mirror contains:

```text
libero_object/
libero_spatial/
libero_goal/
libero_90/
libero_10/
```

The full mirror is about 100GB. Do not download everything for the pilot.

For RunPod and selective Hugging Face download details, see:

- `docs/runpod_setup_ko.md`

## What To Download First

For the first paper pilot, use:

```text
libero_object
```

Why:

- object-centric tasks are easiest to convert into target swap and negation variants
- fewer conceptual edge cases than long-horizon goal tasks
- easier manual annotation

Second choice:

```text
libero_spatial
```

Why:

- useful for left/right/front/back instruction perturbations

Avoid first:

```text
libero_90
libero_100
```

Why:

- too large for a pilot
- too much annotation burden

## How To Download On GPU/SIM Machine

After installing LIBERO:

```bash
cd LIBERO
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object \
  --use-huggingface
```

For spatial tasks:

```bash
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_spatial \
  --use-huggingface
```

The official script also supports:

```bash
python benchmark_scripts/download_libero_datasets.py --datasets all --use-huggingface
```

Do not use `all` for the pilot unless storage and download time are not a problem.

## What To Create From LIBERO

Once LIBERO is installed, run from this repo:

```bash
PYTHONPATH=src python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 30 \
  --out data/libero_object_seed_draft.json
```

This file is an annotation draft. It must be filled before benchmark generation.

Each task needs:

```json
{
  "target_object": "red mug",
  "possible_objects": ["red mug", "blue mug", "tray", "glass bottle"],
  "distractor_objects": ["blue mug"],
  "absent_objects": ["green mug"],
  "ambiguous_targets": ["red mug", "blue mug"],
  "safety_hazards": ["glass bottle"]
}
```

The important field that must stay unchanged:

```json
"metadata": {"libero_task_id": 0}
```

This maps our generated benchmark cases back to the original LIBERO environment.

## What Counts As The Paper Dataset

After annotation:

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --out runs/libero_object/trustvla_pairs.jsonl
```

This generated file is the benchmark dataset we report in the paper:

```text
runs/libero_object/trustvla_pairs.jsonl
```

Recommended first pilot size:

```text
10 LIBERO object tasks
6-8 variants per task
5-10 initial states per task
approximately 300-800 episodes
```

Recommended submission-scale size:

```text
30 LIBERO object/spatial tasks
6-8 variants per task
10 initial states per task
approximately 1,800-2,400 episodes
```

## Can This Be Done Without Downloading HDF5 Demos?

Partially.

For evaluation, LIBERO task definitions and initial states are the critical pieces.
The HDF5 demos are useful for:

- object/hazard annotation
- checking what objects appear in each scene
- writing qualitative examples
- building detector rules

If download budget is tight, start with only the smallest suite needed for annotation
and rollout.

## Dataset Plan For This Paper

Pilot:

1. `libero_object`
2. 5 tasks and 3 initial states for the first real check
3. manual seed annotation plus independent safety-policy review
4. raw and safety-gated OpenVLA rollout
5. report benign compliance, hazardous compliance, over-refusal, and contact traces

Main:

1. `libero_object` + `libero_spatial`
2. 30 tasks total
3. at least two VLA families and one grounding-enhanced condition
4. raw, grounding-enhanced, and grounding-enhanced + policy-gate comparison
5. custom success predicates for every goal-changing counterfactual case
