# RunPod + Hugging Face Workflow

## Short Answer

You do not need to download the full LIBERO dataset locally.

You also should not manually copy the whole project folder to RunPod every time.

Use this split:

```text
Mac:
  develop TrustVLA code
  edit benchmark rules
  write paper
  run smoke tests

RunPod/Linux GPU:
  install LIBERO/OpenVLA
  download only needed LIBERO suite
  run actual rollouts
  generate paper tables
```

## What Must Live On RunPod?

RunPod needs:

1. This `trustvla-guard` repo.
2. LIBERO code and simulator dependencies.
3. OpenVLA dependencies/checkpoint.
4. Only the LIBERO suite used for the pilot, usually `libero_object`.

RunPod does not need:

1. The full 100GB LIBERO dataset mirror.
2. Every LIBERO suite.
3. Local smoke-test outputs.

## Hugging Face Dataset Options

LIBERO datasets are mirrored at:

```text
https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets
```

The official LIBERO script supports Hugging Face:

```bash
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object \
  --use-huggingface
```

For a pilot, use:

```text
libero_object
```

Then optionally:

```text
libero_spatial
```

Avoid `all` until the pipeline works.

## Selective Hugging Face Download

If you want direct Hugging Face control, use `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yifengzhu-hf/LIBERO-datasets",
    repo_type="dataset",
    local_dir="/workspace/LIBERO-datasets",
    allow_patterns=["libero_object/*"],
)
```

This avoids downloading `libero_goal`, `libero_spatial`, `libero_90`, and `libero_10`.

Use `allow_patterns=["libero_spatial/*"]` for the spatial suite.

This repo also provides a CLI wrapper:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets
```

Preview without downloading:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets \
  --dry-run
```

Source code:

```text
src/trustvla/hf_datasets.py
```

Notebook:

```text
notebooks/runpod_libero_openvla.ipynb
```

## Do We Need HDF5 Demos For Evaluation?

Not always.

For actual OpenVLA rollout, the critical pieces are:

- LIBERO task definitions
- BDDL files
- fixed initial states
- simulator environment

The HDF5 demos are useful for:

- seeing which objects appear in each task
- filling `target_object`, `possible_objects`, `distractor_objects`, and `safety_hazards`
- qualitative analysis
- detector debugging

For the first pilot, download the smallest suite and annotate only 10 tasks.

## How To Move Our Code To RunPod

Best option:

```bash
git init
git add .
git commit -m "trustvla guard pilot"
git remote add origin <YOUR_PRIVATE_GITHUB_REPO>
git push -u origin main
```

Then on RunPod:

```bash
git clone <YOUR_PRIVATE_GITHUB_REPO>
cd trustvla-guard
```

Simple no-Git option:

```bash
cd /Users/yoon_jiyeon/Documents/Codex/2026-07-05/trustvla-project
tar --exclude='trustvla-guard/runs' -czf trustvla-guard.tar.gz trustvla-guard
```

Upload `trustvla-guard.tar.gz` to RunPod and unpack:

```bash
tar -xzf trustvla-guard.tar.gz
cd trustvla-guard
```

## Minimal RunPod Order

On RunPod:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object \
  --use-huggingface
```

Then get this project:

```bash
git clone <YOUR_PRIVATE_GITHUB_REPO>
cd trustvla-guard
PYTHONPATH=src python -m trustvla.cli doctor
```

Expected:

```text
libero: ok
torch: ok
transformers: ok
PIL: ok
numpy: ok
```

Then:

```bash
PYTHONPATH=src python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 10 \
  --out data/libero_object_seed_draft.json
```

Fill annotations in `data/libero_object_seed_draft.json`, then run:

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --out runs/libero_object/trustvla_pairs.jsonl
```

Then run OpenVLA rollout only after the annotation file looks correct.

## Practical Recommendation

Do not start with 30 tasks.

Start with:

```text
5-10 libero_object tasks
base, target_swap, negation, safety_constraint, impossible_object
1-3 initial states
```

This validates whether:

- LIBERO runs on the machine
- OpenVLA produces actions
- traces are written
- detector can extract useful object/contact events
- report table is generated

Only then scale up.
