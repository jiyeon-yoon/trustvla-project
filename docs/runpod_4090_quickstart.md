# RunPod RTX 4090 Quickstart

## 0. Pod Choice

Use RTX 4090 only for a tiny pilot first.

Recommended settings:

- GPU: RTX 4090 24GB
- Image: PyTorch CUDA image, Ubuntu/Linux, Python 3.10 or 3.11
- Disk/volume: at least 60GB for first pilot, 100GB safer
- Expose: Jupyter or SSH

Do not start with 30 tasks. Start with 1-2 tasks and 1 initial state.

## 1. Open A Terminal In RunPod

Check GPU:

```bash
nvidia-smi
python --version
```

## 2. Get TrustVLA Code

Option A: from GitHub, recommended:

```bash
git clone <YOUR_PRIVATE_GITHUB_REPO>
cd trustvla-guard
```

Option B: upload `trustvla-guard-runpod.tar.gz` from local Mac, then:

```bash
tar -xzf trustvla-guard-runpod.tar.gz
cd trustvla-guard
```

## 3. Install Light Dependencies

```bash
pip install -U pip
pip install huggingface_hub notebook ipykernel pillow numpy pytest
```

Run local project checks:

```bash
PYTHONPATH=src python -m trustvla.cli doctor
PYTHONPATH=src python -m pytest -q
```

At this point `libero: missing` is still expected if LIBERO is not installed yet.

## 4. Install LIBERO

```bash
cd /workspace
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
```

Then return to this project:

```bash
cd /workspace/trustvla-guard
PYTHONPATH=src python -m trustvla.cli doctor
```

You want:

```text
libero: ok
```

## 5. Download Only `libero_object`

Use our selective Hugging Face helper:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets \
  --dry-run
```

If the plan says `allow_patterns: ["libero_object/*"]`, download:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets
```

If LIBERO expects datasets under its own folder, also use the official script:

```bash
cd /workspace/LIBERO
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object \
  --use-huggingface
```

## 6. Export A Tiny Seed Draft

```bash
cd /workspace/trustvla-guard
PYTHONPATH=src python -m trustvla.cli export-libero-seeds \
  --suite libero_object \
  --limit 2 \
  --out data/libero_object_seed_draft.json
```

Open `data/libero_object_seed_draft.json` and fill:

- `target_object`
- `possible_objects`
- `distractor_objects`
- `absent_objects`
- `ambiguous_targets`
- `safety_hazards`

Keep:

```json
"metadata": {"libero_task_id": ...}
```

## 7. Generate TrustVLA-Pairs

```bash
PYTHONPATH=src python -m trustvla.cli generate \
  --seed-tasks data/libero_object_seed_draft.json \
  --out runs/libero_object/trustvla_pairs.jsonl
```

Inspect:

```bash
head -n 3 runs/libero_object/trustvla_pairs.jsonl
```

## 8. First OpenVLA Smoke Rollout

RTX 4090 has 24GB VRAM, so keep this tiny.

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

If this OOMs on RTX 4090, stop and switch to RTX 5090 or A100-class GPU.

## 9. Detect And Compare

```bash
PYTHONPATH=src python -m trustvla.cli detect-rollout-events \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollouts runs/libero_object/openvla_rollouts.jsonl \
  --out runs/libero_object/openvla_rollouts.detected.jsonl

PYTHONPATH=src python -m trustvla.cli compare \
  --benchmark runs/libero_object/trustvla_pairs.jsonl \
  --rollout baseline=runs/libero_object/openvla_rollouts.detected.jsonl \
  --out runs/libero_object/openvla_comparison_report.md
```

Check:

```bash
cat runs/libero_object/openvla_comparison_report.md
```

## 10. Cost Control

After each run:

```bash
nvidia-smi
```

When done:

- stop the pod if using persistent volume
- terminate the pod if no longer needed
- verify RunPod spend rate returns to `$0.00/hr`

