# RunPod Docker Image Workflow

This document describes how to avoid reinstalling TrustVLA, OpenVLA, LIBERO,
robosuite, and MuJoCo dependencies every time a RunPod Pod is created.

## Cost Rule

Stopping a Pod is not fully free.

- GPU compute billing stops.
- RunPod volume disk storage still bills while stopped.
- Terminating the Pod stops the Pod and volume disk billing, but deletes data
  that is not stored in a Network Volume or external backup.

For repeated work, the practical target is:

```text
Docker image for dependencies + GitHub for code + HF/Network Volume for large data
```

Then a Pod can be terminated when not in use.

## What Is Baked Into The Image

The Docker image created by `docker/Dockerfile.runpod` includes:

- Python 3.10 virtual environment at `/opt/trustvla-env`
- PyTorch / torchvision CUDA wheels
- OpenVLA Hugging Face runtime dependencies
- LIBERO cloned into `/opt/LIBERO`
- LIBERO Python package installed in editable mode
- TrustVLA code copied into `/opt/trustvla-project`

The image does not include:

- OpenVLA 7B model weights
- LIBERO datasets
- Experiment outputs
- Checkpoints

Keep those in `/workspace`, a RunPod Network Volume, Hugging Face cache, or
external storage.

## Build With GitHub Actions

After committing and pushing the Docker files:

```bash
cd /Users/yoon_jiyeon/Documents/Codex/2026-07-05/trustvla-project/trustvla-guard
git add .dockerignore docker .github/workflows/build-runpod-image.yml docs/runpod_docker_image.md
git commit -m "Add RunPod Docker image workflow"
git push
```

Then in GitHub:

1. Open `jiyeon-yoon/trustvla-project`.
2. Go to `Actions`.
3. Select `Build RunPod Image`.
4. Click `Run workflow`.
5. Use tag `v0.1`.

When it succeeds, the image should be:

```text
ghcr.io/jiyeon-yoon/trustvla-runpod:v0.1
```

If the package is private, RunPod may need registry credentials. The easiest
path is to make the GHCR package public or push the same image to Docker Hub.

## Build Locally Instead

If Docker Desktop is available:

```bash
cd /Users/yoon_jiyeon/Documents/Codex/2026-07-05/trustvla-project/trustvla-guard
docker build --platform linux/amd64 \
  -f docker/Dockerfile.runpod \
  -t YOUR_DOCKERHUB_USERNAME/trustvla-runpod:v0.1 .

docker login
docker push YOUR_DOCKERHUB_USERNAME/trustvla-runpod:v0.1
```

Use the pushed Docker Hub image name in the RunPod template.

## Create RunPod Custom Template

In the RunPod console:

1. Go to `Templates`.
2. Click `New Template`.
3. Set container image:

```text
ghcr.io/jiyeon-yoon/trustvla-runpod:v0.1
```

4. Set container disk to at least `50 GB`.
5. Add HTTP port `8888` for Jupyter.
6. Add TCP port `22` for SSH.
7. Save the template.

When deploying a new Pod, select this custom template.

## First Commands In A New Pod

After the Pod starts:

```bash
source /workspace/activate_trustvla.sh
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m trustvla.cli doctor
```

Expected target:

```text
libero: ok
torch: ok
transformers: ok
PIL: ok
numpy: ok
```

## Data And Model Cache

Download LIBERO data into `/workspace`:

```bash
source /workspace/activate_trustvla.sh

PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets
```

OpenVLA weights will be cached under:

```text
/workspace/.cache/huggingface
```

If using a Network Volume, both the dataset and HF cache persist independently
from the Pod.

## Terminate Checklist

Before terminating a Pod:

1. Push code changes to GitHub.
2. Copy important result files from `/workspace/trustvla-project/runs`.
3. Confirm whether datasets/model cache are disposable.
4. Terminate only after results are backed up.

If using only Docker image plus GitHub, datasets and model weights will need to
be downloaded again unless they live in a Network Volume or external storage.
