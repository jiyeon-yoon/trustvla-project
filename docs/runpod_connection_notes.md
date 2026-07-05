# RunPod Connection Notes

Date: 2026-07-05

This file records the current RunPod connection, local project location, remote
workspace location, and commands already used for the TrustVLA project.

## Current RunPod SSH

The direct TCP SSH command that worked:

```bash
ssh root@157.157.221.29 -p 32793 -i ~/.ssh/id_ed25519
```

The RunPod terminal prompt after login looked like:

```text
root@e9a8d43a094b:/workspace/trustvla-project#
```

Important: RunPod IP and port can change when the pod is stopped and started
again. If that happens, update both the IP address and port in the SSH command
and in the VSCode SSH config.

## VSCode Remote SSH Config

Add or update this block in:

```text
/Users/yoon_jiyeon/.ssh/config
```

```sshconfig
Host runpod-trustvla
    HostName 157.157.221.29
    User root
    Port 32793
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 60
```

Then connect in VSCode with:

```text
Remote-SSH: Connect to Host... -> runpod-trustvla
```

Do not select only the raw IP address in VSCode. Select `runpod-trustvla`, so
VSCode also applies the custom port and SSH key.

## Local Project Location

Local Mac project root:

```text
/Users/yoon_jiyeon/Documents/Codex/2026-07-05/trustvla-project/trustvla-guard
```

GitHub repository:

```text
https://github.com/jiyeon-yoon/trustvla-project
```

## RunPod Project Location

Remote RunPod workspace:

```text
/workspace/trustvla-project
```

The repository was expected to be cloned on RunPod with:

```bash
cd /workspace
git clone https://github.com/jiyeon-yoon/trustvla-project.git
cd trustvla-project
```

## Commands Already Run Successfully

Basic package installation:

```bash
pip install -U pip
pip install huggingface_hub notebook ipykernel pillow numpy pytest
```

Tests:

```bash
PYTHONPATH=src python -m pytest -q
```

Observed result:

```text
12 passed in 0.83s
```

Environment check:

```bash
PYTHONPATH=src python -m trustvla.cli doctor
```

Observed result:

```text
libero: missing
torch: ok
transformers: missing
PIL: ok
numpy: ok
```

This means the TrustVLA code itself is working on RunPod. The remaining setup
is to install `transformers` and LIBERO before running real LIBERO/OpenVLA
experiments.

## Next Commands

Install OpenVLA-related Python packages:

```bash
pip install transformers accelerate sentencepiece protobuf
```

Run the doctor command again:

```bash
cd /workspace/trustvla-project
PYTHONPATH=src python -m trustvla.cli doctor
```

Then download a small LIBERO suite from Hugging Face:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets \
  --dry-run
```

If the dry run looks correct, run the real download:

```bash
PYTHONPATH=src python -m trustvla.cli download-libero-hf \
  --suite libero_object \
  --local-dir /workspace/LIBERO-datasets
```

Install LIBERO:

```bash
cd /workspace
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install -e .
```

Return to the project and check again:

```bash
cd /workspace/trustvla-project
PYTHONPATH=src python -m trustvla.cli doctor
```

The target state is:

```text
libero: ok
torch: ok
transformers: ok
PIL: ok
numpy: ok
```

## If SSH Key Changes

If RunPod asks for a password instead of logging in with the SSH key, add the
Mac public key to the pod:

On the Mac:

```bash
cat ~/.ssh/id_ed25519.pub
```

In the RunPod Web Terminal or Jupyter Terminal:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh
echo 'PASTE_PUBLIC_KEY_HERE' >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
```

Then retry:

```bash
ssh root@157.157.221.29 -p 32793 -i ~/.ssh/id_ed25519
```
