#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace

if [ ! -d /workspace/trustvla-project ]; then
  cp -a /opt/trustvla-project /workspace/trustvla-project
fi

cat >/workspace/activate_trustvla.sh <<'EOF'
#!/usr/bin/env bash
source /opt/trustvla-env/bin/activate
cd /workspace/trustvla-project
export PYTHONPATH=/workspace/trustvla-project/src:/opt/LIBERO
export HF_HOME=/workspace/.cache/huggingface
export MUJOCO_GL=egl
EOF
chmod +x /workspace/activate_trustvla.sh

echo ""
echo "TrustVLA runtime is ready."
echo "To enter the environment:"
echo "  source /workspace/activate_trustvla.sh"
echo ""

if [ -x /start.sh ]; then
  exec /start.sh
fi

sleep infinity
