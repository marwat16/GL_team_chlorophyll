#!/bin/bash
# One-time setup on the server. Run it from the ZapBench folder inside tmux: bash setup/setup.sh
# It downloads about 10 GB. Steps that are already done are skipped when it runs again.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

echo "== ZAPBench at commit 9a26d3f"
[ -d zapbench-src/.git ] || git clone -q https://github.com/google-research/zapbench.git zapbench-src
git -C zapbench-src checkout -q 9a26d3fa23e1c933f87f15339e0d20f9b3fa6a97

# TensorFlow (ERA) and JAX (TSMixer) need different CUDA libraries, so each gets its own environment
echo "== Python environments"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
for side in era human; do
  [ -x .venv-$side/bin/python ] || uv venv -q -p 3.12 --python-preference only-system .venv-$side
  uv pip install -q -p .venv-$side/bin/python -r setup/requirements-$side.txt
  uv pip install -q -p .venv-$side/bin/python --no-deps -e zapbench-src tf-keras==2.18.0
done
.venv-era/bin/python -c "import tensorflow as tf; assert tf.config.list_physical_devices('GPU'), 'TensorFlow does not see the GPU'"
.venv-human/bin/python -c "import jax; assert jax.devices()[0].platform == 'gpu', 'JAX does not see the GPU'"

echo "== Data: all frames of the first 36,000 neurons"
if [ ! -f data/zap36k/.complete ]; then
  .venv-era/bin/python setup/get_data.py --out data/zap36k --neurons 36000
  touch data/zap36k/.complete
fi

energibridge --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Setup done."
