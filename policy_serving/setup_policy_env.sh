#!/usr/bin/env bash
#
# Reproducible setup of the POLICY SERVER environment (unrestricted).
#
# Creates a uv venv (default: policy_serving/.venv) and installs yam-policy. This
# env is NOT constrained by ROS or Python 3.10 — add your model's deps here
# (torch / JAX / CUDA, lerobot, or openpi from its repo).
#
#   sh policy_serving/setup_policy_env.sh
#
# Env overrides:  POLICY_PY=3.11
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

PY="${POLICY_PY:-3.11}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] creating venv (python $PY) ..."
uv venv --python "$PY" .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e .

python -c "import yam_policy; print('policy env ready')"

cat <<EOF

[setup] done.  Activate: source $HERE/.venv/bin/activate
  Smoke test (no model):  python -m yam_policy.serve
  Your model:             uv pip install -e /path/to/openpi   # or: uv pip install -e '.[lerobot]'
                          python -m yam_policy.serve --policy <module>:<Class> --config k=v
EOF
