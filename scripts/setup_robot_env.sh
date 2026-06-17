#!/usr/bin/env bash
#
# Reproducible setup of the ROBOT environment (no ROS).
#
# Creates a uv venv (default: <repo>/.venv) and installs i2rt. The robot machine
# runs the portal robot server (i2rt.serving.run_robot_server); it does NOT need
# ROS, rclpy, or Python 3.10 anymore.
#
#   sh scripts/setup_robot_env.sh
#
# Env overrides:  ROBOT_PY=3.11  VENV=/path/to/venv
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PY="${ROBOT_PY:-3.11}"
VENV="${VENV:-$REPO/.venv}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] creating venv at $VENV (python $PY) ..."
uv venv --python "$PY" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[setup] installing i2rt (editable) ..."
uv pip install -e .

python -c "import i2rt, portal; from i2rt.serving import robot_server; print('robot env ready')"

cat <<EOF

[setup] done.
  Activate:    source $VENV/bin/activate
  Bring up CAN: sh scripts/setup_can_ids.sh   (once)  /  sh scripts/reset_all_can.sh (per boot)
  Run server:  scripts/yam teleop   |  scripts/yam dagger   |  scripts/yam wrapper
EOF
