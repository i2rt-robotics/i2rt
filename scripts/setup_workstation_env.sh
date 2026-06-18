#!/usr/bin/env bash
#
# Reproducible setup of the WORKSTATION environment — conda + uv (no ROS).
#
# conda owns the environment (so you can also `pip install` other policy repos into
# it), and uv does the fast installs for THIS repo. Installs i2rt (portal client),
# yam-policy (websocket client for the bridge), and the LeRobot recorder deps.
#
#   sh scripts/setup_workstation_env.sh
#
# Env overrides:  YAM_WS_ENV=yam_ws  WS_PY=3.11
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

ENV="${YAM_WS_ENV:-yam_ws}"
PY="${WS_PY:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] conda not found — install Miniconda first: https://docs.conda.io/en/latest/miniconda.html" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[setup] conda env '$ENV' (python $PY) ..."
conda activate "$ENV" 2>/dev/null || { conda create -y -n "$ENV" python="$PY"; conda activate "$ENV"; }

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] system deps (ffmpeg for LeRobot v3.0 video) ..."
sudo apt-get install -y ffmpeg || echo "  (skip apt; install ffmpeg manually if missing)"

echo "[setup] uv-installing i2rt + yam-policy + recorder deps into conda env '$ENV' ..."
uv pip install -e .                                   # uv targets the active conda env
uv pip install -e policy_serving
uv pip install -r workstation/lerobot_recorder/requirements.txt

echo "[setup] RealSense udev rules (USB permissions) ..."
if [ ! -e /etc/udev/rules.d/99-realsense-libusb.rules ]; then
    git clone --depth 1 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense 2>/dev/null || true
    if [ -f /tmp/librealsense/config/99-realsense-libusb.rules ]; then
        sudo cp /tmp/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
        sudo udevadm control --reload-rules && sudo udevadm trigger
    else
        echo "  (could not fetch udev rules; see workstation/lerobot_recorder/README.md)"
    fi
fi

python -c "import i2rt, yam_policy, lerobot, pyrealsense2; print('workstation env ready')" ||
    echo "  (verify deps; pyrealsense2/lerobot may need a moment)"

cat <<EOF

[setup] done — conda env: $ENV
  Activate:  conda activate $ENV
  Run:       workstation/yam-data record    (auto-activates '$ENV')
  Another policy repo in the SAME env:
             conda activate $ENV && pip install -e /path/to/policy_repo   # or: uv pip install -e ...
EOF
