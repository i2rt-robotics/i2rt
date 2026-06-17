#!/usr/bin/env bash
#
# Reproducible setup of the WORKSTATION environment (no ROS).
#
# Creates a uv venv (default: ~/yam_ws) and installs i2rt (portal RobotClient),
# yam-policy (websocket client for the bridge), and the LeRobot recorder deps.
# The robot link is plain TCP (portal), so there is NO ROS / rclpy / system
# site-packages requirement here.
#
#   sh scripts/setup_workstation_env.sh
#
# Env overrides:  WS_PY=3.11  LEROBOT_ENV=~/yam_ws
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

PY="${WS_PY:-3.11}"
VENV="${LEROBOT_ENV:-$HOME/yam_ws}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    . "$HOME/.local/bin/env"
fi
export PATH="$HOME/.local/bin:$PATH"

echo "[setup] system deps (ffmpeg for LeRobot v3.0 video) ..."
sudo apt-get install -y ffmpeg || echo "  (skip apt; install ffmpeg manually if missing)"

echo "[setup] creating venv at $VENV (python $PY) ..."
uv venv --python "$PY" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "[setup] installing i2rt + yam-policy + recorder deps ..."
uv pip install -e .
uv pip install -e policy_serving
uv pip install -r workstation/lerobot_recorder/requirements.txt

echo "[setup] installing RealSense udev rules (USB permissions) ..."
if [ ! -e /etc/udev/rules.d/99-realsense-libusb.rules ]; then
    git clone --depth 1 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense 2>/dev/null || true
    if [ -f /tmp/librealsense/config/99-realsense-libusb.rules ]; then
        sudo cp /tmp/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
        sudo udevadm control --reload-rules && sudo udevadm trigger
    else
        echo "  (could not fetch udev rules; see workstation/lerobot_recorder/README.md)"
    fi
fi

python -c "import i2rt, yam_policy, lerobot, pyrealsense2; print('workstation env ready')" || \
    echo "  (verify deps; pyrealsense2/lerobot may need a moment)"

cat <<EOF

[setup] done.
  Activate:  source $VENV/bin/activate
  Cameras:   workstation/yam-data cams
  Record:    workstation/yam-data record --robot-host <ROBOT_IP> --serials A,B,C
  Bridge:    workstation/yam-data bridge --robot-host <ROBOT_IP> --policy-host <POLICY_IP>
EOF
