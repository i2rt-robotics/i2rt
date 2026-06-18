# I2RT Python API

A Python client library for interacting with [I2RT](https://i2rt.com/) products — designed for learning-based robotics, teleoperation, and real-world deployment.

[![I2RT](https://github.com/user-attachments/assets/025ac3f0-7af1-4e6f-ab9f-7658c5978f92)](https://i2rt.com/)

## Features

- Plug-and-play Python interface for YAM arms and Flow Base
- Real-time robot control via CAN bus (DM series motors)
- MuJoCo gravity compensation, simulation, and URDF/MJCF models
- Gripper force control and auto-calibration
- Bimanual teleoperation and trajectory record & replay
- Policy-deployment ready — works with standard robot learning pipelines

## Installation

```bash
git clone https://github.com/i2rt-robotics/i2rt.git && cd i2rt
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.11
source .venv/bin/activate
```

```bash
sudo apt update
sudo apt install build-essential python3-dev linux-headers-$(uname -r)
uv pip install -e .
```

## CAN Bus Setup

```bash
# Check detected CAN devices
ls -l /sys/class/net/can*

# Bring up interface at 1 Mbit/s
sudo ip link set can0 up type can bitrate 1000000

# Auto-enable on boot
sudo sh devices/install_devices.sh

# Reset unresponsive adapter
sh scripts/reset_all_can.sh
```

## YAM Arm

### Zero-gravity mode

```bash
python i2rt/robots/motor_chain_robot.py --channel can0 --gripper linear_4310
```

### YAM LeWM Inference Runbook

This is the quick operator path used for running goal-conditioned LeWM + CEM/MPC
on the gem9 bimanual YAM rig. The GPU server runs on RunPod; the robot client runs
on gem9 because it owns the CAN buses, RealSense cameras, and arm drivers.

#### Known gem9 mapping

| Device | Value |
|--------|-------|
| GPU server Tailscale IP | `100.123.144.75` |
| gem9 Tailscale IP | `100.77.141.106` |
| CEM server port | `8017` |
| Left arm CAN | `can1` |
| Right arm CAN | `can0` |
| Exo camera | `353322271538` |
| Left wrist camera | `353322271686` |
| Right wrist camera | `353322270649` |

#### 1. Prepare the checkpoint on RunPod

The deploy directory expects a config JSON and checkpoint weights. On the current
RunPod image these were staged as symlinks:

```bash
cd /workspace/src/research/deploy/lewm-yam
ln -sf /workspace/.stable_worldmodel/checkpoints/yam_swm_lewm_mosaic224_vitb_proprio_decoder/config.json config.json
ln -sf /workspace/.stable_worldmodel/checkpoints/yam_swm_lewm_mosaic224_vitb_proprio_decoder/weights_step_440000.pt weights_step_440000.pt
```

Do not commit these symlinks or the checkpoint. They are machine-local artifacts.

The matching model config is:

- ViT-B/14, image size 224
- 14D proprio
- action encoder input dim 70 (`frameskip=5 * 14D`)
- predictor history size 3

#### 2. Start the GPU server on RunPod

In a RunPod terminal:

```bash
cd /workspace/src/research/deploy/lewm-yam
python serve_lewm.py \
  --config config.json \
  --weights weights_step_440000.pt \
  --port 8017 \
  --device cuda \
  --bf16
```

Wait for:

```text
Warmup done.
Serving on 0.0.0.0:8017
```

In a second terminal on the same RunPod, verify:

```bash
curl http://127.0.0.1:8017/health
```

Expected output:

```text
ok
```

From gem9, use the RunPod Tailscale IP:

```bash
curl http://100.123.144.75:8017/health
```

#### 3. SSH to gem9

Tailnet DNS may not resolve inside RunPod. If this fails:

```bash
ssh pantheon@pantheon-gem9.tailc7b2a4.ts.net
```

use the Tailscale IP and proxy command:

```bash
ssh -o StrictHostKeyChecking=accept-new \
  -o ProxyCommand='tailscale --socket=/var/run/tailscale/tailscaled.sock nc %h %p' \
  pantheon@100.77.141.106
```

#### 4. Check gem9 hardware

On gem9:

```bash
source /home/pantheon/miniconda3/etc/profile.d/conda.sh
conda activate atlas

curl http://100.123.144.75:8017/health
ip -details -statistics link show can0
ip -details -statistics link show can1
python -c "import pyrealsense2 as rs; print([d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()])"
```

Expected:

- server health prints `ok`
- both CAN buses are `UP`, `LOWER_UP`, `ERROR-ACTIVE`, 1 Mbit/s, and zero bus errors
- RealSense lists `353322271538`, `353322271686`, `353322270649`

If a CAN bus was power-cycled or a power cable was replugged:

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

Ping motors:

```bash
for id in 1 2 3 4 5 6 7; do
  echo "PING can0 motor $id"
  timeout 8 python -m i2rt.motor_config_tool.ping_motors --channel can0 --motor_id "$id"
done
```

Repeat with `can1` if needed.

#### 5. Start the robot client

On gem9:

```bash
cd /home/pantheon/lewm-yam
./run_gem9_client.sh 100.123.144.75:8017
```

The script initializes both YAM arms with `zero_gravity_mode=True`. The pose at
startup becomes the stored `home` pose.

Expected prompt:

```text
Ready. Commands:
  g  = capture goal image + goal-pose proprio from current cameras
  c  = capture goal, then smooth return to home
  s  = start inference loop from current pose
  r  = return home, then start CEM/MPC from home
  h  = smooth return to home
  e  = E-stop
  q  = quit
```

#### 6. Normal operation

Use this flow for home-to-goal CEM/MPC:

1. Start the client with the arms physically at the desired home pose.
2. Manually move the arms and scene to the goal pose.
3. Type `c` and press Enter.
   - Captures goal camera frames.
   - Captures current 14D arm state as goal proprio.
   - Sends the goal to the GPU server.
   - Smoothly returns to home.
4. Reset the object/scene to the start state.
5. Type `r` and press Enter.
   - Returns home if needed.
   - Resets planner history while keeping the goal.
   - Runs CEM/MPC from home to goal.

Emergency stop:

```text
e
```

or Ctrl-C while the client is running.

Once the client exits and the terminal prompt is back to
`pantheon@pantheon-gem9:~/lewm-yam$`, the one-letter commands no longer apply.
Typing `e` at the shell prompt is just a shell command, not an E-stop.

#### 7. Common failure recovery

Port already in use on RunPod:

```bash
pkill -f 'serve_lewm.py.*--port 8017'
ss -ltnp | grep 8017 || true
```

Robot client stuck in `Shutting down...`:

```bash
pgrep -af 'inference_lewm.py|run_gem9_client.sh'
pkill -f inference_lewm.py
pkill -f run_gem9_client.sh
```

Camera `VIDIOC_S_FMT` or input/output error:

```bash
pgrep -af 'inference_lewm.py|run_gem9_client.sh'
pkill -f inference_lewm.py
pkill -f run_gem9_client.sh
./run_gem9_client.sh 100.123.144.75:8017 --reset-cameras
```

Wrong home pose:

```text
q
```

Move the arms manually to the desired home pose, then restart the client.

### Python API

```python
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType
import numpy as np

robot = get_yam_robot(channel="can0", gripper_type=GripperType.LINEAR_4310)

# Read joint positions (radians)
q = robot.get_joint_pos()   # shape: (6,)

# Command a target configuration
robot.command_joint_pos(np.zeros(6))
```

### Leader-follower teleoperation

```bash
# Follower arm
python examples/minimum_gello/minimum_gello.py --gripper linear_4310 --mode follower --can-channel can0 --bilateral-kp 0.2

# Leader arm (teaching handle)
python examples/minimum_gello/minimum_gello.py --gripper yam_teaching_handle --mode leader --can-channel can1 --bilateral-kp 0.2
```

- **Top button (press once):** enable synchronisation — follower tracks leader
- **Top button (press again):** disable synchronisation
- `--bilateral-kp` controls resistance felt on the leader (0.1–0.2 recommended)

To inspect leader arm output:

```bash
python scripts/run_yam_leader.py --channel $CAN_CHANNEL
```

### MuJoCo visualiser

```bash
python examples/minimum_gello/minimum_gello.py --mode visualizer_local
```

## Gripper Types

| Gripper | Motor | Notes |
|---------|-------|-------|
| `crank_4310` | DM4310 | Zero-linkage crank — minimises gripper width |
| `linear_3507` | DM3507 | Lightweight linear; start closed or run calibration |
| `linear_4310` | DM4310 | Standard linear; slightly more force than 3507 |
| `yam_teaching_handle` | — | Leader arm handle with trigger + 2 buttons. |

The linear grippers require calibration because their motor travels more than 2π radians over the full stroke — either start with the gripper fully closed, or run the calibration routine.

## Flow Base

```bash
# Joystick demo
python i2rt/flow_base/flow_base_controller.py
```

```python
from i2rt.flow_base.flow_base_client import FlowBaseClient

client = FlowBaseClient(host="172.6.2.20")
client.set_target_velocity([0.1, 0.0, 0.0], frame="local")
```

## Examples

| Example | Location |
|---------|----------|
| Bimanual lead-follower | `examples/bimanual_lead_follower/` |
| Record & replay trajectory | `examples/record_replay_trajectory/` |
| Single motor PD control | `examples/single_motor_position_pd_control/` |
| MuJoCo control interface | `examples/control_with_mujoco/` |

## Advanced: Motor Configuration

### Safety timeout

The factory default is a **400 ms timeout** — motors enter damping mode if no command is received within 400 ms.

```bash
# Disable timeout (advanced users only — run twice)
python i2rt/motor_config_tool/set_timeout.py --channel can0
python i2rt/motor_config_tool/set_timeout.py --channel can0

# Re-enable timeout
python i2rt/motor_config_tool/set_timeout.py --channel can0 --timeout
```

> ⚠️ Without the timeout, a failed gravity-compensation loop can produce uncontrolled torque. If you disable it, always initialise with a PD target:
> ```python
> robot = get_yam_robot(channel="can0", zero_gravity_mode=False)
> ```

### Zero motor offsets

```bash
python i2rt/motor_config_tool/set_zero.py --channel can0 --motor_id 1
```

Run for each motor ID (1–6 for a standard YAM).

## Contributing

Pull requests welcome. Open an issue to request examples or report bugs.

## License

MIT License — see [LICENSE](LICENSE).

## Support

- Email: support@i2rt.com
- Sales: sales@i2rt.com

## Acknowledgments

- [TidyBot++](https://github.com/jimmyyhwu/tidybot2) — Flow Base hardware and control inspired by TidyBot++
- [GELLO](https://github.com/wuphilipp/gello_software) — Teleoperation design inspired by GELLO
