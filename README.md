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

## Networking & deployment

The YAM rig is exposed to a workstation over **`portal`** (plain TCP); policy
inference is a separate **websocket** link (openpi-compatible). Bimanual by default
(2 leaders + 2 followers).

```bash
source .venv/bin/activate                       # robot env (uv; scripts/setup_robot_env.sh)
python -m i2rt.serving.run_robot_server teleop  --sim   # auto home/engage teleop
python -m i2rt.serving.run_robot_server dagger  --sim   # HG-DAgger: policy + button takeover
python -m i2rt.serving.run_robot_server wrapper --sim   # followers track an external command

# …or the shortcut launcher (activates the env for you):
scripts/yam teleop --sim                        # also: dagger / wrapper / can / canup
```

Targets are rate-limited and gravity compensation is always on, so policy↔human
takeovers ramp smoothly.

### Data collection & deployment stack

| Subsystem | Path | What it is |
|-----------|------|------------|
| Robot serving (portal) | [`i2rt/serving/`](i2rt/serving/README.md) | teleop / DAgger / wrapper servers + `RobotClient`; snapshot contract; safety (e-stop, joint/effort limits, link-loss watchdog), EEF FK + safe resolved-rate OSC |
| Policy serving (websocket) | [`policy_serving/`](policy_serving/README.md) | openpi-compatible `WebsocketPolicyServer`/`Client` + `serve.py` + policy templates |
| Workstation tools | [`workstation/lerobot_recorder/`](workstation/lerobot_recorder/README.md) | LeRobot recorder (teleop/dagger/eval), replay+overlay, policy bridge; modern themed GUI with status banner, health, live stats, audio cues, success/fail/discard labeling |

Quick CLIs (workstation): `workstation/yam-data {record\|replay\|bridge\|cams\|doctor}`.

**One config for everything**: edit the tracked [`config.yaml`](config.yaml) at the
**repo root** — robot host/port, control gains/limits, camera serials, recorder
defaults, tasks, and the policy endpoint. Every tool auto-discovers `<repo>/config.yaml`
(no env var, no matter the directory); no `--config` needed. Precedence:
**CLI flag > `config.yaml` > default**.

**Envs**: the **robot** is uv-managed — `scripts/yam …` uses `uv run`, nothing to
activate. The **workstation** is a **conda** env (`scripts/setup_workstation_env.sh`)
with this repo installed via uv, so you can `pip install` other policy repos into the
same env. The **policy server** env is unconstrained (its own conda/uv).

**Safety & ops highlights**: network E-STOP, per-joint position + optional effort
(collision) limits, command-staleness watchdog (link loss → hold), async dataset
writer (collect while saving), camera-disconnect auto-reconnect, disk-space guard,
and a per-episode `outcomes.jsonl` (`yam-data doctor` summarizes success rates).

**Verify on hardware**: follow [`docs/hardware-checklist.md`](docs/hardware-checklist.md)
— an ordered, runnable confirmation list for every feature.

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
