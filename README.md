# I2RT Python API

A Python client library for interacting with [I2RT](https://i2rt.com/) products — designed for learning-based robotics, teleoperation, and real-world deployment.

[![I2RT](https://github.com/user-attachments/assets/025ac3f0-7af1-4e6f-ab9f-7658c5978f92)](https://i2rt.com/)

> 📚 **Full documentation:** [doc.i2rt.com](https://doc.i2rt.com)

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

### Arm variants

Every arm is six-DOF. Pick one with `--arm` on any CLI, or `arm_type=` in Python. A hardware
revision is its own variant — `yam_ultra_2` is revision 2 of the YAM Ultra.

| Arm | `--arm` value | Arm mass | Chain length | Notes |
|-----|---------------|---------:|-------------:|-------|
| YAM | `yam` | 4.292 kg | 0.807 m | [Model docs](i2rt/robot_models/arm/yam/v1/README.md) |
| YAM Pro | `yam_pro` | 4.349 kg | 0.798 m | [Model docs](i2rt/robot_models/arm/yam_pro/v1/README.md) |
| YAM Ultra | `yam_ultra` | 4.521 kg | 0.813 m | [Model docs](i2rt/robot_models/arm/yam_ultra/v1/README.md) |
| YAM Ultra 2 | `yam_ultra_2` | 4.597 kg | 0.813 m | Revision 2 — kinematically identical to `yam_ultra`; revised link3/link4 inertials and a DM4340 on joint 4. [Model docs](i2rt/robot_models/arm/yam_ultra/v2/README.md) |
| Big YAM | `big_yam` | 5.307 kg | 1.060 m | [Model docs](i2rt/robot_models/arm/big_yam/v1/README.md) |
| *(none)* | `no_arm` | — | — | Gripper-only robot |

"Arm mass" excludes the gripper and tips. "Chain length" is the sum of the parent-to-child
joint-origin translation norms — a measure of the kinematic chain, not of reach. Each arm's
model README tabulates full masses, COMs, inertias, joint frames, DH parameters, and screw axes.

### Zero-gravity mode

```bash
python i2rt/robots/motor_chain_robot.py --channel can0 --arm yam --gripper linear_4310
```

### Python API

```python
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType
import numpy as np

robot = get_yam_robot(
    channel="can0",
    arm_type=ArmType.YAM,
    gripper_type=GripperType.LINEAR_4310,
)

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
| `flexible_4310` | DM4310 | Linear with flexible soft tips for compliant grasps |
| `yam_teaching_handle` | — | Leader arm handle with trigger + 2 buttons. |
| `no_gripper` | — | Bare terminal mount; no gripper motor on the chain |

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
| Viser (browser) control interface | `examples/control_with_viser/` |
| Drive Flow Base from Viser | `examples/drive_flow_base_viser/` |
| Plot Flow Base telemetry in Rerun | `examples/plot_flow_base_rerun/` |

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

### Motor registers

Read and write a motor's configuration registers (IDs, control mode, protection thresholds, MIT
scaling, loop gains) over CAN. Full register table and safety notes:
[dm_motor_registers.md](i2rt/motor_config_tool/dm_motor_registers.md).

```bash
# List every register the tool knows about (no CAN bus needed)
python i2rt/motor_config_tool/dm_motor_registers.py list-registers

# Read one register, or dump them all
python i2rt/motor_config_tool/dm_motor_registers.py read sw_ver --motor-id 1 --channel can0
python i2rt/motor_config_tool/dm_motor_registers.py read-all --motor-id 1 --channel can0

# Change a value: write puts it in RAM, save commits it to Flash
python i2rt/motor_config_tool/dm_motor_registers.py write MAX_SPD --value 30.0 --motor-id 1
python i2rt/motor_config_tool/dm_motor_registers.py save MAX_SPD --motor-id 1
```

> ⚠️ Stop any program using the bus first, and only ever `save` a register you have just successfully
> written — saving an unwritten register has been observed to revert *other* unsaved registers to their
> stored values. Changing `ESC_ID`, `MST_ID`, `CTRL_MODE`, `can_br`, `TIMEOUT`, `PMAX`, `VMAX` or `TMAX`
> asks for confirmation on the console first.

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
