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

Every arm is six-DOF except the two `yam_7dof_*` arms. Pick one with `--arm` on any CLI, or
`arm_type=` in Python. A hardware revision is its own variant — `yam_ultra_2` is revision 2 of the
YAM Ultra — and so is each side of a mirrored pair: `yam_7dof_left` and `yam_7dof_right` are separate
arms, not one arm with a side flag.

| Arm | `--arm` value | Arm mass | Chain length | Notes |
|-----|---------------|---------:|-------------:|-------|
| YAM | `yam` | 4.292 kg | 0.807 m | [Model docs](i2rt/robot_models/arm/yam/v1/README.md) |
| YAM Pro | `yam_pro` | 4.349 kg | 0.798 m | [Model docs](i2rt/robot_models/arm/yam_pro/v1/README.md) |
| YAM Ultra | `yam_ultra` | 4.521 kg | 0.813 m | [Model docs](i2rt/robot_models/arm/yam_ultra/v1/README.md) |
| YAM Ultra 2 | `yam_ultra_2` | 4.597 kg | 0.813 m | Revision 2 — kinematically identical to `yam_ultra`; revised link3/link4 inertials and a DM4340 on joint 4. [Model docs](i2rt/robot_models/arm/yam_ultra/v2/README.md) |
| Big YAM | `big_yam` | 5.307 kg | 1.060 m | [Model docs](i2rt/robot_models/arm/big_yam/v1/README.md) |
| YAM 7-DOF (right) | `yam_7dof_right` | 5.844 kg | 0.822 m | Seven joints, plate-mounted. Its gripper is on CAN `0x08`. [See below](#7-dof-arm) |
| YAM 7-DOF (left) | `yam_7dof_left` | 5.844 kg | 0.822 m | The right arm mirrored about the world x–z plane. [See below](#7-dof-arm) |
| *(none)* | `no_arm` | — | — | End effector on its own. A motorized gripper keeps its motor; either teaching handle is passive, so it is read-only (trigger and buttons from its encoder) |

"Arm mass" excludes the gripper and tips. "Chain length" is the sum of the parent-to-child
joint-origin translation norms — a measure of the kinematic chain, not of reach. Each arm's
model README tabulates full masses, COMs, inertias, joint frames, DH parameters, and screw axes.

Whichever end effector you select adds its own mass at the flange, and gravity compensation sees
it: `flexible_4310` 0.697 kg, `linear_4310` 0.695 kg, `crank_4310` 0.600 kg, `linear_3507`
0.445 kg, `yam_teaching_handle_v1` 0.258 kg, and `yam_teaching_handle_v2` 0.244 kg (`no_gripper`
is a bare mount and adds nothing). Each teaching handle has a model README of its own
([v1](i2rt/robot_models/gripper/yam_teaching_handle/v1/README.md),
[v2](i2rt/robot_models/gripper/yam_teaching_handle/v2/README.md)); the motorized grippers
are documented by their MJCF files.

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
python examples/minimum_gello/minimum_gello.py --gripper yam_teaching_handle_v1 --mode leader --can-channel can1 --bilateral-kp 0.2
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

## 7-DOF Arm

Seven-DOF arms, built and composed exactly like the six-DOF ones: `arm_type=ArmType.YAM_7DOF_RIGHT`
(or `..._LEFT`) on `get_yam_robot`, with any of the interchangeable grippers below. Two things set
them apart — they are a **mirrored left/right pair**, registered as two arm types rather than one
plus a side flag, and their gripper sits on CAN `0x08` because their own seven joints have taken
`0x01`–`0x07`.

| Property | Value |
|----------|------:|
| Arm DOF | 7 (`joint1`–`joint7`) |
| Gripper DOF | 1 — two equality-coupled fingers (`joint8`/`joint9`), `q=0` closed |
| Arm mass | 5.844 kg (`base`–`link6`; the end effector's mass arrives with the gripper) |
| Total mass, with `linear_4310` | 6.539 kg |
| Reach, origin → TCP at home | 0.808 m |
| Motors | DM6248 ×2, DM4340 ×3, DM4310 ×2, plus the gripper's |
| CAN ids | `0x01`–`0x07` arm, `0x08` gripper |
| Mounting | Plate mount: mounting face is the world x–z plane through the origin; `joint1` about −Y; arm hangs along −Z at home. Face normal +Y on the right arm, −Y on the left |

Model docs tabulate full masses, COMs, inertias, joint frames and screw axes, per arm:
[right](i2rt/robot_models/arm/yam_7dof_right/v1/README.md),
[left](i2rt/robot_models/arm/yam_7dof_left/v1/README.md).

> **`gravity_comp_factor` is bench-tuned against this model**, replacing values found against an
> earlier ONShape export whose links 2–5 were 0.536 kg lighter. The right arm's model docs tabulate
> the per-joint ratios. The left arm has its own, fitted from torque sweeps on a left arm.

```python
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType

robot = get_yam_robot(channel="can0", arm_type=ArmType.YAM_7DOF_RIGHT)            # or sim=True
robot = get_yam_robot(channel="can1", arm_type=ArmType.YAM_7DOF_LEFT)              # the mirrored arm
robot = get_yam_robot(channel="can0", arm_type=ArmType.YAM_7DOF_RIGHT,             # any gripper
                      gripper_type=GripperType.CRANK_4310)
robot.command_joint_pos([0, 0, 0, 0, 0, 0, 0, 0])   # 7 arm joints + gripper (0 closed, 1 open)
```

Drive it from an interactive MuJoCo viewer with
[`examples/control_with_mujoco/`](examples/control_with_mujoco/) or from a browser with
[`examples/control_with_viser/`](examples/control_with_viser/), in both cases with
`--arm yam_7dof_right` or `--arm yam_7dof_left`.

### Sides

The left arm is the right one **reflected about the world x-z plane** (`y → -y`), generated by
`.claude/skills/align-urdf-mjcf/scripts/mirror_arm_model.py`. It is the same hardware mounted on the
opposite face of the plate, not a separate revision: masses, inertias, link lengths and every value in
`yam_7dof_<side>_v1.yml` are shared. Each arm is nonetheless its own family, with its own model dir,
config and gripper-mount block, so the two can carry different bench-tuned values.

The two sides are **true mirror images**: an identical joint vector puts them in mirror-symmetric
poses, so `q_left = q_right` joint for joint, and their joint ranges are identical. Their `directions`
differ on joint3, joint5 and joint6, whose motors on the left do not reverse with their axes.
What differs is the world direction each joint turns about, because a rotation axis is a pseudovector
— the joints whose axis lies along the mirror normal (`±y`) agree between the sides, and those lying
in the mirror plane oppose:

| Joint | j1 | j2 | j3 | j4 | j5 | j6 | j7 |
|-------|----|----|----|----|----|----|----|
| Right, world axis at home | `-y` | `-x` | `-z` | `-y` | `-z` | `-x` | `-y` |
| Left, world axis at home | `-y` | `+x` | `+z` | `-y` | `+z` | `+x` | `-y` |

The [left model docs](i2rt/robot_models/arm/yam_7dof_left/v1/README.md) give the full derivation.

The **gripper** is not mirrored: it is one physical part that bolts to either arm. So a left arm is
the mirror of a right one through the mount frame and at the tool centre, but its two fingers are
not each other's reflection.

### Where the models live

`i2rt/robot_models/arm/yam_7dof_right/v1/` and `.../yam_7dof_left/v1/` — two ordinary arm families,
laid out exactly like every other: `<family>.urdf` is the source of truth, `<family>.xml` is
generated from it and is arm-only, `assets/` sits beside them, and the terminal `gripper` body is a
`mass=1e-6` mount placeholder that runtime composition fills in. A future hardware revision would be
`v2` under both families.

The file basenames and the MJCF `<mujoco model>` name must equal the family name: `get_arm_xml_path`
builds the path from it, and `sync_gripper_mounts.py` keys the gripper-mount block off it.

## Gripper Types

| Gripper | Motor | Notes |
|---------|-------|-------|
| `crank_4310` | DM4310 | Zero-linkage crank — minimises gripper width |
| `linear_3507` | DM3507 | Lightweight linear; start closed or run calibration |
| `linear_4310` | DM4310 | Standard linear; slightly more force than 3507 |
| `flexible_4310` | DM4310 | Linear with flexible soft tips for compliant grasps |
| `yam_teaching_handle_v1` | — | Leader arm handle with trigger + 2 buttons. |
| `yam_teaching_handle_v2` | — | The same handle with a 38 mm shorter shell. |
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
