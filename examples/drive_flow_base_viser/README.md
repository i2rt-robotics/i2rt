# Drive Flow Base (Viser)

Slider-driven teleoperation for the i2rt flow_base mobile base, with live odometry visualization
in [viser](https://github.com/nerfstudio-project/viser). Three velocity sliders set the base's
local-frame velocity directly — vx (forward/back), vy (strafe), and ω (yaw) — a coordinate
**axis** in the browser scene tracks the base's odometry pose on a ground grid, and the panel
shows the live **control-loop rate (Hz)**.

The flow_base is a 4-module swerve drive, so it is holonomic: vx, vy, and ω are independently
drivable.

With `--linearbot` the base is a **LinearBot** (flow base + linear rail lift): a fourth slider
drives the rail, and the scene shows the LinearBot model
([i2rt/robot_models/linearbot/](../../i2rt/robot_models/linearbot/), `linearbot.urdf` and the
`linearbot.xml` generated from it) following the odometry pose, with its rail holder at the
reported height and its casters steering and wheels turning from the motor feedback.
`--left-arm` / `--right-arm` mount an arm on each end of the rail carriage, with a slider per
joint (and the gripper) in its own panel folder. `--sim` replaces all the hardware with
stand-ins, so the whole thing runs on any machine.

## Quick Start

```bash
# On a machine on the base's CAN bus (drives a local Vehicle):
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py

# From a workstation, talking to a controller already running on the base:
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --host 172.6.2.20

# No hardware (kinematic sim):
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --sim

# A LinearBot, combined with any of the above:
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --linearbot --sim
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --linearbot --host 172.6.2.20

# ... with two arms on the carriage (each on its own CAN channel):
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --linearbot --sim --left-arm yam --right-arm yam
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --linearbot --host 172.6.2.20 \
    --left-arm yam --left-gripper linear_4310 --left-channel can_left \
    --right-arm yam --right-gripper linear_4310 --right-channel can_right
```

Then open `http://localhost:8080` in your browser, drag the sliders to drive, and press **Stop**
to re-center all sliders to zero. Quit with `Ctrl+C` in the terminal.

## Controls

| Control | Action |
|---------|--------|
| `vx` slider | Forward / back velocity (m/s) |
| `vy` slider | Strafe left / right velocity (m/s) |
| `ω` slider | Yaw velocity (rad/s) |
| `vz` slider | Linear rail velocity, positive = up (m/s); `--linearbot` only |
| **enable robot** checkbox | Send the base (and arm) commands; unchecked, the base halts and nothing is commanded |
| **stop** button | Re-center all velocity sliders to 0 |
| **reset odometry** button | Reset odometry to the origin |
| **left / right arm** `mode` | `viz (grav comp)`: the arm floats in gravity compensation, movable by hand, and its sliders follow it. `sliders`: its sliders command it |
| **left / right arm** `j1`…`jN`, `gripper` | Joint angles (deg) over the model's joint ranges, gripper 0 (closed) to 1 (open); `sliders` mode only |

Velocity is commanded in the **local (body) frame**, so the sliders are relative to the base's
current heading. The sliders **hold** their value, so the base keeps moving until you re-center
them (or press **Stop**) — there is no auto-stop.

Each arm starts in **viz** mode. Switch it to **sliders** to drive it; the sliders command the arm
only while "enable robot" is checked, and **follow the arm** otherwise, so neither switching mode
nor enabling ever jumps it. Switching back to viz drops the position hold and returns the arm to
gravity compensation.

A slider command that would collide is **blocked** (the terminal says so) until the sliders leave
the collision — the same rule the YAM control interfaces apply, checked against the whole robot:
the arm into itself, the other arm, the rail carriage, the column or the base. The check composes
the LinearBot model with both arms at their mounts, so the other arm is checked where it actually
is, and the base where the rail and casters actually are.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | unset | FlowBase controller host. If unset, drives a local `Vehicle` over CAN. |
| `--channel` | `can_flow_base` | CAN channel for the local `Vehicle` (ignored when `--host` is set). |
| `--port` | `8080` | Viser server port. |
| `--sim` | off | Drive a kinematic stand-in instead of a base: the pose integrates the slider velocities. |
| `--linearbot` | off | The base carries a linear rail: add the `vz` slider and show the LinearBot model. |
| `--left-arm` / `--right-arm` | unset | Arm on that end of the carriage (`yam`, `yam_pro`, ...); needs `--linearbot`. |
| `--left-gripper` / `--right-gripper` | `linear_4310` | Gripper on that arm. |
| `--left-channel` / `--right-channel` | `can_left` / `can_right` | CAN channel of that arm (ignored with `--sim`). |
| `--max-linear` | `0.5` | The vx / vy (and rail vz) sliders range over ±this (m/s). |
| `--max-angular` | `1.57` | The ω slider ranges over ±this (rad/s). |
| `--control-hz` | `50.0` | Control-loop rate: command + scene refresh frequency. Keep it well above the base's 0.25 s (4 Hz) safety timeout. |

## Notes

- **Hardware required.** The local path needs the base's CAN bus; the `--host` path needs a
  flow_base controller already running on the base (see `i2rt/flow_base/README.md`), started with
  `--linear-rail` for `--linearbot`. `--sim` needs neither.
- **Local `--linearbot` homes the rail at startup — it moves** before you enable anything, exactly
  as the base's own controller does, because homing is what calibrates the rail's height. The brake's
  USB-GPIO port is `/dev/ttyUSB0` unless `I2RT_USB_GPIO_PORT` names another (native GPIO on a Pi).
- **`--sim` is kinematic:** the pose follows the commanded velocity instantly (no acceleration
  limits, no safety timeout) and the rail height is clipped to its 1 m stroke. The casters steer
  at the rates the real controller commands, so they swing into their velocity rather than snap.
  Its arms are `SimRobot`s.
- **Real arms start in gravity compensation** (viz mode) — powered, but movable by hand — as soon
  as the script runs; the first slider command puts an arm under position control. The arms
  are connected locally on their CAN channels even with `--host`.
- **Collision blocks slider commands only.** An arm moved by hand in viz mode, or a rail or base
  motion, is not stopped. MuJoCo collides meshes by their convex hulls, so the check is
  conservative around concave parts such as the column's braces.
- **Safety timeout.** The base stops on its own if it receives no command for 0.25 s, so a crash
  or disconnect of this script will not leave it driving.
