# Drive Flow Base (Viser)

Slider-driven teleoperation for the i2rt flow_base mobile base, with live odometry visualization
in [viser](https://github.com/nerfstudio-project/viser). Three velocity sliders set the base's
local-frame velocity directly — vx (forward/back), vy (strafe), and ω (yaw) — a coordinate
**axis** in the browser scene tracks the base's odometry pose on a ground grid, and the panel
shows the live **control-loop rate (Hz)**.

The flow_base is a 4-module swerve drive, so it is holonomic: vx, vy, and ω are independently
drivable.

## Quick Start

```bash
# On a machine on the base's CAN bus (drives a local Vehicle):
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py

# From a workstation, talking to a controller already running on the base:
uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --host 172.6.2.20
```

Then open `http://localhost:8080` in your browser, drag the sliders to drive, and press **Stop**
to re-center all sliders to zero. Quit with `Ctrl+C` in the terminal.

## Controls

| Control | Action |
|---------|--------|
| `vx` slider | Forward / back velocity (m/s) |
| `vy` slider | Strafe left / right velocity (m/s) |
| `ω` slider | Yaw velocity (rad/s) |
| **stop** button | Re-center all velocity sliders to 0 |
| **reset odometry** button | Reset odometry to the origin |

Velocity is commanded in the **local (body) frame**, so the sliders are relative to the base's
current heading. The sliders **hold** their value, so the base keeps moving until you re-center
them (or press **Stop**) — there is no auto-stop.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | unset | FlowBase controller host. If unset, drives a local `Vehicle` over CAN. |
| `--channel` | `can_flow_base` | CAN channel for the local `Vehicle` (ignored when `--host` is set). |
| `--port` | `8080` | Viser server port. |
| `--max-linear` | `0.5` | The vx / vy sliders range over ±this (m/s). |
| `--max-angular` | `1.57` | The ω slider ranges over ±this (rad/s). |
| `--control-hz` | `50.0` | Control-loop rate: command + scene refresh frequency. Keep it well above the base's 0.25 s (4 Hz) safety timeout. |

## Notes

- **Hardware required.** The local path needs the base's CAN bus; the `--host` path needs a
  flow_base controller already running on the base (see `i2rt/flow_base/README.md`).
- **Safety timeout.** The base stops on its own if it receives no command for 0.25 s, so a crash
  or disconnect of this script will not leave it driving.
