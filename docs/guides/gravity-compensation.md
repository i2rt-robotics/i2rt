# Gravity & Friction Compensation

YAM arms compute gravity torques from their MuJoCo model and add them to every motor command, so the arm "floats" against gravity. From v1.1.1 the same control law also includes per-joint **Coulomb friction compensation** and per-joint **motor-side idle damping (`grav_comp_kd`)**. Together they make the arm easy to backdrive while keeping idle behaviour stable.

This page covers what each knob does, how to tune them, and how the sim mirrors the hardware.

## How the torque command is built

In gravity-comp idle (`zero_gravity_mode=True`, no active control command), each motor receives:

```
τ = g(q) · gravity_comp_factor  +  coulomb_friction · sign(q_dot)
```

plus motor-side MIT-mode `kd` damping equal to `grav_comp_kd[i]` running at kHz on the motor's onboard loop.

Under an active PD command (`command_joint_pos` / `command_joint_state`):

```
τ = τ_PD  +  g(q) · gravity_comp_factor  +  coulomb_friction · sign(q_dot)
```

`grav_comp_kd` is **not** applied during PD control — `self._kd` is used instead, so PD tracking gains are unchanged. The relevant code lives in `i2rt/robots/motor_chain_robot.py:340-355`.

## Per-arm YAML configuration

Tuning lives in `i2rt/robots/config/<arm>.yml`. The current YAM defaults:

```yaml
gravity_comp_factor: [1.0, 1.1, 1.1, 1.2, 1.0, 1.0]
grav_comp_kd:        [0.1, 0.1, 0.1, 0.3, 0.05, 0.05]
coulomb_friction:    [0.3, 0.3, 0.3, 0.06, 0.06, 0.06]
```

| Field | Units | Purpose |
|-------|-------|---------|
| `gravity_comp_factor` | unitless multiplier on `g(q)` | Compensates per-joint underdrive (gear loss, harness drag). Six elements, one per arm joint. |
| `grav_comp_kd` | MIT-mode kd (motor units) | Passive idle damping on the motor's onboard loop. Six elements; only active when the robot is in gravity-comp idle. |
| `coulomb_friction` | Nm | Magnitude of `friction · sign(q_dot)` injection. Six elements; cancels static stiction so light pushes move the joint. |

The gripper slot is automatically padded with `0.0` in [`i2rt/robots/get_robot.py:206-207`](https://github.com/i2rt-robotics/i2rt/blob/main/i2rt/robots/get_robot.py) — gripper joints are position-commanded and need no passive compensation.

::: tip Per-arm defaults
`yam_pro.yml`, `yam_ultra.yml`, and `big_yam.yml` ship sensible starting values, but the wrist joints in particular vary with payload — expect to retune `grav_comp_kd[3]` (J4) and `coulomb_friction[3]` on a custom build.
:::

## Runtime override

Only `gravity_comp_factor` can be overridden per-call:

```python
import numpy as np
from i2rt.robots.get_robot import get_yam_robot

robot = get_yam_robot(
    channel="can0",
    gravity_comp_factor=np.array([1.0, 1.15, 1.10, 1.25, 1.0, 1.0]),
)
```

`grav_comp_kd` and `coulomb_friction` are YAML-only — they're hardware-tuning constants tied to the specific arm build, so they live with the arm config.

## Mode transitions: `enter_gravity_comp_idle()`

After issuing PD commands you can return the arm to a clean gravity-comp idle:

```python
robot.command_joint_pos(target)   # active PD
# ... do work ...
robot.enter_gravity_comp_idle()   # back to floating, with grav_comp_kd damping
```

This sets `self._commands` to zeros with `kd = grav_comp_kd`, so the arm stops actively tracking the last target and resumes the float-and-damp behaviour. The MuJoCo viewer (`examples/control_with_mujoco/`) calls this automatically on every CONTROL → VIS toggle.

::: warning Why this matters
Before v1.1.1 the viewer used `update_kp_kd(zeros, zeros)` on toggle. That only rewrote member caches and left the stale PD target in `_commands`, so the motors kept holding the last CONTROL pose — felt as per-joint friction after returning to VIS. If you call `update_kp_kd` directly, follow it with `enter_gravity_comp_idle()` to clear the stale target.
:::

## Simulation parity

`SimRobot` runs a daemon **physics thread** (`_physics_loop` in `i2rt/robots/sim_robot.py`) that applies model-based gravity comp at the same `gravity_comp_factor` schedule as hardware. The MuJoCo viewer toggles it via:

```python
robot.enable_gravity_comp()    # VIS mode
robot.disable_gravity_comp()   # CONTROL mode (teleport on; physics paused)
```

`get_yam_robot(sim=True)` passes a unit factor (`np.ones(...)`) so sim uses the unscaled MuJoCo gravity — there's no harness-loss or gear-loss to compensate for. `coulomb_friction` and `grav_comp_kd` flow through but have no observable effect in sim, because the MuJoCo model has no Coulomb friction and no MIT-mode motor model.

## Tuning workflow

Start from the per-arm defaults and adjust in this order:

1. **Set `gravity_comp_factor`.** Lift the arm to a horizontal pose and release. Increase `gravity_comp_factor[i]` on whichever joint sags first; decrease on whichever drifts up. Tune one joint at a time, shoulder-to-wrist.
2. **Set `coulomb_friction`.** Once the arm holds pose, give each joint a small push. If a joint resists at first then "snaps free", raise its `coulomb_friction` slightly. Stop just before joints start drifting on their own.
3. **Set `grav_comp_kd` only if needed.** If a joint chatters or limit-cycles after release, raise its `grav_comp_kd`. Most joints don't need it (defaults near 0.05–0.1); only J4 typically needs more on bigger wrists.

::: tip Quick test loop
Use the zero-gravity entry point for fast iteration:

```bash
python i2rt/robots/get_robot.py --channel can0 --gripper linear_4310
```

Edit the YAML, re-run, repeat. No code changes needed.
:::

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|------|-----|
| Joint sags slowly under gravity | `gravity_comp_factor` too low | Raise the per-joint factor by 0.05–0.10 increments |
| Joint drifts up against gravity | `gravity_comp_factor` too high | Lower the per-joint factor |
| Joint chatters / limit-cycles when released | Stiction interacting with under-compensation; or `grav_comp_kd` too low | First, recheck `gravity_comp_factor`; if clean, raise `grav_comp_kd` |
| Joint feels "sticky" — needs a push to start moving, then moves freely | `coulomb_friction` too low | Raise the per-joint friction value |
| Joint walks on its own after release | `coulomb_friction` too high | Lower the per-joint friction value |
| Arm "holds" the previous PD target after CONTROL → VIS | `enter_gravity_comp_idle()` was not called | Use the v1.1.1 viewer (auto-calls), or call it yourself after `command_joint_pos` |
| Sim arm flops to the floor | `enable_gravity_comp()` was not called or `start_server()` not invoked | Use `mujoco_control_interface` (calls both for you) |

## See also

- [YAM Arm SDK reference](/sdk/yam-arm) — `get_yam_robot()` arguments and the `MotorChainRobot` API.
- [MuJoCo Control example](/examples/control-with-mujoco) — interactive grav-comp testbed.
- [v1.1.1 release notes](/releases/v1.1.1) — the PR that introduced these knobs.
