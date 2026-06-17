# i2rt.serving — ROS-free robot networking

The YAM rig is driven over **plain TCP**, not ROS. The robot machine runs a
`portal` server that owns the real-time control loop; the workstation connects as
a `portal` client. Policy inference is a separate **websocket** link
(openpi-compatible, see [`policy_serving/`](../../policy_serving)).

No `rclpy`, no DDS, no Python-3.10 ABI constraint — every machine is a clean `uv`
env.

## Two topologies

```
DATA COLLECTION / REPLAY
  robot machine                         workstation
  ┌────────────────────────┐  portal    ┌──────────────────────────┐
  │ run_robot_server teleop │ ◀────────▶ │ lerobot recorder         │
  │  (TeleopController)     │  TCP       │  (PortalBridge polls obs) │
  └────────────────────────┘            └──────────────────────────┘

DEPLOYMENT
  robot machine            workstation                    policy server
  ┌──────────────────┐ portal ┌──────────────────┐ ws+msgpack ┌────────────────┐
  │ run_robot_server │◀──────▶│ policy_bridge     │◀──────────▶│ yam_policy.serve│
  │  dagger          │  TCP   │ (openpi client)   │  obs/chunk │  your model     │
  └──────────────────┘        └──────────────────┘            └────────────────┘
```

## Robot side

```bash
source .venv/bin/activate        # robot env (uv; see scripts/setup_robot_env.sh)

scripts/yam teleop  --bilateral-kp 0.15     # auto home/engage bimanual teleop
scripts/yam dagger  --mirror-kp 0.2         # HG-DAgger: policy drives, button takeover
scripts/yam wrapper                          # followers track an external command (replay)
scripts/yam teleop  --sim                    # no hardware
```

Each launches `python -m i2rt.serving.run_robot_server <mode>` (portal port
**11331** by default). The control core (gating, bilateral teleop, takeover,
rate-limited smoothing) is the same transport-agnostic code as before, in
[`teleop_common.py`](teleop_common.py) / [`safety.py`](safety.py) /
[`control_config.py`](control_config.py); the modes live in
[`controllers.py`](controllers.py).

## Workstation side

```python
from i2rt.serving.robot_client import RobotClient

robot = RobotClient(host="192.168.1.10", port=11331)
obs = robot.get_observation()            # {"left": {pos,vel,eff,...}, "right": {...}, "teleop_state", ...}
robot.command({"left": q_l, "right": q_r})        # wrapper/replay: direct follower target
robot.set_policy_action({"left": q_l, "right": q_r})  # dagger: policy target
robot.set_intervention(True)             # dagger: external gate override
robot.set_estop(True)                    # network e-stop: hold, ignore all commands
```

**Safety:**
- `set_estop(True)` makes every controller stop commanding the followers (they hold
  their last pose) until released; the snapshot carries `estop`.
- Every commanded target is clamped to `control_config.FOLLOWER_JOINT_LIMITS`
  (optional per-joint `[lo, hi]`).
- **Link-loss watchdog:** dagger/wrapper followers hold if no fresh
  `set_policy_action`/`command` arrives within `command_timeout` (default 0.5 s) —
  so a workstation crash or network drop can't leave a stale target driving the arm.
**End-effector (EEF):**
- `observation.eef` (per-arm `[x,y,z,qw,qx,qy,qz]`) is computed by the company
  `Kinematics` FK (mink) from `robot.xml_path`'s `grasp_site` and added to the
  snapshot (zeros if a model isn't available).
- **Safe operational-space control** (`run_robot_server wrapper --control eef`):
  each EE-pose target is resolved to joint positions by `Kinematics.ik` (mink QP,
  joint limits + LM damping), seeded at the current pose, then driven through the
  **same joint path** as joint mode — `command_joint_pos` (MIT impedance) +
  `TargetSmoother` rate limit + joint clamp + e-stop + watchdog. This is resolved-rate
  OSC; it avoids the singularity torque spikes of a torque-level OSC. (Motors run
  MIT/torque mode, so a `Jᵀ·F` torque OSC is possible too but intentionally not the
  default for safety.)

## Snapshot schema (`get_observation()`)

| key | meaning |
|-----|---------|
| `mode` | `teleop` / `dagger` / `wrapper` |
| `t` | robot monotonic timestamp |
| `teleop_state` | `HOMING`/`IDLE`/`ENGAGED` (teleop) — the episode gate signal |
| `active` | True iff ENGAGED (teleop) |
| `intervention` | gate state (dagger) |
| `<side>.pos/vel/eff` | follower full state (len `num_dofs`, trailing gripper) |
| `<side>.leader_pos` | leader joints (teleop/dagger) |
| `<side>.applied` | the rate-limited command actually sent (the action) |
| `<side>.human` | leader target while intervening (dagger) |

## Why portal (and not ROS)

The control logic never needed ROS — ROS was only the network bus. `portal` is
already an i2rt dependency (used by `ServerRobot`/`ClientRobot` and the flow base),
it's plain TCP (works across VPN/tailscale, no DDS multicast), and it frees both
machines from the rclpy/Python-3.10 constraint.
