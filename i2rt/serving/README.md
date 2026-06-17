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
```

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
