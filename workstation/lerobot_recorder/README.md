# YAM ↔ LeRobot Recorder

Records [LeRobot](https://github.com/huggingface/lerobot) datasets from the
bimanual YAM teleop rig. Runs on the **workstation** (a different machine / env
than the robot): it connects to the YAM ROS 2 graph **remotely**, reads three
RealSense cameras **locally**, and **auto-starts/stops each episode from the
teleop gate** — no manual record button per episode.

## How recording is triggered (the key idea)

The teleop node (`i2rt.ros2.run_teleop`, on the robot machine) publishes
`/teleop/state` (`HOMING` / `IDLE` / `ENGAGED`). The recorder watches it:

```
 you press "Start collection"  ──▶  gate armed
 both gellos lifted → ENGAGED  ──▶  episode STARTS recording   ◀── auto
 …teleoperate…       (ENGAGED)      every frame is recorded
 gellos back to home → HOMING  ──▶  still recording (the return is included)
 homing done → IDLE            ──▶  episode SAVED              ◀── auto
```

So one episode = **ENGAGED → HOMING → IDLE**. The action stored is the robot's
**`applied_action`** (the rate-limited command actually sent), so a policy
trained on it reproduces the episode precisely.

## Dataset schema

| Key | Shape | Source |
|-----|-------|--------|
| `observation.images.wrist_left`  | (H,W,3) uint8 | D405 (left wrist) |
| `observation.images.wrist_right` | (H,W,3) uint8 | D405 (right wrist) |
| `observation.images.agentview`   | (H,W,3) uint8 | D455 (scene) |
| `observation.state` | (42,) float32 | both arms × [pos(7), vel(7), eff(7)] |
| `action`            | (14,) float32 | both arms × `applied_action`(7) |
| task                | string | the language instruction |

Recorded at **60 fps** (matched to the cameras).

## Setup

This is a **separate environment** from the robot's `i2rt_ros`.

```bash
# 1) a venv that can see ROS 2's rclpy
python -m venv ~/lerobot_env --system-site-packages
source ~/lerobot_env/bin/activate
source /opt/ros/<distro>/setup.bash          # SAME distro as the robot machine
pip install -r workstation/lerobot_recorder/requirements.txt

# 2) remote ROS 2: same domain id on BOTH machines (robot + workstation)
export ROS_DOMAIN_ID=42
#    (and make sure the two machines are on the same network / DDS can discover)

# 3) find your RealSense serials and map them to roles
python -m workstation.lerobot_recorder.cameras --list
```

Put the serials in `config.py` (`default_cameras`) or pass `--serials`.

## Run

```bash
# real:
python -m workstation.lerobot_recorder \
    --repo-id user/yam_pick --task "pick up the cube" \
    --serials <wrist_left_sn>,<wrist_right_sn>,<agentview_sn>

# dry run with no robot/cameras/lerobot (synthetic teleop cycle + fake frames):
python -m workstation.lerobot_recorder --mock
```

In the GUI:

1. Set `repo_id` / `root` / `task`, then **Start** (opens cameras + ROS + dataset).
2. **Start collection** to arm the auto-gate. Now just teleoperate: lift both
   gellos to begin an episode, bring them home to end it (the recorder saves on
   homing-complete). The instruction field can be edited between episodes.
3. **Stop collection** to disarm (a half-finished episode is discarded).

The status row shows the live teleop state, a **REC** indicator, the episode
count, and a small preview of each camera.

## Notes

- The record loop is clocked at `--fps` (60); cameras stream at 60 fps and each
  tick grabs the latest frame, plus the latest robot state/action from ROS.
- `lerobot`'s dataset API has changed across releases; the few version-sensitive
  calls are isolated in `dataset_writer.py` — tweak there if your version differs.
- Nothing here imports `i2rt`; it only needs the ROS topic contract from
  `i2rt/ros2/README.md`.
