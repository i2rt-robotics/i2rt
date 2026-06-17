# YAM ↔ LeRobot Recorder & Replay

Record and replay [LeRobot](https://github.com/huggingface/lerobot) **v3.0**
datasets for the bimanual YAM teleop rig. Runs on the **workstation** (a
different machine / env than the robot): it connects to the YAM ROS 2 graph
**remotely**, reads three RealSense cameras **locally**, and **auto-starts/stops
each episode from the teleop gate** — no manual record button per episode.

Two tools (both have a PyQt GUI):

| Tool | Module | What it does |
|------|--------|--------------|
| **Recorder** | `workstation.lerobot_recorder` | teleop-gated LeRobot capture + review/delete |
| **Replay**   | `workstation.lerobot_recorder.replay_main` | play a dataset back onto the robot |

## How recording is triggered (the key idea)

`i2rt.ros2.run_teleop` (on the robot machine) publishes `/teleop/state`
(`HOMING` / `IDLE` / `ENGAGED`). The recorder watches it:

```
 press "Start collection"      ──▶  gate armed
 both gellos lifted → ENGAGED  ──▶  episode STARTS recording   ◀── auto
 …teleoperate…       (ENGAGED)      every frame recorded
 gellos home → HOMING          ──▶  still recording (return included)
 homing done → IDLE            ──▶  episode ends → REVIEW       ◀── auto
   review playback → Keep / Delete
```

One episode = **ENGAGED → HOMING → IDLE**. The action stored is the robot's
**`applied_action`** (the rate-limited command actually sent) → reproducible.

## Dataset schema (LeRobot v3.0)

| Key | Shape | Source |
|-----|-------|--------|
| `observation.images.wrist_left`  | (H,W,3) uint8 | D405 left wrist |
| `observation.images.wrist_right` | (H,W,3) uint8 | D405 right wrist |
| `observation.images.agentview`   | (H,W,3) uint8 | D455 scene |
| `observation.state` | (42,) float32 | both arms × [pos(7), vel(7), eff(7)] |
| `action`            | (14,) float32 | both arms × `applied_action`(7) |
| task                | string | the language instruction |

Recorded at **60 fps** (matched to the cameras). Uses the official v3.0 API
(`create` / `add_frame` with a `task` key / `save_episode` / `clear_episode_buffer`
/ **`finalize`**); the version-sensitive calls live in `dataset_writer.py`.

---

# One-time setup

### [robot machine] — YAM + ROS 2 (the `.venv-ros` uv venv)

Already covered in [`i2rt/ros2/README.md`](../../i2rt/ros2/README.md). In short:

```bash
source .venv-ros/bin/activate            # ROS 2 Humble auto-sourced
sh scripts/setup_can_ids.sh              # persistent CAN names (once)
export ROS_DOMAIN_ID=64                  # SAME value on both machines
```

### [workstation] — prerequisites

The workstation needs **ROS 2 Humble** (same distro as the robot, so DDS
discovery matches) and **uv**:

```bash
# ROS 2 Humble (Ubuntu 22.04) — skip if already installed.
# Full guide: https://docs.ros.org/en/humble/Installation.html
sudo apt install -y ros-humble-ros-base

# uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
```

### [workstation] — Python env (uv)

The recorder needs `rclpy` (from ROS 2, **not** pip-installable) plus pip deps
(`lerobot`, `pyrealsense2`, `PyQt5`). So the venv must use the **system Python
3.10** that Humble's `rclpy` was built against, with `--system-site-packages` so
the ROS Python packages are visible. (This is the **workstation** env — the robot
env is separate; see [`i2rt/ros2/README.md §1`](../../i2rt/ros2/README.md).)

```bash
sudo apt install -y ffmpeg                          # LeRobot v3.0 video encoding

uv venv --python /usr/bin/python3.10 --system-site-packages ~/lerobot_env
source ~/lerobot_env/bin/activate
echo 'source /opt/ros/humble/setup.bash' >> ~/lerobot_env/bin/activate  # auto-source ROS on activate
source /opt/ros/humble/setup.bash                   # for this shell, now
uv pip install -r workstation/lerobot_recorder/requirements.txt

python -c "import rclpy, pyrealsense2, lerobot; print('ready')"
```

The `yam-data` launcher activates `~/lerobot_env` + sources ROS for you; override
with `LEROBOT_ENV=/path` / `ROS_SETUP=/path/setup.bash` if your paths differ.

### [workstation] — RealSense cameras

The pip `pyrealsense2` wheel ships the SDK bindings but **not** the udev rules, so
without them the cameras open only as root / fail with permission errors. Install
the rules once:

```bash
# Minimal: just the udev permission rules from the librealsense repo.
git clone --depth 1 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense
sudo cp /tmp/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# (optional) realsense-viewer for live preview — needs Intel's apt repo:
#   sudo apt install -y librealsense2-utils
```

Plug the cameras into **USB 3** ports (USB 2 can't sustain 60 fps), then map each
**serial → role**. Each camera's serial is permanent firmware ID — no udev
renaming needed (unlike CAN), you just record which serial is which physical view:

```bash
workstation/yam-data cams        # lists connected RealSense name + serial
```

To find which serial is which view, plug them in **one at a time** and re-run
`cams` (or use `realsense-viewer` to look at the stream). Then pass them in the
fixed order `wrist_left,wrist_right,agentview` at launch (`--serials A,B,C`), or
hard-code them in [`config.py`](config.py) `default_cameras()`.

```bash
echo 'export ROS_DOMAIN_ID=64' >> ~/lerobot_env/bin/activate   # bake in (MUST match the robot)
export ROS_DOMAIN_ID=64          # for the current shell, now
```

> Both machines must share `ROS_DOMAIN_ID` (64 here) and be on the same network
> so DDS can discover the topics. Baking it into the activate script (as above,
> and in the robot's `.venv-ros`) guarantees they never drift.

---

# Runbook — scenarios (run top to bottom)

Commands are tagged **[robot]** (the YAM machine) or **[workstation]**. The
`scripts/yam` and `workstation/yam-data` launchers activate the right env for you.

## A. Bimanual teleop only (no recording)

```bash
# [robot]
scripts/yam canup                 # bring up the 4 CAN interfaces (after boot)
scripts/yam teleop --bilateral-kp 0.15
# lift both gellos to engage; bring both home to stop & auto-return.
```

## B. Data collection (teleop + LeRobot recorder)  ← main flow

```bash
# 1. [robot]   start teleop (publishes state / applied_action / teleop signals)
scripts/yam canup
scripts/yam teleop --bilateral-kp 0.15

# 2. [workstation]   start the recorder GUI
workstation/yam-data record \
    --repo-id user/yam_pick --root ~/lerobot_data \
    --serials <wrist_left_sn>,<wrist_right_sn>,<agentview_sn>
```

Then, in the recorder GUI:

1. Confirm `repo_id` / `root` / `task` → **Start** (opens cameras + ROS + dataset).
2. **Start collection** (arms the gate).
3. Teleoperate: **lift both gellos** → an episode records; **bring both home** →
   it ends and the **review panel** plays it back.
4. **Keep** (save) or **Delete** (discard), then repeat from step 3.
5. **Stop collection** when done, then close the window (this calls `finalize()`
   so the dataset is complete).

Quick dry run with no robot/cameras/lerobot:

```bash
# [workstation]
workstation/yam-data record --mock
```

## C. DAgger (policy + human takeover)

```bash
# [robot]
scripts/yam canup
scripts/yam dagger --bilateral-kp 0.15
# your policy (anywhere on the ROS graph) publishes /<arm>/policy_action;
# press a handle button (or /dagger/intervention_cmd) to take over both arms.
```

(Recording DAgger interventions can reuse the same recorder by subscribing to the
DAgger streams — ask if you want a DAgger-specific recorder preset.)

## D. Replay a dataset onto the robot

```bash
# 1. [robot]   run the wrapper so /<arm>/command drives the followers
scripts/yam canup
scripts/yam wrapper --arm left:can_follower_l --arm right:can_follower_r

# 2. [workstation]   open the replay GUI
workstation/yam-data replay --repo-id user/yam_pick --root ~/lerobot_data
```

In the replay GUI: **Load** → pick an **episode** → tick **Send to robot** →
**Play**. It first ramps the robot from its current pose to the first frame (no
jump), then streams each frame's `action` to `/<arm>/command`. Untick "Send to
robot" to just preview the video. **Pause** / **Stop** / **speed** as needed.

Dry run:

```bash
# [workstation]
workstation/yam-data replay --mock
```

---

## Notes

- **finalize**: closing the recorder window (or `recorder.shutdown()`) calls
  `LeRobotDataset.finalize()`. Skipping it leaves parquet files incomplete.
- The record loop is clocked at 60 fps; cameras stream at 60 fps and each tick
  grabs the latest frame plus the latest robot state/action from ROS.
- If a D405 doesn't support 60 fps at 640×480, lower the resolution/fps in
  `config.py` (`CameraSpec`); the record loop tolerates it (grab-latest).
- `lerobot`'s API can shift between releases — the version-sensitive calls are
  isolated in `dataset_writer.py` and `dataset_reader.py`.
- Nothing here imports `i2rt`; it only relies on the ROS topic contract in
  `i2rt/ros2/README.md`.
