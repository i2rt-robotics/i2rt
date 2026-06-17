# YAM ↔ LeRobot Recorder & Replay

Record and replay [LeRobot](https://github.com/huggingface/lerobot) **v3.0**
datasets for the bimanual YAM teleop rig. Runs on the **workstation** (a
different machine / env than the robot): it connects to the YAM **robot server
over portal** (plain TCP, **no ROS**), reads three RealSense cameras **locally**,
and **auto-starts/stops each episode from the teleop gate** — no manual record
button per episode.

Two tools (both have a PyQt GUI):

| Tool | Module | What it does |
|------|--------|--------------|
| **Recorder** | `workstation.lerobot_recorder` | teleop-gated LeRobot capture + review/delete |
| **Replay**   | `workstation.lerobot_recorder.replay_main` | play a dataset back onto the robot |

## How recording is triggered (the key idea)

`i2rt.serving.run_robot_server teleop` (on the robot machine) reports a
`teleop_state` (`HOMING` / `IDLE` / `ENGAGED`) in its snapshot. The recorder polls
it over portal:

```
 press "Start collection"      ──▶  gate armed
 both gellos lifted → ENGAGED  ──▶  episode STARTS recording   ◀── auto
 …teleoperate…       (ENGAGED)      every frame recorded
 gellos home → HOMING          ──▶  still recording (return included)
 homing done → IDLE            ──▶  episode ends → REVIEW       ◀── auto
   review playback → Keep / Delete
```

One episode = **ENGAGED → HOMING → IDLE**. The action stored is the robot's
**`applied`** (the rate-limited command actually sent) → reproducible.

## Dataset schema (LeRobot v3.0)

| Key | Shape | Source |
|-----|-------|--------|
| `observation.images.wrist_left`  | (H,W,3) uint8 | D405 left wrist |
| `observation.images.wrist_right` | (H,W,3) uint8 | D405 right wrist |
| `observation.images.agentview`   | (H,W,3) uint8 | D455 scene |
| `observation.state` | (42,) float32 | both arms × [pos(7), vel(7), eff(7)] |
| `action`            | (14,) float32 | both arms × `applied`(7) |
| task                | string | the language instruction |

Recorded at **60 fps** (matched to the cameras). Uses the official v3.0 API
(`create` / `add_frame` with a `task` key / `save_episode` / `clear_episode_buffer`
/ **`finalize`**); the version-sensitive calls live in `dataset_writer.py`.

---

# One-time setup

### [robot machine] — YAM robot server (no ROS)

See [`i2rt/serving/README.md`](../../i2rt/serving/README.md) /
[`scripts/setup_robot_env.sh`](../../scripts/setup_robot_env.sh). In short:

```bash
sh scripts/setup_robot_env.sh            # uv venv (.venv) + i2rt, no ROS
source .venv/bin/activate
sh scripts/setup_can_ids.sh              # persistent CAN names (once)
```

### [workstation] — Python env (uv, no ROS)

One script does it (`uv` venv at `~/yam_ws`, installs i2rt + yam-policy + recorder
deps, RealSense udev rules):

```bash
sh scripts/setup_workstation_env.sh
source ~/yam_ws/bin/activate
```

<details><summary>What it installs (manual equivalent)</summary>

```bash
sudo apt install -y ffmpeg                # LeRobot v3.0 video encoding
uv venv ~/yam_ws                          # any Python >= 3.10; NO ROS, NO system-site-packages
source ~/yam_ws/bin/activate
uv pip install -e .                       # i2rt (portal RobotClient)
uv pip install -e policy_serving          # yam-policy (websocket client for the bridge)
uv pip install -r workstation/lerobot_recorder/requirements.txt
python -c "import i2rt, yam_policy, lerobot, pyrealsense2; print('ready')"
```
</details>

The `yam-data` launcher activates `~/yam_ws` for you (override with `LEROBOT_ENV=/path`).
The robot host/port are passed at launch (`--robot-host`/`--robot-port`, default
`127.0.0.1:11331`) — both machines just need to be on the same network (plain TCP,
no `ROS_DOMAIN_ID`, no DDS).

### [workstation] — RealSense cameras

The pip `pyrealsense2` wheel ships the SDK bindings but **not** the udev rules, so
without them the cameras open only as root / fail with permission errors (the setup
script installs them; manual version):

```bash
git clone --depth 1 https://github.com/IntelRealSense/librealsense.git /tmp/librealsense
sudo cp /tmp/librealsense/config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
# (optional) realsense-viewer for live preview: sudo apt install -y librealsense2-utils
```

Plug the cameras into **USB 3** ports (USB 2 can't sustain 60 fps), then map each
**serial → role**. Each camera's serial is a permanent firmware ID — no udev
renaming needed (unlike CAN), you just record which serial is which physical view:

```bash
workstation/yam-data cams        # lists connected RealSense name + serial
```

To find which serial is which view, plug them in **one at a time** and re-run
`cams` (or use `realsense-viewer`). Then pass them in the fixed order
`wrist_left,wrist_right,agentview` at launch (`--serials A,B,C`), or hard-code them
in [`config.py`](config.py) `default_cameras()`.

---

# Runbook — scenarios (run top to bottom)

Commands are tagged **[robot]**, **[workstation]**, or **[policy]**. Replace
`<ROBOT_IP>` / `<POLICY_IP>` with the machine addresses (use `127.0.0.1` if a
component runs locally).

## A. Bimanual teleop only (no recording)

```bash
# [robot]
scripts/yam canup                 # bring up the 4 CAN interfaces (after boot)
scripts/yam teleop --bilateral-kp 0.15
# lift both gellos to engage; bring both home to stop & auto-return.
```

## B. Data collection (teleop + LeRobot recorder)  ← main flow

```bash
# 1. [robot]   start the teleop server (serves state/action/gate over portal)
scripts/yam canup
scripts/yam teleop --bilateral-kp 0.15

# 2. [workstation]   start the recorder GUI
workstation/yam-data record \
    --robot-host <ROBOT_IP> \
    --repo-id user/yam_pick --root ~/lerobot_data \
    --serials <wrist_left_sn>,<wrist_right_sn>,<agentview_sn>
```

Then, in the recorder GUI:

1. Confirm `repo_id` / `root` / `task` → **Start** (opens cameras + robot link + dataset).
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

## C. Deployment (policy + human takeover via DAgger)

```bash
# 1. [robot]    dagger server (policy drives followers; handle button = takeover)
scripts/yam canup
scripts/yam dagger --mirror-kp 0.2

# 2. [policy]   serve your policy (own env; openpi-compatible websocket)
#               see policy_serving/README.md
python -m yam_policy.serve --policy <module>:<Class> --config k=v     # :8000

# 3. [workstation]  the bridge: robot (portal) <-> policy (websocket)
workstation/yam-data bridge \
    --robot-host <ROBOT_IP> --policy-host <POLICY_IP> \
    --serials <wrist_left_sn>,<wrist_right_sn>,<agentview_sn> \
    --prompt "pick up the cube"
# press a handle button (or RobotClient.set_intervention) to take over both arms.
```

**Collect HG-DAgger data** by running the recorder against the same dagger server
with `--source dagger`: an episode = one intervention segment, and the recorded
action is the **human (leader) action** (`observation.state` is the follower state):

```bash
# [workstation]  (while the dagger server runs and you take over via the handle)
workstation/yam-data record --source dagger --robot-host <ROBOT_IP> \
    --repo-id user/yam_dagger --serials A,B,C
```

This closes the loop: train → deploy (bridge) → intervene → collect → retrain.

## D. Replay a dataset onto the robot

```bash
# 1. [robot]   wrapper server so the followers track an external command
scripts/yam canup
scripts/yam wrapper

# 2. [workstation]   open the replay GUI
workstation/yam-data replay --robot-host <ROBOT_IP> --repo-id user/yam_pick --root ~/lerobot_data
```

In the replay GUI: **Load** → pick an **episode** → tick **Send to robot** →
**Play**. It first ramps the robot from its current pose to the first frame (no
jump), then streams each frame's `action` to the robot via portal. Untick "Send to
robot" to just preview the video. **Pause** / **Stop** / **speed** as needed.

Dry run: `workstation/yam-data replay --mock`.

---

## Notes

- **finalize**: closing the recorder window (or `recorder.shutdown()`) calls
  `LeRobotDataset.finalize()`. Skipping it leaves parquet files incomplete.
- The record loop is clocked at 60 fps; cameras stream at 60 fps and each tick
  grabs the latest frame plus the latest robot state/action polled over portal.
- If a D405 doesn't support 60 fps at 640×480, lower the resolution/fps in
  `config.py` (`CameraSpec`); the record loop tolerates it (grab-latest).
- `lerobot`'s API can shift between releases — the version-sensitive calls are
  isolated in `dataset_writer.py` and `dataset_reader.py`.
- **Outcome labels**: in the recorder GUI, **Keep (success)** / **Keep (fail)** tag
  each episode; the label + task + frame count are appended to `outcomes.jsonl` in
  the dataset root (a sidecar, since LeRobot has no per-episode label slot).
- **Resume**: `--resume` appends to an existing dataset at `--root` instead of
  creating a new one (episode indices continue).
- **Always-on provenance (fixed schema)**: every frame carries
  `observation.state(42)`, `observation.leader(12)`, `observation.eef(14)` (FK from
  the company `Kinematics`; zeros if no model), `observation.control_mode(1)`
  (teleop/policy/intervention), and `action(14)`. The schema is **predefined from the
  robot's known outputs** (no runtime probe).
- **Async writer**: a finished episode is queued and saved by a background worker
  (one at a time), so LeRobot's per-trajectory encoding never blocks the next
  collection. The GUI shows the pending `queue` depth.
- **Labeling**: in the review panel use the mouse, **keyboard** ([S] keep success,
  [F] keep fail, [D] delete), or the **leader handle buttons** — button 1 = success,
  2 = fail, 0 = discard. A label button also starts homing on the robot, so one
  press ends + labels + saves the trajectory (records through the homing return).
- **Eval rollouts**: `--source eval` records a continuous policy rollout from
  Start to Stop (action = the executed command, labeled policy/intervention) — for
  saving evaluation episodes as datasets.
- **Camera fault tolerance**: a disconnected RealSense shows a red ⚠ warning,
  recording pauses (no garbage frames), and the manager auto-reconnects.
- **Replay overlay**: tick **Overlay live** to blend an episode's first frame with
  the live agentview, so you can place objects to match the dataset before Play.
- The robot link is the snapshot contract in
  [`i2rt/serving/README.md`](../../i2rt/serving/README.md); the policy link is in
  [`policy_serving/README.md`](../../policy_serving/README.md).
