# YAM ↔ LeRobot Recorder & Replay

Record and replay [LeRobot](https://github.com/huggingface/lerobot) **v3.0**
datasets for the bimanual YAM teleop rig. Runs on the **workstation** (a
different machine / env than the robot): it connects to the YAM **robot server
over portal** (plain TCP), reads three RealSense cameras **locally**,
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

`config.yaml` tunes the flow: **`auto_arm`** arms on START (skip "Start collection"),
**`review_before_save: false`** auto-saves each episode (skip Keep/Delete), and a
leader **button** can end+label in one press (see *Labeling*). The engage/release
thresholds (`control.engage_thr` / `release_thr` / `dwell` / `gate_joints`) live in
`config.yaml` too and are applied by the robot server.

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

### [robot machine] — YAM robot server (uv; nothing to activate)

```bash
sh scripts/setup_robot_env.sh            # optional: pre-create .venv + install i2rt
sh scripts/setup_can_ids.sh              # persistent CAN names (once)
```

You don't need to activate anything — `scripts/yam` uses **`uv run`**, which resolves
(and on first run creates) the env automatically. Already inside a conda/venv? set
`YAM_NO_UV=1` and it uses plain `python`.

### [workstation] — conda env + uv

conda owns the env (so you can also `pip install` other policy repos into it); uv
does the fast installs for this repo:

```bash
sh scripts/setup_workstation_env.sh       # conda create yam_ws + uv pip install + udev rules
conda activate yam_ws
```

<details><summary>What it does (manual equivalent)</summary>

```bash
conda create -y -n yam_ws python=3.11      # any Python >= 3.10
conda activate yam_ws
sudo apt install -y ffmpeg                  # LeRobot v3.0 video encoding
uv pip install -e .                         # i2rt (portal RobotClient) — uv targets the conda env
uv pip install -e policy_serving            # yam-policy (websocket client for the bridge)
uv pip install -r workstation/lerobot_recorder/requirements.txt
# another policy repo in the SAME env:  pip install -e /path/to/policy_repo   (or uv pip install)
```
</details>

The `yam-data` launcher activates the **conda** env for you (default `yam_ws`,
override with `YAM_WS_ENV=...`). The robot host/port come from `config.yaml` (or
`--robot-host`/`--robot-port`, default
`127.0.0.1:11331`) — both machines just need to be on the same network.

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

The recorder opens on a **Setup page**:

1. Confirm `repo_id` / `root` / `task` and the **source** (teleop / dagger / eval).
   The dataset is written to **`<root>/<name>`** (name = last segment of `repo_id`,
   e.g. `~/lerobot_data` + `hello/pick_and_place` → `~/lerobot_data/pick_and_place`).
   The status line shows cameras detected and whether that dataset already exists.
2. Tick **Continue collecting** to resume/append; otherwise **START** creates it
   fresh (and asks **twice** before overwriting an existing folder).
3. **START** connects the robot, opens cameras + dataset, and — with `auto_arm` —
   arms collection immediately.

Then teleoperate: **lift both gellos** → records; **bring both home** → episode ends.
With `review_before_save: true` it's held in the **review panel** for **Keep** (S/F) /
**Delete** (D); with `review_before_save: false` it **auto-saves** each engage→idle.
A **leader handle button** ends + labels in one press (see *Labeling* below). Close
the window when done — this calls `finalize()` so the dataset is complete.

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
- The record loop is clocked at 60 fps. Cameras grab on their **own capture thread**
  and cache the latest frame; the loop and GUI read that cache **non-blocking**, so a
  slow frame or a pipe re-open never stalls recording or freezes the view. Each tick
  pairs the latest cached frame with the latest robot state/action polled over portal.
- **Camera fps fallback**: the requested fps (60) is auto-reduced to the highest the
  device supports (e.g. 30 on USB 2.0) — no config edit needed. Over **USB 2.0** a
  640×480 stream caps at 30 fps and can drop frames under 3-camera load, so use the
  **USB 3 cables** for true 60 fps (a USB-2 cable downgrades the link even on a USB-3 port).
- **Single instance**: the recorder takes a lock so a second instance can't fight
  over the cameras; starting a second one reports a clear error instead of flapping.
- `lerobot`'s API can shift between releases — the version-sensitive calls are
  isolated in `dataset_writer.py` and `dataset_reader.py`.
- **Dataset location**: `root` is a **parent dir**; the dataset lives at
  `<root>/<name>` (name = last segment of `repo_id`), so several datasets can sit
  side by side under one `root`. The reader/replay resolve the same path.
- **Outcome labels**: **Keep (success)** / **Keep (fail)** tag each episode; the
  label + task + frame count are appended to `outcomes.jsonl` **inside the dataset
  folder** (a sidecar, since LeRobot has no per-episode label slot).
- **Resume**: tick **Continue collecting** in the GUI (or `--resume`) to append to
  the existing dataset at `<root>/<name>` instead of creating a new one (episode
  indices continue).
- **Doctor**: `workstation/yam-data doctor --root ~/lerobot_data [--repo-id ...]`
  prints episode counts, success rate, and per-task stats from `outcomes.jsonl`
  (and validates the LeRobot dataset if `--repo-id` is given). The replay episode
  list is annotated with ✓/✗ from the same sidecar.
- **Safety**: E-STOP button in both GUIs (holds the followers); optional collision
  soft-stop (`control_config.FOLLOWER_EFFORT_LIMIT`); disk-space guard refuses to
  save below `min_free_gb`.
- **Always-on provenance (fixed schema)**: every frame carries
  `observation.state(42)`, `observation.leader(12)`, `observation.eef(14)` (FK from
  the company `Kinematics`; zeros if no model), `observation.control_mode(1)`
  (teleop/policy/intervention), and `action(14)`. The schema is **predefined from the
  robot's known outputs** (no runtime probe).
- **Async writer**: a finished episode is queued and saved by a background worker
  (one at a time), so LeRobot's per-trajectory encoding never blocks the next
  collection. The GUI shows the pending `queue` depth.
- **Labeling**: in the review panel use the mouse, **keyboard** ([S] keep success,
  [F] keep fail, [D] delete, [space] toggle collection), or the **leader handle
  buttons**. The button→outcome map is **per-(side, index)** and configurable in
  `config.yaml` under `recorder.buttons` (keyed `<side>.<index>`, upper=0/lower=1).
  Default: **left lower = success, right lower = fail, either upper = discard** — so
  all three outcomes are reachable with two buttons per arm. A label button also
  starts homing, so one press ends + labels + saves (records through homing).
- **Operator UI**: a big color status **banner** (IDLE/ARMED/REC/REVIEW/fault), a
  **health strip** (robot link · cameras · save queue), **live stats** (kept ✓/✗,
  discarded, success rate), **audio cues** (start, keep/fail/delete, fault), and a
  review **scrubber** — so you can collect while watching the robot.
- **Task templates**: `--tasks "pick the cube; stack the blocks; open the drawer"`
  gives a quick-switch dropdown; the active task **persists until you change it**
  (editable — type a new one on the fly).
- **Eval rollouts**: `--source eval` records a continuous policy rollout from
  Start to Stop (action = the executed command, labeled policy/intervention) — for
  saving evaluation episodes as datasets.
- **Camera fault tolerance**: a faulted RealSense shows a red ⚠ warning, recording
  pauses (no garbage frames), and the capture thread auto-reconnects in the
  background (logging only the down/recovered transitions, not every retry).
- **Replay overlay**: tick **Overlay live** to blend an episode's first frame with
  the live agentview, so you can place objects to match the dataset before Play.
- The robot link is the snapshot contract in
  [`i2rt/serving/README.md`](../../i2rt/serving/README.md); the policy link is in
  [`policy_serving/README.md`](../../policy_serving/README.md).
