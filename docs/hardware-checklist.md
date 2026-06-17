# Hardware verification checklist

Things to confirm **on the real robot + cameras** (sim/mock + unit tests already
pass in CI). Go top-to-bottom; each item has how-to-run and what to look for.
Tags: **[robot]** = YAM machine, **[ws]** = workstation, **[policy]** = policy host.

Transports: robot↔workstation = **portal** (TCP, default `:11331`); workstation↔policy
= **websocket** (default `:8000`). No ROS anywhere.

---

## 0. Environments (one-time)

```bash
# [robot]
sh scripts/setup_robot_env.sh && source .venv/bin/activate
# [ws]
sh scripts/setup_workstation_env.sh && source ~/yam_ws/bin/activate
# [policy]  (any host/GPU)
sh policy_serving/setup_policy_env.sh && source policy_serving/.venv/bin/activate
```
- [ ] All three envs import cleanly (`python -c "import i2rt"`, `import yam_policy`, `import lerobot, pyrealsense2`).

## 1. CAN + cameras

```bash
# [robot]
sh scripts/setup_can_ids.sh           # persistent names (once)
scripts/yam canup                      # bring up at 1 Mbit/s (each boot)
# [ws]
workstation/yam-data cams              # list RealSense serials
```
- [ ] `ip link` shows `can_leader_l/r`, `can_follower_l/r` UP.
- [ ] `cams` lists D455 (agentview) + 2× D405 (wrist). Fill serials in `--serials A,B,C` (order: wrist_left,wrist_right,agentview) or `config.py`.

---

## 2. Robot server + workstation link

```bash
# [robot]
scripts/yam teleop --bilateral-kp 0.0
# [ws]
python -c "from i2rt.serving.robot_client import RobotClient; c=RobotClient(host='<ROBOT_IP>'); print(c.metadata); print(sorted(c.get_observation()))"
```
- [ ] Client prints metadata `{mode: teleop, sides:[left,right], ...}` and a snapshot with `left/right`, `teleop_state`, `control_mode`.

## 3. Teleop gate (auto engage/home)

```bash
# [robot]  scripts/yam teleop --bilateral-kp 0.0
```
- [ ] Lift both leaders → state goes **ENGAGED**, followers track.
- [ ] Return leaders home → **HOMING → IDLE**, robot eases home.

## 4. ⚠ Bilateral engage — NO leader yank (the bug fix)

```bash
# [robot]  scripts/yam teleop --bilateral-kp 0.15
```
- [ ] With followers at home and **leaders lifted**, press engage: the **leader is NOT yanked toward home**. Leader stays free until the follower catches up, then you feel bilateral force. (This was the bug.)

## 5. Gentle homing speed

```bash
# [robot]  scripts/yam teleop --bilateral-kp 0.15 --home-speed 0.4
```
- [ ] Homing return is smooth/slow (raise/lower `--home-speed` to taste; engage approach uses `--ramp-speed 0.8`).

## 6. Leader-button end+label+home

- [ ] While ENGAGED, press leader **button 1** → robot starts homing (gently), records through homing, and on IDLE the episode is saved **success**.
- [ ] **button 2** → saved **fail**.  **button 0** → **discarded** (not saved).
  (If your handle's button indices differ, adjust `HOME_BUTTONS` in `i2rt/serving/control_config.py` and `SUCCESS/FAIL/DISCARD_BUTTON` in `workstation/lerobot_recorder/recorder.py`.)

---

## 7. Data collection (recorder)

```bash
# [robot]  scripts/yam teleop --bilateral-kp 0.15
# [ws]
workstation/yam-data record --robot-host <ROBOT_IP> \
    --repo-id user/yam_pick --root ~/lerobot_data \
    --serials <wl>,<wr>,<agent> --task "pick the cube"
```
In the GUI:
- [ ] Live view shows **agentview with both wrist views inset** in the bottom corners.
- [ ] **Start collection**, teleop an episode → review panel plays it back.
- [ ] Label by **mouse**, **keyboard** (`S` success / `F` fail / `D` delete), or leader buttons.
- [ ] Status shows `queue:` depth; collecting the next episode works **while the previous is still saving** (async writer — confirm `queue` rises then drains, no stall).
- [ ] Stop + close window (calls `finalize()`).

Verify the saved dataset schema (fixed, no probe):
```bash
# [ws]
python - <<'PY'
from lerobot.datasets import LeRobotDataset
ds = LeRobotDataset("user/yam_pick", root="~/lerobot_data")
print(ds.features.keys())
PY
cat ~/lerobot_data/outcomes.jsonl
```
- [ ] Features include `observation.state(42)`, `observation.leader(12)`, `observation.control_mode(1)`, `action(14)`, and the 3 images.
- [ ] `outcomes.jsonl` has one line per kept episode with `outcome`/`task`/`frames`/`source`.

## 8. Camera disconnect fallback

- [ ] While recording, unplug one RealSense → GUI shows red **⚠ CAMERA DISCONNECTED**, recording pauses (no garbage frames). Replug → it auto-reconnects and resumes.

## 9. Safety: watchdog + e-stop

```bash
# [robot]  scripts/yam wrapper
# [ws]
python -c "from i2rt.serving.robot_client import RobotClient; import numpy as np,time; c=RobotClient(host='<ROBOT_IP>'); c.command({'left':np.zeros(7),'right':np.zeros(7)}); print('sent')"
```
- [ ] After the client exits (no more commands), the follower **holds** within ~0.5 s instead of replaying the last target (watchdog). Tune `command_timeout`.
- [ ] `RobotClient(...).set_estop(True)` → robot stops commanding (holds); `set_estop(False)` resumes. Snapshot `estop` reflects it.
- [ ] (Optional) set `control_config.FOLLOWER_JOINT_LIMITS` and confirm targets are clamped.

---

## 10. Policy deployment (bridge)

```bash
# [policy]  smoke test with the zero-model "hold" policy:
python -m yam_policy.serve            # :8000
# [robot]  scripts/yam dagger --mirror-kp 0.2
# [ws]
workstation/yam-data bridge --robot-host <ROBOT_IP> --policy-host <POLICY_IP> \
    --serials <wl>,<wr>,<agent> --prompt "do the task"
```
- [ ] Bridge logs the policy **server metadata** and auto-configures `action_horizon`/image keys (no hand-matching).
- [ ] Policy drives the followers; pressing a handle button hands control to the human (intervention), releasing returns to policy.
- [ ] Kill the bridge → robot holds within `command_timeout` (link-loss watchdog).

## 11. HG-DAgger collection

```bash
# [robot]  scripts/yam dagger --mirror-kp 0.2   (+ policy server + bridge as above)
# [ws]
workstation/yam-data record --source dagger --robot-host <ROBOT_IP> \
    --repo-id user/yam_dagger --serials <wl>,<wr>,<agent>
```
- [ ] An episode = one intervention segment; recorded `action` is the human (leader) action; `control_mode` = intervention. Verify in `outcomes.jsonl` (`source: dagger`).

## 12. Eval rollout recording

```bash
# [ws]  (policy + dagger + bridge running)
workstation/yam-data record --source eval --robot-host <ROBOT_IP> \
    --repo-id user/yam_eval --serials <wl>,<wr>,<agent>
```
- [ ] **Start collection** records the whole rollout (action = executed command, `control_mode` policy/intervention); **Stop collection** ends + saves it.

## 13. Replay + scene overlay

```bash
# [robot]  scripts/yam wrapper
# [ws]
workstation/yam-data replay --robot-host <ROBOT_IP> --repo-id user/yam_pick --root ~/lerobot_data
```
- [ ] **Load**, pick an episode, tick **Overlay live** → live agentview is blended with the episode's first frame; arrange objects to match.
- [ ] Tick **Send to robot** + **Play** → robot ramps to the first frame then follows the trajectory. Untick send = video-only preview.

## 14. EEF (note, not a blocker)

- The current `MotorChainRobot` exposes **no** end-effector pose, so `observation.eef`
  is **not** recorded and `--control eef` is a no-op (logs a warning). If you want
  EEF, the robot needs a `get_ee_pose()` (FK) and/or `command_ee_pose()` (IK) —
  ping me and I'll wire it to mink.
- [ ] (Optional) confirm `--control eef` logs "no command_ee_pose(); ignoring" and joint mode still works.
