# Hardware verification checklist

Things to confirm **on the real robot + cameras** (sim/mock + unit tests already
pass in CI). Go top-to-bottom; each item has how-to-run and what to look for.
Tags: **[robot]** = YAM machine, **[ws]** = workstation, **[policy]** = policy host.

Transports: robot↔workstation = **portal** (TCP, default `:11331`); workstation↔policy
= **websocket** (default `:8000`).

---

## 0. Environments (one-time)

```bash
# [robot]   uv-managed; robot/yam uses `uv run` (no activation needed)
sh robot/setup_robot_env.sh                       # optional pre-create + verify
# [ws]      conda env + uv installs
sh workstation/setup_workstation_env.sh && conda activate yam_ws
# [policy]  (any host/GPU)
sh policy_serving/setup_policy_env.sh && source policy_serving/.venv/bin/activate
```
- [ ] Workstation env imports cleanly (`python -c "import i2rt, yam_policy, lerobot, pyrealsense2"`).
- [ ] Robot: `robot/yam teleop --sim` boots (uv resolves the env on first run).
- [ ] Edit the tracked `config.yaml` at the **repo root** (serials + robot.host); every
      tool auto-finds it — no `--config`, no env var, regardless of directory.

## 1. CAN + cameras

```bash
# [robot]
sh robot/setup_can_ids.sh           # persistent names (once)
robot/yam canup                      # bring up at 1 Mbit/s (each boot)
# [ws]
workstation/yam-data cams              # list RealSense serials
```
- [ ] `ip link` shows `can_leader_l/r`, `can_follower_l/r` UP.
- [ ] `cams` lists D455 (agentview) + 2× D405 (wrist). Fill serials in `--serials A,B,C` (order: wrist_left,wrist_right,agentview) or `config.py`.

---

## 2. Robot server + workstation link

```bash
# [robot]
robot/yam teleop --bilateral-kp 0.0
# [ws]
python -c "from i2rt.serving.robot_client import RobotClient; c=RobotClient(host='<ROBOT_IP>'); print(c.metadata); print(sorted(c.get_observation()))"
```
- [ ] Client prints metadata `{mode: teleop, sides:[left,right], ...}` and a snapshot with `left/right`, `teleop_state`, `control_mode`.

## 3. Teleop gate (auto engage/home)

```bash
# [robot]  robot/yam teleop --bilateral-kp 0.0
```
- [ ] Lift both leaders → state goes **ENGAGED**, followers track.
- [ ] Return leaders home → **HOMING → IDLE**, robot eases home.

## 4. Bilateral engage — leader stays put

```bash
# [robot]  robot/yam teleop --bilateral-kp 0.15
```
- [ ] With followers at home and **leaders lifted**, press engage: the **leader is NOT yanked toward home**. Leader stays free until the follower catches up, then you feel bilateral force.

## 5. Gentle homing speed

```bash
# [robot]  robot/yam teleop --bilateral-kp 0.15 --home-speed 0.4
```
- [ ] Homing return is smooth/slow (raise/lower `--home-speed` to taste; engage approach uses `--ramp-speed 0.8`).

## 6. Leader-button end+label+home

- [ ] While ENGAGED, press leader **button 1** → robot starts homing (gently), records through homing, and on IDLE the episode is saved **success**.
- [ ] **button 2** → saved **fail**.  **button 0** → **discarded** (not saved).
  (If your handle's button indices differ, adjust `HOME_BUTTONS` in `i2rt/serving/control_config.py` and `SUCCESS/FAIL/DISCARD_BUTTON` in `workstation/lerobot_recorder/recorder.py`.)

---

## 7. Data collection (recorder)

```bash
# [robot]  robot/yam teleop --bilateral-kp 0.15
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
# [robot]  robot/yam wrapper
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
# [robot]  robot/yam dagger --mirror-kp 0.2
# [ws]
workstation/yam-data bridge --robot-host <ROBOT_IP> --policy-host <POLICY_IP> \
    --serials <wl>,<wr>,<agent> --prompt "do the task"
```
- [ ] Bridge logs the policy **server metadata** and auto-configures `action_horizon`/image keys (no hand-matching).
- [ ] Policy drives the followers; pressing a handle button hands control to the human (intervention), releasing returns to policy.
- [ ] Kill the bridge → robot holds within `command_timeout` (link-loss watchdog).

## 11. HG-DAgger collection

```bash
# [robot]  robot/yam dagger --mirror-kp 0.2   (+ policy server + bridge as above)
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
# [robot]  robot/yam wrapper
# [ws]
workstation/yam-data replay --robot-host <ROBOT_IP> --repo-id user/yam_pick --root ~/lerobot_data
```
- [ ] **Load**, pick an episode, tick **Overlay live** → live agentview is blended with the episode's first frame; arrange objects to match.
- [ ] Tick **Send to robot** + **Play** → robot ramps to the first frame then follows the trajectory. Untick send = video-only preview.

## 14. EEF observation + safe OSC control

EEF now uses the company `Kinematics` (mink) on `robot.xml_path`'s `grasp_site`.
Verified in sim (FK populates obs; IK round-trip holds). Confirm on hardware:

```bash
# [ws]  with a teleop/dagger server running, check the snapshot carries eef
python -c "from i2rt.serving.robot_client import RobotClient; c=RobotClient(host='<ROBOT_IP>'); print(c.get_observation()['left'].get('eef'))"
```
- [ ] `observation.eef` is a 7-vector `[x,y,z,qw,qx,qy,qz]` that **moves sensibly** as you move the arm (and is recorded as `observation.eef(14)` in datasets).
- [ ] **Safe OSC**: `robot/yam wrapper --control eef`, then from the workstation send an EE-pose target via `RobotClient.command({"left": pose7, "right": pose7})`. The arm should track the target smoothly (resolved-rate IK → joint impedance + smoother), **no jump near singularities**. Start with small offsets from the current pose (get it from the snapshot's `eef`).
- [ ] Confirm the IK site is correct for your gripper (default `grasp_site`; teaching handle uses `tcp_site`). If poses look off, set the site in `i2rt/serving/eef.py` (`ArmKinematics(..., site=...)`).
- Note: torque-level OSC (`Jᵀ·F`) is intentionally not the default (singularity safety); ask if you want it as an opt-in since the motors are in MIT/torque mode.
