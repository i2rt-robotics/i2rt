# I2RT ROS 2 Integration

A **build-less** ROS 2 layer for I2RT YAM arms. It uses only standard message
types (`sensor_msgs` / `std_msgs`), so there is **no colcon/ament package to
build** — just activate the environment and run the modules.

It assumes the usual setup: **this machine is the robot side** (arms on CAN), and
a **main workstation** connects over ROS 2 to read state and send actions.
Everything here is **bimanual by default** (2 leaders + 2 followers).

Three components:

| # | Module | What it does |
|---|--------|--------------|
| ① | `i2rt.ros2.run_wrapper` | Wrap follower arm(s): publish state, subscribe to joint targets |
| ② | `i2rt.ros2.run_teleop`  | Bimanual leader→follower teleop + publish all streams |
| ③ | `i2rt.ros2.run_dagger`  | HG-DAgger: policy drives followers; a button hands control to the human |

---

## 1. Environment

A dedicated conda env (`i2rt_ros`, Python 3.10) holds i2rt **and** can import
ROS 2 Humble's `rclpy`. Activating it auto-sources `/opt/ros/humble`.

```bash
conda activate i2rt_ros          # sources ROS 2 Humble automatically
python -c "import rclpy, i2rt; print('ready')"
```

<details>
<summary>How this env was created (one-time, for reference)</summary>

```bash
conda create -y -n i2rt_ros -c conda-forge --override-channels python=3.10
conda activate i2rt_ros
python -m ensurepip --upgrade
pip install -e /home/droid/i2rt_rllab
# auto-source ROS 2 on activate:
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'source /opt/ros/humble/setup.bash' > "$CONDA_PREFIX/etc/conda/activate.d/ros2_humble.sh"
```

The conda Python (3.10) is ABI-compatible with Humble's system Python (3.10), so
the prebuilt `rclpy` loads directly — no RoboStack needed.
</details>

All commands below assume `conda activate i2rt_ros`. Add `--sim` to any of them to
run without hardware (uses `SimRobot`).

### Shortcuts

[`scripts/yam`](../../scripts/yam) activates the env for you and launches a node,
so you don't type the full `conda activate … && python -m …`:

```bash
scripts/yam teleop  --sim                 # ② teleop
scripts/yam dagger  --mirror-kp 0.2 --feedback-kp 0.1   # ③ dagger
scripts/yam wrapper --arm left:can_follower_l   # ① wrapper
scripts/yam can                           # interactive CAN-id setup
scripts/yam canup                         # reset/bring up CAN at 1 Mbit/s
```

For even shorter commands, add the aliases once to `~/.bashrc`:

```bash
echo "source $(pwd)/scripts/ros2_aliases.sh" >> ~/.bashrc && source ~/.bashrc
# then, from anywhere:
yam teleop --sim        # or: yam-teleop --sim
yam-dagger --mirror-kp 0.2 --feedback-kp 0.1
```

(Override the env with `I2RT_ENV=my_env scripts/yam …`.)

---

## 2. Hardware / CAN setup (real bimanual)

The teleop (②) and DAgger (③) nodes are **bimanual by default** and expect four
USB-CAN adapters with these fixed interface names (see
`teleop_common.default_bimanual_specs`):

| side  | leader channel  | follower channel  | arm | leader gripper        | follower gripper |
|-------|-----------------|-------------------|-----|-----------------------|------------------|
| left  | `can_leader_l`  | `can_follower_l`  | yam | `yam_teaching_handle` | `linear_4310`    |
| right | `can_leader_r`  | `can_follower_r`  | yam | `yam_teaching_handle` | `linear_4310`    |

YAM uses candle/candlelight USB-CAN adapters, which appear as **network
interfaces** (`canX`), not `/dev/ttyUSB`. So there is no serial baud to set —
instead you must (1) pin persistent names, (2) bring the interfaces up at
1 Mbit/s, and (3) verify.

### ① Persistent interface names (required for 4 arms)

Plugging four adapters in gives non-deterministic `can0..can3` (plug order), so
the arms would swap randomly. Pin names by serial via udev.

**Easiest — the interactive helper** ([`scripts/setup_can_ids.sh`](../../scripts/setup_can_ids.sh)):
it asks you to plug the four adapters **in one at a time**, auto-detects each
one's serial, writes `/etc/udev/rules.d/90-can.rules`, reloads udev, and brings
the interfaces up:

```bash
sh scripts/setup_can_ids.sh
```

<details>
<summary>Manual alternative</summary>

Full guide: [`docs/guides/set-persistent-can-ids.md`](../../docs/guides/set-persistent-can-ids.md).

```bash
# find each adapter's serial
udevadm info -a -p /sys/class/net/can0 | grep -i serial

# /etc/udev/rules.d/90-can.rules  (names <=13 chars, must start with "can")
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<serial1>", NAME="can_leader_l"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<serial2>", NAME="can_follower_l"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<serial3>", NAME="can_leader_r"
SUBSYSTEM=="net", ACTION=="add", ATTRS{serial}=="<serial4>", NAME="can_follower_r"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```
</details>

> Note: `ip link ... up` does **not** persist across reboot. Re-run the bring-up
> loop below (or `sh scripts/reset_all_can.sh`) after each boot.

### ② Bring interfaces up at 1 Mbit/s

```bash
for c in can_leader_l can_follower_l can_leader_r can_follower_r; do
    sudo ip link set "$c" up type can bitrate 1000000
done
# or, if an adapter is unresponsive:
sh scripts/reset_all_can.sh
```

If you hit permission/driver issues, install the udev rules and group
membership once with `sudo sh devices/install_devices.sh` (the `gs_usb` driver
is normally built into the kernel and auto-loads).

### ③ Verify before running

```bash
ip link show | grep can                                      # all four UP?
python i2rt/motor_config_tool/ping_motors.py --channel can_follower_l   # follower motors
python scripts/read_encoder.py --channel can_leader_l         # leader handle trigger/buttons
```

### Recommended bring-up order

1. Pin the four names via udev, confirm `ip link` shows all four **UP**.
2. `ping_motors` each follower; `read_encoder` each leader handle.
3. Smoke-test one follower through the wrapper:
   `python -m i2rt.ros2.run_wrapper --arm test:can_follower_l`
4. Then run the full teleop: `python -m i2rt.ros2.run_teleop --bilateral-kp 0.15`,
   and press each leader handle's top button to engage sync.

> If your channel names or grippers differ from the table above, either rename
> the interfaces to match, or ask for `--channels`/`--gripper` overrides to be
> added to `run_teleop`/`run_dagger` (currently those defaults are hard-coded).

---

## ① ROS 2 Wrapper — `run_wrapper`

Wraps each **follower** arm as a node: publishes `JointState`, accepts joint
targets.

```bash
# Two simulated followers (no hardware):
python -m i2rt.ros2.run_wrapper --sim

# Real bimanual hardware:
python -m i2rt.ros2.run_wrapper \
    --arm left:can_follower_l --arm right:can_follower_r --gripper linear_4310

# Single arm:
python -m i2rt.ros2.run_wrapper --arm follower:can0
```

Topics per arm `<name>`:

| Dir | Topic | Type | Notes |
|-----|-------|------|-------|
| pub | `<name>/joint_states` | `sensor_msgs/JointState` | `[joint_1..6, gripper]`, pos/vel/eff |
| sub | `<name>/command`      | `sensor_msgs/JointState` | target positions → `command_joint_pos` |
| pub | `<name>/buttons`      | `sensor_msgs/Joy` | only if a teaching handle is present |

The command may carry the full vector (`num_dofs`) or arm-only (`num_dofs-1`, the
current gripper is kept). The gripper value is normalized **0 = closed, 1 = open**.

```bash
# from the workstation: drive the left follower
ros2 topic pub /left/command sensor_msgs/msg/JointState \
  "{name: [joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,gripper],
    position: [0.3,0,0,0,0,0,0.5]}"
```

---

## ② Teleop — `run_teleop`

Activates both leader+follower pairs and drives them with a **single global
auto-gate** (no per-arm buttons — teleop on/off is always both arms together),
publishing every stream for demonstration recording.

```bash
python -m i2rt.ros2.run_teleop --sim
python -m i2rt.ros2.run_teleop --bilateral-kp 0.2 --home 0,0,0,0,0,0   # real hardware
```

### How the gate works (state machine)

| State | Behavior | Leaves when |
|-------|----------|-------------|
| `HOMING`  | robot **and** leaders ramp smoothly to the home pose | ramp reaches home |
| `IDLE`    | sitting at home, leaders free (gravity-comp) | **both** leaders lifted past `--engage-thr` |
| `ENGAGED` | followers track leaders (rate-limited, no jump) | **both** leaders back within `--release-thr` of home for `--dwell` s → `HOMING` |

So you just **lift both gellos to start** teleop, and **bring both back to home to
stop** (the robot and gellos then return home on their own). Hysteresis
(`engage-thr > release-thr`) prevents chattering.

The gate compares the leader to the home pose. **By default it keys on the 2nd
joint alone** (`GATE_JOINTS = [1]`): lift the 2nd joint past `--engage-thr` to
start, lower it back within `--release-thr` to return — regardless of the other
joints (so homing isn't blocked by a stray joint angle). Use `--gate-joints` to
pick other joint(s), or `--gate-joints ""` for the L2 distance over all joints.
The gripper is ignored.

**Steady tracking is direct.** While ENGAGED the follower is commanded straight
to the leader pose (`command_joint_pos`) and tracked by its **own PD gains** —
exactly like the original `minimum_gello` (there is no rate limit on tracking, so
it does not lag). Only the *one-time engage approach* and the *homing return* are
ramped, by `--ramp-speed`.

**Two kinds of arguments — don't confuse them:**

| Arg | Unit | What it controls |
|-----|------|------------------|
| `--ramp-speed` | rad/s | speed of the **one-time engage approach + homing return** only — *lower = smoother/slower return* (default 0.8). Does **not** affect steady tracking. |
| `--engage-thr` / `--release-thr` | rad | how far a leader must move from / return to home to engage / disengage (default 0.6 / 0.3) |
| `--gate-joints` | indices | gate on specific joint(s), e.g. `1` = 2nd joint only; empty = L2 over all |
| `--dwell` | s | release must hold this long before homing (default 0.5) |
| `--home-kp` | gain | **leader stiffness** while homing — pulls the *leader* (gello) back to home (default 0.3) |
| `--bilateral-kp` | gain | **leader stiffness** while engaged — back-drives the *leader* for force feel (0 = free; 0.1–0.2 = light) |
| `--home` | rad | the home pose arm joints (default zeros) |

> **speed** = how fast a *transition ramp* moves (engage approach / homing).
> **kp** = a *stiffness gain* on the *leader* (the gello the human holds) — nothing
> to do with ramp speed or follower tracking.

### Global follow gain — same in teleop, DAgger, and replay

The **follower's tracking gains** (kp/kd) are the single most important thing to
keep identical across collection and replay. They live in one place,
[`i2rt/ros2/control_config.py`](control_config.py): by default they are the arm's
`yam.yml` gains (already global), and `apply_follower_gains()` is called wherever a
follower is built — `run_teleop`, `run_dagger`, and the `run_wrapper` used by
replay — so all three behave the same. To change them globally, set
`FOLLOWER_KP` / `FOLLOWER_KD` there (scalar or per-joint); leave `None` to keep
`yam.yml`. The gate/ramp defaults (`ENGAGE_THR`, `RAMP_SPEED`, `GATE_JOINTS`, …)
live there too, so teleop and DAgger share one source of truth.

### Reproducible logging

The **exact rate-limited target sent to the robot** is published on
`<s>/applied_action` every tick — log that (not the raw leader) so a policy
trained on it reproduces the episode precisely. `/teleop/state` and
`/teleop/active` tell the logger which phase each sample belongs to.

Topics per side `<s>` ∈ {left, right}:

| Dir | Topic | Type |
|-----|-------|------|
| pub | `<s>/leader/joint_states`   | `sensor_msgs/JointState` |
| pub | `<s>/follower/joint_states` | `sensor_msgs/JointState` |
| pub | `<s>/applied_action`        | `sensor_msgs/JointState` (smoothed command to the robot) |
| pub | `<s>/buttons`               | `sensor_msgs/Joy` |

Global:

| Dir | Topic | Type |
|-----|-------|------|
| pub | `/teleop/state`     | `std_msgs/String` (HOMING / IDLE / ENGAGED) |
| pub | `/teleop/active`    | `std_msgs/Bool` (True iff ENGAGED) |
| sub | `/teleop/sim_engage`| `std_msgs/Bool` (debug: force ENGAGED in `--sim`) |

> DAgger's gate is likewise **global** — one button (or `/dagger/intervention_cmd`)
> takes over both arms at once.

---

## ③ DAgger — `run_dagger`

**HG-DAgger** interactive takeover, **bimanual, single gate**. The gate is
**hold-to-engage** (level, not a toggle): while the human **holds** a handle
button the human drives; **releasing** hands control back to the policy.

- **Normal (policy):** the workstation **policy** publishes `<s>/policy_action`;
  this node applies it to the followers and **back-drives the leaders** (at the
  **mirror** gain) so a human resting on the handles feels what the policy intends.
- **Intervention (button held):** the gate (either handle's top button, or
  `/dagger/intervention_cmd`) switches **both** arms to human control — each
  follower tracks its leader, leader back-driven at the **feedback** gain — and
  streams `<s>/human_action` plus `/dagger/intervention=true` so the workstation
  can aggregate `(obs, action)`.
- **Release:** control returns to the policy.

The two leader gains are **separate** and live in `control_config.py`:

| Arg / config | Phase | Default |
|---|---|---|
| `--mirror-kp` / `DAGGER_MIRROR_KP` | policy driving (leader mirrors policy) | 0.2 |
| `--feedback-kp` / `DAGGER_FEEDBACK_KP` | human intervening (force feel) | 0.1 |

```bash
python -m i2rt.ros2.run_dagger --sim
python -m i2rt.ros2.run_dagger --mirror-kp 0.2 --feedback-kp 0.1   # real hardware
```

Topics:

| Dir | Topic | Type | Notes |
|-----|-------|------|-------|
| sub | `<s>/policy_action`        | `sensor_msgs/JointState` | from the policy |
| pub | `<s>/follower/joint_states`| `sensor_msgs/JointState` | |
| pub | `<s>/leader/joint_states`  | `sensor_msgs/JointState` | |
| pub | `<s>/applied_action`       | `sensor_msgs/JointState` | what was actually sent |
| pub | `<s>/human_action`         | `sensor_msgs/JointState` | leader q; valid while intervening |
| pub | `<s>/buttons`              | `sensor_msgs/Joy` | |
| pub | `/dagger/intervention`     | `std_msgs/Bool` | gate state (both arms) |
| sub | `/dagger/intervention_cmd` | `std_msgs/Bool` | external gate (sim / no handle) |

**Workstation side** (sketch): subscribe to `/<s>/follower/joint_states` for
observations, publish `/<s>/policy_action` at your policy rate (e.g. 10–30 Hz),
and log a transition into the DAgger buffer whenever `/dagger/intervention` is
`true`, using `/<s>/human_action` as the label.

---

## Notes

- The real-time 250 Hz motor loop lives inside `MotorChainRobot`; these nodes only
  sample state and forward targets, so a modest ROS rate (50–120 Hz) is plenty.
- Bilateral back-driving needs the real `MotorChainRobot` (`update_kp_kd`); in
  `--sim` it is skipped automatically.
- All nodes run in one process per component via an executor — no launch files,
  no build, no custom messages.
