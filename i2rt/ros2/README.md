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

Activates both leader+follower pairs and runs the bilateral teleop loop in
process, publishing every stream for demonstration recording.

```bash
python -m i2rt.ros2.run_teleop --sim
python -m i2rt.ros2.run_teleop --bilateral-kp 0.2      # real hardware
```

- The **handle top button** toggles sync (edge-triggered), exactly like
  `examples/minimum_gello`. While synced, the follower tracks the leader and the
  leader is back-driven (`--bilateral-kp`, 0.1–0.2 typical) so the human feels
  contact forces.
- Without a handle (e.g. sim), publish `Bool` to `<side>/sync_cmd` to force sync.

Topics per side `<s>` ∈ {left, right}:

| Dir | Topic | Type |
|-----|-------|------|
| pub | `<s>/leader/joint_states`   | `sensor_msgs/JointState` |
| pub | `<s>/follower/joint_states` | `sensor_msgs/JointState` |
| pub | `<s>/buttons`               | `sensor_msgs/Joy` |
| pub | `<s>/sync`                  | `std_msgs/Bool` |
| sub | `<s>/sync_cmd`              | `std_msgs/Bool` (external override) |

---

## ③ DAgger — `run_dagger`

**HG-DAgger** interactive takeover, **bimanual, single gate**:

- **Normal:** the workstation **policy** publishes `<s>/policy_action`; this node
  applies it to the followers and **back-drives the leaders** so a human resting
  on the handles feels what the policy intends.
- **Intervention:** pressing the gate (either handle's top button, or
  `/dagger/intervention_cmd`) switches **both** arms to human control — each
  follower tracks its leader — and streams `<s>/human_action` plus
  `/dagger/intervention=true` so the workstation can aggregate `(obs, action)`.
- **Release:** control returns to the policy.

```bash
python -m i2rt.ros2.run_dagger --sim
python -m i2rt.ros2.run_dagger --bilateral-kp 0.15     # real hardware
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
