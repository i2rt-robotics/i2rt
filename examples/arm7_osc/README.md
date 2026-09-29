# 7-DOF operational-space control

Operational-space control with nullspace regularization for the YAM 7-DOF arm. The tool holds
position while the elbow stays free to be pushed aside.

This file is a runbook. Follow it in order; every stage assumes the one before it passed.

---

## 1. Simulation

No hardware needed.

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .

python examples/arm7_osc/arm7_osc_demo.py --headless          # prints measurements
python examples/arm7_osc/arm7_osc_demo.py --viser             # browser view, localhost:8080
python examples/arm7_osc/arm7_osc_demo.py --headless --mode full
```

Expected: tool hold error 0.001 mm, loop 500.0 Hz, p99 2.12 ms.

---

## 2. Hardware prerequisites

**Arm.** YAM 7-DOF, 7 Damiao motors on one CAN channel. Motor types must match
`i2rt/robots/config/yam_7dof_right_v1.yml`.

**Motor ids.** Every motor needs `MASTER_ID == ID`. The i2rt default is `ID + 16`, which the
EtherCAT-to-CAN board cannot use — the CAN side goes completely silent while the EtherCAT layer
looks healthy. Check and fix before anything else:

```bash
# read-only, bridge must be stopped
~/ecat_probe/ecatsetid/build/ecat_setid enp99s0 --ids 1,2,3,4,5,6,7

# writes Flash, one motor at a time; power-cycle afterwards and re-check
~/ecat_probe/ecatsetid/build/ecat_setid enp99s0 --ids 1 --write
```

**Bridge.** A patched `EncosBridge` that subscribes to `/motors/cmd_torque`. Stock has the
`cmd_torque_` member and feeds it into the MIT frame but never fills it, so nothing can be
commanded. Its config must carry unit directions and zero offsets — calibration is applied on the
i2rt side.

**Probe tools.** `~/ecat_probe/`. Give them raw-socket access once (lost on every rebuild):

```bash
sudo setcap cap_net_raw,cap_net_admin=eip ~/ecat_probe/ecatscan/build/ecat_scan
sudo setcap cap_net_raw,cap_net_admin=eip ~/ecat_probe/ecatsetid/build/ecat_setid
```

---

## 3. Start the bridge

**Terminal A**, and leave it running:

```bash
source /opt/ros/humble/setup.bash
source ~/ros_ws/install/setup.bash
export ROS_LOCALHOST_ONLY=1
export ETHERCAT_INTERFACE=enp99s0
export MOTOR_CONFIG_PATH=$HOME/humanoid_internal/hw_interface/encos_motor_control_interface/config/motor_config_yam7dof.yaml
~/ros_ws/install/encos_motor_control_interface/lib/encos_motor_control_interface/encosbridge
```

All five lines are required. Do **not** use `run_encosbridge.sh`: it runs under `sudo`, and a
root-created Fast DDS shared-memory segment is unreadable by the client.

Expected on startup: 7 motors listed, `Operational state reached for all slaves`.

**Terminal B**, for every command in section 4:

```bash
source /opt/ros/humble/setup.bash
source ~/ros_ws/install/setup.bash
export ROS_LOCALHOST_ONLY=1
cd ~/i2rt
```

Source ROS **before** activating the venv. `ROS_LOCALHOST_ONLY` must match terminal A or the two
nodes never see each other.

---

## 4. Stage order

Shorthand used below:

```bash
B="python examples/arm7_osc/arm7_bringup.py --transport ethercat --loop-hz 500"
```

`--loop-hz 500` is the value section 4.2 measures. Passing less uncaps the host damping and the
lowest-friction joint diverges on first touch.

### 4.1 All seven motors present — read-only

```bash
~/ecat_probe/ecatscan/build/ecat_scan enp99s0     # bridge stopped
```

**Pass:** seven rows, ids 1–7, MOS temperature 33–40 °C. A row for id 8 with temperature 0 is the
sentinel, not a motor. **Any missing row: stop.** A run with one joint off the bus produces
plausible numbers that mean nothing.

`--stage check` reads motor registers but needs `--transport can`; the bridge exposes no registers.

### 4.2 Bus rates — energises nothing

```bash
$B --stage survey --max-torque 18
```

**Pass:** per-motor feedback rate printed for all seven. **Record it and pass it as `--loop-hz`.**

| | measured |
|---|---|
| Per-motor feedback rate | 500.0 Hz |
| Inter-motor skew | 0.00 ms |
| Raw poll rate | 1621 Hz |

### 4.3 Loop timing (P0)

```bash
$B --stage jitter --max-torque 18 --seconds 60
```

**Energises the motors with no gravity compensation.** The arm sinks. Park it resting or supported
first. `--motor-kd` defaults to 1.0 so it sinks rather than drops.

**Pass:** p99 ≤ 3 ms. Measured 2.831 ms.

### 4.4 Gravity compensation (A1)

```bash
$B --stage float --max-torque 18
```

`--max-torque 18` is required; the default 3 N·m is below the 14–17 N·m joint1 carries.

**Do:** move the arm to a pose by hand, let go, **do not touch it for 60 s**, watch the `holding`
line. Repeat at three poses with different shoulder loading. Ctrl-C when done; the trace path is
printed at the start.

**Pass:** `tool` ≤ 10 mm per hold. Measured 0.38 / 0.00 / 0.24 mm.

### 4.5 Nullspace

```bash
$B --stage null --max-torque 18 --motor-kd 2.0
```

**Hands off.** The arm drives itself along a 25° arc planned to hold the tool fixed.

**Do:** watch the gripper against a fixed mark. How far it physically moves is the result the
encoders cannot give.

**Pass:** gripper visibly near-still; printed tool displacement around 11 mm. Measured 10.6 mm.

### 4.6 Task space, `kinematic`

```bash
$B --stage task --max-torque 18 --motor-kd 5.0 --gain-scale 5.0 --trace-hz 200
```

**Do**, in this order:
1. Let it settle, **hands off for 10 s**.
2. Push the tool sideways ~2 cm, release, wait until fully stopped. Repeat 6–8 times.
3. Push the elbow sideways to 10° or more, hold, release. Repeat 4–5 times.

**Pass:** no buzzing; peak `|dq|` well under 4.0 rad/s; the tool returns after release. Working
point 5.0x at `--motor-kd 5.0`, validated at five start poses spanning `cond(J)` 14.8–43.1.

Raising `--gain-scale` requires raising `--motor-kd` with it. Measured pairs that hold: 2.0/2.0,
3.0/3.5, 4.0/4.5, 5.0/5.0. `3.0/2.0` diverges. `--motor-kd` above 5.0 is clamped by firmware.

Other poses to repeat at:

```bash
$B --stage task --max-torque 18 --motor-kd 5.0 --gain-scale 5.0 --trace-hz 200 \
   --start-pose "0,20,0,-25,0,20,0"        # extended
$B ... --start-pose "0,70,30,-95,0,30,0"   # folded
$B ... --start-pose "-40,50,35,-70,20,35,0" # side reach
$B ... --start-pose "0,45,20,-60,60,30,40"  # wrist rolled
```

### 4.7 Tracking (A3)

```bash
$B --stage track --max-torque 18 --motor-kd 2.0 --gain-scale 3.0 --trace-hz 200
```

**Hands off.** 8 s sinusoid × 3 cycles. The plan is checked against joint limits and `cond(J)`
before and after the goto; it refuses rather than driving a bad path.

**Pass:** tool position RMS ≤ 8 mm. Measured 6.99 mm at 3.0x. At 2.0x it is 9.5 mm.

### 4.8 Watchdog (A4)

```bash
$B --stage watchdog --max-torque 18 --motor-kd 2.0
```

Holds the current pose, then stops commanding on purpose. **Put the arm in a mid-range pose first**
and keep a hand near it.

**Do:** when it stops, push the arm gently — it must hold itself up and be back-drivable, neither
stiff nor limp. Then press Enter to test `resume()`.

**Pass:** stops in ≤ 320 ms, state `damped`, resume returns `active`. Measured 284 ms.

### 4.9 Task space, `full`

```bash
$B --stage task --mode full --max-torque 18 --motor-kd 5.0 \
   --gain-scale 4.0 --cart-zeta 1.0 --trace-hz 200
```

The modal check does not constrain this mode — it passes at every gain scale tried. Start at
`--gain-scale 0.25`, settle 10 s untouched, touch with one finger, then work up.

**Working point:** 4.0x, `--cart-zeta 1.0`, `--motor-kd 5.0`. Bracketed on both sides: 2.0x and
3.0x recover worse, 5.0x recovers worse, `--cart-zeta 0.7` recovers worse, `--cart-zeta 1.4`
produces a sustained 21.8 Hz limit cycle.

`--gain-scale` here is `Kp6 = 40 × scale` with units of 1/s², not N/m. Task bandwidth is `√Kp6`.

---

## 5. Tuning a new arm

Section 4 uses the working point for the arm this was developed on. On a different arm, or after a
mechanical change, derive it in this order. Each step feeds the next.

### 5.1 Friction and the gravity model — `--stage sweep`

Per joint, once. Writes the two vectors everything else reads.

```bash
$B --stage sweep --max-torque 18 --joint 1
python examples/arm7_osc/analyze_sweep.py /tmp/arm7_sweep_j1_*.csv --joint 1 --applied-factor 1.2
```

`--joint` must match the sweep (it defaults to 4). `--applied-factor` is the
`gravity_comp_factor` entry the controller was running for that joint during the sweep.

The joint creeps both ways under gravity compensation with a slow torque bias. The analysis prints
the gravity-model error and the Coulomb friction separately.

Put the results in `i2rt/robots/config/yam_7dof_<side>_v1.yml`:

| field | from |
|---|---|
| `coulomb_friction` | the friction column |
| `gravity_comp_factor` | 1.0 minus the model error, per joint |

Repeat for all seven. `SWEEP_CARRIERS` in `arm7_bringup.py` holds the pose each joint must be swept
from; a roll axis at the hanging pose has zero gravity torque and measures nothing.

**Check:** friction must be positive on every joint. A negative value means the fit is meaningless —
re-run that joint.

### 5.2 Feedback rate — `--stage survey`

```bash
$B --stage survey --max-torque 18
```

Record the per-motor rate and pass it as `--loop-hz` everywhere from here on. Everything in 5.4
depends on it.

### 5.3 Gravity — `--stage float`

```bash
$B --stage float --max-torque 18
```

Hold three poses for 60 s each. If a joint drifts, adjust **that joint's** `gravity_comp_factor`,
not a gain:

| Observation | Action |
|---|---|
| Joint sags | raise its factor by 0.05 |
| Joint rises | lower its factor by 0.05 |
| Drifts only at one pose | the model is wrong there; re-run 5.1 for that joint |

Iterate with `--gravity-factor "1.2,1.0,0.6,..."` to try values without editing the config, then
write the final set back.

**Stop when:** tool drift ≤ 10 mm over 60 s at three poses. Do not proceed with a drifting arm —
every later measurement rides on top of it.

### 5.4 Task gains, `kinematic` — `--stage task`

Two knobs, moved together. `--gain-scale` multiplies `KP_TASK_JOINT`; `--motor-kd` sets the
motor-side damping. Host-side damping is capped automatically from the friction measured in 5.1.

Start at `--gain-scale 2.0 --motor-kd 2.0` and step up:

```bash
$B --stage task --max-torque 18 --motor-kd 2.0 --gain-scale 2.0 --trace-hz 200
$B --stage task --max-torque 18 --motor-kd 3.5 --gain-scale 3.0 --trace-hz 200
$B --stage task --max-torque 18 --motor-kd 4.5 --gain-scale 4.0 --trace-hz 200
$B --stage task --max-torque 18 --motor-kd 5.0 --gain-scale 5.0 --trace-hz 200
```

At each step: settle 10 s untouched, push the tool 6–8 times, push the elbow 4–5 times.

| Observation | Action |
|---|---|
| Holds, quiet when settled | step up |
| One joint diverges on touch | raise `--motor-kd` by 1.0 and retry the same `--gain-scale` |
| Diverges again at the same scale, `--motor-kd` at 5.0 | the previous step is the working point |
| Buzzes continuously below 4 rad/s | the previous step is the working point |
| Velocity guard trips | the previous step is the working point |

`--motor-kd` is firmware-clamped at 5.0, so the last usable pair is whatever holds at that value.

Then repeat the working point at four more start poses spanning the workspace (section 4.6 lists
the ones used here). **The lowest scale that holds at every pose is the working point**, not the
best one.

Reading the trace afterwards is what separates a stable run from a quiet-looking one:

```python
# non-"start" rows, settled segments
rms = np.sqrt((dq[label == "settled"] ** 2).mean())   # should be ~0.001-0.003 rad/s
```

Count sign changes in `dq` for the worst joint in the 50 samples before its peak: a push is one
excursion, an instability alternates and grows. Threshold the count above the noise floor
(~0.05 rad/s) or a quiet run scores worst.

### 5.5 Tracking gains — `--stage track`

This stage runs matched gains (nullspace equal to task), so its scale is tuned separately from 5.4.

```bash
$B --stage track --max-torque 18 --motor-kd 2.0 --gain-scale 2.0 --trace-hz 200
$B --stage track --max-torque 18 --motor-kd 2.0 --gain-scale 3.0 --trace-hz 200
```

| Observation | Action |
|---|---|
| RMS above target, joint lag small | raise `--gain-scale`; the residual is stiction dead zone |
| RMS above target, one joint's amplitude ratio far from 1.0 | that joint is not tracking — check it individually |
| Faults mid-run | drop one step |

### 5.6 `full` mode — `--stage task --mode full`

`--motor-kd` stays at 5.0 and does not need re-balancing here. Two knobs:

```bash
$B --stage task --mode full --max-torque 18 --motor-kd 5.0    --gain-scale 0.25 --cart-zeta 1.0 --trace-hz 200
```

Work `--gain-scale` up from 0.25 by observation — the modal check does not constrain this mode.
Then try `--cart-zeta` either side of 1.0.

| Observation | Action |
|---|---|
| Push does not return at all | raise `--gain-scale`; it is a bandwidth floor |
| Recovery stops improving as the scale rises | you are at the working point |
| Sustained oscillation after raising `--cart-zeta` | lower it; 1.0 is the default for a reason |
| Recovery worse after lowering `--cart-zeta` | put it back to 1.0 |

Measure recovery from the trace: for each excursion above 15 mm, take the minimum tool
displacement within 3 s of the peak. Compare residual as a *fraction* of the excursion, not in mm —
push size varies between attempts.

---

## 6. Reference results

All on the EtherCAT bridge with seven joints live.

| | required | measured |
|---|---|---|
| P0 loop jitter p99 | ≤ 3 ms | 2.831 ms |
| Per-motor feedback rate | record | 500.0 Hz |
| Inter-motor skew | record | 0.00 ms |
| A1 gravity float, 3 × 60 s | ≤ 10 mm | 0.38 / 0.00 / 0.24 mm |
| A3 tracking RMS | ≤ 8 mm | 6.99 mm |
| A4 watchdog stop | ≤ 320 ms | 284 ms |
| Nullspace arc, tool | — | 10.6 mm |
| Tool per degree of elbow, `kinematic` | — | 4.9–11.7 mm/deg |
| Tool per degree of elbow, `full` | — | 1.5–2.2 mm/deg |
| Push recovery, `kinematic` 5.0x | — | 4–7 % of the excursion |
| Push recovery, `full` 4.0x | — | 11–12 % |

Not measured: A2 and A6 (need a dial indicator and a spring scale), A5 (2-hour soak),
direction-uniformity of tool stiffness, bridge dropout under load.

---

## 7. Troubleshooting

| Symptom | Action |
|---|---|
| Bridge: `No slaves found` | `ETHERCAT_INTERFACE` wrong, or the cable is in the board's OUT port |
| Bridge: `ROS_LOCALHOST_ONLY is not set` | export it in the bridge's own shell |
| Client hangs waiting for state | `ROS_LOCALHOST_ONLY` differs between terminals |
| `ModuleNotFoundError: rclpy` | source ROS in that terminal, before the venv |
| `ecat_scan` shows fewer than 7 motors | check power and the CAN connector on the missing joint |
| One joint reads but the stack ignores it | `MASTER_ID != ID`; reflash and power-cycle |
| A joint stops responding mid-session, no fault | bridge stall latch (bit 0x08). Only a bridge restart clears it |
| One joint diverges on first touch | `--loop-hz` too low — pass the surveyed rate |
| Buzzing that never decays, velocity below 4 rad/s | limit cycle; the guard will not catch it. Ctrl-C and lower the gain |
| Velocity guard trips at 4 rad/s | back off one gain step; raise `--motor-kd` first |
| Arm still stiff after Ctrl-C | run `release_arm.py`, then `pkill -f encosbridge` |
| The goto inside a stage lands >10° short | raise `--max-torque`, or a joint is against a stop |
| Tool drifts while floating | `gravity_comp_factor`, not gains. Re-run `--stage sweep` |

---

## 8. Safety

- e-stop within reach of the hand that is not on the keyboard, second person present.
- `--max-torque` starts at 3 N·m, below what the shoulder needs. Raise it deliberately; 18 is the
  working value for this arm.
- `--stage jitter` and `--stage mirror` apply no gravity compensation. The arm sinks or falls.
- `gravity_comp_factor` above 1.0 is positive feedback — joints 1 and 7 ship at 1.2 and can run to
  their stops once something moves the arm past stiction.
- Limits live at the top of `i2rt/robots/arm7_osc.py`: enable ramp, per-joint torque and slew
  bounds, joint-limit clamping, velocity stop, command timeout requiring an explicit `resume()`.
- Power-on state is gravity-comp float, not zero torque. `go_limp()` is the explicit release.
