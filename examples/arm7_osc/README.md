# 7-DOF operational-space control

Operational-space control with nullspace regularization for the YAM 7-DOF arm. The tool holds
position while the elbow stays free to be pushed aside.

This file is a runbook. Follow it in order; every stage assumes the one before it passed.

---

## 1. Simulation

No hardware needed.

```bash
uv sync && source .venv/bin/activate      # also builds the EtherCAT master (Linux; cmake + a C++ compiler)

python examples/arm7_osc/arm7_osc_demo.py --headless          # prints measurements
python examples/arm7_osc/arm7_osc_demo.py --viser             # browser view, localhost:8080
python examples/arm7_osc/arm7_osc_demo.py --headless --mode full
```

Expected: tool hold error 0.001 mm, loop 500.0 Hz, p99 2.12 ms.

---

## 2. Hardware prerequisites

**Arm.** YAM 7-DOF, 7 Damiao motors on one CAN channel. Motor types must match
`i2rt/robots/config/yam_7dof_right_v1.yml`.

**joint6 is behind a four-bar.** Its motor turns a crank and the joint follows at a ratio of
0.14–0.87 (`i2rt/robots/four_bar.py`). `arm7_bringup.py` now declares the linkage to `Arm7OSC` and
converts it in `_read_joint_state`, so joint6 is controlled and reported in joint space. The
measurements below were taken **before** that, with the motor angle read as the joint's. Re-check
joint6's gains on the bench (`--stage float`, then `track`) before trusting them: its effective
stiffness is now what the gains say, where before it was 1/ratio² times higher (1.3x mid-travel,
up to 50x near the upper stop).

Outside this runbook, `get_yam_robot(arm_type=ArmType.YAM_7DOF_RIGHT, control_mode="osc")` (or
`"osc_full"`) builds the same controller behind the ordinary `Robot` API, gripper included.

**EtherCAT.** The EtherCAT-to-CAN board on its own NIC (`enp99s0` below), the arm on its CAN
channel 1. The EtherCAT master runs inside the bring-up script: `uv sync` builds it
(`i2rt-ecat/`, SOEM on a real-time thread), and nothing else has to be running. Opening the NIC
needs `cap_net_raw`, and the master's real-time priority needs `cap_sys_nice`. Grant both once, to
this venv's interpreter only, as yourself (it asks for sudo):

```bash
scripts/grant_ethercat_caps.sh             # scripts/grant_ethercat_caps.sh --remove takes them away
```

It gives the venv its own copy of the interpreter first -- never `setcap` the path `.venv/bin/python`
links to, which is the shared interpreter, so every Python program using it would get raw sockets --
and makes the other interpreter names **hard** links to that copy: an interpreter holding
capabilities runs in secure mode, and started through a symlink it cannot find its standard library,
which breaks `uv sync` (uv runs `python3`). Re-run it whenever `uv sync` recreates the venv; until
then, `sudo -E .venv/bin/python ...` works. `cap_sys_nice` also lets `Arm7OSC`'s own loop take
real-time priority, which it could not before, so re-measure `--stage jitter` after granting it.

**Motor ids.** Every motor needs `MASTER_ID == ID`: the board's replies are routed by the id they
carry, and the i2rt default of `ID + 16` leaves every motor silent. Read it through the board, and
fix it over a USB-CAN adapter (writes Flash; power-cycle afterwards and re-read):

```bash
python i2rt/motor_config_tool/dm_motor_registers.py read MST_ID --motor-id 1 --channel ecat:enp99s0
python i2rt/motor_config_tool/dm_motor_registers.py write MST_ID --value 1 --motor-id 1 --channel can0
python i2rt/motor_config_tool/dm_motor_registers.py save MST_ID --motor-id 1 --channel can0
```

Every tool in `i2rt/motor_config_tool/` takes `--channel ecat:<nic>` (or `ecat:<nic>:<board>:<can
channel>`). The DM ones (`ping_motors`, `set_zero`, `set_timeout`) use `same` there whatever
`--receive-mode` says; on a CAN adapter, motors flashed this way need `--receive-mode same`, and so
does `--transport can` here.

---

## 3. What changed from the previous transport

The numbers in sections 4-6 were measured through an external ROS 2 bridge, which this in-process
master replaces. Three differences matter when reading them:

- **kd was delivered at 1/8.** The bridge packed motor-side kd into 9 bits of a 12-bit field, so
  every motor got `trunc(kd / 5 * 511) / 4095 * 5`, about 0.125x of the stated value. The master
  here sends it at full resolution. The `--motor-kd` values below are already translated to what the
  motors actually received; the working point written as "5.0" was really 0.62.

  | old `--motor-kd` | delivered, and the value to pass now |
  |---|---|
  | 1.0 | 0.12 |
  | 2.0 | 0.25 |
  | 3.5 | 0.44 |
  | 4.5 | 0.56 |
  | 5.0 | 0.62 |

  The old ceiling of 0.62 is gone: the motors now accept up to 5.0. `MOTOR_KD_*` in
  `i2rt/robots/arm7_osc.py` and `JITTER_MOTOR_KD` now deliver their nominal values, 8x what they
  delivered before.
- **Per-motor feedback rate is 428.6 Hz**, not 500: the bridge republished at 500 Hz, but the board
  refreshes each of seven motors on one CAN channel 3 times in 7 cycles. Pass the rate `--stage
  survey` measures as `--loop-hz`.
- **The board's own rules**, measured on it and handled in the master:
  - a reply arrives two cycles after its command, in the same frame slot;
  - a slot nothing answered keeps its last reply for ~25 ms;
  - a slot only accepts replies carrying the id it last sent.

  So a channel needs at least three motors, or a stopped motor's last reply could read as fresh.
  Register access (`dm_motor_registers.py --channel ecat:<nic>`) primes a slot with a remote frame
  before each request, and the motor answers that with its state, which is discarded.
- **Stopping releases the arm.** If the script stops commanding for 100 ms the master releases every
  motor (zero torque, zero stiffness, kd 1.0), and a normal exit releases and then disables them. The
  bridge held the last command forever. A killed process stops the master too, so what the motors do
  then is up to their own CAN timeout (`set_timeout.py`).

---

## 4. Stage order

Shorthand used below:

```bash
B="python examples/arm7_osc/arm7_bringup.py --transport ethercat --interface enp99s0 --loop-hz 428"
```

`--loop-hz 428` is the value section 4.2 measures. Passing less uncaps the host damping and the
lowest-friction joint diverges on first touch.

### 4.1 All seven motors present — read-only

```bash
$B --stage check
```

Scans the board with every motor disabled, then reads each motor's type and scaling registers.
Writes nothing, energises nothing.

**Pass:** seven ids answering at ~428 Hz, MOS temperature 33–40 °C, then `OK -- every motor holds
the type and scaling this config declares`. **Any silent motor: stop.** A run with one joint off
the bus produces plausible numbers that mean nothing.

### 4.2 Bus rates — energises nothing

```bash
$B --stage survey --max-torque 18
```

**Pass:** per-motor feedback rate printed for all seven. **Record it and pass it as `--loop-hz`.**

| | expected |
|---|---|
| Per-motor feedback rate | 428.6 Hz (3000/7) |
| Master `wkc_errors` | 0 |

### 4.3 Loop timing (P0)

```bash
$B --stage jitter --max-torque 18 --seconds 60
```

**Energises the motors with no gravity compensation.** The arm sinks. Park it resting or supported
first. `--motor-kd` defaults to 1.0 so it sinks rather than drops (0.12 on the previous transport).

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
$B --stage null --max-torque 18 --motor-kd 0.25
```

**Hands off.** The arm drives itself along a 25° arc planned to hold the tool fixed.

**Do:** watch the gripper against a fixed mark. How far it physically moves is the result the
encoders cannot give.

**Pass:** gripper visibly near-still; printed tool displacement around 11 mm. Measured 10.6 mm.

### 4.6 Task space, `kinematic`

```bash
$B --stage task --max-torque 18 --motor-kd 0.62 --gain-scale 5.0 --trace-hz 200
```

**Do**, in this order:
1. Let it settle, **hands off for 10 s**.
2. Push the tool sideways ~2 cm, release, wait until fully stopped. Repeat 6–8 times.
3. Push the elbow sideways to 10° or more, hold, release. Repeat 4–5 times.

**Pass:** no buzzing; peak `|dq|` well under 4.0 rad/s; the tool returns after release. Working
point 5.0x at `--motor-kd 0.62`, validated at five start poses spanning `cond(J)` 14.8–43.1.

Raising `--gain-scale` requires raising `--motor-kd` with it. Measured pairs that hold
(`--gain-scale`/`--motor-kd`): 2.0/0.25, 3.0/0.44, 4.0/0.56, 5.0/0.62. `3.0/0.25` diverges. 0.62
was the previous transport's ceiling, not the motors': above it is untested.

Other poses to repeat at:

```bash
$B --stage task --max-torque 18 --motor-kd 0.62 --gain-scale 5.0 --trace-hz 200 \
   --start-pose "0,20,0,-25,0,20,0"        # extended
$B ... --start-pose "0,70,30,-95,0,30,0"   # folded
$B ... --start-pose "-40,50,35,-70,20,35,0" # side reach
$B ... --start-pose "0,45,20,-60,60,30,40"  # wrist rolled
```

### 4.7 Tracking (A3)

```bash
$B --stage track --max-torque 18 --motor-kd 0.25 --gain-scale 3.0 --trace-hz 200
```

**Hands off.** 8 s sinusoid × 3 cycles. The plan is checked against joint limits and `cond(J)`
before and after the goto; it refuses rather than driving a bad path.

**Pass:** tool position RMS ≤ 8 mm. Measured 6.99 mm at 3.0x. At 2.0x it is 9.5 mm.

### 4.7b Straight line under nullspace turbulence

```bash
$B --stage line --mode full --max-torque 18 --motor-kd 0.62 --gain-scale 4.0 --cart-zeta 1.0 --trace-hz 200
```

**Hands off.** Two passes, each sliding the tool back and forth along an 80 mm line (`--line-mm`,
`--line-axis`, `--track-period`, `--track-cycles`). The second repeats the line while the commanded
posture wanders along the self-motion manifold by up to `--turbulence-deg`: pure self-motion, so
the tool target never leaves the line.

**Pass:** `STILL LINEAR`, i.e. the turbulent pass is within 1 mm RMS of the reference pass. Also
check that `posture moved` is close to what was commanded; if the posture barely moved, the test
proved nothing. In sim this adds 0.3 mm at the default 6 degrees. A fast, large wander leaks more
(3 mm at 25 degrees), because `Arm7OSC` does not compensate velocity-dependent forces.

**On the arm, friction decides it.** A 120 mm vertical line with 15° of turbulence
(`--line-mm 120 --line-axis 0,0,1 --turbulence-deg 15`) gives:

| Mode | plain | turbulent | added |
|---|---|---|---|
| `full` 4.0x | 14.5 mm | 23.8 mm | +9.3 mm |
| `kinematic` 3.0x | 6.4 mm | 7.2 mm | +0.8 mm, `STILL LINEAR` |

`Kp6` is an acceleration gain, so the stiffness it gives a joint follows that joint's inertia, not
its friction. At 4.0x joint3 gets 16 N.m/rad against `kinematic`'s 120. `--kp6` raises the task
gain alone, without the nullspace that `--gain-scale` also scales, and `--cart-zeta 0.5` holds
`Kd6` at the 25.3 the arm has already run. In sim with this arm's friction, `--kp6 640 --cart-zeta 0.5`
takes the same line from 13.8 / 14.0 to 3.9 / 4.5 mm.

On the arm the same setting gives 5.0 / 7.9 mm (+2.9), but **it is not a working point**. Once the
joints move, joints 1 and 4 oscillate at 24 Hz (0.95 / 0.82 N.m RMS above 5 Hz, against 0.26 /
0.14 at 4.0x), with the torque held at the 200 N.m/s slew limit 25–40% of the time. Task holds at
320, 480 and 640 were all quiet: stiction pins a joint at rest, so a hold cannot show this. Judge a
`--kp6` on a moving trace, the line's, by its torque ripple above 5 Hz. 320 and 480 have not been
run on the line yet. Re-check push recovery (§4.9) at any new point too; its bracket found ζ 0.7
recovering worse.

```bash
$B --stage task --mode full --max-torque 18 --motor-kd 0.62 --gain-scale 4.0 --kp6 320 --cart-zeta 0.71 --trace-hz 200
$B --stage line --mode full --max-torque 18 --motor-kd 0.62 --gain-scale 4.0 --kp6 320 --cart-zeta 0.71 \
   --line-mm 120 --line-axis 0,0,1 --turbulence-deg 15 --trace-hz 200
```

### 4.8 Watchdog (A4)

```bash
$B --stage watchdog --max-torque 18 --motor-kd 0.25
```

Holds the current pose, then stops commanding on purpose. **Put the arm in a mid-range pose first**
and keep a hand near it.

**Do:** when it stops, push the arm gently — it must hold itself up and be back-drivable, neither
stiff nor limp. Then press Enter to test `resume()`.

**Pass:** stops in ≤ 320 ms, state `damped`, resume returns `active`. Measured 284 ms.

### 4.9 Task space, `full`

```bash
$B --stage task --mode full --max-torque 18 --motor-kd 0.62 \
   --gain-scale 4.0 --cart-zeta 1.0 --trace-hz 200
```

The modal check does not constrain this mode — it passes at every gain scale tried. Start at
`--gain-scale 0.25`, settle 10 s untouched, touch with one finger, then work up.

**Working point:** 4.0x, `--cart-zeta 1.0`, `--motor-kd 0.62`. Bracketed on both sides: 2.0x and
3.0x recover worse, 5.0x recovers worse, `--cart-zeta 0.7` recovers worse, `--cart-zeta 1.4`
produces a sustained 21.8 Hz limit cycle.

`--gain-scale` here is `Kp6 = 40 × scale` with units of 1/s², not N/m, unless `--kp6` sets it
directly (§4.7b). Task bandwidth is `√Kp6`.

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
| `breakaway_friction` | the friction column |
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

Start at `--gain-scale 2.0 --motor-kd 0.25` and step up:

```bash
$B --stage task --max-torque 18 --motor-kd 0.25 --gain-scale 2.0 --trace-hz 200
$B --stage task --max-torque 18 --motor-kd 0.44 --gain-scale 3.0 --trace-hz 200
$B --stage task --max-torque 18 --motor-kd 0.56 --gain-scale 4.0 --trace-hz 200
$B --stage task --max-torque 18 --motor-kd 0.62 --gain-scale 5.0 --trace-hz 200
```

At each step: settle 10 s untouched, push the tool 6–8 times, push the elbow 4–5 times.

| Observation | Action |
|---|---|
| Holds, quiet when settled | step up |
| One joint diverges on touch | raise `--motor-kd` by 0.12 and retry the same `--gain-scale` |
| Diverges again at the same scale, `--motor-kd` at 0.62 | the previous step is the working point, or try more kd (untested) |
| Buzzes continuously below 4 rad/s | the previous step is the working point |
| Velocity guard trips | the previous step is the working point |

The previous transport could not deliver more than 0.62, so the pairs above stop there; the motors
accept up to 5.0.

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
$B --stage track --max-torque 18 --motor-kd 0.25 --gain-scale 2.0 --trace-hz 200
$B --stage track --max-torque 18 --motor-kd 0.25 --gain-scale 3.0 --trace-hz 200
```

| Observation | Action |
|---|---|
| RMS above target, joint lag small | raise `--gain-scale`; the residual is stiction dead zone |
| RMS above target, one joint's amplitude ratio far from 1.0 | that joint is not tracking — check it individually |
| Faults mid-run | drop one step |

### 5.6 `full` mode — `--stage task --mode full`

`--motor-kd` stays at 0.62 and does not need re-balancing here. Two knobs:

```bash
$B --stage task --mode full --max-torque 18 --motor-kd 0.62 --gain-scale 0.25 --cart-zeta 1.0 --trace-hz 200
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

Right arm, seven joints live. The first table is this in-process master (2026-09-30), the second
the previous transport (section 3), where kd was delivered at 1/8 -- compare at the translated
values.

On this master:

| | required | measured |
|---|---|---|
| P0 loop jitter p99, 60 s | ≤ 3 ms | 2.059 ms (master: 0 overruns, 54 µs max late) |
| Per-motor feedback rate | record | 427.8–428.0 Hz |
| Inter-motor skew | record | 2.00 ms (true per-motor stamps) |
| A1 gravity float at home, 34 s | ≤ 10 mm | 0.3 mm (the three hand-placed poses still to do) |
| A3 tracking RMS, 3.0x, `--motor-kd 0.25` | ≤ 8 mm | 7.27 mm (10.23 mm at 2.0x) |
| A4 watchdog stop | ≤ 320 ms | 281 ms, resume returns `active` |
| Nullspace arc, tool (encoders) | — | 13.1 mm |
| Task hold, `kinematic` 5.0x, `--motor-kd 0.62` | — | 0.30 mm, peak \|dq\| 0.038 rad/s |
| Task hold, `full` 4.0x, `--cart-zeta 1.0`, `--motor-kd 0.62` | — | 0.31 mm, peak \|dq\| 0.043 rad/s |
| Process killed while enabled | — | motors disabled by their own TIMEOUT within 1.7 s |

On the previous transport:

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
direction-uniformity of tool stiffness, dropout under load.

---

## 7. Troubleshooting

| Symptom | Action |
|---|---|
| `cannot open a raw socket ... cap_net_raw` | `scripts/grant_ethercat_caps.sh` (section 2), or run with `sudo -E` |
| `no network interface` / `no EtherCAT slaves` | wrong `--interface`, no carrier, or the cable is in the board's OUT port |
| `maps N output / N input bytes` | not the EtherCAT-to-CAN board this expects, or a different firmware layout |
| `motors [n] did not answer` | power and the CAN connector on that joint, or `MASTER_ID != ID` |
| Master stats show `rt_ok: 0` | `cap_sys_nice` missing: the master runs without real-time priority |
| `watchdog_trips` rising during a run | the script stalled for 100 ms or more; the arm was released each time |
| One joint diverges on first touch | `--loop-hz` too low — pass the surveyed rate |
| Buzzing that never decays, velocity below 4 rad/s | limit cycle; the guard will not catch it. Ctrl-C and lower the gain |
| Velocity guard trips at 4 rad/s | back off one gain step; raise `--motor-kd` first |
| Arm still stiff after a killed process | the motors hold their last command until their CAN timeout; set one with `set_timeout.py --timeout` |
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
- A stage that raised the arm to its start pose walks it to the stand pose `REST_POSE`
  (`0,0,0,90,0,0,0`, forearm horizontal) before releasing it, on soft joint gains with the nullspace
  matched to them. The forearm drops once released there: support it. A faulted (`damped`) arm is not driven
  anywhere: it is released where it stands, and the script says so first. Support it.
