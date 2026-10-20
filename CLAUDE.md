# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`i2rt` is a Python client library for I2RT hardware: YAM 6-DOF arms, their grippers, and the Flow Base
mobile platform. It talks to Damiao (DM) series motors over a CAN bus, provides MuJoCo simulation and
gravity compensation, and ships URDF/MJCF robot models. The package is named `i2rt` and built with **Flit** from
`pyproject.toml`; the root `setup.py` is legacy/unused (stale `version`, minimal deps) — treat
`pyproject.toml` as authoritative for dependencies and metadata.

## Environment & commands

The project uses **uv** (not pip/venv directly). Python 3.11 is what CI runs; `requires-python >= 3.10`.

```bash
# One-time setup
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e .              # or: uv sync --dev  (installs dev tools too)

# Tests (all run in MuJoCo sim — no CAN bus / motors needed)
uv run pytest -n auto            # full suite, parallel (what CI runs)
uv run pytest i2rt/robots/tests/test_urdf_mjcf_alignment.py -v      # one file
uv run pytest -k "yam_pro and linear_4310"                          # by keyword

# Lint / format / type-check (match CI + pre-commit exactly)
ruff check .                     # CI pins ruff==0.15.6; line-length 119, ANN rules on
ruff format .
uv run python -m pyright         # pre-commit only type-checks a scoped allowlist (see below)
pre-commit run --all-files       # uv-lock, uv-sync --locked, ruff, ruff-format, pyright, nbstripout
```

Two GitHub Actions workflows gate PRs: `ruff.yml` (`ruff check`) and `unit_tests.yml` (`uv sync --dev`
then `uv run pytest -n auto`). Both use Python 3.11.

### Tooling conventions

- **All CLIs use `tyro`, never `argparse`.** Every `.py` entry point (in `scripts/`, `i2rt/`, `examples/`)
  parses args with `tyro.cli(...)`. Follow this when adding or editing a CLI.
- **Type-checking is split:** `pyproject.toml` has a `[tool.mypy]` section (targets 3.12, excludes
  `scripts/`), but the pre-commit hook actually runs **pyright**, scoped to a small allowlist
  (`i2rt/utils/(encoder_manager|can_flash).py`). Extend the hook's `files:` regex as more modules are cleaned up.
- Ruff's `ANN` (annotations) rules apply to `scripts/` even though mypy excludes that directory.

## Architecture

### The Robot abstraction (`i2rt/robots/`)

Everything downstream programs against the `Robot` **Protocol** in [robot.py](i2rt/robots/robot.py)
(`get_joint_pos`, `command_joint_pos`, `get_observations`, ...). There are two implementations:

- **`MotorChainRobot`** ([motor_chain_robot.py](i2rt/robots/motor_chain_robot.py)) — real hardware. Runs a
  background control thread over the CAN motor chain, does gravity compensation using a MuJoCo model of the
  robot, and handles gripper force limiting / calibration.
- **`SimRobot`** ([sim_robot.py](i2rt/robots/sim_robot.py)) — MuJoCo-only, in-memory joint state. Implements the
  same protocol so tests and visualizers use it interchangeably. **This is the path the entire test suite exercises.**

Both are constructed through the single factory `get_yam_robot(...)` in
[get_robot.py](i2rt/robots/get_robot.py), switched by a `sim: bool` argument. That factory is where arm +
gripper config, joint limits, gains, and motor offsets get assembled — read it first to understand how a
robot instance comes together.

### Config-driven, runtime-composed models

A robot = an **arm** + a **gripper**, each independently selected. A **hardware revision is its own
arm variant** — `ArmType.YAM_ULTRA_2` (`yam_ultra_2`) is revision 2 of `yam_ultra` — so `ArmType` alone
identifies an arm across the robot-construction API (`get_yam_robot`, `combine_arm_and_gripper_xml`,
`ArmType.get_xml_path`, `_load_arm_config`) and the four robot CLIs (`motor_chain_robot.py` plus the three
`examples/` entry points): none of those take a `version` argument or a `--version` flag. A revision is
still passed explicitly *below* that line, where it is load-bearing and must not be "cleaned up" — the path
resolver `get_arm_xml_path(arm, version)` and the offline pipeline script `sync_gripper_mounts.py --version N`
(both described below). A **mirrored pair is two variants, not one plus a side**: `yam_7dof_left` and
`yam_7dof_right` are separate families with separate models, configs and mount blocks, so nothing about
them needs disambiguating at call time. `ArmType.family` / `ArmType.version` unpack a variant into the on-disk
`(family, v<N>)` pair that the config and model loaders resolve paths from. Registering a revision is a
**two-place edit** in [utils.py](i2rt/robots/utils.py): the `ArmType` member **and** its `(family, revision)`
entry in the `_ARM_VARIANTS` map. The two enums are the entry points:

- `ArmType` / `GripperType` enums live in [utils.py](i2rt/robots/utils.py) and map to:
  - a **hardware YAML** in [i2rt/robots/config/](i2rt/robots/config/) — motor IDs, directions, kp/kd,
    gravity-comp factors, gripper limits, force-torque maps. **Arm** configs are version-suffixed
    (`yam_v1.yml`, `yam_ultra_v1.yml`, `yam_7dof_right_v1.yml`); **gripper** configs are not
    (`linear_4310.yml`).
  - a **robot model** (URDF + MJCF) under [i2rt/robot_models/](i2rt/robot_models/), resolved via the path
    constants in [i2rt/robot_models/__init__.py](i2rt/robot_models/__init__.py).
- **`combine_arm_and_gripper_xml`** (in [utils.py](i2rt/robots/utils.py)) merges the arm MJCF and gripper MJCF
  into one model at runtime (written to `/tmp/`). **Critical subtlety:** it locates the arm's deepest body
  (the `gripper` mount frame) and *overwrites* its `pos`/`quat` and its joint's `axis` from the **gripper**
  config's `last_joint_mount.<arm>.v<N>` block. So the terminal-mount transform is physically arm data but is
  duplicated across every gripper config — a stale copy silently misplaces the gripper for that arm. The
  mount is keyed by arm **and** arm model version because a hardware revision can move the terminal frame;
  an un-versioned `last_joint_mount.<arm>` block (with `pos` directly under the arm key) is still accepted
  for backward compatibility. Seed a new version's blocks with
  `sync_gripper_mounts.py <arm>.xml --version N` rather than hand-editing all six configs. The
  block may also carry a `gripper_pos`/`gripper_quat` offset — see the 7-DOF section for why.
  The mount frame follows the tool convention **`+X` right, `+Y` down, `+Z` forward**: at the home pose its
  `+Z` (the approach axis) points along world `+X`. That holds on every artifact that carries a flange —
  the five YAM-family arm URDFs and MJCFs, their 30 composed models, and the `{side}_gripper` frames in
  `robot_models/station/` — and matches the ROS/OpenCV convention the station camera frames already use.
  On the 7-DOF arms the mount body is `joint7`'s wrist-pitch frame, so the flange is the
  `gripper_pos`/`gripper_quat` frame inside it; it follows the same convention, but since those arms hang,
  its `+Z` points along world `-Z` at home.
- **Both bodies are named `gripper`** — the arm's terminal mount (matching the URDF's last-joint child) and the
  gripper model's root — and MuJoCo body names must be unique, so composition *merges* them instead of nesting:
  the gripper's children move into the mount body under a `<frame>` carrying the gripper root's own
  `pos`/`quat`. Its `<inertial>` is the exception — MuJoCo accepts an `<inertial>` inside a `<frame>` but does
  **not** apply the frame transform to it, so that element is composed into the mount frame explicitly and
  replaces the mount's `mass=1e-6` placeholder. Composition also renumbers the gripper's own joints to
  continue the arm's count, since every gripper model names its fingers for a six-joint arm — a no-op
  unless the arm has more.

Arm variants: `yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, `big_yam`, `yam_7dof_left`, `yam_7dof_right`
(`yam_ultra_2` is the only shipped revision beyond v1). Grippers: `crank_4310`, `linear_3507`, `linear_4310`,
`flexible_4310`, plus the non-gripper `yam_teaching_handle_v1`, `yam_teaching_handle_v2` and
`no_gripper` (`GripperType.is_teaching_handle` covers both handles, which are two revisions of one
handle modelled under `robot_models/gripper/yam_teaching_handle/v<N>/`). `ArmType.NO_ARM` gives a
gripper-only robot, built by `_get_gripper_only_robot`. A teaching handle there is **not** a
one-motor gripper: it has no motor at all, so it forks to `_get_teaching_handle_robot` and the
`TeachingHandleRobot` / `MotorlessCanChain` pair in
[teaching_handle_robot.py](i2rt/robots/teaching_handle_robot.py) — a read-only 0-DOF robot over
the passive trigger encoder, `robot.motor_chain`-shaped so the control interfaces need no
special case.

### The 7-DOF arms (`yam_7dof_left`, `yam_7dof_right`)

Seven-DOF arms, and otherwise arms like any other: `ArmType.YAM_7DOF_LEFT` / `ArmType.YAM_7DOF_RIGHT`,
built by `get_yam_robot(arm_type=..., gripper_type=...)`, composed with any gripper through
`combine_arm_and_gripper_xml`. Two things about them are unlike every YAM:

- **They are a mirrored pair, registered as two families.** The left model is the right one reflected
  about the world x-z plane, but each is its own `ArmType` with its own `arm/<family>/v1/` dir, its own
  `<family>_v1.yml` and its own `last_joint_mount.<family>.v1` block. Nothing in the API or the CLIs
  takes a `side`: the arm name carries it. A hardware revision would be `v2` under both families.
- **Their last joint is a wrist pitch, not the tool roll.** Every YAM gripper bolts coaxially onto
  `joint6`, so the mount block positions the mount body and the gripper attaches at identity inside
  it. Here the gripper bolts ~70 mm *perpendicular* to `joint7`'s axis, which the mount pose cannot
  carry — moving it would move the joint's axis line. `last_joint_mount.<arm>.v<N>` therefore gains an
  optional `gripper_pos`/`gripper_quat`, composed into the `<frame>` that places the gripper. Absent on
  every YAM, where it is identity.

Two smaller consequences of having seven joints:

- **Gripper CAN id.** The arm's joints take `0x01`-`0x07`, so its gripper cannot use the YAM
  grippers' `0x07`. The arm config declares `gripper_can_id: 0x08`; `_ArmHWConfig.gripper_can_id`
  defaults to `0x07` for everyone else. `gripper_motor_direction` beside it overrides the gripper
  config's polarity for this arm (bench-found; it steers the force limiter).
- **Finger joint names.** Every gripper MJCF names its fingers `joint7`/`joint8`, for a six-joint
  arm. That collides here and MuJoCo rejects duplicate joint names, so
  `combine_arm_and_gripper_xml` renumbers the gripper's joints to continue the arm's count —
  `joint8`/`joint9` here, a no-op on every YAM.

Everything else follows the family:

- Model: [arm/yam_7dof_right/v1/](i2rt/robot_models/arm/yam_7dof_right/v1/) and
  [arm/yam_7dof_left/v1/](i2rt/robot_models/arm/yam_7dof_left/v1/), each holding
  `<family>.urdf`/`<family>.xml` and `assets/`. The file basenames and the MJCF `<mujoco model>`
  must equal the family name: the resolver builds the path from it, and `sync_gripper_mounts.py`
  keys the mount block off it. The MJCF is arm-only — `base`, `link1`..`link6`, then a `gripper` mount body
  carrying `joint7` with a `mass=1e-6` placeholder inertial, exactly as `yam.xml` does. It does carry
  one geom, `wrist.stl` — see below — which composition keeps, since only the inertial is replaced.
  Like `yam.urdf`, the **URDF is the whole robot**: its terminal `gripper` link is the canonical
  linear_4310 (mass `0.553219`) with both tips on prismatic joints, which
  `urdf_to_arm_mjcf.py --dof 7` drops. The three end-effector STLs are URDF-only orphans in
  `assets/`, as `arm/yam/v1/assets/`'s have always been.
- The as-built ONShape export fused a **0.103291 kg wrist section** into that gripper link — the
  mesh is 147.8 mm long where the linear_4310 housing alone is 74 mm. That mesh is **split**: the
  proximal 76 mm is `assets/wrist.stl`, cut at the gripper's own mounting face and carried as a geom
  on the mount body, and the distal half arrives with the gripper. Its **mass is deliberately not
  modelled** — a real inertial on the mount would be discarded by composition anyway — so the arm
  mass is the seven links only (`5.843760 kg`) and the composed wrist is ~0.101 kg lighter than the
  as-built model.
- That split cut away the **gripper-mount spigot**. Every YAM's last link ends in a ø38.08 mm boss
  that flares to ø57 mm over the 10.5 mm proximal of the mounting face and seats in the gripper's
  bore; the fusion erased the interface and the cut handed that whole band to the gripper side, so
  the wrist butted flat on the gripper's collar. `wrist.stl` now carries `arm/yam/v1`'s own spigot,
  cut at the same plane and placed at the same coordinates in the gripper-root frame (which is the
  linear_4310's own frame, shared by both arms) — so nothing is scaled or invented and the tool
  centre does not move. `test_wrist_carries_the_same_mount_spigot_as_a_yam` requires the two arms'
  interfaces to measure the same rather than pinning a number.
- **Gravity comp is shared by both sides** (2026-10-13): `gravity_comp_factor`, `grav_comp_kd` and
  `coulomb_friction` are identical in the two configs, and `test_sides_share_hardware_config` keeps
  them so. The first tune against this model (2026-09-22) replaced values found against the first
  ONShape export, whose links 2-5 were 0.536 kg lighter; the right arm's README tabulates those
  per-joint ratios. Factor and friction both rescale with the motor `KT_Value * GREF`, which both
  configs declare under `motor_registers` (joints 1-7, not the gripper) and the startup config check
  refuses to start without.
- Config: [config/yam_7dof_right_v1.yml](i2rt/robots/config/yam_7dof_right_v1.yml) and
  [yam_7dof_left_v1.yml](i2rt/robots/config/yam_7dof_left_v1.yml) — the ordinary arm schema at
  length 7, resolved by `_load_arm_config` from the family name like every other arm's. The two
  start out identical; they are separate files so each can carry its own bench-tuned values.
- The **left** model is the right one reflected about the world x-z plane (`y -> -y`), generated by
  `.claude/skills/align-urdf-mjcf/scripts/mirror_arm_model.py` (re-run it after any change to the
  right model, passing `--name yam_7dof_left` so the output carries its own family name).
  Every rule is a sign flip, so the mirror is bit-exact: positions negate `y`, frame
  orientations go `R -> M R M` (URDF `rpy`: negate roll and yaw; MJCF `quat`: negate x and z), the
  MJCF *principal-axis* `<inertial quat>` instead goes `(w,x,y,z) -> (-y,z,w,-x)` because `-M =
  Ry(pi)` is what re-rights the improper `M R_p`, `Ixy`/`Iyz` negate, and masses, principal moments
  and link lengths are invariant. `apply_urdf_heading.py` and `urdf_align_lib.transform_origin`
  cannot do this: scipy raises on a determinant of -1, and `verify_rigidity` exists to reject
  exactly a whole-tree change. The **gripper** is not mirrored — it is one physical part bolted to
  either arm — so a composed left arm is the mirror of a composed right one through the mount frame
  and at the tool centre, but its two fingers are not each other's reflection.
- Regenerating the model: `urdf_to_arm_mjcf.py --dof 7` now produces the whole arm MJCF, because the
  terminal body really is a placeholder. Two things still need putting back by hand: `link6`'s
  URDF-sourced colour (the PALETTE runs out at `link5`), the terminal `wrist` geom, and the
  `<contact>` section. The `link5`/`gripper` exclude **is** required: the wrist section nests around
  the short `link6` segment and their convex hulls overlap by up to 38 mm at every reachable pose. That
  is the arm's own geometry, so it holds for every gripper, `no_gripper` included. After
  regenerating, re-run `sync_gripper_mounts.py <arm>.xml` for that family; it syncs `pos`/`quat`/`axis`
  from the MJCF and leaves `gripper_pos`/`gripper_quat` alone, since the MJCF cannot express them.
- Joint signs: the requirement is stated in **world** directions at the home pose, not in local
  `<axis>` values. Right arm `-y -x -z -y -z -x -y` for joint1..joint7; left arm is the reflection,
  `-y +x +z -y +z +x -y`. `HOME_WORLD_AXES` in [test_arm_7dof.py](i2rt/robots/tests/test_arm_7dof.py)
  pins it. The local axes that satisfy it are *not* uniform (`joint1`/`joint3`/`joint5`/`joint6` sit
  on `-Z`, the rest on `+Z`), so do not "tidy" them. The export writes every joint on `+Z`, which
  after the heading gives `+y -x +z -y +z +x -y`; those four are the ones re-signed.
- Why the left's differ from the right's: a rotation axis is a pseudovector, `a -> -M a` under the
  reflection, so an axis along the mirror **normal** (+-y) keeps its world direction while one lying
  **in** the mirror plane reverses. A slide axis is an ordinary vector, `a -> M a`. Both leave the
  joint *coordinate* alone, so the two sides are true mirror images at `q_left = q_right`, and their
  `range` values are identical. `mirror_arm_model.py` applies exactly these two
  rules and touches no range.
- `directions` is `[-1, +1, +1, +1, +1, +1, +1]` on the right and `[-1, +1, -1, +1, -1, -1, +1]` on
  the left, whose joint3/joint5/joint6 motors do not reverse with their axes -- a mixed sign, where every YAM
  arm is a uniform `+1`. A specified world direction is not generally its motor's positive rotation,
  and `directions` is the layer that belongs to that: the driver applies it symmetrically on read and
  write, so for a given motor position the torque sent to each motor is bit-identical to the model
  without it, while the limits and `SimRobot` stay right in model coordinates. Contrast a genuine
  model/hardware **mismatch**, which must be fixed in the **model**, because a `directions` flip
  reaches neither the model's joint limits (`_arm_joint_limits` reads the MJCF) nor `SimRobot`, which
  has no notion of motor polarity -- so recording it there fixes gravity comp while leaving the
  limits and sim mirrored. Either kind of sign error is invisible at `q = 0`, which is why zeroing
  the motors looks correct. Re-signing an
  axis must carry its `range` with it, negated and swapped, or the physical travel moves. The left
  arm's `directions` is **not bench-verified**.
- Joint limits are this export's **CAD travel**, shared by both sides (earlier bench-measured
  hardware limits were replaced by it), except `joint3`, whose export limits `[-15, 195]` were wrong
  about where zero sits and are now the bench-measured `[-105, 105]` -- same 210 deg of travel,
  centred on mid-travel where the hardware's zero actually is. `joint6` is narrower than the first
  export said. The right arm's README
  tabulates both units. `test_poses_are_within_the_joint_limits` keeps the posed sweep inside them,
  since they have already changed twice.
- Tests: now that it is an `ArmType`, the generic suites cover it —
  [test_urdf_mjcf_alignment.py](i2rt/robots/tests/test_urdf_mjcf_alignment.py) and its posed twin run
  on **both** sides, and `test_assembly.py` composes it against every gripper.
  [test_arm_7dof.py](i2rt/robots/tests/test_arm_7dof.py) keeps what they cannot express: the mirror
  relation, `HOME_WORLD_AXES`, the `0x08` gripper id, the per-arm mount offset, the gravity ceiling
  and the self-collision sweep. Its cross-side tests must stay **posed**, not just at home: at
  `q = 0` the two arms are reflections of each other whatever their axes say, so every mirror sign
  error hides there.

Provenance: ONShape document `a30d2ef2655863e671c75240`, exported by `onshape-to-robot`. Unlike this
arm's first export it arrives in the shape the pipeline expects — rooted at `base`, `link6` already
named positionally, meshes placed from one clean assembly origin, and the `base` link frame already
on the mounting face — so the re-rooting, the `link8` -> `link6` rename and the `base`/`link1` frame
swap that the first import needed are all gone. `normalize_onshape_urdf.py` still **skips** it (it
shows none of the "raw" signatures the script gates on, and `--force` does not override that branch),
but it needs nothing from that stage. `apply_urdf_heading.py` now **runs**: this export's base
visual, collision and inertial origins share one orientation, the precondition the first export
failed (`big_yam`, the repo's other `onshape-to-robot` export, still fails it). One heading is
applied, `R = Rz(-90 deg)*Ry(90 deg)` (`--rpy "0 1.5707963267948966 -1.5707963267948966"`), giving a
**plate mount**: the mounting face *is* the world x-z plane through the origin, normal +Y, `joint1`
turns about +Y, and the arm hangs along -Z at home. The frame itself is untouched (it stays the
identity world root); the rotation goes onto the base link's visual/collision/inertial origins and
the `base -> link1` joint origin together. Two things are then applied by hand and recorded in the
URDF's `i2rt-port` header: the four `<axis>` re-signings above, and `joint7`'s origin
right-multiplied by `Rz(180 deg)`, which leaves its axis *line* and therefore the kinematics untouched
while keeping the terminal `gripper` block — the canonical linear_4310 plus the split wrist, which is
not export data — in the frame it was derived in.

### CAN motor layer (`i2rt/motor_drivers/dm_driver.py`)

`DMChainCanInterface` owns the low-level Damiao-motor protocol: MIT-mode position/velocity/torque commands,
a background send/receive thread, absolute-position wrap-around tracking, and optional auto-recovery of
errored motors. `MotorChainRobot` sits on top of this. `motor_config_tool/` holds one-off maintenance CLIs
(`set_zero.py`, `set_timeout.py`, `ping_motors.py`).

**Startup register checks.** [motor_drivers/motor_check.py](i2rt/motor_drivers/motor_check.py) holds
both of them, as one function each, and `DMChainCanInterface.__init__` runs whichever the caller asked
for. **Both are opt-in** (`check_motor_types` / `check_motor_config`, each defaulting to `False`), and
both real callers — arms and the Flow Base — opt into both. `_motor_rows` drops a passive `motor_type: ""`
row (`no_gripper`, the teaching handles) before either check sees it — defensively, since no caller
builds such a row any more: a gripper-only teaching handle now takes the motorless path described above.

- `verify_motor_types` reads every motor's read-only `Gr` (gear-ratio) register and refuses to build a
  chain that does not hold the motor types its `motor_list` declares. Writes nothing. **Arms and the
  Flow Base both run it.**
- `verify_motor_config` compares `CTRL_MODE` — *repairing* it, written then verified then saved to
  Flash, unless the caller passed `repair=False` — and compares `PMAX`/`VMAX`/`TMAX` against
  `MotorType.get_motor_constants`, writing none of those three. It also compares, never writes, any
  `expected_registers` its caller declares — `get_yam_robot` passes the arm config's
  `motor_registers` (the 7-DOF arms' `KT_Value`/`GREF`), and any mismatch there refuses. **Arms and
  the Flow Base both run it too**, but an arm is the strictest possible caller:
  `MIT` puts `TMAX` in the blocking set and it names no `loop_critical_motor_ids`, so any of the three
  registers on any of its motors refuses the launch, where the base can only be refused over
  `PMAX`/`VMAX` on its four steering motors. `dm_driver.py --survey-only` surveys a chain's registers
  without building a robot around it, and is the **only** `repair=False` caller: a wrong `CTRL_MODE` is
  reported with the `dm_motor_registers.py write` command that fixes it and still refuses (non-zero
  exit), never written. **Without** that flag the same command repairs `CTRL_MODE` and then goes on to
  enable every motor and hold it at zero command. A survey still needs `--control-mode` — it is the
  value every motor is *required to hold*, and (without `--survey-only`) the one saved to Flash to get
  there, so `--check-motor-config` refuses to guess it: `MIT` for an arm, `VEL` for a base. On any arm
  that is not a YAM v1 it also needs a per-id `--motor-type` list positionally matching `--motor-id`.

Why the type check matters: the driver *encodes* every MIT command with the declared type's constants,
so a wrong declaration silently rescales torque — a `yam_ultra_2` arm run as `--arm yam_ultra` is 2.8x
on joint 4, with no error anywhere. Expected ratios come from `MotorType.get_gear_ratio`
([motor_drivers/utils.py](i2rt/motor_drivers/utils.py)); adding a motor type to a config without an
entry there is a CI failure, and the entry must come from a bench read, not the part number. `Gr` maps
many-to-one (10 is a `DM4310`, `DM4310V` *or* `DM_FLOW_WHEEL`), so it identifies a motor's class, never
which of a same-ratio pair is fitted — it cannot catch a steering/drive swap on the base.

Neither is called directly. `run_startup_checks` is the entry point, and it enforces the two rules that
have to hold for every caller rather than for whichever one remembered them: **types run before config**,
and asking for config alone is a `ValueError` — on the wrong part the scaling registers hold *that* part's
own scale, so a config check running first would report three consequent mismatches, tell the operator to
rewrite `PMAX`, and write `CTRL_MODE` to Flash on a motor whose repair is to be swapped; and **a check
that was asked for and cannot run refuses** rather than warning, because a non-socketcan channel that
started anyway would trust every declared type and scaling unverified. Both flags off is still a no-op on
any channel, which is what keeps an unchecked chain buildable on the `PCAN_USBBUS1` default.

On the chain-building path the call sits **inside the constructor**, after its asserts and before the
branch that opens the socket: everything above that line is pure Python, and register access needs an
**idle bus**, so there is no later window. `--survey-only` is the other call site and builds no chain at
all — it is also the one that passes `repair=False`, which is what makes "survey" true in both senses.
Sim robots reach neither: `get_robot` returns a `SimRobot` before building a chain, so the sim-only
test suite never opens a socket.

Two things `verify_motor_config` cannot derive and takes as arguments. The chain's **control mode**
gives it the `CTRL_MODE` to expect (`1` MIT / `2` pos-speed / `3` speed — `set_control` encodes only MIT
and VEL, so the CLI offers only those two) and decides whether a wrong
`TMAX` blocks: MIT encodes commanded torque through it, a `VEL` frame carries a raw float32 rad/s and
nothing else, so on the base `TMAX` is telemetry-only. **`loop_critical_motor_ids`** says whose feedback
the caller's loop acts on; omit it and every motor can block, which is what an arm wants, while
`flow_base_controller` passes its four `STEER_MOTOR_IDS` so a drive or rail motor's mis-scaled feedback
is logged and the base still starts. The base drives both flags from its one
`--no-verify-motor-config`, so that flag still skips everything it used to.

### Flow Base (`i2rt/flow_base/`)

A four-caster swerve mobile base (8 DM motors) plus an optional linear-rail lift. `flow_base_controller.py`
is the server (gamepad/RC/API teleop + swerve kinematics); `flow_base_client.py` is the network client
(`portal` RPC to `172.6.2.20`). See [i2rt/flow_base/README.md](i2rt/flow_base/README.md) — it is the
authoritative hardware + commissioning + API doc. Note: the base commissioning SOP references material that
lives in a separate internal repo, not here.

### robot_models: URDF is the source of truth

Arm models are **versioned by hardware revision**: each version dir (`i2rt/robot_models/arm/<arm>/v<N>/`) has
`<arm>.urdf`, `<arm>.xml` (MJCF), an `assets/` mesh dir, and a `README.md`. The file basenames are *not*
version-suffixed, so `<mujoco model="yam">` and the `assets/` relative mesh paths are identical across
versions. Prefer `ArmType.get_xml_path()`; the underlying resolver is `get_arm_xml_path(arm, version)` in
[i2rt/robot_models/__init__.py](i2rt/robot_models/__init__.py) — the `ARM_*_XML_PATH` constants remain as v1
aliases for downstream users. **The URDF is the kinematic/dynamic source of truth; the MJCF is generated from it.** The
per-arm README tabulates masses, COMs, inertia tensors, joint frames/axes/ranges, link lengths, home-pose
global frames, Modified-DH parameters, and space/body-frame PoE screw axes. A CI test keeps the URDF and MJCF
honest against each other (masses, COMs, inertia, frames, axes, ranges); the README tables themselves are
hand-maintained and are not parsed by any test.

## URDF ↔ MJCF alignment pipeline (this branch's focus)

The branch `fix/urdf-mjcf-alignment` is about aligning MJCF models with their URDF sources. Two Claude
**skills** cover this and should be preferred over ad-hoc edits — invoke `align-urdf-mjcf` or
`transform-onshape-urdf`. The pipeline is five staged scripts bundled with those skills — stages 1–3 in
[`.claude/skills/transform-onshape-urdf/scripts/`](.claude/skills/transform-onshape-urdf/scripts/), stages 4–5 in
[`.claude/skills/align-urdf-mjcf/scripts/`](.claude/skills/align-urdf-mjcf/scripts/) — split around a
**human visual-inspection checkpoint** (heading choice can't be inferred from CAD):

1. `normalize_onshape_urdf.py` — deterministic cleanup of a raw ONShape export (dedupe links, drop synthetic
   `root`, canonicalize joint/mesh names, bake root rotation `R0` into base mesh **and** inertia together).
2. `view_urdf.py` — load the normalized URDF in MuJoCo against world axes; the operator decides the heading.
3. `apply_urdf_heading.py` — bake the chosen heading rotation (not idempotent; previews unless `--apply`).
4. `urdf_to_arm_mjcf.py` — regenerate the arm-only MJCF (exactly `joint1..joint6`; the terminal `gripper` body
   is an empty mount placeholder; gripper/tip bodies excluded; URDF inertia → MuJoCo principal `quat` +
   `diaginertia`).
5. `sync_gripper_mounts.py` — re-write `last_joint_mount.<arm>` in every gripper config from the regenerated
   MJCF (because of the runtime overwrite described above). **Always run this after regenerating an arm MJCF.**

Shared math/FK helpers live in `.claude/skills/transform-onshape-urdf/scripts/urdf_align_lib.py`. URDF RPY convention throughout: `R = Rz(yaw) Ry(pitch) Rx(roll)`.

### Alignment tests

- [test_urdf_mjcf_alignment.py](i2rt/robots/tests/test_urdf_mjcf_alignment.py) — independently recomputes
  home-pose frames, joint axes/ranges, masses, COMs, and inertias from **both** the URDF and the MJCF and
  asserts they agree. This is what prevents the generated MJCF from drifting from the URDF; the per-arm README
  tables are hand-maintained and are not parsed by any test.
- [test_urdf_mjcf_posed_alignment.py](i2rt/robots/tests/test_urdf_mjcf_posed_alignment.py) — the same idea at
  non-home joint configurations.

`big_yam` is a special case in these tests: its MJCF models the fixed base as a static worldbody geom (no
`base` body) and uses an opposite mount/axis convention, so parts of the comparison exclude it. The terminal
`gripper` body is a mount frame with placeholder inertia (`mass=1e-6`), so only its frame/axis is checked, not
mass/COM/inertia.

## Testing notes

- The whole suite is **sim-only** — no CAN bus or motors required. `conftest.py` defines `sim`/`real` markers,
  but tests are not currently gated by `skipif`; CI just runs everything.
- Tests are heavily **parametrized over arm × gripper combinations** — a change to config or model geometry
  can ripple across many test IDs at once.
