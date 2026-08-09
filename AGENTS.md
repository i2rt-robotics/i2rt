# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
(both described below). `ArmType.family` / `ArmType.version` unpack a variant into the on-disk
`(family, v<N>)` pair that the config and model loaders resolve paths from. Registering a revision is a
**two-place edit** in [utils.py](i2rt/robots/utils.py): the `ArmType` member **and** its `(family, revision)`
entry in the `_ARM_VARIANTS` map. The two enums are the entry points:

- `ArmType` / `GripperType` enums live in [utils.py](i2rt/robots/utils.py) and map to:
  - a **hardware YAML** in [i2rt/robots/config/](i2rt/robots/config/) — motor IDs, directions, kp/kd,
    gravity-comp factors, gripper limits, force-torque maps. **Arm** configs are version-suffixed
    (`yam_v1.yml`, `yam_ultra_v1.yml`); **gripper** configs are not (`linear_4310.yml`).
  - a **robot model** (URDF + MJCF) under [i2rt/robot_models/](i2rt/robot_models/), resolved via the path
    constants in [i2rt/robot_models/__init__.py](i2rt/robot_models/__init__.py).
- **`combine_arm_and_gripper_xml`** (in [utils.py](i2rt/robots/utils.py)) merges the arm MJCF and gripper MJCF
  into one model at runtime (written to `/tmp/`). **Critical subtlety:** it locates the arm's deepest body
  (the `gripper` mount frame) and *overwrites* its `pos`/`quat` and `joint6`'s `axis` from the **gripper**
  config's `last_joint_mount.<arm>.v<N>` block. So the terminal-mount transform is physically arm data but is
  duplicated across every gripper config — a stale copy silently misplaces the gripper for that arm. The
  mount is keyed by arm **and** arm model version because a hardware revision can move the terminal frame;
  an un-versioned `last_joint_mount.<arm>` block (with `pos` directly under the arm key) is still accepted
  for backward compatibility. Seed a new version's blocks with
  `sync_gripper_mounts.py <arm>.xml --version N` rather than hand-editing all six configs.
- **Both bodies are named `gripper`** — the arm's terminal mount (matching the URDF's `joint6` child) and the
  gripper model's root — and MuJoCo body names must be unique, so composition *merges* them instead of nesting:
  the gripper's children move into the mount body under a `<frame>` carrying the gripper root's own
  `pos`/`quat`. Its `<inertial>` is the exception — MuJoCo accepts an `<inertial>` inside a `<frame>` but does
  **not** apply the frame transform to it, so that element is composed into the mount frame explicitly and
  replaces the mount's `mass=1e-6` placeholder.

Arm variants: `yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, `big_yam` (`yam_ultra_2` is the only
shipped revision beyond v1). Grippers: `crank_4310`, `linear_3507`, `linear_4310`,
`flexible_4310`, plus the non-gripper `yam_teaching_handle` and `no_gripper`. `ArmType.NO_ARM` gives a
gripper-only robot.

### CAN motor layer (`i2rt/motor_drivers/dm_driver.py`)

`DMChainCanInterface` owns the low-level Damiao-motor protocol: MIT-mode position/velocity/torque commands,
a background send/receive thread, absolute-position wrap-around tracking, and optional auto-recovery of
errored motors. `MotorChainRobot` sits on top of this. `motor_config_tool/` holds one-off maintenance CLIs
(`set_zero.py`, `set_timeout.py`, `ping_motors.py`).

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

The branch `fix/urdf-mjcf-alignment` is about aligning MJCF models with their URDF sources. Two Codex
**skills** cover this and should be preferred over ad-hoc edits — invoke `align-urdf-mjcf` or
`transform-onshape-urdf`. The pipeline is five staged scripts bundled with those skills — stages 1–3 in
[`.agents/skills/transform-onshape-urdf/scripts/`](.agents/skills/transform-onshape-urdf/scripts/), stages 4–5 in
[`.agents/skills/align-urdf-mjcf/scripts/`](.agents/skills/align-urdf-mjcf/scripts/) — split around a
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

Shared math/FK helpers live in `.agents/skills/transform-onshape-urdf/scripts/urdf_align_lib.py`. URDF RPY convention throughout: `R = Rz(yaw) Ry(pitch) Rx(roll)`.

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
