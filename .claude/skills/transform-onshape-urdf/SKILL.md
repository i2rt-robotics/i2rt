---
name: transform-onshape-urdf
description: >-
  Normalize a raw ONShape-exported YAM-family URDF into the aligned, world-referenced form used
  across i2rt robot models, as a two-stage pipeline around a human inspection checkpoint. Use when
  importing a fresh ONShape URDF export (robot named "urdf_top_assembly", package:// mesh paths, a
  synthetic "root" link, dof_joint* names, and duplicate *_1 links wired by 分组/紧固 fixed joints);
  when removing a synthetic root link and baking its rotation into the base while keeping every
  downstream link in place; when applying a world-axis heading correction that rotates the base mesh
  AND inertia together and rigidly rotates the whole arm; when fixing mesh filename case/paths; or
  when re-running such a transform idempotently. Runs normalize_onshape_urdf.py (stage 1) and
  apply_urdf_heading.py (stage 3), with view_urdf.py as the checkpoint in between.
---

# Transform ONShape URDF → aligned model

Turns a raw ONShape YAM-family export into the clean form established by `yam.urdf`
(commit `8975c25`). This was one script (`transform_onshape_urdf.py`); it is now **two**, split
around a mandatory visual-inspection checkpoint so the heading rotation -- a human judgment -- is
never applied as a default guess:

- [`scripts/normalize_onshape_urdf.py`](scripts/normalize_onshape_urdf.py) — stage 1,
  deterministic cleanup (meshes, joint names, remove root, bake `R0`). No heading.
- [`scripts/view_urdf.py`](scripts/view_urdf.py) — stage 2, the checkpoint: render the
  normalized model in MuJoCo with world axes and ask the user which heading correction is needed.
- [`scripts/apply_urdf_heading.py`](scripts/apply_urdf_heading.py) — stage 3, apply the
  user's rotation to the base (mesh + inertia together) and the arm.

The shared math/XML helpers live in
[`scripts/urdf_align_lib.py`](scripts/urdf_align_lib.py).

The downstream URDF→MJCF alignment (regenerate the `.xml`, then re-sync gripper mounts) is owned by
[`.claude/skills/align-urdf-mjcf/SKILL.md`](../align-urdf-mjcf/SKILL.md). Treat the URDF as the
source of truth; regenerate the `.xml` (MJCF) from the fixed URDF.

## When to use

- A fresh ONShape export needs to become a usable URDF: `<robot name="urdf_top_assembly">`,
  `package://.../meshes/Name.stl` paths, a `root` link above `base`, `dof_joint0..N` joints,
  and duplicate `link*_1` / `分组` / `紧固` structural joints.
- You need to drop a synthetic root and re-home the base heading without disturbing the arm.
- You need to re-run normalization safely (it must be a no-op the second time).

## Run it

```bash
# Stage 1 — normalize (deterministic; edits in place):
python .claude/skills/transform-onshape-urdf/scripts/normalize_onshape_urdf.py <path/to/model.urdf> \
    [--assets-dir assets] [--name NAME] [--dry-run] [--force]

# Stage 2 — inspect (required checkpoint):
python .claude/skills/transform-onshape-urdf/scripts/view_urdf.py <path/to/model.urdf>   # or --screenshot out.png (headless)

# Stage 3 — apply the user's heading (previews by default; --apply writes):
python .claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py <path/to/model.urdf> --axis z --deg <N> [--apply]
```

- `normalize` `--dry-run` prints the verification report and the resulting `base` + `joint1`,
  writing nothing. Always dry-run first.
- `apply_urdf_heading` writes only with `--apply`; without it, it previews. Different YAM arms need
  different headings (each export's baked `R0` differs) — inspect and confirm visually. Common case
  is a yaw about world Z; `yam_pro` needed 0°.

## Stage 1 — the transform (what `normalize_onshape_urdf.py` does)

RPY convention (URDF): `R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`.

1. **Keep only the actuated chain.** The real kinematic tree is the `dof_joint*` joints. Drop
   every other joint (`分组`/`紧固`/fastened duplicates) and every link not on that chain.
   Rename the survivors to canonical names by stripping a trailing `_<n>`
   (`base_1`→`base`, `link2_1`→`link2`, `gripper_1`→`gripper`, `tip_right_1`→`tip_right`).
2. **Fix meshes.** Rewrite each reference to `assets/<file>`, preferring an exact
   case-insensitive match on disk, else the suffix-stripped lowercase name
   (`Gripper.stl`→`gripper.stl`, `link5_1.stl`→`link5.stl`).
3. **Remove the synthetic root, baking `R0`.** The synthetic root is the *parent of the joint whose
   child is `base`* (in a raw export that is `dof_joint0`, `root -> base`). Capture its rotation
   `R0 = Rz(yaw)Ry(pitch)Rx(roll)`, delete the `root` link and that joint, and left-multiply `R0`
   into the `base` **visual** origin, the `base` **inertial** origin (position + rpy), and the
   `base -> link1` joint origin. The inertia tensor components and every joint's local `axis` are
   unchanged; joints 2..N are untouched, so the arm rotates rigidly. Rotating the base mesh and its
   inertia by the *same* `R0` keeps them consistent. **Guard:** if no joint has child `base`, the
   root is already gone and this step is skipped — it never re-bakes and never mistakes a rootless
   `base` for a synthetic root.
4. **Rename joints** `dof_jointN` → `jointN`; set the robot name.
5. **Insert an `onshape-normalize:` marker** recording `R0` and that a heading is still pending.

**No heading is applied here.** That is stage 3, gated by the checkpoint.

## Stage 3 — the heading (what `apply_urdf_heading.py` does)

Construct the requested rotation `R` (`--axis`/`--deg`, or `--rpy "r p y"`) and left-multiply it
into the `base` visual origin, the `base` inertial origin (**the same `R`** — mesh and inertia never
diverge), and the `base -> link1` joint origin (rigidly rotating the whole arm). Because origins are
left-multiplied, this composes with stage 1's `R0` as `R @ R0` — identical to the old one-shot
transform's combined `--heading-deg`. Inertia components and downstream joints are untouched. An
`onshape-heading:` marker is appended. The op is **not idempotent** (re-running stacks another
rotation), so it previews by default and writes only with `--apply`.

## Idempotency contract (stage 1)

Before doing anything, `normalize_onshape_urdf.py` classifies the file structurally:

- **`is_raw`**: a `root` link, `dof_joint*` names, `package://` paths, non-arm (`分组`/`紧固`)
  joints, or `*_<n>` duplicate links. A non-raw file has nothing to normalize (each step is a
  no-op), so it is skipped.
- **`has_marker`**: the `onshape-normalize:` comment is present.

Decision: not raw → skip (already normalized). Raw + marker + no `--force` → skip. Raw and
(`--force` or no marker) → normalize, then write the marker. Every step is individually idempotent,
so re-running is safe.

## Verify (the scripts refuse to write on failure)

Both stage scripts verify before writing and print a report:

- **Rigidity**: parent→child origins for every joint below `base -> link1` are unchanged (the arm
  is a pure rigid rotation).
- **Joint world axes** are recomputed and printed; the FK also proves the tree is acyclic with a
  single `base` root.
- **Base mesh & inertia aligned**: the base visual and inertial orientations must match (the mesh
  and inertia were rotated by the identical `R`).
- **Meshes** all resolve on disk.

After writing, additionally:

```bash
git diff -- <model.urdf>        # base visual+inertial + joint1 rotated; root/dof_joint0 gone;
                                # joints renamed; mesh case fixed; other links byte-identical
```

Then **inspect** with `python .claude/skills/transform-onshape-urdf/scripts/view_urdf.py <model.urdf>` and confirm the arm assembles and
the base faces the intended forward direction — the required heading checkpoint from
`align-urdf-mjcf`. Finally regenerate the MJCF from the URDF (and re-sync gripper mounts — see
`align-urdf-mjcf`).

## Assumptions / limits

- ONShape naming conventions: the actuated chain uses `dof_joint*`; structural duplicates use
  `分组`/`紧固`; per-instance suffixes are `_<n>`. Verified end-to-end on `yam_pro`.
- The heading is a single rotation about one world axis (or an explicit `--rpy`). If a model needs a
  different base orientation, pass a different `--axis`/`--deg` and confirm by rendering.
