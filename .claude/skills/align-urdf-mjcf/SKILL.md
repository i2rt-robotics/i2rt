---
name: align-urdf-mjcf
description: >-
  Align a MuJoCo MJCF robot model with a URDF source while preserving kinematics, geometry, masses,
  centers of mass, and inertia tensors. Runs a staged pipeline: normalize the raw ONShape URDF (fix
  mesh filenames, canonicalize/rename joints, remove the synthetic root) with
  normalize_onshape_urdf.py, visually inspect it against world axes with
  view_urdf.py, apply the user-specified heading rotation with apply_urdf_heading.py
  (base mesh AND inertia rotate together), regenerate the arm MJCF with urdf_to_arm_mjcf.py,
  then re-sync every gripper config's per-arm mount with sync_gripper_mounts.py. Use when
  comparing or synchronizing .urdf and .xml robot descriptions, aligning URDF link frames or mesh
  headings with world axes, changing a robot root or base orientation without moving downstream
  links unintentionally, fixing mesh paths or joint names, matching home-pose link frames and link
  lengths, converting URDF inertias to MuJoCo principal inertias, keeping an arm model at a
  specified DOF count without embedding gripper or tip bodies, or standardizing terminal mounts
  across linear, crank, flexible, no-gripper, and teaching-handle variants while preserving each end
  effector's physical placement.
---

# Align URDF and MJCF

## Goal

Treat the URDF as the source of truth unless the user states otherwise. Produce the smallest MJCF change that makes the requested arm scope numerically equivalent at the home pose and remains compatible with the repository's model-composition code.

## Pipeline (scripts)

The alignment runs as five stages. The first three prepare the URDF around a **mandatory human
inspection checkpoint**; the last two regenerate the MJCF and repair the gripper mounts that depend
on it. The URDF-normalization mechanics (stages 1 and 3) are documented in the
[`transform-onshape-urdf`](../transform-onshape-urdf/SKILL.md) skill.

```
raw ONShape export
  1. python .claude/skills/transform-onshape-urdf/scripts/normalize_onshape_urdf.py <model.urdf>   # deterministic cleanup, no heading
  2. python .claude/skills/transform-onshape-urdf/scripts/view_urdf.py <model.urdf>                # HUMAN CHECKPOINT: world axes
  3. python .claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py <model.urdf> --axis z --deg N --apply   # user's heading
  4. python .claude/skills/align-urdf-mjcf/scripts/urdf_to_arm_mjcf.py <model.urdf> <model.xml>     # URDF -> arm MJCF
  5. python .claude/skills/align-urdf-mjcf/scripts/sync_gripper_mounts.py <model.xml>              # re-sync last_joint_mount.<arm>
```

- Stages 1 and 3 were one script (`transform_onshape_urdf.py`); they were split so the heading
  rotation -- a human judgment -- happens only after the model is inspected, never as a default.
- Treat the URDF as the source of truth; regenerate the `.xml` from the fixed URDF, never the
  reverse.

## Establish the Contract

1. Read repository instructions and inspect the dirty worktree before editing.
2. Identify the source URDF, target MJCF, mesh directory, and any model-composition utilities or tests.
3. State the requested scope explicitly:
   - Which links and joints belong to the arm?
   - Does the terminal body represent a physical link or only an attachment frame?
   - Must gripper, finger, tip, tool, or top bodies be excluded?
   - How is “link length” defined? Default to the norm of the parent-to-child joint translation, but also require the full translation vector to match.
4. Preserve unrelated user changes and assets.
5. Create a short, verifiable task plan before editing.

For the YAM arm-only contract, retain exactly six arm joints named `joint1` through `joint6`. Keep `link6` as the end-effector mount body. Do not copy the URDF gripper mass, gripper geometry, or tip bodies into the arm MJCF.

## Inventory the Models

Use `rg`, `rg --files`, and an XML parser to collect:

- Link names, parent-child relationships, and joint types.
- Joint origins, axes, limits, and names.
- Visual origins, mesh filenames, scales, and colors.
- Masses, centers of mass, inertial-frame orientations, and inertia components.
- MJCF compiler settings, mesh assets, nested body transforms, inertials, joints, geoms, sites, and top-level equality/contact sections.
- Composition assumptions such as “append a gripper to the deepest body.”

Confirm actual mesh filenames on disk. Follow the repository's URI convention in URDF, such as `package://yam/...` when required. In MJCF, resolve the same files through `meshdir` or absolute paths used by the composition utility.

## Align the URDF with World Axes

Normalizing the URDF and correcting its heading is split into three stages with a required human
checkpoint between them. Never infer the forward direction from CAD coordinates or link names.

### Stage 1 — Normalize (deterministic; `normalize_onshape_urdf.py`)

Run `python .claude/skills/transform-onshape-urdf/scripts/normalize_onshape_urdf.py <model.urdf>`. This does only the cleanup that needs
no human judgment:

1. Keep the actuated `dof_joint*` chain and its links; drop the `分组`/`紧固` structural joints and
   the duplicate links; canonicalize names (`base_1` -> `base`, `link2_1` -> `link2`, ...). Every
   parent, child, transmission, control, and configuration reference is updated with the rename.
2. Resolve every mesh URI against the filesystem and match the real filename, capitalization, and
   extension; rewrite to `assets/<file>`. Do not invent or silently rename a mesh file.
3. Remove the synthetic root, **baking its rotation `R0` into the `base` visual AND inertial origins
   (the same `R0` -- mesh and inertia stay aligned) and the `base -> link1` joint origin** (which
   rigidly rotates the whole arm). This preserves the ONShape assembled pose so the model can be
   inspected in its natural orientation. The synthetic root is the *parent of the joint whose child
   is `base`*; if no joint has child `base`, the root is already gone and this step is skipped.
4. Rename `dof_jointN` -> `jointN`.

The script verifies rigidity (downstream joint origins unchanged), that meshes resolve, that the
tree is acyclic with a single `base` root, and that the base visual and inertial share one
orientation, then refuses to write if any check fails. It applies **no heading** -- that is stage 3.

### Stage 2 — Inspect (required checkpoint; `view_urdf.py`)

Run `python .claude/skills/transform-onshape-urdf/scripts/view_urdf.py <model.urdf>` (add `--screenshot out.png` on a headless host). It
opens the normalized model in MuJoCo with the world XYZ triad drawn (X red, Y green, Z blue). Ask
the user: **“Does the base face the intended robot-forward direction? If not, which world axis and
how many degrees of rotation are needed?”** This is a hard gate; do not choose a heading yourself.

### Stage 3 — Apply the heading (`apply_urdf_heading.py`)

With the axis + angle the user gives, run
`python .claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py <model.urdf> --axis z --deg N` to preview, then add `--apply`
to commit. Construct the requested rigid rotation `R` (normally a yaw about world Z) and:

1. Left-multiply `R` into the base visual and base inertial origins **by the same `R`**, so the
   base mesh and its center-of-mass / inertial frame continue to describe the same rotated physical
   base. The inertia tensor components stay unchanged (they live in the rotated inertial frame).
2. Left-multiply the same `R` into the `base -> link1` joint origin, which rigidly rotates every
   non-base home-pose link frame about the base origin. Leave downstream relative joint transforms
   unchanged. Because the origin is left-multiplied, this composes with stage 1's `R0` as
   `R @ R0` -- identical to a single combined rotation.
3. Keep each joint's local `axis` components unchanged when its entire joint frame rotates with the
   new origin; then verify the resulting world axis numerically. Re-express an axis only when the
   intended physical world axis is different. Never rotate both the joint frame and its local axis
   blindly.
4. Preserve masses, joint ranges, and parent-to-child translation norms unless the user requests
   another physical change.

The script verifies that downstream relative kinematics are unchanged, that joint world axes are
correct, and (as a hard guardrail) that the base visual and inertial orientations still match. It is
**not idempotent** -- running it again stacks another rotation -- so it previews by default and only
writes with `--apply`. Present the corrected visual heading (rerun `view_urdf.py`) for confirmation.

## Map URDF Semantics to MJCF

Use these equivalences:

| URDF | MJCF |
| --- | --- |
| Root link frame | Top-level body frame |
| Joint `origin` | Child body `pos` and `quat` |
| Joint `axis` | Joint `axis`, expressed in the joint/body frame |
| Link visual `origin` | Geom `pos` and `quat` |
| Link inertial `origin xyz` | Inertial `pos` |
| Link mass | Inertial `mass` |
| Inertia plus inertial RPY | Principal-axis `quat` plus `diaginertia` |

Represent the base as an explicit MJCF body when it has mass or inertia. A worldbody geom alone cannot represent the URDF base dynamics.

For an arm-only terminal mount:

- Give the mount body the URDF joint6 child-frame pose and joint6.
- Use a tiny valid placeholder inertia only when MuJoCo requires it, for example mass `1e-6` and diagonal inertia `1e-9 1e-9 1e-9`.
- Add no terminal mesh, gripper mass, finger joint, or tip body.
- Let the existing composition utility attach the selected external gripper.

Regenerate the arm MJCF with `python .claude/skills/align-urdf-mjcf/scripts/urdf_to_arm_mjcf.py <model.urdf> <model.xml>`.

## Preserve and Change Poses Correctly

Interpret URDF RPY as:

`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`

Interpret MuJoCo quaternions as `w x y z`. Compare rotations as matrices because `q` and `-q` represent the same rotation.

Compute home-pose global transforms recursively:

`T_world_child = T_world_parent @ T_parent_child`

Apply frame changes deliberately:

- To remove a synthetic root link while preserving every global pose, compose its transform into the new root's children, visuals, and inertials as required.
- To rotate only a base mesh, change the base geom/visual transform; do not rotate the base body frame.
- To rotate all descendant link frames while leaving the base frame fixed, apply the desired world rotation to the first child global transform, solve back for its new parent-relative transform, and leave downstream relative transforms unchanged when the same rigid rotation should affect the entire subtree.
- Recompute joint axes in their declared local frames after changing frame orientations. Do not guess an axis from its global appearance.

After every transform edit, recompute all global link poses at the home configuration and verify that only the requested frames moved.

## Match Geometry and Link Lengths

For each included link:

1. Copy the exact visual-to-link translation and rotation into the MJCF geom.
2. Point the MJCF mesh asset to the same physical file as the URDF mesh.
3. Preserve mesh scale and RGBA.
4. Match the complete parent-to-child joint translation vector.
5. Report link length as the norm of that vector, while treating vector equality as the stronger validation.

MuJoCo may recenter mesh vertices during compilation and store a mesh reference transform. Do not declare a geometry mismatch by comparing compiled `geom_xpos` directly with a URDF visual frame. First account for MuJoCo's mesh reference transform, or compare the source geom transform and underlying mesh file identity.

## Convert Mass and Inertia

Copy mass and center of mass exactly for each physical arm link.

URDF gives a symmetric inertia tensor in its inertial frame:

```text
I_urdf = [[ixx, ixy, ixz],
          [ixy, iyy, iyz],
          [ixz, iyz, izz]]
```

Rotate it into the link/body frame using the URDF inertial-origin rotation:

`I_body = R_inertial @ I_urdf @ R_inertial.T`

Convert `I_body` to MuJoCo principal form:

1. Run a symmetric eigendecomposition `eigenvalues, eigenvectors = eigh(I_body)`.
2. Ensure the eigenvector matrix is a proper rotation with determinant `+1`; flip one eigenvector if necessary.
3. Convert the eigenvector rotation matrix to a normalized MuJoCo `w x y z` quaternion.
4. Write the eigenvalues as `diaginertia`.
5. Reconstruct `I_check = R_quat @ diag(diaginertia) @ R_quat.T` and compare it with `I_body`.

Prefer high-precision `quat` plus `diaginertia` values. MuJoCo can diagonalize `fullinertia`, but that compilation step may introduce a larger reconstruction residual. Do not specify an inertial quaternion together with `fullinertia`.

## Build the MJCF Chain

Nest bodies in kinematic order:

```text
world
└── base
    └── link1 / joint1
        └── link2 / joint2
            └── ...
                └── link6 / joint6
```

Place each joint at `pos="0 0 0"` inside its child body when the body's transform already represents the URDF joint origin. Copy joint type, axis, range, and name exactly. Keep actuator-force metadata only if the existing MJCF convention requires it.

For YAM, expect six arm meshes: `base.stl` and `link1.stl` through `link5.stl`. Exclude `gripper.stl`, `tip_left.stl`, and `tip_right.stl` from the arm MJCF.

Whenever the arm MJCF is regenerated, immediately re-sync the gripper mounts (next section) --
`link6` may have moved, and the composition code trusts the config, not the MJCF.

## Cover All YAM End-Effector Variants

Use the complete YAM end-effector contract when changing shared arm mounts or composition code:

| Variant | Role | Compiled gripper joints | Required sites | Preserve |
| --- | --- | ---: | --- | --- |
| `linear_4310` | Active linear reference | 2 | `tcp_site`, `grasp_site` | DM4310 settings, mass, inertia, linear limiter |
| `linear_3507` | Active linear | 2 | `tcp_site`, `grasp_site` | DM3507 settings, target-specific mass, inertia, linear limiter |
| `crank_4310` | Active crank | 0 | `tcp_site`, `grasp_site` | Root offset, mass, inertia, crank limiter |
| `flexible_4310` | Active flexible linear | 2 | `tcp_site`, `grasp_site` | Root offset, geometry, dynamics, motor direction, linear limiter |
| `no_gripper` | Passive mount placeholder | 0 | `tcp_site`, `grasp_site` | No motor, no calibration, no limiter, six arm DOFs |
| `yam_teaching_handle` | Passive physical handle | 0 | `tcp_site` | Root offset, mass, inertia, no motor, six arm DOFs |

Count only compiled body joints; an equality-section `<joint>` element does not add a MuJoCo joint. Do not add joints, sites, motors, or limiter settings merely to make variants structurally identical.

Use `linear_4310` as the arm-side mount-frame reference for all four arm variants: `yam`, `yam_pro`, `yam_ultra`, and `big_yam`. Move each end-effector-specific rigid offset into its XML root body, but preserve every target-specific property listed above. Do not modify an unaffected variant unless the user requests it or a shared change requires it.

When a shared mount or composition change is in scope, compile the full four-arm by six-end-effector matrix. Verify the declared joint and site contract for each model, and separately verify the robot-interface DOF count because active crank actuation is not represented by a gripper MJCF joint.

## Standardize Interchangeable Gripper Mounts

Keep the arm's terminal joint frame independent of the selected gripper. `combine_arm_and_gripper_xml` (`i2rt/robots/utils.py`) finds the arm's deepest body (`link6`) and **overwrites** its `pos`, `quat`, and first joint's `axis` from the selected gripper config's `last_joint_mount.<arm>` block before attaching the gripper. Treat those fields as arm mount data, not gripper-specific offsets: the block is identical across every gripper config for a given arm.

### Sync gripper mounts after regenerating the arm MJCF

Because composition trusts the gripper config over the MJCF, a `last_joint_mount.<arm>` that is stale
relative to a freshly regenerated `<arm>.xml` does not sit harmlessly -- it actively relocates
`link6` (and can flip the joint axis, inverting the gripper motor direction), misplacing every
gripper on that arm. So after stage 4:

```bash
python .claude/skills/align-urdf-mjcf/scripts/sync_gripper_mounts.py i2rt/robot_models/arm/<arm>/<arm>.xml   # --dry-run to preview
```

It reads `link6`'s `pos`/`quat` and `joint6`'s `axis` from the arm MJCF -- using the same
`_find_deepest_body` composition uses, so the body it reads is the body composition overwrites --
and writes them into `last_joint_mount.<arm>` in every gripper config that has such a block
(`linear_4310`, `linear_3507`, `crank_4310`, `flexible_4310`, `no_gripper`, `yam_teaching_handle`).
It edits only the target arm's three scalar lines, leaving the other arms' blocks and all comments
intact, and skips a block that already matches within `1e-9` (so it is idempotent and produces a
clean diff).

### Rebasing a gripper mount frame (when moving offsets into gripper XML)

Use a known-correct gripper as the reference, such as `linear_4310`:

1. Read the old reference and target mount transforms for every supported arm variant.
2. Derive the target gripper's root transform in the reference mount frame:

   `T_target_root = inverse(T_reference_mount_old) @ T_target_mount_old`

3. Compare `T_target_root` across arm variants. If they agree within the precision of the original configs, encode one high-precision transform in the target gripper XML root body.
4. Copy each arm variant's reference `pos`, `quat`, and `axis` into the target gripper config. Preserve gripper motor polarity, gains, limits, calibration, and limiter settings.
5. Leave the target gripper's inertials, geoms, child-body transforms, joints, TCP site, and grasp site unchanged. The root-body transform carries the complete gripper rigidly.

After rebasing the reference mount itself, calculate the expected target placement as:

`T_target_expected = T_reference_mount_new @ inverse(T_reference_mount_old) @ T_target_mount_old`

The new composition must satisfy:

`T_target_actual = T_reference_mount_new @ T_target_root`

Compare these transforms as translation vectors and rotation matrices. Retain full calculated precision; do not replace a derived sub-micrometer translation or near-principal quaternion with a visually cleaner approximation unless it stays within the required tolerance.

If root transforms differ materially across arm variants, do not force them into one shared gripper XML. Determine whether the variants have genuinely different physical adapters or whether their source configs use inconsistent frames.

## Validate Numerically

Parse both source files and calculate maximum absolute errors for:

- Local body translation and rotation.
- Global home-pose body translation and rotation.
- Parent-to-child translation norm and complete translation vector.
- Joint axis and range.
- Mass and center of mass.
- Reconstructed body-frame inertia tensor.
- Local geom translation and rotation.
- Mesh file identity, scale, and RGBA.

Use tight tolerances appropriate to the source precision. A useful target for values copied from decimal XML is:

- Position: `< 1e-12 m`
- Rotation-matrix entry: `< 1e-12`
- Reconstructed inertia: `< 1e-15 kg·m²`

Compile the MJCF with the repository's supported MuJoCo version. For the standalone YAM arm-only model, verify:

- `nbody == 8`: world plus base and six link bodies.
- `njnt == 6`
- `nq == 6`
- `ngeom == 6`
- Body names are exactly `base`, `link1` through `link6`.
- Joint names begin with exactly `joint1` through `joint6`.
- No gripper or tip geom exists.

Then exercise the repository's arm/gripper composition function for every supported external gripper and compile every generated MJCF. Verify that the first six joints remain `joint1` through `joint6`. Do not infer robot-interface DOF count solely from MuJoCo `njnt`: some grippers add no MuJoCo joint, while coupled linear grippers may add two.

For a gripper-mount standardization:

- Compile every target gripper standalone.
- Assert that each target mount config equals the reference mount config for every arm variant.
- Compile the Cartesian product of affected arms and grippers.
- Verify the expected total joint count for each gripper model.
- Verify `joint1` through `joint6`, `tcp_site`, and `grasp_site` remain present.
- Reconstruct the compiled transform from the arm's last physical link to the gripper root and compare it with `T_target_expected`.
- Report the maximum source and compiled pose-preservation residuals. Interpret residuals against the precision of the original config quaternions.

Finish with:

```bash
git diff --check
git diff -- path/to/model.urdf path/to/model.xml
git status --short --untracked-files=all
```

## Report the Result

Lead with the achieved scope. Report:

- Which files changed.
- Whether the model remains arm-only and at the requested DOF count.
- Whether terminal mount bodies are physical or placeholders.
- Whether terminal mount configs were standardized and gripper-specific offsets moved into gripper XML roots.
- Maximum pose, length, mass, COM, and inertia errors.
- Maximum gripper pose-preservation error across all affected arm/gripper combinations.
- MuJoCo structure counts and compilation result.
- Composition-test results.
- Any pre-existing dirty files or untracked meshes left untouched.

## Avoid Common Errors

- Do not choose a base heading yourself; it is a required user decision made at the `view_urdf.py` checkpoint.
- Do not run `apply_urdf_heading.py` twice for one heading; it stacks rotations (previews by default, writes only with `--apply`).
- Do not regenerate an arm MJCF without re-running `sync_gripper_mounts.py`; a stale mount misplaces every gripper.
- Do not confuse a visual rotation with a link-frame rotation.
- Do not compare Euler angles directly when a rotation-matrix comparison is available.
- Do not mix quaternion conventions.
- Do not rotate an axis in world coordinates and write it as a local axis.
- Do not omit base mass by leaving the base as a world geom.
- Do not rotate the base mesh without applying the identical rotation to the base inertial.
- Do not copy gripper or tip dynamics into an arm-only MJCF.
- Do not replace a composition mount placeholder with a physical gripper body.
- Do not let arm joint6 pose or axis vary by gripper when the physical arm mount is the same.
- Do not move a gripper-specific offset into the XML root without updating every affected per-arm mount config.
- Do not standardize only one arm variant when the gripper XML is shared by several variants.
- Do not round a derived gripper root transform before validating pose preservation.
- Do not round principal inertias or quaternions prematurely.
- Do not treat MuJoCo mesh recentering as a source-transform mismatch.
- Do not modify unrelated formatting, assets, or robot models.
