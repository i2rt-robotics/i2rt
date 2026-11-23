# YAM Teaching Handle v1 Physical Properties

This document records the physical properties of `yam_teaching_handle_v1`, the passive two-part pistol grip that replaces a gripper on the arm flange for leader-arm teleoperation. The shipped model is [`yam_teaching_handle.xml`](yam_teaching_handle.xml). The next revision, with a 38.2 mm shorter shell, is documented in [`../v2/README.md`](../v2/README.md).

Unlike the arms, the handle has no URDF in this repository. Its source is the ONShape export of the YAM + teaching-handle assembly ("YAM handle long"), whose arm chain is bit-identical to [`arm/yam/v1/yam.urdf`](../../../arm/yam/v1/yam.urdf) down to `link5`; that export's `link6` and `link7` are the handle shell and the trigger, and its `dof_joint6`, rotated into the flange's tool convention, is exactly the flange → handle transform tabulated below. The MJCF is therefore the source of truth here, and every value in this document is read from it.

## Scope and conventions

- The handle is two bodies: `gripper` (the shell, and the root) and `trigger` (the squeeze lever), joined by one passive hinge.
- The handle adds **no robot DOF**. The trigger hinge is passive, has no motor and no actuator, and is deliberately not named `joint7`; the robot stays at six DOFs.
- SI units are used: metres, kilograms, radians, and kg·m².
- Frames are given **relative to the arm flange** — the terminal `gripper` mount frame — which makes them arm-independent. The flange's own pose is arm- and revision-specific and lives in [`config/yam_teaching_handle_v1.yml`](../../../../robots/config/yam_teaching_handle_v1.yml); see [Composition and the arm mount](#composition-and-the-arm-mount).
- Frame orientations and joint ranges within floating-point tolerance of a multiple of π are written symbolically. Positions, translations, and axis components with magnitude below `1e-4` are shown as `0`; the model file keeps full precision (the trigger pivot's true x offset is `1e-05 m`, for instance).
- Off-diagonal inertia terms are shown as `0` only where they are CAD symmetry zeros — below `1e-6` of that body's largest moment. This deliberately differs from the arm READMEs' flat `1e-4` cut-off: the handle's entire tensor is of order `1e-3 kg·m²`, and a flat cut-off would erase real coupling terms.

## Summary

| Property | Value |
| --- | ---: |
| Bodies | 2 |
| Robot DOF added | 0 |
| Passive joints | 1 (trigger hinge) |
| Total mass | 0.257594 kg |
| Flange to trigger pivot | 0.1014822104 m |
| Flange to `tcp_site` | 0.2084198765 m |
| Collision geometry | none — both geoms are visual only |

## Body mass and inertial frames

The COM position and inertial-frame RPY are relative to the corresponding body frame.

| Body | Mass (kg) | COM xyz (m) | Inertial-frame RPY (rad) |
| --- | ---: | --- | --- |
| `gripper` (shell) | 0.252595 | `0.0018575 0.02106 -0.0802638` | `0 0 0` |
| `trigger` | 0.0049991 | `0 -0.000646924 -0.00511405` | `0 0 0` |
| **Total** | **0.257594** | — | — |

Both inertial frames are axis-aligned with their body frames, so each tensor below is expressed in the body frame, about that body's COM.

## Body inertia

```text
I = [[Ixx, Ixy, Ixz],
     [Ixy, Iyy, Iyz],
     [Ixz, Iyz, Izz]]
```

| Body | Ixx | Iyy | Izz | Ixy | Ixz | Iyz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gripper` | 0.00105352 | 0.000972567 | 0.000177474 | -1.23341e-05 | -2.68502e-06 | -1.12898e-05 |
| `trigger` | 6.5772e-07 | 7.0783e-07 | 2.30195e-07 | 0 | 0 | -5.49021e-08 |

MuJoCo stores those same tensors in principal-axis form — a rotation from the body frame onto the principal axes, plus the three eigenvalues in ascending order:

| Body | Inertial `quat` (w x y z) | `diaginertia` (kg·m²) |
| --- | --- | --- |
| `gripper` | `0.7059392426719 -0.057289184102795 -0.704380485114871 -0.0470729989484305` | `0.000177304388462799 0.000970897959001009 0.00105535865253619` |
| `trigger` | `0.527433056094875 0.470971730933418 0.470971730933418 -0.527433056094875` | `2.23965486366579e-07 6.5772e-07 7.14059513633421e-07` |

## Frames

| Frame | Parent | pos (m) | quat (w x y z) | RPY (rad) |
| --- | --- | --- | --- | --- |
| `gripper` (handle root) | arm flange | `0 0 0` | `0.38268 0.92388 0 0` | `3π/4 0 0` |
| `trigger` | `gripper` | `0 0.0533184 -0.0863469` | `1 0 0 0` | `0 0 0` |
| `tcp_site` | `gripper` | `-0.000490393 0.0231556 -0.207129` | `0.2705980500730985 -0.6532814824381883 0.6532814824381882 0.2705980500730985` | `-π π/4 -π/2` |

The handle root coincides with the flange: the transform is a pure rotation, `Rx(+135°)`. The export seats the handle at `Rz(+90°) Rx(-45°)` from `joint6`'s child frame — the exact intent behind its `rpy "-0.785398 0 1.5708"`, since ONShape truncates RPY to six significant digits, which would otherwise tilt the whole handle by `2.1e-4 °` and split two quaternion components that are equal by symmetry. That child frame is no longer the flange: the flange now carries the **`+X` right, `+Y` down, `+Z` forward** tool convention, which is `R_fix` (180° about `(1,1,0)/√2`) away from the export's, and `R_fix · Rz(+90°) Rx(-45°)` collapses to a single rotation about x. The translation is zero either way, since `R_fix` fixes the origin.

`tcp_site` sits at the shell's furthest point along the approach axis (flange `+Z`), the same way every gripper's `grasp_site` sits at its fingertips: the one mesh vertex with the largest flange `z`, on the front of the butt, 208.4 mm from the flange. It is on the shell body, not the trigger, so squeezing does not move it. Only its *position* is handle-specific; its orientation is the shared end-effector convention that `test_robot_variants`' grasp-site tests enforce, and those tests check axes only, never position.

| `tcp_site` axis | In the flange frame | In the world frame at home |
| --- | --- | --- |
| X | `0 1 0` | `0 0 -1` (down) |
| Y | `-1 0 0` | `0 1 0` (horizontal) |
| Z | `0 0 1` | `1 0 0` (horizontal) |

So `tcp_site` is the flange frame turned `+90°` about the flange's own `z`: the two share an approach axis and differ only in the roll about it.

Both columns hold for every arm the handle mounts on — `yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, and `big_yam`. `big_yam`'s gripper configs used to roll its mount `+90°` about `z` and flip `joint6`'s sign relative to its URDF; that divergence has since been folded into its own URDF and MJCF, so all five arms now present the same flange.

## Trigger joint

| Property | Value |
| --- | --- |
| Joint name | `gripper` |
| Type | hinge, passive (no motor, no actuator) |
| Body | `trigger` |
| Origin | `0 0 0` (the `trigger` body origin) |
| Axis | `1 0 0` |
| Range | `[-0.680678, 0]` rad = `[-39, 0]` deg |

The joint is named `gripper` — the repo's name for a normalized gripper command — and shares that name with the root body, which MuJoCo allows because body and joint names live in separate namespaces. It is deliberately **not** named `joint7`: `get_robot._load_joint_limits_from_xml` picks up every joint whose name starts with `joint`, and the handle robot must stay at six DOFs.

The range is oriented so that the repo's standard normalized mapping `q = lo + n·(hi − lo)`, for a command `n` in `[0, 1]`, lands the right way round for the "0 = closed, 1 = open" convention in `sim_robot._cmd_to_mj_qpos`:

| `n` | `q` (rad) | `q` (deg) | Trigger state |
| ---: | ---: | ---: | --- |
| 0 | -0.680678 | -39 | fully squeezed — CLOSED |
| 1 | 0 | 0 | at rest — OPEN |

That mapping reduces to `q = -0.680678 · n` exactly, so a single write both matches the arm convention and makes the trigger follow the passive encoder.

Two consequences worth remembering. The hinge needs no actuator: MuJoCo's viewer drives unactuated hinges from its Joint panel, which writes `qpos` directly, and composition drops `<actuator>`, `<keyframe>`, and `<default>` anyway. And because this is the only gripper joint that is a **hinge rather than a slide**, code must not infer an arm's DOF count from the model's hinge count.

## Composition and the arm mount

`combine_arm_and_gripper_xml` merges this model into the arm at runtime. The arm's terminal mount body and this model's root body are both named `gripper`, and MuJoCo body names must be unique, so composition *merges* rather than nests: the root body's children move into the arm's mount body under a `<frame>` carrying the root's own `pos`/`quat`, and the root `<inertial>` is composed in explicitly — MuJoCo accepts an `<inertial>` inside a `<frame>` but does **not** apply the frame transform to it — replacing the mount's `mass="1e-6"` placeholder.

Only `<asset>`, the root body, `<equality>`, and `<contact>` survive that merge, which is why every attribute in the model file is written inline.

Composition also *overwrites* the arm mount body's `pos`, `quat`, and `joint6` `axis` from `last_joint_mount.<arm>.v<N>` in [`config/yam_teaching_handle_v1.yml`](../../../../robots/config/yam_teaching_handle_v1.yml). That block is physically arm data duplicated into every gripper config, so it must be re-synced with `sync_gripper_mounts.py` whenever an arm MJCF is regenerated. It does not affect any frame in this document, all of which are flange-relative.

## Meshes

| Mesh | Body | Triangles | Colour (RGBA) |
| --- | --- | ---: | --- |
| [`assets/handle.stl`](assets/handle.stl) | `gripper` | 8000 | `0.301961 0.427451 0.701961 1` |
| [`assets/trigger.stl`](assets/trigger.stl) | `trigger` | 8000 | `0.94902 0.94902 0.94902 1` |

Both geoms are `contype="0" conaffinity="0"` — visual only, matching the previous geom-less model and the gripper-base convention. Nothing here is meant to grasp.

## Mass provenance

The masses and inertias above are the ONShape export's verbatim values, from a re-export that carries real material assignments. An earlier export of the same assembly had no material and reported mesh volumes at `rho = 1 kg/m³` instead, which made the model weigh 0.158 g in total; the handle then carried a hand-set 0.258 kg to compensate.

The two parts turn out to have **different** densities — the shell's figures scale by ×1654.20 against those volumes and the trigger's by ×973.80 — so no single density relates the two sets. The geometry did not change between exports: both COMs are bit-identical, both mesh origins are identical, and the principal-axis `quat`s above still diagonalize the new tensors to `2e-8` relative, since a uniform density change scales an inertia tensor's eigenvalues but leaves its eigenvectors alone. Only `mass` and `diaginertia` were rewritten.

The 0.257594 kg total lands on the 0.258 kg that had been hand-set, which is an independent check on the export.

This is real load on the flange, and it is live: `MotorChainRobot` hands the composed model to `MuJoCoKDL`, so gravity compensation now sees the handle. Against the old placeholder figures the compensation torques move by up to **1.06 N·m** on joint 3 at the home pose. Handles that differ from stock — older 3D-printed shells in particular — can be corrected at run time with `--ee-mass`, documented in [`examples/minimum_gello`](../../../../../examples/minimum_gello/README.md#overriding-handle-weight); that override sets the shell body's mass only and does not rescale the inertia tensor.
