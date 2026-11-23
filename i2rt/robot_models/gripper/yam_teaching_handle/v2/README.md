# YAM Teaching Handle v2 Physical Properties

This document records the physical properties of `yam_teaching_handle_v2`, revision v2 of the passive two-part pistol grip that replaces a gripper on the arm flange for leader-arm teleoperation. The shipped model is [`yam_teaching_handle.xml`](yam_teaching_handle.xml).

Like v1, this revision has no URDF in this repository. Its source is the ONShape export of the YAM + short-teaching-handle assembly ("YAM  handle short"), whose `link1`–`link5` masses and inertia tensors are bit-identical to [`arm/yam/v1/yam.urdf`](../../../arm/yam/v1/yam.urdf); that export's `link6` and `link7` are the handle shell and the trigger, and its `dof_joint6`, rotated into the flange's tool convention, is exactly the flange → handle transform tabulated below. The MJCF is therefore the source of truth here, and every value in this document is read from it.

The previous revision is documented in [`../v1/README.md`](../v1/README.md). The two share every convention; [Revision v2 versus v1](#revision-v2-versus-v1) lists everything that actually changes.

## Scope and conventions

- The handle is two bodies: `gripper` (the shell, and the root) and `trigger` (the squeeze lever), joined by one passive hinge.
- The handle adds **no robot DOF**. The trigger hinge is passive, has no motor and no actuator, and is deliberately not named `joint7`; the robot stays at six DOFs.
- SI units are used: metres, kilograms, radians, and kg·m².
- Frames are given **relative to the arm flange** — the terminal `gripper` mount frame — which makes them arm-independent. The flange's own pose is arm- and revision-specific and lives in [`config/yam_teaching_handle_v2.yml`](../../../../robots/config/yam_teaching_handle_v2.yml); see [Composition and the arm mount](#composition-and-the-arm-mount).
- Frame orientations and joint ranges within floating-point tolerance of a multiple of π are written symbolically. Positions, translations, and axis components with magnitude below `1e-4` are shown as `0`; the model file keeps full precision (the trigger pivot's true x offset is `1e-05 m`, for instance).
- Off-diagonal inertia terms are shown as `0` only where they are CAD symmetry zeros — below `1e-6` of that body's largest moment. This deliberately differs from the arm READMEs' flat `1e-4` cut-off: the handle's entire tensor is of order `1e-3 kg·m²`, and a flat cut-off would erase real coupling terms.

## Summary

| Property | Value |
| --- | ---: |
| Bodies | 2 |
| Robot DOF added | 0 |
| Passive joints | 1 (trigger hinge) |
| Total mass | 0.24359111 kg |
| Flange to trigger pivot | 0.1019915745 m |
| Flange to `tcp_site` | 0.1759575582 m |
| Collision geometry | none — both geoms are visual only |

## Revision v2 versus v1

Only four things change. Everything else in this document — the root rotation, the `tcp_site` definition and orientation, the trigger joint's name, axis and range, the composition rules — is identical to v1.

| | v1 | v2 | Change |
| --- | ---: | ---: | ---: |
| Shell mass | 0.252595 kg | 0.238592 kg | −14.003 g |
| Total mass | 0.2575941 kg | 0.24359111 kg | −14.003 g |
| Shell COM, z (handle frame) | −0.0802638 m | −0.0630398 m | 17.224 mm toward the flange |
| Shell extent, min z (handle frame) | −214.452 mm | −176.247 mm | **38.205 mm shorter** |
| Trigger pivot reach | 0.1014822104 m | 0.1019915745 m | 0.5094 mm further out |
| `tcp_site` reach | 0.2084198765 m | 0.1759575582 m | 32.462 mm closer, following the butt |
| Trigger rest orientation | — | — | **re-clocked +16° about the hinge axis** |

**The shell is truncated at the butt, not rescaled.** Its bounding box is identical in x and y (79.13 × 99.11 mm) and identical at the flange end (`max z = 26.438 mm` in both); the entire 38.205 mm difference is at the far end. The grip cross-section, the flange interface and the trigger geometry are untouched.

**The trigger is the same physical part**, rotated `+16°` about the hinge axis at rest. Three independent measurements agree: its mass matches v1's to five significant digits (0.00499911 vs 0.0049991 kg); solving `Rx(θ) I Rxᵀ = I'` on the two inertia tensors gives `θ = +15.9998°` and maps the v1 COM onto the v2 one to 0.04 µm; and rotating the v1 trigger's raw 34 710-triangle vertex cloud by `Rx(+16°)` about the pivot reproduces this export's bounding box to 0.006 mm.

That symmetry has a consequence for the tables below. The trigger's principal-axis `quat` satisfies `x == y` and `w == −z` exactly, and a rotation about x preserves that structure — if `q = (w, x, x, −w)` and `q_x = (c, s, 0, 0)`, then `q_x ⊗ q = (cw − sx, cx + sw, cx + sw, −(cw − sx))`. So both handles' trigger quats have the same form, with different values.

**Gravity compensation differs by up to 0.088 N·m.** Swapping a v1 handle for a v2 moves the compensation torques on joints 2 and 3 by 0.088 N·m at the home pose and 0.034 N·m on joint 4 with the arm out front. That is small but not nothing: `MotorChainRobot` hands the composed model to `MuJoCoKDL`, so the model you select is the model the arm compensates for.

## Body mass and inertial frames

The COM position and inertial-frame RPY are relative to the corresponding body frame.

| Body | Mass (kg) | COM xyz (m) | Inertial-frame RPY (rad) |
| --- | ---: | --- | --- |
| `gripper` (shell) | 0.238592 | `0.00222058 0.0206182 -0.0630398` | `0 0 0` |
| `trigger` | 0.00499911 | `0 0.000787745 -0.0050943` | `0 0 0` |
| **Total** | **0.24359111** | — | — |

Both inertial frames are axis-aligned with their body frames, so each tensor below is expressed in the body frame, about that body's COM.

## Body inertia

```text
I = [[Ixx, Ixy, Ixz],
     [Ixy, Iyy, Iyz],
     [Ixz, Iyz, Izz]]
```

| Body | Ixx | Iyy | Izz | Ixy | Ixz | Iyz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gripper` | 0.000757284 | 0.000679585 | 0.00016895 | -1.56247e-05 | 7.92763e-06 | 5.10534e-05 |
| `trigger` | 6.57726e-07 | 7.00643e-07 | 2.3739e-07 | 0 | 0 | 7.99927e-08 |

MuJoCo stores those same tensors in principal-axis form — a rotation from the body frame onto the principal axes, plus the three eigenvalues in ascending order:

| Body | Inertial `quat` (w x y z) | `diaginertia` (kg·m²) |
| --- | --- | --- |
| `gripper` | `0.700697371576107 -0.0328033925254197 -0.705130845057546 -0.103622498780861` | `0.000163745821426384 0.000681758302639947 0.000760314875933669` |
| `trigger` | `0.456755099233804 0.539791422054777 0.539791422054777 -0.456755099233804` | `2.23966163220921e-07 6.57726e-07 7.14066836779079e-07` |

The trigger's two zeroed off-diagonals are what make its `quat` exactly symmetric. The export reports `Ixy = 3.4e-14` and `Ixz = -1.1e-13` against a largest moment of `7.0e-07` — ratios of `5e-08` and `2e-07`, both under the `1e-6` symmetry cut-off — so x is exactly a principal axis. That is why `diaginertia`'s middle entry is exactly the export's `Ixx`, and why the remaining pair comes from the 2 × 2 `yz` block alone. Reconstructing the full export tensor from this principal form is accurate to `1.5e-07` relative; the discrepancy *is* those two zeroed terms. The shell's off-diagonals are real coupling and are kept.

## Frames

| Frame | Parent | pos (m) | quat (w x y z) | RPY (rad) |
| --- | --- | --- | --- | --- |
| `gripper` (handle root) | arm flange | `0 0 0` | `0.38268 0.92388 0 0` | `3π/4 0 0` |
| `trigger` | `gripper` | `0 0.0542396 -0.0863733` | `1 0 0 0` | `0 0 0` |
| `tcp_site` | `gripper` | `-0.000241398 0.0187737 -0.174953` | `0.2705980500730985 -0.6532814824381883 0.6532814824381882 0.2705980500730985` | `-π π/4 -π/2` |

The handle root coincides with the flange: the transform is a pure rotation, `Rx(+135°)`. The export seats the handle at `Rz(+90°) Rx(-45°)` from `joint6`'s child frame — the exact intent behind its `rpy "-0.785398 0 1.5708"`, since ONShape truncates RPY to six significant digits, which would otherwise tilt the whole handle by `2.1e-4 °` and split two quaternion components that are equal by symmetry. That child frame is no longer the flange: the flange now carries the **`+X` right, `+Y` down, `+Z` forward** tool convention, which is `R_fix` (180° about `(1,1,0)/√2`) away from the export's, and `R_fix · Rz(+90°) Rx(-45°)` collapses to a single rotation about x. The translation is zero either way, since `R_fix` fixes the origin.

This root transform is **bit-identical to v1's**. Both exports place `dof_joint6` at the same xyz as `yam.urdf`'s `joint6` (`0.0404996 0 -0.0356`), with zero relative translation, so in both cases the flange → root transform is a pure rotation with the same value. Each export's own `dof_joint6` rotation is `Rz(+90°) Rx(-45°)`, and `joint6` now carries `R_fix` in its `rpy`; composing the two gives the `Rx(+135°)` tabulated above.

`tcp_site` sits at the shell's furthest point along the approach axis (flange `+Z`), the same way every gripper's `grasp_site` sits at its fingertips: the one mesh vertex with the largest flange `z`, on the front of the butt, 176.0 mm from the flange. The truncated butt pulls it 32.5 mm closer than v1's. It is on the shell body, not the trigger, so squeezing does not move it. Only its *position* is handle-specific; its orientation is the shared end-effector convention that `test_robot_variants`' grasp-site tests enforce, and those tests check axes only, never position.

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

Every entry is identical to v1's. The joint is named `gripper` — the repo's name for a normalized gripper command — and shares that name with the root body, which MuJoCo allows because body and joint names live in separate namespaces. It is deliberately **not** named `joint7`: `get_robot._load_joint_limits_from_xml` picks up every joint whose name starts with `joint`, and the handle robot must stay at six DOFs.

The range is oriented so that the repo's standard normalized mapping `q = lo + n·(hi − lo)`, for a command `n` in `[0, 1]`, lands the right way round for the "0 = closed, 1 = open" convention in `sim_robot._cmd_to_mj_qpos`:

| `n` | `q` (rad) | `q` (deg) | Trigger state |
| ---: | ---: | ---: | --- |
| 0 | -0.680678 | -39 | fully squeezed — CLOSED |
| 1 | 0 | 0 | at rest — OPEN |

That mapping reduces to `q = -0.680678 · n` exactly, so a single write both matches the arm convention and makes the trigger follow the passive encoder. Note that the +16° re-clocking changes where the lever *sits*, not how far it *travels*: the swept angle is 39° on both handles.

Two consequences worth remembering. The hinge needs no actuator: MuJoCo's viewer drives unactuated hinges from its Joint panel, which writes `qpos` directly, and composition drops `<actuator>`, `<keyframe>`, and `<default>` anyway. And because this is a **hinge rather than a slide**, code must not infer an arm's DOF count from the model's hinge count.

## Composition and the arm mount

`combine_arm_and_gripper_xml` merges this model into the arm at runtime. The arm's terminal mount body and this model's root body are both named `gripper`, and MuJoCo body names must be unique, so composition *merges* rather than nests: the root body's children move into the arm's mount body under a `<frame>` carrying the root's own `pos`/`quat`, and the root `<inertial>` is composed in explicitly — MuJoCo accepts an `<inertial>` inside a `<frame>` but does **not** apply the frame transform to it — replacing the mount's `mass="1e-6"` placeholder.

Only `<asset>`, the root body, `<equality>`, and `<contact>` survive that merge, which is why every attribute in the model file is written inline.

Composition also *overwrites* the arm mount body's `pos`, `quat`, and `joint6` `axis` from `last_joint_mount.<arm>.v<N>` in [`config/yam_teaching_handle_v2.yml`](../../../../robots/config/yam_teaching_handle_v2.yml). That block is physically arm data duplicated into every gripper config — byte-identical across all seven of them — so it must be re-synced with `sync_gripper_mounts.py` whenever an arm MJCF is regenerated. It does not affect any frame in this document, all of which are flange-relative.

## Meshes

| Mesh | Body | Triangles | Colour (RGBA) |
| --- | --- | ---: | --- |
| [`assets/handle.stl`](assets/handle.stl) | `gripper` | 8000 | `1 0.6 0 1` |
| [`assets/trigger.stl`](assets/trigger.stl) | `trigger` | 8000 | `0.764706 0.878431 0.945098 1` |

Colours are this export's own ONShape materials, which differ from v1's — the v2 shell renders orange where v1's is blue. That is a CAD appearance assignment, not a property of the plastic.

Both geoms are `contype="0" conaffinity="0"` — visual only, matching v1 and the gripper-base convention. Nothing here is meant to grasp.

Both meshes were decimated from the raw export with [`scripts/simplify_mesh.py`](../../../scripts/simplify_mesh.py) at the repo's standard 8000-face budget. The v2 export is tessellated far more finely than v1's, so the shell's reduction is 74× (596 292 → 8000 faces) against v1's 19×; its bounding box grows 0.28 mm as a result, or 0.14% of the 203 mm length. The trigger's reduction is 23× (181 078 → 8000) and its bounding box moves 0.003 mm. Both geoms are visual-only, so neither figure has any physical consequence.

## Mass provenance

The masses and inertias above are the ONShape export's verbatim values; the assembly carries real material assignments, so no density correction is applied. Unlike v1, this revision never shipped with unit-density placeholder figures, so there is no earlier revision to reconcile against.

This is real load on the flange, and it is live: `MotorChainRobot` hands the composed model to `MuJoCoKDL`, so gravity compensation sees the handle. Handles that differ from stock — older 3D-printed shells in particular — can be corrected at run time with `--ee-mass`, documented in [`examples/minimum_gello`](../../../../../examples/minimum_gello/README.md#overriding-handle-weight); that override sets the shell body's mass only and does not rescale the inertia tensor.
