# YAM Pro Arm Physical Properties

This document records the physical and kinematic properties of the six-DOF YAM Pro arm from [`yam_pro.urdf`](yam_pro.urdf). The corresponding arm-only MuJoCo model is [`yam_pro.xml`](yam_pro.xml).

## Scope and conventions

- The physical arm consists of `base` and `link1` through `link5`, connected by `joint1` through `joint6`.
- The URDF names the child of `joint6` `gripper`. The arm-only MJCF represents that same attachment frame as `link6`; it is not a physical arm link.
- Gripper, finger, tip, tool, and top bodies are excluded.
- SI units are used: metres, kilograms, radians, and kg·m².
- Home pose means all six arm joint coordinates are zero.
- URDF RPY uses `R = Rz(yaw) Ry(pitch) Rx(roll)`.
- Joint axes are expressed in their local joint frames.
- “Link length” is the Euclidean norm of the parent-to-child joint-origin translation. The full translation vector remains the authoritative kinematic value.
- Frame orientations (RPY) and joint ranges within floating-point tolerance of `0` or a multiple of π are written as `0` or that multiple (e.g. `π/2`, `-2π/3`); any near-zero numeric component with magnitude below `1e-4` — COM offsets, off-diagonal inertia terms, translation, axis, and global-frame coordinates — is likewise shown as `0` (diagonal inertia moments always keep their value). Degree joint ranges are rounded to the nearest integer. All other values keep their full precision.

## Summary

| Property | Value |
| --- | ---: |
| Arm DOF | 6 |
| Physical arm bodies | 6 |
| Physical arm mass, excluding gripper and tips | 4.348977 kg |
| Sum of parent-to-child translation norms | 0.7977942186 m |

## Link mass and inertial frames

The COM position and inertial-frame RPY are relative to the corresponding link frame.

| Link | Mass (kg) | COM xyz (m) | Inertial-frame RPY (rad) |
| --- | ---: | --- | --- |
| `base` | 0.873654 | `0 0 0.0247677998591` | `π π/2 0` |
| `link1` | 0.101218 | `-0.0233636 -0.00172503 -0.00727828` | `0 0 0` |
| `link2` | 1.52608 | `0.000134495 0.0336247 0.129165` | `0 0 0` |
| `link3` | 0.982553 | `-0.0556042 0.0357986 -0.135083` | `0 0 0` |
| `link4` | 0.462165 | `-0.0548959 -0.0350242 -0.0575534` | `0 0 0` |
| `link5` | 0.403307 | `0.0375847 0 -0.000825911` | `0 0 0` |
| `link6` mount | — | — | — |

The `link6` MJCF placeholder uses mass `1e-6 kg` and diagonal inertia `1e-9 1e-9 1e-9 kg·m²` only to satisfy MuJoCo. These are simulation artifacts and are not included in the physical arm mass.

## Link inertia

Each tensor is about the link COM and expressed in the inertial frame listed above:

```text
I = [[Ixx, Ixy, Ixz],
     [Ixy, Iyy, Iyz],
     [Ixz, Iyz, Izz]]
```

| Link | Ixx | Iyy | Izz | Ixy | Ixz | Iyz |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.00157059 | 0.000678307 | 0.00169404 | 0 | 0 | 0 |
| `link1` | 0.000114689 | 4.76603e-05 | 0.000125996 | 0 | 0 | 0 |
| `link2` | 0.0152377 | 0.0151654 | 0.00083113 | 0 | 0 | 0 |
| `link3` | 0.00758945 | 0.00763097 | 0.000937606 | 0 | -0.00064069 | 0 |
| `link4` | 0.000473594 | 0.000764022 | 0.000604492 | 0 | -0.000238801 | 0 |
| `link5` | 0.000184284 | 0.000219791 | 0.000181053 | 0 | 0 | 0 |

## Joint frames, axes, ranges, and link lengths

The origin is the joint frame relative to its parent link. At zero joint displacement, the joint frame coincides with the child link frame.

| Joint | Parent → child | Origin xyz (m) | Origin RPY (rad) | Axis | Range (rad) | Range (deg) | Link length (m) |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `joint1` | `base → link1` | `0 0 0.0679999999995` | `-π π/2 0` | `-1 0 0` | `[-5π/6, π]` | `[-150, 180]` | 0.0680000000 |
| `joint2` | `link1 → link2` | `-0.0455 -0.0329 -0.02` | `0 0 0` | `0 -1 0` | `[0, 7π/6]` | `[0, 210]` | 0.0596041945 |
| `joint3` | `link2 → link3` | `0 -0.00205 0.264` | `0 0 0` | `0 1 0` | `[0, π]` | `[0, 180]` | 0.2640079592 |
| `joint4` | `link3 → link4` | `-0.0600003 0.0688 -0.244999` | `0 0 0` | `0 1 0` | `[-1.69297, π/2]` | `[-97, 90]` | 0.2614536020 |
| `joint5` | `link4 → link5` | `-0.0403003 -0.0338507 -0.0739989` | `0 0 0` | `1 0 0` | `[-π/2, π/2]` | `[-90, 90]` | 0.0908065046 |
| `joint6` | `link5 → link6 mount` | `0.0404996 0 -0.0356` | `0 0 0` | `0 0 -1` | `[-2π/3, 2π/3]` | `[-120, 120]` | 0.0539219584 |

## Global link and joint frames at home

The base frame is the world/TF root. Each `jointN` frame coincides with its corresponding `linkN` frame at home. The global RPY values below use the canonical extraction with pitch in `[-π/2, π/2]`. The final column gives each joint's rotation axis expressed in the world frame at home — the local `Axis` from the previous table carried through that frame's global orientation; `base` is fixed and has no joint.

| Link / coincident joint frame | Global xyz (m) | Global RPY (rad) | Joint axis, world frame |
| --- | --- | --- | --- |
| `base` | `0 0 0` | `0 0 0` | — |
| `link1 / joint1` | `0 0 0.0679999999995` | `π π/2 0` | `0 0 1` |
| `link2 / joint2` | `0.0199994957879 0.0329000530712 0.113500073463` | `π π/2 0` | `0 1 0` |
| `link3 / joint3` | `-0.244000509648 0.0349493525234 0.113498695306` | `π π/2 0` | `0 -1 0` |
| `link4 / joint4` | `0.000998452523352 -0.03384999735 0.173499895237` | `π π/2 0` | `0 -1 0` |
| `link5 / joint5` | `0.0749971146654 0 0.21380046705` | `π π/2 0` | `0 0 -1` |
| `link6 mount / joint6` | `0.110597263429 0 0.173300997817` | `π π/2 0` | `1 0 0` |

Several home-frame orientations are very close to the RPY pitch singularity at `±π/2`. Equivalent roll/yaw pairs may therefore look different while representing the same rotation; use rotation matrices or the model quaternions for numerical comparisons.

## Classical Denavit–Hartenberg parameters

Standard ("distal") DH convention: `Zᵢ₋₁` lies on joint axis `i`, and `Xᵢ` is the
common normal between `Zᵢ₋₁` and `Zᵢ`. The link transform is

```text
T(i-1, i) = Rot_z(θᵢ) · Trans_z(dᵢ) · Trans_x(aᵢ) · Rot_x(αᵢ)
```

`θᵢ` is the revolute joint variable; the table lists its **home offset** (value at
zero displacement — add the joint coordinate to it). Frame `{0}` is the `base`
frame (`Z₀` on joint axis 1) and frame `{6}` is the `link6` mount frame (the
end-effector frame `M` used below). Joints 2–4 have parallel axes, so the
`aᵢ`/`dᵢ` spanning them are not unique; origins there are placed by perpendicular
projection from the preceding frame, giving the values shown.

| i | θᵢ home offset (rad) | dᵢ (m) | aᵢ (m) | αᵢ (rad) |
| --- | ---: | ---: | ---: | ---: |
| 1 | `0` | `0.1135` | `0.02` | `-π/2` |
| 2 | `π` | `0` | `0.264` | `π` |
| 3 | `-2.901421` | `0` | `0.252239` | `0` |
| 4 | `-0.240173` | `0` | `0.073999` | `π/2` |
| 5 | `π/2` | `0.000199` | `0` | `π/2` |
| 6 | `π/2` | `0.0356` | `0` | `π` |

## Product-of-exponentials screw axes

Space-frame (base-frame) screw axes for the Modern Robotics PoE formulation

```text
T(θ) = e^([S₁]θ₁) · e^([S₂]θ₂) ··· e^([S₆]θ₆) · M
```

where each `Sᵢ = (ωᵢ, vᵢ)` with `ωᵢ` the unit joint axis and `vᵢ = -ωᵢ × qᵢ`
(`qᵢ` a point on the axis), all at the home configuration. The end-effector frame
is the `link6` mount frame; the space frame is the `base` frame.

| Joint | ωᵢ (unit) | vᵢ (m) |
| --- | --- | --- |
| `joint1` | `0 0 1` | `0 0 0` |
| `joint2` | `0 1 0` | `-0.1135 0 0.02` |
| `joint3` | `0 -1 0` | `0.113499 0 0.244` |
| `joint4` | `0 -1 0` | `0.1735 0 -0.000998` |
| `joint5` | `0 0 -1` | `0 0.074998 0` |
| `joint6` | `1 0 0` | `0 0.173301 0` |

Home configuration `M` (end-effector pose at zero joints), as a 4×4 homogeneous
transform in the base frame:

```text
[ +0.000000  +0.000000  -1.000000  +0.110597 ]
[ +0.000000  -1.000000  +0.000000  +0.000000 ]
[ -1.000000  +0.000000  +0.000000  +0.173301 ]
[ +0.000000  +0.000000  +0.000000  +1.000000 ]
```

The same joint axes expressed in the **body** (end-effector) frame give the
body-form PoE, with each `Bᵢ` the space axis `Sᵢ` transformed by the adjoint of `M⁻¹`:

```text
T(θ) = M · e^([B₁]θ₁) · e^([B₂]θ₂) ··· e^([B₆]θ₆)
Bᵢ   = [Ad_{M⁻¹}] Sᵢ
```

| Joint | ωᵢ (unit) | vᵢ (m) |
| --- | --- | --- |
| `joint1` | `-1 0 0` | `0 -0.110598 0` |
| `joint2` | `0 -1 0` | `0.090598 0 -0.059801` |
| `joint3` | `0 1 0` | `-0.354598 0 0.059801` |
| `joint4` | `0 1 0` | `-0.109599 0 -0.000199` |
| `joint5` | `1 0 0` | `0 0.0356 0` |
| `joint6` | `0 0 -1` | `0 0 0` |
