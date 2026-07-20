# Big YAM Arm Physical Properties

This document records the physical and kinematic properties of the six-DOF Big YAM arm from [`big_yam.urdf`](big_yam.urdf). The corresponding arm-only MuJoCo model is [`big_yam.xml`](big_yam.xml).

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
| Physical arm mass, excluding gripper and tips | 5.307106 kg |
| Sum of parent-to-child translation norms | 1.0595561697 m |

## Link mass and inertial frames

The COM position and inertial-frame RPY are relative to the corresponding link frame.

| Link | Mass (kg) | COM xyz (m) | Inertial-frame RPY (rad) |
| --- | ---: | --- | --- |
| `base` | 1.11529 | `0 0 0.0314825` | `0 0 0` |
| `link1` | 0.87455 | `0.0175642 -0.00563025 -0.0562098` | `0 0 0` |
| `link2` | 1.29451 | `-0.196912 -0.0105 0.0197601` | `0 0 0` |
| `link3` | 1.16452 | `0.211196 0 0.0377013` | `0 0 0` |
| `link4` | 0.454929 | `0.0573241 -0.0534721 0.0332447` | `0 0 0` |
| `link5` | 0.403307 | `0.000825911 0 -0.0375847` | `0 0 0` |
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
| `base` | 0.00127239 | 0.00108285 | 0.00120806 | 0 | 0 | 0 |
| `link1` | 0.000718995 | 0.000898846 | 0.000605134 | 0 | 0.000129101 | 0 |
| `link2` | 0.00176716 | 0.0244467 | 0.0248589 | -0.00247342 | 0.000285312 | 0 |
| `link3` | 0.00071823 | 0.0175944 | 0.0173974 | 0 | 0 | 0 |
| `link4` | 0.000606911 | 0.000482841 | 0.000768197 | 0.000253478 | 0 | 0 |
| `link5` | 0.000181053 | 0.000219791 | 0.000184284 | 0 | 0 | 0 |

## Joint frames, axes, ranges, and link lengths

The origin is the joint frame relative to its parent link. At zero joint displacement, the joint frame coincides with the child link frame.

| Joint | Parent → child | Origin xyz (m) | Origin RPY (rad) | Axis | Range (rad) | Range (deg) | Link length (m) |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `joint1` | `base → link1` | `0 0 0.0666` | `π 0 0` | `0 0 -1` | `[-11π/12, 11π/12]` | `[-165, 165]` | 0.0666000000 |
| `joint2` | `link1 → link2` | `0.0219115 0.022425 -0.0639519` | `π/2 0 0` | `0 0 1` | `[0, π]` | `[0, 180]` | 0.0712238722 |
| `joint3` | `link2 → link3` | `-0.378697 -0.068 0.05915` | `-π 0 0` | `0 0 1` | `[0, 3.01942]` | `[0, 173]` | 0.3892738629 |
| `joint4` | `link3 → link4` | `0.381981 0 0.0695705` | `-π 0 0` | `0 0 -1` | `[-1.69297, π/2]` | `[-97, 90]` | 0.3882647793 |
| `joint5` | `link4 → link5` | `0.0739989 -0.0403003 0.0323887` | `π/2 0 0` | `0 0 -1` | `[-π/2, π/2]` | `[-90, 90]` | 0.0902716969 |
| `joint6` | `link5 → link6 mount` | `0.0356 0 -0.0404996` | `-π/2 0 π/2` | `0 0 1` | `[-2π/3, 2π/3]` | `[-120, 120]` | 0.0539219584 |

## Global link and joint frames at home

The base frame is the world/TF root. Each `jointN` frame coincides with its corresponding `linkN` frame at home. The global RPY values below use the canonical extraction with pitch in `[-π/2, π/2]`. The final column gives each joint's rotation axis expressed in the world frame at home — the local `Axis` from the previous table carried through that frame's global orientation; `base` is fixed and has no joint.

| Link / coincident joint frame | Global xyz (m) | Global RPY (rad) | Joint axis, world frame |
| --- | --- | --- | --- |
| `base` | `0 0 0` | `0 0 0` | — |
| `link1 / joint1` | `0 0 0.0666` | `π 0 0` | `0 0 1` |
| `link2 / joint2` | `0.0219115 -0.0224248302978 0.130551959507` | `-π/2 0 0` | `0 1 0` |
| `link3 / joint3` | `-0.3567855 0.0367251003683 0.198552019817` | `π/2 0 0` | `0 -1 0` |
| `link4 / joint4` | `0.0251955 -0.0328453996312 0.19855176427` | `-π/2 0 0` | `0 -1 0` |
| `link5 / joint5` | `0.0991944 -0.000456954603595 0.238852269186` | `0 0 0` | `0 0 -1` |
| `link6 mount / joint6` | `0.1347944 -0.000456789465595 0.198352669186` | `-π/2 0 π/2` | `-1 0 0` |

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
| 1 | `0` | `0.130552` | `0.021912` | `-π/2` |
| 2 | `-2.963923` | `0` | `0.384754` | `π` |
| 3 | `-2.963923` | `0` | `0.381981` | `0` |
| 4 | `0` | `0.000456` | `0.073999` | `π/2` |
| 5 | `π/2` | `0.000199` | `0` | `-π/2` |
| 6 | `π` | `-0.0356` | `0` | `0` |

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
| `joint2` | `0 1 0` | `-0.130552 0 0.021911` |
| `joint3` | `0 -1 0` | `0.198552 0 0.356785` |
| `joint4` | `0 -1 0` | `0.198552 0 -0.025195` |
| `joint5` | `0 0 -1` | `0.000455 0.099194 0` |
| `joint6` | `-1 0 0` | `0 -0.198352 -0.000457` |

Home configuration `M` (end-effector pose at zero joints), as a 4×4 homogeneous
transform in the base frame:

```text
[ +0.000000  +0.000000  -1.000000  +0.134794 ]
[ +1.000000  +0.000000  +0.000000  -0.000457 ]
[ +0.000000  -1.000000  +0.000000  +0.198353 ]
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
| `joint1` | `0 -1 0` | `0.134794 0 -0.000458` |
| `joint2` | `1 0 0` | `0 0.112883 -0.0678` |
| `joint3` | `-1 0 0` | `0 -0.49158 -0.000201` |
| `joint4` | `-1 0 0` | `0 -0.109599 -0.0002` |
| `joint5` | `0 1 0` | `-0.0356 0 0` |
| `joint6` | `0 0 1` | `0 0 0` |
