# YAM station camera extrinsics (`yam_station_linear_4310_d405`)

Two YAM v1 arms with linear_4310 grippers and wrist-mounted Intel RealSense D405 cameras, plus a
third D405 mounted overhead on the gantry crossbar. `yam_station_linear_4310_d405.urdf` and the MJCF
generated from it carry every transform below at full precision. The `crank_4310` counterpart is
[`../yam_station_crank_4310_d405/`](../yam_station_crank_4310_d405/README.md).

`{side}_gripper` is the flange, i.e. the `joint6` output frame. `{side}_camera` and `top_camera` are
massless pure frames carrying no geometry.

## Conventions

- Orientations are quaternions `(w, x, y, z)`, translations are metres. The URDF stores the same
  transforms as `rpy` with `R = Rz(yaw) · Ry(pitch) · Rx(roll)`; read it there if you need Euler.
- **Every camera frame's +Z is its optical axis** (ROS/OpenCV: +X right, +Y down, +Z forward).
- Values are rounded to 3 decimals (1 mm) with trailing zeros dropped. The URDF and MJCF carry full
  precision — read them if you need more digits, or the intermediate `*_bracket` / `*_body` frames.
  Renormalize any quaternion copied from here; rounding leaves them up to 5e-4 off unit length.
- Every rotation here is a whole number of degrees (30°, 65°, ±90°, 180°). The URDF stores them as
  6-significant-figure `rpy` constants (`1.5708`, `3.14159`, `0.523599`, `1.13446`), so the committed
  transforms sit ≤2.3 arcsec from exact.

## `left_base` → `top_camera`

| | |
| --- | --- |
| `xyz` | `-0.166  -0.305  0.954` |
| `quat (w,x,y,z)` | `0.183  -0.683  0.683  -0.183` |

The optical axis sits 60° below horizontal aimed forward (+X) and meets the base plane at
`(0.384, -0.305, 0)` — `0.384` m in front of the arms, exactly on the midline between them.
A quick sanity check when calibrating against this model.

The arm-to-arm offset and the right-arm equivalent:

| Transform | `xyz` | `quat (w,x,y,z)` |
| --- | --- | --- |
| `left_base` → `right_base` | `0  -0.61  0` | `1  0  0  0` |
| `right_base` → `top_camera` | `-0.166  0.305  0.954` | `0.183  -0.683  0.683  -0.183` |

The camera sits on the arms' midline, so the two rows above are exact mirrors in `y`.

## Flange → wrist camera (`{side}_gripper` → `{side}_camera`)

**Identical for both arms** — the two mounts have byte-identical origins in the URDF, so one table
serves both:

| | |
| --- | --- |
| `xyz` | `-0.07  0  -0.077` |
| `quat (w,x,y,z)` | `0.153  0.69  0.69  0.153` |

The optical axis is canted 25° off the flange's −Z approach axis, tilted back toward the gripper
centreline: the ray leaves the camera 70 mm behind and 77 mm below the flange and crosses the
gripper axis at `z = -0.228 m`, i.e. at the fingertips, past `linear_4310`'s `grasp_site` at
`z = -0.1347`.

The whole wrist-camera chain hangs off `joint6`'s output by fixed joints, so this transform holds at
every arm configuration. Because both arms carry the same mount and the same base orientation, it is
also side-independent; only the base→top-camera extrinsics differ between arms.
