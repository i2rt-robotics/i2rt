# YAM station camera extrinsics (`yam_station_linear_4310_d405`)

Two YAM v1 arms with linear_4310 grippers and wrist-mounted Intel RealSense D405 cameras, plus a
third D405 mounted overhead on the gantry crossbar. `yam_station_linear_4310_d405.urdf` and the MJCF
generated from it carry every transform below at full precision. The `crank_4310` counterpart is
[`../yam_station_crank_4310_d405/`](../yam_station_crank_4310_d405/README.md).

`{side}_gripper` is the flange, i.e. the `joint6` output frame. `tcp_left` / `tcp_right` are the
tool-centre frames, copied from `linear_4310`'s `grasp_site`: the flange turned 90° about Z and
carried out to the centre of the two fingertip facets — 144.650 mm along the flange's +Z and
0.104 mm off its axis. It is measured from the tip meshes and holds at every finger opening, because
the fingers slide transversely. Sites are a MuJoCo concept, so these live in the MJCF only.
`{side}_camera` and `top_camera` are the optical frames. Each is also the frame of its camera assembly's merged mesh (one STL per
assembly in [`../assets/`](../assets/), baked from the CAD parts): the mesh sits at identity in the
body, so "where the geometry is" and "where the camera is" are the same frame, with nothing to
compose between them.

## Conventions

- Orientations are quaternions `(w, x, y, z)`, translations are metres. The URDF stores the same
  transforms as `rpy` with `R = Rz(yaw) · Ry(pitch) · Rx(roll)`; read it there if you need Euler.
- **Every camera frame's +Z is its optical axis** (ROS/OpenCV: +X right, +Y down, +Z forward).
- Values are rounded to 3 decimals (1 mm) with trailing zeros dropped. The URDF and MJCF carry full
  precision — read them if you need more digits. Each camera is one link there, so every transform
  below is a single `<origin>` you can read straight off, not a chain to compose.
  Renormalize any quaternion copied from here; rounding leaves them up to 5e-4 off unit length.
- Every rotation here is a whole number of degrees (30°, 65°, ±90°, 180°). Each committed `<origin>`
  carries their composition at full precision, and because the CAD export wrote every factor as a
  6-significant-figure `rpy` constant (`1.5708`, `3.14159`, `0.523599`, `1.13446`), the committed
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
| `xyz` | `0  -0.07  0.077` |
| `quat (w,x,y,z)` | `-0.976  0.216  0  0` |

The optical axis is canted 25° off the flange's +Z approach axis, tilted back toward the gripper
centreline. In this frame the mount is a pure `Rx(-25°)`, so the quaternion above *is* the cant: the
ray leaves the camera 70 mm above and 77 mm ahead of the flange and crosses the
gripper axis at `z = +0.228 m` — 83 mm *beyond* the fingertip plane at `z = +0.14465` where
`tcp_{side}` sits. The tool centre itself is 35.3 mm off the optical ray, so the wrist camera looks
past the fingers rather than at them.

The wrist camera hangs off `joint6`'s output by a fixed joint, so this transform holds at
every arm configuration. Because both arms carry the same mount and the same base orientation, it is
also side-independent; only the base→top-camera extrinsics differ between arms.
