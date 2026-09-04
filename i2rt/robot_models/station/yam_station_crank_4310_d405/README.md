# YAM station camera extrinsics (`yam_station_crank_4310_d405`)

Two YAM v1 arms with crank_4310 grippers and wrist-mounted Intel RealSense D405 cameras, plus a
third D405 mounted overhead on the gantry crossbar. `yam_station_crank_4310_d405.urdf` and the MJCF
generated from it carry every transform below at full precision. The `linear_4310` counterpart is
[`../yam_station_linear_4310_d405/`](../yam_station_linear_4310_d405/README.md).

`{side}_gripper` is the flange, i.e. the `joint6` output frame. `tcp_left` / `tcp_right` are the
tool-centre frames, copied from `crank_4310`'s `grasp_site`: the flange turned 180° about X and
carried out to the centre of the two claw points — 146.764 mm along the flange's −Z and 44.591 mm
off its axis. That lateral offset is the geometry rather than an error: each claw tapers to a point
near one end of its jaw, so the two points meet well off the flange axis, and unlike the linear
station's this transform is gripper-specific. It is measured from the tip meshes and holds at every
opening, because the claws slide transversely. Sites are a MuJoCo concept, so these live in the
MJCF only.
`{side}_camera` and `top_camera` are the optical frames. Each is also the frame of its camera assembly's merged mesh (one STL per
assembly in [`../assets/`](../assets/), baked from the CAD parts): the mesh sits at identity in the
body, so "where the geometry is" and "where the camera is" are the same frame, with nothing to
compose between them.

## Conventions

- Orientations are quaternions `(w, x, y, z)`, translations are metres. The URDF stores the same
  transforms as `rpy` with `R = Rz(yaw) · Ry(pitch) · Rx(roll)`; read it there if you need Euler.
- **Every camera frame's +Z is its optical axis** (ROS/OpenCV: +X right, +Y down, +Z forward).
- Values are rounded to 3 decimals (1 mm) with trailing zeros dropped, *except* where a rounded
  digit would misrepresent the geometry — see the flange→camera table. The URDF and MJCF carry full
  precision — read them if you need more digits. Each camera is one link there, so every transform
  below is a single `<origin>` you can read straight off, not a chain to compose.
  Renormalize any quaternion copied from here; rounding leaves them up to 5e-4 off unit length.
- Every rotation here is a whole number of degrees (30°, 40°, ±90°, 180°). Each committed `<origin>`
  carries their composition at full precision, and because the CAD export wrote every factor as a
  6-significant-figure `rpy` constant (`0.523599`, `0.698132`, `1.5708`, `3.14159`), the committed
  transforms sit ≤1.5 arcsec from exact.

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

The camera sits on the arms' midline, so the two rows above are exact mirrors in `y`. All three rows
are identical to the linear station's — the top camera is shared verbatim, mesh included.

## Flange → wrist camera (`{side}_gripper` → `{side}_camera`)

**This is the only extrinsic that differs from the linear station.** It is identical for both arms —
the two mounts have byte-identical origins in the URDF, so one table serves both:

| | |
| --- | --- |
| `xyz` | `-0.08  0.0017  -0.066` |
| `quat (w,x,y,z)` | `0.299  0.641  0.641  0.299` |

`y` is quoted at its full `0.0017`, not rounded to `0.002`: that 0.3 mm is 18% of the value, and it is
the whole reason the optical ray misses the gripper axis rather than crossing it (below). The linear
station's `y` is `-7.7e-08`, so 3 decimals cost it nothing.

The optical axis is canted **50°** off the flange's −Z approach axis, tilted back toward the gripper
centreline. The ray leaves the camera 80 mm behind and 66 mm below the flange and passes **within
1.700 mm** of the gripper axis, at its closest at `z = -0.1329`. It never actually meets the axis:
the camera sits at `y = +0.0017` and its optical axis has an exactly zero `y` component, so the ray
stays in the plane `y = +0.0017`. The tool centre is 18.1 mm off that ray, because `tcp_{side}` sits
44.6 mm off the flange axis out at the claw points while the ray tracks the axis itself.

The two cant angles follow from the mount rotations: the D405 sits at `Rz(90°)·Rx(40°)` here against
`Rz(90°)·Rx(65°)` on the linear mount, and `90° − 40° = 50°` against `90° − 65° = 25°`. Aiming at the
fingers more nearly than 35 mm past them is the practical difference between the two wrist views.

The wrist camera hangs off `joint6`'s output by a fixed joint, so this transform holds at
every arm configuration. Because both arms carry the same mount and the same base orientation, it is
also side-independent; only the base→top-camera extrinsics differ between arms.
