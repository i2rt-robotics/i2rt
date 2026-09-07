# yambox camera extrinsics (`yambox_linear_4310_d405_decxin`)

Two YAM v1 arms with linear_4310 grippers and wrist-mounted Intel RealSense D405 cameras, 0.61 m
apart — the same arm-to-arm offset as the YAM station — plus a DECXIN camera looking down over the
shared workspace. `yambox_linear_4310_d405_decxin.urdf` and the MJCF generated from it carry every
transform below at full precision.

`{side}_gripper` is the flange, i.e. the `joint6` output frame. `tcp_left` / `tcp_right` are the
tool-centre frames, copied from `linear_4310`'s `grasp_site`: the flange turned 180° about X and
carried out to the centre of the two fingertip facets — 144.650 mm along the flange's −Z and
0.104 mm off its axis. It is measured from the tip meshes and holds at every finger opening, because
the fingers slide transversely. Sites are a MuJoCo concept, so these live in the MJCF only.
`{side}_camera` and `top_camera` are the optical frames. Each is also the frame of its camera
assembly's merged mesh (one STL per assembly in [`../assets/`](../assets/), baked from the CAD
parts): the mesh sits at identity in the body, so "where the geometry is" and "where the camera is"
are the same frame, with nothing to compose between them.

Provenance: the arms and their wrist cameras are byte-identical to
[`../yam_station_linear_4310_d405/`](../yam_station_linear_4310_d405/README.md) — every arm and
wrist link and joint of it — so the wrist table below and that station's agree exactly. The DECXIN is
modelled as three CAD parts, its bracket, its cover and the camera body, baked into the single
`../assets/decxin_top.stl`; the per-part frames and meshes are in git history.

## What is not modelled

**The yambox chassis, and the mast the camera bracket clamps to** (along with the bracket's own arm
down to it, see below)**.** `left_base` is the model root and
the only structure here is the two arm bases and the camera assembly, so nothing in this file fixes
the arms' height above the floor or their pose inside the box — only their offset from each other and
the camera's pose relative to the left arm.

## Conventions

- Orientations are quaternions `(w, x, y, z)`, translations are metres. The URDF stores the same
  transforms as `rpy` with `R = Rz(yaw) · Ry(pitch) · Rx(roll)`; read it there if you need Euler.
- **Every camera frame's +Z is its optical axis** (ROS/OpenCV: +X right, +Y down, +Z forward).
- Values are rounded to 3 decimals (1 mm) with trailing zeros dropped. The URDF and MJCF carry full
  precision — read them if you need more digits. Each camera is one link there, so every transform
  below is a single `<origin>` you can read straight off, not a chain to compose.
  Renormalize any quaternion copied from here; rounding leaves them up to 5e-4 off unit length.
- Every rotation in the tables below is a whole number of degrees, and each committed `<origin>`
  carries its chain's composition at full precision. The wrist chain was built from ONShape's
  6-significant-figure `rpy` constants (`1.13446` = 65°, `1.5708`, `3.14159`), so the flange→camera
  rotation sits 2.3 arcsec from exact; the top-camera chain carried full double precision, so its
  46° and 180° hold to better than 1e-12 rad.

## `left_base` → `top_camera`

| | |
| --- | --- |
| `xyz` | `0.027  -0.305  0.627` |
| `quat (w,x,y,z)` | `0.265  -0.656  0.656  -0.265` |

`top_camera_joint` holds this transform in a single line: `xyz = (0.0273, -0.305, 0.6273)`,
`quat = (0.26488686, -0.65561799, 0.65561799, -0.26488686)`, the rotation being
`Rz(-90°) · Rx(-136°)`. Its committed digits carry the sub-nanometre residual of the solve that
placed the assembly rather than these round numbers.

The optical axis sits **46° below horizontal** aimed forward (+X) and meets the base plane at
`(0.633, -0.305, 0)` — 0.633 m in front of the arms, exactly on the midline between them, 0.872 m
along the ray. The camera sits 0.627 m above `left_base` and 0.698 m from either arm base.

The arm-to-arm offset and the right-arm equivalent:

| Transform | `xyz` | `quat (w,x,y,z)` |
| --- | --- | --- |
| `left_base` → `right_base` | `0  -0.61  0` | `1  0  0  0` |
| `right_base` → `top_camera` | `0.027  0.305  0.627` | `0.265  -0.656  0.656  -0.265` |

The two bases share an orientation and the offset is pure `−y`, so the arms' midline is the plane
`y = -0.305`; the camera sits on it, which makes the two rows above exact `y` mirrors.

### The top-camera assembly

`left_base` → `top_camera` is one fixed joint. The bracket, the cover and the camera body are baked
into `../assets/decxin_top.stl` expressed in the optical frame, which sits on the lens face rolled
180° about Z from the export's own camera frame — the roll is what puts +Y down the image, as the
ROS/OpenCV convention requires. The bracket mounts axis-aligned, at a pure 180° yaw about Z to
0.6 arcsec.

**The bracket's mast arm is trimmed off.** The CAD part continues as a 16.9 × 16.9 mm post 153 mm
down to a clamp foot at `z = 0.465`, all of it gripping a mast this model does not carry, so the
mesh is cut at `z = 0.6194` — flush with the underside of the bracket's head plate, 74 µm below it
— and capped. What is left spans `z = 0.618 … 0.667`, 49 mm tall, and nothing hangs below the
camera. Everything above the cut is the CAD geometry untouched.

The link's inertial is the sum of the three parts' unit-density inertials, the bracket's taken over
the trimmed solid: the merged "mass" `4.096e-05` is literally the assembly's mesh volume in m³, the
same placeholder convention the stations' cameras use. Do not use it for dynamics.

## Flange → wrist camera (`{side}_gripper` → `{side}_camera`)

**Identical for both arms** — the two mounts have byte-identical origins in the URDF, so one table
serves both:

| | |
| --- | --- |
| `xyz` | `-0.07  0  -0.077` |
| `quat (w,x,y,z)` | `0.153  0.69  0.69  0.153` |

The optical axis is canted 25° off the flange's −Z approach axis, tilted back toward the gripper
centreline: the ray leaves the camera 70 mm behind and 77 mm below the flange and crosses the gripper
axis at `z = -0.228 m` — 83 mm beyond `tcp_{side}` (`z = -0.14465`) and 81 mm beyond the deepest
fingertip vertex (`z = -0.1468`). In the TCP's own plane the ray is 38.9 mm off the centreline, and
the tool centre sits 35.3 mm off the ray itself: the grasp is inside the D405's field of view, not on
its optical axis.

The wrist camera hangs off `joint6`'s output by a single fixed joint, so this transform holds at
every arm configuration. Because both arms carry the same mount and the same base orientation, it is
also side-independent.
