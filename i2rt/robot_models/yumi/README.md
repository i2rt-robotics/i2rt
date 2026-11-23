# yumi device model (`yumi.urdf`, `yumi.xml`)

The yumi handheld UMI device ([`i2rt/yumi/`](../../yumi/README.md)): its body (handle, MF camera housing and
tracker seat), two fingers on prismatic rails, and the HTC VIVE Tracker 3.0 screwed to the seat. The URDF adds three
frames the CAD export does not name: `tcp`, `camera` and `vive`.
[`examples/visualize_yumi_viser`](../../../examples/visualize_yumi_viser/README.md) renders it live.

`yumi.xml` is its MJCF twin (`YUMI_XML_PATH`): the same bodies, joints, meshes and colours, with `tcp`, `camera` and
`vive` as sites on `base`. The URDF is the source; `i2rt/yumi/tests/test_yumi_model.py` holds the two equal. With no
`<inertial>` to copy, MuJoCo derives the MJCF's masses from the meshes at its default density.

## Links and joints

| Link | What it is | Mesh |
| --- | --- | --- |
| `base` | The body. **+X forward** (the way the fingers point), **+Y left**, **+Z up** (toward the tracker); origin on the bottom face of the handle | `assets/base.stl` |
| `left_finger`, `right_finger` | The fingers, on `left_finger_joint` (+Y) and `right_finger_joint` (-Y): **0 closed** (the fingertips touch) to **0.047 m open** each | `assets/{left,right}_finger.stl` |
| `vive` | The tracker frame, which also carries the tracker mesh | `assets/vive_tracker.stl` |
| `tcp`, `camera` | Frames only | — |

Every mesh is expressed in its own link frame, so every `<visual><origin>` is identity.

## Frames

The three frames share **+X = the device's right**, so any two of them differ by a rotation about X alone.

| `base` → | `xyz` (m) | `quat (w,x,y,z)` | `rpy` |
| --- | --- | --- | --- |
| `tcp` | `0.1683  -0.0019  0.0912` | `0.5  -0.5  0.5  -0.5` | `-90°  0  -90°` |
| `camera` | `0.1053  -0.0015  0.1731` | `0.3799  -0.5964  0.5964  -0.3799` | `-115°  0  -90°` |
| `vive` | `0.0564  -0.0015  0.1885` | `0.1228  -0.6964  0.6964  -0.1228` | `-160°  0  -90°` |

Between the frames, which is what a tracked recording needs:

| | `xyz` (m) | rotation |
| --- | --- | --- |
| `vive` → `camera` | `0  -0.0407  0.0312` | Rx(45°), `quat (0.9239, 0.3827, 0, 0)` |
| `vive` → `tcp` | `0.0003  -0.0719  0.1298` | Rx(70°), `quat (0.8192, 0.5736, 0, 0)` |
| `camera` → `tcp` | `0.0003  0.0476  0.0918` | Rx(25°), `quat (0.9763, 0.2164, 0, 0)` |

The TCP is 103.4 mm from the lens face, 27.4° off the optical axis, straight below the image centre.

### `tcp`: the fingertips

- **Origin**: midway between the two fingertip end faces, at the fingertips' furthest point, 168.3 mm forward of
  the handle. Each fingertip ends in a flat 8.3 mm² face; the origin is the midpoint of their centroids, which lies
  on the fingers' midplane, the plane the two finger joints are symmetric about.
- **Axes**: the tool convention, **+Z forward** (along the fingers), **+Y down**, **+X right**.
- It holds at every opening as long as the two fingers open by the same amount.

### `camera`: the MF camera's optical frame

- **Origin**: the centre of the lens face, an 18 mm disc.
- **Axes**: OpenCV, **+X image right**, **+Y image down**, **+Z along the optical axis**. The optical axis is the
  lens face's normal: forward and 25° down, toward the fingertips.
- **Roll**: image-down is taken as the direction toward the fingers, so the fingertips appear at the bottom of the
  image and the device's left on the image's left. The camera board is square, so the CAD cannot show which way
  the sensor is turned. If a live image comes out rotated about the optical axis, the board is mounted another
  way and `camera_joint` needs the matching roll.

### `vive`: the tracker frame

- **Origin**: the centre of the seat's ¼″ screw hole (6.6 mm), on the flat face the tracker's mounting pad sits
  against.
- **Axes**: the tracker frame of `vive_pose`, as drawn in [`i2rt/yumi/README.md`](../../yumi/README.md#frames).
  **+Z** points out of the tracker's mounting face, away from its dome, so into the seat. **+Y** points from the
  screw toward the tracker's pogo pins. **+X** completes a right-handed frame.
- **How the rotation is fixed**: the seat carries a stabilizing pin 13.97 mm from the screw, which sits in the
  tracker's stabilizing-pin hole, 14 mm from its screw on the side away from the pogo pins. So +Y points from
  the pin through the screw. The seat face is tilted so that the tracker's dome faces up and 20° back.
- **Tracker mesh**: HTC's VIVE Tracker 3.0 model, moved into this frame: its mounting pad at z = 0, the screw axis
  at x = y = 0.

## Conventions

- Translations are metres and quaternions `(w, x, y, z)`. The URDF writes each rotation as `rpy` with
  `R = Rz(yaw) · Ry(pitch) · Rx(roll)`. Every angle above is a whole number of degrees, as measured on the CAD.
- The tables round to 0.1 mm; the URDF carries full precision.

## What is not modelled

- **Inertia.** No link has an `<inertial>`: the export's masses were part volumes at unit density, and the device
  is passive.
- **Collision geometry.**
- **The finger coupling.** The two fingers are independent joints; on the device they open together.
- **Full mesh detail.** The meshes are decimated: the body from 390k to 107k faces, the tracker from 229k to 72k,
  each finger from 51k to 26k. 99% of their vertices lie within 0.16 mm of the CAD surface; the worst lie 2.1 mm
  off on the body, 1.1 mm on the tracker and 0.2 mm on the fingers. All three frames were measured on the
  undecimated CAD.
