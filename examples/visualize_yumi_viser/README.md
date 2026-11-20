# Visualize yumi (Viser)

A live 3D view of the yumi handheld device in [viser](https://github.com/nerfstudio-project/viser):

- the **yumi model** ([`i2rt/robot_models/yumi/`](../../i2rt/robot_models/yumi/README.md)) following the Vive
  tracker's pose, hidden while the tracker has no optical lock, with its fingers following the gripper opening;
- a **camera frustum** at the model's `camera` frame, textured with the live MF camera stream (raw fisheye frames);
- a panel with the **gripper** opening (0 closed … 1 open), the encoder's two **buttons**, and each stream's health.

## Quick Start

The yumi's three sensors must each be set up first: the gripper's CAN bus up and its encoder zeroed, and SteamVR
installed. [`i2rt/yumi/README.md`](../../i2rt/yumi/README.md#hardware-setup) covers all three, and its bring-up
readout checks them on their own:

```bash
python i2rt/yumi/yumi.py          # all three sensors, printed at ~10 Hz
```

Then:

```bash
uv run examples/visualize_yumi_viser/visualize_yumi_viser.py
```

Without serials it opens the first camera plugged in and follows the first tracker SteamVR reports, and logs which
it picked (a warning names the others when there are several). To choose, name them; `python
i2rt/yumi/vive_tracker.py` lists the tracker serials:

```bash
uv run examples/visualize_yumi_viser/visualize_yumi_viser.py \
    --tracker-serial LHR-XXXXXXXX --camera-serial MF150_2026061109
```

Startup waits for all three sensors, including SteamVR and a first valid tracker pose (up to about 30 s). Then open
`http://localhost:8080` in a browser. Quit with `Ctrl+C` in the terminal.

## What you see

| Element | Meaning |
|---------|---------|
| Floor grid + large axes | OpenVR's standing universe: metres, **+Y up**, origin and yaw from the SteamVR room setup |
| Small axes | The tracker frame (red X, green Y, blue Z). Its axes are drawn in [`i2rt/yumi/README.md`](../../i2rt/yumi/README.md#frames) |
| Model | The yumi: body, fingers and Vive tracker, from `i2rt/robot_models/yumi/yumi.urdf` |
| Frustum | The camera, at the model's `camera` frame: looking forward and 25° down, toward the fingertips |
| **gripper** slider | `gripper_pos`: 0 = closed … 1 = open (read-only) |
| **button 1 / 2** | The handle encoder's two digital inputs, checked while pressed (read-only) |
| Status line | `tracking` / `pose lost`, and any stream that has gone stale |

**Placement.** The model hangs off the tracker by its `vive` frame, the tracker frame of `vive_pose`, and the
frustum sits at its `camera` frame (OpenCV: +X right, +Y down, +Z forward, the same convention as viser's
frustum). Both are the CAD design values in the URDF, not a per-unit calibration. In the tracker frame the camera
is 40.7 mm along -Y and 31.2 mm along +Z from the screw, turned 45° about X, so its optical axis sits midway
between the tracker's -Y and +Z. [`i2rt/robot_models/yumi/README.md`](../../i2rt/robot_models/yumi/README.md)
tabulates every frame.

**Fingers.** Each finger is drawn at `gripper_pos` times its 47 mm travel, which assumes the travel is linear in the
handle encoder's angle.

**Camera image.** The frustum shows the raw fisheye frame, not rectified, at a fixed display angle (70° tall). The
driver carries no intrinsics, so pixels are not on their true rays; the frustum shows where the camera points and
what it sees, not a metric projection.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tracker-serial` | first tracker | Serial of the Vive tracker on the handle; defaults to the first tracker SteamVR reports. |
| `--camera-serial` | first camera | USB short serial of the MF camera, e.g. `MF150_2026061109`; defaults to the first camera plugged in. |
| `--can-channel` | `can0` | CAN channel of the gripper's handle encoder. |
| `--range-rad` | `5.1` | Encoder angle (rad) from the full-open zero that reads as fully closed. |
| `--port` | `8080` | Viser server port. |

## Notes

- **Hardware required.** The yumi's three sensors, and SteamVR on an x86-64 host (the OpenVR library is x86-64
  only).
- **Losing the pose is normal.** The model and frustum hide while the tracker has no line of sight to a base
  station, and reappear when it does. A stream that stops updating is reported in the status line and does not
  stop the viewer; a device process that dies does.
