# yumi — handheld UMI data-collection device

The yumi is a handheld UMI-style device with three sensors:

| Sensor | What it measures | Module |
|---|---|---|
| Passive UMI gripper | Jaw opening, from a passive CAN encoder on the handle | [`gripper.py`](gripper.py) |
| Vive tracker | 6-DoF pose of the handle, through SteamVR / OpenVR | [`vive_tracker.py`](vive_tracker.py) |
| i2rt MF150 camera | 150° USB (UVC) fisheye images; the MF118 is the 118° variant | [`mf_camera.py`](../camera/mf_camera.py) |

[`yumi.py`](yumi.py) combines them into `Yumi`, an i2rt `Robot`. **Each sensor runs in its own process** and
pushes every new sample to the main process over a pipe ([`device_process.py`](device_process.py)). A slow
frame decode therefore never jitters the gripper stream, a stuck driver stalls only its own stream, and a native
crash in one device is reported instead of taking the recording down.

```python
from i2rt.yumi.yumi import Yumi

with Yumi(can_channel="can0", tracker_serial="LHR-XXXXXXXX", camera_serial="MF150_2026061109") as yumi:
    obs = yumi.get_observations()   # {"gripper_pos": (1,), "vive_pose": (4, 4), "vive_pose_valid": (1,)}
    image = yumi.get_image()        # (H, W, 3) uint8 RGB
```

Every CLI in this directory uses tyro, so `--help` lists its flags. All of them run from the repo root.

## Contents

1. [Hardware setup](#hardware-setup): gripper, Vive tracker, MF camera
2. [Python API](#python-api)

---

## Hardware setup

### Gripper (passive encoder on CAN)

The gripper's jaw drives a passive absolute encoder that reports over CAN (device `0x50E`), one encoder per
CAN channel.

1. **Bring the CAN interface up** at 1 Mbit/s:

   ```bash
   sudo ip link set can0 up type can bitrate 1000000   # or: scripts/reset_all_can.sh
   ```

   Stable per-device names such as `can_left` are covered in
   [`docs/guides/set-persistent-can-ids.md`](../../docs/guides/set-persistent-can-ids.md).

2. **Zero the encoder at the jaws' full-open mechanical stop.** Hold the jaws fully open and run:

   ```bash
   python devices/config_passive_encoder.py --bus can0 reset-zero-position
   ```

   The mapping is `gripper_pos = 1 − |clip(angle, ±5.1 rad)| / 5.1`:
   - 1.0 is open, 0.0 is closed. This is the i2rt `gripper_pos` convention.
   - 5.1 rad is the measured full-squeeze travel of the handle. It can be changed with `range_rad` / `--range-rad`.
   - Because of the `abs()`, the zero must sit at the stop. A zero anywhere else reads as "more open" on both
     sides of it.
   - The value is not calibrated per unit, so a handle whose squeeze travel differs from 5.1 rad reads slightly
     off 0.0 when fully closed.

3. **Check it:** `python i2rt/yumi/gripper.py --channel can0` prints the position, plus the raw encoder angle
   and velocity and the two buttons (1 = pressed), at about 20 Hz.

> **Startup cost.** Opening the gripper validates the encoder, as every i2rt passive-encoder reader does. This
> takes about 2 s and may write the encoder's report frequency to 0 (polled mode). It fails if no encoder
> answers on the channel.

### Vive tracker (SteamVR)

Vive tracking needs **SteamVR** running on an **x86-64** Linux host. The OpenVR client library bundled with the
`openvr` package is x86-64 only, so on arm64 hosts `vive_tracker.py` imports but refuses to start.

**Setup** — installing Steam and SteamVR, turning off the headset requirement, connecting the (wired, so
unpaired) tracker and troubleshooting — is in [`docs/guides/steamvr-vive-tracker-setup.md`](../../docs/guides/steamvr-vive-tracker-setup.md).

`ViveReader` starts SteamVR itself if it is not already running, and waits up to 20 s for it.

#### Find the tracker's serial

```bash
python i2rt/yumi/vive_tracker.py
```

This lists every device SteamVR reports (class, serial, model), then redraws their live 4x4 poses. Tracker
serials look like `LHR-XXXXXXXX`. Moving a tracker and watching which matrix changes tells you which serial it is.

#### Frames

- **World frame.** OpenVR's standing universe, in metres, right-handed, **+Y up**. Its origin and yaw come from
  the SteamVR room setup.
- **Tracker frame.** The origin is at the centre of the mounting face, the face with the ¼″ screw and the pogo
  pins.
  - **+Z** points straight out of the mounting face, away from the tracker's dome.
  - **+Y** points from the screw toward the pogo pins.
  - **+X** completes a right-handed frame. Looking at the mounting face with the pogo pins below the screw, +X
    points left.

  ```text
         mounting face, seen from the front        side view
                    ( screw )                      dome )|--> +Z
          +X <-------  o                                 |  (out of the mounting face)
                       |
                       v +Y  (toward the pogo pins)
  ```

- `vive_pose` maps tracker coordinates to world coordinates. No tracker-to-camera or tracker-to-TCP transform is
  applied online; those are per-device extrinsics, applied offline. Their CAD design values are the `camera` and
  `tcp` frames of the device model, relative to its `vive` frame (this tracker frame), in
  [`i2rt/robot_models/yumi/`](../robot_models/yumi/README.md#frames).

### MF camera (UVC fisheye)

1. **Permissions:** the user must be in the `video` group: `sudo usermod -aG video "$USER"`, then log out and back
   in afterwards.

2. **Find the camera's USB serial.** The camera is opened by serial, not by `/dev/videoN`, because node numbers
   move between boots:

   ```bash
   udevadm info --query=property --name=/dev/video0 | grep ID_SERIAL_SHORT
   ```

   Any serial opens: it is only the lookup key, matched exactly, and its name is never checked.

3. **Check it:**

   ```bash
   python i2rt/camera/mf_camera.py --serial MF150_2026061109 --show
   ```

   This prints the measured FPS and shows the stream; press q or ESC to quit. The default capture mode is
   `800x600` MJPG at 30 fps.
   - A mode the camera does not support is refused rather than silently clamped. List the supported modes with
     `v4l2-ctl --device=/dev/videoN --list-formats-ext`.
   - If `import cv2` fails on a minimal host, install `libgl1 libglib2.0-0`.

**No intrinsics.** `MFCamera` is only a driver: it returns the raw fisheye frames and carries no K or distortion
coefficients.

---

## Python API

```python
Yumi(can_channel, tracker_serial=None, camera_serial=None, range_rad=5.1, camera_resolution=(800, 600), camera_fps=30,
     tracker_startup_timeout_s=30.0)
```

`Yumi` is an i2rt `Robot` (`get_robot_type() == RobotType.UMI`).

**Construction.**
1. A cheap check runs first: the camera serial must be plugged in. Without a `camera_serial`, the first plugged-in
   capture-capable camera is used (lowest `/dev/videoN`), whatever its serial, so on a laptop pass the serial to
   avoid opening the webcam.
2. Then all three device processes start in parallel. Without a `tracker_serial`, the Vive process follows the first
   tracker SteamVR reports (lowest device index), waiting up to `tracker_startup_timeout_s` for one to connect;
   `yumi.tracker_serial` names it.
3. Construction returns once each stream has delivered a first sample and the tracker has reported a valid pose.

Either choice is logged, and a warning names the others when more than one camera or tracker is connected.

| Method | Returns |
|---|---|
| `get_observations()` | `gripper_pos` `(1,)` float64, 0 = closed … 1 = open; `vive_pose` `(4, 4)` float64, tracker → OpenVR world (metres, +Y up), identity when invalid; `vive_pose_valid` `(1,)` bool. All are copies. |
| `get_image()` | the latest frame, `(H, W, 3)` uint8 **RGB** (a copy) |
| `get_joint_pos()` | `[gripper_pos]`; `num_dofs() == 1` (the passive gripper) |
| `command_*` | raise `NotImplementedError`, because the device is passive |
| `close()` | stops the camera, gripper and Vive processes; idempotent; also runs as a context manager |

**Per-stream samples with timestamps.** Every timestamp is `time.time()`, comparable across sensors.
- `yumi.gripper.latest()` is a `GripperState(timestamp, position, raw_angle, raw_velocity, buttons)`.
  `raw_angle` is the encoder angle as reported (rad, signed, not clipped), for checking the zero and the range.
  `buttons` is the encoder's two digital inputs, True = pressed.
- `yumi.vive.latest()` is a `TrackerPose(pose, valid, timestamp)`.
- `yumi.camera.latest()` is a `CameraFrame(image, timestamp)`. The timestamp is taken when the frame arrives,
  before decode.
- `yumi.camera.info` holds the camera's `serial`, `device_path`, and negotiated `resolution` and `fps`.
- Each stream's `sample_count` and `pid` are available for diagnostics. The device processes forward every
  gripper (250 Hz), Vive (250 Hz) and camera (30 fps) sample.

**Errors.**
- **Stale stream:** a getter raises `TimeoutError` when its stream's newest sample is too old (gripper 1 s,
  tracker 0.5 s, camera 0.2 s). Age is measured from when the device process *took* the sample, so a sample that
  sat in the pipe counts as old. The message carries the device's last error.
- **Lost pose:** losing the tracker's optical lock is normal for a handheld device. It shows up as
  `vive_pose_valid == False` with an identity pose, never as an exception. Rotation-only tracking counts as
  invalid, because its position drifts.
- **Dead process:** a device process that exited raises `RuntimeError`, as does any getter after `close()`.
- **Construction failure:** an error a device raises while starting is re-raised in the main process with its
  original type (or as a `RuntimeError` naming it, if the exception cannot be pickled), with the device process's
  traceback attached as the cause.

**Bring-up readout:**

```bash
python i2rt/yumi/yumi.py --can-channel can0 --tracker-serial LHR-XXXXXXXX --camera-serial MF150_2026061109
```

Both serials are optional; without them it uses the first camera and tracker detected, as above. This prints
`gripper_pos`, the two buttons, the tracker position, the valid flag and each stream's rate at about 10 Hz.

**Using the sensors without `Yumi`.** `UmiGripper`, `ViveReader` and `MFCamera` also work on their own, in the
calling process. `ViveReader` is a process-wide singleton, because OpenVR is process-global, so closing it closes
it for every holder in that process.
