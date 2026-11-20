"""The yumi handheld UMI device as an i2rt ``Robot``: passive gripper + Vive tracker + MF camera.

``Yumi`` owns no hardware logic of its own. It runs each sensor reader (``gripper.py``, ``vive_tracker.py``,
``i2rt/camera/mf_camera.py``) **in its own process**, streaming samples back over a pipe (``device_process.py``),
and puts the three streams behind the ``Robot`` protocol, so code written against ``Robot`` -- recorders,
visualizers, teleop plumbing -- can consume a handheld device the same way it consumes an arm.

Why it is shaped the way it is:

* **One process per device.** The camera's JPEG decode, the 250 Hz Vive poll and the 250 Hz CAN poll do not
  share a GIL, a stuck driver call stalls only its own stream, and a native crash is reported instead of taking
  the recording down. Each process pushes every new sample; this process keeps the latest of each.
* **One passive DOF.** The gripper is the only joint (``num_dofs() == 1``, ``get_joint_pos() == [gripper_pos]``);
  the tracker pose is an observation, not a degree of freedom. Nothing can be commanded, so every ``command_*``
  raises instead of inheriting the protocol's silent no-op, which would swallow a teleop command unnoticed.
* **Cheap checks first, then a parallel start.** A mistyped camera serial is caught
  in milliseconds, before any process starts. The three processes then start together, so the encoder
  validation (~2 s), the SteamVR start (up to 20 s) and the camera open overlap instead of adding up.
* **The tracker must be live before startup returns.** A freshly powered tracker can take a while to appear
  and gain optical lock, so construction waits up to ``tracker_startup_timeout_s`` for a valid pose.
* **Loud staleness, quiet pose loss.** A stale stream raises ``TimeoutError`` from the getters; losing optical
  lock is an ordinary event for a handheld device, so it is reported as ``vive_pose_valid == False`` with an
  identity pose, never as an exception. A device process that died raises ``RuntimeError``, as does every getter
  after ``close()``.
* **No extrinsics online.** ``vive_pose`` is the raw tracker pose in OpenVR's standing frame (metres,
  right-handed, +Y up); the tracker-to-camera and tracker-to-TCP transforms are per-device and applied offline.

Per-sensor samples with their timestamps are ``yumi.gripper.latest()`` (``GripperState``),
``yumi.vive.latest()`` (``TrackerPose``) and ``yumi.camera.latest()`` (``CameraFrame``); the camera's mode is in
``yumi.camera.info``. See ``i2rt/yumi/README.md`` for setup.

Bring-up readout (prints the gripper, its buttons, the tracker translation and each stream's rate at about 10 Hz)::

    python i2rt/yumi/yumi.py --can-channel can0 --tracker-serial LHR-XXXXXXXX --camera-serial MF150_2026061109
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from types import TracebackType

import numpy as np
import tyro

from i2rt.camera.mf_camera import (
    MF_DEFAULT_RESOLUTION,
    CameraFrame,
    MFCamera,
    find_mf_camera_serials,
    find_video_device_by_serial,
)
from i2rt.robots.robot import Robot, RobotType
from i2rt.utils.utils import override_log_level
from i2rt.yumi.device_process import DeviceStream, close_all, wait_ready_all
from i2rt.yumi.gripper import DEFAULT_RANGE_RAD, GripperState, UmiGripper
from i2rt.yumi.vive_tracker import TrackerPose, ViveTracker

logger = logging.getLogger(__name__)

# How long each device may take to come up in its process. The Vive budget covers the SteamVR start (up to 20 s),
# OpenVR init and the first poll; the others cover the encoder validation / camera open plus a first sample.
_GRIPPER_READY_TIMEOUT_S = 30.0
_VIVE_READY_TIMEOUT_S = 60.0
_CAMERA_READY_TIMEOUT_S = 30.0
# How long without a new sample before a stream counts as stale in this process. Each device also enforces its
# own staleness and then stops producing samples, which is what trips these.
_GRIPPER_STALE_S = 1.0
_VIVE_STALE_S = 0.5
_CAMERA_STALE_S = 0.2
# The Vive and gripper processes poll their device at 1 kHz, the camera at 500 Hz, forwarding each sample they see
# once. That catches every gripper and Vive (250 Hz each) and camera (30 fps) sample.
_FAST_SAMPLE_PERIOD_S = 0.001
_CAMERA_SAMPLE_PERIOD_S = 0.002
# How often the startup wait re-checks the tracker.
_TRACKER_WAIT_POLL_S = 0.05
# The CLI readout period (~10 Hz).
_READOUT_PERIOD_S = 0.1

_NOT_COMMANDABLE = "Yumi is a passive handheld device and cannot be commanded"


def _first_mf_camera_serial() -> str:
    """The serial of the first plugged-in camera (lowest ``/dev/videoN``), or raise ``RuntimeError``."""
    serials = find_mf_camera_serials()
    if not serials:
        raise RuntimeError(
            "No camera found: no capture-capable video device has a USB serial. "
            "Plug one in, or pass its serial (check a node with: udevadm info --query=property "
            "--name=/dev/videoX | grep ID_SERIAL_SHORT)."
        )
    if len(serials) > 1:
        logger.warning(
            f"{len(serials)} cameras are plugged in ({', '.join(serials)}); using the first, {serials[0]}. Pass "
            "its serial to pick another."
        )
    else:
        logger.info(f"Using camera {serials[0]}, the only one plugged in.")
    return serials[0]


class Yumi(Robot):
    """The yumi handheld device: a passive UMI gripper, a Vive tracker and an i2rt MF camera, one process each.

    Attributes:
        gripper: the gripper stream; ``latest()`` is a ``GripperState`` (position, raw angle, raw velocity, buttons,
            timestamp).
        vive: the tracker stream; ``latest()`` is a ``TrackerPose`` for ``tracker_serial``.
        camera: the camera stream; ``latest()`` is a ``CameraFrame``. ``camera.info`` holds ``serial``,
            ``device_path`` and the negotiated ``resolution`` and ``fps``.
        tracker_serial: the Vive tracker whose pose ``get_observations`` reports: the one asked for, or the first
            one SteamVR reported.
    """

    def __init__(
        self,
        can_channel: str,
        tracker_serial: str | None = None,
        camera_serial: str | None = None,
        range_rad: float = DEFAULT_RANGE_RAD,
        camera_resolution: tuple[int, int] = MF_DEFAULT_RESOLUTION,
        camera_fps: int = 30,
        tracker_startup_timeout_s: float = 30.0,
    ) -> None:
        """Check the camera serial, then start the three device processes and wait for them.

        Args:
            can_channel: SocketCAN channel of the gripper's handle encoder, e.g. ``"can0"``.
            tracker_serial: serial of the Vive tracker on the handle (``python i2rt/yumi/vive_tracker.py`` lists them).
                ``None`` follows the first tracker SteamVR reports, waiting up to ``tracker_startup_timeout_s`` for one.
            camera_serial: the MF camera's USB short serial, e.g. ``"MF150_2026061109"``. ``None`` opens the first
                plugged-in capture-capable camera, whatever its serial.
            range_rad: encoder angle from the full-open zero that reads as fully closed.
            camera_resolution: camera capture mode ``(width, height)``.
            camera_fps: requested camera frame rate.
            tracker_startup_timeout_s: how long to wait for the tracker to report a valid pose (and, without a
                ``tracker_serial``, as long again for a tracker to connect).

        Raises:
            RuntimeError: the camera serial matches no capture node (or, without one, no camera is plugged in),
                the tracker did not report a valid pose in time, or a device process failed or exited.
            TimeoutError: a device did not come up in time, or, without a ``tracker_serial``, no tracker connected.
            Exception: any error a device's constructor raised in its process, re-raised here with its type (or as a
                ``RuntimeError`` naming it, if the exception cannot be pickled).
        """
        self._closed = False
        resolution = (int(camera_resolution[0]), int(camera_resolution[1]))

        # A cheap check with no side effects, before the encoder validation and the SteamVR wait.
        if camera_serial is None:
            camera_serial = _first_mf_camera_serial()
        find_video_device_by_serial(camera_serial)

        streams: list[DeviceStream] = []
        try:
            self.gripper: DeviceStream[GripperState] = DeviceStream(
                "gripper",
                UmiGripper,
                {"channel": can_channel, "range_rad": range_rad},
                sample_method="get_state",
                stale_after_s=_GRIPPER_STALE_S,
                sample_period_s=_FAST_SAMPLE_PERIOD_S,
            )
            streams.append(self.gripper)
            # The tracker is picked in the Vive process, the only one with OpenVR running; it reports its choice.
            self.vive: DeviceStream[TrackerPose] = DeviceStream(
                "vive",
                ViveTracker,
                {"serial": tracker_serial, "detect_timeout_s": tracker_startup_timeout_s},
                sample_method="get_pose",
                stale_after_s=_VIVE_STALE_S,
                sample_period_s=_FAST_SAMPLE_PERIOD_S,
                info_attrs=("serial",),
            )
            streams.append(self.vive)
            self.camera: DeviceStream[CameraFrame] = DeviceStream(
                "camera",
                MFCamera,
                {"serial": camera_serial, "resolution": resolution, "fps": camera_fps},
                sample_method="read",
                stale_after_s=_CAMERA_STALE_S,
                sample_period_s=_CAMERA_SAMPLE_PERIOD_S,
                info_attrs=("serial", "device_path", "resolution", "fps"),
            )
            streams.append(self.camera)
            # All three are already starting and are waited on together, so the first failure surfaces at once
            # rather than after the slower devices come up. Each is ready once a fresh first sample has arrived, so
            # the getters work the moment construction returns -- except the Vive, whose first sample needs the
            # tracker; the tracker wait below covers it with a clearer error.
            detect_s = tracker_startup_timeout_s if tracker_serial is None else 0.0
            wait_ready_all(
                [
                    (self.gripper, _GRIPPER_READY_TIMEOUT_S, True),
                    (self.vive, _VIVE_READY_TIMEOUT_S + detect_s, False),
                    (self.camera, _CAMERA_READY_TIMEOUT_S, True),
                ]
            )
            self.tracker_serial: str = self.vive.info["serial"]
            self._wait_for_tracker(tracker_startup_timeout_s)
        except BaseException:
            try:
                close_all(list(reversed(streams)))
            except Exception:
                logger.exception("Closing the device streams after a failed start failed")
            raise

    def _wait_for_tracker(self, timeout_s: float) -> None:
        """Block until the tracker stream carries a valid pose; raise ``RuntimeError`` on timeout."""
        logger.info(f"Waiting up to {timeout_s:.0f} s for Vive tracker {self.tracker_serial} to report a valid pose")
        deadline = time.monotonic() + timeout_s
        seen = False
        while True:
            try:
                if self.vive.latest().valid:
                    return
                seen = True
            except TimeoutError:
                pass  # no pose yet: the tracker has not been seen (the Vive process reports why in last_error)
            if time.monotonic() >= deadline:
                break
            time.sleep(_TRACKER_WAIT_POLL_S)

        if seen:
            reason = (
                "was seen but its pose never became valid -- check that it has line of sight to a base station "
                "and that its LED is solid green"
            )
        else:
            reason = (
                "was never seen -- check the serial (python i2rt/yumi/vive_tracker.py lists them) and the tracker's "
                "USB cable"
            )
        raise RuntimeError(
            f"Vive tracker {self.tracker_serial!r} {reason}; waited {timeout_s:.1f} s. Last report from the Vive "
            f"process: {self.vive.last_error}. See docs/guides/steamvr-vive-tracker-setup.md."
        )

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Yumi is closed")

    # ------------------------------------------------------------------ Robot protocol

    def num_dofs(self) -> int:
        """The gripper; the tracker pose is an observation, not a DOF."""
        return 1

    def get_joint_pos(self) -> np.ndarray:
        """``[gripper_pos]``, 0 = closed ... 1 = open. Raises ``TimeoutError`` if the gripper stream is stale."""
        self._check_open()
        return np.array([self.gripper.latest().position], dtype=np.float64)

    def get_observations(self) -> dict[str, np.ndarray]:
        """The gripper opening and the tracker pose, as fresh arrays.

        Returns:
            ``"gripper_pos"``: shape ``(1,)`` float64, 0 = closed ... 1 = open.
            ``"vive_pose"``: ``(4, 4)`` float64, tracker -> OpenVR standing world (metres, +Y up); identity when
            invalid.
            ``"vive_pose_valid"``: shape ``(1,)`` bool.

        Raises:
            TimeoutError: the gripper or tracker stream is stale (losing the pose is ``vive_pose_valid``, not this).
            RuntimeError: this Yumi is closed, or a device process exited.
        """
        self._check_open()
        gripper = self.gripper.latest()
        tracker = self.vive.latest()
        return {
            "gripper_pos": np.array([gripper.position], dtype=np.float64),
            "vive_pose": np.array(tracker.pose, dtype=np.float64, copy=True),
            "vive_pose_valid": np.array([tracker.valid], dtype=bool),
        }

    def get_image(self) -> np.ndarray:
        """A copy of the latest camera frame, ``HxWx3`` uint8 RGB. Raises ``TimeoutError`` if the frame is stale."""
        self._check_open()
        return self.camera.latest().image.copy()

    def get_robot_type(self) -> RobotType:
        return RobotType.UMI

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        raise NotImplementedError(_NOT_COMMANDABLE)

    def command_target_vel(self, joint_vel: np.ndarray) -> None:
        raise NotImplementedError(_NOT_COMMANDABLE)

    def command_joint_state(self, joint_state: dict[str, np.ndarray]) -> None:
        raise NotImplementedError(_NOT_COMMANDABLE)

    def close(self) -> None:
        """Stop the camera, gripper and Vive processes together, each even if another fails. Idempotent."""
        if self._closed:
            return
        self._closed = True
        close_all([self.camera, self.gripper, self.vive])

    def __enter__(self) -> Yumi:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


@dataclass
class _CliArgs:
    """Bring-up readout for the yumi: print the gripper, its buttons, the tracker translation and stream rates at ~10 Hz."""

    tracker_serial: str | None = None
    """Serial of the Vive tracker on the handle (python i2rt/yumi/vive_tracker.py lists them); defaults to the first
    tracker SteamVR reports."""
    camera_serial: str | None = None
    """USB short serial of the MF camera, e.g. MF150_2026061109; defaults to the first camera plugged in."""
    can_channel: str = "can0"
    """CAN channel of the gripper's handle encoder."""
    range_rad: float = DEFAULT_RANGE_RAD
    """Encoder angle (rad) from the full-open zero that reads as fully closed."""


def main(args: _CliArgs) -> None:
    with Yumi(
        args.can_channel,
        args.tracker_serial,
        args.camera_serial,
        range_rad=args.range_rad,
    ) as yumi:
        info = yumi.camera.info
        width, height = info["resolution"]
        print(
            f"Yumi up: gripper on {args.can_channel}, tracker {yumi.tracker_serial}, camera {info['serial']} at "
            f"{info['device_path']} ({width}x{height} @ {info['fps']} fps). Ctrl-C to quit."
        )
        streams = (yumi.gripper, yumi.vive, yumi.camera)
        counts = [stream.sample_count for stream in streams]
        window_start = time.monotonic()
        try:
            while True:
                time.sleep(_READOUT_PERIOD_S)
                now = time.monotonic()
                new_counts = [stream.sample_count for stream in streams]
                rates = [(new - old) / (now - window_start) for new, old in zip(new_counts, counts, strict=True)]
                counts, window_start = new_counts, now
                obs = yumi.get_observations()
                buttons = yumi.gripper.latest().buttons
                x, y, z = obs["vive_pose"][:3, 3]
                print(
                    f"\rgripper_pos {obs['gripper_pos'][0]:5.3f}  buttons {int(buttons[0])} {int(buttons[1])}  "
                    f"pose xyz [{x:+7.3f} {y:+7.3f} {z:+7.3f}] m  "
                    f"valid {bool(obs['vive_pose_valid'][0])!s:<5}  "
                    f"rates: gripper {rates[0]:5.0f} Hz, vive {rates[1]:5.0f} Hz, camera {rates[2]:4.1f} fps",
                    end="",
                    flush=True,
                )
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    override_log_level(logging.INFO)
    main(tyro.cli(_CliArgs))
