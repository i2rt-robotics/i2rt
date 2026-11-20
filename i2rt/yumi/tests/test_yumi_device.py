"""Tests for ``Yumi``, the ``Robot`` that runs the yumi's gripper, Vive tracker and MF camera one process each.

``Yumi`` owns no hardware logic -- each sensor module is tested against its own fake bus or runtime, and the
process/pipe machinery in ``test_yumi_device_process.py``. What it does own is the composition, and each piece of
that is easy to get subtly wrong: the ``Robot`` surface downstream code programs against (a passive device that
silently accepted ``command_joint_pos`` would hide a teleop bug), the observation shapes, dtypes and copies, the
cheap checks that must run before an EEPROM-writing encoder validation and a 20 s SteamVR wait, the startup wait for
the tracker, and that a failure halfway through construction stops every device already started.

Most tests replace ``DeviceStream`` with an in-process fake that builds the (fake) part exactly as the child would,
so they stay fast; the three part classes and the udev lookup are fakes inside the ``yumi`` module, logging when
they ran. One test at the end runs the real processes, to pin that each device really lives in a process of its own.
"""

import copy
import functools
import os
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest

from i2rt.camera.mf_camera import CameraFrame
from i2rt.robots.robot import Robot, RobotType
from i2rt.yumi import yumi as yumi_module
from i2rt.yumi.device_process import wait_ready_all
from i2rt.yumi.gripper import GripperState
from i2rt.yumi.tests import yumi_fake_devices
from i2rt.yumi.vive_tracker import TrackedDeviceInfo, TrackerPose
from i2rt.yumi.yumi import Yumi

_CAN = "can_fake"
_TRACKER = "LHR-0000TEST"
_CAMERA = "MF150_2026061109"
_POSE = np.array(
    [[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 1.2], [0.0, 0.0, 1.0, -0.3], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
)
# Generous against a 0.2 s patched timeout: shared CI runners stall.
_MAX_TIMEOUT_WALL_S = 5.0


class _Rig:
    """What the fakes report, and an ordered log of what was built and closed."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.gripper_position = 0.25
        self.gripper_error: BaseException | None = None
        self.poses: dict[str, TrackerPose] = {_TRACKER: TrackerPose(_POSE.copy(), True, time.time())}
        self.devices = [TrackedDeviceInfo(0, "tracker", _TRACKER, "VIVE Tracker 3.0", True)]
        self.invalid_polls = 0
        self.pose_error: BaseException | None = None
        self.video_lookup_error: BaseException | None = None
        self.camera_error: BaseException | None = None
        self.camera_serials = [_CAMERA]
        self.close_errors: dict[str, BaseException] = {}
        self.stream_close_errors: dict[str, BaseException] = {}
        self.streams: list[_FakeStream] = []
        self.image = np.arange(600 * 800 * 3, dtype=np.uint32).reshape(600, 800, 3).astype(np.uint8)
        self.parts: dict[str, _FakePart] = {}


class _FakePart:
    name = ""

    def __init__(self, rig: _Rig) -> None:
        self.rig = rig
        self.close_calls = 0
        rig.parts[self.name] = self
        rig.events.append(self.name)

    def close(self) -> None:
        self.close_calls += 1
        self.rig.events.append(f"close {self.name}")
        error = self.rig.close_errors.get(self.name)
        if error is not None:
            raise error

    def __enter__(self) -> "_FakePart":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _FakeGripper(_FakePart):
    name = "gripper"

    def __init__(self, rig: _Rig, channel: str, range_rad: float) -> None:
        super().__init__(rig)
        self.channel = channel
        self.range_rad = range_rad

    def get_state(self) -> GripperState:
        if self.rig.gripper_error is not None:
            raise self.rig.gripper_error
        position = self.rig.gripper_position
        return GripperState(
            timestamp=time.time(),
            position=position,
            raw_angle=(1.0 - position) * self.range_rad,
            raw_velocity=0.0,
            buttons=(False, False),
        )


class _FakeVive(_FakePart):
    """Stands in for ``ViveTracker``: without a serial it follows the first tracker in ``rig.devices``."""

    name = "vive"

    def __init__(self, rig: _Rig, serial: str | None = None, detect_timeout_s: float = 30.0) -> None:
        super().__init__(rig)
        self.polls = 0
        self.detect_timeout_s = detect_timeout_s
        self.serial = (
            serial if serial is not None else next(d.serial for d in rig.devices if d.device_class == "tracker")
        )

    def get_pose(self) -> TrackerPose:
        self.polls += 1
        if self.rig.pose_error is not None:
            raise self.rig.pose_error
        if self.serial not in self.rig.poses:
            raise KeyError(f"Vive device {self.serial!r} has not been seen; known serials: {sorted(self.rig.poses)}")
        if self.polls <= self.rig.invalid_polls:
            return TrackerPose(np.eye(4), False, time.time())
        # The stored record itself, not a copy: Yumi's own copy is what the copy test pins.
        return self.rig.poses[self.serial]


class _FakeCamera(_FakePart):
    name = "camera"

    def __init__(
        self,
        rig: _Rig,
        serial: str,
        resolution: tuple[int, int],
        fps: int,
    ) -> None:
        if rig.camera_error is not None:
            rig.events.append("camera failed")
            raise rig.camera_error
        super().__init__(rig)
        self.serial = serial
        self.device_path = "/dev/video0"
        self.resolution = resolution
        self.fps = fps
        self.last_frame: CameraFrame | None = None

    def read(self) -> CameraFrame:
        self.last_frame = CameraFrame(image=self.rig.image.copy(), timestamp=time.time())
        return self.last_frame


class _FakeStream:
    """In-process stand-in for ``DeviceStream``: builds the part at spawn time and samples it on ``latest()``.

    It keeps the real stream's contract: construction errors surface from ``wait_ready`` (after which the stream
    is closed), a failing sample call reads as a stale stream carrying that error, a part's own close error stays
    in the "child" (logged, not raised), and every sample is its own copy, as an unpickled one would be.
    """

    def __init__(
        self,
        rig: _Rig,
        name: str,
        factory: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
        *,
        sample_method: str,
        stale_after_s: float,
        sample_period_s: float = 0.001,
        info_attrs: tuple[str, ...] = (),
    ) -> None:
        self.rig = rig
        rig.streams.append(self)
        self.name = name
        self.stale_after_s = stale_after_s
        self.info: dict[str, Any] = {}
        self.last_error: str | None = None
        self.sample_count = 0
        self.pid = os.getpid()
        self._sample_method, self._info_attrs = sample_method, info_attrs
        self._closed = False
        self._finished = False
        self._polled = False
        self._device: Any = None
        self._init_error: BaseException | None = None
        try:
            self._device = factory(**(kwargs or {}))
        except BaseException as exc:
            self._init_error = exc

    def poll_ready(self, first_sample: bool = True) -> bool:
        if not self._polled:
            self._polled = True
            self.rig.events.append(f"wait {self.name} first_sample={first_sample}")
        if self._init_error is not None:
            raise self._init_error
        self.info = {attr: getattr(self._device, attr) for attr in self._info_attrs}
        return True

    def describe_pending(self, timeout_s: float) -> str:
        return f"The {self.name} device did not come up within {timeout_s:.1f} s"

    def wait_ready(self, timeout_s: float, first_sample: bool = True) -> dict[str, Any]:
        wait_ready_all([(self, timeout_s, first_sample)])
        return self.info

    def latest(self) -> Any:
        if self._closed:
            raise RuntimeError(f"The {self.name} stream is closed")
        try:
            sample = getattr(self._device, self._sample_method)()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise TimeoutError(f"The {self.name} stream is stale; last error: {self.last_error}") from None
        self.sample_count += 1
        return copy.deepcopy(sample)

    def request_close(self) -> None:
        self._closed = True

    def finish_close(self) -> None:
        self._closed = True
        if self._finished:
            return
        self._finished = True
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass  # the real child logs a part's close error; it never reaches the parent
        error = self.rig.stream_close_errors.get(self.name)
        if error is not None:
            raise error

    def close(self) -> None:
        self.request_close()
        self.finish_close()


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> _Rig:
    rig = _Rig()

    def find_video_device_by_serial(serial: str) -> str:
        rig.events.append("video lookup")
        if rig.video_lookup_error is not None:
            raise rig.video_lookup_error
        return "/dev/video0"

    monkeypatch.setattr(yumi_module, "DeviceStream", functools.partial(_FakeStream, rig))
    monkeypatch.setattr(yumi_module, "UmiGripper", functools.partial(_FakeGripper, rig))

    def find_mf_camera_serials() -> list[str]:
        rig.events.append("camera detection")
        return list(rig.camera_serials)

    monkeypatch.setattr(yumi_module, "ViveTracker", functools.partial(_FakeVive, rig))
    monkeypatch.setattr(yumi_module, "MFCamera", functools.partial(_FakeCamera, rig))
    monkeypatch.setattr(yumi_module, "find_video_device_by_serial", find_video_device_by_serial)
    monkeypatch.setattr(yumi_module, "find_mf_camera_serials", find_mf_camera_serials)
    return rig


def _make_yumi(camera_serial: str = _CAMERA, **kwargs: Any) -> Yumi:
    kwargs.setdefault("tracker_startup_timeout_s", 2.0)
    return Yumi(_CAN, _TRACKER, camera_serial, **kwargs)


@pytest.fixture
def yumi(rig: _Rig) -> Iterator[Yumi]:
    device = _make_yumi()
    try:
        yield device
    finally:
        device.close()


# ---------------------------------------------------------------------- Robot surface and observations


def test_yumi_is_a_passive_one_dof_robot(yumi: Yumi) -> None:
    assert isinstance(yumi, Robot)
    assert yumi.num_dofs() == 1
    assert yumi.joint_pos_spec().shape == (1,)
    assert yumi.get_robot_type() is RobotType.UMI
    np.testing.assert_array_equal(yumi.get_joint_pos(), yumi.get_observations()["gripper_pos"])
    commands: list[Callable[[], None]] = [
        lambda: yumi.command_joint_pos(np.zeros(1)),
        lambda: yumi.command_target_vel(np.zeros(1)),
        lambda: yumi.command_joint_state({"pos": np.zeros(1), "vel": np.zeros(1)}),
    ]
    for command in commands:
        with pytest.raises(NotImplementedError, match="passive"):
            command()


def test_observation_keys_shapes_and_dtypes(yumi: Yumi, rig: _Rig) -> None:
    obs = yumi.get_observations()

    assert set(obs) == {"gripper_pos", "vive_pose", "vive_pose_valid"}
    assert obs["gripper_pos"].shape == (1,)
    assert obs["gripper_pos"].dtype == np.float64
    assert obs["gripper_pos"][0] == pytest.approx(rig.gripper_position)
    assert obs["vive_pose"].shape == (4, 4)
    assert obs["vive_pose"].dtype == np.float64
    np.testing.assert_array_equal(obs["vive_pose"], _POSE)
    assert obs["vive_pose_valid"].shape == (1,)
    assert obs["vive_pose_valid"].dtype == np.bool_
    assert bool(obs["vive_pose_valid"][0])


def test_observations_are_copies(yumi: Yumi, rig: _Rig) -> None:
    first = yumi.get_observations()
    for value in first.values():
        value[...] = 0

    np.testing.assert_array_equal(rig.poses[_TRACKER].pose, _POSE)
    second = yumi.get_observations()
    np.testing.assert_array_equal(second["vive_pose"], _POSE)
    assert second["gripper_pos"][0] == pytest.approx(rig.gripper_position)
    assert bool(second["vive_pose_valid"][0])


def test_lost_pose_is_reported_through_the_valid_flag(yumi: Yumi, rig: _Rig) -> None:
    rig.poses[_TRACKER] = TrackerPose(np.eye(4), False, time.time())

    obs = yumi.get_observations()

    assert not bool(obs["vive_pose_valid"][0])
    np.testing.assert_array_equal(obs["vive_pose"], np.eye(4))


@pytest.mark.parametrize("stale_part", ["gripper", "vive"])
def test_stale_stream_raises_timeout(yumi: Yumi, rig: _Rig, stale_part: str) -> None:
    error = TimeoutError(f"{stale_part} stream is stale")
    if stale_part == "gripper":
        rig.gripper_error = error
    else:
        rig.pose_error = error

    with pytest.raises(TimeoutError, match=stale_part):
        yumi.get_observations()


def test_get_image_returns_a_copy_of_the_latest_frame(yumi: Yumi, rig: _Rig) -> None:
    image = yumi.get_image()

    camera = rig.parts["camera"]
    assert isinstance(camera, _FakeCamera) and camera.last_frame is not None
    assert image.shape == (600, 800, 3) and image.dtype == np.uint8
    np.testing.assert_array_equal(image, rig.image)
    image[...] = 0
    np.testing.assert_array_equal(yumi.get_image(), rig.image)


def test_camera_info_is_reported_from_its_process(yumi: Yumi) -> None:
    info = yumi.camera.info
    assert info["serial"] == _CAMERA
    assert info["device_path"] == "/dev/video0"
    assert tuple(info["resolution"]) == (800, 600)
    assert info["fps"] == 30


# ---------------------------------------------------------------------- construction


def test_parts_are_built_in_order_with_the_constructor_arguments(rig: _Rig) -> None:
    with _make_yumi(range_rad=3.0, camera_resolution=(1280, 1024), camera_fps=60):
        pass

    assert rig.events[:4] == ["video lookup", "gripper", "vive", "camera"]
    # Getters must work the moment construction returns, so the gripper and camera wait for a first sample; the
    # Vive does not, because the tracker wait (with the clearer error) covers its first sample.
    assert rig.events[4:7] == [
        "wait gripper first_sample=True",
        "wait vive first_sample=False",
        "wait camera first_sample=True",
    ]
    gripper, camera = rig.parts["gripper"], rig.parts["camera"]
    assert isinstance(gripper, _FakeGripper) and isinstance(camera, _FakeCamera)
    assert (gripper.channel, gripper.range_rad) == (_CAN, 3.0)
    assert (camera.serial, camera.resolution, camera.fps) == (_CAMERA, (1280, 1024), 60)
    assert [stream.name for stream in rig.streams] == ["gripper", "vive", "camera"]


def test_startup_waits_for_the_tracker_to_become_valid(rig: _Rig) -> None:
    rig.invalid_polls = 3

    with _make_yumi():
        pass

    vive = rig.parts["vive"]
    assert isinstance(vive, _FakeVive) and vive.polls > rig.invalid_polls
    assert "camera" in rig.events


def test_without_serials_the_first_detected_camera_and_tracker_are_used(rig: _Rig) -> None:
    rig.camera_serials = [_CAMERA, "MF118_2026061101"]
    rig.devices = [
        TrackedDeviceInfo(1, "tracking_reference", "LHB-BASE", "Valve SR Imp", True),
        TrackedDeviceInfo(3, "tracker", _TRACKER, "VIVE Tracker 3.0", True),
        TrackedDeviceInfo(4, "tracker", "LHR-SECOND", "VIVE Tracker 3.0", True),
    ]

    with Yumi(_CAN, tracker_startup_timeout_s=2.0) as device:
        assert device.tracker_serial == _TRACKER
        np.testing.assert_array_equal(device.get_observations()["vive_pose"], _POSE)
        camera = rig.parts["camera"]
        assert isinstance(camera, _FakeCamera) and camera.serial == _CAMERA
    # The detected camera still goes through the cheap check, before any device starts.
    assert rig.events[:3] == ["camera detection", "video lookup", "gripper"]


def test_no_camera_detected_fails_before_any_device_starts(rig: _Rig) -> None:
    rig.camera_serials = []

    with pytest.raises(RuntimeError, match="No camera found"):
        Yumi(_CAN, _TRACKER)

    assert rig.events == ["camera detection"]


def test_a_given_serial_skips_detection(rig: _Rig) -> None:
    with _make_yumi() as device:
        assert device.tracker_serial == _TRACKER
    assert "camera detection" not in rig.events


@pytest.mark.parametrize("seen", [False, True], ids=["never_seen", "seen_but_invalid"])
def test_tracker_timeout_raises_and_releases_the_built_parts(rig: _Rig, seen: bool) -> None:
    rig.devices = [
        TrackedDeviceInfo(0, "tracker", "LHR-OTHER", "VIVE Tracker 3.0", True),
        TrackedDeviceInfo(1, "tracking_reference", "LHB-BASE", "Valve SR Imp", True),
    ]
    rig.poses = {"LHR-OTHER": TrackerPose(_POSE.copy(), True, time.time())}
    if seen:
        rig.poses[_TRACKER] = TrackerPose(np.eye(4), False, time.time())

    started = time.monotonic()
    with pytest.raises(RuntimeError) as info:
        _make_yumi(tracker_startup_timeout_s=0.2)

    assert time.monotonic() - started < _MAX_TIMEOUT_WALL_S
    message = str(info.value)
    assert _TRACKER in message
    assert ("never became valid" in message) is seen
    assert ("never seen" in message) is not seen
    if not seen:
        # The Vive process's last report (its get_pose KeyError) lists what SteamVR did see.
        assert "LHR-OTHER" in message
    # The devices start in parallel, so the camera is already up and must be stopped too.
    assert all(rig.parts[name].close_calls == 1 for name in ("gripper", "vive", "camera"))


@pytest.mark.parametrize("error", [RuntimeError("driver refused 800x600"), KeyboardInterrupt()])
def test_camera_failure_closes_the_gripper_and_vive(rig: _Rig, error: BaseException) -> None:
    rig.camera_error = error

    with pytest.raises(type(error)):
        _make_yumi()

    assert "camera failed" in rig.events
    assert rig.parts["gripper"].close_calls == 1
    assert rig.parts["vive"].close_calls == 1


def test_unplugged_camera_serial_fails_before_the_gripper_is_built(rig: _Rig) -> None:
    rig.video_lookup_error = RuntimeError("MF camera with serial 'MF150_2026061109' not found")

    with pytest.raises(RuntimeError, match="not found"):
        _make_yumi()

    assert rig.parts == {}


def test_a_serial_without_a_lens_token_starts(rig: _Rig) -> None:
    with _make_yumi("CAM_2026061109"):
        pass

    camera = rig.parts["camera"]
    assert isinstance(camera, _FakeCamera) and camera.serial == "CAM_2026061109"


# ---------------------------------------------------------------------- close


def test_close_order_is_camera_gripper_vive_and_close_is_idempotent(rig: _Rig) -> None:
    device = _make_yumi()
    built = len(rig.events)

    device.close()
    device.close()

    assert rig.events[built:] == ["close camera", "close gripper", "close vive"]
    for method in (device.get_observations, device.get_joint_pos, device.get_image):
        with pytest.raises(RuntimeError, match="closed"):
            method()


def test_close_stops_every_stream_even_when_one_fails(rig: _Rig) -> None:
    device = _make_yumi()
    rig.close_errors["gripper"] = RuntimeError("CAN bus gone")  # a part's own error stays in its process
    rig.stream_close_errors["camera"] = OSError("pipe broken")  # a stream's close failing must not skip the rest

    with pytest.raises(OSError, match="pipe broken"):
        device.close()

    assert all(part.close_calls == 1 for part in rig.parts.values())


def test_context_manager_closes_every_part(rig: _Rig) -> None:
    with _make_yumi() as device:
        assert isinstance(device, Yumi)
        assert all(part.close_calls == 0 for part in rig.parts.values())

    assert set(rig.parts) == {"gripper", "vive", "camera"}
    assert all(part.close_calls == 1 for part in rig.parts.values())


# ---------------------------------------------------------------------- real device processes


def test_each_device_runs_in_its_own_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real DeviceStream, with importable fake devices: three child processes, three live streams."""
    monkeypatch.setattr(yumi_module, "UmiGripper", yumi_fake_devices.FakeGripper)
    monkeypatch.setattr(yumi_module, "ViveTracker", yumi_fake_devices.FakeViveTracker)
    monkeypatch.setattr(yumi_module, "MFCamera", yumi_fake_devices.FakeCamera)
    monkeypatch.setattr(yumi_module, "find_video_device_by_serial", lambda serial: "/dev/video-fake")

    with Yumi(_CAN, "LHR-0000TEST", _CAMERA, camera_resolution=(320, 240)) as device:
        pids = {device.gripper.pid, device.vive.pid, device.camera.pid}
        assert len(pids) == 3 and os.getpid() not in pids and None not in pids

        obs = device.get_observations()
        assert obs["gripper_pos"][0] == pytest.approx(0.25)
        np.testing.assert_array_equal(obs["vive_pose"], yumi_fake_devices.POSE)
        assert bool(obs["vive_pose_valid"][0])
        image = device.get_image()
        assert image.shape == (240, 320, 3) and int(image[0, 0, 0]) == 7
        assert device.camera.info["device_path"] == "/dev/video-fake"

        # Every stream keeps flowing: new samples keep arriving from each process.
        counts = [stream.sample_count for stream in (device.gripper, device.vive, device.camera)]
        time.sleep(0.3)
        assert all(
            stream.sample_count > count
            for stream, count in zip((device.gripper, device.vive, device.camera), counts, strict=True)
        )
        processes = [device.gripper._process, device.vive._process, device.camera._process]

    assert all(not process.is_alive() for process in processes)
    assert all(process.exitcode == 0 for process in processes)
