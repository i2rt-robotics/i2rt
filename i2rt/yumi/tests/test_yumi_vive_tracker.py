"""Tests for the yumi Vive reader, against a fake ``openvr`` module (no SteamVR, no tracker).

The reader's failure modes are all quiet ones: a rotation-only pose that still says "valid", a tracker that
drops out and comes back under a different ordinal name, a poll thread that died while the last pose keeps
being served, a singleton that cannot be reopened after close, and SteamVR's quit killing the whole process.
Each test pins one of those down, using ctypes structs laid out like openvr's so the matrix conversion runs
on the same kind of object the real binding returns.
"""

import ctypes
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from i2rt.yumi import vive_tracker
from i2rt.yumi.vive_tracker import TrackedDeviceInfo, TrackerPose, ViveReader, ViveTracker

_RUNNING_OK = 200
_FALLBACK_ROTATION_ONLY = 300
_CLASS_HMD, _CLASS_CONTROLLER, _CLASS_TRACKER, _CLASS_BASE = 1, 2, 3, 4
_ACTIVATED, _DEACTIVATED, _QUIT = 100, 101, 700
_PROP_MODEL, _PROP_SERIAL = 1001, 1002

# A tracker pose: 90 deg about world +Y, then a translation, as OpenVR's row-major 3x4.
_POSE_3X4 = np.array([[0.0, 0.0, 1.0, 0.5], [0.0, 1.0, 0.0, 1.25], [-1.0, 0.0, 0.0, -0.75]])


class _Matrix34(ctypes.Structure):
    _fields_ = [("m", (ctypes.c_float * 4) * 3)]


class _Vector3(ctypes.Structure):
    _fields_ = [("v", ctypes.c_float * 3)]


class _Pose(ctypes.Structure):
    _fields_ = [
        ("mDeviceToAbsoluteTracking", _Matrix34),
        ("vVelocity", _Vector3),
        ("vAngularVelocity", _Vector3),
        ("eTrackingResult", ctypes.c_uint),
        ("bPoseIsValid", ctypes.c_ubyte),
        ("bDeviceIsConnected", ctypes.c_ubyte),
    ]


class _Event(ctypes.Structure):
    _fields_ = [("eventType", ctypes.c_uint32), ("trackedDeviceIndex", ctypes.c_uint32)]


class _FakeOpenVRError(Exception):
    pass


class _FakeTrackedPropertyError(_FakeOpenVRError):
    pass


class _FakeInitError(_FakeOpenVRError):
    pass


@dataclass
class _Device:
    device_class: int
    serial: str | None
    model: str | None = "fake model"
    matrix: np.ndarray = field(default_factory=_POSE_3X4.copy)
    pose_valid: bool = True
    tracking_result: int = _RUNNING_OK
    connected: bool = True


class _FakeVRSystem:
    """The subset of ``openvr.IVRSystem`` the reader calls."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.devices: dict[int, _Device] = {}
        self.events: deque[tuple[int, int]] = deque()
        self.fail_next = 0
        self.fail_always = False
        self.polls = 0
        self.acknowledged_quit = False

    def getDeviceToAbsoluteTrackingPose(self, origin: int, predicted: float, count: int) -> Any:
        assert origin == 1  # TrackingUniverseStanding
        with self.lock:
            self.polls += 1
            if self.fail_always:
                raise OSError("fake sustained poll failure")
            if self.fail_next > 0:
                self.fail_next -= 1
                raise OSError("fake one-off poll failure")
            poses = (_Pose * count)()
            for index, device in self.devices.items():
                pose = poses[index]
                for row in range(3):
                    for col in range(4):
                        pose.mDeviceToAbsoluteTracking.m[row][col] = device.matrix[row, col]
                pose.bPoseIsValid = device.pose_valid
                pose.eTrackingResult = device.tracking_result
                pose.bDeviceIsConnected = device.connected
        return poses

    def getTrackedDeviceClass(self, index: int) -> int:
        with self.lock:
            device = self.devices.get(index)
        return 0 if device is None else device.device_class

    def getStringTrackedDeviceProperty(self, index: int, prop: int) -> str:
        with self.lock:
            device = self.devices[index]
        value = {_PROP_SERIAL: device.serial, _PROP_MODEL: device.model}[prop]
        if value is None:
            raise _FakeTrackedPropertyError("TrackedProp_UnknownProperty")
        return value

    def pollNextEvent(self, event: Any) -> bool:
        with self.lock:
            if not self.events:
                return False
            event.eventType, event.trackedDeviceIndex = self.events.popleft()
        return True

    def acknowledgeQuit_Exiting(self) -> None:
        self.acknowledged_quit = True

    def post(self, event_type: int, index: int = 0) -> None:
        with self.lock:
            self.events.append((event_type, index))


class _FakeOpenVR:
    """Stands in for the ``openvr`` module: the constants, init/shutdown, and the error hierarchy."""

    VRApplication_Other = 0
    TrackingUniverseStanding = 1
    k_unMaxTrackedDeviceCount = 16
    TrackedDeviceClass_HMD = _CLASS_HMD
    TrackedDeviceClass_Controller = _CLASS_CONTROLLER
    TrackedDeviceClass_GenericTracker = _CLASS_TRACKER
    TrackedDeviceClass_TrackingReference = _CLASS_BASE
    Prop_SerialNumber_String = _PROP_SERIAL
    Prop_ModelNumber_String = _PROP_MODEL
    VREvent_TrackedDeviceActivated = _ACTIVATED
    VREvent_TrackedDeviceDeactivated = _DEACTIVATED
    VREvent_Quit = _QUIT
    TrackingResult_Running_OK = _RUNNING_OK
    TrackingResult_Fallback_RotationOnly = _FALLBACK_ROTATION_ONLY
    VREvent_t = _Event

    class error_code:
        OpenVRError = _FakeOpenVRError
        TrackedPropertyError = _FakeTrackedPropertyError

    def __init__(self) -> None:
        self.system = _FakeVRSystem()
        self.init_calls = 0
        self.shutdown_calls = 0
        self.init_error: Exception | None = None

    def init(self, application_type: int) -> _FakeVRSystem:
        assert application_type == self.VRApplication_Other
        if self.init_error is not None:
            raise self.init_error
        self.init_calls += 1
        return self.system

    def shutdown(self) -> None:
        self.shutdown_calls += 1


_TRACKER_SERIAL = "LHR-TRACKER1"


@pytest.fixture(autouse=True)
def fake_openvr(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeOpenVR]:
    """Patch in the fake openvr (one tracker, one base station) and patch out SteamVR; close readers after."""
    fake = _FakeOpenVR()
    fake.system.devices[1] = _Device(_CLASS_BASE, "LHB-BASE1", matrix=np.eye(4)[:3])
    fake.system.devices[3] = _Device(_CLASS_TRACKER, _TRACKER_SERIAL, model="VIVE Tracker 3.0")
    monkeypatch.setattr(vive_tracker, "openvr", fake)
    monkeypatch.setattr(vive_tracker, "_ensure_steamvr_running", lambda: None)
    # Generous: a loaded CI runner can starve the poll thread; the staleness tests tighten it themselves.
    monkeypatch.setattr(ViveReader, "STALE_AFTER_S", 5.0)
    yield fake
    # Torn down before monkeypatch undoes the patches, so close() still reaches the fake openvr.
    if ViveReader._instance is not None:
        ViveReader._instance.close()
    ViveReader._instance = None


def _wait_until(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"condition not met within {timeout_s} s")


def _raises(reader: ViveReader, serial: str, exc_type: type[Exception]) -> bool:
    try:
        reader.get_pose(serial)
    except exc_type:
        return True
    return False


def test_pose_is_the_3x4_matrix_as_4x4(fake_openvr: _FakeOpenVR) -> None:
    before = time.time()
    reader = ViveReader()
    pose = reader.get_pose(_TRACKER_SERIAL)

    assert isinstance(pose, TrackerPose)
    assert pose.valid
    assert pose.pose.shape == (4, 4)
    assert pose.pose.dtype == np.float64
    np.testing.assert_allclose(pose.pose[:3, :4], _POSE_3X4, atol=1e-6)
    np.testing.assert_array_equal(pose.pose[3], [0.0, 0.0, 0.0, 1.0])
    # time.time() of the poll: comparable to other sensors' wall-clock stamps.
    assert before - 1.0 <= pose.timestamp <= time.time() + 1.0


def test_list_devices_reports_discovered_devices_by_index(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()

    assert reader.list_devices() == [
        TrackedDeviceInfo(1, "tracking_reference", "LHB-BASE1", "fake model", True),
        TrackedDeviceInfo(3, "tracker", _TRACKER_SERIAL, "VIVE Tracker 3.0", True),
    ]


@pytest.mark.parametrize(
    ("pose_valid", "tracking_result"),
    [(False, _RUNNING_OK), (True, _FALLBACK_ROTATION_ONLY)],
    ids=["pose_invalid", "rotation_only_fallback"],
)
def test_untrusted_pose_is_identity_and_invalid(
    fake_openvr: _FakeOpenVR, pose_valid: bool, tracking_result: int
) -> None:
    fake_openvr.system.devices[3].pose_valid = pose_valid
    fake_openvr.system.devices[3].tracking_result = tracking_result
    reader = ViveReader()

    pose = reader.get_pose(_TRACKER_SERIAL)

    assert not pose.valid
    np.testing.assert_array_equal(pose.pose, np.eye(4))


def test_deactivated_tracker_reports_invalid_not_keyerror(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()
    assert reader.get_pose(_TRACKER_SERIAL).valid

    fake_openvr.system.post(_DEACTIVATED, 3)
    _wait_until(lambda: not reader.get_pose(_TRACKER_SERIAL).valid)

    np.testing.assert_array_equal(reader.get_pose(_TRACKER_SERIAL).pose, np.eye(4))
    assert [d.connected for d in reader.list_devices() if d.serial == _TRACKER_SERIAL] == [False]

    fake_openvr.system.post(_ACTIVATED, 3)
    _wait_until(lambda: reader.get_pose(_TRACKER_SERIAL).valid)


def test_device_activated_after_start_is_picked_up(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()
    with fake_openvr.system.lock:
        fake_openvr.system.devices[5] = _Device(_CLASS_CONTROLLER, "LHR-CTRL1")
    fake_openvr.system.post(_ACTIVATED, 5)

    _wait_until(lambda: any(d.serial == "LHR-CTRL1" for d in reader.list_devices()))
    _wait_until(lambda: not _raises(reader, "LHR-CTRL1", KeyError))
    assert reader.get_pose("LHR-CTRL1").valid


def test_never_seen_serial_raises_keyerror_listing_known_serials(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()

    with pytest.raises(KeyError, match="LHR-NOPE") as excinfo:
        reader.get_pose("LHR-NOPE")
    assert _TRACKER_SERIAL in str(excinfo.value)
    assert "LHB-BASE1" in str(excinfo.value)


def test_tracker_without_a_serial_follows_the_first_tracker(
    fake_openvr: _FakeOpenVR, caplog: pytest.LogCaptureFixture
) -> None:
    fake_openvr.system.devices[7] = _Device(_CLASS_TRACKER, "LHR-TRACKER2")
    fake_openvr.system.devices[5] = _Device(_CLASS_CONTROLLER, "LHR-CTRL1")

    with caplog.at_level(logging.WARNING, logger=vive_tracker.__name__):
        tracker = ViveTracker()

    # The lowest-index tracker; a controller is never picked, and the choice among several is announced.
    assert tracker.serial == _TRACKER_SERIAL
    assert "LHR-TRACKER2" in caplog.text
    assert tracker.get_pose().valid


def test_tracker_without_a_serial_waits_for_one_to_connect(fake_openvr: _FakeOpenVR) -> None:
    del fake_openvr.system.devices[3]

    def connect() -> None:
        time.sleep(0.2)
        with fake_openvr.system.lock:
            fake_openvr.system.devices[3] = _Device(_CLASS_TRACKER, _TRACKER_SERIAL)
        fake_openvr.system.post(_ACTIVATED, 3)

    threading.Thread(target=connect, daemon=True).start()
    assert ViveTracker(detect_timeout_s=5.0).serial == _TRACKER_SERIAL


def test_tracker_without_a_serial_times_out_naming_what_steamvr_reports(fake_openvr: _FakeOpenVR) -> None:
    del fake_openvr.system.devices[3]

    with pytest.raises(TimeoutError, match="No Vive tracker") as excinfo:
        ViveTracker(detect_timeout_s=0.2)
    assert "LHB-BASE1" in str(excinfo.value)
    # A failed start closes the reader it opened.
    assert ViveReader._instance is None


def test_tracker_with_a_serial_does_not_wait_for_it(fake_openvr: _FakeOpenVR) -> None:
    tracker = ViveTracker("LHR-NOT-YET", detect_timeout_s=60.0)

    assert tracker.serial == "LHR-NOT-YET"
    with pytest.raises(KeyError, match="LHR-NOT-YET"):
        tracker.get_pose()


def test_missing_serial_property_falls_back_to_index_name(fake_openvr: _FakeOpenVR) -> None:
    fake_openvr.system.devices[3].serial = None
    fake_openvr.system.devices[3].model = None
    reader = ViveReader()

    tracker = next(d for d in reader.list_devices() if d.index == 3)
    assert tracker.serial == "device_index_3"
    assert tracker.model == ""
    assert reader.get_pose("device_index_3").valid


def test_get_pose_returns_a_copy(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()
    first = reader.get_pose(_TRACKER_SERIAL)
    first.pose[:] = 0.0

    np.testing.assert_allclose(reader.get_pose(_TRACKER_SERIAL).pose[:3, :4], _POSE_3X4, atol=1e-6)


def test_one_failed_poll_does_not_kill_the_thread(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()
    polls_before = fake_openvr.system.polls
    fake_openvr.system.fail_next = 1

    _wait_until(lambda: reader._error_count >= 1)
    _wait_until(lambda: fake_openvr.system.polls > polls_before + 5)

    assert reader._thread is not None and reader._thread.is_alive()
    assert reader.get_pose(_TRACKER_SERIAL).valid


def test_sustained_poll_failure_raises_timeout_with_last_error(
    fake_openvr: _FakeOpenVR, monkeypatch: pytest.MonkeyPatch
) -> None:
    reader = ViveReader()
    monkeypatch.setattr(ViveReader, "STALE_AFTER_S", 0.3)
    fake_openvr.system.fail_always = True

    _wait_until(lambda: _raises(reader, _TRACKER_SERIAL, TimeoutError))

    with pytest.raises(TimeoutError, match="fake sustained poll failure") as excinfo:
        reader.get_pose(_TRACKER_SERIAL)
    assert "poll error(s)" in str(excinfo.value)
    assert reader._thread is not None and reader._thread.is_alive()


def test_dead_poll_thread_raises_timeout(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()
    reader._stop.set()
    assert reader._thread is not None
    reader._thread.join(timeout=5.0)

    with pytest.raises(TimeoutError, match="dead"):
        reader.get_pose(_TRACKER_SERIAL)


def test_first_poll_timeout_fails_construction_and_releases_openvr(
    fake_openvr: _FakeOpenVR, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ViveReader, "FIRST_POLL_TIMEOUT_S", 0.3)
    fake_openvr.system.fail_always = True

    with pytest.raises(TimeoutError, match="fake sustained poll failure"):
        ViveReader()

    assert fake_openvr.init_calls == 1
    assert fake_openvr.shutdown_calls == 1
    assert ViveReader._instance is None


def test_openvr_init_error_is_a_runtime_error_pointing_at_the_setup_guide(fake_openvr: _FakeOpenVR) -> None:
    fake_openvr.init_error = _FakeInitError("InitError_Init_HmdNotFound")

    with pytest.raises(RuntimeError, match=r"docs/guides/steamvr-vive-tracker-setup\.md") as excinfo:
        ViveReader()

    assert isinstance(excinfo.value.__cause__, _FakeInitError)
    assert fake_openvr.shutdown_calls == 0
    assert ViveReader._instance is None


def test_singleton_reinitialises_after_close(fake_openvr: _FakeOpenVR) -> None:
    first = ViveReader()
    assert ViveReader() is first
    assert fake_openvr.init_calls == 1

    first.close()
    first.close()  # idempotent
    assert fake_openvr.shutdown_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        first.get_pose(_TRACKER_SERIAL)
    with pytest.raises(RuntimeError, match="closed"):
        first.list_devices()

    second = ViveReader()
    assert second is not first
    assert fake_openvr.init_calls == 2
    assert second.get_pose(_TRACKER_SERIAL).valid


def test_context_manager_closes(fake_openvr: _FakeOpenVR) -> None:
    with ViveReader() as reader:
        assert reader.get_pose(_TRACKER_SERIAL).valid

    assert fake_openvr.shutdown_calls == 1
    assert ViveReader._instance is None


def test_steamvr_quit_acknowledges_and_shuts_down(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()
    fake_openvr.system.post(_QUIT)

    _wait_until(lambda: _raises(reader, _TRACKER_SERIAL, RuntimeError))

    with pytest.raises(RuntimeError, match="SteamVR quit"):
        reader.get_pose(_TRACKER_SERIAL)
    assert fake_openvr.system.acknowledged_quit
    assert fake_openvr.shutdown_calls == 1
    assert not any(d.connected for d in reader.list_devices())
    assert reader._thread is not None
    reader._thread.join(timeout=5.0)
    assert not reader._thread.is_alive()

    reader.close()
    assert fake_openvr.shutdown_calls == 1  # the reader already shut OpenVR down; close() must not do it again


def test_missing_openvr_raises_a_clear_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vive_tracker, "openvr", None)

    with pytest.raises(ImportError, match="x86-64"):
        ViveReader()
    assert ViveReader._instance is None


def test_poll_thread_is_a_daemon(fake_openvr: _FakeOpenVR) -> None:
    reader = ViveReader()

    assert reader._thread is not None
    assert reader._thread.daemon
