"""Tests for the MF camera driver, against a fake udev and a fake ``cv2.VideoCapture``.

The driver's job is to refuse the ways a USB camera silently produces wrong data: opening the metadata node
that never delivers a frame, a mode the driver clamped without saying so, a frame that stopped updating. Each of those passes unnoticed on the bench until a dataset is already recorded,
so each is pinned here without hardware. The device is always released on a failed construction, because a
leaked V4L2 file descriptor makes the retry fail with EBUSY.
"""

import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pytest
import pyudev
import tyro

from i2rt.camera import mf_camera
from i2rt.camera.camera import Camera
from i2rt.camera.mf_camera import CameraFrame, MFCamera, find_mf_camera_serials, find_video_device_by_serial

SERIAL = "MF150_2026061109"
UNLABELLED_SERIAL = "2026061109"
CAPTURE_NODE = "/dev/video4"
METADATA_NODE = "/dev/video5"
# Generous: shared CI runners can starve the capture thread for well over the 0.2 s production default.
FRESH_S = 5.0


# ---------------------------------------------------------------------------------------------------------------
# Fake udev
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class _FakeDevice:
    device_node: str | None
    properties: dict[str, str]


def _v4l_device(node: str, serial: str, capture: bool) -> _FakeDevice:
    return _FakeDevice(node, {"ID_SERIAL_SHORT": serial, "ID_V4L_CAPABILITIES": ":capture:" if capture else ":"})


def _install_udev(monkeypatch: pytest.MonkeyPatch, devices: list[_FakeDevice]) -> list[dict[str, Any]]:
    """Replace ``pyudev.Context`` with one listing ``devices``; returns the recorded ``list_devices`` kwargs."""
    calls: list[dict[str, Any]] = []

    class _FakeContext:
        def list_devices(self, **kwargs: Any) -> list[_FakeDevice]:
            calls.append(kwargs)
            return list(devices) if kwargs.get("subsystem") == "video4linux" else []

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(pyudev, "Context", _FakeContext)
    return calls


# ---------------------------------------------------------------------------------------------------------------
# Fake VideoCapture
# ---------------------------------------------------------------------------------------------------------------


@dataclass
class _Behaviour:
    """How the fake capture behaves; tests mutate it, including while the capture thread runs."""

    opened: bool = True
    negotiated: tuple[int, int] | None = None  # None: whatever was requested
    reported_fps: float | None = None  # None: whatever was requested
    rejected_props: set[int] = field(default_factory=set)
    grab_ok: bool = True
    frame_period_s: float = 0.005
    retrieve_delay_s: float = 0.0
    # Consumed one per retrieve() call: "ok", "fail" (returns (False, None)) or "raise".
    retrieve_script: list[str] = field(default_factory=list)
    # When set, grab() blocks until it is set (or 10 s pass).
    grab_gate: threading.Event | None = None


class _FakeCapture:
    def __init__(self, path: str, api: int, behaviour: _Behaviour) -> None:
        self.path = path
        self.api = api
        self.behaviour = behaviour
        self.sets: list[tuple[int, float]] = []
        self.props: dict[int, float] = {}
        self.released = False
        self.released_while_grabbing = False
        self.in_grab = False
        self.frame_index = -1
        self.grab_end_wall: dict[int, float] = {}
        self.retrieve_end_wall: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.behaviour.opened

    def set(self, prop: int, value: float) -> bool:
        self.sets.append((prop, value))
        self.props[prop] = value
        return prop not in self.behaviour.rejected_props

    def get(self, prop: int) -> float:
        b = self.behaviour
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(b.negotiated[0] if b.negotiated else self.props.get(prop, 0.0))
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(b.negotiated[1] if b.negotiated else self.props.get(prop, 0.0))
        if prop == cv2.CAP_PROP_FPS:
            return float(b.reported_fps if b.reported_fps is not None else self.props.get(prop, 0.0))
        return float(self.props.get(prop, 0.0))

    def grab(self) -> bool:
        self.in_grab = True
        try:
            if self.behaviour.grab_gate is not None:
                self.behaviour.grab_gate.wait(10.0)
            time.sleep(self.behaviour.frame_period_s)
            if not self.behaviour.grab_ok:
                return False
            self.frame_index += 1
            self.grab_end_wall[self.frame_index] = time.time()
            return True
        finally:
            self.in_grab = False

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        step = self.behaviour.retrieve_script.pop(0) if self.behaviour.retrieve_script else "ok"
        if step == "raise":
            raise cv2.error("fake decode failure")
        if step == "fail":
            return False, None
        time.sleep(self.behaviour.retrieve_delay_s)
        width, height = int(self.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bgr = np.empty((height, width, 3), dtype=np.uint8)
        bgr[..., 0] = 10  # B
        bgr[..., 1] = 20  # G
        bgr[..., 2] = self.frame_index % 256  # R carries the frame index
        self.retrieve_end_wall[self.frame_index] = time.time()
        return True, bgr

    def release(self) -> None:
        self.released_while_grabbing = self.in_grab
        self.released = True


@dataclass
class _Rig:
    behaviour: _Behaviour
    captures: list[_FakeCapture]
    threads_before: set[threading.Thread]

    @property
    def cap(self) -> _FakeCapture:
        assert len(self.captures) == 1
        return self.captures[0]


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Rig]:
    """A fake camera at CAPTURE_NODE (plus its metadata node) that opens and streams."""
    _install_udev(
        monkeypatch,
        [_v4l_device(METADATA_NODE, SERIAL, capture=False), _v4l_device(CAPTURE_NODE, SERIAL, capture=True)],
    )
    behaviour = _Behaviour()
    captures: list[_FakeCapture] = []

    def _factory(path: str, api: int) -> _FakeCapture:
        captures.append(_FakeCapture(path, api, behaviour))
        return captures[-1]

    monkeypatch.setattr(cv2, "VideoCapture", _factory)
    monkeypatch.delenv("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", raising=False)
    yield _Rig(behaviour, captures, set(threading.enumerate()))
    behaviour.grab_ok = True
    if behaviour.grab_gate is not None:
        behaviour.grab_gate.set()


@pytest.fixture
def open_camera(rig: _Rig) -> Iterator[Callable[..., MFCamera]]:
    """Construct cameras against the rig and always close them."""
    cameras: list[MFCamera] = []

    def _open(**kwargs: Any) -> MFCamera:
        kwargs.setdefault("max_frame_age_s", FRESH_S)
        cameras.append(MFCamera(kwargs.pop("serial", SERIAL), **kwargs))
        return cameras[-1]

    yield _open
    for camera in cameras:
        camera.close()


def _assert_released_without_thread(rig: _Rig) -> None:
    cap = rig.cap
    assert cap.released
    assert not cap.released_while_grabbing
    started = set(threading.enumerate()) - rig.threads_before
    assert not [t for t in started if t.name.startswith("MFCamera-") and t.is_alive()]


# ---------------------------------------------------------------------------------------------------------------
# Device lookup
# ---------------------------------------------------------------------------------------------------------------


def test_lookup_skips_the_metadata_node_that_shares_the_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_udev(
        monkeypatch,
        [
            _v4l_device("/dev/video0", "OTHER_CAMERA", capture=True),
            _v4l_device(METADATA_NODE, SERIAL, capture=False),
            _v4l_device(CAPTURE_NODE, SERIAL, capture=True),
            _FakeDevice(None, {"ID_SERIAL_SHORT": SERIAL, "ID_V4L_CAPABILITIES": ":capture:"}),
        ],
    )
    assert find_video_device_by_serial(SERIAL) == CAPTURE_NODE
    assert calls == [{"subsystem": "video4linux"}]


def test_lookup_is_byte_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_udev(monkeypatch, [_v4l_device(CAPTURE_NODE, SERIAL, capture=True)])
    with pytest.raises(RuntimeError, match="ID_SERIAL_SHORT") as info:
        find_video_device_by_serial(SERIAL.lower())
    # The serials that ARE present are named, so a typo is obvious.
    assert SERIAL in str(info.value)


def test_lookup_refuses_a_serial_carried_only_by_a_metadata_node(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_udev(monkeypatch, [_v4l_device(METADATA_NODE, SERIAL, capture=False)])
    with pytest.raises(RuntimeError, match="non-capture") as info:
        find_video_device_by_serial(SERIAL)
    assert "udevadm info" in str(info.value)


def test_lookup_prefers_the_lowest_node_when_two_capture_nodes_match(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _install_udev(
        monkeypatch,
        [_v4l_device("/dev/video10", SERIAL, capture=True), _v4l_device("/dev/video2", SERIAL, capture=True)],
    )
    with caplog.at_level(logging.WARNING, logger=mf_camera.__name__):
        assert find_video_device_by_serial(SERIAL) == "/dev/video2"
    assert "More than one capture node" in caplog.text


def test_camera_serials_list_every_capture_serial_once_in_node_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_udev(
        monkeypatch,
        [
            _v4l_device("/dev/video10", "MF118_2026061101", capture=True),
            _v4l_device("/dev/video0", UNLABELLED_SERIAL, capture=True),  # no lens token: listed all the same
            _v4l_device(METADATA_NODE, SERIAL, capture=False),
            _v4l_device(CAPTURE_NODE, SERIAL, capture=True),
            _v4l_device("/dev/video6", SERIAL, capture=True),  # a second capture node of the same camera
            _v4l_device("/dev/video7", "METADATA_ONLY", capture=False),
            _FakeDevice("/dev/video9", {"ID_V4L_CAPABILITIES": ":capture:"}),  # no serial at all
        ],
    )
    assert find_mf_camera_serials() == [UNLABELLED_SERIAL, SERIAL, "MF118_2026061101"]


def test_camera_serials_are_empty_without_a_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_udev(monkeypatch, [_v4l_device(METADATA_NODE, SERIAL, capture=False)])
    assert find_mf_camera_serials() == []


def test_lookup_refuses_non_linux_hosts_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux only"):
        find_video_device_by_serial(SERIAL)


# ---------------------------------------------------------------------------------------------------------------
# Open, configure, negotiate
# ---------------------------------------------------------------------------------------------------------------


def test_opens_the_capture_node_on_v4l2_with_settings_in_reference_order(
    rig: _Rig, open_camera: Callable[..., MFCamera]
) -> None:
    camera = open_camera()
    cap = rig.cap
    assert (cap.path, cap.api) == (CAPTURE_NODE, cv2.CAP_V4L2)
    assert cap.sets == [
        (cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG")),
        (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
        (cv2.CAP_PROP_FRAME_WIDTH, 800),
        (cv2.CAP_PROP_FRAME_HEIGHT, 600),
        (cv2.CAP_PROP_FPS, 30),
        (cv2.CAP_PROP_BUFFERSIZE, 2),
    ]
    assert camera.serial == SERIAL
    assert camera.device_path == CAPTURE_NODE
    assert camera.resolution == (800, 600)
    assert camera.fps == 30


def test_select_timeout_is_defaulted_but_never_overridden(
    rig: _Rig, open_camera: Callable[..., MFCamera], monkeypatch: pytest.MonkeyPatch
) -> None:
    open_camera()
    assert os.environ["OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT"] == "1"
    monkeypatch.setenv("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "3")
    open_camera()
    assert os.environ["OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT"] == "3"


def test_a_rejected_setting_is_warned_about_not_fatal(
    rig: _Rig, open_camera: Callable[..., MFCamera], caplog: pytest.LogCaptureFixture
) -> None:
    rig.behaviour.rejected_props = {cv2.CAP_PROP_AUTO_EXPOSURE}
    with caplog.at_level(logging.WARNING, logger=mf_camera.__name__):
        open_camera()
    assert "rejected AUTO_EXPOSURE=0.25" in caplog.text


def test_a_clamped_mode_is_refused_and_the_device_released(rig: _Rig) -> None:
    # set() reported success, but the driver latched another mode -- only the read-back tells.
    rig.behaviour.negotiated = (640, 480)
    with pytest.raises(RuntimeError, match=r"refused 800x600 and negotiated 640x480"):
        MFCamera(SERIAL)
    _assert_released_without_thread(rig)


def test_a_device_that_does_not_open_is_released(rig: _Rig) -> None:
    rig.behaviour.opened = False
    with pytest.raises(RuntimeError, match="failed to open"):
        MFCamera(SERIAL)
    _assert_released_without_thread(rig)


@pytest.mark.parametrize(("reported", "expected"), [(29.97, 30), (60.0, 60), (0.0, 30)])
def test_negotiated_fps_is_rounded_and_falls_back_to_the_request(
    rig: _Rig, open_camera: Callable[..., MFCamera], reported: float, expected: int
) -> None:
    rig.behaviour.reported_fps = reported
    assert open_camera(fps=30).fps == expected


# ---------------------------------------------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------------------------------------------


def test_frames_are_rgb_uint8_copies(rig: _Rig, open_camera: Callable[..., MFCamera]) -> None:
    camera = open_camera()
    assert isinstance(camera, Camera)
    frame = camera.read()
    assert isinstance(frame, CameraFrame)
    assert frame.image.shape == (600, 800, 3)
    assert frame.image.dtype == np.uint8
    # The fake delivers BGR (10, 20, index); RGB puts the index first and B last.
    assert frame.image[0, 0, 1] == 20
    assert frame.image[0, 0, 2] == 10
    frame.image[...] = 0
    assert camera.read().image[0, 0, 2] == 10


def test_timestamp_is_taken_at_grab_before_decode(rig: _Rig, open_camera: Callable[..., MFCamera]) -> None:
    rig.behaviour.retrieve_delay_s = 0.05
    camera = open_camera()
    frame = camera.read()
    index = int(frame.image[0, 0, 0])
    cap = rig.cap
    assert frame.timestamp >= cap.grab_end_wall[index]
    # retrieve() slept 50 ms after the stamp, so a stamp taken after decode could not be this early.
    assert cap.retrieve_end_wall[index] - frame.timestamp >= 0.045


def test_a_failed_or_raising_retrieve_does_not_kill_the_thread(
    rig: _Rig, open_camera: Callable[..., MFCamera]
) -> None:
    rig.behaviour.retrieve_script = ["fail", "raise"]
    camera = open_camera()
    first = camera.read()
    deadline = time.monotonic() + FRESH_S
    while camera.read().timestamp == first.timestamp and time.monotonic() < deadline:
        time.sleep(0.01)
    assert camera.read().timestamp > first.timestamp
    assert camera._thread is not None and camera._thread.is_alive()
    assert camera._error_count == 2
    assert camera._last_error is not None and "fake decode failure" in camera._last_error


def test_a_stream_that_stops_updating_raises_timeout_with_the_last_error(
    rig: _Rig, open_camera: Callable[..., MFCamera]
) -> None:
    camera = open_camera(max_frame_age_s=0.3)
    rig.behaviour.grab_ok = False
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            camera.read()
        except TimeoutError as exc:
            message = str(exc)
            break
        time.sleep(0.05)
    else:
        pytest.fail("read() never reported the stale stream")
    assert "grab() returned False" in message
    assert "capture errors" in message


def test_no_first_frame_times_out_and_releases(rig: _Rig) -> None:
    rig.behaviour.grab_ok = False
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="no frame within"):
        MFCamera(SERIAL, first_frame_timeout_s=0.3)
    assert time.monotonic() - started < 5.0
    _assert_released_without_thread(rig)


def test_capture_thread_is_a_daemon(rig: _Rig, open_camera: Callable[..., MFCamera]) -> None:
    camera = open_camera()
    assert camera._thread is not None and camera._thread.daemon


# ---------------------------------------------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------------------------------------------


def test_close_is_idempotent_and_releases_after_the_thread_exits(rig: _Rig) -> None:
    with MFCamera(SERIAL, max_frame_age_s=FRESH_S) as camera:
        camera.read()
    _assert_released_without_thread(rig)
    camera.close()
    camera.close()
    with pytest.raises(RuntimeError, match="closed"):
        camera.read()


def test_close_is_bounded_and_does_not_release_under_a_stuck_thread(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    camera = MFCamera(SERIAL, max_frame_age_s=FRESH_S)
    gate = threading.Event()
    rig.behaviour.grab_gate = gate
    time.sleep(0.1)  # let the thread enter the blocking grab()
    monkeypatch.setattr(mf_camera, "_JOIN_TIMEOUT_S", 0.2)
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger=mf_camera.__name__):
        camera.close()
    assert time.monotonic() - started < 2.0
    assert not rig.cap.released
    assert "did not exit" in caplog.text
    gate.set()
    assert camera._thread is not None
    camera._thread.join(5.0)
    assert not camera._thread.is_alive()


# ---------------------------------------------------------------------------------------------------------------
# Any serial, any mode
# ---------------------------------------------------------------------------------------------------------------


def test_a_serial_without_a_lens_token_opens(
    monkeypatch: pytest.MonkeyPatch, rig: _Rig, open_camera: Callable[..., MFCamera]
) -> None:
    _install_udev(monkeypatch, [_v4l_device(CAPTURE_NODE, UNLABELLED_SERIAL, capture=True)])
    camera = open_camera(serial=UNLABELLED_SERIAL)
    assert camera.serial == UNLABELLED_SERIAL
    camera.read()


def test_any_mode_the_driver_accepts_opens(open_camera: Callable[..., MFCamera]) -> None:
    camera = open_camera(resolution=(1920, 1080))
    assert camera.read().image.shape == (1080, 1920, 3)


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------


def test_cli_requires_serial_and_defaults_to_800x600() -> None:
    args = tyro.cli(mf_camera._CliArgs, args=["--serial", SERIAL])
    assert (args.serial, args.resolution, args.fps, args.show) == (SERIAL, (800, 600), 30, False)
    with pytest.raises(SystemExit):
        tyro.cli(mf_camera._CliArgs, args=[])
