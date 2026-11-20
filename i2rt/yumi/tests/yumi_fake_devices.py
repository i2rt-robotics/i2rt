"""Fake devices for tests that spawn REAL device processes (``device_process.DeviceStream``).

A spawned child imports its device class by reference, so fakes that run in a child must live in an importable
module -- not inside a test file, and not behind a ``monkeypatch`` (which never crosses a process boundary).
Behaviour is chosen through constructor keyword arguments, which ``DeviceStream`` pickles to the child. Import
this module as ``i2rt.yumi.tests.yumi_fake_devices`` so the child resolves the same module path.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from i2rt.camera.mf_camera import CameraFrame
from i2rt.yumi.gripper import GripperState
from i2rt.yumi.vive_tracker import TrackerPose

POSE = np.array(
    [[0.0, -1.0, 0.0, 0.1], [1.0, 0.0, 0.0, 1.2], [0.0, 0.0, 1.0, -0.3], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
)


def _inheritable_fds() -> list[int]:
    fds = []
    for name in os.listdir("/proc/self/fd"):
        fd = int(name)
        if fd <= 2:
            continue
        try:
            if os.get_inheritable(fd):
                fds.append(fd)
        except OSError:
            pass  # the descriptor listdir itself used, already closed
    return fds


@dataclass(frozen=True)
class Tick:
    """One CountingDevice sample: ``index`` counts periods since construction."""

    timestamp: float
    index: int
    pid: int


class CountingDevice:
    """Produces a new ``Tick`` every ``period_s`` (and repeats it in between, so deduplication is observable).

    Args:
        period_s: how often a new sample appears.
        stop_after: from this index on, ``read`` raises ``TimeoutError`` (the stream goes stale).
        crash_after: at this index the process exits hard (``os._exit``), as a native crash would.
        fail_init: raise ``ValueError(fail_init)`` from the constructor.
        init_sleep_s: block this long in the constructor (a slow or stuck device).
        closed_flag: a file written by ``close()``, so a test can see the device was closed in the child.
        close_sleep_s: block this long in ``close()`` (a device that is slow to release).
    """

    def __init__(
        self,
        period_s: float = 0.005,
        stop_after: int | None = None,
        crash_after: int | None = None,
        fail_init: str | None = None,
        init_sleep_s: float = 0.0,
        closed_flag: str | None = None,
        close_sleep_s: float = 0.0,
    ) -> None:
        if init_sleep_s:
            time.sleep(init_sleep_s)
        if fail_init is not None:
            raise ValueError(fail_init)
        self.period_s = period_s
        self.stop_after = stop_after
        self.crash_after = crash_after
        self.closed_flag = closed_flag
        self.close_sleep_s = close_sleep_s
        self.label = "counting"
        self.pid = os.getpid()
        # Descriptors a process this device launched would inherit (should be none beyond stdio).
        self.inheritable_fds = _inheritable_fds()
        self._start_mono = time.monotonic()
        self._start_wall = time.time()

    def read(self) -> Tick:
        index = int((time.monotonic() - self._start_mono) / self.period_s)
        if self.crash_after is not None and index >= self.crash_after:
            os._exit(3)
        if self.stop_after is not None and index >= self.stop_after:
            raise TimeoutError(f"stopped producing after sample {self.stop_after}")
        return Tick(timestamp=self._start_wall + index * self.period_s, index=index, pid=self.pid)

    def close(self) -> None:
        if self.close_sleep_s:
            time.sleep(self.close_sleep_s)
        if self.closed_flag is not None:
            Path(self.closed_flag).write_text(f"closed by {os.getpid()}")


class FakeGripper:
    """Stands in for ``UmiGripper`` (same constructor keywords)."""

    def __init__(self, channel: str, range_rad: float) -> None:
        self.channel = channel
        self.range_rad = range_rad

    def get_state(self) -> GripperState:
        return GripperState(
            timestamp=time.time(),
            position=0.25,
            raw_angle=0.75 * self.range_rad,
            raw_velocity=0.0,
            buttons=(False, False),
        )

    def close(self) -> None:
        pass


class FakeViveTracker:
    """Stands in for ``ViveTracker`` (same constructor keywords): ``LHR-0000TEST``, the only tracker, always valid at
    ``POSE``."""

    def __init__(self, serial: str | None = None, detect_timeout_s: float = 30.0) -> None:
        self.serial = serial if serial is not None else "LHR-0000TEST"

    def get_pose(self) -> TrackerPose:
        if self.serial != "LHR-0000TEST":
            raise KeyError(f"Vive device {self.serial!r} has not been seen; known serials: ['LHR-0000TEST']")
        return TrackerPose(pose=POSE.copy(), valid=True, timestamp=time.time())

    def close(self) -> None:
        pass


class FakeCamera:
    """Stands in for ``MFCamera`` (same constructor keywords): a new constant frame every 1/fps."""

    def __init__(self, serial: str, resolution: tuple[int, int], fps: int) -> None:
        self.serial = serial
        self.device_path = "/dev/video-fake"
        self.resolution = tuple(resolution)
        self.fps = fps
        self._start_mono = time.monotonic()
        self._start_wall = time.time()
        width, height = self.resolution
        self._image = np.full((height, width, 3), 7, dtype=np.uint8)

    def read(self) -> CameraFrame:
        index = int((time.monotonic() - self._start_mono) * self.fps)
        return CameraFrame(image=self._image, timestamp=self._start_wall + index / self.fps)

    def close(self) -> None:
        pass
