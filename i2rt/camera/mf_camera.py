"""Driver for the i2rt MF150 / MF118 USB fisheye cameras (UVC over V4L2), opened by USB serial.

Why this is more than ``cv2.VideoCapture("/dev/videoN")``:

* **The device is found by serial, not by node number.** ``/dev/videoN`` numbering follows enumeration order and
  moves between boots and replugs, while the USB short serial (``ID_SERIAL_SHORT``) is flashed per unit. A UVC
  camera also exposes a second, metadata-only node under the same serial, which opens fine and then never
  delivers a frame -- so the lookup accepts only nodes that udev reports as capture-capable.
* **The backend is pinned to V4L2.** ``CAP_ANY`` can pick a backend that accepts every ``set()`` and ignores it.
  Even on V4L2, ``set()`` returns True when the driver clamps to a different mode, so the negotiated width and
  height are read back and a mismatch refuses to open, so the frames are always the size that was asked for.
* **A background thread keeps only the latest frame.** ``read()`` never blocks on the device; it returns the
  newest frame or raises ``TimeoutError`` if that frame is older than ``max_frame_age_s``. The timestamp is taken
  when ``grab()`` returns, before the JPEG decode, so it marks when the frame arrived rather than when it
  finished decoding.

Linux only (pyudev + V4L2); a non-Linux host raises when a camera is opened, not at import.

Bring-up check::

    python i2rt/camera/mf_camera.py --serial MF150_2026061109 [--resolution 800 600] [--fps 30] [--show]
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from types import TracebackType

import cv2
import pyudev

from i2rt.camera.camera import Camera, CameraFrame

logger = logging.getLogger(__name__)

# The mode yumi records at.
MF_DEFAULT_RESOLUTION = (800, 600)

# OpenCV's V4L2 backend waits this many seconds in select() for a frame before grab() gives up (default 10).
# Bounding it bounds how long close() can wait for the capture thread to leave grab().
_SELECT_TIMEOUT_ENV = "OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT"
_SELECT_TIMEOUT_S = "1"
# Longer than one select() timeout plus a decode, so a healthy thread always exits within it.
_JOIN_TIMEOUT_S = 2.0

# The value the MF cameras have always been run at. V4L2 has no 0.25 value, so what it latches is device-specific;
# the driver logs the read-back, and ``v4l2-ctl -d /dev/videoN --get-ctrl auto_exposure`` shows it on the bench.
_AUTO_EXPOSURE = 0.25
# Two buffers keep the driver from queueing stale frames behind the one the thread is about to grab.
_BUFFER_SIZE = 2

_ERROR_BACKOFF_S = 0.01
_ERROR_LOG_INTERVAL_S = 1.0

_UDEVADM_HINT = "udevadm info --query=property --name=/dev/videoX | grep ID_SERIAL_SHORT"
_VIDEO_NODE_INDEX = re.compile(r"(\d+)$")


def _video_node_index(node: str) -> tuple[int, str]:
    match = _VIDEO_NODE_INDEX.search(node)
    return (int(match.group(1)) if match else sys.maxsize, node)


def _video_nodes() -> list[tuple[str, str | None, bool]]:
    """``(node, ID_SERIAL_SHORT, capture-capable)`` for every video4linux node, in ``/dev/videoN`` order.

    Raises:
        RuntimeError: on a non-Linux host.
    """
    if sys.platform != "linux":
        raise RuntimeError(
            f"MF cameras are supported on Linux only (pyudev + V4L2); got sys.platform={sys.platform!r}"
        )
    nodes = [
        (
            device.device_node,
            device.properties.get("ID_SERIAL_SHORT"),
            ":capture:" in device.properties.get("ID_V4L_CAPABILITIES", ""),
        )
        for device in pyudev.Context().list_devices(subsystem="video4linux")
        if device.device_node is not None
    ]
    return sorted(nodes, key=lambda node: _video_node_index(node[0]))


def find_video_device_by_serial(serial: str) -> str:
    """Return the capture-capable ``/dev/videoN`` whose USB short serial is exactly ``serial``, or raise.

    The match is byte-exact against ``ID_SERIAL_SHORT`` (no case folding): serials are flashed by hand, and a
    fuzzy match could open the wrong unit. Nodes without ``:capture:`` in ``ID_V4L_CAPABILITIES`` -- the UVC
    metadata node every camera also exposes -- are skipped.

    Raises:
        RuntimeError: on a non-Linux host, or when no capture-capable node carries ``serial``.
    """
    nodes = _video_nodes()
    capture_nodes = [node for node, node_serial, capture in nodes if node_serial == serial and capture]
    if not capture_nodes:
        non_capture_nodes = [node for node, node_serial, capture in nodes if node_serial == serial and not capture]
        other_serials = sorted({s for _, s, capture in nodes if capture and s and s != serial})
        found = (
            f"only non-capture node(s) {non_capture_nodes} carry it"
            if non_capture_nodes
            else "no video4linux device carries it"
        )
        raise RuntimeError(
            f"MF camera with serial {serial!r} not found: {found}. Capture-capable serials present: "
            f"{other_serials}. The match is exact and case-sensitive; check a node with: {_UDEVADM_HINT}"
        )

    if len(capture_nodes) > 1:
        logger.warning(
            "More than one capture node carries serial %r (%s); using %s. Two units flashed with the same serial "
            "cannot be told apart.",
            serial,
            capture_nodes,
            capture_nodes[0],
        )
    return capture_nodes[0]


def find_mf_camera_serials() -> list[str]:
    """Return the USB short serials of every plugged-in capture-capable camera, in ``/dev/videoN`` order.

    Every serial is accepted, whatever it is named, so a laptop webcam is listed too; each camera is listed once,
    whatever number of nodes it exposes.

    Raises:
        RuntimeError: on a non-Linux host.
    """
    serials: list[str] = []
    for _, serial, capture in _video_nodes():
        if capture and serial and serial not in serials:
            serials.append(serial)
    return serials


class MFCamera(Camera):
    """An i2rt MF camera opened by USB serial, with a background thread holding the latest RGB frame.

    Construction finds the capture node, opens it on V4L2, configures and verifies the capture mode, starts the
    capture thread and waits for the first frame; any failure after the device is opened releases it before
    re-raising.

    Attributes:
        serial: the USB short serial the camera was opened by.
        device_path: the ``/dev/videoN`` capture node it resolved to.
        resolution: the NEGOTIATED ``(width, height)`` (always equal to the request; a mismatch refuses to open).
        fps: the NEGOTIATED frame rate, or the requested one when the driver does not report it.
    """

    def __init__(
        self,
        serial: str,
        resolution: tuple[int, int] = MF_DEFAULT_RESOLUTION,
        fps: int = 30,
        max_frame_age_s: float = 0.2,
        first_frame_timeout_s: float = 5.0,
    ) -> None:
        """Open the camera and block until its first frame arrives.

        Args:
            serial: USB short serial (``ID_SERIAL_SHORT``), matched exactly, e.g. ``MF150_2026061109``.
            resolution: requested ``(width, height)``; must be a mode the camera supports.
            fps: requested frame rate.
            max_frame_age_s: ``read()`` raises ``TimeoutError`` once the latest frame is older than this.
            first_frame_timeout_s: how long construction waits for the first frame.

        Raises:
            RuntimeError: the camera is not found, cannot be opened, or refuses the requested mode.
            TimeoutError: no frame arrived within ``first_frame_timeout_s``.
        """
        requested = (int(resolution[0]), int(resolution[1]))
        self.serial = serial
        self.resolution = requested
        self.fps = int(fps)
        self.max_frame_age_s = max_frame_age_s

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._first_frame = threading.Event()
        self._frame: CameraFrame | None = None
        self._frame_monotonic = 0.0
        self._error_count = 0
        self._last_error: str | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

        self.device_path = find_video_device_by_serial(serial)
        logger.info("MF camera %s found at %s", serial, self.device_path)

        # Must precede the first VideoCapture: the V4L2 backend reads it when a device is opened.
        os.environ.setdefault(_SELECT_TIMEOUT_ENV, _SELECT_TIMEOUT_S)
        self._cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
        try:
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"MF camera {serial}: failed to open {self.device_path} with the V4L2 backend. Check that the "
                    f"user is in the 'video' group and that no other process holds the device."
                )
            self._configure(self._cap, requested, self.fps)
            self.resolution, self.fps = self._negotiated_mode(self._cap, requested, self.fps)

            self._thread = threading.Thread(target=self._capture_loop, name=f"MFCamera-{serial}", daemon=True)
            self._thread.start()
            if not self._first_frame.wait(first_frame_timeout_s):
                with self._lock:
                    errors, last_error = self._error_count, self._last_error
                raise TimeoutError(
                    f"MF camera {serial} ({self.device_path}): no frame within {first_frame_timeout_s:.1f}s "
                    f"({errors} capture errors, last: {last_error})"
                )
        except BaseException:
            self.close()
            raise

    def _set(self, cap: cv2.VideoCapture, prop: int, value: float, name: str) -> None:
        if not cap.set(prop, value):
            logger.warning("MF camera %s: the driver rejected %s=%s", self.serial, name, value)

    def _configure(self, cap: cv2.VideoCapture, resolution: tuple[int, int], fps: int) -> None:
        """Apply the capture settings, format first, since the pixel format decides which modes exist."""
        self._set(cap, cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"), "FOURCC")
        self._set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE, "AUTO_EXPOSURE")
        self._set(cap, cv2.CAP_PROP_FRAME_WIDTH, resolution[0], "FRAME_WIDTH")
        self._set(cap, cv2.CAP_PROP_FRAME_HEIGHT, resolution[1], "FRAME_HEIGHT")
        self._set(cap, cv2.CAP_PROP_FPS, fps, "FPS")
        self._set(cap, cv2.CAP_PROP_BUFFERSIZE, _BUFFER_SIZE, "BUFFERSIZE")
        logger.info(
            "MF camera %s: AUTO_EXPOSURE reads back %s after requesting %s",
            self.serial,
            cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            _AUTO_EXPOSURE,
        )

    def _negotiated_mode(
        self, cap: cv2.VideoCapture, requested: tuple[int, int], fps: int
    ) -> tuple[tuple[int, int], int]:
        """Read back the mode the driver actually latched; refuse a size mismatch, tolerate FPS jitter."""
        negotiated = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "MF camera %s (%s) negotiated %dx%d @ %.1f fps",
            self.serial,
            self.device_path,
            negotiated[0],
            negotiated[1],
            raw_fps,
        )
        if negotiated != requested:
            raise RuntimeError(
                f"MF camera {self.serial} ({self.device_path}): the driver refused {requested[0]}x{requested[1]} and "
                f"negotiated {negotiated[0]}x{negotiated[1]}. List the supported modes with: "
                f"v4l2-ctl --device={self.device_path} --list-formats-ext"
            )
        # Drivers report jitter (29.97 for 30) and some report 0, so round and fall back rather than refuse.
        return negotiated, round(raw_fps) if raw_fps > 0 else fps

    def _capture_loop(self) -> None:
        cap = self._cap
        assert cap is not None
        last_log = -float("inf")
        while not self._stop.is_set():
            try:
                if not cap.grab():
                    raise RuntimeError("grab() returned False")
                timestamp = time.time()
                stamp_monotonic = time.monotonic()
                ok, bgr = cap.retrieve()
                if not ok or bgr is None:
                    raise RuntimeError("retrieve() returned no frame")
                frame = CameraFrame(image=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), timestamp=timestamp)
                with self._lock:
                    self._frame = frame
                    self._frame_monotonic = stamp_monotonic
                self._first_frame.set()
            except Exception as exc:
                with self._lock:
                    self._error_count += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    errors = self._error_count
                now = time.monotonic()
                if now - last_log >= _ERROR_LOG_INTERVAL_S:
                    last_log = now
                    logger.warning("MF camera %s: capture error #%d: %s", self.serial, errors, exc)
                self._stop.wait(_ERROR_BACKOFF_S)

    def read(self) -> CameraFrame:
        """Return a copy of the latest frame.

        Raises:
            RuntimeError: the camera is closed.
            TimeoutError: the latest frame is older than ``max_frame_age_s`` or the capture thread has died.
        """
        with self._lock:
            closed = self._closed
            frame = self._frame
            stamp = self._frame_monotonic
            errors, last_error = self._error_count, self._last_error
        if closed:
            raise RuntimeError(f"MF camera {self.serial} is closed")
        alive = self._thread is not None and self._thread.is_alive()
        age = time.monotonic() - stamp if frame is not None else float("inf")
        if frame is None or not alive or age > self.max_frame_age_s:
            raise TimeoutError(
                f"MF camera {self.serial} ({self.device_path}): latest frame is {age:.3f}s old "
                f"(max {self.max_frame_age_s:.3f}s), capture thread alive={alive}, "
                f"{errors} capture errors, last: {last_error}"
            )
        return CameraFrame(image=frame.image.copy(), timestamp=frame.timestamp)

    def close(self) -> None:
        """Stop the capture thread and release the device; idempotent.

        The device is released only once the thread has left ``grab()``: releasing a capture that another thread
        is inside is undefined behaviour in OpenCV. If the thread does not exit in time it is left to finish (it is
        a daemon) and the device is not released.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=_JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(
                    "MF camera %s: capture thread did not exit within %.1fs; not releasing %s",
                    self.serial,
                    _JOIN_TIMEOUT_S,
                    self.device_path,
                )
                return
        if self._cap is not None:
            self._cap.release()

    def __enter__(self) -> MFCamera:
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
    """Bring-up check for an i2rt MF camera: open it by USB serial, print the measured FPS, optionally show it."""

    serial: str
    """USB short serial (ID_SERIAL_SHORT), e.g. MF150_2026061109. Find it with: udevadm info --query=property
    --name=/dev/videoX | grep ID_SERIAL_SHORT"""
    resolution: tuple[int, int] = MF_DEFAULT_RESOLUTION
    """Capture mode as width height."""
    fps: int = 30
    """Requested frame rate."""
    show: bool = False
    """Show the live stream in an OpenCV window (q or ESC quits)."""


def main(args: _CliArgs) -> None:
    """Open the camera and stream it."""
    window = f"MF camera {args.serial}"
    with MFCamera(args.serial, resolution=args.resolution, fps=args.fps) as camera:
        width, height = camera.resolution
        print(f"Opened {camera.serial} at {camera.device_path}: {width}x{height} @ {camera.fps} fps")
        last_timestamp: float | None = None
        frames = 0
        window_start = time.monotonic()
        try:
            while True:
                frame = camera.read()
                if frame.timestamp != last_timestamp:
                    last_timestamp = frame.timestamp
                    frames += 1
                    if args.show:
                        cv2.imshow(window, cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR))
                if args.show:
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
                else:
                    time.sleep(0.001)
                now = time.monotonic()
                if now - window_start >= 1.0:
                    print(f"Camera frequency: {frames / (now - window_start):.2f} FPS")
                    frames = 0
                    window_start = now
        except KeyboardInterrupt:
            pass
        finally:
            if args.show:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    import tyro

    from i2rt.utils.utils import override_log_level

    override_log_level(logging.INFO)
    main(tyro.cli(_CliArgs))
