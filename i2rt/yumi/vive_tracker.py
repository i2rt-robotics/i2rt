"""Vive tracker poses through OpenVR/SteamVR, read by one process-wide background thread.

The yumi handle carries a Vive tracker. SteamVR owns the lighthouse tracking; this module is an OpenVR
client that polls every tracked device's pose and serves the latest one per **serial**, because the serial
is the only identity that survives a reconnect (the tracked-device index is a slot, and ordinal names like
"tracker_1" silently re-point when a device drops out and another comes back first).

Why it is shaped the way it is:

* **One reader per process.** ``openvr.init`` / ``openvr.shutdown`` are process-global, so ``ViveReader()``
  returns the live shared instance instead of a second connection. Closing it closes it for every holder;
  the next ``ViveReader()`` after a close initialises a fresh one.
* **Validity is strict.** A pose counts as valid only when OpenVR marks it valid *and* its tracking result
  is ``TrackingResult_Running_OK``. The rotation-only fallback (the IMU coasting without optical lock) keeps
  ``bPoseIsValid`` set while its translation drifts, which is exactly the pose a recording must not trust.
  An invalid pose is reported as the identity with ``valid=False``, never as an exception: losing optical
  lock is an ordinary event for a handheld device.
* **A dead stream is loud.** The poll thread survives any one failing iteration, but ``get_pose`` raises
  ``TimeoutError`` once the stream is older than ``STALE_AFTER_S``, so a stuck reader cannot keep serving
  its last pose as if it were current.
* **SteamVR quitting is survivable.** On ``VREvent_Quit`` the reader acknowledges the quit, marks every
  pose invalid and shuts OpenVR down itself; a client that stays connected is killed by SteamVR, and it
  would take the whole process (camera, gripper, recording) with it.

Frames: poses are OpenVR's standing universe, in metres, right-handed, **+Y up**; ``pose`` maps tracker
coordinates to world coordinates. SteamVR setup (with the tracker wired over USB, so no pairing) is in
``docs/guides/steamvr-vive-tracker-setup.md``; the tracker's own axes are in ``i2rt/yumi/README.md``.

``ViveTracker`` follows one tracker from the shared reader -- by serial, or the first one SteamVR reports -- and
is what ``Yumi`` runs in its Vive process.

Run ``python i2rt/yumi/vive_tracker.py`` to list the connected devices (and so find a tracker's serial),
then watch their live poses.
"""

from __future__ import annotations

import atexit
import dataclasses
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import tyro

# ``import openvr`` dlopens a bundled x86-64 library, so on an arm64 host it raises OSError rather than
# ImportError. The guard keeps this module importable everywhere; ViveReader() raises a clear error instead.
_OPENVR_IMPORT_ERROR: str | None = None
if TYPE_CHECKING:
    import openvr
    import openvr.error_code
else:
    try:
        import openvr
        import openvr.error_code
    except (ImportError, OSError) as _exc:
        openvr = None
        _OPENVR_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

logger = logging.getLogger(__name__)

_SETUP_GUIDE = "docs/guides/steamvr-vive-tracker-setup.md"

# SteamVR's Steam app id, and the processes whose presence means it is up.
_STEAMVR_APP_ID = "250820"
_STEAMVR_PROCESSES = ("vrserver", "vrcompositor")
_STEAMVR_START_TIMEOUT_S = 20.0

# The poll period (250 Hz, the gripper's rate). SteamVR serves predicted poses at any rate, so this sets the pose
# stream's rate; 250 Hz is well above a 30 fps camera or any recorded trajectory, and faster only costs CPU.
_POLL_PERIOD_S = 1.0 / 250.0
_ERROR_BACKOFF_S = 0.01
_ERROR_LOG_INTERVAL_S = 1.0
_CLOSE_JOIN_TIMEOUT_S = 1.0
# How often ViveTracker re-checks for a tracker while waiting for the first one.
_DETECT_POLL_S = 0.05


@dataclass(frozen=True)
class TrackerPose:
    """One device's pose from one poll."""

    pose: np.ndarray
    """4x4 float64 homogeneous transform, tracker -> OpenVR standing world (metres). Identity when invalid."""
    valid: bool
    """``bPoseIsValid`` and ``TrackingResult_Running_OK``; False for a disconnected or rotation-only device."""
    timestamp: float
    """``time.time()`` of the poll that produced this pose."""


@dataclass(frozen=True)
class TrackedDeviceInfo:
    """One tracked device as SteamVR reported it."""

    index: int
    """OpenVR tracked-device index (a slot, not an identity)."""
    device_class: str
    """``"tracker"``, ``"controller"``, ``"hmd"`` or ``"tracking_reference"`` (a base station)."""
    serial: str
    """Device serial, or ``device_index_<index>`` if SteamVR would not report one. The key of ``get_pose``."""
    model: str
    """Model number string ("" if SteamVR would not report one)."""
    connected: bool
    """False once SteamVR has deactivated the device (or quit)."""


def _require_openvr() -> None:
    if openvr is None:
        raise ImportError(
            "openvr is unavailable, so Vive tracking cannot run on this host: either the package is not "
            "installed or its bundled native library (x86-64 only) failed to load "
            f"({_OPENVR_IMPORT_ERROR}). See {_SETUP_GUIDE}."
        )


def _is_steamvr_running() -> bool:
    for name in _STEAMVR_PROCESSES:
        result = subprocess.run(
            ["pgrep", "-x", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        if result.returncode == 0:
            return True
    return False


def _try_launch_steamvr() -> None:
    # The URL-scheme launcher first, then the Steam CLI; either may be missing on a headless host. A new session
    # keeps a Ctrl-C in this terminal from also killing the SteamVR (or Steam) it started.
    for cmd in (["xdg-open", f"steam://rungameid/{_STEAMVR_APP_ID}"], ["steam", "-applaunch", _STEAMVR_APP_ID]):
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except FileNotFoundError:
            continue
        logger.info("Launching SteamVR with: %s", " ".join(cmd))
        return
    logger.warning("Could not launch SteamVR automatically; start it manually (see %s).", _SETUP_GUIDE)


def _ensure_steamvr_running() -> None:
    """Start SteamVR if it is not running, and wait briefly for it. A no-op when it is already up."""
    if shutil.which("pgrep") is None:
        logger.warning("pgrep is not available, so whether SteamVR is running cannot be checked; assuming it is.")
        return
    if _is_steamvr_running():
        return
    _try_launch_steamvr()
    deadline = time.monotonic() + _STEAMVR_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if _is_steamvr_running():
            return
        time.sleep(0.5)
    logger.warning(
        "SteamVR did not start within %.0f s; OpenVR initialisation will probably fail.", _STEAMVR_START_TIMEOUT_S
    )


def _device_class_name(device_class: int) -> str | None:
    # Built per call rather than at import: openvr may be None at import, and tests swap the module.
    return {
        openvr.TrackedDeviceClass_GenericTracker: "tracker",
        openvr.TrackedDeviceClass_Controller: "controller",
        openvr.TrackedDeviceClass_HMD: "hmd",
        openvr.TrackedDeviceClass_TrackingReference: "tracking_reference",
    }.get(device_class)


def _string_property(vr: Any, index: int, prop: int) -> str:
    # openvr raises a TrackedPropertyError subclass (rather than returning "") when a property is unavailable.
    try:
        return vr.getStringTrackedDeviceProperty(index, prop)
    except openvr.error_code.TrackedPropertyError as exc:
        logger.debug("Vive device %d: property %d unavailable (%r)", index, prop, exc)
        return ""


def _pose_to_matrix(pose: Any) -> np.ndarray:
    """OpenVR's row-major 3x4 device-to-world matrix as a 4x4 float64 homogeneous transform."""
    matrix = np.eye(4)
    matrix[:3, :4] = np.asarray(pose.mDeviceToAbsoluteTracking.m, dtype=np.float64)
    return matrix


def _pose_is_valid(pose: Any) -> bool:
    return bool(pose.bPoseIsValid) and pose.eTrackingResult == openvr.TrackingResult_Running_OK


class ViveReader:
    """The process-wide OpenVR client: ``ViveReader()`` returns the live instance, creating it if needed.

    Construction starts SteamVR if it is not running (waiting up to 20 s), initialises OpenVR, and waits up to
    ``FIRST_POLL_TIMEOUT_S`` for the first poll. The reader is shared: ``close()`` closes it for every holder,
    after which getters raise ``RuntimeError`` and the next ``ViveReader()`` initialises a fresh reader.
    """

    STALE_AFTER_S: ClassVar[float] = 0.5
    """``get_pose`` raises ``TimeoutError`` once the last successful poll is older than this."""
    FIRST_POLL_TIMEOUT_S: ClassVar[float] = 5.0
    """How long construction waits for the first successful poll."""

    _instance: ClassVar[ViveReader | None] = None
    # Reentrant because a failed construction closes itself while __new__ still holds it. It is held across the
    # whole of construction and close, which serialises the process-global openvr.init / openvr.shutdown.
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __new__(cls) -> ViveReader:
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._start()
                cls._instance = instance
            return cls._instance

    def _start(self) -> None:
        _require_openvr()
        self._state_lock = threading.Lock()
        self._stop = threading.Event()
        self._first_poll = threading.Event()
        # Guarded by _state_lock: the device table (index -> info) and the latest pose of every serial ever seen.
        self._devices: dict[int, TrackedDeviceInfo] = {}
        self._poses: dict[str, TrackerPose] = {}
        self._last_poll_mono: float | None = None
        self._error_count = 0
        self._last_error: str | None = None
        self._quit = False
        self._closed = False
        # Touched only by the poll thread (and by close() once that thread has exited).
        self._discovered = False
        self._last_error_log_mono = float("-inf")
        self._thread: threading.Thread | None = None
        self._vr: Any = None

        _ensure_steamvr_running()
        try:
            self._vr = openvr.init(openvr.VRApplication_Other)
        except openvr.error_code.OpenVRError as exc:
            raise RuntimeError(
                f"OpenVR failed to initialise ({exc!r}). Check that SteamVR is installed and running and that "
                f'"requireHmd" is false in its default.vrsettings; see {_SETUP_GUIDE}.'
            ) from exc
        try:
            atexit.register(self.close)
            self._thread = threading.Thread(target=self._run, name="vive_reader", daemon=True)
            self._thread.start()
            self._wait_for_first_poll()
            devices = self.list_devices()
            logger.info(
                "OpenVR initialised; %d device(s): %s",
                len(devices),
                ", ".join(f"{d.device_class} {d.serial}" for d in devices) or "none",
            )
        except BaseException:
            self.close()
            raise

    def _wait_for_first_poll(self) -> None:
        deadline = time.monotonic() + self.FIRST_POLL_TIMEOUT_S
        while not self._first_poll.wait(0.05):
            if self._quit:
                raise RuntimeError("SteamVR quit while the Vive reader was starting.")
            if time.monotonic() >= deadline:
                with self._state_lock:
                    errors, last_error = self._error_count, self._last_error
                raise TimeoutError(
                    f"No Vive poll succeeded within {self.FIRST_POLL_TIMEOUT_S:.1f} s "
                    f"({errors} poll error(s), last: {last_error}). Is SteamVR running? See {_SETUP_GUIDE}."
                )

    # ------------------------------------------------------------------ poll thread

    def _run(self) -> None:
        while not self._stop.is_set() and not self._quit:
            started = time.monotonic()
            try:
                self._poll_once()
            except Exception as exc:
                self._record_error(exc)
                self._stop.wait(_ERROR_BACKOFF_S)
                continue
            # Sleep the rest of the period, not a fixed amount: the poll already used part of it.
            self._stop.wait(max(0.0, _POLL_PERIOD_S - (time.monotonic() - started)))

    def _record_error(self, exc: Exception) -> None:
        with self._state_lock:
            self._error_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            count = self._error_count
        now = time.monotonic()
        if now - self._last_error_log_mono >= _ERROR_LOG_INTERVAL_S:
            self._last_error_log_mono = now
            logger.warning("Vive poll failed (%d error(s) so far): %r", count, exc)

    def _poll_once(self) -> None:
        vr = self._vr
        self._drain_events(vr)
        if self._quit:
            return
        poses = vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
        )
        wall = time.time()
        mono = time.monotonic()
        if not self._discovered:
            # Devices already connected at init raise no activation event, so the first poll finds them.
            for index in range(len(poses)):
                if poses[index].bDeviceIsConnected:
                    self._add_device(vr, index)
            self._discovered = True

        with self._state_lock:
            devices = list(self._devices.values())
            seen_serials = list(self._poses)
        # A serial seen earlier but not connected now keeps reporting (identity, invalid): KeyError is reserved
        # for a serial that was never seen, so a tracker dropping out reads as "lost", not as "wrong serial".
        records = {serial: TrackerPose(np.eye(4), False, wall) for serial in seen_serials}
        for info in devices:
            if info.connected and info.index < len(poses) and _pose_is_valid(poses[info.index]):
                records[info.serial] = TrackerPose(_pose_to_matrix(poses[info.index]), True, wall)
            else:
                records.setdefault(info.serial, TrackerPose(np.eye(4), False, wall))
        with self._state_lock:
            self._poses = records
            self._last_poll_mono = mono
        self._first_poll.set()

    def _drain_events(self, vr: Any) -> None:
        event = openvr.VREvent_t()
        while vr.pollNextEvent(event):
            if event.eventType == openvr.VREvent_TrackedDeviceActivated:
                self._add_device(vr, event.trackedDeviceIndex)
            elif event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
                self._deactivate_device(event.trackedDeviceIndex)
            elif event.eventType == openvr.VREvent_Quit:
                self._handle_quit(vr)
                return

    def _add_device(self, vr: Any, index: int) -> None:
        device_class = _device_class_name(vr.getTrackedDeviceClass(index))
        if device_class is None:
            return
        serial = _string_property(vr, index, openvr.Prop_SerialNumber_String) or f"device_index_{index}"
        model = _string_property(vr, index, openvr.Prop_ModelNumber_String)
        info = TrackedDeviceInfo(index=index, device_class=device_class, serial=serial, model=model, connected=True)
        with self._state_lock:
            self._devices[index] = info
        logger.info("Vive %s %s (%s) connected at index %d", device_class, serial, model, index)

    def _deactivate_device(self, index: int) -> None:
        with self._state_lock:
            info = self._devices.get(index)
            if info is not None:
                self._devices[index] = dataclasses.replace(info, connected=False)
        if info is not None:
            logger.warning("Vive %s %s disconnected", info.device_class, info.serial)

    def _handle_quit(self, vr: Any) -> None:
        logger.warning("SteamVR is quitting; the Vive reader is shutting OpenVR down so this process survives.")
        wall = time.time()
        with self._state_lock:
            self._quit = True
            self._devices = {i: dataclasses.replace(d, connected=False) for i, d in self._devices.items()}
            self._poses = {serial: TrackerPose(np.eye(4), False, wall) for serial in self._poses}
        try:
            vr.acknowledgeQuit_Exiting()
        finally:
            try:
                openvr.shutdown()
            finally:
                self._vr = None

    # ------------------------------------------------------------------ public API

    def _check_stream_locked(self) -> None:
        if self._closed:
            raise RuntimeError("ViveReader is closed; construct a new ViveReader().")
        if self._quit:
            raise RuntimeError("SteamVR quit; restart SteamVR, then close() this reader and construct a new one.")
        thread_alive = self._thread is not None and self._thread.is_alive()
        age = None if self._last_poll_mono is None else time.monotonic() - self._last_poll_mono
        if not thread_alive or age is None or age > self.STALE_AFTER_S:
            age_text = "never polled" if age is None else f"last poll {age:.3f} s ago"
            raise TimeoutError(
                f"Vive pose stream is stale ({age_text}, limit {self.STALE_AFTER_S:.3f} s; poll thread "
                f"{'alive' if thread_alive else 'dead'}; {self._error_count} poll error(s), last: {self._last_error})"
            )

    def get_pose(self, serial: str) -> TrackerPose:
        """The latest pose of the device with this serial (a copy).

        Raises:
            KeyError: the serial was never seen; the message lists the serials that were.
            TimeoutError: the poll stream is stale or its thread has died.
            RuntimeError: the reader is closed, or SteamVR quit.
        """
        with self._state_lock:
            self._check_stream_locked()
            record = self._poses.get(serial)
            known = sorted(self._poses)
        if record is None:
            raise KeyError(f"Vive device {serial!r} has not been seen; known serials: {known}")
        return TrackerPose(pose=record.pose.copy(), valid=record.valid, timestamp=record.timestamp)

    def list_devices(self) -> list[TrackedDeviceInfo]:
        """A snapshot of every device seen since init, by index, including ones since disconnected."""
        with self._state_lock:
            if self._closed:
                raise RuntimeError("ViveReader is closed; construct a new ViveReader().")
            return [self._devices[index] for index in sorted(self._devices)]

    def close(self) -> None:
        """Stop the poll thread and shut OpenVR down, for every holder of this shared reader. Idempotent."""
        cls = type(self)
        with cls._lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
            if cls._instance is self:
                cls._instance = None
            atexit.unregister(self.close)
            self._stop.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
            if thread is not None and thread.is_alive():
                logger.warning(
                    "Vive poll thread did not exit within %.1f s; leaving OpenVR initialised.", _CLOSE_JOIN_TIMEOUT_S
                )
                return
            if self._vr is not None:
                self._vr = None
                openvr.shutdown()
            logger.info("Vive reader closed.")

    def __enter__(self) -> ViveReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class ViveTracker:
    """One tracker's pose stream, from the process's shared ``ViveReader``: what ``Yumi`` runs in its Vive process.

    Without a serial it follows the first tracker SteamVR reports (the lowest device index), waiting up to
    ``detect_timeout_s`` for one to connect, and keeps following that tracker for its lifetime. With a serial it
    does not wait: a tracker that has not appeared yet reads as ``KeyError`` from ``get_pose`` until it does.

    Attributes:
        serial: the tracker being followed.
    """

    def __init__(self, serial: str | None = None, detect_timeout_s: float = 30.0) -> None:
        """Start (or join) the shared reader, and pick the first tracker if no serial is given.

        Raises:
            TimeoutError: no serial was given and no tracker connected within ``detect_timeout_s``.
        """
        self._reader = ViveReader()
        try:
            self.serial = serial if serial is not None else self._first_tracker(detect_timeout_s)
        except BaseException:
            self._reader.close()
            raise

    def _first_tracker(self, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while True:
            devices = self._reader.list_devices()
            trackers = [d.serial for d in devices if d.device_class == "tracker" and d.connected]
            if len(trackers) == 1:
                logger.info("Following Vive tracker %s, the only one connected.", trackers[0])
                return trackers[0]
            if trackers:
                logger.warning(
                    "%d Vive trackers are connected (%s); following the first, %s. Pass its serial to pick another.",
                    len(trackers),
                    ", ".join(trackers),
                    trackers[0],
                )
                return trackers[0]
            if time.monotonic() >= deadline:
                seen = ", ".join(f"{d.device_class} {d.serial}" for d in devices) or "nothing"
                raise TimeoutError(
                    f"No Vive tracker connected within {timeout_s:.1f} s (SteamVR reports: {seen}). Check the "
                    f"tracker's USB cable; see {_SETUP_GUIDE}."
                )
            time.sleep(_DETECT_POLL_S)

    def get_pose(self) -> TrackerPose:
        """The followed tracker's latest pose; raises as ``ViveReader.get_pose`` does."""
        return self._reader.get_pose(self.serial)

    def close(self) -> None:
        """Close the shared reader (for every holder in this process). Idempotent."""
        self._reader.close()


def _format_device(info: TrackedDeviceInfo) -> str:
    state = "connected" if info.connected else "disconnected"
    return f"[{info.index:2d}] {info.device_class:<18} serial={info.serial}  model={info.model}  ({state})"


def main() -> None:
    """List the Vive devices SteamVR reports (class, serial, model), then redraw their live 4x4 poses.

    This is how to find a tracker's serial. Press Ctrl-C to exit.
    """
    with ViveReader() as reader:
        devices = reader.list_devices()
        print(f"Found {len(devices)} tracked device(s):")
        for info in devices:
            print(f"  {_format_device(info)}")
        time.sleep(2.0)
        try:
            while True:
                lines = []
                for info in reader.list_devices():
                    lines.append(_format_device(info))
                    try:
                        pose = reader.get_pose(info.serial)
                    except (TimeoutError, KeyError) as exc:
                        # KeyError: activated since the last poll, so it has no pose yet.
                        lines.append(f"  {exc}")
                        continue
                    lines.append(f"  valid={pose.valid}")
                    matrix = np.array2string(pose.pose, formatter={"float_kind": lambda x: f"{x:7.3f}"})
                    lines.extend(f"  {row}" for row in matrix.splitlines())
                # Clear the screen and home the cursor, so the poses redraw in place instead of scrolling.
                print("\x1b[2J\x1b[H" + "\n".join(lines or ["No tracked devices."]), flush=True)
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    from i2rt.utils.utils import override_log_level

    override_log_level(logging.INFO)
    tyro.cli(main)
