"""Read the yumi's passive UMI gripper through the passive CAN encoder on its handle.

The gripper has no motor: the operator squeezes the jaws by hand, and the only sensor is a passive joint
encoder on the handle, answering on its own CAN bus. ``UmiGripper`` owns that bus for its whole lifetime and
polls the encoder from a single background thread -- ``CanInterface`` has no locking, so exactly one thread may
talk on it, and a reader that shared the bus with anything else would misread the other party's frames.

**Construction side effects.** Building the ``PassiveEncoderReader`` runs ``PassiveJointEncoder.validate_encoders``,
which opens its own socket on the channel, takes about 2 s, raises when no encoder answers, and may write the
encoder's EEPROM (``report_freq=0``, the polled mode this reader needs). SocketCAN hands every socket its own
copy of each frame, so whatever validation left queued on *our* socket is drained before polling starts.

**Mapping.** ``position = 1 - |clip(angle, +-range_rad)| / range_rad``: 1.0 is open and 0.0 is closed, the i2rt
``gripper_pos`` convention. The range is fixed rather than calibrated, which has two consequences:

* The encoder must be zeroed at the jaws' **full-open mechanical stop**
  (``python devices/config_passive_encoder.py --bus <can> reset-zero-position`` with the jaws held open). The
  ``abs()`` folds the sign, so a zero anywhere else reads as "open" on both sides of it.
* Open reads about 1.0, and a full squeeze reads ``1 - theta_squeeze / range_rad`` -- 0.0 only if the squeeze
  spans at least ``range_rad`` (5.1 rad by default, the measured full-squeeze travel).

Nothing here re-zeroes the encoder or refuses to start on a suspicious reading: a check that writes EEPROM or
gates startup stays opt-in until the fleet has been surveyed.

**Rate and staleness.** A CAN round trip is about 1.4 ms, so an unthrottled loop would free-run near 700 Hz and
keep the bus busy for no benefit; 250 Hz is well above a 30 fps camera or any recorded gripper trace. One failed
read blocks for about 0.3 s (15 retries) before it raises, so the 1 s staleness default is more than three failed
reads in a row -- a single dropped frame never surfaces as an exception, a disconnected encoder does.

Usage (bring-up; prints the position, the raw encoder angle and velocity, and the buttons at about 20 Hz)::

    python i2rt/yumi/gripper.py --channel can0
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from types import TracebackType

import tyro

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.dm_driver import PassiveEncoderReader
from i2rt.utils.encoder_manager import EncoderCanID, drain_bus
from i2rt.utils.utils import override_log_level

logger = logging.getLogger(__name__)

DEFAULT_RANGE_RAD = 5.1
"""Encoder angle (rad) from the full-open zero that maps to fully closed: the measured full-squeeze travel of the
yumi handle (about 5.1 rad). A smaller range would clip before full close; a larger one would never reach 0.0."""

# Pause after a failed read, so a dead bus is retried promptly without spinning.
_ERROR_BACKOFF_S = 0.01
# A failing encoder raises on every read; log it about once a second, not at the poll rate.
_ERROR_LOG_INTERVAL_S = 1.0
# Longer than one failed read (~0.3 s), so close() normally sees the thread exit.
_CLOSE_JOIN_TIMEOUT_S = 1.0


@dataclass(frozen=True)
class GripperState:
    """One gripper reading."""

    timestamp: float
    """``time.time()`` of the read, comparable across the yumi's sensors."""
    position: float
    """Jaw opening: 0.0 = closed ... 1.0 = open."""
    raw_angle: float
    """Encoder angle in rad from its zero, as reported (signed, not clipped to ``range_rad``)."""
    raw_velocity: float
    """Encoder angular velocity in rad/s, as reported (signed, not normalized)."""
    buttons: tuple[bool, bool]
    """The encoder's two digital inputs (its ``io_inputs``), True = pressed."""


class UmiGripper:
    """Polls the passive UMI gripper's handle encoder on one CAN channel from a background thread."""

    def __init__(
        self,
        channel: str,
        range_rad: float = DEFAULT_RANGE_RAD,
        poll_hz: float = 250.0,
        stale_after_s: float = 1.0,
        first_sample_timeout_s: float = 5.0,
    ) -> None:
        """Open ``channel``, validate the encoder, and wait for its first reading.

        Args:
            channel: SocketCAN channel the handle encoder is on, e.g. ``"can0"``.
            range_rad: Encoder angle from the full-open zero that reads as fully closed.
            poll_hz: Encoder poll rate.
            stale_after_s: ``get_state`` raises ``TimeoutError`` when the newest reading is older than this.
            first_sample_timeout_s: How long construction waits for the first reading before raising.
        """
        if range_rad <= 0 or poll_hz <= 0:
            raise ValueError(f"range_rad and poll_hz must be positive, got {range_rad} and {poll_hz}")
        self.channel = channel
        self.range_rad = range_rad
        self._period_s = 1.0 / poll_hz
        self._stale_after_s = stale_after_s

        self._lock = threading.Lock()
        self._latest: GripperState | None = None
        self._latest_monotonic = -math.inf
        self._error_count = 0
        self._last_error: str | None = None
        self._last_error_log = -math.inf
        self._closed = False
        self._first_sample = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._can = CanInterface(channel=channel, use_buffered_reader=False)
        try:
            self._reader = PassiveEncoderReader(self._can)
            drained = drain_bus(self._can.bus)
            if drained:
                logger.info(f"Drained {drained} stale CAN frames on {channel} after encoder validation")
            self._thread = threading.Thread(target=self._poll_loop, name=f"umi-gripper-{channel}", daemon=True)
            self._thread.start()
            if not self._first_sample.wait(first_sample_timeout_s):
                raise TimeoutError(
                    f"No gripper encoder reading on {channel} within {first_sample_timeout_s} s; {self._health()}"
                )
        except BaseException:
            self.close()
            raise

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                # One raw read per poll: the position is derived here, so the raw angle is kept alongside it.
                info = self._reader.read_raw_encoder(EncoderCanID.REQ)
                angle = float(info.position)
                state = GripperState(
                    timestamp=time.time(),
                    position=1.0 - min(abs(angle), self.range_rad) / self.range_rad,
                    raw_angle=angle,
                    raw_velocity=float(info.velocity),
                    buttons=(bool(info.io_inputs[0]), bool(info.io_inputs[1])),
                )
                with self._lock:
                    self._latest = state
                    self._latest_monotonic = time.monotonic()
                self._first_sample.set()
            except Exception as exc:
                self._record_error(exc)
                self._stop.wait(_ERROR_BACKOFF_S)
                continue
            # Sleep the rest of the period, not a fixed amount: the read already used part of it.
            self._stop.wait(max(0.0, self._period_s - (time.monotonic() - started)))

    def _record_error(self, exc: Exception) -> None:
        now = time.monotonic()
        with self._lock:
            self._error_count += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            should_log = now - self._last_error_log >= _ERROR_LOG_INTERVAL_S
            if should_log:
                self._last_error_log = now
            health = self._health_locked()
        if should_log:
            logger.warning(f"Gripper encoder read failed on {self.channel}; {health}")

    def _health_locked(self) -> str:
        return f"{self._error_count} read errors, last error: {self._last_error or 'none'}"

    def _health(self) -> str:
        with self._lock:
            return self._health_locked()

    def get_state(self) -> GripperState:
        """Return the newest reading.

        Raises:
            RuntimeError: The gripper is closed (as ``ViveReader`` and ``MFCamera`` do), so "closed" is never
                mistaken for a stale stream.
            TimeoutError: The newest reading is older than ``stale_after_s`` or the poll thread has died. The
                message carries the reading's age, the error count and the last error.
        """
        with self._lock:
            state = self._latest
            age = time.monotonic() - self._latest_monotonic
            health = self._health_locked()
        if self._closed:
            raise RuntimeError(f"UmiGripper on {self.channel} is closed")
        if self._thread is None or not self._thread.is_alive():
            raise TimeoutError(f"Gripper poll thread on {self.channel} is not running; {health}")
        if state is None or age > self._stale_after_s:
            raise TimeoutError(
                f"Gripper reading on {self.channel} is {age:.3f} s old (stale after {self._stale_after_s} s); {health}"
            )
        return state

    def close(self) -> None:
        """Stop polling and release the CAN bus. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=_CLOSE_JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning(
                    f"Gripper poll thread on {self.channel} did not stop within {_CLOSE_JOIN_TIMEOUT_S} s; "
                    "leaving the CAN bus open"
                )
                return
        try:
            self._can.close()
        except Exception:
            logger.warning(f"Error closing the CAN interface on {self.channel}", exc_info=True)

    def __enter__(self) -> UmiGripper:
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
    """Print the yumi gripper position, raw encoder angle and velocity, and buttons at about 20 Hz, for bring-up."""

    channel: str = "can0"
    """CAN channel of the gripper's handle encoder."""
    range_rad: float = DEFAULT_RANGE_RAD
    """Encoder angle (rad) from the full-open zero that reads as fully closed."""


def main(args: _CliArgs) -> None:
    with UmiGripper(args.channel, range_rad=args.range_rad) as gripper:
        print("Open and squeeze the gripper; Ctrl-C to quit.")
        try:
            while True:
                state = gripper.get_state()
                print(
                    f"\rposition {state.position:6.3f} (0 closed, 1 open)  raw angle {state.raw_angle:+7.3f} rad  "
                    f"raw velocity {state.raw_velocity:+8.3f} rad/s  "
                    f"buttons {int(state.buttons[0])} {int(state.buttons[1])} (1 pressed)",
                    end="",
                    flush=True,
                )
                time.sleep(0.05)
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    override_log_level(logging.INFO)
    main(tyro.cli(_CliArgs))
