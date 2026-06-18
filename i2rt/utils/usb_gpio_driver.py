"""USB-to-GPIO converter driver and an RPi.GPIO-compatible backend.

The linear-rail brake and limit switches are driven through GPIO. On a Raspberry
Pi this is the Pi's native GPIO (``from RPi import GPIO``). On an x86 / non-Pi
host the same lines are wired to a bestep USB-to-16-channel GPIO converter
(hardware id ``ZT-DPI/SY``) exposed as a serial port (default ``/dev/ttyUSB0``).

This module provides:

* :class:`UsbToGpio` -- a small synchronous driver for the converter's binary
  serial protocol (115200 baud, 8N1), ported from the reference implementation.
* :class:`UsbGpioBackend` -- an object exposing the same surface as the
  ``RPi.GPIO`` module (``setmode``/``setup``/``output``/``input``/
  ``add_event_detect``/``remove_event_detect``/``cleanup`` and the ``BCM``/``OUT``/
  ``IN``/``PUD_UP``/``HIGH``/``LOW``/``BOTH`` constants). The converter has no
  hardware edge interrupts, so ``add_event_detect`` is emulated with a background
  polling thread that debounces and fires the callback on level changes.
* :func:`get_gpio_backend` -- a factory that returns the native ``RPi.GPIO``
  module on a Pi (ARM) and a singleton :class:`UsbGpioBackend` everywhere else.

``import serial`` is deliberately done lazily inside :class:`UsbToGpio` so this
module imports cleanly on a Pi (where ``pyserial`` is not installed and the
converter is never constructed).
"""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from collections.abc import Callable
from enum import IntEnum
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BAUDRATE = 115200
# Replies are a few bytes at 115200 baud, so 0.2 s is generous. Kept short on
# purpose: the poll thread holds the serial lock for up to one read timeout, so a
# stalled converter must not block a brake output() for ~1 s (see UsbGpioBackend.output).
DEFAULT_TIMEOUT_S = 0.2
NUM_GPIO = 16  # channels are numbered 1..16

# Opening the port toggles DTR, which resets the board; once it finishes booting
# it emits a one-time 0x2A 0xFF frame. Since the request path flushes the input
# buffer *before* writing, this frame can arrive afterwards and be read in place
# of the first reply, so _request drops it and resends once.
BOOT_FRAME = bytes((0x2A, 0xFF))

# Factory configuration.
DEFAULT_PORT = "/dev/ttyUSB0"
ENV_PORT_VAR = "I2RT_USB_GPIO_PORT"
_RASPBERRY_PI_MACHINES = ("aarch64", "armv7l", "armv6l")


def is_raspberry_pi() -> bool:
    """True on a Raspberry Pi (ARM), where GPIO is native and no USB-GPIO converter is used."""
    return platform.machine().lower() in _RASPBERRY_PI_MACHINES

# Polling cadence for the add_event_detect emulation (≫ faster than the 50 ms
# debounce and the homing loop's 10 ms tick, so limit hits are caught promptly).
POLL_INTERVAL_S = 0.005

# When the converter stops responding, the poll read fails every iteration. Log
# the first failure and every Nth after that (instead of ~200/sec at POLL_INTERVAL_S),
# and back off the poll interval up to POLL_FAIL_BACKOFF_MAX_S while it keeps failing.
POLL_FAIL_LOG_EVERY = 200
POLL_FAIL_BACKOFF_MAX_S = 0.5


class UsbToGpioError(Exception):
    """Raised when the board returns a truncated, mismatched, or malformed reply."""


class Command(IntEnum):
    """Frame head bytes for the protocol commands and replies used here."""

    SET_PINS = 0x3A
    SET_PINS_REPLY = 0x2A
    READ_PIN = 0x3F
    READ_PIN_REPLY = 0x2F
    READ_INPUTS = 0x5B
    READ_INPUTS_REPLY = 0x4B


class PullMode(IntEnum):
    """Input pull configuration used by range reads (0x5B)."""

    PULL_UP = 0
    PULL_DOWN = 1


def _check_channel(channel: int) -> None:
    if not 1 <= channel <= NUM_GPIO:
        raise ValueError(f"GPIO channel {channel} out of range 1..{NUM_GPIO}")


class UsbToGpio:
    """Synchronous driver for the USB-to-16-GPIO converter.

    Open a port and call one method per protocol command. Instances are NOT
    thread-safe (one request/reply at a time); guard with your own lock if you
    share an instance across threads. Usable as a context manager.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        import serial  # local import: pyserial is only needed on x86 (see module docstring)

        self.port = port
        self.baudrate = baudrate
        # pyserial defaults to 8N1, which is what the board expects.
        self._serial = serial.Serial(port, baudrate=baudrate, timeout=timeout_s)
        logger.info(f"UsbToGpio opened on {port} (baud={baudrate})")

    def __enter__(self) -> "UsbToGpio":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying serial port."""
        if self._serial.is_open:
            self._serial.close()
            logger.info(f"UsbToGpio on {self.port} closed")

    # -- commands --------------------------------------------------------

    def set_pins(self, levels: dict[int, bool]) -> None:
        """Set specific channels high/low (0x3A) and confirm the board's echo.

        USB-serial writes are unacked, but the board acknowledges a set by echoing
        ``0x2A`` followed by the same ``(channel, level)`` pairs it received
        (``3A 03 01`` -> ``2A 03 01``). This reads that echo back and raises if it is
        missing or doesn't match the command, so a silently-dropped write (e.g. to the
        brake) is detected rather than ignored. Channels omitted keep their state.
        """
        if not levels:
            raise ValueError("set_pins requires at least one channel")
        pairs = bytearray()
        for pin, level in levels.items():
            _check_channel(pin)
            pairs += bytes((pin, int(level)))
        frame = bytes([Command.SET_PINS]) + bytes(pairs)
        reply = self._request(frame, 1 + len(pairs), Command.SET_PINS_REPLY)
        if reply[1:] != bytes(pairs):
            raise UsbToGpioError(f"set_pins echo {reply.hex(' ')} does not match command {frame.hex(' ')}")

    def read_pin(self, pin: int) -> bool:
        """Read one channel's level (0x3F -> 0x2F). True if high.

        Note: 0x3F has no pull-config byte; use :meth:`read_inputs` when a
        pull-up/down must be applied.
        """
        _check_channel(pin)
        reply = self._request(bytes([Command.READ_PIN, pin]), 3, Command.READ_PIN_REPLY)
        if reply[1] != pin:
            raise UsbToGpioError(f"read_pin reply for channel {reply[1]}, expected {pin}")
        return bool(reply[2])

    def read_inputs(self, start: int, end: int, pull: PullMode = PullMode.PULL_UP) -> dict[int, bool]:
        """Read channels ``start..end`` as inputs with a pull config (0x5B -> 0x4B).

        Returns ``{channel: level}`` (True if high) for each channel in the range.
        """
        _check_channel(start)
        _check_channel(end)
        if end < start:
            raise ValueError(f"range end {end} precedes start {start}")
        count = end - start + 1
        frame = bytes([Command.READ_INPUTS, start, end, int(pull)])
        reply = self._request(frame, 4 + count, Command.READ_INPUTS_REPLY)
        return {start + i: bool(reply[4 + i]) for i in range(count)}

    def get_version(self) -> str:
        """Query firmware/hardware version via the ASCII ``ver`` command."""
        self._serial.reset_input_buffer()
        self._write(b"ver")
        return self._read_ascii()

    # -- transport -------------------------------------------------------

    def _write(self, frame: bytes | bytearray) -> None:
        data = bytes(frame)
        logger.debug(f"-> {data.hex(' ')}")
        self._serial.write(data)

    def _read_frame(self, reply_len: int) -> bytes:
        reply = self._serial.read(reply_len)
        logger.debug(f"<- {reply.hex(' ')}")
        return reply

    def _request(self, frame: bytes, reply_len: int, reply_head: int) -> bytes:
        """Flush stale input, send ``frame``, and read a fixed-length reply.

        If the board's one-time boot frame turns up at the head of the reply,
        drop it and resend once.

        Raises:
            UsbToGpioError: on a short read (timeout) or wrong reply head byte.
        """
        self._serial.reset_input_buffer()
        self._write(frame)
        reply = self._read_frame(reply_len)
        if reply.startswith(BOOT_FRAME):
            self._drain()  # drop the boot frame and any partial reply trailing it
            self._write(frame)
            reply = self._read_frame(reply_len)
        if len(reply) != reply_len:
            raise UsbToGpioError(
                f"timed out on reply to 0x{frame[0]:02X}: expected {reply_len} bytes, "
                f"got {len(reply)} ({reply.hex(' ')})"
            )
        if reply[0] != reply_head:
            raise UsbToGpioError(f"reply head 0x{reply[0]:02X}, expected 0x{reply_head:02X} ({reply.hex(' ')})")
        return reply

    def _read_ascii(self) -> str:
        """Read bytes until a read times out, then return the decoded text."""
        chunks = bytearray()
        while True:
            chunk = self._serial.read(self._serial.in_waiting or 1)
            if not chunk:
                break
            chunks += chunk
        return chunks.decode("ascii", errors="replace").strip()

    def _drain(self) -> None:
        """Read and discard input until a read times out (drops a stale reply)."""
        while self._serial.read(self._serial.in_waiting or 1):
            pass


class UsbGpioBackend:
    """An ``RPi.GPIO``-module-compatible facade backed by :class:`UsbToGpio`.

    Drop-in for the ``GPIO`` module on x86. Pin arguments use the controller's
    BCM numbers; they are translated to converter channels via ``pin_map`` only
    at the serial boundary. ``add_event_detect`` is emulated with a background
    polling thread (the converter has no hardware edge interrupts).
    """

    # Match RPi.GPIO's integer values (the controller only does
    # ``input(pin) == HIGH`` and passes the rest as opaque tokens).
    BCM = 11
    BOARD = 10
    OUT = 0
    IN = 1
    PUD_OFF = 20
    PUD_DOWN = 21
    PUD_UP = 22
    LOW = 0
    HIGH = 1
    RISING = 31
    FALLING = 32
    BOTH = 33

    def __init__(self, port: str, pin_map: Optional[dict[int, int]] = None) -> None:
        self._port = port
        self._pin_map = dict(pin_map) if pin_map else {}
        self._dev: Optional[UsbToGpio] = None
        self._serial_lock = threading.Lock()  # the ONE lock guarding all serial access
        self._mode_set = False

        self._directions: dict[int, int] = {}
        self._pulls: dict[int, int] = {}

        # add_event_detect emulation state. Guarded by _watch_lock (never nested
        # with _serial_lock) so the poll thread and the main thread can mutate it
        # without relying on the CPython GIL. Each entry is (callback, bouncetime_s, edge).
        self._watch_lock = threading.Lock()
        self._watch: dict[int, tuple[Callable[[int], None], float, int]] = {}
        self._last_stable: dict[int, bool] = {}
        self._last_change_ts: dict[int, float] = {}
        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._poll_fail_count = 0

    # -- configuration ---------------------------------------------------

    def set_port(self, device: str) -> None:
        """Point the backend at ``device``. Must be called before the port opens."""
        # Explicit raise (not assert): the guard must hold under ``python -O`` too.
        if self._dev is not None:
            raise RuntimeError("Cannot change USB-GPIO device after the port is opened")
        self._port = device

    def _channel(self, pin: int) -> int:
        return self._pin_map.get(pin, pin)

    def _to_dev_pull(self, pud: int) -> PullMode:
        return PullMode.PULL_DOWN if pud == self.PUD_DOWN else PullMode.PULL_UP

    def _ensure_open(self) -> UsbToGpio:
        # Called outside _serial_lock; the lock is non-reentrant.
        with self._serial_lock:
            if self._dev is None:
                self._dev = UsbToGpio(self._port)
            return self._dev

    # -- RPi.GPIO surface ------------------------------------------------

    def setmode(self, mode: int) -> None:
        # Lazy open: the first GPIO call in the program is setmode (via
        # initialize_brake_gpio). The converter has no BCM/BOARD distinction.
        self._ensure_open()
        self._mode_set = True

    def setup(self, pin: int, direction: int, pull_up_down: Optional[int] = None) -> None:
        self._ensure_open()
        self._directions[pin] = direction
        if direction == self.IN:
            self._pulls[pin] = pull_up_down if pull_up_down is not None else self.PUD_OFF

    def output(self, pin: int, level: int) -> None:
        dev = self._ensure_open()
        ch = self._channel(pin)
        target = bool(level)
        # USB-serial writes are unacked, but the converter echoes a 0x2A <ch> <level>
        # frame for every 0x3A set, so set_pins() confirms the write was received.
        # If the echo is missing/garbled, resend once; if it still fails, log a
        # warning and continue -- never raise, so a flaky echo can't break brake
        # control (a raising verified-write previously broke startup; see git history).
        with self._serial_lock:
            error: Optional[Exception] = None
            for _ in range(2):  # initial write, plus one resend on a bad/missing echo
                try:
                    dev.set_pins({ch: target})
                    return
                except UsbToGpioError as e:
                    error = e
            logger.warning(
                f"USB-GPIO output(pin={pin}, ch={ch}, level={int(target)}) not confirmed by echo "
                f"after resend: {error}"
            )

    def input(self, pin: int) -> int:
        dev = self._ensure_open()
        ch = self._channel(pin)
        pull = self._to_dev_pull(self._pulls.get(pin, self.PUD_UP))
        with self._serial_lock:
            value = dev.read_inputs(ch, ch, pull=pull)[ch]
        return self.HIGH if value else self.LOW

    def add_event_detect(
        self,
        pin: int,
        edge: int,
        callback: Callable[[int], None],
        bouncetime: int = 0,
    ) -> None:
        # Seed the debounce state from a live read so no spurious edge fires at start.
        # input() takes _serial_lock, so seed BEFORE taking _watch_lock (never nest).
        seed = self.input(pin) == self.HIGH
        with self._watch_lock:
            self._last_stable[pin] = seed
            self._last_change_ts[pin] = 0.0
            self._watch[pin] = (callback, (bouncetime or 0) / 1000.0, edge)
        self._start_poll_thread()

    def remove_event_detect(self, pin: int) -> None:
        with self._watch_lock:
            self._watch.pop(pin, None)
            self._last_stable.pop(pin, None)
            self._last_change_ts.pop(pin, None)
            empty = not self._watch
        # Join the poll thread outside _watch_lock (the thread itself takes it).
        if empty:
            self._stop_poll_thread()

    def cleanup(self, pin: Optional[int] = None) -> None:
        # Stop polling and close the serial port. The converter LATCHES its last-driven
        # level across a port close, so we deliberately do NOT touch the output channels
        # here: the brake stays at its last commanded level (LOW == engaged; see
        # set_brake_gpio) and the rail stays held after a single shutdown. (Native
        # RPi.GPIO.cleanup() floats pins instead, releasing the brake on a Pi -- the
        # controller avoids that by cleaning up only the limit-switch input pins.) ``pin``
        # is accepted for RPi.GPIO API compatibility but ignored: cleanup() is only ever
        # called at process shutdown, where full teardown of the single shared port is the
        # only correct action.
        self._stop_poll_thread()  # join FIRST (outside locks) before clearing watch state
        with self._watch_lock:
            self._watch.clear()
            self._last_stable.clear()
            self._last_change_ts.clear()
        self._directions.clear()
        self._pulls.clear()
        with self._serial_lock:
            if self._dev is not None:
                self._dev.close()
                self._dev = None
        self._mode_set = False

    # -- polling-thread edge emulation -----------------------------------

    def _start_poll_thread(self) -> None:
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, name="usb-gpio-poll", daemon=True)
        self._poll_thread.start()

    def _stop_poll_thread(self) -> None:
        self._poll_stop.set()
        thread = self._poll_thread
        # Guard against joining ourselves (a callback runs on the poll thread).
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._poll_thread = None

    def _poll_loop(self) -> None:
        while not self._poll_stop.is_set():
            self._poll_stop.wait(self._poll_once())

    def _poll_once(self) -> float:
        """Run one poll iteration; return how long to wait before the next.

        Reads every watched input in one range round-trip, debounces, and fires
        callbacks whose configured ``edge`` matches the level change. Split out
        from the driver loop so the debounce/edge logic is unit-testable without
        the background thread. The returned wait is backed off after consecutive
        read failures (see POLL_FAIL_BACKOFF_MAX_S).
        """
        with self._watch_lock:
            watched = list(self._watch.items())
            channels = {pin: self._channel(pin) for pin, _ in watched}
        dev = self._dev
        if not watched or dev is None:
            return POLL_INTERVAL_S

        lo, hi = min(channels.values()), max(channels.values())
        # Honor the configured pull like input() does, instead of hardcoding PULL_UP.
        # A single range read can't express mixed pulls, so fall back to PULL_UP if
        # the watched pins disagree (they don't today -- both limits are PUD_UP).
        configured = {self._pulls.get(pin, self.PUD_UP) for pin, _ in watched}
        pull = self._to_dev_pull(configured.pop()) if len(configured) == 1 else PullMode.PULL_UP
        try:
            with self._serial_lock:  # hold the lock ONLY for the read
                readings = dev.read_inputs(lo, hi, pull=pull)
        except Exception as e:
            self._poll_fail_count += 1
            if self._poll_fail_count == 1 or self._poll_fail_count % POLL_FAIL_LOG_EVERY == 0:
                logger.warning(f"USB-GPIO poll read failed (x{self._poll_fail_count}): {e}")
            return min(POLL_INTERVAL_S * self._poll_fail_count, POLL_FAIL_BACKOFF_MAX_S)
        if self._poll_fail_count:
            logger.info(f"USB-GPIO poll read recovered after {self._poll_fail_count} failure(s)")
            self._poll_fail_count = 0
        # Lock released here -- before any callback runs (callbacks re-enter input()).

        now = time.monotonic()
        fired: list[tuple[Callable[[int], None], int]] = []
        with self._watch_lock:
            for pin, (callback, bouncetime, edge) in watched:
                if pin not in self._watch:  # removed since the snapshot
                    continue
                level = readings.get(channels[pin])
                prev = self._last_stable.get(pin)
                if level is None or level == prev:
                    continue
                if now - self._last_change_ts.get(pin, 0.0) >= bouncetime:
                    rising = level and not prev
                    self._last_stable[pin] = level
                    self._last_change_ts[pin] = now
                    if edge == self.BOTH or (edge == self.RISING and rising) or (edge == self.FALLING and not rising):
                        fired.append((callback, pin))
        # Invoke callbacks outside both locks (they re-enter input()).
        for callback, pin in fired:
            try:
                callback(pin)
            except Exception as e:
                logger.error(f"USB-GPIO callback error on pin {pin}: {e}")
        return POLL_INTERVAL_S


_BACKEND_SINGLETON: Optional[UsbGpioBackend] = None


def get_gpio_backend(port: Optional[str] = None, pin_map: Optional[dict[int, int]] = None) -> Any:
    """Return the GPIO backend for this platform.

    On a Raspberry Pi (ARM) this returns the native ``RPi.GPIO`` module so
    behavior is unchanged. Otherwise it returns a process-wide singleton
    :class:`UsbGpioBackend` driving the USB-GPIO converter. Constructing the
    backend does NOT open the serial port (that happens lazily on ``setmode``),
    so importing this module never requires the device to be present.

    Port precedence (highest first): an explicit :meth:`UsbGpioBackend.set_port`
    call > the ``port`` argument > the ``I2RT_USB_GPIO_PORT`` env var > ``/dev/ttyUSB0``.
    """
    if is_raspberry_pi():
        from RPi import GPIO  # native backend on the Pi; behavior unchanged

        return GPIO

    global _BACKEND_SINGLETON
    if _BACKEND_SINGLETON is None:
        resolved = port or os.environ.get(ENV_PORT_VAR, DEFAULT_PORT)
        _BACKEND_SINGLETON = UsbGpioBackend(resolved, pin_map=pin_map)
    return _BACKEND_SINGLETON
