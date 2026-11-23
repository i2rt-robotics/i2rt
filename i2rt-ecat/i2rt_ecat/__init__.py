"""CAN frames over an EtherCAT-to-CAN board, from Python.

A thin ctypes wrapper over ``libi2rt_ecat.so`` (``src/ecat_can_bus.cpp``), which runs the EtherCAT
master (SOEM) on its own real-time thread inside this process. The library knows no motor protocol:
it moves 8-byte CAN payloads. Two modes:

* **cyclic** -- ``EcatCanBus(interface, motors=[(slave, channel, can_id), ...], release=...)``. Each
  motor has a latched frame, re-sent whenever its slot comes round (three per CAN channel per
  cycle), and a latest-reply buffer routed by (slave, channel, reply id). If ``set_frames`` is not
  called for ``watchdog_ms`` the latched frames are replaced by ``release`` until it is.
* **raw** -- ``EcatCanBus(interface)`` with no motors: ``send_raw`` queues a frame to go out once, and
  ``recv_raw`` returns every frame received, in order. For register access and one-shot commands.

Opening needs ``cap_net_raw`` (and ``cap_sys_nice`` for real-time priority): run as root, or grant
them to the interpreter. Every ctypes call releases the GIL.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import threading
import time
from typing import Dict, Iterator, NamedTuple, Optional, Sequence, Tuple

import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libi2rt_ecat.so"))

_u8p = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
_i32p = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_f64p = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_u64p = np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")

_lib.ecb_open.restype = ctypes.c_void_p
_lib.ecb_open.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    _i32p,
    ctypes.c_int,
    ctypes.c_int,
    _u8p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
]
_lib.ecb_set_frames.argtypes = [ctypes.c_void_p, _u8p]
_lib.ecb_read.argtypes = [ctypes.c_void_p, _u8p, _f64p, _u64p]
_lib.ecb_stats.argtypes = [ctypes.c_void_p, _u64p]
_lib.ecb_used_slaves.restype = ctypes.c_int
_lib.ecb_used_slaves.argtypes = [ctypes.c_void_p]
_lib.ecb_raw_send.restype = ctypes.c_int
_lib.ecb_raw_send.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint32,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
]
_lib.ecb_raw_recv.restype = ctypes.c_int
_lib.ecb_raw_recv.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_uint32),
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_double),
]
_lib.ecb_trace_start.argtypes = [ctypes.c_void_p, ctypes.c_int]
_lib.ecb_trace_read.restype = ctypes.c_int
_lib.ecb_trace_read.argtypes = [ctypes.c_void_p, _u8p, ctypes.c_int]
_lib.ecb_loopback_mute.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
_lib.ecb_close.argtypes = [ctypes.c_void_p]

STAT_KEYS = ("ticks", "overruns", "max_late_us", "wkc_errors", "watchdog_trips", "rt_ok", "al_state", "raw_drops")


class RawFrame(NamedTuple):
    slave: int
    channel: int
    can_id: int
    data: bytes
    timestamp: float  # time.time() clock


class EcatCanBus:
    """One EtherCAT-to-CAN master. Only one may be open per process (SOEM keeps global state)."""

    def __init__(
        self,
        interface: str,
        motors: Sequence[Tuple[int, int, int]] = (),
        release: Optional[np.ndarray] = None,
        period_us: int = 1000,
        watchdog_ms: int = 100,
        rt_priority: int = 99,
        cpu: int = -1,
        loopback: bool = False,
    ) -> None:
        self.n = len(motors)
        motor_map = np.ascontiguousarray(np.asarray(motors, dtype=np.int32).reshape(self.n * 3))
        rel = np.zeros((self.n, 8), dtype=np.uint8) if release is None else np.asarray(release, dtype=np.uint8)
        if rel.shape != (self.n, 8):
            raise ValueError(f"release must be ({self.n}, 8) bytes, got {rel.shape}")
        err = ctypes.create_string_buffer(512)
        # A zero-length array still needs a valid pointer for ndpointer.
        motor_map = motor_map if self.n else np.zeros(1, dtype=np.int32)
        rel = np.ascontiguousarray(rel.reshape(-1)) if self.n else np.zeros(1, dtype=np.uint8)
        handle = _lib.ecb_open(
            interface.encode(),
            self.n,
            motor_map,
            period_us,
            watchdog_ms,
            rel,
            int(loopback),
            rt_priority,
            cpu,
            err,
            len(err),
        )
        if not handle:
            raise OSError(err.value.decode(errors="replace"))
        self._h: Optional[int] = handle
        # Calls in flight on the handle, so close() can wait them out before freeing it. Not a lock
        # held across calls: recv_raw blocks for up to its timeout, and a thread looping on it would
        # re-take a lock before any other thread got it.
        self._inflight = 0
        self._cv = threading.Condition()
        self.raw = self.n == 0

    @contextlib.contextmanager
    def _use(self) -> Iterator[int]:
        with self._cv:
            if self._h is None:
                raise RuntimeError("EcatCanBus is closed")
            self._inflight += 1
            handle = self._h
        try:
            yield handle
        finally:
            with self._cv:
                self._inflight -= 1
                self._cv.notify_all()

    def set_frames(self, frames: np.ndarray) -> None:
        """Latch one 8-byte frame per motor (cyclic mode), and feed the watchdog."""
        tx = np.ascontiguousarray(frames, dtype=np.uint8)
        if tx.shape != (self.n, 8):
            raise ValueError(f"frames must be ({self.n}, 8), got {tx.shape}")
        with self._use() as h:
            _lib.ecb_set_frames(h, tx.reshape(-1))

    def read(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Latest reply per motor: ``(payload (n, 8) uint8, stamp (n,) s, count (n,) uint64)``.

        A motor that has never answered has stamp 0 and count 0.
        """
        rx = np.zeros(self.n * 8 or 1, dtype=np.uint8)
        t = np.zeros(self.n or 1, dtype=np.float64)
        c = np.zeros(self.n or 1, dtype=np.uint64)
        with self._use() as h:
            _lib.ecb_read(h, rx, t, c)
        return rx[: self.n * 8].reshape(self.n, 8), t[: self.n], c[: self.n]

    def stats(self) -> Dict[str, int]:
        out = np.zeros(len(STAT_KEYS), dtype=np.uint64)
        with self._use() as h:
            _lib.ecb_stats(h, out)
        return {k: int(v) for k, v in zip(STAT_KEYS, out, strict=True)}

    @property
    def slaves(self) -> int:
        """EtherCAT-to-CAN boards in use."""
        with self._use() as h:
            return int(_lib.ecb_used_slaves(h))

    def send_raw(self, slave: int, channel: int, can_id: int, data: bytes, rtr: bool = False, slot: int = -1) -> None:
        """Queue one standard-id frame to go out once (raw mode).

        ``rtr`` sends a remote frame (``data`` then only sets its length). ``slot`` (0-2) picks which of
        the channel's three frame slots it goes out on, instead of the first free one. The board only
        accepts a reply into a slot whose last frame carried the reply's id, so a slot sent a frame
        with id R keeps listening for R -- which is how a reply with another id than its request's is
        made receivable.
        """
        data = bytes(data)
        with self._use() as h:
            rc = _lib.ecb_raw_send(h, slave, channel, can_id, len(data), data, int(rtr), slot)
        if rc == -1:
            raise ValueError(f"cannot send to slave {slave} channel {channel} ({len(data)} bytes) on this bus")
        if rc == -2:
            raise BufferError("raw transmit queue is full")

    def recv_raw(self, timeout: float = 0.0) -> Optional[RawFrame]:
        """Oldest received frame, waiting up to ``timeout`` seconds; ``None`` if none arrived."""
        s, c, dlc = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        can_id, stamp = ctypes.c_uint32(), ctypes.c_double()
        data = ctypes.create_string_buffer(8)
        with self._use() as h:
            got = _lib.ecb_raw_recv(
                h,
                int(timeout * 1e6),
                ctypes.byref(s),
                ctypes.byref(c),
                ctypes.byref(can_id),
                ctypes.byref(dlc),
                data,
                ctypes.byref(stamp),
            )
        if not got:
            return None
        return RawFrame(s.value, c.value, can_id.value, data.raw[: dlc.value], stamp.value)

    def trace(self, cycles: int) -> Tuple[np.ndarray, np.ndarray]:
        """Record board 0's raw 86-byte output and input image for the next ``cycles`` cycles.

        Blocks until they are captured. Returns ``(sent, received)``, each ``(cycles, 86)`` uint8: what
        the master wrote and what it read back on each cycle, for diagnosing the board's firmware.
        """
        with self._use() as h:
            _lib.ecb_trace_start(h, cycles)
        buf = np.zeros(cycles * 2 * 86, dtype=np.uint8)
        got = 0
        while got < cycles:
            time.sleep(0.01)
            with self._use() as h:
                got = _lib.ecb_trace_read(h, buf, cycles)
        images = buf.reshape(cycles, 2, 86)
        return images[:, 0].copy(), images[:, 1].copy()

    def loopback_mute(self, can_id: int, muted: bool = True) -> None:
        """Loopback only: stop (or resume) answering frames to ``can_id``, like a dead motor."""
        with self._use() as h:
            _lib.ecb_loopback_mute(h, can_id, int(muted))

    def close(self) -> None:
        """Stop the loop, take the bus to INIT and release the NIC. Idempotent."""
        with self._cv:
            handle, self._h = self._h, None
            while self._inflight:
                self._cv.wait()
        if handle is not None:
            _lib.ecb_close(handle)

    def __enter__(self) -> "EcatCanBus":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # The loop thread keeps running while the handle is open; never leak it past the object.
        try:
            self.close()
        except Exception:
            pass
