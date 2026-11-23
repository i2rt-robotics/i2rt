"""DM motors over an EtherCAT-to-CAN board, with the EtherCAT master running in this process.

The board is an EtherCAT slave with two CAN channels; each 1 kHz EtherCAT cycle carries up to three
CAN frames per channel each way. The master is the compiled ``i2rt_ecat`` package (SOEM on its own
real-time thread), which ``uv sync`` builds on Linux. There is no separate bridge process.

Two ways in:

* :class:`EtherCATMotorChain` -- a :class:`~i2rt.motor_drivers.dm_driver.MotorChain` for a control
  loop such as ``Arm7OSC``. Each motor's latest MIT command is latched and re-sent every time its
  slot comes round: with more than three motors on a channel they rotate, so the 7-DOF arm's seven
  motors are each served at 3000/7 = 428.6 Hz, never more than 3 ms apart.
* :class:`EtherCATCanBus` -- a python-can bus over the board, for request/response tools. Every CAN
  tool that takes a channel accepts ``ecat:<nic>`` (or ``ecat:<nic>:<slave>:<channel>``) through
  :func:`i2rt.motor_drivers.can_interface.open_can_bus`.

Motors are identified by the CAN id they reply with, so on this transport each motor must answer on
``MASTER_ID == ID``; ``python i2rt/motor_config_tool/dm_motor_registers.py read MST_ID ...`` shows it.

Opening the NIC needs ``cap_net_raw`` (and ``cap_sys_nice`` for real-time priority); the error
raised when it is missing says how to grant it.
"""

from __future__ import annotations

import queue
import threading
import time
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import can
import numpy as np

from i2rt.motor_drivers.can_interface import ETHERCAT_CHANNEL_PREFIX
from i2rt.motor_drivers.dm_driver import MotorChain
from i2rt.motor_drivers.utils import (
    DM_CLEAR_ERR,
    DM_DISABLE,
    DM_ENABLE,
    MitFeedback,
    MotorErrorCode,
    MotorInfo,
    MotorType,
    pack_mit,
    unpack_mit_feedback,
)

ERROR_CODE_OK = "0x1"

DEFAULT_RELEASE_KD = 1.0
"""Motor-side kd, N.m.s/rad, in the release frame: zero torque and zero stiffness, so a released arm
sinks damped rather than holding or dropping. Damps a fall; does not stop one."""

PRESENCE_TIMEOUT_S = 0.5
ENABLE_TIMEOUT_S = 0.5
STARTUP_PHASE_S = 0.01  # how long each startup frame (clear error, enable) is latched
CLOSE_PHASE_S = 0.02  # how long each shutdown frame (release, disable) is latched

VELOCITY_TAU_S = 0.004
"""Time constant of the first-order filter on the velocity differentiated from position, when
``derive_velocity`` is on."""


def _bus_class() -> Any:
    try:
        from i2rt_ecat import EcatCanBus
    except ImportError as exc:
        raise ImportError(
            "the EtherCAT transport needs the compiled i2rt_ecat package, which `uv sync` builds on "
            "Linux (or: `uv pip install ./i2rt-ecat`)"
        ) from exc
    return EcatCanBus


def _frames(payload: bytes, n: int) -> np.ndarray:
    return np.tile(np.frombuffer(payload, dtype=np.uint8), (n, 1))


def _release_frames(motor_list: Sequence[Tuple[int, str]], release_kd: float) -> np.ndarray:
    consts = [MotorType.get_motor_constants(t) for _, t in motor_list]
    return np.array([np.frombuffer(pack_mit(0, 0, 0, release_kd, 0, c), np.uint8) for c in consts])


def _release_and_close(bus: Any, release: np.ndarray, disable: np.ndarray) -> None:
    """Release every motor, then disable it, then stop the bus. Loud if the release fails."""
    try:
        bus.set_frames(release)
        time.sleep(CLOSE_PHASE_S)
        bus.set_frames(disable)
        time.sleep(CLOSE_PHASE_S)
    except Exception as exc:
        print(
            f"\n  *** RELEASE FAILED: {type(exc).__name__}: {exc}\n"
            "  *** The motors may still hold their last command. Power the arm down.",
            flush=True,
        )
    finally:
        bus.close()


class EtherCATMotorChain(MotorChain):
    """DM motors in MIT mode on one CAN channel of an EtherCAT-to-CAN board.

    Construct it in place of :class:`~i2rt.motor_drivers.dm_driver.DMChainCanInterface`::

        chain = EtherCATMotorChain(cfg_motor_list, motor_direction=directions, interface="enp99s0")
        arm = Arm7OSC(chain, xml_path, ee_site="grasp_site")

    Startup detects every motor before energising any: the bus opens with release frames latched, and
    only once every motor has answered are they cleared and enabled -- so a miswired or misconfigured
    chain raises having sent no enable at all. ``enable=False`` stops after detection and leaves the
    motors disabled, still reporting position.

    A motor's reply timestamp advances only while it reports itself enabled, so a motor that faults
    or disables itself looks like one that went silent, which is what a missing-frame watchdog (such
    as ``Arm7OSC``'s) is keyed to. Such a reply also clears ``running``, which is what
    ``MotorChainRobot`` stops on.

    Args:
        motor_list: ``(can_id, motor_type)`` per motor, in chain order.
        motor_offset: radians, subtracted in the motor frame before the direction flip, as
            ``DMChainCanInterface`` does.
        motor_direction: ``+1`` or ``-1`` per motor.
        interface: the NIC the board is wired to, e.g. ``enp99s0``.
        slave: the board's position on the EtherCAT chain, 0 for the first.
        channel: the board's CAN channel the motors are on, 1 or 2.
        derive_velocity: differentiate each motor's position over its own reply stamps instead of
            using the velocity it reports.
        watchdog_ms: the master re-sends latched commands for this long after the last
            ``set_commands``, then releases every motor until the next one.
        release_kd: motor-side kd in the release frame; see ``DEFAULT_RELEASE_KD``.
        enable: clear errors and enable the motors after detecting them.
        bus: this chain's slice of a master shared with other chains, from :func:`open_shared_master`,
            instead of opening one of its own; ``watchdog_ms`` and ``release_kd`` are then the master's.
    """

    def __init__(
        self,
        motor_list: Sequence[Tuple[int, str]],
        motor_offset: Optional[np.ndarray] = None,
        motor_direction: Optional[np.ndarray] = None,
        interface: str = "",
        slave: int = 0,
        channel: int = 1,
        derive_velocity: bool = False,
        watchdog_ms: int = 100,
        release_kd: float = DEFAULT_RELEASE_KD,
        enable: bool = True,
        bus: Optional["_MasterSlice"] = None,
    ) -> None:
        if not interface:
            raise ValueError("interface is required: the NIC the EtherCAT-to-CAN board is wired to, e.g. enp99s0")
        self.motor_list: List[Tuple[int, str]] = [(int(i), str(t)) for i, t in motor_list]
        n = len(self.motor_list)
        self.motor_offset = np.zeros(n) if motor_offset is None else np.asarray(motor_offset, dtype=float)
        self.motor_direction = np.ones(n) if motor_direction is None else np.asarray(motor_direction, dtype=float)
        if self.motor_offset.shape != (n,) or self.motor_direction.shape != (n,):
            raise ValueError(f"motor_offset and motor_direction must each hold {n} values")
        if not np.all(np.isin(self.motor_direction, (-1.0, 1.0))):
            # kd is sent unscaled, which is only right because a direction is +/-1.
            raise ValueError(f"motor_direction must be +1 or -1, got {self.motor_direction.tolist()}")

        self.interface, self.slave, self.channel = interface, int(slave), int(channel)
        self._n = n
        self._consts = [MotorType.get_motor_constants(t) for _, t in self.motor_list]
        if bus is not None and bus.n != n:
            raise ValueError(f"the shared master's slice carries {bus.n} motors, not {n}")
        self._release = bus.release if bus is not None else _release_frames(self.motor_list, release_kd)
        self._disable = _frames(DM_DISABLE, n)
        self._derive_velocity = bool(derive_velocity)
        self._lock = threading.Lock()
        self._closed = False
        self.running = True

        self._seen = np.zeros(n, dtype=np.uint64)
        self._fb: List[MitFeedback] = [MitFeedback(0, 0.0, 0.0, 0.0, 0.0, 0.0)] * n
        self._stamp = np.zeros(n)
        self._reply_time = np.zeros(n)
        self._derived_vel = np.zeros(n)

        if bus is None:
            bus = _bus_class()(
                interface,
                [(self.slave, self.channel, can_id) for can_id, _ in self.motor_list],
                release=self._release,
                watchdog_ms=watchdog_ms,
            )
        self._bus = bus
        self._opened_at = time.time()
        self._finalizer = weakref.finalize(self, _release_and_close, self._bus, self._release, self._disable)
        try:
            self._await_every_motor(PRESENCE_TIMEOUT_S)
            if enable:
                self._enable()
        except BaseException:
            self.close()
            raise

    # ---- startup ------------------------------------------------------------------------------

    def _ids(self, mask: np.ndarray) -> List[int]:
        return [self.motor_list[i][0] for i in np.flatnonzero(mask)]

    def _await_every_motor(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            self._bus.set_frames(self._release)  # keep the watchdog fed, so its count means a real stall
            _, _, count = self._bus.read()
            silent = count == 0
            if not silent.any():
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"motors with CAN id {self._ids(silent)} did not answer on {self.interface} (board "
                    f"{self.slave}, CAN channel {self.channel}); nothing was enabled. Each motor must reply "
                    "on MASTER_ID == ID, since replies are routed by the id they carry: check with "
                    "`python i2rt/motor_config_tool/dm_motor_registers.py read MST_ID --motor-id N --channel "
                    f"{ETHERCAT_CHANNEL_PREFIX}{self.interface}`."
                )
            time.sleep(0.005)

    def _enable(self) -> None:
        for frame in (DM_CLEAR_ERR, DM_ENABLE):
            self._bus.set_frames(_frames(frame, self._n))
            time.sleep(STARTUP_PHASE_S)
        self._bus.set_frames(self._release)
        _, _, baseline = self._bus.read()
        deadline = time.monotonic() + ENABLE_TIMEOUT_S
        while True:
            self._bus.set_frames(self._release)  # keep the watchdog fed while waiting
            rx, _, count = self._bus.read()
            fresh = count > baseline
            state = np.array([unpack_mit_feedback(bytes(rx[i]), self._consts[i]).error for i in range(self._n)])
            faulted = fresh & (state >= 0x8)
            if faulted.any():
                i = int(np.flatnonzero(faulted)[0])
                self._bus.set_frames(self._disable)
                time.sleep(CLOSE_PHASE_S)
                raise RuntimeError(
                    f"motor {self.motor_list[i][0]} faulted on enable: {MotorErrorCode.get_error_message(state[i])}"
                )
            if (fresh & (state == MotorErrorCode.normal)).all():
                return
            if time.monotonic() > deadline:
                bad = ~(fresh & (state == MotorErrorCode.normal))
                raise RuntimeError(f"motors with CAN id {self._ids(bad)} did not report enabled after enable")
            time.sleep(0.005)

    # ---- MotorChain ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"EtherCATMotorChain(n_motors={self._n}, interface={self.interface}, board={self.slave}, can={self.channel})"

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> Optional[List[MotorInfo]]:
        """Latch one MIT command per motor, in model coordinates, as ``DMChainCanInterface`` takes it.

        ``None`` means zero, so ``set_commands(np.zeros(n))`` leaves every motor limp.
        """
        if self._closed:
            raise RuntimeError("EtherCATMotorChain is closed")
        n = self._n
        torques = np.asarray(torques, dtype=float)
        if torques.size < n:
            raise ValueError(f"need {n} torques, got {torques.size}")
        pos_ = np.zeros(n) if pos is None else np.asarray(pos, dtype=float)
        vel_ = np.zeros(n) if vel is None else np.asarray(vel, dtype=float)
        kp_ = np.zeros(n) if kp is None else np.asarray(kp, dtype=float)
        kd_ = np.zeros(n) if kd is None else np.asarray(kd, dtype=float)
        frames = np.empty((n, 8), dtype=np.uint8)
        for i in range(n):
            d = self.motor_direction[i]
            frames[i] = np.frombuffer(
                pack_mit(
                    pos_[i] * d + self.motor_offset[i], vel_[i] * d, kp_[i], kd_[i], torques[i] * d, self._consts[i]
                ),
                np.uint8,
            )
        self._bus.set_frames(frames)
        return self.read_states(torques=torques) if get_state else None

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        """The latest reply per motor, in model coordinates."""
        rx, stamp, count = self._bus.read()
        with self._lock:
            for i in np.flatnonzero(count != self._seen):
                fb = unpack_mit_feedback(bytes(rx[i]), self._consts[i])
                if self._derive_velocity and self._reply_time[i] > 0.0:
                    dt = stamp[i] - self._reply_time[i]
                    if 1e-4 < dt < 0.1:
                        raw = (fb.position - self._fb[i].position) / dt
                        self._derived_vel[i] += dt / (VELOCITY_TAU_S + dt) * (raw - self._derived_vel[i])
                self._fb[i] = fb
                self._reply_time[i] = stamp[i]
                if fb.error == MotorErrorCode.normal:
                    self._stamp[i] = stamp[i]
                else:
                    self.running = False  # fail fast as DMChainCanInterface does; MotorChainRobot stops on it
                self._seen[i] = count[i]
            fbs = list(self._fb)
            stamps = self._stamp.copy()
            derived = self._derived_vel.copy()
        infos = []
        for i, fb in enumerate(fbs):
            d = self.motor_direction[i]
            vel = derived[i] if self._derive_velocity else fb.velocity
            infos.append(
                MotorInfo(
                    id=self.motor_list[i][0],
                    error_code=hex(fb.error),
                    target_torque=float(torques[i]) if torques is not None else 0.0,
                    vel=float(vel * d),
                    eff=float(fb.torque * d),
                    pos=float((fb.position - self.motor_offset[i]) * d),
                    temp_mos=fb.temperature_mos,
                    temp_rotor=fb.temperature_rotor,
                    timestamp=float(stamps[i]),
                )
            )
        return infos

    @property
    def comm_freq(self) -> float:
        """Slowest motor's mean reply rate since opening, Hz."""
        _, _, count = self._bus.read()
        elapsed = time.time() - self._opened_at
        return float(count.min()) / elapsed if elapsed > 0 and self._n else 0.0

    def stats(self) -> Dict[str, int]:
        """The master's loop counters: ticks, overruns, max lateness, WKC errors, watchdog trips."""
        return self._bus.stats()

    def close(self) -> None:
        """Release every motor, disable it, and stop the master. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.running = False
        self._finalizer()


# ---- one master for several chains --------------------------------------------------------------


class _SharedMaster:
    """One cyclic master carrying several chains' motors, each latching its own rows.

    The master's watchdog trips only once no chain feeds it, so a chain that stops calling
    ``set_frames`` for ``watchdog_ms`` is put back on its release frames by whichever chain still does.
    """

    def __init__(self, bus: Any, releases: Sequence[np.ndarray], watchdog_ms: int) -> None:
        self.bus = bus
        self._frames = np.concatenate(releases)
        self._timeout_s = watchdog_ms / 1000
        self._lock = threading.Lock()
        ends = np.cumsum([len(r) for r in releases])
        self.slices = [_MasterSlice(self, slice(end - len(r), end), r) for r, end in zip(releases, ends, strict=True)]

    def latch(self, owner: "_MasterSlice", frames: np.ndarray) -> None:
        with self._lock:
            now = time.monotonic()
            self._frames[owner.rows] = frames
            owner.last_set = now
            for s in self.slices:
                # A closed chain keeps the disable frames it left latched.
                if not s.closed and now - s.last_set > self._timeout_s:
                    self._frames[s.rows] = s.release
            self.bus.set_frames(self._frames)

    def detach(self, owner: "_MasterSlice") -> None:
        with self._lock:
            owner.closed = True
            if not all(s.closed for s in self.slices):
                return
        self.bus.close()


class _MasterSlice:
    """One chain's rows of a :class:`_SharedMaster`, behind the calls a chain makes on its own bus."""

    def __init__(self, master: _SharedMaster, rows: slice, release: np.ndarray) -> None:
        self.master, self.rows, self.release = master, rows, release
        self.n = len(release)
        self.last_set = 0.0
        self.closed = False

    def set_frames(self, frames: np.ndarray) -> None:
        self.master.latch(self, frames)

    def read(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rx, stamp, count = self.master.bus.read()
        return rx[self.rows], stamp[self.rows], count[self.rows]

    def stats(self) -> Dict[str, int]:
        return self.master.bus.stats()

    def close(self) -> None:
        self.master.detach(self)


def open_shared_master(
    interface: str,
    chains: Sequence[Tuple[int, int, Sequence[Tuple[int, str]]]],
    watchdog_ms: int = 100,
    release_kd: float = DEFAULT_RELEASE_KD,
) -> List[_MasterSlice]:
    """One cyclic master on ``interface`` for several chains, each given as ``(slave, channel, motor_list)``.

    The master allows one bus per process, so chains on one NIC -- the two CAN channels of a board, an
    arm on each -- must share it. Returns one bus per chain, in order, for ``EtherCATMotorChain(bus=...)``;
    every motor is on its release frame until its chain starts, and the master closes once every chain has.
    """
    releases = [_release_frames(motors, release_kd) for _, _, motors in chains]
    slots = [(int(slave), int(channel), int(can_id)) for slave, channel, motors in chains for can_id, _ in motors]
    bus = _bus_class()(interface, slots, release=np.concatenate(releases), watchdog_ms=watchdog_ms)
    return _SharedMaster(bus, releases, watchdog_ms).slices


# ---- a read-only scan -----------------------------------------------------------------------


@dataclass
class ScanRow:
    can_id: int
    replies: int
    rate_hz: float
    state: str
    position: float
    temp_mos: float
    temp_rotor: float


def scan(
    interface: str,
    motor_list: Sequence[Tuple[int, str]],
    slave: int = 0,
    channel: int = 1,
    seconds: float = 2.0,
) -> Tuple[List[ScanRow], Dict[str, int]]:
    """Listen to the motors for ``seconds`` without enabling any, and report each one.

    Every motor is sent the release frame (zero torque, zero stiffness) the whole time, so nothing
    is energised. A motor with 0 replies is not on the bus, or does not reply on MASTER_ID == ID.
    """
    motor_list = [(int(i), str(t)) for i, t in motor_list]
    consts = [MotorType.get_motor_constants(t) for _, t in motor_list]
    release = np.array([np.frombuffer(pack_mit(0, 0, 0, 0, 0, c), np.uint8) for c in consts])
    bus = _bus_class()(interface, [(slave, channel, i) for i, _ in motor_list], release=release)
    try:
        t0 = time.time()
        while time.time() - t0 < seconds:
            bus.set_frames(release)
            time.sleep(0.02)
        rx, _, count = bus.read()
        elapsed = time.time() - t0
        stats = bus.stats()
    finally:
        bus.close()
    rows = []
    for i, (can_id, _) in enumerate(motor_list):
        fb = unpack_mit_feedback(bytes(rx[i]), consts[i])
        heard = count[i] > 0
        rows.append(
            ScanRow(
                can_id=can_id,
                replies=int(count[i]),
                rate_hz=float(count[i]) / elapsed,
                state=MotorErrorCode.get_error_message(fb.error) if heard else "silent",
                position=fb.position if heard else float("nan"),
                temp_mos=fb.temperature_mos if heard else float("nan"),
                temp_rotor=fb.temperature_rotor if heard else float("nan"),
            )
        )
    return rows, stats


# ---- python-can over the board --------------------------------------------------------------


def parse_ethercat_channel(channel: str) -> Tuple[str, int, int]:
    """``ecat:<nic>`` or ``ecat:<nic>:<slave>:<can channel>`` -> ``(nic, slave, can_channel)``."""
    parts = channel[len(ETHERCAT_CHANNEL_PREFIX) :].split(":")
    if len(parts) == 1 and parts[0]:
        return parts[0], 0, 1
    if len(parts) == 3 and parts[0]:
        return parts[0], int(parts[1]), int(parts[2])
    raise ValueError(f"EtherCAT channel {channel!r} must be ecat:<nic> or ecat:<nic>:<slave>:<can channel>")


DM_REGISTER_ID = 0x7FF
"""CAN id of DM register and management requests. The reply comes back on the motor's own id."""

LISTEN_SLOT = 2
"""Channel slot used to prime the board's reply filter. Tool traffic goes out one frame at a time
from slot 0, so this one keeps the filter it was primed with."""

PRIME_TIMEOUT_S = 0.02


class _RawHub:
    """One raw-mode master per NIC, shared by every python-can bus open on it.

    The master allows one bus per process, but CAN tools open several on one channel at once (a
    register interface beside a motor interface) and expect each to see every frame, as SocketCAN
    gives them. The hub reads the master's receive queue and hands every frame to every subscriber.

    It also makes register replies receivable. The board drops a reply unless one of the channel's
    slots last sent a frame with the reply's id, and a register request goes to ``DM_REGISTER_ID``
    while its reply carries the motor's id. So before each one, :meth:`listen_for` sends a remote
    frame with the motor's id from ``LISTEN_SLOT``. A DM motor answers a remote frame with its state
    (it changes nothing); that answer is swallowed here, so no tool mistakes it for the reply it is
    waiting for.
    """

    _hubs: Dict[str, "_RawHub"] = {}
    _hubs_lock = threading.Lock()

    def __init__(self, nic: str) -> None:
        self.nic = nic
        self.bus = _bus_class()(nic)
        self.subscribers: List["queue.Queue[can.Message]"] = []
        self._prime_lock = threading.Lock()
        self._swallow: Optional[Tuple[int, int, int]] = None  # (slave, channel, id) of a prime's answer
        self._swallowed = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, name=f"ecat-can-{nic}", daemon=True)
        self._thread.start()

    @classmethod
    def acquire(cls, nic: str) -> Tuple["_RawHub", "queue.Queue[can.Message]"]:
        with cls._hubs_lock:
            hub = cls._hubs.get(nic)
            if hub is None:
                hub = cls._hubs[nic] = _RawHub(nic)
            q: "queue.Queue[can.Message]" = queue.Queue()
            hub.subscribers.append(q)
            return hub, q

    def release(self, q: "queue.Queue[can.Message]") -> None:
        with self._hubs_lock:
            self.subscribers.remove(q)
            if self.subscribers:
                return
            del self._hubs[self.nic]
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.bus.close()

    def listen_for(self, slave: int, channel: int, can_id: int) -> None:
        """Prime ``LISTEN_SLOT`` so the board accepts a reply carrying ``can_id``; see the class doc.

        Re-primed for every request rather than cached: anything else that goes out from that slot
        moves its filter, and nothing here can see that happen.
        """
        with self._prime_lock:
            self._swallowed.clear()
            self._swallow = (slave, channel, can_id)
            self.bus.send_raw(slave, channel, can_id, bytes(8), rtr=True, slot=LISTEN_SLOT)
            self._swallowed.wait(PRIME_TIMEOUT_S)
            self._swallow = None

    def _pump(self) -> None:
        while not self._stop.is_set():
            frame = self.bus.recv_raw(timeout=0.05)
            if frame is None:
                continue
            if self._swallow == (frame.slave, frame.channel, frame.can_id):
                self._swallow = None
                self._swallowed.set()
                continue
            msg = can.Message(
                arbitration_id=frame.can_id,
                data=frame.data,
                is_extended_id=False,
                timestamp=frame.timestamp,
                channel=f"{self.nic}:{frame.slave}:{frame.channel}",
            )
            with self._hubs_lock:
                subscribers = list(self.subscribers)
            for q in subscribers:
                q.put(msg)


class EtherCATCanBus(can.BusABC):
    """A python-can bus on one CAN channel of an EtherCAT-to-CAN board.

    Standard-id data frames only, which is all the board carries. Frames received on the board's
    other channel are not delivered.
    """

    def __init__(self, channel: str, **kwargs: Any) -> None:
        self._nic, self._slave, self._can = parse_ethercat_channel(channel)
        self._hub, self._queue = _RawHub.acquire(self._nic)
        self._tag = f"{self._nic}:{self._slave}:{self._can}"
        self.channel_info = f"EtherCAT-to-CAN on {self._nic}, board {self._slave}, CAN channel {self._can}"
        super().__init__(channel=channel, **kwargs)

    def send(self, msg: can.Message, timeout: Optional[float] = None) -> None:
        if msg.is_extended_id or msg.is_remote_frame:
            raise can.CanError("the EtherCAT-to-CAN board carries standard-id data frames only")
        try:
            if msg.arbitration_id == DM_REGISTER_ID and len(msg.data) >= 2:
                # Answered on the motor's own id -- equal to its MASTER_ID, as this board requires.
                self._hub.listen_for(self._slave, self._can, msg.data[0] | msg.data[1] << 8)
            self._hub.bus.send_raw(self._slave, self._can, msg.arbitration_id, bytes(msg.data))
        except (ValueError, BufferError) as exc:
            raise can.CanError(str(exc)) from exc

    def _recv_internal(self, timeout: Optional[float]) -> Tuple[Optional[can.Message], bool]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                msg = self._queue.get(timeout=wait)
            except queue.Empty:
                return None, False
            if msg.channel == self._tag:
                return msg, False

    def shutdown(self) -> None:
        if getattr(self, "_hub", None) is not None:
            self._hub.release(self._queue)
            self._hub = None
        super().shutdown()


if __name__ == "__main__":
    import tyro

    from i2rt.robots.utils import ArmName, ArmType, _load_arm_config

    def main(
        interface: str,
        arm: ArmName = "yam_7dof_right",
        slave: int = 0,
        channel: int = 1,
        seconds: float = 2.0,
        extra_ids: Tuple[int, ...] = (),
    ) -> None:
        """Scan an arm's motors through the board without enabling any.

        Args:
            interface: NIC the board is wired to.
            arm: arm whose config supplies the motor list.
            slave: the board's position on the EtherCAT chain.
            channel: CAN channel on the board.
            seconds: how long to listen.
            extra_ids: CAN ids to probe as well, e.g. one known to be absent, which must show 0.
        """
        motors = list(_load_arm_config(ArmType.from_string_name(arm)).motor_list)
        motors += [(i, MotorType.DM4310) for i in extra_ids]
        rows, stats = scan(interface, motors, slave, channel, seconds)
        print(f"{'id':>4} {'replies':>8} {'rate Hz':>8} {'state':<24} {'pos rad':>8} {'MOS C':>6} {'rotor C':>7}")
        for r in rows:
            print(
                f"{r.can_id:>4} {r.replies:>8} {r.rate_hz:>8.1f} {r.state:<24} {r.position:>8.3f} "
                f"{r.temp_mos:>6.0f} {r.temp_rotor:>7.0f}"
            )
        print(stats)

    tyro.cli(main)
