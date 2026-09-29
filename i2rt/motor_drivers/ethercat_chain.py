"""A :class:`~i2rt.motor_drivers.dm_driver.MotorChain` that drives motors over ROS 2 topics.

The bridge is a ROS 2 node owning a 1 kHz EtherCAT loop. It speaks CAN to the motors and exposes
them as ``Float32MultiArray`` topics; this class publishes commands and subscribes to state.

Three rates are involved:

* **1 kHz** EtherCAT tick.
* **428.5 Hz** per-motor CAN refresh. The firmware polls three motors per channel per cycle and
  rotates (``ring_index = (ring_index + 3) % motor_count``), so seven motors on one channel each
  see 3/7 of 1 kHz, with a 3 ms worst-case gap. Three or fewer on a channel get the full rate.
* **500 Hz** DDS publish, carrying all motors in one message. This is the rate state arrives at.

``rclpy`` is imported lazily in :meth:`EtherCATMotorChain.__init__`, so this module imports without
ROS installed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from i2rt.motor_drivers.dm_driver import MotorChain
from i2rt.motor_drivers.utils import MotorInfo

# Healthy, in the hex-string form DMChainCanInterface puts in MotorInfo.error_code.
ERROR_CODE_OK = "0x1"

COMMS_FAULT_BITS = 0x07
"""Bits of ``state_working`` meaning a motor is not being heard from: ``0x01`` DLC 0, ``0x02``
answering id not the configured one, ``0x04`` ~100 ms with no fresh frame."""

BRIDGE_LATCH_BIT = 0x08
"""Bit of ``state_working`` meaning the bridge has latched this motor into its own safe stop and
is holding it at ``kp=0``, ignoring published commands. Set by the bridge's stall and comms-fault
watchdogs. Per-motor and one-way: clearing it requires restarting the bridge. The motor keeps
answering with fresh position while latched."""

DRIVE_FAULT_BITS = COMMS_FAULT_BITS | BRIDGE_LATCH_BIT
"""Any bit meaning this joint is not being driven as commanded, whatever the reason."""


def _error_code(working: int) -> str:
    """The bridge's health byte as the hex string ``MotorInfo.error_code`` carries, or ``"0x1"``
    when healthy.

    ====== ==========================================================
    0x01   the CAN reply had DLC 0
    0x02   the answering CAN id is not the configured one
    0x04   over 100 consecutive ticks (~100 ms) with no fresh frame
    0x08   the bridge has latched this motor into safe-stop
    ====== ==========================================================
    """
    return ERROR_CODE_OK if working == 0 else hex(working)


@dataclass
class BridgeTopics:
    """Everything about the bridge's wire format, in one place.

    Split out because the rest of this file is fixed by the ``MotorChain`` contract and cannot be
    wrong, while these are read off another repository and can be. Confirm against
    ``hw_interface/encos_motor_control_interface/src/EncosBridge.cpp`` before trusting a run.
    """

    cmd_torque: str = "/motors/cmd_torque"
    """Feedforward torque, reaching the motors as ``tau_ff`` in the MIT law
    ``tau = tau_ff + kp (pos - q) + kd (vel - qdot)``.

    **A patched bridge is required.** Stock ``EncosBridge`` has the ``cmd_torque_`` member and
    feeds it into the MIT frame but never subscribes to fill it."""

    cmd_kpkd: str = "/motors/cmd_kpkd"
    """Motor-side gains. This class publishes ``kp`` as zero and ``kd`` as given."""

    state_pos: str = "/motors/state_pos"
    state_vel: str = "/motors/state_vel"
    state_torque: str = "/motors/state_torque"
    state_temperature: str = "/motors/state_temperature"
    state_working: str = "/motors/state_working"

    kpkd_concatenated: bool = True
    """The ``cmd_kpkd`` payload is ``[kp0..kp(N-1), kd0..kd(N-1)]``, length ``2N``, not
    interleaved. The bridge reads ``kp_ = cmd_kpkd_[i]`` and ``kd_ = cmd_kpkd_[i + N]``."""

    kd_max: float = 5.0
    """The firmware silently clamps ``kd`` to ``[0, 5]`` and quantizes it to 9 bits. ``kp`` is
    likewise clamped to 500. Values above these are warned about, not rejected."""

    qos_depth: int = 1
    """Matches the bridge's publishers, which use depth 1."""


VELOCITY_TAU_S = 0.004
"""Seconds. Time constant of the first-order filter on the velocity differentiated from position.

Position is quantised to 16 bits over +/-12.5 rad, so a raw difference at 500 Hz carries about
0.19 rad/s of noise; at this constant about 0.15 rad/s survives, with 36 degrees of phase at
20 Hz."""


@dataclass
class _Latest:
    """The most recent published arrays, and when each motor's reading last changed."""

    pos: np.ndarray
    vel: np.ndarray
    derived_vel: np.ndarray
    last_pos_time: float
    eff: np.ndarray
    temp: np.ndarray
    working: np.ndarray
    stamp: np.ndarray
    arrivals: List[float] = field(default_factory=list)


class EtherCATMotorChain(MotorChain):
    """Presents the EtherCAT-to-CAN bridge as the motor chain ``Arm7OSC`` already knows how to use.

    Construct it in place of :class:`~i2rt.motor_drivers.dm_driver.DMChainCanInterface`::

        chain = EtherCATMotorChain(motor_list=cfg["motor_list"], node_name="arm7_osc")
        arm = Arm7OSC(chain, xml_path, ee_site="grasp_site")

    ``motor_offset`` and ``motor_direction`` are applied here, with the same formulas
    ``DMChainCanInterface`` uses. **The bridge's own config must therefore carry unit directions and
    zero offsets**, or the two layers compound.

    Position is passed through as published: the bridge accumulates the angle itself, and this
    class relies on it being continuous in time.
    """

    def __init__(
        self,
        motor_list: Sequence[Tuple[int, str]],
        motor_offset: Optional[np.ndarray] = None,
        motor_direction: Optional[np.ndarray] = None,
        node_name: str = "i2rt_ethercat_chain",
        topics: Optional[BridgeTopics] = None,
        state_timeout_s: float = 2.0,
        derive_velocity: bool = True,
    ) -> None:
        """
        Args:
            motor_list: ``(can_id, motor_type)`` per motor, in the order the bridge's own
                ``motors:`` list has them. The order is the topic's array order and there is nothing
                in the payload to check it against, so a mismatch here is silent.
            motor_offset: radians, subtracted in the motor frame before the direction flip, exactly
                as ``DMChainCanInterface`` does. Defaults to zeros, which is what the arm uses --
                its zeros live in motor Flash.
            motor_direction: ``+1`` or ``-1`` per joint, from the config's ``directions``. Defaults
                to ones, which is **wrong for this arm** and is only a default so that a caller with
                an already-aligned bridge can opt out deliberately.
            derive_velocity: differentiate the published position instead of trusting the published
                velocity. Default, and it should stay the default. Position carries one range,
                ``+/-12.5`` rad, for every DM type, so it decodes correctly whatever the config
                declares; velocity is encoded against the motor's own ``VMAX`` and decoded against
                the declared type's, and on this arm two joints read exactly 2.0x and 1/3x as a
                result. A damping term and a velocity limit both act on this number, so a wrong
                scale is not merely a reporting defect.
            node_name: ROS 2 node name. One per process.
            topics: wire format; see :class:`BridgeTopics`.
            state_timeout_s: how long ``__init__`` waits for the first state message before giving
                up. Failing loudly at construction is much better than a controller that starts
                against an all-zero pose.
        """
        # Imported here, not at module scope: this module is imported by code paths that run on
        # machines without ROS, including the whole sim-only test suite.
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray, UInt8MultiArray

        self.motor_list: List[Tuple[int, str]] = list(motor_list)
        self.topics = topics or BridgeTopics()
        n = len(self.motor_list)

        # Attributes DMChainCanInterface exposes that callers read directly, and that this class
        # applies for the same reason it does: see the class docstring.
        self.motor_offset = np.zeros(n) if motor_offset is None else np.asarray(motor_offset, dtype=float)
        self.motor_direction = np.ones(n) if motor_direction is None else np.asarray(motor_direction, dtype=float)
        if self.motor_offset.shape != (n,) or self.motor_direction.shape != (n,):
            raise ValueError(f"motor_offset and motor_direction must each hold {n} values")
        if not np.all(np.isin(self.motor_direction, (-1.0, 1.0))):
            # The write path relies on d*d == 1 to leave kd unscaled, as the CAN chain does.
            raise ValueError(f"motor_direction must be +1 or -1, got {self.motor_direction.tolist()}")
        self.running = True

        self._derive_velocity = bool(derive_velocity)
        self._n = n
        self._lock = threading.RLock()
        self._closed = False
        # One log line per motor the bridge latches, not one per 500 Hz tick.
        self._latch_warned = np.zeros(n, dtype=bool)
        self._latest = _Latest(
            pos=np.zeros(n),
            vel=np.zeros(n),
            derived_vel=np.zeros(n),
            last_pos_time=0.0,
            eff=np.zeros(n),
            temp=np.full(n, -1.0),
            working=np.zeros(n, dtype=np.uint8),
            # Zero, so every motor looks infinitely stale until its first message arrives.
            stamp=np.zeros(n),
        )

        if not rclpy.ok():
            # No rclpy signal handler, so SIGINT arrives as a plain KeyboardInterrupt and the
            # context is still valid when `close()` publishes the zero-torque release.
            from rclpy.signals import SignalHandlerOptions

            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._rclpy = rclpy
        self._node: Any = Node(node_name)

        qos = self.topics.qos_depth
        self._pub_torque = self._node.create_publisher(Float32MultiArray, self.topics.cmd_torque, qos)
        self._pub_kpkd = self._node.create_publisher(Float32MultiArray, self.topics.cmd_kpkd, qos)

        # Pre-allocated and written in place; `set_commands` allocates nothing per tick.
        self._msg_torque = Float32MultiArray()
        self._msg_torque.data = [0.0] * n
        self._msg_kpkd = Float32MultiArray()
        self._msg_kpkd.data = [0.0] * (2 * n)

        self._node.create_subscription(Float32MultiArray, self.topics.state_pos, self._on_pos, qos)
        self._node.create_subscription(Float32MultiArray, self.topics.state_vel, self._on_vel, qos)
        self._node.create_subscription(Float32MultiArray, self.topics.state_torque, self._on_eff, qos)
        self._node.create_subscription(Float32MultiArray, self.topics.state_temperature, self._on_temp, qos)
        self._node.create_subscription(UInt8MultiArray, self.topics.state_working, self._on_working, qos)

        # The executor runs in its own thread, so `set_commands` returns the most recent state
        # without spinning it.
        from rclpy.executors import SingleThreadedExecutor

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._spin, name=f"{node_name}_spin", daemon=True)
        self._spin_thread.start()

        deadline = time.time() + state_timeout_s
        while time.time() < deadline:
            with self._lock:
                if np.any(self._latest.stamp > 0.0):
                    break
            time.sleep(0.01)
        else:
            self.close()
            raise RuntimeError(
                f"no message on {self.topics.state_pos} within {state_timeout_s:.1f}s -- "
                "is the bridge running, and are ROS_DOMAIN_ID / RMW settings the same on both sides?"
            )

    # ---- ROS plumbing --------------------------------------------------------

    def _spin(self) -> None:
        """Run the executor until close. A callback that raises is logged and the next cycle
        runs, so one bad message does not stop state delivery."""
        while self.running and not self._closed:
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception as exc:
                if self._closed or not self.running:
                    return
                self._node.get_logger().error(f"subscription callback raised: {exc!r}")
                time.sleep(0.01)

    def _stamp(self, field_name: str, values: np.ndarray) -> None:
        """Store a published array and advance the timestamps of the motors ``state_working``
        reports as healthy.

        A motor with any bit of ``DRIVE_FAULT_BITS`` set stops being stamped, so its age grows and
        the controller's missing-frame watchdog sees it. The published payload carries no header or
        sequence number, so this is the only per-motor freshness signal available; a silence
        shorter than the bridge's own ~100 ms threshold is not visible here.
        """
        now = time.time()
        with self._lock:
            previous = getattr(self._latest, field_name)
            if values.shape != previous.shape:
                self._node.get_logger().warn(
                    f"{field_name}: bridge published {values.size} values for {previous.size} motors"
                )
                return
            if field_name == "pos" and self._latest.last_pos_time > 0.0:
                dt = now - self._latest.last_pos_time
                if 1e-4 < dt < 0.1:
                    raw = (values - previous) / dt
                    alpha = dt / (VELOCITY_TAU_S + dt)
                    self._latest.derived_vel += alpha * (raw - self._latest.derived_vel)
            setattr(self._latest, field_name, values)
            if field_name == "pos":
                self._latest.last_pos_time = now
                latched = (self._latest.working & BRIDGE_LATCH_BIT) != 0
                for i in np.flatnonzero(latched & ~self._latch_warned):
                    self._latch_warned[i] = True
                    self._node.get_logger().error(
                        f"the bridge has LATCHED motor {i + 1} (can id {self.motor_list[i][0]}) "
                        "into its own safe stop and is ignoring commanded torque for it. "
                        "Nothing in the bridge clears this latch -- restart the bridge. Its "
                        "stall watchdog trips on |current| near peak while the joint is not moving."
                    )
                healthy = (self._latest.working & DRIVE_FAULT_BITS) == 0
                self._latest.stamp[healthy] = now
                self._latest.arrivals.append(now)
                if len(self._latest.arrivals) > 200:
                    del self._latest.arrivals[:100]

    # Radians and radians per second, confirmed against the bridge: it converts the config's
    # degree offsets at load time and publishes `(pos - offset) * direction` in the motor's own
    # units. No conversion here.
    def _on_pos(self, msg: Any) -> None:
        self._stamp("pos", np.asarray(msg.data, dtype=float))

    def _on_vel(self, msg: Any) -> None:
        self._stamp("vel", np.asarray(msg.data, dtype=float))

    def _on_eff(self, msg: Any) -> None:
        self._stamp("eff", np.asarray(msg.data, dtype=float))

    def _on_temp(self, msg: Any) -> None:
        self._stamp("temp", np.asarray(msg.data, dtype=float))

    def _on_working(self, msg: Any) -> None:
        self._stamp("working", np.asarray(msg.data, dtype=np.uint8))

    # ---- MotorChain protocol -------------------------------------------------

    def __len__(self) -> int:
        return self._n

    def __repr__(self) -> str:
        return f"EtherCATMotorChain(n_motors={self._n}, cmd={self.topics.cmd_torque})"

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> Optional[List[MotorInfo]]:
        """Publish one command and return the most recent state, without waiting for a reply.

        ``pos``, ``vel`` and ``kp`` are accepted for signature parity and **must be zero or absent**.
        ``Arm7OSC`` never passes them: it computes the whole joint torque itself and wants the motor
        to add only damping, ``tau = tau_ff + kd (0 - qdot)``. A non-zero ``kp`` here would give the
        motor a position target of its own to pull towards, which fights the controller rather than
        assisting it, so passing one is refused rather than silently forwarded.

        ``None`` means zero, not "leave unchanged" -- the same as ``DMChainCanInterface``. This is
        what makes ``set_commands(np.zeros(n))`` mean "go limp", which is how both
        ``Arm7OSC.close()`` and the bringup script's teardown release the arm.
        """
        for name, value in (("pos", pos), ("vel", vel), ("kp", kp)):
            if value is not None and np.any(np.asarray(value) != 0.0):
                raise ValueError(
                    f"{name} must be zero through this transport: the bridge's MIT law would then "
                    "drive the motor towards its own target and fight the OSC torque"
                )

        torques = np.asarray(torques, dtype=float)
        if torques.size < self._n:
            raise ValueError(f"need {self._n} torques, got {torques.size}")
        kd_vec = np.zeros(self._n) if kd is None else np.asarray(kd, dtype=float)[: self._n]

        # Written in place into the pre-allocated messages; see __init__ on why.
        data = self._msg_torque.data
        for i in range(self._n):
            # Model frame to motor frame. kd is left unscaled, which is only correct because the
            # direction is +/-1 -- the same assumption DMChainCanInterface makes.
            data[i] = float(torques[i] * self.motor_direction[i])
        self._pub_torque.publish(self._msg_torque)

        kpkd = self._msg_kpkd.data
        if np.any(kd_vec > self.topics.kd_max):
            self._node.get_logger().warn(
                f"kd up to {kd_vec.max():.2f} exceeds the firmware's silent clamp of "
                f"{self.topics.kd_max}; the motors will apply {self.topics.kd_max}"
            )
        for i in range(self._n):
            kpkd[i] = 0.0  # kp: the host owns the stiffness, the motor only damps
            kpkd[self._n + i] = float(kd_vec[i])
        self._pub_kpkd.publish(self._msg_kpkd)

        if get_state:
            return self.read_states(torques=torques)
        return None

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        """The most recent published state, one :class:`MotorInfo` per motor, in chain order."""
        with self._lock:
            pos = self._latest.pos.copy()
            vel = (self._latest.derived_vel if self._derive_velocity else self._latest.vel).copy()
            eff = self._latest.eff.copy()
            temp = self._latest.temp.copy()
            stamp = self._latest.stamp.copy()
            working = self._latest.working.copy()
        return [
            MotorInfo(
                id=self.motor_list[i][0],
                error_code=_error_code(int(working[i])),
                target_torque=float(torques[i]) if torques is not None else 0.0,
                # Motor frame to model frame: offset off first, then the flip, as on the CAN chain.
                vel=float(vel[i] * self.motor_direction[i]),
                eff=float(eff[i] * self.motor_direction[i]),
                pos=float((pos[i] - self.motor_offset[i]) * self.motor_direction[i]),
                temp_mos=float(temp[i]),
                temp_rotor=float(temp[i]),
                timestamp=float(stamp[i]),
            )
            for i in range(self._n)
        ]

    @property
    def comm_freq(self) -> float:
        """Mean rate of the state stream, Hz, over the recent arrivals.

        The bridge's publish rate, 500 Hz, decimated from its 1 kHz loop. Every motor appears in
        every message, so this is the rate each motor's state arrives at. It is not the rate at
        which a motor's value is genuinely new; the firmware rotates through them at 3/7 of 1 kHz.
        """
        with self._lock:
            arrivals = list(self._latest.arrivals)
        if len(arrivals) < 2:
            return 0.0
        span = arrivals[-1] - arrivals[0]
        return (len(arrivals) - 1) / span if span > 0 else 0.0

    def close(self) -> None:
        """Publish zero torque and zero gains three times, then tear the node down. Idempotent.

        The bridge re-applies the last command it received every 1 ms with no timeout, so exiting
        without this leaves the arm holding the last gains indefinitely.
        """
        if self._closed:
            return
        try:
            for i in range(self._n):
                self._msg_torque.data[i] = 0.0
                self._msg_kpkd.data[i] = 0.0
                self._msg_kpkd.data[self._n + i] = 0.0
            for _ in range(3):
                self._pub_torque.publish(self._msg_torque)
                self._pub_kpkd.publish(self._msg_kpkd)
                time.sleep(0.002)
        except Exception as exc:
            # Never silent. A swallowed release leaves the arm holding whatever it last had, and
            # the operator has no way to know from the console that it happened.
            print(
                f"\n  *** RELEASE FAILED: {type(exc).__name__}: {exc}\n"
                "  *** The arm may still be holding torque. The bridge re-applies the last\n"
                "  *** command indefinitely and has no timeout. Run ~/release_arm.py now, or\n"
                "  *** power the arm down.",
                flush=True,
            )
        self._closed = True
        self.running = False
        # Shut down, join, then destroy: the spin thread must not touch a destroyed context.
        executor = getattr(self, "_executor", None)
        if executor is not None:
            try:
                executor.shutdown()
            except Exception:
                pass
        thread = getattr(self, "_spin_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        node = getattr(self, "_node", None)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
