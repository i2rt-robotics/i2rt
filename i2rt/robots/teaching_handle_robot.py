"""The passive YAM teaching handle as a standalone robot (``--arm no_arm``).

The handle is not a gripper.  It has no motor -- only the passive trigger encoder, which
also reports the two grip buttons.  Bolted to an arm, that encoder rides the arm's motor
chain: ``DMChainCanInterface`` polls it once per control cycle as a "same bus device", and
every consumer reads it back through ``robot.motor_chain``.  On its own there is no motor
chain to ride, and ``DMChainCanInterface`` cannot stand in -- it asserts at least one motor
and brings each one up before returning, so a gripper-only handle used to fail at
``motor_on(0x07)`` for a motor that does not exist.

This module supplies the two missing halves, each shaped like its arm-mounted counterpart
so no consumer has to special-case them:

* :class:`MotorlessCanChain` -- the polling half.  The same ``same_bus_device_driver`` /
  ``get_same_bus_device_states()`` / ``comm_freq`` surface ``DMChainCanInterface`` offers
  for a same-bus device, minus every motor.
* :class:`TeachingHandleRobot` -- a read-only ``Robot`` over that chain.
"""

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.robots.robot import Robot, RobotType
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.utils import RateRecorder

logger = logging.getLogger(__name__)

_POLL_PERIOD_S = 1.0 / 250.0


class MotorlessCanChain:
    """A ``DMChainCanInterface`` stand-in for a CAN bus that carries no motors.

    Owns the bus, polls its one passive device in a background thread, and hands out the
    latest snapshot under the same lock-and-cache contract ``DMChainCanInterface`` uses --
    so ``robot.motor_chain`` reads identically whether or not an arm is present.  ``len()``
    is 0: there is nothing here to command.

    A read failure stops the loop and clears ``running`` (``PassiveEncoderReader`` already
    retries 15 times internally, so reaching here means the device is really gone).
    :class:`TeachingHandleRobot` turns that into a raised error rather than let consumers
    keep serving a frozen snapshot.

    It runs none of the startup register checks every real motor chain opts into: those read
    ``Gr``, ``CTRL_MODE`` and the ``PMAX``/``VMAX``/``TMAX`` scaling registers off motors, and
    there are none here.  The encoder is screened instead by ``PassiveJointEncoder``'s own
    firmware validation, which ``PassiveEncoderReader`` runs when it is constructed.
    """

    def __init__(
        self,
        channel: str,
        get_same_bus_device_driver: Callable[[CanInterface], Any],
        name: str = "motorless_chain",
    ) -> None:
        self.channel = channel
        self.name = name
        self.can_interface = CanInterface(channel=channel, name=name, use_buffered_reader=False)
        self.same_bus_device_lock = threading.Lock()
        try:
            self.same_bus_device_driver = get_same_bus_device_driver(self.can_interface)

            # Encoder validation opens its own socket on this channel, and SocketCAN copies every
            # frame to every socket, so ours is holding replies it already consumed.  Same drain
            # DMChainCanInterface does between encoder validation and motor bring-up.
            drained = self.can_interface._drain_bus(timeout_s=0.2)
            if drained:
                logger.info(f"Drained {drained} stale frames before polling {name}")

            # Read once here so a dead device fails construction instead of leaving the robot
            # reporting `None` until someone notices.
            self.same_bus_device_states = self.same_bus_device_driver.read_states()
        except BaseException:  # incl. Ctrl-C during the encoder's retries: don't leak the socket
            self.can_interface.close()
            raise
        self.running = True

        self._rate_recorder = RateRecorder(name=name)
        self._thread = threading.Thread(target=self._poll_loop, name=name, daemon=True)
        self._thread.start()

    def __len__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"MotorlessCanChain(channel={self.channel}, name={self.name})"

    @property
    def comm_freq(self) -> float:
        return self._rate_recorder.last_rate

    def _poll_loop(self) -> None:
        with self._rate_recorder:
            while self.running:
                try:
                    states = self.same_bus_device_driver.read_states()
                except Exception:
                    self.running = False
                    logger.exception(f"{self}: passive device read failed, stopping the poll loop")
                    return
                with self.same_bus_device_lock:
                    self.same_bus_device_states = states
                self._rate_recorder.track()
                time.sleep(_POLL_PERIOD_S)

    def get_same_bus_device_states(self) -> Any:
        with self.same_bus_device_lock:
            return self.same_bus_device_states

    def close(self) -> None:
        self.running = False
        self._thread.join(timeout=1.0)
        self.can_interface.close()


class TeachingHandleRobot(Robot):
    """A teaching handle on its own CAN bus, with no arm under it.

    A sensor, not an actuator.  ``num_dofs()`` is 0 -- the same number of commandable joints
    the handle contributes when bolted to an arm -- so only an empty command is accepted, and
    there are no gains, joint limits or gravity compensation to configure.  The trigger
    squeeze reaches consumers through the two channels an arm-mounted handle already uses:
    ``get_observations()["gripper_pos"]``, and ``motor_chain.get_same_bus_device_states()``
    for the buttons and the raw encoder frame.
    """

    def __init__(
        self,
        motor_chain: MotorlessCanChain,
        xml_path: str,
        gripper_type: GripperType,
    ) -> None:
        # Named `motor_chain` even though it drives no motors: every teaching-handle consumer
        # (the viser and MuJoCo interfaces, control_with_mujoco) reaches the encoder through
        # `robot.motor_chain`, which is where an arm-mounted handle puts the real chain.
        self.motor_chain = motor_chain
        self.xml_path = os.path.expanduser(xml_path)
        self._gripper_type = gripper_type

    def __repr__(self) -> str:
        return f"TeachingHandleRobot(gripper_type={self._gripper_type}, motor_chain={self.motor_chain})"

    def _read_encoder(self) -> Any:
        """Latest encoder snapshot, or None before the first poll completes."""
        if not self.motor_chain.running:
            raise RuntimeError(f"{self}: the encoder poll loop stopped; the teaching handle is no longer readable")
        states = self.motor_chain.get_same_bus_device_states()
        return states[0] if states else None

    def _handle_gripper_pos(self) -> Optional[float]:
        """Trigger squeeze as a normalized 0..1 gripper position.

        Reported in the arm convention (0 = closed, 1 = open) via
        ``PassiveEncoderInfo.gripper_pos``, matching ``MotorChainRobot._handle_gripper_pos``
        so a consumer cannot tell an arm-mounted handle from this one.
        """
        state = self._read_encoder()
        if state is None:
            return None
        return float(np.clip(state.gripper_pos, 0.0, 1.0))

    # ----------------- Robot protocol ----------------- #

    def num_dofs(self) -> int:
        """0: the handle is passive, so there is nothing to command."""
        return 0

    def get_joint_pos(self) -> np.ndarray:
        # Empty, but not free: this is the per-frame call that surfaces a dead encoder.
        self._read_encoder()
        return np.zeros(0)

    def get_joint_state(self) -> Dict[str, np.ndarray]:
        return {"pos": self.get_joint_pos(), "vel": np.zeros(0)}

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """No-op for an empty command; raises on anything else.

        The empty case stays silent -- control interfaces command every tick while in IK or
        joint mode, and with zero DOFs they are commanding an empty array.  A non-empty
        command is a caller aimed at the wrong robot (say, a teleop script on the wrong bus),
        which dropping it would hide.
        """
        if len(joint_pos):
            raise ValueError(f"{self} has 0 DOFs; got a {len(joint_pos)}-joint command")

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Same contract as :meth:`command_joint_pos`."""
        self.command_joint_pos(joint_state["pos"])

    def move_joints(self, target_joint_positions: np.ndarray, time_interval_s: float = 2.0) -> None:
        raise NotImplementedError(f"{self} is passive: there are no joints to move")

    def start_mcap_recording(self) -> None:
        raise NotImplementedError(f"{self} is passive: there is no motor feedback to record")

    def get_robot_type(self) -> RobotType:
        return RobotType.TEACHING_HANDLE

    def get_observations(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {
            "joint_pos": np.zeros(0),
            "joint_vel": np.zeros(0),
            "joint_eff": np.zeros(0),
        }
        handle_pos = self._handle_gripper_pos()
        if handle_pos is not None:
            obs["gripper_pos"] = np.array([handle_pos])
        return obs

    def get_robot_info(self) -> Dict[str, Any]:
        # No kp/kd keys: there are no motors, and control interfaces key their PD-gains
        # panel off their presence.
        return {
            "arm_type": ArmType.NO_ARM,
            "gripper_type": self._gripper_type,
            "joint_limits": None,
            "gripper_limits": None,
            "gripper_index": None,
        }

    def close(self) -> None:
        self.motor_chain.close()
