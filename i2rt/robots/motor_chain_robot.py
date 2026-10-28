import copy
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    MotorChain,
    MotorInfo,
    PassiveEncoderInfo,
)
from i2rt.robots.four_bar import FourBarLinkage
from i2rt.robots.robot import Robot
from i2rt.robots.utils import (
    ArmName,
    ArmType,
    GripperForceLimiter,
    GripperType,
    JointMapper,
    detect_gripper_limits,
)
from i2rt.utils.mujoco_utils import MuJoCoKDL
from i2rt.utils.recording import RobotMcapRecorder

# How far past its measured travel a linkage joint's motor may read before it is refused. A reading
# beyond the hard stops means a wrong zero or a wrong `directions` sign, and the map stops meaning
# anything there: NaN a few degrees below joint6's lower stop, folding back past the dead point ~6 deg
# above its upper one. Kept clear of that dead point, where the ratio already falls toward zero.
TRANSMISSION_MOTOR_MARGIN = np.radians(3.0)


@dataclass
class JointStates:
    names: List[str]
    pos: np.ndarray
    vel: np.ndarray
    eff: np.ndarray
    temp_mos: np.ndarray  # MOS temperature (float): Motor MOS temperature.
    temp_rotor: np.ndarray  # ROTOR temperature (float): Motor ROTOR temperature.
    timestamp: float

    def asdict(self) -> Dict[str, Any]:
        return {
            "names": self.names,
            "pos": self.pos.flatten().tolist(),
            "vel": self.vel.flatten().tolist(),
            "eff": self.eff.flatten().tolist(),
        }


@dataclass
class JointCommands:
    torques: np.ndarray

    pos: np.ndarray
    vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    indices: Optional[List[int]] = None

    @classmethod
    def init_all_zero(cls, n_joints: int) -> "JointCommands":
        return cls(
            torques=np.zeros(n_joints),
            pos=np.zeros(n_joints),
            vel=np.zeros(n_joints),
            kp=np.zeros(n_joints),
            kd=np.zeros(n_joints),
        )


class MotorChainRobot(Robot):
    """A generic Robot protocol."""

    # Joints whose motor drives them through a linkage rather than directly, by joint index. Direct
    # drive is the norm: the arm config declares any linkage (the 7-DOF arms' joint6) and ``__init__``
    # rebinds this from its ``transmissions`` argument. Declared on the class so the conversion is a
    # no-op by default and never an AttributeError; never mutated.
    _transmissions: Dict[int, FourBarLinkage] = {}

    def __init__(
        self,
        motor_chain: MotorChain,
        xml_path: Optional[str] = None,
        use_gravity_comp: bool = True,
        gravity: Optional[np.ndarray] = None,
        gravity_comp_factor: Optional[np.ndarray] = None,
        gripper_index: Optional[int] = None,  # Zero starting index: if you have a 6 dof arm and last one is gripper: 6
        kp: Union[float, List[float]] = 10.0,
        kd: Union[float, List[float]] = 1.0,
        grav_comp_kd: Optional[np.ndarray] = None,  # per-joint MIT-mode kd, active only in grav-comp idle
        coulomb_friction: Optional[
            np.ndarray
        ] = None,  # per-joint Coulomb friction (Nm); applied as coulomb_friction * sign(q_dot)
        use_coulomb_friction: bool = False,  # if True, add the Coulomb friction feedforward in the grav-comp loop
        joint_limits: Optional[np.ndarray] = None,  # if provided, override the mujoco xml joint limits
        gripper_limits: Optional[np.ndarray] = None,  # [closed, open]
        limit_gripper_force: float = -1,  # whether to limit the gripper effort when it is blocked. -1 means no limit.
        clip_motor_torque: float = np.inf,  # clip the offset motor torque, real motor torque can still still be larger than this setting depending on the motor onboard PID loop
        gripper_type: GripperType = GripperType.LINEAR_4310,
        arm_type: ArmType = ArmType.YAM,
        transmissions: Optional[Dict[int, FourBarLinkage]] = None,  # {joint index: linkage}, from the arm config
        temp_record_flag: bool = False,  # whether record the motor's temperature
        enable_gripper_calibration: bool = False,  # whether to auto-detect gripper limits
        zero_gravity_mode: bool = True,
        # below are calibration parameters
        test_torque: float = 0.5,  # test torque for gripper detection (Nm)
        test_duration: float = 2.0,  # max test duration for each direction (s)
        position_threshold: float = 0.01,  # minimum position change to consider motor still moving (rad)
        check_interval: float = 0.05,  # time interval between checks (s)
        pinned_cpu: int | None = None,
        joint_state_saver_factory: Optional[Callable[[], Any]] = None,
        set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
        enable_auto_recovery: Optional[bool] = None,  # None: inherit motor_chain's setting; True/False: override it
    ) -> None:
        # Set up CPU pinning and real-time scheduling if requested
        if pinned_cpu is not None and set_realtime_and_pin_callback is not None:
            set_realtime_and_pin_callback(pinned_cpu)

        self._joint_state_saver_factory = joint_state_saver_factory
        self._set_realtime_and_pin_callback = set_realtime_and_pin_callback
        self._arm_type = arm_type
        self._gripper_type = gripper_type
        self.temp_record_flag = temp_record_flag

        # make sure kp, kd are float number not int
        self._kp = (
            np.array(
                [
                    kp,
                ]
                * len(motor_chain)
            )
            if isinstance(kp, float)
            else np.array(kp)
        )
        self._kd = (
            np.array(
                [
                    kd,
                ]
                * len(motor_chain)
            )
            if isinstance(kd, float)
            else np.array(kd)
        )
        self._grav_comp_kd = (
            np.array(grav_comp_kd, dtype=float) if grav_comp_kd is not None else np.zeros(len(motor_chain))
        )
        assert len(self._grav_comp_kd) == len(motor_chain), (
            f"grav_comp_kd length {len(self._grav_comp_kd)} != motor_chain length {len(motor_chain)}"
        )
        self.use_coulomb_friction = use_coulomb_friction
        # What ``enter_gravity_comp_idle`` restores to after ``enter_zero_torque_idle`` turned it off.
        self._coulomb_friction_configured = use_coulomb_friction
        self._coulomb_friction = (
            np.array(coulomb_friction, dtype=float) if coulomb_friction is not None else np.zeros(len(motor_chain))
        )
        assert len(self._coulomb_friction) == len(motor_chain), (
            f"coulomb_friction length {len(self._coulomb_friction)} != motor_chain length {len(motor_chain)}"
        )

        if gripper_index is not None:
            assert gripper_index == len(motor_chain) - 1, (
                "Gripper index should be the last one, but got {gripper_index}"
            )

            # Auto-detect gripper limits if enabled and gripper_limits is None
            print(
                f"initializing motorchain robot, gripper_limits: {gripper_limits}, enable_gripper_calibration: {enable_gripper_calibration}"
            )
            if gripper_limits is None and enable_gripper_calibration:
                logger = logging.getLogger(__name__)
                logger.info("Auto-detecting gripper limits...")
                detected_limits = detect_gripper_limits(
                    motor_chain=motor_chain,
                    gripper_index=gripper_index,
                    # The control loop is not running yet, so the arm hangs on this PD hold alone.
                    # The configured gains are what keep the droop at gravity_torque / kp.
                    hold_kp=self._kp,
                    hold_kd=self._kd,
                    test_torque=test_torque,
                    max_duration=test_duration,
                    position_threshold=position_threshold,
                    check_interval=check_interval,
                )
                gripper_limits = np.array(detected_limits)
                logger.info(f"Gripper limits auto-detected: {gripper_limits}")
            elif gripper_limits is None:
                raise ValueError(
                    f"{self}: Gripper limits are required if gripper index is provided and auto-calibration is disabled."
                )
            else:
                # Use the provided gripper_limits
                logger = logging.getLogger(__name__)
                logger.info(f"Using provided gripper limits: {gripper_limits}")

        self._last_gripper_command_qpos = 1  # initialize as fully open
        assert clip_motor_torque >= 0.0
        self._clip_motor_torque = clip_motor_torque
        self.motor_chain = motor_chain
        # None means inherit whatever the chain was constructed with; an explicit value overrides it.
        # The chain reads this flag live each control-loop iteration, so a late set is safe.
        if enable_auto_recovery is not None:
            self.motor_chain.enable_auto_recovery = enable_auto_recovery
        self.use_gravity_comp = use_gravity_comp
        # What ``enter_gravity_comp_idle`` restores to. A robot built without an XML path has no
        # KDL to compute gravity torques with, so "back to gravity comp" must stay off for it.
        self._gravity_comp_available = use_gravity_comp
        self.gravity_comp_factor = (
            gravity_comp_factor if gravity_comp_factor is not None else np.ones(len(motor_chain))
        )

        if transmissions:
            self._transmissions = transmissions

        # variables for gripper effort limiting
        self._gripper_index = gripper_index
        self.remapper = JointMapper({}, len(motor_chain))  # so it works without gripper
        self._gripper_limits = gripper_limits
        self._gripper_force_limiter: Optional[GripperForceLimiter] = None
        self._limit_gripper_force: float = -1.0

        if self._gripper_index is not None:
            self._gripper_force_limiter = GripperForceLimiter(
                max_force=limit_gripper_force, gripper_type=gripper_type, arm_type=arm_type, kp=self._kp[gripper_index]
            )  # force in newton
            self._limit_gripper_force = limit_gripper_force

            self.remapper = JointMapper(
                index_range_map={gripper_index: gripper_limits},
                total_dofs=len(motor_chain),
            )

        self._joint_limits: Optional[np.ndarray] = None
        if xml_path is not None:
            self.xml_path = os.path.expanduser(xml_path)
            self.kdl = MuJoCoKDL(self.xml_path)
            if gravity is not None:
                self.kdl.set_gravity(gravity)
            # Load the joint limits from the xml file
            self._joint_limits = self.kdl.joint_limits
        else:
            assert use_gravity_comp is False, "Gravity compensation requires a valid XML path."

        # override the xml joint limits with the provided joint_limits
        if joint_limits is not None:
            joint_limits = np.array(joint_limits)
            assert np.all(joint_limits[:, 0] < joint_limits[:, 1]), (
                "Lower joint limits must be smaller than upper limits"
            )
            self._joint_limits = joint_limits
        # Initialize joint state saver if factory is provided
        if self._joint_state_saver_factory is not None:
            self._joint_state_saver = self._joint_state_saver_factory()
        else:
            self._joint_state_saver = None

        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._mcap_lock = threading.Lock()
        self._mcap_recorder: Optional[RobotMcapRecorder] = None
        self._joint_state: Optional[JointStates] = None
        while self._joint_state is None:
            # wait to recive joint data
            time.sleep(0.05)
            self._joint_state = self._motor_state_to_joint_state(self.motor_chain.read_states())
        self._commands = JointCommands.init_all_zero(len(motor_chain))
        if zero_gravity_mode:
            self._commands.kd = self._grav_comp_kd.copy()
        else:
            # Hold the startup pose from the loop's very first iteration. init_all_zero leaves kp at
            # zero, so the window this closes was not a lunge toward zero -- the MIT controller
            # ignores a position target at kp=0 -- but the arm was briefly unheld, carried only by
            # gravity comp, until the post-start command landed a few milliseconds later. The state is
            # in joint space, so it is converted back to motor angles like any other command.
            self._commands.pos = self._joint_to_motor_pos(
                self.remapper.to_robot_joint_pos_space(self._joint_state.pos)
            )
            self._commands.kp = self._kp.copy()
            self._commands.kd = self._kd.copy()
        # For SWE-454, check if the current qpos is in the joint limits
        self._check_current_qpos_in_joint_limits()

        self._last_motor_torques: Optional[np.ndarray] = None
        self._stop_event = threading.Event()  # Add a stop event
        self._server_thread = threading.Thread(target=self.start_server, name="robot_server")
        self._server_thread.start()

    def __repr__(self) -> str:
        return f"MotorChainRobot(arm_type={self._arm_type}, gripper_type={self._gripper_type}, motor_chain={self.motor_chain})"

    def _check_current_qpos_in_joint_limits(self, buffer_rad: float = 0.1) -> None:
        """Check if the self._joint_state is in the joint limits.
        If violated, raise an error.
        """
        if self._joint_state is None or self._joint_limits is None:
            raise RuntimeError(
                f"{self}: Joint limits:{self._joint_limits} or joint state:{self._joint_state} are not set."
            )

        current_pos = self._joint_state.pos

        # Check arm joints (exclude gripper if present)
        if self._gripper_index is not None:
            # Only check arm joints, not the gripper
            arm_pos = current_pos[: self._gripper_index]
            arm_limits = self._joint_limits
        else:
            # Check all joints
            arm_pos = current_pos
            arm_limits = self._joint_limits

        # Check if any joint is outside its limits
        lower_limits = arm_limits[:, 0] - buffer_rad
        upper_limits = arm_limits[:, 1] + buffer_rad

        # Find joints that violate lower limits
        lower_violations = arm_pos < lower_limits
        # Find joints that violate upper limits
        upper_violations = arm_pos > upper_limits

        if np.any(lower_violations) or np.any(upper_violations):
            violation_details = []

            for i, (pos, lower, upper) in enumerate(zip(arm_pos, lower_limits, upper_limits, strict=False)):
                if pos < lower:
                    violation_details.append(f"Joint {i}: {pos:.4f} < {lower:.4f} (lower limit)")
                elif pos > upper:
                    violation_details.append(f"Joint {i}: {pos:.4f} > {upper:.4f} (upper limit)")

            violation_msg = "; ".join(violation_details)
            # turn off the main motor control thread as well.
            self.motor_chain.running = False
            raise RuntimeError(
                f"{self}: Joint limit violation detected: {violation_msg}, the root reason should be zero position offset. possible solution: 1. move the arm to zero position and power cycle the robot. 2. Recalibrate the motor zero position."
            )

    def get_robot_info(self) -> Dict[str, Any]:
        """Get the robot information, such as kp, kd, joint limits, gripper limits, etc."""
        info: Dict[str, Any] = {
            "arm_type": self._arm_type,
            "gripper_type": self._gripper_type,
            "kp": self._kp,
            "kd": self._kd,
            "grav_comp_kd": self._grav_comp_kd,
            "coulomb_friction": self._coulomb_friction,
            "use_coulomb_friction": self.use_coulomb_friction,
            "joint_limits": self._joint_limits,
            "gripper_limits": self._gripper_limits,
            "gravity_comp_factor": self.gravity_comp_factor,
            "gripper_index": self._gripper_index,
            "enable_auto_recovery": getattr(self.motor_chain, "enable_auto_recovery", False),
        }
        if self._gripper_index is not None:
            info["limit_gripper_effort"] = self._limit_gripper_force
        return info

    def start_server(self) -> None:
        """Start the server."""
        last_time = time.time()
        iteration_count = 0
        self.update()

        logging.info("initializing, ....")

        while not self._stop_event.is_set():  # Check the stop event
            current_time = time.time()
            elapsed_time = current_time - last_time

            self.update()
            if not self.motor_chain.running:
                raise RuntimeError(f"{self}: motor_chain_robot's motor chain is not running, exiting the robot server")
            time.sleep(0.001)

            iteration_count += 1
            if elapsed_time >= 10.0:
                control_frequency = iteration_count / elapsed_time
                # Overwrite the current line with the new frequency information
                logging.info(f"{self}: Grav Comp Control Frequency: {control_frequency:.2f} Hz")
                if control_frequency < 100:
                    logging.warning(
                        f"{self}: Gravity compensation control loop is slow, current frequency: {control_frequency:.2f} Hz"
                    )
                # Reset the counter and timer
                last_time = current_time
                iteration_count = 0

    def update(self) -> None:
        """Update the robot.

        Send Torques and update the joint state.
        """
        with self._command_lock:
            joint_commands = copy.deepcopy(self._commands)
        with self._state_lock:
            g = self._compute_gravity_compensation(self._joint_state)
            friction_comp = (
                self._coulomb_friction * np.sign(self._joint_state.vel) if self.use_coulomb_friction else 0.0
            )
            motor_torques = joint_commands.torques + g * self.gravity_comp_factor + friction_comp
            # Everything summed above is a torque on the *joint*: the caller's commanded torque, the
            # model's gravity term and the friction term alike. Behind a linkage the motor sees that
            # scaled by the transmission ratio -- on joint6 a ~30% correction mid-travel, and ~7x
            # near the upper stop where the ratio falls to 0.14. Scaled before the motor-side clip,
            # which is a limit on what the motor may be asked for.
            for idx, linkage in self._transmissions.items():
                motor_torques[idx] *= linkage.ratio(linkage.motor_from_joint(self._joint_state.pos[idx]))
            motor_torques = np.clip(motor_torques, -self._clip_motor_torque, self._clip_motor_torque)
            self._last_motor_torques = motor_torques.copy()

            if self._gripper_index is not None:
                if self._limit_gripper_force > 0 and self._joint_state is not None:
                    # Get current gripper state in raw robot joint pos space
                    gripper_state = {
                        "target_qpos": joint_commands.pos[self._gripper_index],
                        "current_qpos": self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[
                            self._gripper_index
                        ],
                        "current_qvel": self._joint_state.vel[self._gripper_index],
                        "current_eff": self._joint_state.eff[self._gripper_index],
                        "current_normalized_qpos": self._joint_state.pos[self._gripper_index],
                        "target_normalized_qpos": self.remapper.to_command_joint_pos_space(joint_commands.pos)[
                            self._gripper_index
                        ],
                        "last_command_qpos": self._last_gripper_command_qpos,
                    }

                    joint_commands.pos[self._gripper_index] = self._gripper_force_limiter.update(gripper_state)

                # add final clip so the gripper won't be over-adjusted
                joint_commands.pos[self._gripper_index] = np.clip(
                    joint_commands.pos[self._gripper_index],
                    min(self._gripper_limits),
                    max(self._gripper_limits),
                )
                self._last_gripper_command_qpos = joint_commands.pos[self._gripper_index]
            self._update_joint_state(motor_torques, joint_commands)

    def _update_joint_state(
        self,
        motor_torques: np.ndarray,
        joint_commands: "JointCommands",
        encoder_infos: Optional[List[PassiveEncoderInfo]] = None,
    ) -> None:
        """Send commands to motor chain, update joint state, and optionally save to disk."""
        if (
            hasattr(self.motor_chain, "get_same_bus_device_states")
            and callable(self.motor_chain.get_same_bus_device_states)
            and self.motor_chain.same_bus_device_driver is not None
        ):
            has_gripper_encoder = True
            encoder_infos = self.motor_chain.get_same_bus_device_states()
            assert len(encoder_infos) == 1, "Only one encoder is supported"
            assert isinstance(encoder_infos[0], PassiveEncoderInfo), "Encoder info must be a PassiveEncoderInfo"
        else:
            has_gripper_encoder = False

        motor_state = self.motor_chain.set_commands(
            motor_torques,
            pos=joint_commands.pos,
            vel=joint_commands.vel,
            kp=joint_commands.kp,
            kd=joint_commands.kd,
        )
        self._joint_state = self._motor_state_to_joint_state(motor_state)

        with self._mcap_lock:
            if self._mcap_recorder is not None:
                try:
                    self._mcap_recorder.add(
                        timestamp=self._joint_state.timestamp,
                        position=self._joint_state.pos,
                        velocity=self._joint_state.vel,
                        effort=self._joint_state.eff,
                        required_torque=motor_torques,
                        temp_mos=self._joint_state.temp_mos,
                        temp_rotor=self._joint_state.temp_rotor,
                    )
                except RuntimeError:
                    logging.exception("MCAP recording stopped after its writer failed")
                    self._mcap_recorder = None

        # For SWE-454: keep monitoring qpos during runtime
        self._check_current_qpos_in_joint_limits()

        if self._joint_state_saver is not None:
            assert not (has_gripper_encoder and self._gripper_index is not None), (
                "Either has_gripper_encoder=True or self._gripper_index is not None"
            )
            ee_pos = ee_vel = ee_eff = None
            if has_gripper_encoder:
                # gripper_pos, not position: this channel has to be in the arm convention
                # (0 = closed, 1 = open) to match the gripper_index branch below, which
                # records the already-remapped joint position.  The raw encoder reads the
                # other way round, so recording it here mirrored teaching-handle episodes
                # against motorized-gripper ones.
                ee_pos = np.array([info.gripper_pos for info in encoder_infos])
                # Still the raw signed rad/s, unlike the normalized units/s the
                # gripper_index branch records -- needs range_rad and a bench sign check.
                ee_vel = np.array([info.velocity for info in encoder_infos])
            elif self._gripper_index is not None:
                ee_pos = self._joint_state.pos[self._gripper_index]
                ee_vel = self._joint_state.vel[self._gripper_index]
                ee_eff = self._joint_state.eff[self._gripper_index]

            if self._gripper_index is None:
                pos = self._joint_state.pos
                vel = self._joint_state.vel
                eff = self._joint_state.eff
            else:
                pos = self._joint_state.pos[: self._gripper_index]
                vel = self._joint_state.vel[: self._gripper_index]
                eff = self._joint_state.eff[: self._gripper_index]

            self._joint_state_saver.add(
                timestamp=self._joint_state.timestamp,
                pos=pos,
                vel=vel,
                eff=eff,
                ee_pos=ee_pos,
                ee_vel=ee_vel,
                ee_eff=ee_eff,
            )

    def _joint_to_motor_pos(self, pos: np.ndarray) -> np.ndarray:
        """Invert the transmissions, so what reaches the CAN frames is motor angles.

        The inverse is bracketed inside each linkage's usable travel, so a command can never be
        solved onto the far side of a dead point -- which would send the motor most of a turn from
        where it is. Returns the input untouched on a direct-drive arm.
        """
        if not self._transmissions:
            return pos
        pos = np.asarray(pos, dtype=float).copy()
        for idx, linkage in self._transmissions.items():
            pos[idx] = linkage.motor_from_joint(pos[idx])
        return pos

    def _joint_to_motor_vel(self, vel: np.ndarray) -> np.ndarray:
        """Joint rates to motor rates, the ratio taken at the joint's present position."""
        if not self._transmissions:
            return vel
        vel = np.asarray(vel, dtype=float).copy()
        for idx, linkage in self._transmissions.items():
            vel[idx] /= linkage.ratio(linkage.motor_from_joint(self._joint_state.pos[idx]))
        return vel

    def _motor_state_to_joint_state(self, motor_state: List[MotorInfo]) -> JointStates:
        """Convert motor state to joint state.

        Args:
            motor_state (List[Any]): The motor state.

        Returns:
            Dict[str, np.ndarray]: The joint state.
        """
        names = [f"joint{i + 1}" for i in range(len(motor_state))]
        if self._gripper_index is not None:
            names[self._gripper_index] = "gripper"
        pos = np.array([motor.pos for motor in motor_state])
        vel = np.array([motor.vel for motor in motor_state])
        eff = np.array([motor.eff for motor in motor_state])
        # Joints driven through a linkage read their motor's angle, not their own. Convert here, at
        # the one place motor feedback becomes joint state, so everything above -- the observations,
        # the joint limits, gravity comp's `q`, MCAP -- is in joint space without knowing about it.
        for idx, linkage in self._transmissions.items():
            lo, hi = linkage.motor_range
            # Checked before converting: past the travel the converted angle can still land inside the
            # joint limits, and NaN fails every comparison, so neither later guard would see it.
            if not lo - TRANSMISSION_MOTOR_MARGIN <= pos[idx] <= hi + TRANSMISSION_MOTOR_MARGIN:
                self.motor_chain.running = False
                raise RuntimeError(
                    f"{self}: joint{idx + 1}'s motor reads {np.degrees(pos[idx]):.2f} deg, outside its linkage's "
                    f"travel [{np.degrees(lo):.2f}, {np.degrees(hi):.2f}] deg. Check the motor's zero and its "
                    f"`directions` sign."
                )
            ratio = linkage.ratio(pos[idx])
            pos[idx] = linkage.joint_from_motor(pos[idx])
            vel[idx] *= ratio
            eff[idx] /= ratio
        pos = self.remapper.to_command_joint_pos_space(pos)
        vel = self.remapper.to_command_joint_vel_space(vel)
        temp_mos = np.array([motor.temp_mos for motor in motor_state])
        temp_rotor = np.array([motor.temp_rotor for motor in motor_state])
        timestamp = motor_state[0].timestamp
        return JointStates(
            names=names,
            pos=pos,
            vel=vel,
            eff=eff,
            temp_mos=temp_mos,
            temp_rotor=temp_rotor,
            timestamp=timestamp,
        )

    def _compute_gravity_compensation(self, joint_state: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        if joint_state is None or not self.use_gravity_comp:
            return np.zeros(len(self.motor_chain))
        elif self.use_gravity_comp:
            q = joint_state.pos[: self._gripper_index] if self._gripper_index is not None else joint_state.pos
            t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            # print gravity torque to 2f
            if not np.all(np.isfinite(t)) or np.max(np.abs(t)) > 25.0:
                print([f"{s:.2f}" for s in t])
                raise RuntimeError(f"{self}: too large torques")
            if self._gripper_index is None:
                return self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            else:
                t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
                return np.append(t, 0.0)

    # ----------------- Server Functions ----------------- #

    def num_dofs(self) -> int:
        """Get the number of joints of the robot, including the gripper.

        Returns:
            int: The number of joints of the robot.
        """
        return len(self.motor_chain)

    def get_motor_torques(self) -> Optional[np.ndarray]:
        """Return the last computed motor torques (gravity comp + any command torques)."""
        return self._last_motor_torques

    def get_joint_pos(self) -> np.ndarray:
        """Get the current state of the leader robot, including the gripper in radian.

        Returns:
            T: The current state of the leader robot.
        """
        with self._state_lock:
            return self._joint_state.pos

    def _clip_robot_joint_pos_command(self, pos: np.ndarray) -> np.ndarray:
        """Clip the robot joint pos command to the joint limits. Do not clip the gripper pos.
        Args:
            pos (np.ndarray): The joint pos command to clip.
        Returns:
            np.ndarray: The clipped joint pos command.
        """

        if self._joint_limits is not None:
            if self._gripper_index is not None:
                pos[: self._gripper_index] = np.clip(
                    pos[: self._gripper_index],
                    self._joint_limits[:, 0],
                    self._joint_limits[:, 1],
                )
            else:
                pos = np.clip(pos, self._joint_limits[:, 0], self._joint_limits[:, 1])
        return pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_pos (np.ndarray): The state to command the leader robot to.
        """
        pos = self._clip_robot_joint_pos_command(joint_pos)
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.pos = self._joint_to_motor_pos(self.remapper.to_robot_joint_pos_space(pos))
            self._commands.kp = self._kp
            self._commands.kd = self._kd

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        pos = self._clip_robot_joint_pos_command(joint_state["pos"])
        vel = joint_state["vel"]
        self._commands = JointCommands.init_all_zero(len(self.motor_chain))
        kp = joint_state.get("kp", self._kp)
        kd = joint_state.get("kd", self._kd)
        with self._command_lock:
            self._commands.pos = self._joint_to_motor_pos(self.remapper.to_robot_joint_pos_space(pos))
            self._commands.vel = self._joint_to_motor_vel(self.remapper.to_robot_joint_vel_space(vel))
            self._commands.kp = kp
            self._commands.kd = kd

    def zero_torque_mode(self) -> None:
        logging.info(f"Entering zero_torque_mode for {self}")
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._kp = np.zeros(len(self.motor_chain))
            self._kd = np.zeros(len(self.motor_chain))

    def _handle_gripper_pos(self) -> Optional[float]:
        """Teaching-handle trigger squeeze as a normalized 0..1 gripper position.

        Reported in the arm convention (0 = closed, 1 = open) via
        ``PassiveEncoderInfo.gripper_pos``, so a consumer reading ``gripper_pos`` cannot
        tell a squeezed trigger from a closed motorized gripper.  None on robots without
        a passive encoder on the same CAN bus.
        """
        if getattr(self.motor_chain, "same_bus_device_driver", None) is None:
            return None
        states = self.motor_chain.get_same_bus_device_states()
        if not states:
            return None
        return float(np.clip(states[0].gripper_pos, 0.0, 1.0))

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        # Outside _state_lock: it takes the chain's encoder lock, and nesting the two would let a
        # slow encoder read stall update(), which needs _state_lock every control cycle.
        handle_pos = self._handle_gripper_pos() if self._gripper_index is None else None
        with self._state_lock:
            if self._gripper_index is None:
                result = {
                    "joint_pos": self._joint_state.pos,
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }
                if handle_pos is not None:
                    result["gripper_pos"] = np.array([handle_pos])
            else:
                result = {
                    "joint_pos": self._joint_state.pos[: self._gripper_index],
                    "joint_vel": self._joint_state.vel[: self._gripper_index],
                    "joint_eff": self._joint_state.eff[: self._gripper_index],
                    "gripper_pos": np.array([self._joint_state.pos[self._gripper_index]]),
                    "gripper_vel": np.array([self._joint_state.vel[self._gripper_index]]),
                    "gripper_eff": np.array([self._joint_state.eff[self._gripper_index]]),
                }
            if self.temp_record_flag:
                result["temp_mos"] = self._joint_state.temp_mos
                result["temp_rotor"] = self._joint_state.temp_rotor
            return result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context related to this object."""
        self.close()

    def move_joints(self, target_joint_positions: np.ndarray, time_interval_s: float = 2.0) -> None:
        """Move the robot to a given joint positions."""
        with self._state_lock:
            current_pos = self._joint_state.pos
        assert len(current_pos) == len(target_joint_positions)
        steps = 50  # 50 steps over time_interval_s
        for i in range(steps + 1):
            alpha = i / steps  # Interpolation factor
            target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
            self.command_joint_pos(target_pos)
            time.sleep(time_interval_s / steps)

    def close(self) -> None:
        """Safely close the robot by setting all torques to zero."""
        # self.move_to_zero()
        self._stop_event.set()  # Signal the thread to stop
        self._server_thread.join()  # Wait for the thread to finish
        try:
            self.motor_chain.close()
        finally:
            self.stop_mcap_recording()
        print("Robot closed with all torques set to zero.")

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        assert kp.shape == self._kp.shape == kd.shape
        self._kp = kp
        self._kd = kd

    def enter_gravity_comp_idle(self) -> None:
        """Reset active commands to gravity-comp idle.

        Sets ``self._commands`` to zeros with ``kd = self._grav_comp_kd``,
        mirroring the ``zero_gravity_mode=True`` branch in ``__init__``. Use
        this after running PD control (e.g. ``command_joint_pos``) to re-enter
        grav-comp idle without leaving the previous target pose/gains active.
        Leaves ``self._kp`` / ``self._kd`` unchanged so subsequent control
        commands still use the configured control gains.

        Also turns gravity compensation and the configured Coulomb friction feedforward back on, so
        this is the way back from ``enter_zero_torque_idle``. A robot built without a model never had
        gravity comp and stays off.
        """
        with self._command_lock:
            self.use_gravity_comp = self._gravity_comp_available
            self.use_coulomb_friction = self._coulomb_friction_configured
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.kd = self._grav_comp_kd.copy()

    def enter_zero_torque_idle(self) -> None:
        """Command literally nothing: zero torque, zero PD gains, no gravity comp and no friction comp.

        Friction comp has to go too: it pushes along ``sign(q_dot)``, so on an unheld arm it would
        help the sag along. The same frame ``dm_driver.py``'s bring-up loop sends. The motors stay enabled and keep
        answering, so joint state still streams, but **nothing holds the arm** -- it is carried by
        its own friction alone and will sag or fall. Support the arm before calling this.

        Distinct from the two neighbours it is easy to confuse it with:
        ``enter_gravity_comp_idle`` still sends the gravity torque, and ``zero_torque_mode``
        leaves ``use_gravity_comp`` on -- so it does *not* zero the torque -- while permanently
        zeroing ``_kp``/``_kd``, which no later command restores. ``enter_gravity_comp_idle`` is
        the way back from here.
        """
        logging.info(f"{self}: entering zero-torque idle -- nothing is holding the arm")
        with self._command_lock:
            self.use_gravity_comp = False
            self.use_coulomb_friction = False
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))

    def start_recording(self, save_dir: str) -> bool:
        """Start recording joint state data asynchronously."""
        if self._joint_state_saver is None:
            raise RuntimeError("Joint state saver factory not provided, recording not available")
        self._joint_state_saver.start_recording(save_dir)
        return True

    def stop_recording(self, prefix: str = "") -> Tuple[bool, str]:
        """Stop recording joint state data asynchronously."""
        if self._joint_state_saver is None:
            raise RuntimeError("Joint state saver not available")
        succ = self._joint_state_saver.stop_recording(prefix)
        if succ:
            return succ, "Recording stopped successfully"
        return succ, "Recording failed to stop"

    def start_mcap_recording(self) -> Path:
        """Start recording every motor feedback frame to a timestamped ROS 2 MCAP file."""
        with self._mcap_lock:
            if self._mcap_recorder is not None:
                raise RuntimeError(f"MCAP recording is already active: {self._mcap_recorder.path}")
            assert self._joint_state is not None
            self._mcap_recorder = RobotMcapRecorder.create(self._joint_state.names)
            return self._mcap_recorder.path

    def stop_mcap_recording(self) -> None:
        """Finish the active MCAP recording, if any."""
        with self._mcap_lock:
            recorder = self._mcap_recorder
            self._mcap_recorder = None
        if recorder is not None:
            recorder.close()


@dataclass
class _CliArgs:
    """Drive a YAM-family arm over CAN."""

    arm: ArmName = "yam"
    """Arm variant."""
    gripper: str = "linear_4310"
    """Gripper variant."""
    channel: str = "can0"
    """CAN channel."""
    operation_mode: Literal["gravity_comp", "test_gripper", "stay_current_qpos"] = "gravity_comp"
    """Operation mode: gravity compensation, gripper cycling, or holding the startup joint positions."""
    record: bool = False
    """Record motor feedback and computed required torques to a ROS 2 CDR MCAP file."""


if __name__ == "__main__":
    import tyro

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.utils.utils import override_log_level

    override_log_level(level=logging.INFO)

    args = tyro.cli(_CliArgs)

    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    print(f"Initializing robot with arm_type: {arm_type}, gripper_type: {gripper_type}")
    robot = get_yam_robot(args.channel, arm_type=arm_type, gripper_type=gripper_type)

    try:
        if args.record:
            print(f"Recording motor feedback to {robot.start_mcap_recording()}")
        if args.operation_mode == "gravity_comp":
            while True:
                time.sleep(1)
        elif args.operation_mode == "test_gripper":
            assert not gripper_type.is_teaching_handle, (
                f"test_gripper is not supported for {gripper_type.name}, teaching handle is a passive device"
            )
            for _ in range(30):
                for gripper_pos in [0.8, 0.0]:
                    print(f"gripper_pos: {gripper_pos}")
                    # Sized from the chain, not written out: the gripper is always the last motor,
                    # but an arm with seven joints makes that index 7 rather than 6.
                    command = np.zeros(robot.num_dofs())
                    command[-1] = gripper_pos
                    robot.command_joint_pos(command)
                    time.sleep(4)
                    print(robot.get_observations())
        elif args.operation_mode == "stay_current_qpos":
            current_qpos = robot.get_joint_pos()
            robot.command_joint_pos(current_qpos)
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        robot.close()
