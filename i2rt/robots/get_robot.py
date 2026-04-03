import logging
import xml.etree.ElementTree as ET
from functools import partial
from typing import Any, Callable, Optional

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    PassiveEncoderReader,
    ReceiveMode,
)
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import (
    ArmType,
    GripperType,
    _load_arm_config,
    combine_arm_and_gripper_xml,
)

logger = logging.getLogger(__name__)


def _load_joint_limits_from_xml(*xml_paths: str) -> np.ndarray:
    """Parse joint limits (range attributes) from one or more XML files.

    Collects all ``<joint name="jointN" range="lo hi">`` elements across the
    given XML files.  Returns an (N, 2) array of [lower, upper] limits,
    ordered by joint name (joint1, joint2, ...).  Duplicate joint names are
    ignored (first occurrence wins).
    """
    seen: set[str] = set()
    joints: list[tuple[str, float, float]] = []
    for xml_path in xml_paths:
        logger.info(f"Loading joint limits from XML: {xml_path}")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for joint_elem in root.iter("joint"):
            name = joint_elem.get("name", "")
            range_str = joint_elem.get("range")
            if range_str and name.startswith("joint") and name not in seen:
                lo, hi = (float(x) for x in range_str.split())
                joints.append((name, lo, hi))
                seen.add(name)

    limits = np.array([[lo, hi] for _, lo, hi in joints])
    logger.info(f"  joint limits ({len(joints)} joints):")
    for name, lo, hi in joints:
        logger.info(f"    {name}: [{lo:.5f}, {hi:.5f}]")
    return limits


def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    return EncoderChain([0x50E], passive_encoder_reader)


def _get_gripper_only_robot(
    channel: str = "can0",
    gripper_type: GripperType = GripperType.LINEAR_4310,
    sim: bool = False,
) -> "Robot":
    """Create a gripper-only robot (no arm).

    Args:
        channel: CAN interface name (e.g. "can0"). Ignored in sim mode.
        gripper_type: Which gripper to load. Must not be NO_GRIPPER.
        sim: If True, return a SimRobot instead of connecting to real hardware.
    """
    if gripper_type == GripperType.NO_GRIPPER:
        raise ValueError("gripper_type cannot be NO_GRIPPER when arm_type is NO_ARM")

    xml_path = gripper_type.get_xml_path()
    # One motor drives the gripper; extra XML joints are coupled via equality constraints.
    n_dofs = 1

    nominal_arm = ArmType.YAM
    gripper_limits = gripper_type.get_gripper_limits(nominal_arm)
    gripper_needs_cal = gripper_type.get_gripper_needs_calibration(nominal_arm)

    if sim:
        from i2rt.robots.sim_robot import SimRobot

        sim_gripper_limits = gripper_limits
        if sim_gripper_limits is None:
            sim_gripper_limits = np.array([0.0, 1.0])

        return SimRobot(
            xml_path=xml_path,
            n_dofs=n_dofs,
            gripper_index=0,
            gripper_limits=sim_gripper_limits,
        )

    # --- Real hardware path ---------------------------------------------------
    motor_type = gripper_type.get_motor_type(nominal_arm)
    gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd(nominal_arm)
    direction = gripper_type.get_motor_direction(nominal_arm)

    motor_chain = DMChainCanInterface(
        [[0x07, motor_type]],
        [0.0],
        [direction],
        channel,
        motor_chain_name="gripper_only",
        receive_mode=ReceiveMode.p16,
        start_thread=True,
    )

    return MotorChainRobot(
        motor_chain=motor_chain,
        xml_path=xml_path,
        use_gravity_comp=False,
        joint_limits=None,
        kp=np.array([gripper_kp]),
        kd=np.array([gripper_kd]),
        gripper_index=0,
        gripper_limits=gripper_limits,
        enable_gripper_calibration=gripper_needs_cal,
        gripper_type=gripper_type,
        arm_type=nominal_arm,
        zero_gravity_mode=False,
    )


def get_yam_robot(
    channel: str = "can0",
    arm_type: ArmType = ArmType.YAM,
    gripper_type: GripperType = GripperType.LINEAR_4310,
    zero_gravity_mode: bool = True,
    ee_mass: Optional[float] = None,
    ee_inertia: Optional[np.ndarray] = None,
    gravity_comp_factor: Optional[np.ndarray] = None,
    sim: bool = False,
    joint_state_saver_factory: Optional[Callable[[], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
) -> "Robot":
    """Create a YAM-family robot (real or sim).

    Args:
        channel: CAN interface name (e.g. "can0"). Ignored in sim mode.
        arm_type: Which arm variant to use. Use ``ArmType.NO_ARM`` for gripper-only.
        gripper_type: Which gripper (or NO_GRIPPER / YAM_TEACHING_HANDLE).
        zero_gravity_mode: Start in gravity-compensation mode.
        ee_mass: Optional end-effector mass override (kg) for MuJoCo inertial.
        ee_inertia: Optional 10-element inertia override [ipos(3), quat(4), diaginertia(3)].
        gravity_comp_factor: Per-joint array (6 elements, arm joints only) multiplied against gravity torques.
            Overrides the arm-type default when provided.
        sim: If True, return a SimRobot instead of connecting to real hardware.
    """
    # --- Gripper-only path (no arm) -------------------------------------------
    if arm_type == ArmType.NO_ARM:
        return _get_gripper_only_robot(channel=channel, gripper_type=gripper_type, sim=sim)

    with_gripper = gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER)
    with_teaching_handle = gripper_type == GripperType.YAM_TEACHING_HANDLE

    hw = _load_arm_config(arm_type)
    effective_gravity_comp = hw.gravity_comp_factor if gravity_comp_factor is None else gravity_comp_factor
    if with_gripper:
        effective_gravity_comp = np.append(effective_gravity_comp, 1.0)

    model_path = combine_arm_and_gripper_xml(
        arm_type,
        gripper_type,
        ee_mass=ee_mass,
        ee_inertia=ee_inertia,
    )

    # Load limits for motor-driven joints only (arm joints + last wrist joint from gripper XML).
    all_joint_limits = _load_joint_limits_from_xml(arm_type.get_xml_path(), gripper_type.get_xml_path())
    n_arm_joints = len(hw.motor_list)
    joint_limits = all_joint_limits[:n_arm_joints]
    joint_limits[:, 0] -= 0.15  # safety buffer
    joint_limits[:, 1] += 0.15

    # Build mutable lists from the frozen arm config, then extend for gripper.
    motor_list = [[can_id, mtype] for can_id, mtype in hw.motor_list]
    directions = list(hw.directions)
    kp = hw.kp.copy()
    kd = hw.kd.copy()
    motor_offsets = [0.0] * len(motor_list)

    if with_gripper:
        motor_type = gripper_type.get_motor_type(arm_type)
        gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd(arm_type)
        logging.info(f"adding gripper motor type={motor_type}, kp={gripper_kp}, kd={gripper_kd}")
        motor_list.append([0x07, motor_type])
        motor_offsets.append(0.0)
        directions.append(gripper_type.get_motor_direction(arm_type))
        kp = np.append(kp, gripper_kp)
        kd = np.append(kd, gripper_kd)

    gripper_limits = gripper_type.get_gripper_limits(arm_type) if with_gripper else None
    gripper_needs_cal = gripper_type.get_gripper_needs_calibration(arm_type) if with_gripper else False

    if sim:
        from i2rt.robots.sim_robot import SimRobot

        # In sim mode, grippers that need calibration have no limits yet — use [0, 1] default.
        sim_gripper_limits = gripper_limits
        if with_gripper and sim_gripper_limits is None:
            sim_gripper_limits = np.array([0.0, 1.0])

        return SimRobot(
            xml_path=model_path,
            n_dofs=len(motor_list),
            joint_limits=joint_limits,
            gripper_index=n_arm_joints if with_gripper else None,
            gripper_limits=sim_gripper_limits,
        )

    # --- Real hardware path ---------------------------------------------------

    # Single pass: create chain, read positions, fix wrap-around offsets in-place, then start thread.
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        start_thread=False,
        get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
        use_buffered_reader=False,
    )
    motor_states = motor_chain.read_states()
    logging.debug(f"motor_states: {motor_states}")

    logging.info(f"current_pos: {[m.pos for m in motor_states]}")
    for idx, state in enumerate(motor_states):
        if state.pos < -np.pi:
            logging.info(f"motor {idx} pos={state.pos:.3f}, offset -2π")
            motor_chain.motor_offset[idx] -= 2 * np.pi
        elif state.pos > np.pi:
            logging.info(f"motor {idx} pos={state.pos:.3f}, offset +2π")
            motor_chain.motor_offset[idx] += 2 * np.pi

    logging.info(f"adjusted motor_offsets: {motor_chain.motor_offset.tolist()}")

    # Start the control thread with corrected offsets.
    motor_chain.start_thread()
    logging.info(f"YAM initial motor_states: {motor_chain.read_states()}")

    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=effective_gravity_comp,
        joint_limits=joint_limits,
        kp=kp,
        kd=kd,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
    )

    if with_gripper:
        return get_robot(
            gripper_index=n_arm_joints,
            gripper_limits=gripper_limits,
            enable_gripper_calibration=gripper_needs_cal,
            gripper_type=gripper_type,
            arm_type=arm_type,
            limit_gripper_force=50.0,
        )
    return get_robot()
