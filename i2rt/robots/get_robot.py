import logging
import time
from dataclasses import dataclass
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
from i2rt.robots.utils import (
    ArmType,
    GripperType,
    combine_arm_and_gripper_xml,
)

# ---------------------------------------------------------------------------
# Per-arm joint limits (rad) — match XML joint ranges exactly.
# A 0.15 rad safety buffer is added at runtime before passing to MotorChainRobot.
# ---------------------------------------------------------------------------
_ARM_JOINT_LIMITS: dict[ArmType, np.ndarray] = {
    ArmType.YAM: np.array(
        [[-2.618, 3.054], [0.0, 3.65], [0.0, 3.665], [-1.571, 1.571], [-1.571, 1.571], [-2.094, 2.094]]
    ),
    ArmType.YAM_PRO: np.array(
        [[-2.618, 3.054], [0.0, 3.65], [0.0, 3.665], [-1.571, 1.571], [-1.571, 1.571], [-2.094, 2.094]]
    ),
    ArmType.YAM_ULTRA: np.array(
        [[-2.618, 3.054], [0.0, 3.65], [0.0, 3.142], [-1.571, 1.571], [-1.571, 1.571], [-2.094, 2.094]]
    ),
    ArmType.BIG_YAM: np.array(
        [[-2.618, 3.130], [0.0, 3.650], [0.0, 3.130], [-1.650, 1.650], [-1.571, 1.571], [-2.094, 2.094]]
    ),
}


# ---------------------------------------------------------------------------
# Per-arm hardware config — motor types, polarities, PD gains, gravity factor.
# Only covers the 6 arm joints; the gripper motor (0x07) is appended at runtime.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ArmHWConfig:
    motor_list: tuple  # ((can_id, motor_type_str), ...) — 6 arm joints
    directions: tuple  # motor polarity (+1 / -1), one per arm joint
    kp: np.ndarray  # position gain, one per arm joint
    kd: np.ndarray  # damping gain,  one per arm joint
    gravity_comp_factor: float


# YAM / YAM Pro / YAM Ultra: 3xDM4340 (shoulder) + 3xDM4310 (elbow/wrist)
_YAM_HW = _ArmHWConfig(
    motor_list=(
        (0x01, "DM4340"),
        (0x02, "DM4340"),
        (0x03, "DM4340"),
        (0x04, "DM4310"),
        (0x05, "DM4310"),
        (0x06, "DM4310"),
    ),
    directions=(1, 1, 1, 1, 1, 1),
    kp=np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0]),
    kd=np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5]),
    gravity_comp_factor=1.3,
)

# big_yam: heavier arm - joints 1-2 use DM6248, joints 3-4 use DM4340, joints 5-6 use DM4310.
# Joint 2 direction is reversed relative to YAM family.
_BIG_YAM_HW = _ArmHWConfig(
    motor_list=(
        (0x01, "DM6248"),
        (0x02, "DM6248"),
        (0x03, "DM4340"),
        (0x04, "DM4340"),
        (0x05, "DM4310"),
        (0x06, "DM4310"),
    ),
    directions=(1, -1, 1, 1, 1, 1),
    kp=np.array([80.0, 80.0, 80.0, 40.0, 40.0, 10.0]),
    kd=np.array([5.0, 5.0, 5.0, 3.0, 1.5, 1.5]),
    gravity_comp_factor=1.0,
)

_ARM_HW_CONFIGS: dict[ArmType, _ArmHWConfig] = {
    ArmType.YAM: _YAM_HW,
    ArmType.YAM_PRO: _YAM_HW,
    ArmType.YAM_ULTRA: _YAM_HW,
    ArmType.BIG_YAM: _BIG_YAM_HW,
}


def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    return EncoderChain([0x50E], passive_encoder_reader)


def get_yam_robot(
    channel: str = "can0",
    arm_type: ArmType = ArmType.YAM,
    gripper_type: GripperType = GripperType.LINEAR_4310,
    zero_gravity_mode: bool = True,
    ee_mass: Optional[float] = None,
    ee_inertia: Optional[np.ndarray] = None,
    sim: bool = False,
    joint_state_saver_factory: Optional[Callable[[], Any]] = None,
    set_realtime_and_pin_callback: Optional[Callable[[int], None]] = None,
) -> "MotorChainRobot":
    """Create a YAM-family robot (real or sim).

    Args:
        channel: CAN interface name (e.g. "can0"). Ignored in sim mode.
        arm_type: Which arm variant to use.
        gripper_type: Which gripper (or NO_GRIPPER / YAM_TEACHING_HANDLE).
        zero_gravity_mode: Start in gravity-compensation mode.
        ee_mass: Optional end-effector mass override (kg) for MuJoCo inertial.
        ee_inertia: Optional 10-element inertia override [ipos(3), quat(4), diaginertia(3)].
        sim: If True, return a SimRobot instead of connecting to real hardware.
    """
    with_gripper = gripper_type not in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER)
    with_teaching_handle = gripper_type == GripperType.YAM_TEACHING_HANDLE

    hw = _ARM_HW_CONFIGS[arm_type]

    model_path = combine_arm_and_gripper_xml(arm_type.get_xml_path(), gripper_type.get_xml_path(), ee_mass, ee_inertia)

    joint_limits = _ARM_JOINT_LIMITS[arm_type].copy()
    joint_limits[:, 0] -= 0.15  # safety buffer
    joint_limits[:, 1] += 0.15

    # Build mutable lists from the frozen arm config, then extend for gripper.
    motor_list = [[can_id, mtype] for can_id, mtype in hw.motor_list]
    directions = list(hw.directions)
    kp = hw.kp.copy()
    kd = hw.kd.copy()
    motor_offsets = [0.0] * len(motor_list)

    if with_gripper:
        motor_type = gripper_type.get_motor_type()
        gripper_kp, gripper_kd = gripper_type.get_motor_kp_kd()
        logging.info(f"adding gripper motor type={motor_type}, kp={gripper_kp}, kd={gripper_kd}")
        motor_list.append([0x07, motor_type])
        motor_offsets.append(0.0)
        directions.append(1)
        kp = np.append(kp, gripper_kp)
        kd = np.append(kd, gripper_kd)

    gripper_limits = gripper_type.get_gripper_limits() if with_gripper else None
    gripper_needs_cal = gripper_type.get_gripper_needs_calibration() if with_gripper else False

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
            gripper_index=6 if with_gripper else None,
            gripper_limits=sim_gripper_limits,
        )

    # --- Real hardware path ---------------------------------------------------

    # First pass: read current positions to compute wrap-around offsets.
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        start_thread=False,
    )
    motor_states = motor_chain.read_states()
    print(f"motor_states: {motor_states}")
    motor_chain.close()

    logging.info(f"current_pos: {[m.pos for m in motor_states]}")
    for idx, state in enumerate(motor_states):
        if state.pos < -np.pi:
            logging.info(f"motor {idx} pos={state.pos:.3f}, offset -2π")
            motor_offsets[idx] -= 2 * np.pi
        elif state.pos > np.pi:
            logging.info(f"motor {idx} pos={state.pos:.3f}, offset +2π")
            motor_offsets[idx] += 2 * np.pi

    time.sleep(0.5)
    logging.info(f"adjusted motor_offsets: {motor_offsets}")

    # Second pass: start the control loop with corrected offsets.
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
        use_buffered_reader=False,
    )
    logging.info(f"YAM initial motor_states: {motor_chain.read_states()}")

    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=hw.gravity_comp_factor,
        joint_limits=joint_limits,
        kp=kp,
        kd=kd,
        zero_gravity_mode=zero_gravity_mode,
        joint_state_saver_factory=joint_state_saver_factory,
        set_realtime_and_pin_callback=set_realtime_and_pin_callback,
    )

    if with_gripper:
        return get_robot(
            gripper_index=6,
            gripper_limits=gripper_limits,
            enable_gripper_calibration=gripper_needs_cal,
            gripper_type=gripper_type,
            limit_gripper_force=50.0,
        )
    return get_robot()


if __name__ == "__main__":
    import argparse

    arm_choices = [a.value for a in ArmType]
    gripper_choices = [g.value for g in GripperType]

    parser = argparse.ArgumentParser(description="Initialize a YAM robot")
    parser.add_argument("--arm", type=str, default="yam", choices=arm_choices)
    parser.add_argument("--gripper", type=str, default="linear_4310", choices=gripper_choices)
    parser.add_argument("--sim", action="store_true", help="Use sim mode instead of real hardware")
    parser.add_argument("--channel", type=str, default="can0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    robot = get_yam_robot(
        channel=args.channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        sim=args.sim,
    )
    print(f"Robot initialized: arm={args.arm}, gripper={args.gripper}, sim={args.sim}")

    while True:
        print(robot.get_observations())
        time.sleep(1)
