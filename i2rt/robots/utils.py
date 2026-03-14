import enum
import logging
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from i2rt.motor_drivers.dm_driver import DMChainCanInterface

I2RT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Arm XML paths
ARM_YAM_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arm/yam/yam.xml")
ARM_YAM_PRO_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arm/yam_pro/yam_pro.xml")
ARM_YAM_ULTRA_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arm/yam_ultra/yam_ultra.xml")
ARM_BIG_YAM_XML_PATH = os.path.join(I2RT_ROOT, "robot_models/arm/big_yam/big_yam.xml")

# Gripper XML paths
GRIPPER_CRANK_4310_PATH = os.path.join(I2RT_ROOT, "robot_models/gripper/crank_4310/crank_4310.xml")
GRIPPER_LINEAR_3507_PATH = os.path.join(I2RT_ROOT, "robot_models/gripper/linear_3507/linear_3507.xml")
GRIPPER_LINEAR_4310_PATH = os.path.join(I2RT_ROOT, "robot_models/gripper/linear_4310/linear_4310.xml")
GRIPPER_TEACHING_HANDLE_PATH = os.path.join(
    I2RT_ROOT, "robot_models/gripper/yam_teaching_handle/yam_teaching_handle.xml"
)
GRIPPER_NO_GRIPPER_PATH = os.path.join(I2RT_ROOT, "robot_models/gripper/no_gripper/no_gripper.xml")


def combine_arm_and_gripper_xml(
    arm_path: str,
    gripper_path: str,
    ee_mass: Optional[float] = None,
    ee_inertia: Optional[np.ndarray] = None,
) -> str:
    """Combine arm and gripper XML files into a single XML string.

    Replaces the <body name="link_6"> subtree in the arm XML with the one from the
    gripper XML (if present). If ee_mass or ee_inertia are provided, update the
    inertial properties of the resulting link_6. Returns path to combined XML in /tmp/.

    Args:
        arm_path: Path to the arm MuJoCo XML file.
        gripper_path: Path to the gripper MuJoCo XML file. If falsy, the arm XML
            is used as-is (no gripper replacement).
        ee_mass: Optional end-effector mass (kg) to override in link_6's inertial.
        ee_inertia: Optional end-effector inertia array. Expected as a flat array of
            10 elements: [ipos(3), quat(4), diaginertia(3)].

    Returns:
        Path to the combined XML file written to /tmp/.
    """
    arm_tree = ET.parse(arm_path)
    arm_root = arm_tree.getroot()

    # Resolve arm mesh paths to absolute
    arm_dir = os.path.dirname(os.path.abspath(arm_path))
    arm_compiler = arm_root.find("compiler")
    arm_meshdir = arm_compiler.get("meshdir", "") if arm_compiler is not None else ""
    arm_asset = arm_root.find("asset")
    if arm_asset is not None:
        for child in arm_asset:
            if child.get("file") and not os.path.isabs(child.get("file")):
                abs_file = os.path.join(arm_dir, arm_meshdir, child.get("file"))
                child.set("file", os.path.abspath(abs_file))

    # Remove meshdir from compiler (all paths now absolute)
    if arm_compiler is not None and arm_compiler.get("meshdir"):
        del arm_compiler.attrib["meshdir"]

    # attempt to load gripper and replace link_6 if available
    if gripper_path:
        try:
            grip_tree = ET.parse(gripper_path)
            grip_root = grip_tree.getroot()
            grip_body = grip_root.find(".//body[@name='link_6']")
            if grip_body is None:
                grip_body = grip_root.find(".//body[@name='link6']")
        except Exception:
            grip_root = None
            grip_body = None

        # merge assets (avoid duplicates), resolving gripper mesh paths to absolute
        if grip_root is not None:
            grip_dir = os.path.dirname(os.path.abspath(gripper_path))
            grip_compiler = grip_root.find("compiler")
            grip_meshdir = grip_compiler.get("meshdir", "") if grip_compiler is not None else ""

            grip_asset = grip_root.find("asset")
            if grip_asset is not None:
                if arm_asset is None:
                    arm_asset = ET.Element("asset")
                    worldbody = arm_root.find("worldbody")
                    if worldbody is not None:
                        arm_root.insert(list(arm_root).index(worldbody), arm_asset)
                    else:
                        arm_root.append(arm_asset)
                existing = {(c.tag, c.get("name")) for c in arm_asset}
                for child in grip_asset:
                    key = (child.tag, child.get("name"))
                    if key not in existing:
                        elem = deepcopy(child)
                        if elem.get("file") and not os.path.isabs(elem.get("file")):
                            abs_file = os.path.join(grip_dir, grip_meshdir, elem.get("file"))
                            elem.set("file", os.path.abspath(abs_file))
                        arm_asset.append(elem)
                        existing.add(key)

        # replace arm's link_6 with gripper's if found
        if grip_body is not None:
            replaced = False
            for parent in arm_root.iter():
                children = list(parent)
                for idx, child in enumerate(children):
                    if child.tag == "body" and child.get("name") in ("link_6", "link6"):
                        parent.remove(child)
                        parent.insert(idx, deepcopy(grip_body))
                        replaced = True
                        break
                if replaced:
                    break

        # merge optional top-level sections (equality, contact) from gripper
        if grip_root is not None:
            for section_tag in ("equality", "contact"):
                grip_section = grip_root.find(section_tag)
                if grip_section is None:
                    continue
                arm_section = arm_root.find(section_tag)
                if arm_section is None:
                    arm_section = ET.SubElement(arm_root, section_tag)
                for child in grip_section:
                    arm_section.append(deepcopy(child))

    # find resulting link_6 and apply end-effector overrides (mass/inertia)
    if ee_mass is not None or ee_inertia is not None:
        res_body = arm_root.find(".//body[@name='link_6']")
        if res_body is None:
            res_body = arm_root.find(".//body[@name='link6']")
        if res_body is not None:
            inertial = res_body.find("inertial")
            if inertial is None:
                inertial = ET.SubElement(res_body, "inertial")

            if ee_mass is not None:
                inertial.set("mass", str(float(ee_mass)))

            if ee_inertia is not None:
                arr = np.asarray(ee_inertia).ravel()
                ipos = " ".join(str(float(x)) for x in arr[:3])
                inertial.set("ipos", ipos)
                quat = " ".join(str(float(x)) for x in arr[3:7])
                inertial.set("quat", quat)
                diagin = " ".join(str(float(x)) for x in arr[-3:])
                inertial.set("diaginertia", diagin)

    # write combined xml to /tmp/ and return filepath
    out_path = tempfile.NamedTemporaryFile(suffix=".xml", prefix="i2rt_combined_", delete=False, dir="/tmp").name
    arm_tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


class ArmType(enum.Enum):
    YAM = "yam"
    YAM_PRO = "yam_pro"
    YAM_ULTRA = "yam_ultra"
    BIG_YAM = "big_yam"

    @classmethod
    def from_string_name(cls, name: str) -> "ArmType":
        try:
            return cls(name)
        except ValueError:
            raise ValueError(
                f"Unknown arm type: {name}, arm has to be one of the following: {ArmType.available_arms()}"
            ) from None

    @classmethod
    def available_arms(cls) -> List[str]:
        return [arm.value for arm in cls]

    def get_xml_path(self) -> str:
        _xml_map = {
            ArmType.YAM: ARM_YAM_XML_PATH,
            ArmType.YAM_PRO: ARM_YAM_PRO_XML_PATH,
            ArmType.YAM_ULTRA: ARM_YAM_ULTRA_XML_PATH,
            ArmType.BIG_YAM: ARM_BIG_YAM_XML_PATH,
        }
        if self not in _xml_map:
            raise ValueError(f"Unknown arm type: {self}")
        return _xml_map[self]


class GripperType(enum.Enum):
    CRANK_4310 = "crank_4310"  # a 4310 motor with a crank
    LINEAR_3507 = "linear_3507"  # a 3507 motor with a linear actuator
    LINEAR_4310 = "linear_4310"  # a 4310 motor with a linear actuator

    # technically not a gripper
    YAM_TEACHING_HANDLE = "yam_teaching_handle"
    NO_GRIPPER = "no_gripper"

    @classmethod
    def from_string_name(cls, name: str) -> "GripperType":
        try:
            return cls(name)
        except ValueError:
            raise ValueError(
                f"Unknown gripper type: {name!r}, must be one of: {GripperType.available_grippers()}"
            ) from None

    @classmethod
    def available_grippers(cls) -> List[str]:
        return [gripper.value for gripper in GripperType]

    def get_gripper_limits(self) -> Optional[tuple[float, float]]:
        if self == GripperType.CRANK_4310:
            return 0.0, -2.7
        return None

    def get_gripper_needs_calibration(self) -> bool:
        return self in (GripperType.LINEAR_3507, GripperType.LINEAR_4310)

    def get_xml_path(self) -> str:
        _xml_map = {
            GripperType.CRANK_4310: GRIPPER_CRANK_4310_PATH,
            GripperType.LINEAR_3507: GRIPPER_LINEAR_3507_PATH,
            GripperType.LINEAR_4310: GRIPPER_LINEAR_4310_PATH,
            GripperType.YAM_TEACHING_HANDLE: GRIPPER_TEACHING_HANDLE_PATH,
            GripperType.NO_GRIPPER: GRIPPER_NO_GRIPPER_PATH,
        }
        if self not in _xml_map:
            raise ValueError(f"Unknown gripper type: {self}")
        return _xml_map[self]

    def get_motor_kp_kd(self) -> tuple[float, float]:
        if self in (GripperType.CRANK_4310, GripperType.LINEAR_4310):
            return 20, 0.5
        elif self == GripperType.LINEAR_3507:
            return 10, 0.3
        elif self in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER):
            return -1.0, -1.0
        else:
            raise ValueError(f"Unknown gripper type: {self}")

    def get_motor_type(self) -> str:
        if self in (GripperType.CRANK_4310, GripperType.LINEAR_4310):
            return "DM4310"
        elif self == GripperType.LINEAR_3507:
            return "DM3507"
        elif self in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER):
            return ""
        else:
            raise ValueError(f"Unknown gripper type: {self}")

    def get_gripper_limiter_params(self) -> tuple[float, float, float, callable]:
        """
        clog_force_threshold: float,
        clog_speed_threshold: float,
        sign: float,
        gripper_force_torque_map: callable,
        """
        if self == GripperType.CRANK_4310:
            return (
                0.5,
                0.2,
                1.0,
                partial(
                    zero_linkage_crank_gripper_force_torque_map,
                    motor_reading_to_crank_angle=lambda x: -x + 0.174,
                    gripper_close_angle=8 / 180.0 * np.pi,
                    gripper_open_angle=170 / 180.0 * np.pi,
                    gripper_stroke=0.071,  # unit in meter
                ),
            )
        elif self == GripperType.LINEAR_3507:
            return (
                0.5,
                0.3,
                1.0,
                partial(
                    linear_gripper_force_torque_map,
                    motor_stroke=6.57,
                    gripper_stroke=0.096,
                ),
            )
        elif self == GripperType.LINEAR_4310:
            return (
                0.5,
                0.3,
                1.0,
                partial(
                    linear_gripper_force_torque_map,
                    motor_stroke=6.57,
                    gripper_stroke=0.096,
                ),
            )
        elif self in (GripperType.YAM_TEACHING_HANDLE, GripperType.NO_GRIPPER):
            return -1.0, -1.0, -1.0, None
        else:
            raise ValueError(f"Unknown gripper type: {self}")


class JointMapper:
    def __init__(self, index_range_map: Dict[int, Tuple[float, float]], total_dofs: int):
        """_summary_
        This class is used to map the joint positions from the command space to the robot joint space.

        Args:
            index_range_map (Dict[int, Tuple[float, float]]): 0 indexed
            total_dofs (int): num of joints in the robot including the gripper if the girpper is the second robot
        """
        self.empty = len(index_range_map) == 0
        if not self.empty:
            self.joints_one_hot = np.zeros(total_dofs).astype(bool)
            self.joint_limits = []
            for idx, (start, end) in index_range_map.items():
                self.joints_one_hot[idx] = True
                self.joint_limits.append((start, end))
            self.joint_limits = np.array(self.joint_limits)
            self.joint_range = self.joint_limits[:, 1] - self.joint_limits[:, 0]

    def to_robot_joint_pos_space(self, command_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_pos
        command_joint_pos = np.asarray(command_joint_pos, order="C")
        result = command_joint_pos.copy()
        needs_remapping = command_joint_pos[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range + self.joint_limits[:, 0]
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_robot_joint_vel_space(self, command_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return command_joint_vel
        result = command_joint_vel.copy()
        needs_remapping = command_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping * self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_vel_space(self, robot_joint_vel: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_vel
        result = robot_joint_vel.copy()
        needs_remapping = robot_joint_vel[self.joints_one_hot]
        needs_remapping = needs_remapping / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result

    def to_command_joint_pos_space(self, robot_joint_pos: np.ndarray) -> np.ndarray:
        if self.empty:
            return robot_joint_pos
        result = robot_joint_pos.copy()
        needs_remapping = robot_joint_pos[self.joints_one_hot]
        needs_remapping = (needs_remapping - self.joint_limits[:, 0]) / self.joint_range
        result[self.joints_one_hot] = needs_remapping
        return result


def linear_gripper_force_torque_map(
    motor_stroke: float, gripper_stroke: float, gripper_force: float, current_angle: float
) -> float:
    """Maps the motor stroke required to achieve a given gripper force.

    Args:
        motor_stroke (float): in rad
        gripper_stroke (float): in meter
        gripper_force (float): in newton
    """
    # force = torque * motor_stroke / gripper_stroke
    return gripper_force * gripper_stroke / motor_stroke


def zero_linkage_crank_gripper_force_torque_map(
    gripper_close_angle: float,
    gripper_open_angle: float,
    motor_reading_to_crank_angle: Callable[[float], float],
    gripper_stroke: float,
    current_angle: float,
    gripper_force: float,
) -> float:
    """Maps the motor crank torque required to achieve a given gripper force. For Yam style gripper (zero linkage crank)

    Args:
        gripper_close_angle (float): Angle of the crank in radians at the closed position.
        gripper_open_angle (float): Angle of the crank in radians at the open position.
        gripper_stroke (float): Linear displacement of the gripper in meters.
        current_angle (float): Current crank angle in radians (relative to the closed position).
        gripper_force (float): Required gripping force in Newtons (N).

    Returns:
        float: Required motor torque in Newton-meters (Nm).
    """
    current_angle = motor_reading_to_crank_angle(current_angle)
    # Compute crank radius based on the total stroke and angle change
    crank_radius = gripper_stroke / (2 * (np.cos(gripper_close_angle) - np.cos(gripper_open_angle)))
    # gripper_position = crank_radius * (np.cos(gripper_close_angle) - np.cos(current_angle))
    grad_gripper_position = crank_radius * np.sin(current_angle)

    # Compute the required torque
    target_torque = gripper_force * grad_gripper_position
    return target_torque


class LockFreeCircularBuffer:
    """
    Lock-free circular buffer.
    There is a ~microsecond level race condition for this, but we're only using it to tell if the gripper is clogged or not.
    So 1 stale reading out of 1000 is not a big deal (FOR THAT PARTICULAR USE CASE!!!).
    """

    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.timestamps = np.zeros(maxsize)
        self.values = np.zeros(maxsize)
        self.write_idx = 0

    def put(self, timestamp: float, value: float) -> None:
        """Add a timestamped value to the buffer."""
        idx = self.write_idx % self.maxsize
        self.timestamps[idx] = timestamp
        self.values[idx] = value
        self.write_idx += 1

    def get_recent_values(self, time_window: float, current_time: Optional[float] = None) -> np.ndarray:
        """Get values within the specified time window."""
        if current_time is None:
            current_time = time.time()

        valid_mask = self.timestamps > (current_time - time_window)
        return self.values[valid_mask]


class GripperForceLimiter:
    def __init__(
        self,
        max_force: float,
        gripper_type: GripperType,
        kp: float,
        average_torque_window: float = 0.1,  # in seconds
        debug: bool = False,
    ):
        self.max_force = max_force
        self.gripper_type = gripper_type
        self._is_clogged = False
        self._gripper_adjusted_qpos = None
        self._kp = kp
        self._past_gripper_effort_buffer = LockFreeCircularBuffer(maxsize=1000)
        self.average_torque_window = average_torque_window
        self.debug = debug
        (self.clog_force_threshold, self.clog_speed_threshold, self.sign, _gripper_force_torque_map) = (
            self.gripper_type.get_gripper_limiter_params()
        )
        self.gripper_force_torque_map = partial(
            _gripper_force_torque_map,
            gripper_force=self.max_force,
        )

    def compute_target_gripper_torque(self, gripper_state: Dict[str, float]) -> float:
        current_speed = gripper_state["current_qvel"]
        relevant_history_effort = self._past_gripper_effort_buffer.get_recent_values(self.average_torque_window)
        if len(relevant_history_effort) > 0:
            average_effort = np.abs(np.mean(relevant_history_effort))
        else:
            average_effort = 0.0

        if self.debug:
            print(f"average_effort: {average_effort}")

        if self._is_clogged:
            normalized_current_qpos = gripper_state["current_normalized_qpos"]
            normalized_target_qpos = gripper_state["target_normalized_qpos"]
            # 0 close 1 open
            if (normalized_current_qpos < normalized_target_qpos) or average_effort < 0.2:  # want to open
                self._is_clogged = False
        elif average_effort > self.clog_force_threshold and np.abs(current_speed) < self.clog_speed_threshold:
            self._is_clogged = True

        if self._is_clogged:
            target_eff = self.gripper_force_torque_map(current_angle=gripper_state["current_qpos"])
            self._is_clogged = True
            return target_eff + 0.3  # this is to compensate the friction
        else:
            return None

    def update(self, gripper_state: Dict[str, float]) -> None:
        current_ts = time.time()
        self._past_gripper_effort_buffer.put(current_ts, gripper_state["current_eff"])
        target_eff = self.compute_target_gripper_torque(gripper_state)

        if target_eff is not None:
            command_sign = np.sign(gripper_state["target_qpos"] - gripper_state["current_qpos"]) * self.sign
            current_zero_eff_pos = (
                gripper_state["last_command_qpos"] - command_sign * np.abs(gripper_state["current_eff"]) / self._kp
            )
            target_gripper_raw_pos = current_zero_eff_pos + command_sign * np.abs(target_eff) / self._kp
            if self.debug:
                print("clogged")
                print(f"gripper_state: {gripper_state}")
                print("current zero eff")
                print(current_zero_eff_pos)
                print(f"target_gripper_raw_pos: {target_gripper_raw_pos}")
            # Update gripper target position
            a = 0.1
            if self._gripper_adjusted_qpos is None:  # initialize it to the target position
                self._gripper_adjusted_qpos = target_gripper_raw_pos
            self._gripper_adjusted_qpos = (1 - a) * self._gripper_adjusted_qpos + a * target_gripper_raw_pos
            return self._gripper_adjusted_qpos
        else:
            if self.debug:
                print("unclogged")
            self._gripper_adjusted_qpos = gripper_state["current_qpos"]
            return gripper_state["target_qpos"]


def detect_gripper_limits(
    motor_chain: DMChainCanInterface,
    gripper_index: int = 6,
    test_torque: float = 0.2,
    max_duration: float = 2.0,
    position_threshold: float = 0.01,
    check_interval: float = 0.1,
) -> List[float]:
    """
    Detect gripper limits by applying test torques and monitoring position changes.

    Args:
        motor_chain: Motor chain interface
        gripper_index: Index of gripper motor
        test_torque: Test torque for gripper detection (Nm)
        max_duration: Maximum test duration for each direction (s)
        position_threshold: Minimum position change to consider motor still moving (rad)
        check_interval: Time interval between checks (s)

    Returns:
        List of detected limits [limit1, limit2]
    """
    logger = logging.getLogger(__name__)
    positions = []
    num_motors = len(motor_chain.motor_list)
    zero_torques = np.zeros(num_motors)

    # Get motor direction for the gripper
    motor_direction = motor_chain.motor_direction[gripper_index]

    # Record initial position
    initial_states = motor_chain.read_states()
    init_torque = np.array([state.eff for state in initial_states])
    initial_pos = initial_states[gripper_index].pos
    positions.append(initial_pos)
    logger.info(f"Gripper calibration starting from position: {initial_pos:.4f}")

    # Test both directions
    for direction in [1, -1]:
        logger.info(f"Testing gripper direction: {direction}")
        test_torques = init_torque
        test_torques[gripper_index] = direction * test_torque

        start_time = time.time()
        last_pos = None
        position_stable_count = 0

        while time.time() - start_time < max_duration:
            motor_chain.set_commands(torques=test_torques)
            time.sleep(check_interval)

            states = motor_chain.read_states()
            current_pos = states[gripper_index].pos
            positions.append(current_pos)

            # Check if position has stopped changing (gripper hit limit)
            if last_pos is not None:
                pos_change = abs(current_pos - last_pos)
                if pos_change < position_threshold:
                    position_stable_count += 1
                else:
                    position_stable_count = 0

                # Check if gripper has hit limit (position stable)
                if position_stable_count >= 3:
                    logger.info(f"Gripper limit detected: pos={current_pos:.4f}")
                    break

            last_pos = current_pos

        time.sleep(0.3)

    # Calculate detected limits
    min_pos = min(positions)
    max_pos = max(positions)

    # Order based on motor direction
    if motor_direction > 0:
        # Positive direction: [max, min]
        detected_limits = [max_pos, min_pos]
    else:
        # Negative direction: [min, max]
        detected_limits = [min_pos, max_pos]

    logger.info(f"Motor direction: {motor_direction}, detected limits: {detected_limits}")
    return detected_limits
