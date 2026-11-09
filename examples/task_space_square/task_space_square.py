"""Trace a 15 cm square in the xz plane with one or two YAM arms in lockstep."""

import time
from itertools import pairwise

import mink
import numpy as np
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml


def channels(arm_left: str | None = None, arm_right: str | None = None) -> list[str]:
    """Name one arm to run it alone; name neither to run both on can0/can1."""
    return [c for c in (arm_left, arm_right) if c] or ["can0", "can1"]


ARM, GRIPPER, SITE = ArmType.YAM, GripperType.LINEAR_4310, "grasp_site"
READY = [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
SIZE, SPEED, DT, HOLD = 0.15, 0.05, 0.02, 2.0

xml_path = combine_arm_and_gripper_xml(ARM, GRIPPER)
kin = Kinematics(xml_path, SITE)
model = kin._configuration.model
limits = [mink.ConfigurationLimit(model)]
seed = np.r_[READY, np.zeros(model.nq - 6)]

# Solve the whole trajectory first, so an unreachable corner shows up before the arm moves.
target = kin.fk(seed)
corners = target[:3, 3] + SIZE / 2 * np.array([[1, 0, 1], [-1, 0, 1], [-1, 0, -1], [1, 0, -1], [1, 0, 1]])
q, traj = seed, []
for a, b in pairwise(corners):
    for p in np.linspace(a, b, int(SIZE / (SPEED * DT))):
        target[:3, 3] = p
        ok, q = kin.ik(target, SITE, init_q=q, limits=limits)
        assert ok, f"IK failed at {np.round(p, 4)}"
        traj.append(q[:6].copy())
    traj += [traj[-1]] * int(HOLD / DT)

robots = [get_yam_robot(channel=ch, arm_type=ARM, gripper_type=GRIPPER) for ch in tyro.cli(channels)]
try:
    homes = [r.get_joint_pos() for r in robots]
    # Ramp in through joint space: the home pose sits on joint2/joint3's lower bound.
    for r, home in zip(robots, homes, strict=True):
        r.move_joints(np.r_[traj[0], home[6:]], time_interval_s=4.0)
    time.sleep(HOLD)
    for q6 in traj:
        for r, home in zip(robots, homes, strict=True):
            r.command_joint_pos(np.r_[q6, home[6:]])
        time.sleep(DT)
    for r, home in zip(robots, homes, strict=True):
        r.move_joints(home, time_interval_s=4.0)
finally:
    for r in robots:
        r.close()
