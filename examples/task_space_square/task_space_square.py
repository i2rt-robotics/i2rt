"""Trace a 15 cm square in the xz plane with a YAM end-effector."""

import time
from itertools import pairwise

import mink
import mujoco
import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

SITE = "grasp_site"
READY = np.array([0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
SIZE = 0.15
SPEED = 0.05
DT = 0.02
HOLD = 2.0

xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
model = mujoco.MjModel.from_xml_path(xml_path)
kin = Kinematics(xml_path, SITE)
limits = [mink.ConfigurationLimit(model)]

seed = np.zeros(model.nq)
seed[:6] = READY
pose = kin.fk(seed, SITE)
rot, center = pose[:3, :3], pose[:3, 3]

half = SIZE / 2
corners = [center + np.array([su * half, 0.0, sv * half]) for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))]


def solve(point: np.ndarray, q: np.ndarray) -> np.ndarray:
    target = np.eye(4)
    target[:3, :3] = rot
    target[:3, 3] = point
    ok, q = kin.ik(target, SITE, init_q=q, limits=limits)
    assert ok, f"IK failed at {np.round(point, 4)}"
    return q


# Solve the whole trajectory first, so an unreachable corner shows up before the arm moves.
q = solve(corners[0], seed)
start = q[:6].copy()
traj = [start] * int(HOLD / DT)
for a, b in pairwise(corners):
    n = max(int(np.ceil(np.linalg.norm(b - a) / (SPEED * DT))), 1)
    for i in range(1, n + 1):
        q = solve(a + (b - a) * (i / n), q)
        traj.append(q[:6].copy())
    traj.extend([traj[-1]] * int(HOLD / DT))

robot = get_yam_robot(channel="can0", arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310)
try:
    cmd = robot.get_joint_pos()
    # Ramp into the first corner in joint space: the home pose sits on joint2/joint3's lower bound.
    q0 = cmd[:6].copy()
    for i in range(1, 201):
        cmd[:6] = q0 + (start - q0) * (i / 200)
        robot.command_joint_pos(cmd)
        time.sleep(DT)
    for q6 in traj:
        cmd[:6] = q6
        robot.command_joint_pos(cmd)
        time.sleep(DT)
    for i in range(1, 201):
        cmd[:6] = traj[-1] + (q0 - traj[-1]) * (i / 200)
        robot.command_joint_pos(cmd)
        time.sleep(DT)
finally:
    robot.close()
