"""Run the YAM robot server on the robot machine (no ROS).

    python -m i2rt.serving.run_robot_server teleop  [--sim] [--bilateral-kp 0.15]
    python -m i2rt.serving.run_robot_server dagger  [--sim] [--mirror-kp 0.2]
    python -m i2rt.serving.run_robot_server wrapper [--sim]            # replay target

The workstation connects with :class:`i2rt.serving.robot_client.RobotClient`.
"""

from __future__ import annotations

import argparse
import logging

from i2rt.serving import control_config as cc
from i2rt.serving.controllers import (
    DaggerConfig,
    DaggerController,
    TeleopConfig,
    TeleopController,
    WrapperConfig,
    WrapperController,
)
from i2rt.serving.robot_server import DEFAULT_PORT, RobotServer


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="YAM robot server (portal, no ROS)")
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("teleop", help="auto home/engage bimanual teleop")
    pt.add_argument("--port", type=int, default=DEFAULT_PORT)
    pt.add_argument("--sim", action="store_true")
    pt.add_argument("--home", default="")
    pt.add_argument("--engage-thr", type=float, default=cc.ENGAGE_THR)
    pt.add_argument("--release-thr", type=float, default=cc.RELEASE_THR)
    pt.add_argument("--dwell", type=float, default=cc.DWELL_S)
    pt.add_argument("--home-kp", type=float, default=cc.HOME_KP)
    pt.add_argument("--bilateral-kp", type=float, default=cc.BILATERAL_KP)
    pt.add_argument("--rate", type=float, default=120.0)
    pt.add_argument("--ramp-speed", type=float, default=cc.RAMP_SPEED)
    pt.add_argument("--gate-joints", default=",".join(str(j) for j in cc.GATE_JOINTS))

    pd = sub.add_parser("dagger", help="HG-DAgger policy + button takeover")
    pd.add_argument("--port", type=int, default=DEFAULT_PORT)
    pd.add_argument("--sim", action="store_true")
    pd.add_argument("--mirror-kp", type=float, default=cc.DAGGER_MIRROR_KP)
    pd.add_argument("--feedback-kp", type=float, default=cc.DAGGER_FEEDBACK_KP)
    pd.add_argument("--rate", type=float, default=120.0)
    pd.add_argument("--max-joint-speed", type=float, default=1.5)

    pw = sub.add_parser("wrapper", help="followers track an external command (replay)")
    pw.add_argument("--port", type=int, default=DEFAULT_PORT)
    pw.add_argument("--sim", action="store_true")
    pw.add_argument("--arm-type", default="yam")
    pw.add_argument("--gripper", default="linear_4310")
    pw.add_argument("--rate", type=float, default=100.0)
    pw.add_argument("--max-joint-speed", type=float, default=1.5)
    pw.add_argument("--control", choices=["joint", "eef"], default="joint", help="command space (eef is experimental)")

    args = p.parse_args()

    if args.mode == "teleop":
        ctrl = TeleopController(
            TeleopConfig(
                sim=args.sim,
                home=args.home,
                engage_thr=args.engage_thr,
                release_thr=args.release_thr,
                dwell=args.dwell,
                home_kp=args.home_kp,
                bilateral_kp=args.bilateral_kp,
                rate=args.rate,
                ramp_speed=args.ramp_speed,
                gate_joints=args.gate_joints,
            )
        )
    elif args.mode == "dagger":
        ctrl = DaggerController(
            DaggerConfig(
                sim=args.sim,
                mirror_kp=args.mirror_kp,
                feedback_kp=args.feedback_kp,
                rate=args.rate,
                max_joint_speed=args.max_joint_speed,
            )
        )
    else:  # wrapper
        ctrl = WrapperController(
            WrapperConfig(
                sim=args.sim,
                arm_type=args.arm_type,
                gripper=args.gripper,
                rate=args.rate,
                max_joint_speed=args.max_joint_speed,
                control=args.control,
            )
        )

    RobotServer(ctrl, port=args.port, rate_hz=args.rate).serve()


if __name__ == "__main__":
    main()
