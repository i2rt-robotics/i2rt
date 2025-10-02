#!/usr/bin/env python3
# minimum_gello.py — follower/leader/visualizer + optional leader-state telemetry server

import time
import threading
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import portal
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot

DEFAULT_ROBOT_PORT = 11333


class ServerRobot:
    """Portal server exposing a robot over RPC."""
    def __init__(self, robot: Robot, port: int | str):
        self._robot = robot
        self._server = portal.Server(str(port))
        print(f"[ServerRobot] Binding to port {port}, Robot: {robot}")

        # RPCs expected by clients
        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        self._server.start()


class ClientRobot(Robot):
    """Thin Portal client matching ServerRobot RPCs."""
    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self._client.get_observations().result()


class YAMLeaderRobot:
    """
    Wrapper providing leader-side info (teaching handle).
    Exposes: get_info() -> (qpos_with_gripper, button_inputs)
    """
    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> np.ndarray:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


class LeaderStateServer:
    """
    Read-only telemetry server exposing *leader (handle)* joint state so recorders
    can log actions without touching CAN.
    """
    def __init__(self, leader_robot: YAMLeaderRobot, port: int, host: str = "127.0.0.1"):
        self._leader = leader_robot
        self._server = portal.Server(f"{host}:{port}")
        print(f"[LeaderStateServer] Serving leader joints on {host}:{port}")

        # Minimal API expected by recorders
        self._server.bind("num_dofs", self._num_dofs)
        self._server.bind("get_joint_pos", self._get_joint_pos)
        self._server.bind("get_observations", self._get_observations)

    def _num_dofs(self) -> int:
        qpos, _ = self._leader.get_info()
        return int(len(qpos))

    def _get_joint_pos(self):
        qpos, _ = self._leader.get_info()
        return qpos

    def _get_observations(self):
        qpos, buttons = self._leader.get_info()
        return {"joint_pos": qpos, "buttons": buttons}

    def start(self) -> None:
        self._server.start()


@dataclass
class Args:
    gripper: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = "yam_teaching_handle"
    mode: Literal["follower", "leader", "visualizer_local", "visualizer_remote"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT              # follower server port (leader connects here)
    can_channel: str = "can0"
    bilateral_kp: float = 0.0

    # NEW: optional telemetry server port for leader state (actions).
    # If provided in leader mode, we start a read-only Portal server here.
    teleop_server_port: Optional[int] = None

def main(args: Args) -> None:
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper)

    if "remote" not in args.mode:
        robot = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type)
    if args.mode == "follower":
        server_robot = ServerRobot(robot, args.server_port)
        server_robot.serve()
    elif args.mode == "leader":
        robot = YAMLeaderRobot(robot)
        robot_current_kp = robot._robot._kp
        client_robot = ClientRobot(args.server_port, host=args.server_host)

        if args.teleop_server_port is not None:
            leader_state_server = LeaderStateServer(robot, port=args.teleop_server_port, host=args.server_host)
            # Start the server in a separate thread so it doesn't block
            server_thread = threading.Thread(target=leader_state_server.start, daemon=True)
            server_thread.start()
            print(f"[Leader] Telemetry server up on {args.server_host}:{args.teleop_server_port}")

        # sync the robot state
        current_joint_pos, current_button = robot.get_info()
        current_follower_joint_pos = client_robot.get_joint_pos()
        print(f"Current leader joint pos: {current_joint_pos}")
        print(f"Current follower joint pos: {current_follower_joint_pos}")

        def slow_move(joint_pos: np.ndarray, duration: float = 1.0) -> None:
            for i in range(100):
                current_joint_pos = joint_pos
                follower_command_joint_pos = current_joint_pos * i / 100 + current_follower_joint_pos * (1 - i / 100)
                client_robot.command_joint_pos(follower_command_joint_pos)
                time.sleep(0.03)

        synchronized = False
        while True:
            current_joint_pos, current_button = robot.get_info()
            if current_button[0] > 0.5:
                if not synchronized:
                    robot.update_kp_kd(kp=robot_current_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                    robot.command_joint_pos(current_joint_pos[:6])
                    slow_move(current_joint_pos)
                else:
                    print("clear bilateral pd")
                    robot.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                    robot.command_joint_pos(current_follower_joint_pos[:6])
                synchronized = not synchronized
                while current_button[0] > 0.5:
                    time.sleep(0.03)
                    current_joint_pos, current_button = robot.get_info()

            current_follower_joint_pos = client_robot.get_joint_pos()

            if synchronized:
                client_robot.command_joint_pos(current_joint_pos)
                # this will set the bilateral force in joint space proportional to the bilateral kp
                robot.command_joint_pos(current_follower_joint_pos[:6])

            time.sleep(0.01)
    elif "visualizer" in args.mode:
        import mujoco
        import mujoco.viewer
        if args.mode == "visualizer_remote":
            robot = ClientRobot(args.server_port, host=args.server_host)
        xml_path = gripper_type.get_xml_path()
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        dt: float = 0.01
        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                step_start = time.time()
                joint_pos = robot.get_joint_pos()
                data.qpos[:] = joint_pos[: model.nq]

                # sync the model state
                mujoco.mj_kinematics(model, data)
                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main(tyro.cli(Args))
