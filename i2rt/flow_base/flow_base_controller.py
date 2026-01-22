# Modified from https://github.com/jimmyyhwu/tidybot2
import os

from i2rt.robots.robot import Robot, RobotType

os.environ["CTR_TARGET"] = "Hardware"  # pylint: disable=wrong-import-position

import atexit
import logging
import math
import os
import queue
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import portal
from dm_env.specs import Array
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig
from threadpoolctl import threadpool_limits

from i2rt.flow_base.linear_rail_controller import (
    LinearRailController,
    SingleMotorControlInterface,
    initialize_brake_gpio,
)
from i2rt.motor_drivers.dm_driver import ControlMode, DMChainCanInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("FlowBaseController")

BASE_DEFAULT_PORT = 11323
POLICY_CONTROL_FREQ = 10
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
h_x, h_y = 0.2 * np.array([1.0, 1.0, -1.0, -1.0]), 0.2 * np.array([-1.0, 1.0, 1.0, -1.0])


def remove_pid_file(pid_file_path: str) -> None:
    # Remove PID file if it corresponds to the current process
    if pid_file_path.exists():
        with open(pid_file_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
        if pid == os.getpid():
            pid_file_path.unlink()


def create_pid_file(name: str) -> None:
    # Check if PID file already exists
    pid_file_path = Path(f"/tmp/{name}.pid")
    if pid_file_path.exists():
        # Get PID of other process from lock file
        with open(pid_file_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())

        # Check if PID matches current process
        if pid != os.getpid():
            # PID does not match current process, check if other process is still running
            try:
                os.kill(pid, 0)
            except OSError:
                print(f"Removing stale PID file (PID {pid})")
                pid_file_path.unlink()
            else:
                raise Exception(f"Another instance of the {name} is already running (PID {pid})")

    # Write PID of current process to the file
    pid_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file_path, "w", encoding="utf-8") as f:
        f.write(f"{os.getpid()}\n")

    # Register cleanup function to remove PID file upon exit
    atexit.register(remove_pid_file, pid_file_path)


# Vehicle
CONTROL_FREQ = 200
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms
NUM_CASTERS = 4

# Caster
b_x = -0.020
b_y = -0.0  # Lateral caster offset (m)
r = 0.05  # Wheel radius (m)
N_s = 1  # Steer gear ratio
N_r1 = 1  # Drive gear ratio (1st stage)
N_r2 = 1  # Drive gear ratio (2nd stage)
N_w = 1  # Wheel gear ratio
N_r1_r2_w = N_r1 * N_r2 * N_w
N_s_r2_w = N_s * N_r2 * N_w
TWO_PI = 2 * math.pi


class VehicleMotorController:
    def __init__(
        self,
        steering_offset: List[float],
        steering_direction: List[int],
        channel_name_or_motor_interface: str | DMChainCanInterface = "can_flow_base",
        num_casters: int = 4,
        homing_check_callback: Optional[callable] = None,
    ):
        self.num_casters = num_casters
        if isinstance(channel_name_or_motor_interface, str):
            self.motor_interface = self._initialize_motor_chain(
                channel_name_or_motor_interface, steering_offset, steering_direction
            )
        else:
            self.motor_interface = channel_name_or_motor_interface

        self.motor_offsets = self.motor_interface.motor_offset
        self.motor_directions = self.motor_interface.motor_direction
        self.kd = np.array(
            [
                2,
                2,
            ]
            * self.num_casters
        )
        self.homing_check_callback = homing_check_callback  # Callback to check if homing is in progress

        print(f"dm chain can interface: {self.motor_interface} initialized")

    def _initialize_motor_chain(
        self, channel: str, steering_offset: List[float], steering_direction: List[int]
    ) -> DMChainCanInterface:
        motor_list = []
        motor_offsets = []

        motor_directions = []
        for caster_idx in [1, 2, 3, 0]:
            motor_offsets.append(steering_offset[caster_idx])
            motor_offsets.append(0)  # drive motor no need to set offset
            motor_directions.append(steering_direction[caster_idx])
            motor_directions.append((-1) ** (caster_idx))  # drive motor direction is always 1

            caster_idx = caster_idx + 1
            steering_motor_id = caster_idx * 2 - 1
            drive_motor_id = caster_idx * 2
            motor_list.append([steering_motor_id, "DM4310V"])
            motor_list.append([drive_motor_id, "DM_FLOW_WHEEL"])

        motor_interface = DMChainCanInterface(
            motor_list,
            motor_offsets,
            motor_directions,
            channel=channel,
            motor_chain_name="holonomic_base",
            control_mode=ControlMode.VEL,
        )
        return motor_interface

    def get_state(self) -> Dict[str, Any]:
        motor_states = self.motor_interface.read_states()
        steer_pos, drive_pos = [], []
        steer_vel, drive_vel = [], []
        for idx in range(self.num_casters):
            steer_idx = idx * 2
            drive_idx = idx * 2 + 1
            steer_pos.append(motor_states[steer_idx].pos)
            drive_pos.append(motor_states[drive_idx].pos)
            steer_vel.append(motor_states[steer_idx].vel)
            drive_vel.append(motor_states[drive_idx].vel)
        result_dict = {
            "steer_pos": steer_pos,
            "drive_pos": drive_pos,
            "steer_vel": steer_vel,
            "drive_vel": drive_vel,
        }
        return result_dict

    def get_positions(self) -> List[float]:
        steer_pos, drive_pos, _, _ = self.get_state()
        return steer_pos + drive_pos

    def get_velocities(self) -> List[float]:
        _, _, steer_vel, drive_vel = self.get_state()
        return steer_vel + drive_vel

    def set_velocities(self, input_dict: Dict[str, Any]) -> None:
        steer_vel, drive_vel = input_dict["steer_vel"], input_dict["drive_vel"]
        num_motors_in_chain = len(self.motor_interface)
        num_base_motors = 2 * self.num_casters

        # Build base motor velocities (steer and drive alternating)
        vels = np.zeros(num_motors_in_chain)
        for i in range(self.num_casters):
            vels[i * 2] = steer_vel[i]  # Steer motor
            vels[i * 2 + 1] = drive_vel[i]  # Drive motor

        if num_motors_in_chain > num_base_motors:
            with self.motor_interface.command_lock:
                current_commands = self.motor_interface.commands
                if current_commands and len(current_commands) == num_motors_in_chain:
                    vels[num_base_motors:] = [cmd.vel for cmd in current_commands[num_base_motors:]]
                elif self.homing_check_callback is not None:
                    try:
                        if self.homing_check_callback():
                            logger.warning(
                                "Linear rail homing in progress but current_commands unavailable. "
                                "Linear rail velocity may be set to zero."
                            )
                    except Exception as e:
                        logger.warning(f"Error checking homing status: {e}")

        self.motor_interface.set_commands(
            torques=np.zeros(num_motors_in_chain),
            pos=np.zeros(num_motors_in_chain),
            vel=vels,
            kp=np.zeros(num_motors_in_chain),
            kd=2.0 * np.ones(num_motors_in_chain),
            get_state=False,
        )

    def set_neutral(self) -> None:
        num_motors_in_chain = len(self.motor_interface)
        self.motor_interface.set_commands(
            torques=np.zeros(num_motors_in_chain),
            pos=np.zeros(num_motors_in_chain),
            vel=np.zeros(num_motors_in_chain),
            kp=np.zeros(num_motors_in_chain),
            kd=0.5 * np.ones(num_motors_in_chain),
        )


class CommandType(Enum):
    POSITION = "position"
    VELOCITY = "velocity"


# Currently only used for velocity commands
class FrameType(Enum):
    GLOBAL = "global"
    LOCAL = "local"


class Vehicle(Robot):
    def __init__(
        self,
        max_vel: Tuple[float, float, float] = (0.5, 0.5, 1.57),
        max_accel: Tuple[float, float, float] = (0.25, 0.25, 0.79),
        channel: str | DMChainCanInterface = "can_flow_base",
        auto_start: bool = True,
    ):
        self.max_vel = np.array(max_vel)
        self.max_accel = np.array(max_accel)

        # Use PID file to enforce single instance
        create_pid_file("base-controller")

        # Initialize hardware module
        steering_offset = [0.0, 0.0, 0.0, 0.0]
        steering_direction = [1, 1, 1, 1]
        self.num_casters = len(steering_offset)

        self.caster_module_controller = VehicleMotorController(steering_offset, steering_direction, channel)

        # Joint space
        num_motors = 2 * NUM_CASTERS
        self.q = np.zeros(num_motors)
        self.dq = np.zeros(num_motors)

        # Operational space (global frame)
        self._lock = threading.Lock()
        self.num_dofs = 3  # (x, y, theta)
        self.x = np.zeros(self.num_dofs)
        self.dx = np.zeros(self.num_dofs)

        # C matrix relating operational space velocities to joint velocities
        self.C = np.zeros((num_motors, self.num_dofs))
        self.C_steer = self.C[::2]
        self.C_drive = self.C[1::2]

        # C_p matrix relating operational space velocities to wheel velocities at the contact points
        self.C_p = np.zeros((num_motors, self.num_dofs))
        self.C_p_steer = self.C_p[::2]
        self.C_p_drive = self.C_p[1::2]
        self.C_p_steer[:, :2] = [1.0, 0.0]
        self.C_p_drive[:, :2] = [0.0, 1.0]

        # C_qp^# matrix relating joint velocities to operational space velocities
        self.C_pinv = np.zeros((num_motors, self.num_dofs))
        self.CpT_Cqinv = np.zeros((self.num_dofs, num_motors))
        self.CpT_Cqinv_steer = self.CpT_Cqinv[:, ::2]
        self.CpT_Cqinv_drive = self.CpT_Cqinv[:, 1::2]

        # OTG (online trajectory generation)
        # Note: It would be better to couple x and y using polar coordinates
        self.otg = Ruckig(self.num_dofs, CONTROL_PERIOD)
        self.otg_inp = InputParameter(self.num_dofs)
        self.otg_out = OutputParameter(self.num_dofs)
        self.otg_res = Result.Working
        self.otg_inp.max_velocity = self.max_vel
        self.otg_inp.max_acceleration = self.max_accel

        # Control loop
        self.command_queue = queue.Queue(1)
        self.control_loop_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_loop_running = False
        if auto_start:
            self.start_control()

    def update_state(self) -> None:
        # Joint positions and velocities
        now = time.time()
        state_dict = self.caster_module_controller.get_state()
        steer_pos, drive_pos, steer_vel, drive_vel = (
            state_dict["steer_pos"],
            state_dict["drive_pos"],
            state_dict["steer_vel"],
            state_dict["drive_vel"],
        )
        for i in range(self.num_casters):
            self.q[i * 2] = steer_pos[i]
            self.q[i * 2 + 1] = drive_pos[i]
            self.dq[i * 2] = steer_vel[i]
            self.dq[i * 2 + 1] = drive_vel[i]

        q_steer = self.q[::2]
        s = np.sin(q_steer)
        c = np.cos(q_steer)

        # C matrix
        self.C_steer[:, 0] = s / b_x
        self.C_steer[:, 1] = -c / b_x
        self.C_steer[:, 2] = (-h_x * c - h_y * s) / b_x - 1.0
        self.C_drive[:, 0] = c / r - b_y * s / (b_x * r)
        self.C_drive[:, 1] = s / r + b_y * c / (b_x * r)
        self.C_drive[:, 2] = (h_x * s - h_y * c) / r + b_y * (h_x * c + h_y * s) / (b_x * r)

        # C_p matrix
        self.C_p_steer[:, 2] = -b_x * s - b_y * c - h_y
        self.C_p_drive[:, 2] = b_x * c - b_y * s + h_x

        # C_qp^# matrix
        self.CpT_Cqinv_steer[0] = b_x * s + b_y * c
        self.CpT_Cqinv_steer[1] = -b_x * c + b_y * s
        self.CpT_Cqinv_steer[2] = b_x * (-h_x * c - h_y * s - b_x) + b_y * (h_x * s - h_y * c - b_y)
        self.CpT_Cqinv_drive[0] = r * c
        self.CpT_Cqinv_drive[1] = r * s
        self.CpT_Cqinv_drive[2] = r * (h_x * s - h_y * c - b_y)
        with threadpool_limits(limits=1, user_api="blas"):  # Prevent excessive CPU usage
            self.C_pinv = np.linalg.solve(self.C_p.T @ self.C_p, self.CpT_Cqinv)

        # Odometry
        with self._lock:
            dx_local = self.C_pinv @ self.dq
            theta_avg = self.x[2] + 0.5 * dx_local[2] * CONTROL_PERIOD
            R = np.array(
                [
                    [math.cos(theta_avg), -math.sin(theta_avg), 0.0],
                    [math.sin(theta_avg), math.cos(theta_avg), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.dx = R @ dx_local
            self.x += self.dx * CONTROL_PERIOD
        time.sleep(0.0005)

    def start_control(self) -> None:
        if self.control_loop_thread is None:
            print("To initiate a new control loop, please create a new instance of Vehicle.")
            return
        self.control_loop_running = True
        self.control_loop_thread.start()

    def stop_control(self) -> None:
        self.control_loop_running = False
        if self.control_loop_thread is not None:
            self.control_loop_thread.join()
            self.control_loop_thread = None

    def control_loop(self) -> None:
        # Set real-time scheduling policy
        try:
            os.sched_setscheduler(
                0,
                os.SCHED_FIFO,
                os.sched_param(os.sched_get_priority_max(os.SCHED_FIFO)),
            )
        except PermissionError:
            print("Failed to set real-time scheduling policy, please edit /etc/security/limits.d/99-realtime.conf")

        disable_motors = True
        last_command_time = time.time()
        last_step_time = time.time()

        while self.control_loop_running:
            # Maintain the desired control frequency
            while time.time() - last_step_time < CONTROL_PERIOD:
                time.sleep(0.0001)
            curr_time = time.time()
            step_time = curr_time - last_step_time
            last_step_time = curr_time
            if step_time > 0.01:  # 10 ms
                logger.warning(f"Step time {1000 * step_time:.3f} ms in {self.__class__.__name__} control_loop")

            # Update state
            self.update_state()
            # Global to local frame conversion
            theta = self.x[2]
            R = np.array(
                [
                    [math.cos(theta), math.sin(theta), 0.0],
                    [-math.sin(theta), math.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            # Check for new command
            if not self.command_queue.empty():
                command = self.command_queue.get()
                last_command_time = time.time()
                target = command["target"]

                # Velocity command
                if command["type"] == CommandType.VELOCITY:
                    if command["frame"] == FrameType.LOCAL:
                        target = R.T @ target
                    self.otg_inp.control_interface = ControlInterface.Velocity
                    self.otg_inp.target_velocity = np.clip(target, -self.max_vel, self.max_vel)

                # Position command
                elif command["type"] == CommandType.POSITION:
                    self.otg_inp.control_interface = ControlInterface.Position
                    self.otg_inp.target_position = target
                    self.otg_inp.target_velocity = np.zeros_like(self.dx)

                self.otg_res = Result.Working
                disable_motors = False
            # Maintain current pose if command stream is disrupted
            if time.time() - last_command_time > 2.5 * POLICY_CONTROL_PERIOD:
                self.otg_inp.target_position = self.otg_out.new_position
                self.otg_inp.target_velocity = np.zeros_like(self.dx)
                self.otg_inp.current_velocity = self.dx  # Set this to prevent lurch when command stream resumes
                self.otg_res = Result.Working
                disable_motors = True

            # Slow down base during caster flip
            # Note: At low speeds, this section can be disabled for smoother movement
            if np.max(np.abs(self.dq[::2])) > 12.56:  # Steer joint speed > 720 deg/s
                if self.otg_inp.control_interface == ControlInterface.Position:
                    self.otg_inp.target_position = self.otg_out.new_position
                elif self.otg_inp.control_interface == ControlInterface.Velocity:
                    self.otg_inp.target_velocity = np.zeros_like(self.dx)

            # Update OTG
            if self.otg_res == Result.Working:
                self.otg_inp.current_position = self.x
                self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
                self.otg_out.pass_to_input(self.otg_inp)

            disable_motors = False
            if disable_motors:
                # Send motor neutral commands
                self.caster_module_controller.set_neutral()

            else:
                # Operational space velocity
                dx_d = self.otg_out.new_velocity

                dx_d_local = R @ dx_d

                # Joint velocities
                dq_d = self.C @ dx_d_local

                vel_dict = {
                    "steer_vel": np.asarray(dq_d[::2], order="C"),
                    "drive_vel": np.asarray(dq_d[1:][::2], order="C"),
                }
                self.caster_module_controller.set_velocities(vel_dict)

    def _enqueue_command(self, command_type: CommandType, target: Any, frame: Optional[FrameType] = None) -> None:
        if self.command_queue.full():
            print("Warning: Command queue is full. Is control loop running?")
        else:
            command = {"type": command_type, "target": target}
            if frame is not None:
                command["frame"] = FrameType(frame)
            self.command_queue.put(command, block=False)

    def get_odometry(self, input_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            return {
                "translation": self.x[:2],
                "rotation": self.x[2],
            }

    def reset_odometry(self, input_dict: Dict[str, Any] | None = None) -> None:
        with self._lock:
            self.x = np.zeros(self.num_dofs)
            self.dx = np.zeros(self.num_dofs)

    def set_target_velocity(self, velocity: Any, frame: str = "local") -> None:
        self._enqueue_command(CommandType.VELOCITY, velocity, frame)

    def set_target_position(self, position: Any) -> None:
        self._enqueue_command(CommandType.POSITION, position)

    def command_target_vel(self, joint_vel: np.ndarray) -> None:
        self.set_target_velocity(joint_vel)

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self.caster_module_controller.get_state()

    def get_robot_type(self) -> RobotType:
        return RobotType.MOBILE_BASE

    def joint_state_spec(self) -> Array:
        return Array(
            shape=(3,),
            dtype=np.float32,
        )

    def num_dofs(self) -> int:
        return self.num_dofs

    def running(self) -> bool:
        return self.caster_module_controller.motor_interface.running

    def close(self) -> None:
        """Clean up resources: stop control loop and set motors to neutral."""
        try:
            if self.control_loop_running:
                self.stop_control()
            if hasattr(self, "caster_module_controller"):
                self.caster_module_controller.set_neutral()
            logger.info("Vehicle closed successfully")
        except Exception as e:
            logger.error(f"Vehicle close error: {e}")


class LinearRailVehicle(Vehicle):
    def __init__(
        self,
        vehicle_max_vel: Tuple[float, float, float] = (0.5, 0.5, 1.57),
        vehicle_max_accel: Tuple[float, float, float] = (0.25, 0.25, 0.79),
        lift_max_vel: float = 14.0,
        channel: str = "can_linear_rail",
        auto_start: bool = True,
        lift_motor_id: int = 9,
        lift_motor_type: str = "DM8009",
        auto_home: bool = True,
        homing_timeout: float = 30.0,
        enable_linear_rail: bool = True,
    ):
        """
        Initialize LinearRailVehicle with optional linear rail lift module.

        Args:
            vehicle_max_vel: Maximum velocity for vehicle base (x, y, theta)
            vehicle_max_accel: Maximum acceleration for vehicle base (x, y, theta)
            lift_max_vel: Maximum velocity for linear rail lift (rad/s)
            channel: CAN channel name for motor communication
            auto_start: Whether to automatically start the control loop
            lift_motor_id: Motor ID for the linear rail motor
            lift_motor_type: Motor type (e.g., "DM4310", "DM8009")
            auto_home: Whether to automatically home the linear rail on initialization
            homing_timeout: Timeout for homing procedure (seconds)
            enable_linear_rail: Whether to enable linear rail. If False, only base (8 motors) will be initialized.
        """
        # Create base motor list (8 motors: 4 casters * 2 motors each)
        motor_list = []
        motor_offsets = []
        motor_directions = []

        steering_offset = [0.0, 0.0, 0.0, 0.0]
        steering_direction = [1, 1, 1, 1]

        for caster_idx in [1, 2, 3, 0]:
            motor_offsets.append(steering_offset[caster_idx])
            motor_offsets.append(0)  # drive motor no need to set offset
            motor_directions.append(steering_direction[caster_idx])
            motor_directions.append((-1) ** (caster_idx))  # drive motor direction

            caster_idx = caster_idx + 1
            steering_motor_id = caster_idx * 2 - 1
            drive_motor_id = caster_idx * 2
            motor_list.append([steering_motor_id, "DM4310V"])
            motor_list.append([drive_motor_id, "DM_FLOW_WHEEL"])

        # Conditionally add linear rail motor (9th motor)
        if enable_linear_rail:
            motor_list.append([lift_motor_id, lift_motor_type])
            motor_offsets.append(0.0)
            motor_directions.append(1)

        # Create unified motor chain
        unified_motor_chain = DMChainCanInterface(
            motor_list=motor_list,
            motor_offset=np.array(motor_offsets),
            motor_direction=np.array(motor_directions),
            channel=channel,
            motor_chain_name="linear_rail_vehicle" if enable_linear_rail else "holonomic_base",
            control_mode=ControlMode.VEL,
        )

        # Initialize brake GPIO only if linear rail is enabled
        if enable_linear_rail:
            initialize_brake_gpio()

        # Initialize vehicle base with the unified motor chain using super().__init__()
        super().__init__(
            max_vel=vehicle_max_vel,
            max_accel=vehicle_max_accel,
            channel=unified_motor_chain,  # Pass the unified motor chain
            auto_start=auto_start,
        )

        # Initialize linear rail only if enabled
        self.linear_rail = None
        if enable_linear_rail:
            # Create single motor control interface for the linear rail (9th motor, index 8)
            single_motor_interface = SingleMotorControlInterface.from_multi_motor_chain(
                unified_motor_chain, target_motor_idx=8
            )

            # Initialize linear rail controller (without auto_home to initialize GPIO first)
            self.linear_rail = LinearRailController(
                single_motor_control_interface=single_motor_interface,
                rail_speed=lift_max_vel,
                auto_home=False,  # Don't auto home yet, initialize GPIO first
                homing_timeout=homing_timeout,
            )

            # Initialize GPIO early, before starting homing
            self.linear_rail.initialize_gpio()

            # Now start homing if requested
            if auto_home:
                self.linear_rail._initialize_linear_rail()

            # Set homing check callback for VehicleMotorController to prevent overwriting homing velocity
            if hasattr(self, "caster_module_controller"):
                self.caster_module_controller.homing_check_callback = (
                    lambda: self.linear_rail.is_homing() if self.linear_rail else False
                )

    def set_target_velocity(self, velocity: Any, frame: str = "local") -> None:
        """Set target velocity for both base and linear rail.

        Args:
            velocity: Target velocity. Can be:
                - 3D array [x, y, theta] for base only
                - 4D array [x, y, theta, linear_rail_vel] for base + linear rail
            frame: Frame for base velocity ("local" or "global")
        """
        velocity = np.asarray(velocity)

        if velocity.shape == (3,):
            # Base only
            super().set_target_velocity(velocity, frame)
        elif velocity.shape == (4,):
            # Base + linear rail
            base_velocity = velocity[:3]
            linear_rail_velocity = velocity[3]
            super().set_target_velocity(base_velocity, frame)
            if self.linear_rail is not None:
                self.set_linear_rail_velocity(linear_rail_velocity)
        else:
            raise ValueError(
                f"Velocity must be 3D [x, y, theta] or 4D [x, y, theta, linear_rail_vel], got shape {velocity.shape}"
            )

    def get_linear_rail_state(self, input_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Get the current state of the linear rail.

        Args:
            input_dict: Optional dictionary (unused, for API compatibility)
        """
        if self.linear_rail is None:
            return {"error": "Linear rail not available"}
        state = self.linear_rail.get_state()
        if "motor_state" in state:
            state = {k: v for k, v in state.items() if k != "motor_state"}
        return state

    def set_linear_rail_velocity(self, velocity: float) -> None:
        """Set the velocity of the linear rail.

        Args:
            velocity: Target velocity in rad/s
        """
        if self.linear_rail is None:
            logger.warning("Linear rail not available, ignoring velocity command")
            return
        try:
            self.linear_rail.set_velocity(velocity)
        except AssertionError as e:
            logger.warning(f"Linear rail velocity command rejected: {e}")
        except Exception as e:
            logger.error(f"Failed to set linear rail velocity: {e}", exc_info=True)

    def close(self) -> None:
        """Clean up resources: stop linear rail and engage brake."""
        if hasattr(self, "linear_rail"):
            self.linear_rail.cleanup()
        super().close()


if __name__ == "__main__":
    import argparse
    import os
    import sys
    import time

    import pygame

    from i2rt.utils.gamepad_utils import Gamepad

    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, default="can0")
    parser.add_argument(
        "--no-linear-rail",
        action="store_true",
        help="Disable linear rail (use only 8 base motors)",
    )

    # Initialize pygame and joystick
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick/gamepad connected!")
        exit()

    joy = pygame.joystick.Joystick(0)
    CALIBRATION_RETRY_DELAY = 1
    DEADZONE = 0.05  # Deadzone for base control (x, y, theta)
    RAIL_DEADZONE = 0.15  # Larger deadzone for linear rail to prevent unwanted movement
    args = parser.parse_args()

    max_vel = np.array([0.8, 0.8, 3.0])
    max_accel = np.array([0.8, 0.8, 3.0])
    lift_max_vel = 14.0  # Maximum velocity for linear rail (rad/s)

    # Use LinearRailVehicle instead of Vehicle
    # Use --no-linear-rail flag if you only have base (8 motors) without linear rail
    vehicle = LinearRailVehicle(
        vehicle_max_vel=max_vel,
        vehicle_max_accel=max_accel,
        lift_max_vel=lift_max_vel,
        channel=args.channel,
        auto_home=True,
        enable_linear_rail=not args.no_linear_rail,  # Enable by default, disable with --no-linear-rail
    )

    # Register cleanup function to ensure brake is engaged on exit
    def close_vehicle() -> None:
        try:
            vehicle.close()
        except Exception as e:
            logger.error(f"Error during atexit close: {e}")

    atexit.register(close_vehicle)

    class TimeoutRemoteCommand:
        """Unified remote command handler for LinearRailVehicle (base + linear rail)"""

        def __init__(self, timeout: float = 0.2):
            self.timeout = timeout
            self.last_update_time = time.time() - 1000000
            self.command = np.zeros(4)  # Support 4D: [x, y, theta, linear_rail]
            self.frame = "local"
            self._lock = threading.Lock()

        def is_command_valid(self) -> bool:
            return time.time() - self.last_update_time < self.timeout

        def remote_set_target_velocity(self, input_dict: Dict[str, Any]) -> None:
            """Set target velocity for base (and optionally linear rail)"""
            target_velocity = input_dict["target_velocity"]
            frame = input_dict["frame"]
            with self._lock:
                # If 3D command, only update base part, preserve linear_rail value
                if len(target_velocity) == 3:
                    # Ensure command is 4D
                    if len(self.command) < 4:
                        self.command = np.append(self.command, 0.0) if len(self.command) == 3 else np.zeros(4)
                    # Update only base part [x, y, theta], preserve linear_rail
                    self.command[:3] = target_velocity
                else:
                    # 4D command: update everything
                    self.command = target_velocity
                self.frame = frame
                self.last_update_time = time.time()

        def get_command(self) -> Tuple[np.ndarray, str]:
            """Get base command [x, y, theta, linear_rail] and frame"""
            with self._lock:
                return self.command, self.frame

    remote_command = TimeoutRemoteCommand()

    # setup server for remote calls
    server = portal.Server(BASE_DEFAULT_PORT)
    server.bind("get_odometry", vehicle.get_odometry)
    server.bind("reset_odometry", vehicle.reset_odometry)
    server.bind("set_target_velocity", remote_command.remote_set_target_velocity)

    # Bind linear rail APIs if vehicle has linear rail
    if hasattr(vehicle, "linear_rail"):
        server.bind("get_linear_rail_state", vehicle.get_linear_rail_state)
        logger.info("Linear rail APIs bound to server")

    server.start(block=False)

    print(f"Joystick Name: {joy.get_name()}")
    print(f"Number of Axes: {joy.get_numaxes()}")
    print(f"Number of Buttons: {joy.get_numbuttons()}")

    # Check all x, y, th are 0 at the beginning, if not ask user to check joystick
    while True:
        # Pump events to update joystick state
        pygame.event.pump()
        four_axis = [joy.get_axis(1), joy.get_axis(0), joy.get_axis(2), joy.get_axis(3)]
        if all(np.abs(axis) < DEADZONE for axis in four_axis):
            logger.info("Joystick is at rest, please check joystick")
            break
        else:
            logger.warning(f"four_axis: {four_axis}")
            logger.warning("Joystick's rest position is not at the center, please check joystick")
            time.sleep(CALIBRATION_RETRY_DELAY)

    # Main loop to read joystick inputs
    gamepad = Gamepad()
    gamepad_command_frame = "local"
    gamepad_command_override = True

    last_gampad_mode_togged = False
    count = 0
    last_rail_log_time = time.time()
    RAIL_LOG_INTERVAL = 1.0  # Log linear rail position every 1 second
    try:
        while True:
            gamepad_cmd = gamepad.get_user_cmd()  # 3D: [x, y, theta]
            gamepad_button = gamepad.get_button_reading()

            if gamepad_button["key_mode"] and not last_gampad_mode_togged:
                last_gampad_mode_togged = True
                gamepad_command_frame = "global" if gamepad_command_frame == "local" else "local"
            else:
                last_gampad_mode_togged = False

            # Handle reset odometry (key_left_1)
            if gamepad_button["key_left_1"]:
                vehicle.reset_odometry()

            lift_vel = 0.0
            if joy.get_numaxes() > 3:
                right_stick_y = joy.get_axis(3)  # Right stick Y-axis
                # Apply larger deadzone for linear rail to prevent unwanted movement
                # Invert: up (negative axis value) = positive velocity
                if np.abs(right_stick_y) > RAIL_DEADZONE:
                    lift_vel = -right_stick_y  # Invert: up (negative axis) = positive velocity

            cmd_4d = np.append(gamepad_cmd, lift_vel)

            is_remote_command_valid = remote_command.is_command_valid()

            if is_remote_command_valid:
                user_cmd, user_frame = remote_command.get_command()
                gamepad_command_override = False

                if gamepad_button["key_left_2"]:
                    gamepad_command_override = True
            else:
                gamepad_command_override = True
            if not vehicle.running():
                print("Motor interface is not running, exiting...")
                print("Please check the E stop or the motor connection. ")
                break
            if gamepad_command_override:
                cmd = cmd_4d
                frame = gamepad_command_frame
            else:
                cmd = user_cmd
                frame = user_frame
            if count % 20 == 0:
                # print up 1 float point
                # print(f"frame: {frame}, cmd: {cmd[0]:.1f}, {cmd[1]:.1f}, {cmd[2]:.1f}, rail: {cmd[3]:.1f}")
                sys.stdout.write(f"\rframe: {frame} cmd: {cmd[0]:.1f} {cmd[1]:.1f} {cmd[2]:.1f} rail: {cmd[3]:.1f}")
                sys.stdout.flush()

            # Log linear rail position and velocity every 1 second
            current_time = time.time()
            if current_time - last_rail_log_time >= RAIL_LOG_INTERVAL:
                try:
                    if hasattr(vehicle, "linear_rail"):
                        rail_state = vehicle.get_linear_rail_state()
                        position = rail_state.get("position")
                        velocity = rail_state.get("velocity")
                        if position is not None and velocity is not None:
                            print(f"Linear rail - pos: {position:.4f} rad, vel: {velocity:.4f} rad/s")
                        else:
                            print(f"Linear rail - pos: {position}, vel: {velocity}")
                except Exception as e:
                    print(f"Failed to get linear rail state: {e}")
                last_rail_log_time = current_time

            count += 1

            # Set target velocity (supports both 3D and 4D)
            if len(cmd) == 4:
                # 4D: [x, y, theta, linear_rail] - cmd is already normalized [-1, 1]
                base_cmd = cmd[:3] * max_vel
                rail_cmd = cmd[3] * lift_max_vel
                vehicle.set_target_velocity(np.append(base_cmd, rail_cmd), frame=frame)
            else:
                # 3D: [x, y, theta] (backward compatibility)
                vehicle.set_target_velocity(cmd * max_vel, frame=frame)

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Ensure close is always called, even on Ctrl+C
        try:
            vehicle.close()
        except Exception as e:
            logger.error(f"Error during close: {e}")
        pygame.quit()
