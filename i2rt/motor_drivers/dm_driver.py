import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

import can
import numpy as np
import tyro

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.motor_check import run_startup_checks
from i2rt.motor_drivers.utils import (
    FeedbackFrameInfo,
    MotorErrorCode,
    MotorInfo,
    MotorType,
    ReceiveMode,
    float_to_uint,
    uint_to_float,
)
from i2rt.utils.encoder_manager import EncoderConfig, PassiveJointEncoder
from i2rt.utils.utils import RateRecorder

log_level = os.getenv("LOGLEVEL", "ERROR").upper()

# if no log_level is set, set it to WARNING
logging.basicConfig(level=log_level)

# set control frequence
CONTROL_FREQ = 250
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms

EXPECTED_CONTROL_PERIOD = 0.007
REPORT_INTERVAL = 30.0

# How long close() waits for the control thread to leave its loop before shutting the CAN bus down.
# One cycle is a few ms, so this only ever expires if the thread is wedged on the bus.
_CONTROL_THREAD_JOIN_TIMEOUT = 2.0


class ControlMode:
    MIT = "MIT"
    POS_VEL = "POS_VEL"
    VEL = "VEL"

    @classmethod
    def get_id_offset(cls, control_mode: str) -> int:
        if control_mode == cls.MIT:
            return 0x000
        elif control_mode == cls.POS_VEL:
            return 0x100
        elif control_mode == cls.VEL:
            return 0x200
        else:
            raise ValueError(f"Control mode '{control_mode}' not recognized.")


######### for passive encoder #########
@dataclass
class PassiveEncoderInfo:
    """The encoder report."""

    id: int
    """The device number, uint8."""
    position: float
    """Position, in radian (encoder) or axis [-1, 1] center 0 (joystick analog)."""
    velocity: float
    """Velocity, in radian/s."""
    io_inputs: List[bool]
    """The discrete inputs, list of boolean."""


class PassiveEncoderReader:
    def __init__(
        self,
        can_interface: CanInterface,
        receive_mode: ReceiveMode = ReceiveMode.plus_one,
        range_rad: float = 0.7,
        encoder_config: EncoderConfig = None,  # type: ignore
    ):
        if encoder_config is None:
            encoder_config = EncoderConfig(adc_freq=255, report_freq=0, firmware=">=2.2.12")
        self.can_interface = can_interface
        # assert self.can_interface.use_buffered_reader, "Passive encoder reader must use buffered reader"
        self.range_rad = range_rad
        self.receive_mode = receive_mode
        # check the encoder config, the report frequency must be set to 0 for passive mode
        # Keep the validate_encoders result so callers can read per-encoder
        # firmware versions without re-probing the CAN bus (ROB-1311).
        self.encoder_info: Dict[int, Dict[str, Any]] = PassiveJointEncoder.validate_encoders(
            self.can_interface.channel, encoder_config
        )

    def read_encoder(self, encoder_id: int) -> PassiveEncoderInfo:
        # this encoder's trigger message is 0x02
        data = [0xFF, 0x02]
        message = self.can_interface._send_message_get_response(
            encoder_id, encoder_id, data, expected_id=self.receive_mode.get_receive_id(0x50E), max_retry=15
        )
        pos, vel, button_state = self._parse_encoder_message(message)
        pos_range = [-self.range_rad, self.range_rad]
        pos = np.clip(pos, pos_range[0], pos_range[1])
        # normalize pos to 1 - 0
        delta = np.abs(0.0 - pos)
        pos = delta / self.range_rad
        result = PassiveEncoderInfo(id=encoder_id, position=pos, velocity=vel, io_inputs=button_state)
        return result

    def _parse_encoder_message(self, message: can.Message) -> PassiveEncoderInfo:
        # Standard format
        struct_format = "!B h h B"
        _device_id, position, velocity, digital_inputs = struct.unpack(struct_format, message.data)

        # Convert position and velocity to radians
        position_rad = position * 2 * np.pi / 4096
        velocity_rad = velocity * 2 * np.pi / 4096
        button_state = [digital_inputs % 2, digital_inputs // 2]

        return position_rad, velocity_rad, button_state


class EncoderChain:
    def __init__(self, encoder_ids: List[int], encoder_interface: PassiveEncoderReader):
        self.encoder_ids = encoder_ids
        self.encoder_interface = encoder_interface

    def read_states(self) -> List[PassiveEncoderInfo]:
        return [self.encoder_interface.read_encoder(encoder_id) for encoder_id in self.encoder_ids]


class DMSingleMotorCanInterface(CanInterface):
    """Class for CAN interface with a single motor."""

    def __init__(
        self,
        control_mode: str = ControlMode.MIT,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "socketcan",
        bitrate: int = 1000000,
        receive_mode: ReceiveMode = ReceiveMode.p16,
        name: str = "default_can_DM_interface",
        use_buffered_reader: bool = False,
    ):
        super().__init__(
            channel, bustype, bitrate, receive_mode=receive_mode, name=name, use_buffered_reader=use_buffered_reader
        )
        self.control_mode = control_mode
        self.cmd_idoffset = ControlMode.get_id_offset(self.control_mode)
        self.receive_mode = receive_mode

    def _get_frame_id(self, motor_id: int) -> int:
        """Calculate the Control Frame ID for a given motor."""
        return self.cmd_idoffset + motor_id

    def motor_on(self, motor_id: int, motor_type: str) -> None:
        """Turn on the motor.

        Args:
            motor_id (int): The ID of the motor to turn on.
        """
        current_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

        id = motor_id  # self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFC]

        message = self._send_message_get_response(id, motor_id, data)

        # dummy motor type just check motor status
        motor_info = self.parse_recv_message(message, MotorType.DM4310, ignore_error=True)
        if int(motor_info.error_code, 16) != MotorErrorCode.normal:
            while int(motor_info.error_code, 16) != MotorErrorCode.normal:
                logging.info(f"motor {motor_id} error: {motor_info.error_message}")
                self.clean_error(motor_id=motor_id)
                self.try_receive_message()
                logging.info(f"motor {motor_id} error cleaned")
                # enable again

                message = self._send_message_get_response(id, motor_id, data)
                motor_info = self.parse_recv_message(message, motor_type, ignore_error=True)
        else:
            logging.info(f"motor {motor_id} is already on")
        logging.getLogger().setLevel(current_level)
        motor_info = self.parse_recv_message(message, motor_type)
        return motor_info

    def clean_error(self, motor_id: int) -> None:
        # self.try_receive_message()
        id = motor_id  # self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFB]
        logging.info("clear error")
        message = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        for _ in range(3):
            try:
                self.bus.send(message)
            except Exception as e:
                logging.warning(e)
                logging.warning(
                    "\033[91m" + "CAN Error: Failed to communicate with motor over can bus. Retrying..." + "\033[0m"
                )
        # message = self._send_message_get_response(id, data)

    def motor_off(self, motor_id: int) -> None:
        """Turn off the motor.

        Args:
            motor_id (int): The ID of the motor to turn off.
        """
        id = self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFD]
        message = self._send_message_get_response(id, motor_id, data)

    def save_zero_position(self, motor_id: int) -> None:
        """Save the current position as zero position.

        Args:
            motor_id (int): The ID of the motor to save zero position.
        """
        id = self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFE]
        try:
            message = self._send_message_get_response(id, motor_id, data, 2)
        except AssertionError:
            pass
        # check if set zero position success
        current_state = self.set_control(id, MotorType.DM4310, 0, 0, 0, 0, 0)
        diff = abs(current_state.position)
        if diff < 0.01:
            logging.info(f"motor {motor_id} set zero position success, current position: {current_state.position}")
        # message = self._receive_message(timeout=0.5)

    def set_control(
        self,
        motor_id: int,
        motor_type: str,
        pos: float,
        vel: float,
        kp: float,
        kd: float,
        torque: float,
    ) -> FeedbackFrameInfo:
        """Set the control of the motor and return its status.

        Args:
            motor_id (int): The ID of the motor. Check GUI for the CAN ID.
            motor_type (str): The type of the motor. Check MotorType class for available motor types.
            pos (float): The target position value.
            vel (float): The target velocity value.
            kp (float): The proportional gain value.
            kd (float): The derivative gain value.
            torque (float): The target torque value.

        Returns:
            FeedbackFrameInfo: The current state of the motor, including motor id, error code, position, velocity, torque, temperature.
        """
        frame_id = self._get_frame_id(motor_id)
        # Prepare the CAN message
        data = bytearray(8)
        if self.control_mode == ControlMode.MIT:
            const = MotorType.get_motor_constants(motor_type)

            pos_tmp = float_to_uint(pos, const.POSITION_MIN, const.POSITION_MAX, 16)
            vel_tmp = float_to_uint(vel, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
            kp_tmp = float_to_uint(kp, const.KP_MIN, const.KP_MAX, 12)
            kd_tmp = float_to_uint(kd, const.KD_MIN, const.KD_MAX, 12)
            tor_tmp = float_to_uint(torque, const.TORQUE_MIN, const.TORQUE_MAX, 12)

            # "& 0xFF" (bitwise AND with 0xFF) is used to ensure that only the lowest 8 bits (one byte) of a value are kept,
            # and any higher bits are discarded.
            data[0] = (pos_tmp >> 8) & 0xFF
            data[1] = pos_tmp & 0xFF
            data[2] = (vel_tmp >> 4) & 0xFF
            data[3] = ((vel_tmp & 0xF) << 4) | (kp_tmp >> 8)
            data[4] = kp_tmp & 0xFF
            data[5] = (kd_tmp >> 4) & 0xFF
            data[6] = ((kd_tmp & 0xF) << 4) | (tor_tmp >> 8)
            data[7] = tor_tmp & 0xFF
        elif self.control_mode == ControlMode.VEL:
            # system will only response to vel command
            can_data = struct.pack("<f", vel)
            data[0:4] = can_data[0:4]
        else:
            # Without this, an unencodable mode falls through and transmits the zero-initialised
            # bytearray above -- a valid frame commanding nothing, on every motor, with no error
            # anywhere. ControlMode.get_id_offset already refuses an unknown mode this way; POS_VEL
            # passes that one (it has a frame id) and reaches here, which is the gap.
            raise ValueError(
                f"cannot encode a command in {self.control_mode}: set_control implements MIT and VEL "
                "only, and any other mode would silently transmit an all-zero frame."
            )

        # Send the CAN message
        message = self._send_message_get_response(frame_id, motor_id, data, max_retry=15)

        # Parse the received message to extract motor information
        motor_info = self.parse_recv_message(message, motor_type)
        return motor_info

    def parse_recv_message(
        self, message: can.Message, motor_type: str, ignore_error: bool = False
    ) -> FeedbackFrameInfo:
        """Parse the received message to extract motor information.

        Args:
            message (can.Message): The received message.

        Returns:
            FeedbackFrameInfo: The current state of the motor.
        """
        data = message.data
        error_int = (data[0] & 0xF0) >> 4  # TODO: error code seems incorrect, double check

        # convert error into hex
        error_hex = hex(error_int)
        error_message = MotorErrorCode.get_error_message(error_int)

        motor_id_of_this_response = self.receive_mode.to_motor_id(message.arbitration_id)
        if error_hex != "0x1":
            logging.warning(
                f"motor id: {motor_id_of_this_response}, error: {error_message} at {self.name} and channel {self.bus.channel_info}"
            )
            if not ignore_error:
                logging.error(
                    f"motor id: {motor_id_of_this_response}, error: {error_message} at {self.name} and channel {self.bus.channel_info}"
                )
                raise RuntimeError(
                    f"Motor error detected: motor id: {motor_id_of_this_response}, error: {error_message}"
                )
        p_int = (data[1] << 8) | data[2]
        v_int = (data[3] << 4) | (data[4] >> 4)
        t_int = ((data[4] & 0xF) << 8) | data[5]
        temporature_mos = data[6]
        temperature_rotor = data[7]

        const = MotorType.get_motor_constants(motor_type)
        position = uint_to_float(p_int, const.POSITION_MIN, const.POSITION_MAX, 16)
        velocity = uint_to_float(v_int, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
        torque = uint_to_float(t_int, const.TORQUE_MIN, const.TORQUE_MAX, 12)
        temperature_mos = float(temporature_mos)
        temperature_rotor = float(temperature_rotor)

        return FeedbackFrameInfo(
            id=motor_id_of_this_response,
            error_code=error_hex,
            error_message=error_message,
            position=position,
            velocity=velocity,
            torque=torque,
            temperature_mos=temperature_mos,
            temperature_rotor=temperature_rotor,
        )


@dataclass
class MotorCmd:
    type: str = "pos_vel_torque"
    pos: float = 0.0
    vel: float = 0.0
    torque: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


class MotorChain(Protocol):
    """Class for CAN interface with multiple motors."""

    def __len__(self) -> int:
        """Get the number of motors in the chain."""
        raise NotImplementedError

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> List[MotorInfo]:
        """Set commands to the motors in the chain."""
        raise NotImplementedError


class DMChainCanInterface(MotorChain):
    def __init__(
        self,
        motor_list: List[Tuple[int, str]],
        motor_offset: np.ndarray,
        motor_direction: np.ndarray,
        channel: str = "PCAN_USBBUS1",
        bitrate: int = 1000000,
        start_thread: bool = True,  # If true, will start the internal motor reading loop
        motor_chain_name: str = "default_motor_chain",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        control_mode: ControlMode = ControlMode.MIT,
        get_same_bus_device_driver: Optional[Callable] = None,
        use_buffered_reader: bool = False,  # buffered reader is not very stable, the latest encoder fix allows us to use the non-buffered reader
        report_interval: float = REPORT_INTERVAL,
        control_freq: float = CONTROL_FREQ,  # Control loop frequency (Hz), used for the CAN bandwidth check
        enable_auto_recovery: bool = False,  # if True, try to clean+re-enable errored motors in the control loop instead of failing fast
        check_motor_types: bool = False,  # read Gr and refuse a chain not holding its declared motor types
        check_motor_config: bool = False,  # repair CTRL_MODE and compare PMAX/VMAX/TMAX; writes Flash
        loop_critical_motor_ids: Optional[Sequence[int]] = None,  # motors whose feedback the caller's loop acts on
    ):
        assert not use_buffered_reader, (
            "buffered reader is not very stable, the latest encoder fix allows us to use the non-buffered reader"
        )
        assert len(motor_list) > 0
        assert len(motor_list) == len(motor_offset) == len(motor_direction), (
            f"len{len(motor_list)}, len{len(motor_offset)}, len{len(motor_direction)}"
        )
        motor_ids = [motor_id for motor_id, _ in motor_list]
        assert motor_ids == sorted(set(motor_ids)), (
            f"motor_list IDs must be strictly ascending (unique), got {motor_ids}"
        )
        self.motor_list = motor_list
        # float dtype so a homing routine can write a radian zero offset (see set_zero_position)
        self.motor_offset = np.array(motor_offset, dtype=float)
        self.motor_direction = np.array(motor_direction)
        self.channel = channel
        # Read live each control-loop iteration; must be set before _motor_on()/start_thread() since
        # some callers (e.g. _get_gripper_only_robot) start the thread inside this constructor.
        self.enable_auto_recovery = enable_auto_recovery
        logging.info(f"Channel: {channel}, Bitrate: {bitrate}")
        # Last moment the bus is idle: everything above is pure Python, and the branch below opens the
        # socket. Register reads need an idle bus, so there is no later window -- see motor_check's
        # module docstring. Which checks may run, and in what order, is run_startup_checks' business:
        # both rules have to hold for every caller, and this is not the only call site.
        run_startup_checks(
            channel,
            motor_list,
            check_motor_types=check_motor_types,
            check_motor_config=check_motor_config,
            control_mode=control_mode,
            loop_critical_motor_ids=loop_critical_motor_ids,
            # Stated rather than defaulted: this is the path that is about to run the chain, and a motor
            # left in the wrong mode cannot be commanded at all. --survey-only is the caller that says False.
            repair=True,
        )
        if "can" in channel:
            self.motor_interface = DMSingleMotorCanInterface(
                channel=channel,
                bustype="socketcan",
                receive_mode=receive_mode,
                name=motor_chain_name,
                control_mode=control_mode,
                use_buffered_reader=use_buffered_reader,
            )
        else:
            self.motor_interface = DMSingleMotorCanInterface(
                channel=channel,
                bitrate=bitrate,
                name=motor_chain_name,
                use_buffered_reader=use_buffered_reader,
            )
        # CAN bus bandwidth check with 1.1x safety factor
        CAN_FRAME_BITS = 130  # approximate bits per CAN 2.0A frame including overhead
        frames_per_cycle = len(motor_list) * 2  # send + receive per motor
        bits_per_second = frames_per_cycle * CAN_FRAME_BITS * control_freq
        max_bits_per_second = bitrate / 1.1
        if bits_per_second > max_bits_per_second:
            max_safe_freq = max_bits_per_second / (frames_per_cycle * CAN_FRAME_BITS)
            logging.warning(
                f"CAN bus bandwidth exceeded: {bits_per_second:.0f} bps > {max_bits_per_second:.0f} bps "
                f"(bitrate={bitrate}, motors={len(motor_list)}, freq={control_freq}Hz). "
                f"Max safe frequency: {max_safe_freq:.0f} Hz"
            )

        self.state = None
        self.state_lock = threading.Lock()
        self._report_interval = report_interval
        self._rate_recorder = RateRecorder(name=self, report_interval=report_interval)

        self.same_bus_device_states = None
        self.same_bus_device_lock = threading.Lock()

        with self.same_bus_device_lock:
            if get_same_bus_device_driver is not None:
                self.same_bus_device_driver = get_same_bus_device_driver(self.motor_interface)
            else:
                self.same_bus_device_driver = None

            if self.same_bus_device_driver is not None:
                drained = self.motor_interface._drain_bus(timeout_s=0.2)
                if drained:
                    logging.info(f"Drained {drained} stale frames before motor bring-up")

            self.absolute_positions = None
            self._motor_on()
        starting_command = []
        for motor_state in self.state:
            starting_command.append(MotorCmd(torque=motor_state.torque))
        logging.info(f"Initializing motorchain with starting command: {starting_command}")
        self.commands = starting_command
        self.command_lock = threading.RLock()

        self.start_thread_flag = False
        self._control_thread: Optional[threading.Thread] = None
        if start_thread:
            self.start_thread()

    @property
    def comm_freq(self) -> float:
        return self._rate_recorder.last_rate

    def __repr__(self) -> str:
        return f"DMChainCanInterface(channel={self.channel})"

    def _update_absolute_positions(self, motor_feedback: List[MotorInfo]) -> None:
        init_mode = False
        if self.absolute_positions is None:
            self.absolute_positions = np.zeros(len(self.motor_list))
            init_mode = True

        for idx, motor_info in enumerate(self.motor_list):
            _motor_id, motor_type = motor_info
            const = MotorType.get_motor_constants(motor_type)
            position_min = const.POSITION_MIN
            position_max = const.POSITION_MAX
            position_range = position_max - position_min

            # Current position from feedback
            current_position = motor_feedback[idx].position

            # Previous absolute position
            previous_position = self.absolute_positions[idx]

            # Handle wrap-around
            delta_position = current_position - (previous_position % position_range)
            if delta_position > position_range / 2:  # Wrap backward
                delta_position -= position_range
            elif delta_position < -position_range / 2:  # Wrap forward
                delta_position += position_range

            if init_mode:
                self.absolute_positions[idx] = current_position
            else:
                self.absolute_positions[idx] += delta_position

    def __len__(self):
        return len(self.motor_list)

    def _joint_position_real_to_sim(self, joint_position_real: float) -> float:
        return (joint_position_real - self.motor_offset) * self.motor_direction

    def _joint_position_real_to_sim_idx(self, joint_position_real: float, idx: int) -> float:
        return (joint_position_real - self.motor_offset[idx]) * self.motor_direction[idx]

    def _joint_position_sim_to_real_idx(self, joint_position_sim: float, idx: int) -> float:
        return joint_position_sim * self.motor_direction[idx] + self.motor_offset[idx]

    def _motor_on(self) -> None:
        motor_feedback = []
        self.motor_interface._drain_bus(timeout_s=0.05)
        for motor_id, motor_type in self.motor_list:
            logging.info(f"Turning on motor_id: {motor_id}, motor_type: {motor_type}")
            time.sleep(0.003)
            motor_feedback.append(self.motor_interface.motor_on(motor_id, motor_type))
        self._update_absolute_positions(motor_feedback)
        self.state = motor_feedback
        self.running = True

    def start_thread(self) -> None:
        if self.start_thread_flag:
            return
        logging.info("starting separate thread for control loop")
        thread = threading.Thread(target=self._set_torques_and_update_state)
        thread.start()
        self._control_thread = thread
        self.start_thread_flag = True
        time.sleep(0.1)
        while self.state is None:
            time.sleep(0.1)
            logging.info("waiting for the first state")

    def _set_torques_and_update_state(self) -> None:
        """
        Control loop for updating motor torques and states at a fixed frequency.
        If step_time > EXPECTED_CONTROL_PERIODs, it will report the number of step_time > EXPECTED_CONTROL_PERIODs and mean step_time every REPORT_INTERVAL seconds.
        """
        last_step_time = time.time()
        step_time_exceed_count = 0
        step_time_sum = 0.0
        step_time_count = 0
        max_step_time = 0.0
        report_start_time = time.time()
        with self._rate_recorder:
            while self.running:
                try:
                    curr_time = time.time()
                    step_time = curr_time - last_step_time
                    last_step_time = curr_time

                    # Statistics
                    step_time_sum += step_time
                    step_time_count += 1
                    max_step_time = max(max_step_time, step_time)
                    if step_time > EXPECTED_CONTROL_PERIOD:
                        step_time_exceed_count += 1

                    # If step_time > EXPECTED_CONTROL_PERIOD, report every report_interval seconds
                    if step_time_exceed_count > 0 and curr_time - report_start_time >= self._report_interval:
                        mean_step_time = step_time_sum / step_time_count if step_time_count > 0 else 0.0
                        logging.info(
                            f"[{self} {self._report_interval}s Report] step_time > {EXPECTED_CONTROL_PERIOD}s: {step_time_exceed_count} times, mean step_time: {mean_step_time:.6f} s, max step_time: {max_step_time:.6f} s"
                        )
                        step_time_exceed_count = 0
                        step_time_sum = 0.0
                        step_time_count = 0
                        max_step_time = 0.0
                        report_start_time = curr_time

                    # Update state
                    with self.command_lock:
                        try:
                            motor_feedback = self._set_commands(self.commands)
                        except RuntimeError as e:
                            if self.enable_auto_recovery and "Motor error detected" in str(e):
                                logging.warning(f"Motor error in control loop, attempting recovery: {e}")
                                if self._try_recover_motors():
                                    logging.warning("Motor recovery successful, continuing control loop")
                                    continue
                                self.running = False
                                raise
                            raise

                        errors = np.array([motor_feedback[i].error_code != "0x1" for i in range(len(motor_feedback))])
                        if np.any(errors):
                            if self.enable_auto_recovery:
                                logging.warning(f"Motor errors detected in feedback: {errors}, attempting recovery")
                                if self._try_recover_motors(motor_feedback):
                                    logging.warning("Motor recovery successful, continuing control loop")
                                    continue
                            self.running = False
                            logging.error(f"motor errors: {errors}")
                            raise Exception(f"motor errors detected: {errors}, stopping control loop")

                    with self.state_lock:
                        self.state = motor_feedback
                        self._update_absolute_positions(motor_feedback)
                    if self.same_bus_device_driver is not None:
                        time.sleep(0.001)
                        with self.same_bus_device_lock:
                            # assume the same bus device is a passive input device (no commands to send) for now.
                            self.same_bus_device_states = self.same_bus_device_driver.read_states()
                    time.sleep(0.0005)  # yield GIL so other threads can acquire locks
                    self._rate_recorder.track()
                except Exception as e:
                    print(f"DM Error in control loop: {e}")
                    self.running = False
                    raise e

    def _try_recover_motors(self, motor_feedback: Optional[List[MotorInfo]] = None, max_retries: int = 3) -> bool:
        """Attempt to recover motors that report errors.

        For each motor with an error, clean the error and re-enable.
        Returns True if ALL motors recovered successfully, False otherwise.
        """
        for attempt in range(max_retries):
            # Determine which motors need recovery
            if motor_feedback is not None:
                error_indices = [i for i, fb in enumerate(motor_feedback) if fb.error_code != "0x1"]
            else:
                error_indices = list(range(len(self.motor_list)))

            if not error_indices:
                return True

            for idx in error_indices:
                motor_id, motor_type = self.motor_list[idx]
                logging.warning(f"Recovering motor {motor_id} ({motor_type}), attempt {attempt + 1}/{max_retries}")
                self.motor_interface.clean_error(motor_id)
                time.sleep(0.003)
                self.motor_interface.try_receive_message(timeout=0.002)
                try:
                    self.motor_interface.motor_on(motor_id, motor_type)
                except Exception as e:
                    logging.warning(f"Motor {motor_id} re-enable failed: {e}")
                    continue

            # Verify recovery by sending commands
            time.sleep(0.01)
            try:
                with self.command_lock:
                    motor_feedback = self._set_commands(self.commands)
                    if all(fb.error_code == "0x1" for fb in motor_feedback):
                        logging.warning("All motors recovered successfully")
                        with self.state_lock:
                            self.state = motor_feedback
                            self._update_absolute_positions(motor_feedback)
                        return True
            except RuntimeError:
                continue

        return False

    def _set_commands(self, commands: List[MotorCmd]) -> List[MotorInfo]:
        motor_feedback = []
        for idx, motor_info in enumerate(self.motor_list):
            motor_id, motor_type = motor_info
            torque = commands[idx].torque * self.motor_direction[idx]
            pos = self._joint_position_sim_to_real_idx(commands[idx].pos, idx)

            vel = commands[idx].vel * self.motor_direction[idx]
            kp = commands[idx].kp
            kd = commands[idx].kd
            try:
                fd_back = self.motor_interface.set_control(
                    motor_id=motor_id,
                    motor_type=motor_type,
                    pos=pos,
                    vel=vel,
                    kp=kp,
                    kd=kd,
                    torque=torque,
                )
            except Exception as e:
                logging.error(f"{idx}th motor at DMChainCanInterface {self} failed with info {motor_info}")
                raise e

            motor_feedback.append(fd_back)
        return motor_feedback

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        motor_infos = []
        timestamp = time.time()
        with self.state_lock:
            for idx in range(len(self.motor_list)):
                state = self.state[idx]
                motor_infos.append(
                    MotorInfo(
                        id=state.id,
                        error_code=state.error_code,
                        target_torque=torques[idx] if torques is not None else 0.0,
                        vel=state.velocity * self.motor_direction[idx],
                        eff=state.torque * self.motor_direction[idx],
                        pos=self._joint_position_real_to_sim_idx(self.absolute_positions[idx], idx),
                        temp_rotor=state.temperature_rotor,
                        temp_mos=state.temperature_mos,
                        timestamp=timestamp,
                    )
                )
        return motor_infos

    def set_zero_position(self, motor_idx: int) -> None:
        """Set a motor's current position as zero by shifting its software offset.

        Leaves the raw absolute-position accumulator intact so wrap-around unwrapping keeps working.
        """
        with self.state_lock:
            self.motor_offset[motor_idx] = self.absolute_positions[motor_idx]

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> List[MotorInfo]:
        command = []
        for idx in range(len(self.motor_list)):
            command.append(
                MotorCmd(
                    torque=torques[idx],
                    pos=pos[idx] if pos is not None else 0.0,
                    vel=vel[idx] if vel is not None else 0.0,
                    kp=kp[idx] if kp is not None else 0.0,
                    kd=kd[idx] if kd is not None else 0.0,
                )
            )
        with self.command_lock:
            self.commands = command
        if get_state:
            return self.read_states(torques=torques)

    def get_same_bus_device_states(self) -> Any:
        with self.same_bus_device_lock:
            return self.same_bus_device_states

    def close(self) -> None:
        """Stop the control loop and shut the CAN bus down, in that order.

        The control thread sends on ``self.motor_interface``'s socket every cycle, so shutting the bus
        down while it is still running pulls the file descriptor out from under an in-flight
        ``set_control`` and the thread dies with ``ValueError: file descriptor cannot be a negative
        integer (-1)``. Joining first lets the loop observe ``running = False`` and exit between cycles.
        """
        self.running = False
        thread = self._control_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_CONTROL_THREAD_JOIN_TIMEOUT)
            if thread.is_alive():
                logging.warning(
                    f"{self} control loop did not stop within {_CONTROL_THREAD_JOIN_TIMEOUT}s; "
                    "closing the CAN bus anyway"
                )
        self._control_thread = None
        self.start_thread_flag = False
        self.motor_interface.close()


class MultiDMChainCanInterface(MotorChain):
    """Class for interfacing with multiple asynchronous CAN interfaces."""

    def __init__(
        self,
        interfaces: List[DMChainCanInterface],
    ):
        self.interfaces = interfaces

    def __len__(self):
        return sum([len(inter) for inter in self.interfaces])

    @property
    def comm_freq(self) -> float:
        """Return the minimum comm_freq across all sub-interfaces."""
        freqs = [inter.comm_freq for inter in self.interfaces]
        return min(freqs) if freqs else 0.0

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> List[MotorInfo]:
        start_idx = 0
        motor_infos = []
        for inter in self.interfaces:
            inter_len = len(inter)
            end_idx = start_idx + inter_len
            inter_torques = torques[start_idx:end_idx]
            inter_pos = pos[start_idx:end_idx] if pos is not None else None
            inter_vel = vel[start_idx:end_idx] if vel is not None else None
            inter_kp = kp[start_idx:end_idx] if kp is not None else None
            inter_kd = kd[start_idx:end_idx] if kd is not None else None
            infos = inter.set_commands(inter_torques, inter_pos, inter_vel, inter_kp, inter_kd)
            motor_infos.extend(infos)
            start_idx = end_idx
        return motor_infos


_CLI_DEFAULT_MOTOR_TYPES: Dict[int, str] = {
    0x01: MotorType.DM4340,
    0x02: MotorType.DM4340,
    0x03: MotorType.DM4340,
    0x04: MotorType.DM4310,
    0x05: MotorType.DM4310,
    0x06: MotorType.DM4310,
    0x07: MotorType.DM4310,
}
"""Motor type per CAN id for the CLI below, used when ``--motor-type`` is not given.

This is a YAM v1 arm plus a 4310 gripper: ids 1-3 are DM4340 and 4-6 DM4310 in ``yam_v1.yml``,
``yam_pro_v1.yml`` and ``yam_ultra_v1.yml`` alike, and crank/linear/flexible_4310 each put a DM4310 at
0x07. One type for the whole chain was the old default and is wrong on every one of those arms -- it
declares joints 1-3 as DM4310, which encodes their torque against TORQUE_MAX 10 instead of 28 -- so
now that ``DMChainCanInterface`` verifies ``Gr``, that default would simply refuse to start. Naming the
ids is what makes the default describe real hardware.

An id outside this map has no obvious default and asks for ``--motor-type`` rather than being guessed
at: id 9 is the Flow Base rail's DM8009, not a DM4310.
"""


def _cli_motor_list(motor_ids: Tuple[int, ...], motor_type: Optional[Tuple[str, ...]]) -> List[List[Any]]:
    """Pair each CAN id with a motor type: ``motor_type`` if given, else the per-id default.

    ``motor_type`` is one type per id, positionally, in the order the ids were given, because every chain
    this tool points at is mixed: a YAM v1 arm is DM4340 at 1-3 and DM4310 at 4-7, a big_yam is DM6248 at
    1-2, a Flow Base alternates DM4310V and DM_FLOW_WHEEL. A single type for a whole chain cannot describe
    one of those, which is why surveying a big_yam or a yam_ultra_v2 was impossible before. A single type
    is still accepted and broadcast to every id -- correct for a homogeneous sub-chain (``--motor-id 1 2 3
    --motor-type DM4340``) and for the one-motor case, and a wrong broadcast can no longer pass quietly
    now that ``--check-motor-types`` reads Gr.
    """
    if motor_type is None:
        unknown = [motor_id for motor_id in motor_ids if motor_id not in _CLI_DEFAULT_MOTOR_TYPES]
        if unknown:
            raise SystemExit(
                f"no default motor type for CAN id(s) {unknown}; pass --motor-type explicitly. The "
                f"defaults cover ids {sorted(_CLI_DEFAULT_MOTOR_TYPES)}, a YAM v1 arm plus a 4310 gripper."
            )
        return [[motor_id, _CLI_DEFAULT_MOTOR_TYPES[motor_id]] for motor_id in motor_ids]
    # Above the length check, not a case of it: broadcasting one type is a deliberate exception.
    if len(motor_type) == 1:
        return [[motor_id, motor_type[0]] for motor_id in motor_ids]
    if len(motor_type) != len(motor_ids):
        raise SystemExit(
            f"--motor-type was given {len(motor_type)} type(s) {list(motor_type)} for the "
            f"{len(motor_ids)} CAN id(s) {list(motor_ids)}: pass one type per id in the same order, or a "
            "single type to use for every id, or omit it for the per-id default."
        )
    return [[motor_id, declared] for motor_id, declared in zip(motor_ids, motor_type, strict=True)]


def _cli_control_mode(control_mode: Optional[str], check_motor_config: bool) -> str:
    """The mode to build the chain in, refusing to guess it when the config check depends on it.

    This value is the whole expectation the config check compares against: it decides the ``CTRL_MODE``
    every motor must hold and -- without ``--survey-only`` -- the value written to Flash to get there.
    There is no mode that is right for both machines, so when that check is on, the operator says which
    chain this is. Required even for a survey: defaulted to MIT, a Flow Base surveys as eight false
    mismatches, which is how an operator learns to pass ``--no-verify-motor-config`` and stop looking.
    A usage mistake, so ``SystemExit`` with a usage line rather than a library error.
    """
    if control_mode is not None:
        return control_mode
    if check_motor_config:
        raise SystemExit(
            "--check-motor-config needs --control-mode: it is the CTRL_MODE every motor is required to "
            "hold, and (without --survey-only) the value written to Flash to get there. The right one "
            "depends on the machine -- MIT for an arm or gripper chain, VEL for a Flow Base chain (which "
            "must hold speed mode). Defaulting it would silently reconfigure, or falsely condemn, "
            "whichever one you did not mean."
        )
    return ControlMode.MIT


def main(
    channel: str = "can0",
    motor_id: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    motor_type: Optional[Tuple[str, ...]] = None,
    control_mode: Optional[Literal["MIT", "VEL"]] = None,
    print_state: bool = False,
    print_pos: bool = False,
    report_interval: float = REPORT_INTERVAL,
    check_motor_types: bool = False,
    check_motor_config: bool = False,
    survey_only: bool = False,
) -> None:
    """Hold a motor chain at zero command and print its state, to check the chain answers. Ctrl-C to stop.

    "Zero command" is zero torque in MIT and zero velocity in VEL -- the two modes set_control encodes.

    Args:
        channel: SocketCAN channel the motor chain is on.
        motor_id: CAN ids to bring up, in decimal -- ids on these buses are 1-9.
        motor_type: Motor type per CAN id, in the same order as --motor-id, since a real chain is mixed
            (a big_yam is DM6248 at joints 1-2, a yam_ultra_v2 DM4340 at joint 4). A single type is
            broadcast to every id. Omit for the per-id default: DM4340 on ids 1-3 and DM4310 on 4-7,
            i.e. a YAM v1 arm plus a 4310 gripper.
        control_mode: Mode to command the chain in, and the CTRL_MODE --check-motor-config requires every
            motor to hold -- repairing it to this value unless --survey-only. REQUIRED with
            --check-motor-config: MIT for an arm or gripper chain, VEL for a Flow Base chain. Defaults to
            MIT otherwise. Only the two modes set_control can encode are offered.
        print_state: Print every motor's full state each cycle.
        print_pos: Print every motor's position each cycle.
        report_interval: Rate/step-time report interval in seconds.
        check_motor_types: Read every motor's Gr register and refuse a chain that is not holding the
            declared motor types. Off by default because this tool is also how you poke a chain whose
            types you are still working out; turn it on to confirm a chain you believe you know.
        check_motor_config: Read every motor's CTRL_MODE, repairing and saving to Flash any that is not
            --control-mode, and compare PMAX/VMAX/TMAX against the driver's constants for the declared
            type. With --survey-only the repair is suppressed and the mismatch only reported. Requires
            --check-motor-types, since the scaling registers on a wrong part hold that part's own scale.
            Off by default here because it writes to Flash and this tool is the one place you may be
            pointing at a chain you have not identified yet; a real arm or Flow Base runs it on every
            launch.
        survey_only: Run the requested checks and exit, without building a chain, enabling any motor or
            energising anything. This is the read-only survey, and it reads in both senses: a CTRL_MODE
            that disagrees with --control-mode is reported with the dm_motor_registers.py command that
            fixes it, not written. Without this flag the tool repairs CTRL_MODE and then holds the chain
            at zero command, which means every motor is enabled.
    """
    from i2rt.utils.utils import override_log_level

    override_log_level(level=logging.INFO)

    motor_list = _cli_motor_list(motor_id, motor_type)
    resolved_mode = _cli_control_mode(control_mode, check_motor_config)
    if survey_only and not (check_motor_types or check_motor_config):
        raise SystemExit(
            "--survey-only with neither --check-motor-types nor --check-motor-config would read nothing "
            "and exit 0. Ask for at least one check, or drop --survey-only to hold the chain at zero command."
        )
    # The chain as declared, logged before anything is read: it is what every command below is encoded
    # with, and on this tool it is typed by hand. verify_motor_types logs it too, but only when it runs.
    logging.info(
        "chain on %s in %s: %s",
        channel,
        resolved_mode,
        ", ".join(f"{declared_id} {declared_type}" for declared_id, declared_type in motor_list),
    )
    if survey_only:
        run_startup_checks(
            channel,
            motor_list,
            check_motor_types=check_motor_types,
            check_motor_config=check_motor_config,
            control_mode=resolved_mode,
            # What makes --survey-only a survey: a wrong CTRL_MODE is reported with the command that
            # fixes it, never written. Without this the flag skipped building the chain but still
            # Flash-saved CTRL_MODE, so pointing the arm line at a Flow Base reconfigured all 8 motors.
            repair=False,
        )
        return
    motor_chain = DMChainCanInterface(
        motor_list,
        [0] * len(motor_list),
        [1] * len(motor_list),
        channel=channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        control_mode=resolved_mode,
        report_interval=report_interval,
        start_thread=False,
        check_motor_types=check_motor_types,
        check_motor_config=check_motor_config,
    )
    motor_chain.start_thread()
    while True:
        motor_chain.set_commands(np.zeros(len(motor_list)))
        if print_state:
            print(motor_chain.read_states())
        if print_pos:
            print([state.pos for state in motor_chain.read_states()])
        time.sleep(0.1)


if __name__ == "__main__":
    tyro.cli(main)
