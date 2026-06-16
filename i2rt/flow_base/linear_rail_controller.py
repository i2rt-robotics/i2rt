import logging
import sys
import threading
import time
from typing import Any, Dict, Literal, Optional

import numpy as np
from RPi import GPIO

from i2rt.motor_drivers.dm_driver import DMChainCanInterface
from i2rt.motor_drivers.utils import MotorInfo

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LinearRailController")

# GPIO Pins Definition
BRAKE_CONTROL_GPIO = 12  # Brake control GPIO
UPPER_LIMIT_GPIO = 5  # Upper limit GPIO
LOWER_LIMIT_GPIO = 6  # Lower limit GPIO
HOMING_SPEED_RATIO = 0.5
HOMING_TIMEOUT = 30.0  # Timeout for homing procedure in seconds
COMMAND_TIMEOUT = 0.25  # Timeout for command stream (2.5 * POLICY_CONTROL_PERIOD, where POLICY_CONTROL_PERIOD = 0.1s)


def initialize_brake_gpio() -> None:
    """Initialize brake control GPIO pin as an independent function.

    This function can be called before LinearRailController initialization
    to avoid blocking during class initialization.
    """
    try:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BRAKE_CONTROL_GPIO, GPIO.OUT)
        logger.info("Brake GPIO initialized")
    except RuntimeError as e:
        # GPIO mode already set, this is fine
        if "mode" in str(e).lower() or "already" in str(e).lower():
            try:
                GPIO.setup(BRAKE_CONTROL_GPIO, GPIO.OUT)
                logger.debug("Brake GPIO setup completed (mode was already set)")
            except Exception as setup_error:
                logger.warning(f"Brake GPIO setup failed: {setup_error}")
        else:
            logger.warning(f"Failed to initialize brake GPIO: {e}")
    except Exception as e:
        logger.error(f"Failed to initialize brake GPIO: {e}")
        raise


def set_brake_gpio(engaged: bool) -> None:
    """Set brake GPIO state (engaged=True: brake on, engaged=False: brake off)."""
    try:
        GPIO.output(BRAKE_CONTROL_GPIO, GPIO.LOW if engaged else GPIO.HIGH)
        action = "engaged" if engaged else "released"
        logger.info(f"Brake {action}")
    except Exception as e:
        action = "engage" if engaged else "release"
        logger.error(f"Failed to {action} brake: {e}")
        raise


class SingleMotorControlInterface:
    """Single motor control interface for motor chains"""

    def __init__(self, motor_chain: DMChainCanInterface, target_motor_idx: int = -1):
        """Initialize single motor control interface"""
        if len(motor_chain) == 0:
            raise ValueError(f"Motor chain must contain at least 1 motor, got {len(motor_chain)}")

        self.motor_chain = motor_chain
        self.target_motor_idx = target_motor_idx

        if target_motor_idx < 0 or target_motor_idx >= len(motor_chain):
            raise ValueError(f"Motor index {target_motor_idx} out of range [0, {len(motor_chain)})")

        self.motor_id = motor_chain.motor_list[target_motor_idx][0]

    def set_velocity(self, vel: float) -> None:
        """Set motor velocity"""
        num_motors = len(self.motor_chain)

        velocities = np.zeros(num_motors)
        velocities[self.target_motor_idx] = vel

        # Preserve velocities of other motors (e.g., base motors) by reading current commands
        with self.motor_chain.command_lock:
            current_commands = self.motor_chain.commands
            if current_commands and len(current_commands) == num_motors:
                # Preserve velocities of other motors
                for idx in range(num_motors):
                    if idx != self.target_motor_idx:
                        velocities[idx] = current_commands[idx].vel

        torques = np.zeros(num_motors)
        self.motor_chain.set_commands(torques=torques, vel=velocities, pos=None, kp=None, kd=None, get_state=False)

    def get_state(self) -> MotorInfo:
        """Get motor state"""
        return self.motor_chain.read_states()[self.target_motor_idx]

    def set_zero_position(self) -> None:
        """Set the current motor position as the zero reference"""
        self.motor_chain.set_zero_position(self.target_motor_idx)

    @classmethod
    def from_multi_motor_chain(
        cls, motor_chain: DMChainCanInterface, target_motor_idx: int
    ) -> "SingleMotorControlInterface":
        """Create SingleMotorControlInterface from an existing multi-motor chain using motor index"""
        return cls(motor_chain, target_motor_idx=target_motor_idx)


class LinearRailController:
    def __init__(
        self,
        single_motor_control_interface: SingleMotorControlInterface,
        rail_speed: float = 14.0,
        auto_home: bool = True,  # Automatically start homing after initialization
        homing_timeout: float = HOMING_TIMEOUT,  # Timeout for homing procedure in seconds
        total_stroke_m: float = 1.0,
    ):
        """Initialize linear rail controller

        Args:
            single_motor_control_interface: Motor control interface (required)
            rail_speed: Maximum rail speed in rad/s
            auto_home: Whether to automatically home after initialization
            homing_timeout: Timeout for homing procedure in seconds
            total_stroke_m: Physical stroke between upper and lower limit switches, in meters.
                Used during the top-then-bottom startup calibration to convert motor radians
                into linear meters: meters_per_rad = total_stroke_m / (theta_upper - theta_lower).
        """
        self.single_motor_control_interface = single_motor_control_interface
        self.rail_speed = rail_speed
        self.auto_home = auto_home
        self.homing_timeout = homing_timeout
        self.total_stroke_m = total_stroke_m
        self.meters_per_rad: Optional[float] = None

        self.initialized = False
        self.brake_on = True

        self._lock = threading.Lock()
        self.upper_limit_triggered = False
        self.lower_limit_triggered = False
        self._gpio_mode_set = False
        self._gpio_initialized = False  # Flag to prevent duplicate GPIO initialization
        self._homing_event = threading.Event()
        self._homing_start_time = None
        self.homing_speed_ratio = HOMING_SPEED_RATIO

        # Command timeout tracking (similar to base control)
        self.last_command_time = time.time() - 1000000  # Initialize to far past
        self.command_timeout = COMMAND_TIMEOUT

        # GPIO initialization is now done manually in flow_base_controller.py
        # to avoid blocking during initialization

        if self.auto_home:
            self._initialize_linear_rail()
        else:
            # If auto_home is False, mark as initialized but don't release brake yet
            # Brake will be released after GPIO is initialized in flow_base_controller.py
            with self._lock:
                self.initialized = True
            logger.info("Linear rail initialized without auto-homing (GPIO will be initialized separately)")

    def _ensure_gpio_mode(self) -> None:
        """Ensure GPIO mode is set"""
        if not self._gpio_mode_set:
            try:
                GPIO.setmode(GPIO.BCM)
                self._gpio_mode_set = True
            except RuntimeError as e:
                if "mode" in str(e).lower() or "already" in str(e).lower():
                    self._gpio_mode_set = True
                else:
                    raise

    def initialize_gpio(self) -> None:
        """Initialize GPIO pins for limit switches and brake control with event callbacks.

        This method should be called manually after LinearRailController initialization
        to avoid blocking during __init__.
        """
        # Prevent duplicate initialization
        if self._gpio_initialized:
            logger.debug("GPIO already initialized, skipping")
            return

        try:
            self._ensure_gpio_mode()
            # Brake GPIO is already initialized by initialize_brake_gpio() function
            # Only setup limit switch GPIOs here
            GPIO.setup(UPPER_LIMIT_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.setup(LOWER_LIMIT_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_UP)

            # Read initial limit switch states
            initial_upper_limit = GPIO.input(UPPER_LIMIT_GPIO) == GPIO.HIGH
            initial_lower_limit = GPIO.input(LOWER_LIMIT_GPIO) == GPIO.HIGH

            with self._lock:
                self.upper_limit_triggered = initial_upper_limit
                self.lower_limit_triggered = initial_lower_limit

                if initial_lower_limit:
                    logger.info("Linear rail is at lower limit")
                elif initial_upper_limit:
                    logger.info("Linear rail is at upper limit")
                else:
                    logger.info("Linear rail initialized - not at any limit")

            GPIO.add_event_detect(
                UPPER_LIMIT_GPIO,
                GPIO.BOTH,  # Detect both rising and falling edges
                callback=self._upper_limit_callback,
                bouncetime=50,  # Debounce time in milliseconds
            )
            GPIO.add_event_detect(
                LOWER_LIMIT_GPIO,
                GPIO.BOTH,  # Detect both rising and falling edges
                callback=self._lower_limit_callback,
                bouncetime=50,  # Debounce time in milliseconds
            )

            self._gpio_initialized = True
            logger.info("GPIO initialized successfully with event callbacks for limit switches")
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")
            raise

    def _limit_switch_callback(self, channel: int, is_upper: bool) -> None:
        """Generic callback function for limit switch GPIO events - updates limit state and stops motor immediately

        The GPIO interrupt will still trigger and stop the motor when limit switches are activated.

        Args:
            channel: GPIO channel number
            is_upper: True for upper limit, False for lower limit
        """
        try:
            gpio_pin = UPPER_LIMIT_GPIO if is_upper else LOWER_LIMIT_GPIO
            limit_state = GPIO.input(gpio_pin) == GPIO.HIGH
            limit_name = "upper" if is_upper else "lower"

            with self._lock:
                if is_upper:
                    self.upper_limit_triggered = limit_state
                else:
                    self.lower_limit_triggered = limit_state

            if limit_state:
                logger.warning(f"{limit_name.capitalize()} limit switch triggered!")
                self.single_motor_control_interface.set_velocity(0.0)
            else:
                logger.info(f"{limit_name.capitalize()} limit switch released")
        except Exception as e:
            limit_name = "upper" if is_upper else "lower"
            logger.error(f"Error in {limit_name} limit callback: {e}")

    def _upper_limit_callback(self, channel: int) -> None:
        """Callback wrapper for upper limit switch"""
        self._limit_switch_callback(channel, is_upper=True)

    def _lower_limit_callback(self, channel: int) -> None:
        """Callback wrapper for lower limit switch"""
        self._limit_switch_callback(channel, is_upper=False)

    def set_brake(self, engaged: bool) -> None:
        """Set brake state (engaged=True: brake on, engaged=False: brake off)"""
        try:
            # Use the independent brake GPIO function
            set_brake_gpio(engaged)
            with self._lock:
                self.brake_on = engaged
        except Exception as e:
            action = "engage" if engaged else "release"
            logger.error(f"Failed to {action} brake: {e}")

    def _initialize_linear_rail(self) -> None:
        """Calibrate linear rail by driving to the upper limit, then the lower limit.

        Captures the motor angle at each limit, computes meters_per_rad assuming the
        physical stroke between limits is ``self.total_stroke_m``, then zeroes the
        encoder at the lower limit so encoder 0 corresponds to the bottom of travel.
        """
        try:
            # Release brake before any motion
            self.set_brake(engaged=False)

            # Phase 1: drive to upper limit (skip if already triggered there)
            with self._lock:
                already_at_upper = self.upper_limit_triggered
            if not already_at_upper:
                self._move_until_limit(direction="up")
            time.sleep(0.2)  # settle so the encoder reading is stable
            theta_upper = self.single_motor_control_interface.get_state().pos

            # Phase 2: drive to lower limit (skip if already triggered there)
            with self._lock:
                already_at_lower = self.lower_limit_triggered
            if not already_at_lower:
                self._move_until_limit(direction="down")
            time.sleep(0.2)
            theta_lower = self.single_motor_control_interface.get_state().pos

            # Phase 3: compute meters-per-radian from the captured motor angles.
            # Sign of delta is whatever motor_direction yields; carrying it through gives a
            # signed conversion factor so linear_pos = motor_pos * meters_per_rad always
            # grows from 0 (bottom) to total_stroke_m (top) regardless of motor direction.
            delta = theta_upper - theta_lower
            if abs(delta) < 1e-3:
                raise RuntimeError(
                    f"Linear-rail calibration failed: |theta_upper - theta_lower| = "
                    f"{abs(delta):.6f} rad is too small to calibrate "
                    f"(theta_upper={theta_upper:.3f}, theta_lower={theta_lower:.3f})"
                )
            with self._lock:
                self.meters_per_rad = self.total_stroke_m / delta
            logger.info(
                f"Linear rail calibrated: theta_upper={theta_upper:.3f} rad, "
                f"theta_lower={theta_lower:.3f} rad, delta={delta:.3f} rad, "
                f"meters_per_rad={self.meters_per_rad:.6f} m/rad "
                f"(stroke={self.total_stroke_m:.3f} m)"
            )

            # Phase 4: zero encoder at lower limit so encoder 0 == bottom of travel
            self._set_home_zero()
            with self._lock:
                self.initialized = True

        except Exception as e:
            logger.error(f"Linear rail initialization failed: {e}")
            self.initialized = False
            try:
                self.single_motor_control_interface.set_velocity(0.0)
            except Exception as stop_error:
                logger.error(f"Failed to stop linear-rail motor during cleanup: {stop_error}")
            with self._lock:
                self._homing_event.clear()
                self._homing_start_time = None
            raise  # Re-raise the exception (timeout RuntimeError or other errors)

    def _move_until_limit(self, direction: Literal["up", "down"]) -> None:
        """Drive the rail until the corresponding limit switch triggers, or time out.

        Continuously re-applies the homing velocity (every 50 ms) so the base controller's
        own ``set_commands`` calls don't race-overwrite the rail motor's velocity. Sets
        ``_homing_event`` while running so ``is_homing()`` reflects the in-progress state.
        """
        if direction == "up":
            motor_velocity = self.rail_speed * HOMING_SPEED_RATIO
        elif direction == "down":
            motor_velocity = -self.rail_speed * HOMING_SPEED_RATIO
        else:
            raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")

        logger.info(f"Homing started with velocity {motor_velocity:.3f} rad/s (moving {direction})")

        with self._lock:
            self._homing_event.set()
            self._homing_start_time = time.time()

        start_time = time.time()
        last_velocity_set_time = start_time
        velocity_set_interval = 0.05  # Re-apply every 50 ms

        self.single_motor_control_interface.set_velocity(motor_velocity)

        try:
            while time.time() - start_time < self.homing_timeout:
                current_time = time.time()

                if current_time - last_velocity_set_time >= velocity_set_interval:
                    self.single_motor_control_interface.set_velocity(motor_velocity)
                    last_velocity_set_time = current_time

                with self._lock:
                    triggered = self.upper_limit_triggered if direction == "up" else self.lower_limit_triggered
                if triggered:
                    self.single_motor_control_interface.set_velocity(0.0)
                    elapsed = current_time - start_time
                    logger.info(f"{direction.capitalize()} limit reached in {elapsed:.1f}s")
                    return

                time.sleep(0.01)

            # Timeout
            self.single_motor_control_interface.set_velocity(0.0)
            raise RuntimeError(f"Homing timed out after {self.homing_timeout}s moving {direction}")
        finally:
            with self._lock:
                self._homing_event.clear()
                self._homing_start_time = None

    def _stop_homing(self) -> None:
        """Stop homing procedure and reset state (assumes lock is held)"""
        self.single_motor_control_interface.set_velocity(0.0)
        self._homing_event.clear()
        self._homing_start_time = None

    def _set_home_zero(self) -> None:
        """Zero the encoder at the current lower-limit (home) position so encoder 0 == bottom of travel"""
        self.single_motor_control_interface.set_zero_position()
        logger.info("Linear rail encoder zeroed at lower limit (encoder 0 = home)")

    def is_homing(self) -> bool:
        """Check if linear rail is currently homing"""
        with self._lock:
            return self._homing_event.is_set()

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the linear rail.

        Returns position and velocity in BOTH units:
          - position.motor / velocity.motor: motor encoder in rad / rad/s.
          - position.linear / velocity.linear: linear travel in m / m/s, derived from
            ``meters_per_rad`` captured during startup calibration. ``None`` until the
            controller is calibrated (e.g. when ``auto_home=False``).
        """
        motor_state = self.single_motor_control_interface.get_state()

        with self._lock:
            brake_on = self.brake_on
            initialized = self.initialized
            upper_limit = self.upper_limit_triggered
            lower_limit = self.lower_limit_triggered
            meters_per_rad = self.meters_per_rad

        if meters_per_rad is not None:
            linear_pos: Optional[float] = motor_state.pos * meters_per_rad
            linear_vel: Optional[float] = motor_state.vel * meters_per_rad
        else:
            linear_pos = None
            linear_vel = None

        return {
            "position": {
                "motor": motor_state.pos,
                "linear": linear_pos,
            },
            "velocity": {
                "motor": motor_state.vel,
                "linear": linear_vel,
            },
            "brake_on": brake_on,
            "initialized": initialized,
            "upper_limit_triggered": upper_limit,
            "lower_limit_triggered": lower_limit,
            "meters_per_rad": meters_per_rad,
            "motor_state": motor_state,
        }

    def set_velocity(self, vel: float) -> None:
        """Set the velocity of the linear rail, unit in rad/s (motor velocity)"""
        assert self.initialized, "Linear rail must be initialized before setting velocity"
        assert not self.brake_on, "Brake must be released before setting velocity"

        with self._lock:
            current_time = time.time()
            if current_time - self.last_command_time > self.command_timeout:
                try:
                    self.single_motor_control_interface.set_velocity(0.0)
                    logger.warning(
                        f"Linear rail command timeout ({self.command_timeout:.2f}s) detected, "
                        "but new command received (command stream recovered)"
                    )
                except Exception as e:
                    logger.error(f"Failed to stop linear rail on timeout: {e}")

            # Update last command time when receiving a command (command stream active)
            self.last_command_time = current_time

            if vel > 0.0 and self.upper_limit_triggered:
                logger.warning("Upper limit triggered, cannot move forward")
                self.single_motor_control_interface.set_velocity(0.0)
                return
            if vel < 0.0 and self.lower_limit_triggered:
                logger.warning("Lower limit triggered, cannot move backward")
                self.single_motor_control_interface.set_velocity(0.0)
                if self._homing_event.is_set():
                    self._set_home_zero()
                    elapsed_time = time.time() - self._homing_start_time if self._homing_start_time else 0.0
                    logger.info(f"Homing success! Zero position found in {elapsed_time:.1f}s")
                    self._stop_homing()
                return

            try:
                self.single_motor_control_interface.set_velocity(vel)
                if vel != 0.0:
                    logger.info(f"Linear rail velocity set to {vel:.3f} rad/s")
            except Exception as e:
                logger.error(f"Failed to set linear rail velocity: {e}")
                raise

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            self.single_motor_control_interface.set_velocity(0.0)
            if self.initialized:
                # Ensure GPIO mode is set before cleanup operations
                try:
                    self._ensure_gpio_mode()
                    self.set_brake(engaged=True)
                    GPIO.remove_event_detect(UPPER_LIMIT_GPIO)
                    GPIO.remove_event_detect(LOWER_LIMIT_GPIO)
                    GPIO.cleanup()
                except Exception as gpio_error:
                    logger.warning(f"GPIO cleanup error (may be expected if GPIO was not initialized): {gpio_error}")
            logger.info("Linear rail controller cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
