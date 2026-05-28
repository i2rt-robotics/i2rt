import sys
import threading
import time
from dataclasses import dataclass
from pprint import pprint
from typing import Any, Literal

import numpy as np
import portal
import tyro

from i2rt.flow_base.flow_base_controller import BASE_DEFAULT_PORT


class FlowBaseClient:
    def __init__(self, host: str = "localhost", with_linear_rail: bool = False):
        self.with_linear_rail = with_linear_rail
        self.client = portal.Client(f"{host}:{BASE_DEFAULT_PORT}")
        self.num_dofs = 3 if not self.with_linear_rail else 4
        self.command = {"target_velocity": np.zeros(self.num_dofs), "frame": "local"}
        self._lock = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._update_command)
        self._thread.start()

    def _update_command(self) -> None:
        while self.running:
            with self._lock:
                self.client.set_target_velocity(self.command).result()
            time.sleep(0.02)

    def get_odometry(self) -> Any:
        return self.client.get_odometry({}).result()

    def get_wheel_speeds(self) -> Any:
        """Return per-motor angular velocities (rad/s) for the 4 casters: {steer, drive}."""
        return self.client.get_wheel_speeds({}).result()

    def get_observation(self) -> Any:
        """Return combined observation: odometry, wheel speeds, and linear rail state if enabled."""
        obs: dict[str, Any] = {"odometry": self.get_odometry()}
        if self.with_linear_rail:
            obs["linear_rail"] = self.get_linear_rail_state()
        obs["wheel_speeds"] = self.get_wheel_speeds()
        return obs

    def reset_odometry(self) -> Any:
        return self.client.reset_odometry({}).result()

    def set_target_velocity(self, target_velocity: np.ndarray, frame: str = "local") -> None:
        """Set target velocity for base and optionally linear rail.

        Args:
            target_velocity: [x, y, theta] or [x, y, theta, linear_rail_vel]
            frame: "local" or "global"
        """
        assert target_velocity.shape == (self.num_dofs,), f"Target velocity must have shape ({self.num_dofs},)"
        assert frame in ["local", "global"], "Frame must be either local or global"

        with self._lock:
            self.command["target_velocity"] = target_velocity
            self.command["frame"] = frame

    def get_linear_rail_state(self) -> Any:
        """Get the current state of the linear rail."""
        if not self.with_linear_rail:
            raise ValueError("Linear rail not enabled. Initialize FlowBaseClient with with_linear_rail=True")
        return self.client.get_linear_rail_state({}).result()

    def set_linear_rail_velocity(self, velocity: float) -> None:
        """Set the velocity of the linear rail.

        Args:
            velocity (float): Target velocity in rad/s
        """
        if not self.with_linear_rail:
            raise ValueError("Linear rail not enabled. Initialize FlowBaseClient with with_linear_rail=True")
        with self._lock:
            if len(self.command["target_velocity"]) < 4:
                self.command["target_velocity"] = np.append(self.command["target_velocity"], 0.0)
            self.command["target_velocity"][3] = velocity

    def close(self) -> None:
        """Stop the client and clean up resources."""
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def _format_rail_state(rail_state: dict) -> str:
    """Format a one-line summary of get_linear_rail_state() for the CLI."""
    position = rail_state.get("position", {})
    velocity = rail_state.get("velocity", {})
    pos_motor = position.get("motor")
    pos_linear = position.get("linear")
    vel_motor = velocity.get("motor")
    vel_linear = velocity.get("linear")
    motor_part = (
        f"motor: {pos_motor:+.3f} rad / {vel_motor:+.3f} rad/s"
        if pos_motor is not None and vel_motor is not None
        else f"motor: {pos_motor} rad / {vel_motor} rad/s"
    )
    linear_part = (
        f"linear: {pos_linear:+.4f} m / {vel_linear:+.4f} m/s"
        if pos_linear is not None and vel_linear is not None
        else "linear: not calibrated"
    )
    return (
        f"{motor_part}, {linear_part} "
        f"upper_limit: {rail_state.get('upper_limit_triggered')} "
        f"lower_limit: {rail_state.get('lower_limit_triggered')}"
    )


@dataclass
class Args:
    command: Literal[
        "get_odometry",
        "get_observation",
        "get_wheel_speeds",
        "reset_odometry",
        "test_command",
        "test_linear_rail",
        "get_linear_rail_state",
    ] = "get_odometry"
    """Command to run against the FlowBase server."""
    host: str = "localhost"
    """Host running the FlowBase server."""
    with_linear_rail: bool = False
    """Enable linear rail support (auto-enabled for linear-rail commands)."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    linear_rail_commands = ("test_linear_rail", "get_linear_rail_state")
    use_linear_rail = args.with_linear_rail or args.command in linear_rail_commands

    client = FlowBaseClient(args.host, with_linear_rail=use_linear_rail)

    if args.command == "get_odometry":
        print(client.get_odometry())
        client.close()
        exit()
    elif args.command == "get_observation":
        pprint(client.get_observation(), sort_dicts=False, width=100)
        client.close()
        exit()
    elif args.command == "get_wheel_speeds":
        pprint(client.get_wheel_speeds(), sort_dicts=False, width=100)
        client.close()
        exit()
    elif args.command == "reset_odometry":
        client.reset_odometry()
        client.close()
        exit()
    elif args.command == "test_command":
        client.set_target_velocity(np.array([0.0, 0.0, 0.1]), "local")
        while True:
            odo_reading = client.get_odometry()
            pos = odo_reading["position"]
            vw = odo_reading["velocity"]["world"]
            vb = odo_reading["velocity"]["body"]
            px, py, pz = pos["translation"]
            wx, wy, wz = vw["translation"]
            bx, by, bz = vb["translation"]
            sys.stdout.write(
                f"\r pos.t: [{px:+.3f} {py:+.3f} {pz:+.3f}] pos.r: {pos['rotation']:+.3f} "
                f"world.t: [{wx:+.3f} {wy:+.3f} {wz:+.3f}] world.r: {vw['rotation']:+.3f} "
                f"body.t: [{bx:+.3f} {by:+.3f} {bz:+.3f}] body.r: {vb['rotation']:+.3f}"
            )
            sys.stdout.flush()
            time.sleep(0.02)
    elif args.command == "test_linear_rail":
        try:
            client.set_linear_rail_velocity(0.5)
            while True:
                rail_state = client.get_linear_rail_state()
                sys.stdout.write("\r" + _format_rail_state(rail_state))
                sys.stdout.flush()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
            client.set_linear_rail_velocity(0.0)
            time.sleep(0.5)
    elif args.command == "get_linear_rail_state":
        print("Monitoring linear rail state (Press Ctrl+C to exit)")
        try:
            while True:
                rail_state = client.get_linear_rail_state()
                sys.stdout.write("\r" + _format_rail_state(rail_state))
                sys.stdout.flush()
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nExiting")
