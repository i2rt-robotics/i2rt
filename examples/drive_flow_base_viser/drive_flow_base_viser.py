"""Drive the i2rt flow_base with GUI sliders while visualizing its odometry in viser.

The base is disabled at startup: no velocity commands are sent until you check "enable
robot" in the panel. Three velocity sliders then set the base's local-frame velocity
directly: vx (forward/back), vy (strafe), and ω (yaw). A coordinate axis in the browser
scene tracks the base's odometry pose on a ground grid, and the panel shows a live
control-loop rate (Hz). Uncheck "enable robot" to halt the base; press Stop to re-center
the sliders; quit with Ctrl+C in the terminal.

Usage:
    # On a machine on the base's CAN bus (drives a local Vehicle):
    uv run examples/drive_flow_base_viser/drive_flow_base_viser.py

    # From a workstation, talking to a controller already running on the base:
    uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --host 172.6.2.20

Then open http://localhost:8080 in a browser.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import tyro
import viser


@dataclass
class Args:
    host: str | None = None
    """Host running the FlowBase controller server. If unset, drive a local Vehicle over CAN."""
    channel: str = "can_flow_base"
    """CAN channel for the local Vehicle (ignored when --host is set)."""
    port: int = 8080
    """Viser server port."""
    max_linear: float = 0.5
    """The vx / vy sliders range over ±this (m/s)."""
    max_angular: float = 1.57
    """The ω slider ranges over ±this (rad/s)."""
    control_hz: float = 50.0
    """Control-loop rate (Hz): how often the command is sent and the scene refreshed. Keep it
    well above the base's 0.25 s (4 Hz) command-safety timeout."""


def make_backend(args: Args) -> object:
    """Local Vehicle when --host is unset, otherwise a remote FlowBaseClient. Both expose
    set_target_velocity / get_odometry / reset_odometry / close."""
    if args.host is None:
        from i2rt.flow_base.flow_base_controller import Vehicle

        return Vehicle(
            max_vel=(args.max_linear, args.max_linear, args.max_angular),
            channel=args.channel,
            auto_start=True,
        )
    from i2rt.flow_base.flow_base_client import FlowBaseClient

    return FlowBaseClient(
        host=args.host,
        with_linear_rail=False,
        max_vel_x=args.max_linear,
        max_vel_y=args.max_linear,
        max_vel_theta=args.max_angular,
    )


def yaw_to_wxyz(theta: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for a rotation of `theta` about +Z."""
    half = 0.5 * theta
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def main(args: Args) -> None:
    backend = make_backend(args)

    server = viser.ViserServer(host="0.0.0.0", port=args.port, label="flow_base teleop")
    server.scene.add_grid("/ground", width=10.0, height=10.0, plane="xy")
    server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.01)
    base_frame = server.scene.add_frame("/base", axes_length=0.4, axes_radius=0.02)

    vx = server.gui.add_slider(
        "vx — forward (m/s)", min=-args.max_linear, max=args.max_linear, step=0.05, initial_value=0.0
    )
    vy = server.gui.add_slider(
        "vy — strafe left (m/s)", min=-args.max_linear, max=args.max_linear, step=0.05, initial_value=0.0
    )
    wz = server.gui.add_slider(
        "ω — turn left (rad/s)", min=-args.max_angular, max=args.max_angular, step=0.05, initial_value=0.0
    )
    enable = server.gui.add_checkbox("enable robot", initial_value=False)
    stop_button = server.gui.add_button("stop")
    reset_button = server.gui.add_button("reset odometry")
    status = server.gui.add_markdown("")

    @stop_button.on_click
    def _(_: object) -> None:
        vx.value = 0.0
        vy.value = 0.0
        wz.value = 0.0

    @reset_button.on_click
    def _(_: object) -> None:
        backend.reset_odometry()

    print(
        f"[viser] open http://localhost:{args.port} — check 'enable robot', then drag the sliders "
        "to drive (Ctrl+C to quit)"
    )

    period = 1.0 / args.control_hz
    comm_hz = args.control_hz
    last = time.time()
    was_enabled = False

    try:
        while True:
            cmd = np.array([float(vx.value), float(vy.value), float(wz.value)])
            enabled = bool(enable.value)
            if enabled:
                backend.set_target_velocity(cmd, frame="local")
            elif was_enabled:
                # just disabled — halt now instead of coasting on the base's safety timeout
                backend.set_target_velocity(np.zeros(3), frame="local")
            was_enabled = enabled

            odo = backend.get_odometry()
            x, y, z = (float(v) for v in odo["position"]["translation"])
            theta = float(odo["position"]["rotation"])
            base_frame.position = np.array([x, y, z])
            base_frame.wxyz = yaw_to_wxyz(theta)

            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                comm_hz = 0.9 * comm_hz + 0.1 / dt

            state = "ENABLED — sending commands" if enabled else "DISABLED — idle"
            status.content = (
                f"**state**  {state}\n\n"
                f"**pose**  x = {x:+.3f} m   y = {y:+.3f} m   z = {z:+.3f} m   θ = {np.degrees(theta):+.1f}°\n\n"
                f"**cmd**  vx = {cmd[0]:+.2f}   vy = {cmd[1]:+.2f}   ω = {cmd[2]:+.2f}\n\n"
                f"**comm**  {comm_hz:.1f} Hz"
            )

            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        backend.set_target_velocity(np.zeros(3), frame="local")
        backend.close()
        print("\n[flow_base] stopped and closed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
