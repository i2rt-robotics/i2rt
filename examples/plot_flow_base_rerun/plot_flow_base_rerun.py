"""Plot the i2rt flow_base's realtime observation in rerun.

This is a read-only monitor: it never commands the base. It polls odometry and per-wheel
telemetry and streams them to a rerun viewer. A 3D view tracks the base's odometry pose and
trajectory on a ground grid; time-series panels plot position, world- and body-frame velocity,
and the four casters' steer/drive speeds. Drive the base however you like — the gamepad
controller, or examples/drive_flow_base_viser — and watch the observation update live.

Usage:
    # Read from a controller already running on the base (recommended): drive it with the
    # gamepad, observe here. Works on the base or any machine on its network.
    uv run examples/plot_flow_base_rerun/plot_flow_base_rerun.py --host 172.6.2.20

    # Or read a local Vehicle directly over CAN (nothing else may own the bus):
    uv run examples/plot_flow_base_rerun/plot_flow_base_rerun.py

A rerun viewer is spawned automatically. For a headless base, pass --serve and open the printed
URL from a browser, or point an already-running viewer at this process with
--connect rerun+http://<host>:9876/proxy. Quit with Ctrl+C in the terminal.
"""

import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import tyro


@dataclass
class Args:
    host: str | None = None
    """Host running the FlowBase controller server. If unset, read a local Vehicle over CAN."""
    channel: str = "can_flow_base"
    """CAN channel for the local Vehicle (ignored when --host is set)."""
    rate_hz: float = 50.0
    """How often to poll the observation and log it to rerun (Hz)."""
    trajectory_points: int = 2000
    """Max number of past poses kept in the odometry trajectory line."""
    connect: str | None = None
    """Stream to an already-running viewer at this gRPC URL instead of spawning one
    (e.g. rerun+http://127.0.0.1:9876/proxy)."""
    serve: bool = False
    """Serve a rerun web viewer (for a headless base); open the printed URL in a browser."""


class _RemoteReader:
    """Passive portal client: reads odometry and wheel speeds from a running FlowBase controller
    without ever sending a command. (FlowBaseClient streams zero-velocity targets, which would
    override gamepad/teleop input and freeze the base — not what a monitor wants.)"""

    def __init__(self, host: str) -> None:
        import portal

        from i2rt.flow_base.flow_base_controller import BASE_DEFAULT_PORT

        self._client = portal.Client(f"{host}:{BASE_DEFAULT_PORT}")

    def get_odometry(self) -> dict:
        return self._client.get_odometry({}).result()

    def get_wheel_speeds(self) -> dict:
        return self._client.get_wheel_speeds({}).result()

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def make_backend(args: Args) -> object:
    """Read-only telemetry source. A local Vehicle over CAN when --host is unset (it holds zero
    velocity, so it never drives itself), otherwise a passive reader to the remote controller.
    Both expose get_odometry / get_wheel_speeds / close."""
    if args.host is None:
        from i2rt.flow_base.flow_base_controller import Vehicle

        return Vehicle(channel=args.channel, auto_start=True)
    return _RemoteReader(args.host)


# Coordinate-axis colors (X, Y, Z) shared by the world frame and the base's body triad.
AXIS_COLORS = [[230, 80, 80], [80, 200, 120], [90, 130, 235]]
# Per-caster colors; steer/drive of the same caster share a color and differ by line width.
CASTER_COLORS = [[230, 80, 80], [80, 200, 120], [90, 130, 235], [220, 180, 70]]
# name + color for each single-line time series (path -> (legend name, rgb)).
SCALAR_SERIES = {
    "pose/x": ("x (m)", [230, 80, 80]),
    "pose/y": ("y (m)", [80, 200, 120]),
    "pose/theta_deg": ("theta (deg)", [90, 130, 235]),
    "vel_world/vx": ("vx (m/s)", [230, 80, 80]),
    "vel_world/vy": ("vy (m/s)", [80, 200, 120]),
    "vel_world/omega": ("omega (rad/s)", [90, 130, 235]),
    "vel_body/vx": ("vx (m/s)", [230, 80, 80]),
    "vel_body/vy": ("vy (m/s)", [80, 200, 120]),
    "vel_body/omega": ("omega (rad/s)", [90, 130, 235]),
    "loop/rate_hz": ("rate (Hz)", [200, 200, 80]),
}


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/world", name="odometry"),
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/pose", name="pose (x, y / theta)"),
                rrb.TimeSeriesView(origin="/vel_world", name="velocity - world frame"),
                rrb.TimeSeriesView(origin="/vel_body", name="velocity - body frame"),
                rrb.TimeSeriesView(origin="/wheels", name="wheel speeds (rad/s)"),
                rrb.TimeSeriesView(origin="/loop", name="control loop"),
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )


def _connect_viewer(args: Args) -> None:
    rr.init("flow_base_observation")
    blueprint = _blueprint()
    if args.connect is not None:
        rr.connect_grpc(args.connect, default_blueprint=blueprint)
        print(f"[rerun] streaming to viewer at {args.connect}")
    elif args.serve:
        uri = rr.serve_grpc(default_blueprint=blueprint)
        rr.serve_web_viewer(open_browser=False, connect_to=uri)
        print("[rerun] serving web viewer on port 9090 — open http://<this-host>:9090 in a browser")
    else:
        rr.spawn(default_blueprint=blueprint)


def _log_static_scene() -> None:
    """Log everything that never changes: frame orientation, axis triads, and series styling."""
    rr.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "/world/axes",
        rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=[[0.3, 0, 0], [0, 0.3, 0], [0, 0, 0.3]], colors=AXIS_COLORS),
        static=True,
    )
    # Logged under /world/base, so it inherits the base transform and rides along as a body triad.
    rr.log(
        "/world/base/axes",
        rr.Arrows3D(origins=[[0, 0, 0]] * 3, vectors=[[0.4, 0, 0], [0, 0.4, 0], [0, 0, 0.4]], colors=AXIS_COLORS),
        static=True,
    )
    for path, (name, color) in SCALAR_SERIES.items():
        rr.log(path, rr.SeriesLines(names=[name], colors=[color]), static=True)
    for i in range(4):
        rr.log(
            f"wheels/steer/{i}",
            rr.SeriesLines(names=[f"steer c{i}"], colors=[CASTER_COLORS[i]], widths=[1.0]),
            static=True,
        )
        rr.log(
            f"wheels/drive/{i}",
            rr.SeriesLines(names=[f"drive c{i}"], colors=[CASTER_COLORS[i]], widths=[2.5]),
            static=True,
        )


def main(args: Args) -> None:
    backend = make_backend(args)

    _connect_viewer(args)
    _log_static_scene()
    print("[rerun] logging observation — drive the base elsewhere to see it move (Ctrl+C to quit)")

    trajectory: deque[list[float]] = deque(maxlen=args.trajectory_points)
    period = 1.0 / args.rate_hz
    rate_hz = args.rate_hz
    start = time.time()
    last = start

    try:
        while True:
            odo = backend.get_odometry()
            pos = odo["position"]
            x = float(pos["translation"][0])
            y = float(pos["translation"][1])
            z = float(pos["translation"][2])
            theta = float(pos["rotation"])
            vw, vb = odo["velocity"]["world"], odo["velocity"]["body"]
            wheels = backend.get_wheel_speeds()
            steer = np.asarray(wheels["steer"], dtype=float)
            drive = np.asarray(wheels["drive"], dtype=float)

            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                rate_hz = 0.9 * rate_hz + 0.1 / dt

            rr.set_time("elapsed", duration=now - start)

            rr.log("pose/x", rr.Scalars(x))
            rr.log("pose/y", rr.Scalars(y))
            rr.log("pose/z", rr.Scalars(z))
            rr.log("pose/theta_deg", rr.Scalars(np.degrees(theta)))
            rr.log("vel_world/vx", rr.Scalars(float(vw["translation"][0])))
            rr.log("vel_world/vy", rr.Scalars(float(vw["translation"][1])))
            rr.log("vel_world/vz", rr.Scalars(float(vw["translation"][2])))
            rr.log("vel_world/omega", rr.Scalars(float(vw["rotation"])))
            rr.log("vel_body/vx", rr.Scalars(float(vb["translation"][0])))
            rr.log("vel_body/vy", rr.Scalars(float(vb["translation"][1])))
            rr.log("vel_body/vz", rr.Scalars(float(vb["translation"][2])))
            rr.log("vel_body/omega", rr.Scalars(float(vb["rotation"])))
            rr.log("loop/rate_hz", rr.Scalars(rate_hz))
            for i in range(4):
                rr.log(f"wheels/steer/{i}", rr.Scalars(steer[i]))
                rr.log(f"wheels/drive/{i}", rr.Scalars(drive[i]))

            rr.log(
                "/world/base",
                rr.Transform3D(translation=[x, y, z], rotation=rr.RotationAxisAngle([0, 0, 1], radians=theta)),
            )
            trajectory.append([x, y, z])
            rr.log("/world/trajectory", rr.LineStrips3D([list(trajectory)], colors=[[255, 170, 0]]))

            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        print("\n[flow_base] observer closed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
