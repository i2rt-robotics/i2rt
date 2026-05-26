#!/usr/bin/env python3

"""
Flow Base live plot (Redis -> Dash)

How to run:
1) Start the flow base controller (enable telemetry only if you need plotting):
   - Example:
     python submodules/i2rt/i2rt/flow_base/flow_base_controller.py --plot-traj --redis-host localhost

2) Start this plot script:
   - Example:
     python xdof/scripts/plot_flow_base_state.py --redis-host localhost

If you see: "Failed to set real-time scheduling policy, please edit /etc/security/limits.d/99-realtime.conf"

Method 1 (Recommended): configure system security limits once
  - Create a group named 'realtime' (if not exists):
    sudo groupadd realtime
  - Add your current user to this group:
    sudo usermod -aG realtime $USER
    Note: log out and log back in (or reboot) to take effect.
  - Create the limits config file:
    sudo nano /etc/security/limits.d/99-realtime.conf
  - Put the following lines in the file, then save:
    @realtime   -   rtprio  99
    @realtime   -   memlock unlimited

Method 2 (Quick but not recommended): run with sudo temporarily
  sudo python submodules/i2rt/i2rt/flow_base/flow_base_controller.py
"""

import argparse
import atexit
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import plotly.graph_objects as go
import redis
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


def parse_vec3(s: str) -> List[float]:
    parts = [p for p in s.split(" ") if p]
    if len(parts) < 3:
        return []
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def make_odom_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Flow Base Odometry (XY)",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def make_vel_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title="Flow Base Velocity",
        xaxis_title="Time (s)",
        yaxis_title="Velocity",
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot flow base odometry/velocity from Redis using Plotly Dash")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host (default: localhost)")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port (default: 6379)")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis DB index (default: 0)")
    parser.add_argument("--history", type=int, default=20000, help="History length (default: 1000)")
    parser.add_argument("--interval_ms", type=int, default=100, help="UI refresh interval ms (default: 100)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Dash host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8051, help="Dash port (default: 8051)")
    args = parser.parse_args()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)

    # Ring buffers
    hist_xy: Deque[Tuple[float, float]] = deque(maxlen=args.history)
    hist_t: Deque[float] = deque(maxlen=args.history)
    hist_dx: Deque[Tuple[float, float, float]] = deque(maxlen=args.history)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H4("Flow Base — Live Plot (Redis -> Dash)"),
            html.Div(
                [
                    dcc.Graph(id="odom-graph", figure=make_odom_figure(), style={"height": "48vh"}),
                    dcc.Graph(id="vel-graph", figure=make_vel_figure(), style={"height": "38vh"}),
                ]
            ),
            dcc.Interval(id="tick", interval=args.interval_ms, n_intervals=0),
            html.Div(id="status"),
        ]
    )

    @app.callback(
        Output("odom-graph", "figure"),
        Output("vel-graph", "figure"),
        Output("status", "children"),
        Input("tick", "n_intervals"),
    )
    def update(_n: int) -> tuple[go.Figure, go.Figure, str]:  # type: ignore
        # Pull one sample per tick if available
        x_raw = r.get("x")
        dx_raw = r.get("dx")

        if isinstance(x_raw, str) and isinstance(dx_raw, str):
            x_vec = parse_vec3(x_raw)
            dx_vec = parse_vec3(dx_raw)
            if len(x_vec) == 3 and len(dx_vec) == 3:
                hist_xy.append((x_vec[0], x_vec[1]))
                hist_t.append(float(len(hist_t)))
                hist_dx.append((dx_vec[0], dx_vec[1], dx_vec[2]))

        # Odometry fig
        fig_o = make_odom_figure()
        if len(hist_xy) > 0:
            arr_xy = np.array(hist_xy, dtype=np.float32)
            fig_o.add_trace(go.Scatter(x=arr_xy[:, 0], y=arr_xy[:, 1], mode="lines", line=dict(color="#1f77b4", width=3)))
            fig_o.add_trace(
                go.Scatter(x=[arr_xy[-1, 0]], y=[arr_xy[-1, 1]], mode="markers", marker=dict(color="#d62728", size=8))
            )

        # Velocity fig
        fig_v = make_vel_figure()
        if len(hist_dx) > 0:
            t = np.array(hist_t, dtype=np.float32)
            arr_dx = np.array(hist_dx, dtype=np.float32)
            fig_v.add_trace(go.Scatter(x=t, y=arr_dx[:, 0], name="vx", mode="lines"))
            fig_v.add_trace(go.Scatter(x=t, y=arr_dx[:, 1], name="vy", mode="lines"))
            fig_v.add_trace(go.Scatter(x=t, y=arr_dx[:, 2], name="vθ", mode="lines"))

        status = f"Redis {args.redis_host}:{args.redis_port}/{args.redis_db} | points={len(hist_xy)}"
        return fig_o, fig_v, status

    def _cleanup() -> None:
        r.close()

    atexit.register(_cleanup)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
