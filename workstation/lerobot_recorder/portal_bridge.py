"""Portal bridge: poll the YAM robot server, expose the latest fused snapshot.

Connects (plain TCP) to the robot machine running
``i2rt.serving.run_robot_server`` and polls ``get_observation()``. ``get_snapshot()``
returns the latest fused frame the recorder records:

    {teleop_state, control_mode, state(42,), leader, eef, action, buttons,
     intervention, stamp}

``state``/``leader``/``eef``/``action`` are float32 vectors (or None if missing).
``mock=True`` synthesizes a teleop cycle + fake joints so the pipeline runs with no
robot.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import numpy as np

from workstation.lerobot_recorder.config import ARM_DOF, ARMS, CONTROL_MODE, RecorderConfig

logger = logging.getLogger(__name__)

_EMPTY = {
    "teleop_state": "IDLE",
    "control_mode": CONTROL_MODE["teleop"],
    "state": None,
    "leader": None,
    "eef": None,
    "action": None,
    "buttons": {},
    "intervention": False,
    "stamp": 0.0,
}


class PortalBridge:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._snap = dict(_EMPTY)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client = None
        self._connected = False
        self._estop_req = False
        self._estop_sent: Optional[bool] = None
        self._last_err: Optional[str] = None

    @property
    def connected(self) -> bool:
        """True once the robot server has answered (always True in mock)."""
        return self.cfg.mock or self._connected

    def set_estop(self, flag: bool) -> None:
        """Request a robot e-stop; applied (and re-applied on reconnect) by the poll loop."""
        self._estop_req = bool(flag)

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        target = self._mock_loop if self.cfg.mock else self._poll_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_snapshot(self) -> dict:
        with self._lock:
            return dict(self._snap)

    def _server_reachable(self) -> bool:
        """Fast TCP preflight. portal's ``send`` blocks forever waiting to connect, so
        we never construct the client until something is actually listening — this is
        what stops the recorder from hanging silently against a down/unreachable server."""
        import socket

        try:
            with socket.create_connection((self.cfg.robot_host, self.cfg.robot_port), timeout=1.0):
                return True
        except OSError:
            return False

    # ------------------------------------------------------------------ poll
    def _poll_loop(self) -> None:
        # Connect inside the thread so start() never blocks the GUI if the robot
        # server isn't up yet; on a dropped connection, reconnect on the next tick.
        from i2rt.serving.robot_client import RobotClient

        period = 1.0 / max(self.cfg.fps * 2, 1)
        while not self._stop.is_set():
            try:
                if self._client is None:
                    if not self._server_reachable():
                        raise ConnectionError(
                            f"no server listening on {self.cfg.robot_host}:{self.cfg.robot_port}"
                        )
                    # TCP is open; finite timeout guards the (now non-blocking) handshake/reads.
                    self._client = RobotClient(
                        host=self.cfg.robot_host, port=self.cfg.robot_port, timeout=2.0
                    )
                obs = self._client.get_observation()
                with self._lock:
                    self._snap = self._assemble(obs)
                if not self._connected:
                    logger.info("robot connected (%s:%d)", self.cfg.robot_host, self.cfg.robot_port)
                self._connected = True
                self._last_err = None
                if self._estop_sent != self._estop_req:  # apply e-stop (and re-apply on reconnect)
                    self._client.set_estop(self._estop_req)
                    self._estop_sent = self._estop_req
            except Exception as e:
                self._client = None  # drop and retry next tick
                self._connected = False
                self._estop_sent = None  # force re-apply after reconnect
                # Surface *why* the link is down instead of failing silently — but only
                # when the reason changes, so a persistently-down server doesn't spam.
                msg = f"{type(e).__name__}: {e}"
                if msg != self._last_err:
                    self._last_err = msg
                    logger.warning("robot link down (%s:%d) — %s", self.cfg.robot_host, self.cfg.robot_port, msg)
            time.sleep(period)

    @staticmethod
    def _fuse(sides: list, fields: tuple, per_arm: Optional[int] = None) -> Optional[np.ndarray]:
        """Concatenate ``fields`` over both arms, or None if any is missing.

        With ``per_arm`` set, also require each arm's vector to match that size
        (used for the fixed-width state/action); otherwise accept whatever dim the
        robot provides (leader / eef, which vary by embodiment).
        """
        parts = []
        for s in sides:
            if not s or any(s.get(f) is None for f in fields):
                return None
            vec = np.concatenate([np.asarray(s[f], dtype=np.float32).reshape(-1) for f in fields])
            if per_arm is not None and vec.size != per_arm:
                return None
            parts.append(vec)
        return np.concatenate(parts).astype(np.float32)

    def _assemble(self, obs: Dict) -> dict:
        sides = [obs.get(a) for a in ARMS]
        state = self._fuse(sides, ("pos", "vel", "eff"), ARM_DOF * 3)
        leader = self._fuse(sides, ("leader_pos",))  # variable per-arm dof; saved when present
        eef = self._fuse(sides, ("eef",))  # follower end-effector pose (FK), when the robot provides it
        intervening = bool(obs.get("intervention"))
        buttons = {a: (obs.get(a, {}) or {}).get("buttons", []) for a in ARMS}

        if self.cfg.record_source == "dagger":
            # HG-DAgger: an episode = an intervention segment; the label is the
            # human (leader) action, recorded only while intervening.
            teleop_state = "ENGAGED" if intervening else "IDLE"
            action = self._fuse(sides, ("human",), ARM_DOF) if intervening else None
            control_mode = CONTROL_MODE["intervention"] if intervening else CONTROL_MODE["policy"]
        elif self.cfg.record_source == "eval":
            # Evaluation rollout: record the executed action every tick, labeled by
            # who produced it (policy vs human intervention). Episode = arm..disarm.
            teleop_state = "ENGAGED"
            action = self._fuse(sides, ("applied",), ARM_DOF)
            control_mode = CONTROL_MODE["intervention"] if intervening else CONTROL_MODE["policy"]
        else:
            teleop_state = obs.get("teleop_state") or ("ENGAGED" if intervening else "IDLE")
            action = self._fuse(sides, ("applied",), ARM_DOF)
            control_mode = CONTROL_MODE["teleop"]

        return {
            "teleop_state": teleop_state,
            "control_mode": int(control_mode),
            "state": state,
            "leader": leader,
            "eef": eef,
            "action": action,
            "buttons": buttons,
            "intervention": intervening,
            "stamp": float(obs.get("t", 0.0)),
        }

    # ------------------------------------------------------------------ mock
    def _mock_loop(self) -> None:
        cycle = [("IDLE", 1.0), ("ENGAGED", 3.0), ("HOMING", 1.5)]
        i = 0
        t0 = time.time()
        seg_start = t0
        while not self._stop.is_set():
            name, dur = cycle[i]
            now = time.time()
            phase = now - t0
            pos = 0.3 * np.sin(phase + np.arange(ARM_DOF))
            vel = 0.3 * np.cos(phase + np.arange(ARM_DOF))
            eff = 0.1 * np.ones(ARM_DOF)
            state = np.concatenate([np.concatenate([pos, vel, eff]) for _ in ARMS]).astype(np.float32)
            action = np.concatenate([pos for _ in ARMS]).astype(np.float32)
            leader = np.concatenate([pos[:6] for _ in ARMS]).astype(np.float32)  # 6-dof leader per arm
            with self._lock:
                self._snap = {
                    **_EMPTY,
                    "teleop_state": name,
                    "control_mode": CONTROL_MODE["teleop"],
                    "state": state,
                    "leader": leader,
                    "action": action,
                    "stamp": now,
                }
            if now - seg_start >= dur:
                i = (i + 1) % len(cycle)
                seg_start = now
            time.sleep(0.01)
