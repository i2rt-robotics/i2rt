# lerobot/robots/i2rt/i2rt_robot.py
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import portal  # pip install portal-python (same one used by minimum_gello)
from lerobot.cameras.utils import make_cameras_from_configs

from lerobot.robots import Robot
from config import I2RTFollowerConfig, i2rtLeaderConfig, LeaderArmCfg
from lerobot.teleoperators import Teleoperator

# ---- Thin client matching minimum_gello ServerRobot bindings ----
class PortalFollowerClient:
    def __init__(self, host: str, port: int):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        return int(self._client.num_dofs().result())

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray(self._client.get_joint_pos().result(), dtype=np.float32)

    def command_joint_pos(self, q: np.ndarray) -> None:
        self._client.command_joint_pos(np.asarray(q, dtype=np.float32))

    def command_joint_state(self, state: Dict[str, np.ndarray]) -> None:
        self._client.command_joint_state({k: np.asarray(v) for k, v in state.items()})

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self._client.get_observations().result()

class PortalLeaderTeleop(Teleoperator):
    """
    Reads joint positions for multiple arms (left/right) from Portal leader-state server(s)
    and emits LeRobot-style actions: '{prefix}.j{k}.pos' -> float.
    """
    config_class = i2rtLeaderConfig
    name = "portal_leader_teleop"
    
    def __init__(self, cfg: i2rtLeaderConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self._clients: Dict[Tuple[str, int], portal.Client] = {}
        self._arm_dofs: Dict[str, int] = {}
        self._action_keys: List[str] = []
        self._ema_state: Dict[str, np.ndarray] = {}  # Initialize EMA state
        self._configured: bool = False
        self._calibrated: bool = False

    def configure(self, **kwargs) -> None:
        self._configured = True

    def calibrate(self, **kwargs) -> None:
        # No calibration needed for read-only teleop; mark as done.
        self._calibrated = True

    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def action_features(self) -> Dict[str, type]:
        # Map every action key to its dtype (float)
        return {k: float for k in self._action_keys}

    @property
    def feedback_features(self) -> Dict[str, type]:
        # No haptic/force feedback channel for this teleop
        return {}

    def send_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        # No-op: nothing to send back to the leader hardware
        return feedback
    
    @property
    def is_connected(self) -> bool:
        return len(self._clients) > 0

    def _client(self, host: str, port: int) -> portal.Client:
        key = (host, port)
        if key not in self._clients:
            self._clients[key] = portal.Client(f"{host}:{port}")
        return self._clients[key]

    def connect(self) -> None:
        for arm in self.cfg.arms:
            cli = self._client(arm.host, arm.port)
            try:
                dofs = int(cli.num_dofs().result())
                self._arm_dofs[arm.prefix] = dofs
                print(f"✓ Connected to {arm.prefix} arm: {dofs} DOFs")
            except Exception as e:
                print(f"✗ Failed to connect to {arm.prefix} arm: {e}")
                raise

    def disconnect(self) -> None:
        self._clients.clear()
        self._arm_dofs.clear()
        self._ema_state.clear()

    def _read_arm(self, arm: LeaderArmCfg) -> np.ndarray:
        cli = self._client(arm.host, arm.port)
        q_all = np.asarray(cli.get_joint_pos().result(), dtype=np.float32)

        nd = self._arm_dofs.get(arm.prefix, -1)
        q = q_all if nd <= 0 else q_all[:nd]

        return q

    def get_action(self) -> Dict[str, float]:
        act: Dict[str, float] = {}
        for arm in self.cfg.arms:
            q = self._read_arm(arm)
            for j, v in enumerate(q):
                act[f"{arm.prefix}.j{j}.pos"] = float(v)
        return act
    
# ---- LeRobot adapter ----
class I2RTRobot(Robot):
    """LeRobot-compatible adapter for portal-based I2RT followers."""
    config_class = I2RTFollowerConfig
    name = "i2rt"

    def __init__(self, config: I2RTFollowerConfig):
        super().__init__(config)
        self.config = config

        # simple lifecycle flags required by abstract API
        self._configured: bool = False
        self._calibrated: bool = False

        self.cameras = make_cameras_from_configs(config.cameras)

        # One portal client per follower endpoint
        self._followers: Dict[str, PortalFollowerClient] = {}
        self._follower_dofs: Dict[str, int] = {}

        self._motor_keys: List[str] = []
        self.logs: Dict[str, float] = {}

    # ---------- abstract API required by base Robot ----------
    def configure(self, **kwargs) -> None:
        self._configured = True

    def calibrate(self, **kwargs) -> None:
        self._calibrated = True

    def is_calibrated(self) -> bool:
        """Return whether the robot is calibrated."""
        return self._calibrated

    # ---------- Feature schemas ----------
    @property
    def camera_features(self) -> Dict[str, tuple[int | None, int | None, int]]:
        return {cam: (c.height, c.width, 3) for cam, c in self.cameras.items()}

    @property
    def motors_features(self) -> Dict[str, type]:
        if not self._motor_keys:
            return {}
        return {k: float for k in self._motor_keys}

    @property
    def observation_features(self) -> Dict[str, Any]:
        return {**self.motors_features, **self.camera_features}

    @property
    def action_features(self) -> Dict[str, type]:
        return self.motors_features

    # ---------- Lifecycle ----------
    @property
    def is_connected(self) -> bool:
        return len(self._followers) == len(self.config.followers)

    def connect(self) -> None:
        if not self._configured:
            self.configure()

        # Connect to each follower and discover DOFs
        for ep in self.config.followers:
            print(f"Connecting to follower {ep.name} at {ep.host}:{ep.port}")
            try:
                client = PortalFollowerClient(ep.host, ep.port)
                dofs = client.num_dofs()
                self._followers[ep.name] = client
                self._follower_dofs[ep.name] = dofs
                print(f"✓ Connected to {ep.name}: {dofs} DOFs")
            except Exception as e:
                print(f"✗ Failed to connect to {ep.name}: {e}")

        # Build flat motor key list (e.g., "right.j0.pos", ...)
        keys: List[str] = []
        for ep in self.config.followers:
            prefix = getattr(ep, "prefix", None) or ep.name
            dofs = self._follower_dofs[ep.name]
            print(f"Building motor keys for {ep.name}: {dofs} DOFs, prefix={prefix}")
            
            if dofs == 0:
                print(f"  - WARNING: {ep.name} has 0 DOFs, using default 7 DOFs")
                dofs = 7  # Default to 7 DOFs for testing
            
            for j in range(dofs):
                key = f"{prefix}.j{j}.pos"
                keys.append(key)
                print(f"  - Added key: {key}")
        self._motor_keys = keys
        print(f"Total motor keys: {len(self._motor_keys)}")
        print(f"Motor keys: {self._motor_keys}")

        # Cameras
        for cam in self.cameras.values():
            cam.connect()

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()

    # ---------- Helpers ----------
    def _split_action_by_follower(self, flat_action: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Group flat '<name>.jK.pos' into per-follower joint arrays in follower order."""
        per_follow: Dict[str, List[float]] = {ep.name: [] for ep in self.config.followers}
        current = self._read_all_joint_pos()

        # initialize with current, then overwrite keys provided in action
        for ep in self.config.followers:
            per_follow[ep.name] = list(current[ep.name])

        for key, val in flat_action.items():
            # Expect "<prefix>.jK.pos"
            prefix, rest = key.split(".", 1)
            j_str = rest.split(".", 1)[0]  # "jK"
            j_idx = int(j_str[1:])
            per_follow[prefix][j_idx] = float(val)

        return {name: np.asarray(vals, dtype=np.float32) for name, vals in per_follow.items()}

    def _read_all_joint_pos(self) -> Dict[str, np.ndarray]:
        result = {}
        for ep in self.config.followers:
            if ep.name in self._followers:
                result[ep.name] = self._followers[ep.name].get_joint_pos()
            else:
                # Return zeros for not connected follower
                dofs = self._follower_dofs.get(ep.name, 7)
                result[ep.name] = np.zeros(dofs, dtype=np.float32)
        return result

    # ---------- I/O ----------
    def get_observation(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, Any] = {}

        # Motors
        all_q = self._read_all_joint_pos()
        for ep in self.config.followers:
            prefix = getattr(ep, "prefix", None) or ep.name
            q = np.asarray(all_q[ep.name], dtype=np.float32).ravel()
            for j, v in enumerate(q):
                obs[f"{prefix}.j{j}.pos"] = float(v)
        
        # Cameras
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
        
        return obs

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Position control: expects flat dict of '<name>.jK.pos': float."""
        if not action:
            return action

        per_follow = self._split_action_by_follower(action)
        for ep in self.config.followers:
            q_cmd = per_follow[ep.name]
            self._followers[ep.name].command_joint_pos(q_cmd)
        return action
