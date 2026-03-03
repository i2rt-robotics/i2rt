"""MuJoCo control interface for i2rt robots.

Starts in gravity-comp (vis-only) mode, mirroring the robot's joint state.
Press SPACE in the viewer to toggle mocap control mode, then double-click
the target marker, ctrl+right-drag to translate, ctrl+left-drag to rotate.

See examples/control_with_mujoco/ for a runnable entry-point and README.
"""

import os
import tempfile
import time
import xml.etree.ElementTree as ET
from enum import Enum, auto
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from i2rt.robots.kinematics import Kinematics
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot


class Mode(Enum):
    VIS = auto()
    CONTROL = auto()


# Button indicator colours  (inactive → active)
_BTN_OFF_RGBA = np.array([0.35, 0.35, 0.35, 0.7])
_BTN_ON_RGBA = np.array([0.1, 0.9, 0.1, 1.0])

# World-space positions of the two button indicator spheres (beside the robot base)
_BTN_POSITIONS = ["0.13 0 0.06", "0.20 0 0.06"]
_BTN_NAMES = ["btn_top_geom", "btn_grip_geom"]


class MujocoControlInterface:
    """MuJoCo viewer with gravity-comp visualisation and mocap IK control.

    For robots with a teaching handle, two small spheres are rendered near the
    base showing the live state of the two handle buttons (green = pressed).
    """

    MOCAP_BODY_NAME = "mocap_target"
    _VIS_RGBA = np.array([0.2, 0.8, 0.2, 0.3])
    _CTRL_RGBA = np.array([0.9, 0.2, 0.2, 0.6])

    def __init__(
        self,
        robot: Robot,
        xml_path: str,
        ee_site: str = "grasp_site",
        dt: float = 0.02,
    ):
        self._robot = robot
        self._ee_site = ee_site
        self._dt = dt
        self._mode = Mode.VIS

        # tcp_site is exclusive to the teaching handle; works for both sim and real hardware
        with_buttons = ee_site == "tcp_site" or self._has_teaching_handle(robot)
        self._model = self._build_model(xml_path, with_button_indicators=with_buttons)
        self._data = mujoco.MjData(self._model)
        self._kin = Kinematics(xml_path, ee_site)

        self._nq = self._model.nq
        self._n_arm = sum(1 for j in range(self._model.njnt) if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE)

        # Validate ee_site
        self._ee_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if self._ee_site_id == -1:
            available = [mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self._model.nsite)]
            raise ValueError(f"Site {ee_site!r} not found in model. Available: {available}")

        self._mocap_id = self._model.body(self.MOCAP_BODY_NAME).mocapid[0]
        self._mocap_geom_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, "mocap_target_geom")

        # Button indicator geom IDs — only populated when teaching handle is present
        if with_buttons:
            self._btn_geom_ids = [
                mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in _BTN_NAMES
            ]
        else:
            self._btn_geom_ids = []

    @classmethod
    def from_robot(
        cls,
        robot: MotorChainRobot,
        ee_site: str = "grasp_site",
        dt: float = 0.02,
    ) -> "MujocoControlInterface":
        return cls(robot, robot.xml_path, ee_site, dt)

    # ---- model construction ---------------------------------------------------

    @staticmethod
    def _has_teaching_handle(robot: Robot) -> bool:
        """Return True if robot has a teaching handle (needs button indicators)."""
        if not isinstance(robot, MotorChainRobot):
            return False
        chain = robot.motor_chain
        return (
            hasattr(chain, "get_same_bus_device_states")
            and hasattr(chain, "same_bus_device_driver")
            and chain.same_bus_device_driver is not None
        )

    @classmethod
    def _build_model(cls, xml_path: str, with_button_indicators: bool = False) -> mujoco.MjModel:
        """Load robot XML and inject mocap target + optionally button indicator geoms."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")

        # Mocap target marker
        mocap = ET.SubElement(worldbody, "body")
        mocap.set("name", cls.MOCAP_BODY_NAME)
        mocap.set("mocap", "true")
        mocap.set("pos", "0 0 0.3")
        geom = ET.SubElement(mocap, "geom")
        geom.set("name", "mocap_target_geom")
        geom.set("type", "sphere")
        geom.set("size", "0.02")
        geom.set("rgba", " ".join(f"{v:.2f}" for v in cls._VIS_RGBA))
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

        # Button indicator spheres — only for teaching handle
        if with_button_indicators:
            for name, pos in zip(_BTN_NAMES, _BTN_POSITIONS, strict=False):
                btn = ET.SubElement(worldbody, "geom")
                btn.set("name", name)
                btn.set("type", "sphere")
                btn.set("size", "0.022")
                btn.set("pos", pos)
                btn.set("rgba", " ".join(f"{v:.2f}" for v in _BTN_OFF_RGBA))
                btn.set("contype", "0")
                btn.set("conaffinity", "0")

        dir_path = os.path.dirname(os.path.abspath(xml_path))
        fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=dir_path)
        os.close(fd)
        try:
            tree.write(tmp_path, xml_declaration=True)
            model = mujoco.MjModel.from_xml_path(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return model

    # ---- helpers --------------------------------------------------------------

    def _sync_mocap_to_ee(self) -> None:
        """Snap the mocap marker to the current end-effector pose."""
        site = self._data.site(self._ee_site_id)
        self._data.mocap_pos[self._mocap_id] = site.xpos.copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, site.xmat.flatten())
        self._data.mocap_quat[self._mocap_id] = quat

    def _mocap_pose_4x4(self) -> np.ndarray:
        """Return the mocap target pose as a 4x4 homogeneous matrix."""
        pos = self._data.mocap_pos[self._mocap_id]
        quat = self._data.mocap_quat[self._mocap_id]
        mat = np.eye(4)
        mat[:3, 3] = pos
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, quat)
        mat[:3, :3] = rot.reshape(3, 3)
        return mat

    def _mirror_robot(self) -> None:
        """Copy robot joint positions into the MuJoCo model and forward-compute."""
        qpos = self._robot.get_joint_pos()
        n = min(len(qpos), self._nq)
        self._data.qpos[:n] = qpos[:n]
        self._denormalize_slide_joints(n)
        self._enforce_eq_constraints()
        mujoco.mj_forward(self._model, self._data)

    def _denormalize_slide_joints(self, n_set: int) -> None:
        """Scale normalized [0,1] slide joint values to physical range (metres)."""
        for j in range(self._model.njnt):
            adr = self._model.jnt_qposadr[j]
            if adr >= n_set:
                continue
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self._model.jnt_range[j]
                self._data.qpos[adr] = lo + self._data.qpos[adr] * (hi - lo)

    def _enforce_eq_constraints(self) -> None:
        """Project qpos to satisfy joint equality constraints (e.g. coupled fingers)."""
        for i in range(self._model.neq):
            if self._model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            adr1 = self._model.jnt_qposadr[self._model.eq_obj1id[i]]
            adr2 = self._model.jnt_qposadr[self._model.eq_obj2id[i]]
            coef = self._model.eq_data[i, :5]
            q1 = self._data.qpos[adr1]
            self._data.qpos[adr2] = np.polyval(coef[::-1], q1)

    def _set_marker_color(self, rgba: np.ndarray) -> None:
        self._model.geom_rgba[self._mocap_geom_id] = rgba

    def _get_button_states(self) -> Optional[list[bool]]:
        """Read teaching-handle button states from real hardware, or None if unavailable."""
        if not isinstance(self._robot, MotorChainRobot):
            return None
        chain = self._robot.motor_chain
        if not hasattr(chain, "get_same_bus_device_states"):
            return None
        states = chain.get_same_bus_device_states()
        if not states:
            return None
        return list(states[0].io_inputs)  # [button_top, button_grip]

    def _update_button_indicators(self) -> None:
        """Refresh button indicator sphere colours from live hardware state."""
        buttons = self._get_button_states()
        for idx, geom_id in enumerate(self._btn_geom_ids):
            if geom_id == -1:
                continue
            pressed = bool(buttons[idx]) if (buttons and idx < len(buttons)) else False
            self._model.geom_rgba[geom_id] = _BTN_ON_RGBA if pressed else _BTN_OFF_RGBA

    # ---- key callback ---------------------------------------------------------

    def _on_key(self, key: int) -> None:
        if key != 32:  # SPACE
            return
        if self._mode is Mode.VIS:
            self._mode = Mode.CONTROL
            self._sync_mocap_to_ee()
            self._set_marker_color(self._CTRL_RGBA)
            print(
                "[control] CONTROL mode — double-click target, ctrl+right-drag to translate, ctrl+left-drag to rotate"
            )
        else:
            self._mode = Mode.VIS
            self._set_marker_color(self._VIS_RGBA)
            if isinstance(self._robot, MotorChainRobot):
                n = self._robot.num_dofs()
                self._robot.update_kp_kd(np.zeros(n), np.zeros(n))
            print("[control] VIS mode — gravity comp, mirroring robot")

    # ---- main loop ------------------------------------------------------------

    def run(self) -> None:
        """Open the viewer and run the vis / control loop."""
        print("[control] Starting in VIS (gravity comp) mode")
        print("[control] Press SPACE to toggle CONTROL mode")

        with mujoco.viewer.launch_passive(
            self._model,
            self._data,
            key_callback=self._on_key,
        ) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            try:
                while viewer.is_running():
                    self._mirror_robot()
                    self._update_button_indicators()

                    if self._mode is Mode.VIS:
                        self._sync_mocap_to_ee()
                    else:
                        target = self._mocap_pose_4x4()
                        init_q = self._data.qpos[: self._nq].copy()
                        ok, ik_q = self._kin.ik(target, self._ee_site, init_q=init_q)
                        if ok:
                            cmd = self._robot.get_joint_pos().copy()
                            cmd[: self._n_arm] = ik_q[: self._n_arm]
                            self._robot.command_joint_pos(cmd)

                    viewer.sync()
                    time.sleep(self._dt)
            except KeyboardInterrupt:
                pass

        print("[control] Viewer closed")
