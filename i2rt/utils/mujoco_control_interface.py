"""MuJoCo control interface for i2rt robots.

Starts in gravity-comp (vis-only) mode, mirroring the robot's joint state.
Press SPACE in the viewer to toggle control mode, which offers two ways to
command the robot:

  1. **Per-joint sliders** — open the viewer's *Control* panel (right-click
     the viewer → "Control") to adjust each joint individually.
  2. **Mocap IK** — double-click the target marker, then ctrl+right-drag to
     translate and ctrl+left-drag to rotate.

Both inputs stay in sync: moving sliders updates the mocap target via FK,
and dragging the mocap target updates the sliders via IK.

See examples/control_with_mujoco/ for a runnable entry-point and README.
"""

import logging
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

logger = logging.getLogger(__name__)


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
        log: bool = False,
    ):
        self._robot = robot
        self._ee_site = ee_site
        self._dt = dt
        self._mode = Mode.VIS
        self._logging = log
        self._log_prev_time: float = 0.0
        self._log_freq: float | None = None

        n_dofs = robot.num_dofs()
        robot_info = robot.get_robot_info() if hasattr(robot, "get_robot_info") else {}
        self._gripper_index: Optional[int] = robot_info.get("gripper_index")
        self._gripper_limits: Optional[np.ndarray] = robot_info.get("gripper_limits")
        self._is_sim: bool = robot_info.get("sim", False)

        # tcp_site is exclusive to the teaching handle; works for both sim and real hardware
        with_buttons = ee_site == "tcp_site" or self._has_teaching_handle(robot)
        self._model = self._build_model(
            xml_path,
            n_dofs=n_dofs,
            gripper_index=self._gripper_index,
            with_button_indicators=with_buttons,
        )
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
        self._check_data = mujoco.MjData(self._model)  # scratch buffer for collision checks
        self._in_collision = False

        # Slider tracking: detect when the user moves a slider in the Control panel
        self._prev_ctrl = np.zeros(self._model.nu)
        self._prev_mocap_pos = np.zeros(3)
        self._prev_mocap_quat = np.array([1.0, 0.0, 0.0, 0.0])

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
        log: bool = False,
    ) -> "MujocoControlInterface":
        return cls(robot, robot.xml_path, ee_site, dt, log=log)

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
    def _build_model(
        cls,
        xml_path: str,
        n_dofs: int = 0,
        gripper_index: Optional[int] = None,
        with_button_indicators: bool = False,
    ) -> mujoco.MjModel:
        """Load robot XML, inject mocap target, joint actuators, and optionally button geoms.

        Position actuators are added for each joint so the MuJoCo viewer's
        Control panel exposes per-joint sliders (used in CONTROL mode).
        """
        # Load the base model first to read joint info
        base_model = mujoco.MjModel.from_xml_path(xml_path)

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

        # Position actuators for per-joint sliders in the viewer Control panel
        actuator_el = root.find("actuator")
        if actuator_el is None:
            actuator_el = ET.SubElement(root, "actuator")
        for j in range(min(n_dofs, base_model.njnt)):
            jnt_name = mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jnt_name is None:
                continue
            act = ET.SubElement(actuator_el, "position")
            act.set("name", "gripper" if j == gripper_index else jnt_name)
            act.set("joint", jnt_name)
            act.set("kp", "100")
            if j == gripper_index:
                act.set("ctrlrange", "0 1")
                act.set("ctrllimited", "true")
            elif base_model.jnt_limited[j]:
                lo, hi = base_model.jnt_range[j]
                act.set("ctrlrange", f"{lo} {hi}")
                act.set("ctrllimited", "true")

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
        self._denormalize_slide_joints_on(self._data, n_set)

    def _denormalize_slide_joints_on(self, data: mujoco.MjData, n_set: int) -> None:
        """Scale normalized [0,1] slide joint values to physical range (metres)."""
        for j in range(self._model.njnt):
            adr = self._model.jnt_qposadr[j]
            if adr >= n_set:
                continue
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self._model.jnt_range[j]
                data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)

    def _enforce_eq_constraints(self) -> None:
        self._enforce_eq_constraints_on(self._data)

    def _enforce_eq_constraints_on(self, data: mujoco.MjData) -> None:
        """Project qpos to satisfy joint equality constraints (e.g. coupled fingers)."""
        for i in range(self._model.neq):
            if self._model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            adr1 = self._model.jnt_qposadr[self._model.eq_obj1id[i]]
            adr2 = self._model.jnt_qposadr[self._model.eq_obj2id[i]]
            coef = self._model.eq_data[i, :5]
            q1 = data.qpos[adr1]
            data.qpos[adr2] = np.polyval(coef[::-1], q1)

    def _robot_cmd_to_qpos(self, cmd: np.ndarray) -> np.ndarray:
        """Expand a robot-command array (num_dofs) into a full model qpos array (nq).

        For models where nq > num_dofs (e.g. linear grippers with coupled finger
        slide joints), this copies the command into a qpos-sized buffer,
        denormalizes any slide joints, and fills in coupled joints via equality
        constraints.
        """
        qpos = np.zeros(self._nq)
        n = min(len(cmd), self._nq)
        qpos[:n] = cmd[:n]
        # Denormalize slide joints from [0,1] to physical range
        for j in range(self._model.njnt):
            adr = self._model.jnt_qposadr[j]
            if adr >= n:
                continue
            if self._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self._model.jnt_range[j]
                qpos[adr] = lo + qpos[adr] * (hi - lo)
        # Fill in coupled joints via equality constraints
        for i in range(self._model.neq):
            if self._model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            adr1 = self._model.jnt_qposadr[self._model.eq_obj1id[i]]
            adr2 = self._model.jnt_qposadr[self._model.eq_obj2id[i]]
            coef = self._model.eq_data[i, :5]
            qpos[adr2] = np.polyval(coef[::-1], qpos[adr1])
        return qpos

    def _set_marker_color(self, rgba: np.ndarray) -> None:
        self._model.geom_rgba[self._mocap_geom_id] = rgba

    def _has_self_collision(self, target_q: np.ndarray, n: int) -> bool:
        """Return True if *target_q* joint positions would cause self-collision.

        Uses a scratch ``MjData`` so the render state is not corrupted.
        Contacts involving the ground plane or adjacent (parent-child) bodies
        are ignored — only unexpected link-link penetrations count.
        """
        self._check_data.qpos[:n] = target_q[:n]
        self._denormalize_slide_joints_on(self._check_data, n)
        self._enforce_eq_constraints_on(self._check_data)
        mujoco.mj_forward(self._model, self._check_data)
        for i in range(self._check_data.ncon):
            c = self._check_data.contact[i]
            if c.dist >= -1e-3:
                continue
            if (
                self._model.geom_type[c.geom1] == mujoco.mjtGeom.mjGEOM_PLANE
                or self._model.geom_type[c.geom2] == mujoco.mjtGeom.mjGEOM_PLANE
            ):
                continue
            b1 = self._model.geom_bodyid[c.geom1]
            b2 = self._model.geom_bodyid[c.geom2]
            if self._model.body_parentid[b1] == b2 or self._model.body_parentid[b2] == b1:
                continue
            return True
        return False

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

    def _sync_sliders_to_robot(self) -> None:
        """Copy current robot joint positions into the actuator ctrl array (viewer sliders)."""
        qpos = self._robot.get_joint_pos()
        n = min(len(qpos), self._model.nu)
        for i in range(n):
            self._data.ctrl[i] = qpos[i]
        self._prev_ctrl[:] = self._data.ctrl
        self._prev_mocap_pos[:] = self._data.mocap_pos[self._mocap_id]
        self._prev_mocap_quat[:] = self._data.mocap_quat[self._mocap_id]

    def _sliders_changed(self) -> bool:
        """Return True if the user has moved any slider since the last check."""
        return not np.allclose(self._data.ctrl, self._prev_ctrl, atol=1e-6)

    def _mocap_changed(self) -> bool:
        """Return True if the user has moved the mocap target since the last check."""
        pos_changed = not np.allclose(self._data.mocap_pos[self._mocap_id], self._prev_mocap_pos, atol=1e-6)
        quat_changed = not np.allclose(self._data.mocap_quat[self._mocap_id], self._prev_mocap_quat, atol=1e-6)
        return pos_changed or quat_changed

    def _cmd_from_sliders(self) -> np.ndarray:
        """Build a joint-position command array from the current slider (ctrl) values."""
        cmd = self._robot.get_joint_pos().copy()
        n = min(len(cmd), self._model.nu)
        for i in range(n):
            cmd[i] = self._data.ctrl[i]
        return cmd

    def _sync_mocap_to_sliders(self) -> None:
        """Update the mocap target to match the FK of the current slider joint values."""
        q = self._robot_cmd_to_qpos(self._cmd_from_sliders())
        pose = self._kin.fk(q, self._ee_site)
        self._data.mocap_pos[self._mocap_id] = pose[:3, 3]
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, pose[:3, :3].flatten())
        self._data.mocap_quat[self._mocap_id] = quat

    def _sync_sliders_to_ik(self, ik_q: np.ndarray) -> None:
        """Update slider ctrl values to reflect the IK solution (arm joints only)."""
        n = min(len(ik_q), self._model.nu)
        for i in range(n):
            if i == self._gripper_index:
                continue  # don't overwrite gripper slider from IK
            self._data.ctrl[i] = ik_q[i]

    def _update_button_indicators(self) -> None:
        """Refresh button indicator sphere colours from live hardware state."""
        buttons = self._get_button_states()
        for idx, geom_id in enumerate(self._btn_geom_ids):
            if geom_id == -1:
                continue
            pressed = bool(buttons[idx]) if (buttons and idx < len(buttons)) else False
            self._model.geom_rgba[geom_id] = _BTN_ON_RGBA if pressed else _BTN_OFF_RGBA

    # ---- logging -------------------------------------------------------------

    def _compute_sim_torques(self) -> np.ndarray:
        """Compute gravity-compensation torques via MuJoCo inverse dynamics.

        Sets velocity and acceleration to zero so the returned torques are
        exactly the forces needed to hold the current pose against gravity.
        """
        inv_data = mujoco.MjData(self._model)
        inv_data.qpos[:] = self._data.qpos[:]
        inv_data.qvel[:] = 0.0
        inv_data.qacc[:] = 0.0
        mujoco.mj_inverse(self._model, inv_data)
        # qfrc_inverse has one entry per DoF
        return inv_data.qfrc_inverse[: self._n_arm].copy()

    def _format_log_table(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_eff: np.ndarray,
        required: np.ndarray,
        diff: np.ndarray,
        gripper_pos: float | None = None,
        loop_freq: float | None = None,
        can_freq: float | None = None,
        temp_mos: Optional[np.ndarray] = None,
        temp_rotor: Optional[np.ndarray] = None,
    ) -> str:
        """Build a Unicode box-drawing table for joint state, torques, and temperatures."""
        lw, cw = 7, 16  # label width, column width
        tw = 10  # temperature column width
        mode = "SIM" if self._is_sim else "REAL"
        has_temp = temp_mos is not None and temp_rotor is not None

        cols = ["Pos (rad)", "Vel (rad/s)", "Effort (Nm)", "Required (Nm)", "Diff (Nm)"]
        top = f"┌{'─' * lw}" + "".join(f"┬{'─' * cw}" for _ in cols)
        hdr = f"│{'Joint':^{lw}}" + "".join(f"│{c:^{cw}}" for c in cols)
        sep = f"├{'─' * lw}" + "".join(f"┼{'─' * cw}" for _ in cols)
        btm = f"└{'─' * lw}" + "".join(f"┴{'─' * cw}" for _ in cols)
        if has_temp:
            top += f"┬{'─' * tw}┬{'─' * tw}┐"
            hdr += f"│{'MOS °C':^{tw}}│{'Rotor °C':^{tw}}│"
            sep += f"┼{'─' * tw}┼{'─' * tw}┤"
            btm += f"┴{'─' * tw}┴{'─' * tw}┘"
        else:
            top += "┐"
            hdr += "│"
            sep += "┤"
            btm += "┘"

        freq_str = f"  [loop {loop_freq:.1f} Hz]" if loop_freq is not None else ""
        can_str = f"  [CAN {can_freq:.1f} Hz]" if can_freq is not None and can_freq > 0 else ""
        rows = [f" Log [{mode}]{freq_str}{can_str}", top, hdr, sep]

        for i in range(len(joint_pos)):
            line = f"│{f'j{i + 1}':^{lw}}"
            line += f"│{joint_pos[i]:>+12.4f}    "
            line += f"│{joint_vel[i]:>+12.4f}    "
            line += f"│{joint_eff[i]:>+12.4f}    "
            if i < len(required):
                line += f"│{required[i]:>+12.4f}    "
                line += f"│{diff[i]:>+12.4f}    "
            else:
                line += f"│{'':>16}│{'':>16}"
            if has_temp and i < len(temp_mos):
                line += f"│{temp_mos[i]:>{tw - 1}.0f} │{temp_rotor[i]:>{tw - 1}.0f} │"
            elif has_temp:
                line += f"│{'':>{tw}}│{'':>{tw}}│"
            else:
                line += "│"
            rows.append(line)

        if gripper_pos is not None:
            line = f"│{'grip':^{lw}}"
            line += f"│{gripper_pos:>12.4f}    "
            line += f"│{'':>16}" * 4
            if has_temp:
                line += f"│{'':>{tw}}│{'':>{tw}}│"
            else:
                line += "│"
            rows.append(line)

        rows.append(btm)
        return "\n".join(rows)

    def _log(self) -> None:
        """Display joint state and torque table, clearing the console each iteration."""
        now = time.monotonic()
        if self._log_prev_time > 0:
            dt = now - self._log_prev_time
            if dt > 0:
                self._log_freq = 1.0 / dt
        self._log_prev_time = now

        obs = self._robot.get_observations()
        joint_pos = obs["joint_pos"]
        joint_vel = obs["joint_vel"]
        joint_eff = obs["joint_eff"]
        gripper_pos = obs.get("gripper_pos")
        if gripper_pos is not None:
            gripper_pos = float(gripper_pos[0])

        required = self._compute_sim_torques()
        n = min(len(joint_eff[: self._n_arm]), len(required))
        actual_arm = joint_eff[: self._n_arm][:n]
        diff = actual_arm - required[:n]

        can_freq = None
        if hasattr(self._robot, "motor_chain") and hasattr(self._robot.motor_chain, "comm_freq"):
            can_freq = self._robot.motor_chain.comm_freq

        temp_mos = None
        temp_rotor = None
        if hasattr(self._robot, "_joint_state") and self._robot._joint_state is not None:
            temp_mos = self._robot._joint_state.temp_mos[: self._n_arm]
            temp_rotor = self._robot._joint_state.temp_rotor[: self._n_arm]

        table = self._format_log_table(
            joint_pos,
            joint_vel,
            joint_eff,
            required[:n],
            diff,
            gripper_pos=gripper_pos,
            loop_freq=self._log_freq,
            can_freq=can_freq,
            temp_mos=temp_mos[:n] if temp_mos is not None else None,
            temp_rotor=temp_rotor[:n] if temp_rotor is not None else None,
        )
        print("\033[2J\033[H" + table, flush=True)

    # ---- key callback ---------------------------------------------------------

    def _on_key(self, key: int) -> None:
        if key != 32:  # SPACE
            return
        if self._is_sim:
            # Sim mode: always stay in CONTROL mode
            return
        if self._mode is Mode.VIS:
            self._mode = Mode.CONTROL
            self._sync_mocap_to_ee()
            self._set_marker_color(self._CTRL_RGBA)
            # Seed sliders with current robot joint positions
            self._sync_sliders_to_robot()
            print("[control] CONTROL mode — use sliders for per-joint control,")
            print("[control]   or double-click target + ctrl+drag for IK")
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
        if self._is_sim:
            self._mode = Mode.CONTROL
            print("[control] Starting in CONTROL mode (sim)")
        else:
            print("[control] Starting in VIS (gravity comp) mode")
            print("[control] Press SPACE to toggle CONTROL mode")

        with mujoco.viewer.launch_passive(
            self._model,
            self._data,
            key_callback=self._on_key,
        ) as viewer:
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            if self._is_sim:
                self._set_marker_color(self._CTRL_RGBA)
                self._mirror_robot()
                self._sync_mocap_to_ee()
                self._sync_sliders_to_robot()
            try:
                while viewer.is_running():
                    self._mirror_robot()
                    if self._logging:
                        self._log()
                    self._update_button_indicators()

                    if self._mode is Mode.VIS:
                        self._sync_mocap_to_ee()
                    else:
                        sliders_moved = self._sliders_changed()
                        mocap_moved = self._mocap_changed()

                        if sliders_moved:
                            # User moved a slider — command joints directly from slider values
                            cmd = self._cmd_from_sliders()
                            n = min(len(cmd), self._nq)
                            if self._has_self_collision(cmd, n):
                                if not self._in_collision:
                                    print("[control] Collision detected — command blocked")
                                    self._in_collision = True
                            else:
                                self._robot.command_joint_pos(cmd)
                                logger.info(f"joint command: {cmd}")
                                if self._in_collision:
                                    print("[control] Collision cleared — commands resumed")
                                    self._in_collision = False
                            # Update mocap target to follow the new slider FK
                            self._sync_mocap_to_sliders()
                        elif mocap_moved:
                            # User dragged the mocap target — solve IK
                            target = self._mocap_pose_4x4()
                            init_q = self._data.qpos[: self._nq].copy()
                            ok, ik_q = self._kin.ik(target, self._ee_site, init_q=init_q)
                            if ok:
                                cmd = self._robot.get_joint_pos().copy()
                                cmd[: self._n_arm] = ik_q[: self._n_arm]
                                n = min(len(cmd), self._nq)
                                if self._has_self_collision(cmd, n):
                                    if not self._in_collision:
                                        print("[control] Collision detected — command blocked")
                                        self._in_collision = True
                                else:
                                    self._robot.command_joint_pos(cmd)
                                    logger.info(f"IK joint command: {cmd}")
                                    # Update sliders to reflect the IK solution
                                    self._sync_sliders_to_ik(ik_q)
                                    if self._in_collision:
                                        print("[control] Collision cleared — commands resumed")
                                        self._in_collision = False

                        # Snapshot for next iteration's change detection
                        self._prev_ctrl[:] = self._data.ctrl
                        self._prev_mocap_pos[:] = self._data.mocap_pos[self._mocap_id]
                        self._prev_mocap_quat[:] = self._data.mocap_quat[self._mocap_id]

                    viewer.sync()
                    time.sleep(self._dt)
            except KeyboardInterrupt:
                pass

        print("[control] Viewer closed")
