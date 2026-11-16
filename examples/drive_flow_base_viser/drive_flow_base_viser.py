"""Drive the i2rt flow_base with GUI sliders while visualizing its odometry in viser.

The base is disabled at startup: no velocity commands are sent until you check "enable
robot" in the panel. Three velocity sliders then set the base's local-frame velocity
directly: vx (forward/back), vy (strafe), and ω (yaw). A coordinate axis in the browser
scene tracks the base's odometry pose on a ground grid, and the panel shows a live
control-loop rate (Hz). Uncheck "enable robot" to halt the base; press Stop to re-center
the sliders; quit with Ctrl+C in the terminal. With --linearbot a fourth slider drives the
linear rail and the scene shows the LinearBot model: rail holder at the reported height, casters
steering and wheels turning from the motor feedback, and up to two arms (--left-arm / --right-arm)
on the carriage, each switchable between viz (gravity comp, movable by hand) and sliders (a slider
per joint commands the arm while "enable robot" is checked).

Usage:
    # On a machine on the base's CAN bus (drives a local Vehicle):
    uv run examples/drive_flow_base_viser/drive_flow_base_viser.py

    # From a workstation, talking to a controller already running on the base:
    uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --host 172.6.2.20

    # No hardware: a kinematic stand-in integrates the slider velocities:
    uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --sim

    # A LinearBot (flow base + linear rail), with any of the above:
    uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --linearbot --sim

    # ... with two arms on its carriage:
    uv run examples/drive_flow_base_viser/drive_flow_base_viser.py --linearbot --sim --left-arm yam --right-arm yam

Then open http://localhost:8080 in a browser.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
import tyro
import viser

from i2rt.flow_base.flow_base_controller import Vehicle, b_x, h_x, h_y
from i2rt.flow_base.flow_base_controller import r as wheel_radius
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmName, ArmType, GripperType

LINEARBOT_XML = REPO_ROOT / "i2rt" / "robot_models" / "linearbot" / "linearbot.xml"


@dataclass
class Args:
    host: str | None = None
    """Host running the FlowBase controller server. If unset, drive a local Vehicle over CAN."""
    channel: str = "can_flow_base"
    """CAN channel for the local Vehicle (ignored when --host is set)."""
    port: int = 8080
    """Viser server port."""
    max_linear: float = 0.5
    """The vx / vy (and --linearbot rail) sliders range over ±this (m/s)."""
    max_angular: float = 1.57
    """The ω slider ranges over ±this (rad/s)."""
    control_hz: float = 50.0
    """Control-loop rate (Hz): how often the command is sent and the scene refreshed. Keep it
    well above the base's 0.25 s (4 Hz) command-safety timeout."""
    verify_motor_config: bool = True
    """Check every motor's control mode and feedback scaling before opening the chain. Pass
    --no-verify-motor-config only for bench work with motors missing or a busy bus; this example
    drives the base, so a mis-scaled steering motor is not something to start on. Ignored when
    --host is set: the controller already running on the base did its own check at launch."""
    sim: bool = False
    """Drive a kinematic stand-in instead of a base: no CAN bus or --host needed."""
    linearbot: bool = False
    """The base carries a linear rail (a LinearBot): add a rail slider and show the LinearBot model."""
    left_arm: ArmName | None = None
    """Arm on the LinearBot carriage's left mount (needs --linearbot). Unset: no left arm."""
    left_gripper: str = "linear_4310"
    """Gripper on the left arm."""
    left_channel: str = "can_left"
    """CAN channel of the left arm (ignored with --sim)."""
    right_arm: ArmName | None = None
    """Arm on the LinearBot carriage's right mount (needs --linearbot). Unset: no right arm."""
    right_gripper: str = "linear_4310"
    """Gripper on the right arm."""
    right_channel: str = "can_right"
    """CAN channel of the right arm (ignored with --sim)."""


class SimBase:
    """The --sim backend: the pose integrates the last commanded body twist and the rail height its
    rate, clipped to the 1 m stroke. The casters' steer and drive angles integrate the rates the real
    controller commands them at, so they swing into their velocity as the real trailing casters do.
    No acceleration limits and no command timeout."""

    def __init__(self) -> None:
        self.cmd = np.zeros(4)  # vx, vy, ω, rail vz
        self.pose = np.zeros(3)  # x, y, θ
        self.z = 0.0
        self.steer = np.zeros(4)  # per caster, in the controller's order (see flow_base_controller h_x/h_y)
        self.drive = np.zeros(4)
        self.t = time.time()

    def set_target_velocity(self, velocity: np.ndarray, frame: str = "local") -> None:
        self.cmd[: len(velocity)] = velocity

    def get_odometry(self) -> dict:
        now = time.time()
        dt, self.t = now - self.t, now
        self.pose, _ = Vehicle._integrate_pose(self.pose, self.cmd[:3], dt)
        self.z = float(np.clip(self.z + self.cmd[3] * dt, 0.0, 1.0))
        # Vehicle.update_state's C matrix (b_y = 0) times the body twist. With the wheel trailing its axis
        # (b_x < 0) the steer angle settles onto the axis' velocity with a time constant of |b_x| / speed,
        # 40 ms at 0.5 m/s, so it is integrated in <= 5 ms steps to stay stable through a slow loop tick.
        vx, vy, w = self.cmd[:3]
        steps = max(1, int(np.ceil(dt / 0.005)))
        for _ in range(steps):
            s, c = np.sin(self.steer), np.cos(self.steer)
            self.steer += dt / steps * ((s * vx - c * vy - (h_x * c + h_y * s) * w) / b_x - w)
            self.drive += dt / steps * (c * vx + s * vy + (h_x * s - h_y * c) * w) / wheel_radius
        return {"position": {"translation": np.array([self.pose[0], self.pose[1], self.z]), "rotation": self.pose[2]}}

    def get_wheel_states(self) -> dict:
        return {"steer": {"pos": self.steer.copy()}, "drive": {"pos": self.drive.copy()}}

    def reset_odometry(self) -> None:
        self.pose = np.zeros(3)  # like a real LinearBot, the rail height is not odometry and stays

    def close(self) -> None:
        pass


def make_backend(args: Args) -> object:
    """SimBase with --sim, a local Vehicle (LinearRailVehicle with --linearbot) when --host is unset,
    otherwise a remote FlowBaseClient. All expose set_target_velocity / get_odometry / reset_odometry / close."""
    if args.sim:
        return SimBase()
    if args.host is None:
        from i2rt.flow_base.flow_base_controller import LinearRailVehicle, Vehicle

        if args.linearbot:
            # Homes the rail at startup (it moves), as the base's own controller does. The brake's
            # USB-GPIO port is /dev/ttyUSB0 unless I2RT_USB_GPIO_PORT names another.
            return LinearRailVehicle(
                vehicle_max_vel=(args.max_linear, args.max_linear, args.max_angular),
                channel=args.channel,
                verify_motor_config=args.verify_motor_config,
            )
        return Vehicle(
            max_vel=(args.max_linear, args.max_linear, args.max_angular),
            channel=args.channel,
            auto_start=True,
            verify_motor_config=args.verify_motor_config,
        )
    from i2rt.flow_base.flow_base_client import FlowBaseClient

    return FlowBaseClient(
        host=args.host,
        with_linear_rail=args.linearbot,
        max_vel_x=args.max_linear,
        max_vel_y=args.max_linear,
        max_vel_theta=args.max_angular,
        max_vel_z=args.max_linear,
    )


def mat_to_wxyz(mat: np.ndarray) -> np.ndarray:
    wxyz = np.empty(4)
    mujoco.mju_mat2Quat(wxyz, mat.ravel())
    return wxyz


class MeshView:
    """A MuJoCo model's mesh geoms in the viser scene under `prefix`, posed from `self.data` by show()."""

    def __init__(self, server: viser.ViserServer, prefix: str, xml_path: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.handles = {}
        for g in range(self.model.ngeom):
            if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = self.model.geom_dataid[g]
            v, f = self.model.mesh_vertadr[mesh], self.model.mesh_faceadr[mesh]
            self.handles[g] = server.scene.add_mesh_simple(
                f"{prefix}/geom_{g}",
                self.model.mesh_vert[v : v + self.model.mesh_vertnum[mesh]],
                self.model.mesh_face[f : f + self.model.mesh_facenum[mesh]],
                color=tuple(int(c * 255) for c in self.model.geom_rgba[g][:3]),
            )
        self.show()

    def show(self) -> None:
        mujoco.mj_kinematics(self.model, self.data)
        for g, handle in self.handles.items():
            handle.position = self.data.geom_xpos[g].copy()
            handle.wxyz = mat_to_wxyz(self.data.geom_xmat[g])


class LinearBotView(MeshView):
    """The LinearBot under /base, so it follows the odometry pose, posed from the rail height and wheel states."""

    def __init__(self, server: viser.ViserServer) -> None:
        super().__init__(server, "/base/linearbot", str(LINEARBOT_XML))
        self.rail = self.model.joint("linear_rail").qposadr[0]
        # Match each controller caster to the model's joints by position: the export's motorN names are
        # not the base's CAN ids. Its casters also sit at arbitrary angles (some wheels lead the steering
        # axis), so the zero is read off the geometry too: the controller's steer angle is the heading
        # the wheel rolls along, with the wheel trailing the axis (b_x < 0).
        hinges = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        steer_joints = [j for j in hinges if abs(self.data.xaxis[j][2]) > 0.9]
        self.casters = []
        for hx, hy in zip(h_x, h_y, strict=True):
            s = min(steer_joints, key=lambda j: np.hypot(*(self.data.xanchor[j][:2] - (hx, hy))))
            d = next(
                j
                for j in range(self.model.njnt)
                if self.model.body_parentid[self.model.jnt_bodyid[j]] == self.model.jnt_bodyid[s]
            )
            axle = self.data.xaxis[d]
            trail = self.data.xanchor[d] - self.data.xanchor[s]
            heading = -(trail - axle * (trail @ axle))  # the hub's offset along the axle dropped
            heading /= np.linalg.norm(heading)
            self.casters.append(
                (
                    self.model.jnt_qposadr[s],
                    np.sign(self.data.xaxis[s][2]),  # steer joint about +z or -z
                    np.arctan2(heading[1], heading[0]),  # heading at joint zero
                    self.model.jnt_qposadr[d],
                    np.sign(axle @ np.cross([0.0, 0.0, 1.0], heading)),  # rolling along the heading turns z x heading
                )
            )

    def update(self, z: float, wheels: dict) -> None:
        self.data.qpos[self.rail] = z
        for (s, s_sign, heading0, d, d_sign), steer, drive in zip(
            self.casters, wheels["steer"]["pos"], wheels["drive"]["pos"], strict=True
        ):
            self.data.qpos[s] = s_sign * (steer - heading0)
            self.data.qpos[d] = d_sign * drive
        self.show()

    def mount(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """Position and wxyz, in the base frame, of the arm base frame on the carriage's `side` mount."""
        body = self.data.body(f"{side}_arm_mount")
        return body.xpos.copy(), mat_to_wxyz(body.xmat)


VIZ, SLIDERS = "viz (grav comp)", "sliders"


class ArmView(MeshView):
    """An arm (with its gripper) on a LinearBot mount, in one of two modes, as in ViserControlInterface:
    viz holds the arm in gravity compensation (movable by hand) and the sliders follow it; sliders
    commands the arm from a slider per joint while "enable robot" is checked. Following until then is
    what keeps a switch or an enable from jumping the arm. A command that would collide is blocked, as
    on a YAM, but against the whole robot (see RobotCollision)."""

    def __init__(self, server: viser.ViserServer, side: str, robot: Robot) -> None:
        self.robot = robot
        self.side = side
        self.frame = server.scene.add_frame(f"/base/{side}_arm", show_axes=False)
        super().__init__(server, f"/base/{side}_arm", robot.xml_path)
        self.check = mujoco.MjData(self.model)  # a command, mapped to this arm's qpos for RobotCollision
        self.collision: RobotCollision | None = None  # set once every arm exists
        self.blocked = False
        info = robot.get_robot_info()
        has_gripper = info.get("gripper_index") is not None and info.get("gripper_limits") is not None
        self.n_arm = info["gripper_index"] if info.get("gripper_index") is not None else robot.num_dofs()
        hinges = [j for j in range(self.model.njnt) if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        with server.gui.add_folder(f"{side} arm (deg)"):
            self.mode = server.gui.add_dropdown("mode", options=(VIZ, SLIDERS), initial_value=VIZ)
            self.sliders = [
                server.gui.add_slider(
                    f"j{i + 1}", min=float(np.degrees(lo)), max=float(np.degrees(hi)), step=0.1, initial_value=0.0
                )
                for i, (lo, hi) in enumerate(self.model.jnt_range[hinges[: self.n_arm]])
            ]
            self.gripper = (
                server.gui.add_slider("gripper", min=0.0, max=1.0, step=0.01, initial_value=0.0)
                if has_gripper
                else None
            )
        self.controls = self.sliders + ([self.gripper] if self.gripper is not None else [])
        for s in self.controls:
            s.disabled = True
        self.active = VIZ  # the mode the robot is in; update() applies a dropdown change from the loop thread

    def _switch(self, mode: str) -> None:
        # ViserControlInterface._enter_control_grav_comp / _enter_vis_grav_comp: sim pauses or resumes its
        # gravity comp; a real arm drops any PD hold and is carried by gravity comp either way, until the
        # next slider command supplies the gains.
        if hasattr(self.robot, "enable_gravity_comp"):
            self.robot.disable_gravity_comp() if mode == SLIDERS else self.robot.enable_gravity_comp()
        else:
            self.robot.enter_gravity_comp_idle()
        for s in self.controls:
            s.disabled = mode != SLIDERS
        self.active, self.blocked = mode, False

    def _set_qpos(self, data: mujoco.MjData, q: np.ndarray) -> None:
        # The same mapping as ViserControlInterface._mirror_robot: grippers report a normalised [0, 1]
        # slide, and coupled fingers follow their leader through the model's joint equalities.
        n = min(len(q), self.model.nq)
        data.qpos[:n] = q[:n]
        for j in range(self.model.njnt):
            adr = self.model.jnt_qposadr[j]
            if adr < n and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = self.model.jnt_range[j]
                data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)
        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                leader, follower = (
                    self.model.jnt_qposadr[self.model.eq_obj1id[i]],
                    self.model.jnt_qposadr[self.model.eq_obj2id[i]],
                )
                data.qpos[follower] = np.polyval(self.model.eq_data[i, :5][::-1], data.qpos[leader])

    def _collides(self, cmd: np.ndarray) -> bool:
        self._set_qpos(self.check, cmd)
        return self.collision.collides(self.side, self.check.qpos)

    def update(self, position: np.ndarray, wxyz: np.ndarray, enabled: bool) -> None:
        if self.mode.value != self.active:
            self._switch(self.mode.value)
        q = self.robot.get_joint_pos()
        if enabled and self.active == SLIDERS:
            cmd = q.copy()
            cmd[: self.n_arm] = np.radians([s.value for s in self.sliders])
            if self.gripper is not None:
                cmd[self.n_arm] = self.gripper.value
            blocked = self._collides(cmd)
            if blocked != self.blocked:
                print(f"[viser] {self.side} arm: " + ("collision, command blocked" if blocked else "commands resumed"))
                self.blocked = blocked
            if not blocked:
                self.robot.command_joint_pos(cmd)
        else:
            for s, v in zip(self.sliders, np.degrees(q[: self.n_arm]), strict=True):
                s.value = float(v)
            if self.gripper is not None:
                self.gripper.value = float(q[self.n_arm])
        self._set_qpos(self.data, q)
        self.show()
        self.frame.position, self.frame.wxyz = position, wxyz


class RobotCollision:
    """The LinearBot with its arms attached at their mounts, as one MuJoCo model, to check an arm command
    with the YAM interfaces' rule (ViserControlInterface._has_self_collision: penetration over 1 mm
    between non-adjacent bodies, the ground ignored) against the whole robot: the arm into itself, the
    other arm, the carriage, the column or the base. Only contacts the commanding arm takes part in
    count, so the export's own resting overlaps (base and steer motors, column and carriage) never block."""

    def __init__(self, bot: LinearBotView, arms: dict[str, ArmView]) -> None:
        spec = mujoco.MjSpec.from_file(str(LINEARBOT_XML))
        for geom in spec.geoms:
            geom.contype = geom.conaffinity = 1  # the export ships visual-only meshes
        for side, arm in arms.items():
            frame = spec.body(f"{side}_arm_mount").add_frame()
            spec.attach(mujoco.MjSpec.from_file(arm.robot.xml_path), prefix=f"{side}/", frame=frame)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.bot, self.arms = bot, arms
        # Each part's qpos addresses, in its own model and in this one, matched by joint name.
        self.adr = {
            key: (
                part.jnt_qposadr,
                np.array([self.model.joint(prefix + part.joint(j).name).qposadr[0] for j in range(part.njnt)]),
            )
            for key, part, prefix in [(None, bot.model, "")] + [(s, a.model, f"{s}/") for s, a in arms.items()]
        }
        self.bodies = {
            side: {b for b in range(self.model.nbody) if self.model.body(b).name.startswith(f"{side}/")}
            for side in arms
        }

    def collides(self, side: str, arm_qpos: np.ndarray) -> bool:
        """Whether the `side` arm at `arm_qpos` (its own model's qpos) would collide, the rest as it is now."""
        for key, qpos in [(None, self.bot.data.qpos)] + [(s, a.data.qpos) for s, a in self.arms.items()]:
            src, dst = self.adr[key]
            self.data.qpos[dst] = (arm_qpos if key == side else qpos)[src]
        mujoco.mj_forward(self.model, self.data)
        m, plane = self.model, mujoco.mjtGeom.mjGEOM_PLANE
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.dist >= -1e-3 or plane in (m.geom_type[c.geom1], m.geom_type[c.geom2]):
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                continue
            if b1 in self.bodies[side] or b2 in self.bodies[side]:
                return True
        return False


def yaw_to_wxyz(theta: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for a rotation of `theta` about +Z."""
    half = 0.5 * theta
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def caster_fault_reported(backend: object) -> bool:
    """Print and return True once the base has latched a steering fault and ramped itself to a stop.

    Without this the failure is silent from here: the base stops and its control loop exits, but the
    chain stays alive, so ``running()`` keeps returning True and the sliders carry on sending commands
    nothing will ever act on. Only a local ``Vehicle`` exposes the fault -- with ``--host`` the
    controller runs in its own process and reports it there.
    """
    fault = getattr(backend, "caster_fault", lambda: None)()
    if fault is None:
        return False
    print(f"\n{fault.render()}", file=sys.stderr)
    return True


def main(args: Args) -> None:
    sides = [
        (side, arm, gripper, channel)
        for side, arm, gripper, channel in (
            ("left", args.left_arm, args.left_gripper, args.left_channel),
            ("right", args.right_arm, args.right_gripper, args.right_channel),
        )
        if arm is not None
    ]
    if sides and not args.linearbot:
        raise SystemExit("--left-arm / --right-arm mount on the LinearBot's carriage: add --linearbot")

    backend = make_backend(args)
    ndof = 4 if args.linearbot else 3  # [vx, vy, ω] (+ rail vz)

    server = viser.ViserServer(host="0.0.0.0", port=args.port, label="flow_base teleop")
    server.scene.add_grid("/ground", width=10.0, height=10.0, plane="xy")
    server.scene.add_frame("/world", axes_length=0.3, axes_radius=0.01)
    base_frame = server.scene.add_frame("/base", axes_length=0.4, axes_radius=0.02)
    bot = LinearBotView(server) if args.linearbot else None
    arms: dict[str, ArmView] = {}
    robots: list[Robot] = []

    vx = server.gui.add_slider(
        "vx — forward (m/s)", min=-args.max_linear, max=args.max_linear, step=0.05, initial_value=0.0
    )
    vy = server.gui.add_slider(
        "vy — strafe left (m/s)", min=-args.max_linear, max=args.max_linear, step=0.05, initial_value=0.0
    )
    wz = server.gui.add_slider(
        "ω — turn left (rad/s)", min=-args.max_angular, max=args.max_angular, step=0.05, initial_value=0.0
    )
    vz = (
        server.gui.add_slider(
            "vz — rail up (m/s)", min=-args.max_linear, max=args.max_linear, step=0.01, initial_value=0.0
        )
        if args.linearbot
        else None
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
        if vz is not None:
            vz.value = 0.0

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
        # Inside the try, so the finally closes whatever was opened if a later arm fails to come up.
        # A real arm starts in gravity compensation (movable by hand); the viewer only mirrors it.
        for side, arm, gripper, channel in sides:
            robots.append(
                get_yam_robot(
                    channel=channel,
                    arm_type=ArmType.from_string_name(arm),
                    gripper_type=GripperType.from_string_name(gripper),
                    sim=args.sim,
                )
            )
            arms[side] = ArmView(server, side, robots[-1])
        if arms:
            collision = RobotCollision(bot, arms)
            for arm in arms.values():
                arm.collision = collision
        while True:
            if caster_fault_reported(backend):
                status.content = "**STOPPED — caster steering fault.** See the terminal; the base will not move."
                break
            cmd = np.array([vx.value, vy.value, wz.value] + ([vz.value] if vz is not None else []), dtype=float)
            enabled = bool(enable.value)
            if enabled:
                backend.set_target_velocity(cmd, frame="local")
            elif was_enabled:
                # just disabled — halt now instead of coasting on the base's safety timeout
                backend.set_target_velocity(np.zeros(ndof), frame="local")
            was_enabled = enabled

            odo = backend.get_odometry()
            x, y, z = (float(v) for v in odo["position"]["translation"])
            theta = float(odo["position"]["rotation"])
            base_frame.position = np.array([x, y, 0.0])  # on a LinearBot z is the rail height, not the base's
            base_frame.wxyz = yaw_to_wxyz(theta)
            if bot is not None:
                bot.update(z, backend.get_wheel_states())
                for side, arm in arms.items():
                    arm.update(*bot.mount(side), enabled)

            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                comm_hz = 0.9 * comm_hz + 0.1 / dt

            state = "ENABLED — sending commands" if enabled else "DISABLED — idle"
            status.content = (
                f"**state**  {state}\n\n"
                f"**pose**  x = {x:+.3f} m   y = {y:+.3f} m   z = {z:+.3f} m   θ = {np.degrees(theta):+.1f}°\n\n"
                f"**cmd**  vx = {cmd[0]:+.2f}   vy = {cmd[1]:+.2f}   ω = {cmd[2]:+.2f}"
                + (f"   vz = {cmd[3]:+.2f}" if args.linearbot else "")
                + "\n\n"
                f"**comm**  {comm_hz:.1f} Hz"
            )

            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        backend.set_target_velocity(np.zeros(ndof), frame="local")
        backend.close()
        for robot in robots:
            robot.close()
        print("\n[flow_base] stopped and closed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
