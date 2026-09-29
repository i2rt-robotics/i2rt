"""Operational-space control on a 7-DOF arm, in simulation.

Shows the thing a redundant arm can do that a 6-DOF arm cannot: hold its tool still while the
elbow stays free to be pushed aside, and drift back when released. Both come from one commanded
joint vector -- the task target is ``FK(joint_cmd)`` and the nullspace target is ``joint_cmd``
itself -- which is why teleoperation and policy rollouts need no new interface to get it.

Two ways to run it.

**Browser** (``--viser``). Serves the live arm over HTTP/WebSocket, so it renders in a browser
tab. This is the one to use over ssh: it needs one forwarded TCP port and no X server, no GLX and
no GPU context on either end. Drag the gizmo on the elbow to push the arm; the GUI panel carries
the enable/float buttons and the live numbers. Rendering happens in the browser rather than in
this process, so it also costs the control loop far less than a local window does.

**Viewer** (default). Opens the MuJoCo passive viewer on the simulated plant, so the mouse acts on
the real thing:

    python examples/arm7_osc/arm7_osc_demo.py
    python examples/arm7_osc/arm7_osc_demo.py --trajectory circle
    python examples/arm7_osc/arm7_osc_demo.py --arm left
    python examples/arm7_osc/arm7_osc_demo.py --mode full

    ctrl + right-drag on the elbow   push it; the tool should barely move
    space                            toggle OSC / gravity-comp float
    p                                pulse a fixed sideways force on the elbow
    r                                resume after a watchdog timeout
    q                                quit

On macOS the passive viewer only opens from the process ``mjpython`` started, so run
``mjpython examples/arm7_osc/arm7_osc_demo.py``. Over ssh, prefer ``--viser``: a forwarded
OpenGL window goes through indirect GLX, which cannot supply the context MuJoCo asks for.

**Headless** (``--headless``). Runs a scripted hold-push-release sequence and prints the measured
numbers -- end-effector hold error, how much of the push landed in the nullspace, recovery, and
loop timing. No display needed, so this is what to run over ssh or in a report.

**Recorded** (``--record out.mp4``). Same scripted sequence, rendered offscreen through EGL --
GPU-accelerated and needing no X server at all -- with a marker pinned to the tool's starting
position and the live numbers burned in. This is the reliable way to *see* it over ssh: an
interactive OpenGL window forwarded over ``ssh -X`` goes through indirect GLX, which cannot supply
the context MuJoCo asks for, and the remote GPU is not used for it either way.

Both paths drive the same :class:`~i2rt.robots.arm7_osc.Arm7OSC` through the same ``MotorChain``
interface a real CAN bus presents, so nothing here is simulation-only glue.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tyro

# MuJoCo resolves its GL backend when it is imported, not when a context is first needed, so the
# choice has to be made before any import below pulls it in. Recording renders offscreen and wants
# EGL; the interactive viewer wants GLFW, which is the default.
if any(a == "--record" or a.startswith("--record=") for a in sys.argv[1:]):
    os.environ.setdefault("MUJOCO_GL", "egl")

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from i2rt.robot_models import ARM_7DOF_LEFT_XML_PATH, ARM_7DOF_RIGHT_XML_PATH
from i2rt.robots.arm7_osc import (
    ARM_DOF,
    COMMAND_TIMEOUT_S,
    FEEDBACK_TIMEOUT_CYCLES,
    KD_TASK_JOINT,
    KP_TASK_JOINT,
    Arm7OSC,
    ArmState,
    ControlMode,
    arm7_xml,
)
from i2rt.robots.sim_motor_chain import SimMotorChain

# 7 arm motors plus the gripper. The two fingers are one commanded DOF, coupled by an equality
# constraint in the model, so 8 motors drive a 9-DOF model.
MOTOR_LIST = [
    (0x01, "DM4340"),
    (0x02, "DM4340"),
    (0x03, "DM4340"),
    (0x04, "DM4310"),
    (0x05, "DM4310"),
    (0x06, "DM4310"),
    (0x07, "DM4310"),
    (0x08, "DM4310"),
]
DOF_INDICES = list(range(8))

# The two finger DOFs are held by the plant, standing in for a non-backdrivable leadscrew. Without
# this their unbalanced weight leaks through the equality constraint into the arm and looks exactly
# like broken gravity compensation. See SimMotorChain.mechanism_held_dofs.
FINGER_DOFS = [7, 8]

# A mid-workspace pose, clear of the joint limits and of singularity, with the elbow well off the
# shoulder-wrist line so there is a visible elbow to push.
Q_START = np.array([0.3, 0.9, 0.4, -1.2, 0.3, -0.5, 0.2])


# Poses for the gravity panel, in degrees. Spread deliberately from the gravitational minimum to
# full reach: near the hanging pose every joint needs a few tenths of a N.m, so a wrong model and a
# right one look identical there. That is how a factor set tuned only at the hang gets away with
# being wrong -- the poses that constrain it are the raised ones.
_Z7 = np.zeros(ARM_DOF)

GRAVITY_POSES: Dict[str, np.ndarray] = {
    "Hang": np.zeros(ARM_DOF),
    "Elbow bent": np.radians([0, 45, 20, -60, 0, 30, 0]),
    "Near horizontal": np.radians([0, 85, 20, -30, 0, 20, 0]),
    "Side reach": np.radians([-57, 115, 46, -29, -40, 29, -23]),
}

ELBOW_BODY = "link3"
PUSH_FORCE_N = 12.0
PUSH_DURATION_S = 2.0


@dataclass
class Args:
    """Operational-space control on a 7-DOF arm, in simulation."""

    arm: str = "right"
    """Which arm model: right or left."""
    mode: str = "kinematic"
    """`kinematic` (geometry only) or `full` (inertia shaping, dynamically consistent nullspace)."""
    trajectory: str = "hold"
    """`hold` keeps the start pose; `circle` walks the tool around a small circle via IK."""
    headless: bool = False
    """Run a scripted measurement instead of opening the viewer."""
    gate: bool = False
    """Run the acceptance table (A1/A2/A3/A5/A6 + the P1b defect injection) and report pass/fail."""
    gate_seconds: float = 5.0
    """Seconds to hold for each --gate measurement. The spec's own runs are far longer."""
    control_hz: float = 500.0
    """Control-loop rate."""
    duration: float = 0.0
    """Seconds to run, 0 for unlimited (viewer mode only)."""
    viser: bool = False
    """Serve the live arm to a browser instead of opening a native window. Best over ssh."""
    port: int = 8080
    """Port for --viser."""
    poll_groups: bool = False
    """Simulate the firmware's three-at-a-time polling. Each motor is then heard from at a third
    of the loop rate, which is what the missing-frame watchdog has to tolerate."""
    encoder_bits: Optional[int] = None
    """Simulate a quantized rotor encoder, e.g. 12. Velocity is differenced from the quantized
    position, as it is on hardware."""
    record: Optional[str] = None
    """Render the scripted sequence offscreen to this .mp4 instead of opening a viewer."""
    fps: int = 30
    """Frame rate for --record, and the browser update rate for --viser."""
    width: int = 1280
    """Frame width for --record."""
    height: int = 720
    """Frame height for --record."""
    panel: str = "gravity"
    """Which --viser task panel opens first: gravity, osc or faults. Switchable in the browser."""
    gravity_factor: str = ""
    """Per-joint gravity_comp_factor, 7 comma-separated numbers, e.g. "1,1,1,0.9,1,1,1".

    The plant does not see this, only the controller does, so the mismatch it creates is exactly
    ``(factor - 1) * g(q)``. That makes it the one gravity experiment simulation can actually
    answer: with the plant and the model identical, a factor of 1.0 holds perfectly by
    construction, so any drift here is attributable to the factor alone and not to model error.
    Empty uses 1.0 everywhere."""


def _xml_path(arm: str) -> str:
    """Composed arm-plus-gripper MJCF. The shipped arm file is arm-only: no ``grasp_site`` for the
    Jacobian and no gripper mass for gravity, so simulating it would simulate a different robot."""
    if arm not in ("right", "left"):
        raise SystemExit(f"--arm must be 'right' or 'left', got {arm!r}")
    shipped = ARM_7DOF_RIGHT_XML_PATH if arm == "right" else ARM_7DOF_LEFT_XML_PATH
    if not Path(shipped).is_file():
        return shipped  # let _build report the missing model
    return arm7_xml(arm)


def _gravity_factor(args: Args) -> np.ndarray:
    """``--gravity-factor`` parsed, or all ones."""
    if not args.gravity_factor:
        return np.ones(ARM_DOF)
    vals = [float(x) for x in args.gravity_factor.replace(" ", "").split(",") if x]
    if len(vals) != ARM_DOF:
        raise SystemExit(f"--gravity-factor needs {ARM_DOF} numbers, got {len(vals)}")
    return np.array(vals)


def _build(args: Args) -> tuple[SimMotorChain, Arm7OSC]:
    xml = _xml_path(args.arm)
    if not Path(xml).is_file():
        raise SystemExit(
            f"7-DOF model not found at {xml}.\n"
            "It ships as a separate archive; extract it into i2rt/robot_models/ so that\n"
            "i2rt/robot_models/arm/yam_7dof_{left,right}/v1/ exist."
        )
    chain = SimMotorChain(
        xml,
        MOTOR_LIST,
        dof_indices=DOF_INDICES,
        control_dt=1.0 / args.control_hz,
        initial_qpos=np.concatenate([Q_START, [0.0, 0.0]]),
        mechanism_held_dofs=FINGER_DOFS,
        poll_groups=[[0, 1, 2], [3, 4, 5], [6, 7]] if args.poll_groups else None,
        encoder_bits=args.encoder_bits,
    )
    if args.mode not in ("kinematic", "full"):
        raise SystemExit(f"--mode must be 'kinematic' or 'full', got {args.mode!r}")

    # Derate when the polling defect is on. The shipped gains assume every motor is heard from
    # every tick; three groups leave a third of that, and the usable gain follows the per-motor
    # feedback rate rather than the loop rate -- so running the shipped gains here would simply
    # show a diverging arm, which teaches the wrong lesson about why.
    gains: dict = {}
    if args.poll_groups:
        if args.mode == "full":
            gains = dict(kp_cartesian=np.full(6, 10.0), kd_cartesian=np.full(6, 6.3), kp_null=1.0, kd_null=0.2)
            note = "Kp6 40 -> 10, kd_null 0.8 -> 0.2"
        else:
            # `kinematic` needs a far harder derate than `full`, because it has no Mx to normalise
            # the 150x inertia spread: the wrist stays the weak link and sets the limit for the
            # whole arm. x0.05 looks stable for ten seconds and then diverges around twelve, so
            # this is x0.02 -- at which point the task layer is barely doing anything, which is the
            # honest conclusion about this mode under these defects rather than a tuning success.
            gains = dict(kp_task=KP_TASK_JOINT * 0.02, kd_task=KD_TASK_JOINT * 0.14, kp_null=0.2, kd_null=0.02)
            note = "task gain x0.02, kd_null 0.8 -> 0.02"
        print("[defects] three-group polling on: each motor is heard from at a third of the loop")
        print(f"[defects] rate, so the gains are derated to match -- {note}.")
        print("[defects] The nullspace damping is the term that diverges first, not the task layer.")
        if args.mode == "kinematic":
            print("[defects] Note `full` tolerates these defects at a much higher gain: Mx flattens")
            print("[defects] the 150x inertia spread that leaves the wrist as kinematic's weak link.")

    arm = Arm7OSC(
        chain,
        xml,
        ee_site="grasp_site",
        mode=args.mode,
        control_hz=args.control_hz,
        gravity_comp_factor=_gravity_factor(args),
        **gains,
    )
    return chain, arm


def _nullspace_fraction(arm: Arm7OSC, q_ref: np.ndarray, q_now: np.ndarray) -> float:
    """Share of the joint displacement that does not move the tool.

    Always measured with the **orthogonal** projector, whichever projector the controller uses.
    "How much of this motion is invisible to the tool" is a geometric question and the orthogonal
    projection is its answer. ``N_dyn`` answers a different one -- which torques leave the tool
    unaccelerated -- and being oblique it can lengthen a vector, so a norm ratio taken with it runs
    past 100% and is meaningless as a share.
    """
    delta = q_now - q_ref
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        return float("nan")
    return float(np.linalg.norm(arm.nullspace_projector(q_ref) @ delta) / norm)


class _CircleTarget:
    """Walks the tool around a small circle, converting to ``joint_cmd`` with IK.

    Deliberately updated at a teleoperation-like rate rather than the control rate: the upstream
    command is allowed to be slow and irregular, and IK through ``mink`` allocates, so it has no
    business inside a 500 Hz loop.
    """

    RADIUS_M = 0.06
    PERIOD_S = 8.0

    def __init__(self, xml_path: str, q_start: np.ndarray) -> None:
        from i2rt.robots.kinematics import Kinematics

        self._kin = Kinematics(xml_path, "grasp_site")
        self._home = self._kin.fk(np.concatenate([q_start, [0.0, 0.0]]))
        self._q = np.concatenate([q_start, [0.0, 0.0]])
        self._t0 = time.monotonic()

    def __call__(self) -> np.ndarray:
        phase = 2.0 * np.pi * (time.monotonic() - self._t0) / self.PERIOD_S
        target = self._home.copy()
        target[1, 3] += self.RADIUS_M * np.cos(phase) - self.RADIUS_M
        target[2, 3] += self.RADIUS_M * np.sin(phase)
        ok, q = self._kin.ik(target, "grasp_site", init_q=self._q, max_iters=40)
        if ok:
            self._q = q.copy()
        return self._q[:ARM_DOF]


def run_headless(args: Args) -> None:
    """Scripted hold, push, release -- and report what was measured."""
    chain, arm = _build(args)
    try:

        def drive(seconds: float, target: np.ndarray) -> None:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                arm.command_joint_pos(target)
                time.sleep(0.005)

        print("settling under gravity compensation ...")
        drive(1.0, Q_START)

        arm.enable_arm()
        print(f"OSC enabled (mode={arm.mode.value}); easing onto the target, then holding ...")
        # Ease onto the target: the arm has drifted while floating, and a step from there is a
        # lunge the safe stop would rightly catch.
        arm.move_joints(Q_START, duration=2.0)
        drive(3.0, Q_START)
        if arm.state is not ArmState.ACTIVE:
            raise SystemExit(f"arm left ACTIVE before the measurement: {arm.state}, fault={arm.fault}")

        q_ref = arm.get_joint_pos().copy()
        ee_ref = chain.get_site_pose("grasp_site")[:3, 3].copy()
        obs = arm.get_observations()
        print("\n-- hold --")
        print(f"  joint error vs command : {np.abs(q_ref - Q_START).max() * 1e3:8.3f} mrad (max)")
        print(f"  tool position error    : {np.linalg.norm(obs['ee_pos_err']) * 1e3:8.3f} mm")

        print(f"\n-- pushing {ELBOW_BODY} with {PUSH_FORCE_N:.0f} N for {PUSH_DURATION_S:.0f} s --")
        chain.apply_body_force(ELBOW_BODY, np.array([0.0, PUSH_FORCE_N, 0.0]))
        drive(PUSH_DURATION_S, Q_START)
        q_push = arm.get_joint_pos().copy()
        ee_push = chain.get_site_pose("grasp_site")[:3, 3].copy()
        frac = _nullspace_fraction(arm, q_ref, q_push)
        print(f"  elbow (joint3) moved   : {(q_push - q_ref)[2] * 1e3:+8.1f} mrad")
        print(f"  largest joint motion   : {np.abs(q_push - q_ref).max() * 1e3:8.1f} mrad")
        print(f"  in the nullspace       : {frac * 100:8.1f} %")
        print(f"  tool moved             : {np.linalg.norm(ee_push - ee_ref) * 1e3:8.2f} mm")
        if arm.mode is ControlMode.FULL:
            print("    (the projector guarantees nullspace *torque* adds no tool acceleration; an")
            print("     external push is not a nullspace torque, so what remains is the task layer's")
            print("     finite stiffness rejecting it. Compare modes at equal elbow displacement.)")
        else:
            print("    (nonzero by design: N_kin is an orthogonal projector, so it perturbs the task.")
            print("     mode='full' is about 10x better at equal elbow displacement.)")

        print("\n-- released --")
        chain.apply_body_force(ELBOW_BODY, np.zeros(3))
        drive(3.0, Q_START)
        q_rel = arm.get_joint_pos().copy()
        print(f"  returned to within     : {np.abs(q_rel - q_ref).max() * 1e3:8.1f} mrad of the hold pose")

        stats = arm.loop_stats()
        print("\n-- control loop --")
        print(
            f"  {stats['mean_hz']:.1f} Hz mean over {int(stats['count'])} cycles;"
            f" period p50 {stats['p50_ms']:.3f} ms, p99 {stats['p99_ms']:.3f} ms,"
            f" max {stats['max_ms']:.3f} ms"
        )
        print(f"  state={arm.state.value} fault={arm.fault}")
    finally:
        arm.close()
        chain.close()


# The scripted sequence both --headless and --record follow: (label, seconds, elbow force N).
_SEQUENCE = (
    ("holding position", 2.0, 0.0),
    (f"PUSHING the elbow, {PUSH_FORCE_N:.0f} N", 3.0, PUSH_FORCE_N),
    ("released - elbow returns", 3.5, 0.0),
)


def run_record(args: Args) -> None:
    """Render the scripted sequence offscreen to an mp4, with the numbers burned in.

    Uses the real controller against the real plant; the renderer only watches the plant's MjData.
    A marker is pinned where the tool started, so the point is visible and not merely measurable.
    """
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    if os.environ.get("MUJOCO_GL") != "egl":
        raise SystemExit("--record needs MUJOCO_GL=egl; re-run so it is set before mujoco is imported")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("--record needs ffmpeg on PATH")

    chain, arm = _build(args)
    # The delivered models declare no <visual>, so MuJoCo's offscreen buffer defaults to 640x480.
    # Raise it here rather than editing a model file.
    chain.model.vis.global_.offwidth = max(args.width, chain.model.vis.global_.offwidth)
    chain.model.vis.global_.offheight = max(args.height, chain.model.vis.global_.offheight)

    renderer = mujoco.Renderer(chain.model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(chain.model, cam)
    cam.distance, cam.azimuth, cam.elevation = 1.05, 152.0, -12.0

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 22)
        font_b = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = font_b = ImageFont.load_default()

    proc = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{args.width}x{args.height}",
            "-framerate",
            str(args.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            args.record,
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None

    try:
        arm.enable_arm()
        arm.move_joints(Q_START, duration=2.0)
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            arm.command_joint_pos(Q_START)
            time.sleep(0.004)

        with chain.lock:
            ee_ref = chain.get_site_pose("grasp_site")[:3, 3].copy()
            base_id = mujoco.mj_name2id(chain.model, mujoco.mjtObj.mjOBJ_BODY, "base")
            cam.lookat[:] = 0.5 * (chain.data.xpos[base_id] + ee_ref)
        q_ref = arm.get_joint_pos().copy()
        projector = arm.nullspace_projector(q_ref)

        frames = 0
        for label, duration, force in _SEQUENCE:
            chain.apply_body_force(ELBOW_BODY, np.array([0.0, force, 0.0]))
            for _ in range(int(duration * args.fps)):
                deadline = time.monotonic() + 1.0 / args.fps
                while time.monotonic() < deadline:
                    arm.command_joint_pos(Q_START)
                    time.sleep(0.002)

                with chain.lock:
                    renderer.update_scene(chain.data, cam)
                    ee = chain.data.site_xpos[arm._site_id].copy()
                scene = renderer.scene
                if scene.ngeom < scene.maxgeom:
                    mujoco.mjv_initGeom(
                        scene.geoms[scene.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.022, 0.0, 0.0]),
                        ee_ref,
                        np.eye(3).flatten(),
                        np.array([1.0, 0.2, 0.2, 0.55], dtype=np.float32),
                    )
                    scene.ngeom += 1

                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                delta = arm.get_joint_pos() - q_ref
                norm = float(np.linalg.norm(delta))
                frac = float(np.linalg.norm(projector @ delta) / norm) if norm > 1e-9 else float("nan")
                draw.text(
                    (30, 26),
                    f"7-DOF OSC   {arm.mode.value} mode   {args.control_hz:.0f} Hz",
                    fill=(190, 190, 190),
                    font=font,
                )
                draw.text((30, 58), label, fill=(255, 170, 60) if force else (150, 220, 255), font=font_b)
                draw.text(
                    (30, 108),
                    f"tool moved from marker  {np.linalg.norm(ee - ee_ref) * 1e3:6.1f} mm",
                    fill=(235, 235, 235),
                    font=font,
                )
                draw.text(
                    (30, 138), f"elbow (joint3) moved   {delta[2] * 1e3:+7.1f} mrad", fill=(235, 235, 235), font=font
                )
                draw.text((30, 168), f"motion in nullspace     {frac * 100:6.1f} %", fill=(235, 235, 235), font=font)
                draw.text((30, 212), "red sphere = where the tool started", fill=(255, 130, 130), font=font)
                proc.stdin.write(img.tobytes())
                frames += 1
    finally:
        arm.close()
        chain.close()
        proc.stdin.close()
        proc.wait()

    print(f"wrote {args.record}  ({frames} frames, {frames / args.fps:.1f} s)")
    stats = arm.loop_stats()
    if stats:
        print(
            f"loop {stats['mean_hz']:.1f} Hz mean, period p50 {stats['p50_ms']:.3f} ms, "
            f"p99 {stats['p99_ms']:.3f} ms, max {stats['max_ms']:.3f} ms"
        )
        print(
            "  NOTE: rendering and encoding run in this thread and take GIL time from the control\n"
            "  loop, so these numbers are worse than the controller's. Quote --headless for timing."
        )


class _CommandFeeder:
    """Streams ``joint_cmd`` from its own thread at a steady rate.

    Rendering is not the command source. A single frame on a software GL stack, or a browser
    that stops asking for updates, can take longer than ``COMMAND_TIMEOUT_S``, and the arm would
    then stop because the *viewer* stalled -- the watchdog doing its job for the wrong reason. A
    real system's teleoperation or policy loop is likewise independent of any visualiser.
    """

    def __init__(self, arm: Arm7OSC, q0: np.ndarray, rate_hz: float = 100.0) -> None:
        self._arm = arm
        self._rate = rate_hz
        self.target = q0.copy()
        self._lock = threading.Lock()
        self._ease: Optional[tuple[np.ndarray, np.ndarray, float, float]] = None
        self._paused_until = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="arm7_osc_demo_cmd", daemon=True)

    def pause(self, seconds: float) -> None:
        """Stop sending for a while, as an upstream process falling over would.

        The thread keeps running and simply does not publish, which is the case the command
        timeout is written for -- distinct from the stream sending a stale value, which looks
        healthy from the controller's side and is the watchdog's blind spot by design.
        """
        with self._lock:
            self._paused_until = time.monotonic() + seconds

    def hold_here(self, q: np.ndarray) -> None:
        """Re-target to ``q`` immediately, cancelling any ease in progress.

        Used after the arm has stopped and drifted: the stale target is now far away, and
        streaming it again is the very lunge that stopped the arm in the first place.
        """
        with self._lock:
            self._ease = None
            self.target = np.asarray(q, dtype=float).copy()

    def ease_to(self, q: np.ndarray, duration: float = 2.0) -> None:
        """Walk the target to ``q`` over ``duration``. Non-blocking.

        The feeder is the only writer of ``joint_cmd``, so interpolating here rather than calling
        ``Arm7OSC.move_joints`` keeps a GUI callback from fighting this thread for the command.
        """
        with self._lock:
            self._ease = (self.target.copy(), np.asarray(q, dtype=float).copy(), time.monotonic(), duration)

    @property
    def easing(self) -> bool:
        with self._lock:
            return self._ease is not None

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if time.monotonic() < self._paused_until:
                    target = None
                elif self._ease is not None:
                    start, goal, t0, dur = self._ease
                    alpha = 1.0 if dur <= 0 else min(1.0, (time.monotonic() - t0) / dur)
                    self.target = start + (goal - start) * alpha
                    if alpha >= 1.0:
                        self._ease = None
                    target = self.target
                else:
                    target = self.target
            if target is not None:
                self._arm.command_joint_pos(target)
            time.sleep(1.0 / self._rate)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


# Acceptance thresholds, from the spec's table. Kept next to the check that uses them so a
# revised threshold cannot drift away from what is actually being measured.
A1_FLOAT_DRIFT_MM = 10.0
A2_HOLD_MM, A2_HOLD_DEG = 3.0, 1.0
A3_TRACK_RMS_MM = 8.0
A5_JITTER_P99_MS = 3.0
SINE_PERIOD_S = 6.0  # A3 asks for "period >= 5 s or thereabouts"
SINE_AMP_RAD = 0.15


def run_gate(args: Args) -> None:
    """Run the acceptance table and print pass/fail against the spec's own thresholds.

    Everything here is simulation, so it stands in for the real measurements rather than replacing
    them -- A1/A2/A3/A5 are hardware rows. What it does do is catch a regression against the
    numbers already achieved, and produce the record in the units the table specifies.
    """
    from i2rt.robots.arm7_osc import ARM_DOF as _N
    from i2rt.robots.arm7_osc import rotation_error

    groups = [[0, 1, 2], [3, 4, 5], [6, 7]]
    results: List[Tuple[str, str, bool]] = []

    def record(row: str, detail: str, ok: bool) -> None:
        results.append((row, detail, ok))
        print(f"  {row:34s} {detail:44s} {'PASS' if ok else 'FAIL'}")

    # ---- A1: gravity-comp float, tool drift ----------------------------------
    print(f"\nA1  gravity-comp float, tool drift <= {A1_FLOAT_DRIFT_MM:.0f} mm")
    from i2rt.utils.mujoco_utils import MuJoCoKDL

    kdl = MuJoCoKDL(_xml_path(args.arm))
    worst = 0.0
    for q0 in (
        Q_START,
        np.array([0.3, 0.4, 0.5, -1.6, 0.4, -0.6, 0.2]),
        np.array([-1.0, 2.0, 0.8, -0.5, -0.7, 0.5, -0.4]),
    ):
        chain = SimMotorChain(
            _xml_path(args.arm),
            MOTOR_LIST,
            dof_indices=DOF_INDICES,
            control_dt=1 / args.control_hz,
            initial_qpos=np.concatenate([q0, [0.0, 0.0]]),
            mechanism_held_dofs=FINGER_DOFS,
        )
        try:
            ee0 = chain.get_site_pose("grasp_site")[:3, 3].copy()
            tau = np.zeros(len(MOTOR_LIST))
            for _ in range(int(args.gate_seconds * args.control_hz)):
                q = chain.get_true_state()["pos"][:_N]
                tau[:_N] = kdl.compute_inverse_dynamics(q, np.zeros(_N), np.zeros(_N))[:_N]
                chain.set_commands(tau, get_state=False)
            worst = max(worst, float(np.linalg.norm(chain.get_site_pose("grasp_site")[:3, 3] - ee0) * 1e3))
        finally:
            chain.close()
    record("A1 float drift", f"{worst:.3f} mm over {args.gate_seconds:.0f} s x 3 poses", worst <= A1_FLOAT_DRIFT_MM)

    # ---- A2, A3, A5 per mode -------------------------------------------------
    for mode in ("kinematic", "full"):
        print(f"\nA2/A3/A5  mode={mode}")
        chain = SimMotorChain(
            _xml_path(args.arm),
            MOTOR_LIST,
            dof_indices=DOF_INDICES,
            control_dt=1 / args.control_hz,
            initial_qpos=np.concatenate([Q_START, [0.0, 0.0]]),
            mechanism_held_dofs=FINGER_DOFS,
        )
        arm = Arm7OSC(chain, _xml_path(args.arm), ee_site="grasp_site", mode=mode, control_hz=args.control_hz)
        feeder = _CommandFeeder(arm, Q_START)
        try:
            arm.enable_arm()
            feeder.start()
            feeder.ease_to(Q_START, duration=2.0)
            time.sleep(4.0)
            if arm.state is not ArmState.ACTIVE:
                record(f"A2 hold [{mode}]", f"faulted: {arm.fault}", False)
                continue

            tgt_pos = arm.get_ee_pose(Q_START)[:3, 3]
            tgt_rot = arm.get_ee_pose(Q_START)[:3, :3]
            pos_e, rot_e = [], []
            end = time.monotonic() + args.gate_seconds
            while time.monotonic() < end:
                q = chain.get_true_state()["pos"][:_N]
                pose = arm.get_ee_pose(q)
                pos_e.append(np.linalg.norm(pose[:3, 3] - tgt_pos))
                rot_e.append(np.linalg.norm(rotation_error(tgt_rot, pose[:3, :3])))
                time.sleep(0.004)
            hold_mm = float(np.mean(pos_e)) * 1e3
            hold_deg = float(np.mean(rot_e)) * 180 / np.pi
            record(
                f"A2 hold [{mode}]",
                f"{hold_mm:.3f} mm, {hold_deg:.3f} deg",
                hold_mm <= A2_HOLD_MM and hold_deg <= A2_HOLD_DEG,
            )

            arm.timer.reset()
            t0 = time.monotonic()
            track = []
            while time.monotonic() - t0 < 3 * SINE_PERIOD_S:
                t = time.monotonic() - t0
                cmd = Q_START.copy()
                cmd[1] += SINE_AMP_RAD * np.sin(2 * np.pi * t / SINE_PERIOD_S)
                cmd[3] += SINE_AMP_RAD * np.sin(2 * np.pi * t / SINE_PERIOD_S)
                feeder.target = cmd
                if t > SINE_PERIOD_S:
                    q = chain.get_true_state()["pos"][:_N]
                    track.append(np.linalg.norm(arm.get_ee_pose(q)[:3, 3] - arm.get_ee_pose(cmd)[:3, 3]))
                time.sleep(0.004)
            rms = float(np.sqrt(np.mean(np.square(track)))) * 1e3
            record(f"A3 track [{mode}]", f"{rms:.2f} mm RMS, {SINE_PERIOD_S:.0f} s sine", rms <= A3_TRACK_RMS_MM)

            st = arm.loop_stats()
            record(
                f"A5 loop [{mode}]",
                f"{st['mean_hz']:.0f} Hz, p99 {st['p99_ms']:.2f} ms, max {st['max_ms']:.2f} ms",
                st["p99_ms"] <= A5_JITTER_P99_MS,
            )
        finally:
            feeder.stop()
            arm.close()
            chain.close()

    # ---- A6: the nullspace unit test ----------------------------------------
    print("\nA6  nullspace correctness (simulation unit test, `full` only)")
    chain = SimMotorChain(
        _xml_path(args.arm),
        MOTOR_LIST,
        dof_indices=DOF_INDICES,
        initial_qpos=np.concatenate([Q_START, [0.0, 0.0]]),
        mechanism_held_dofs=FINGER_DOFS,
    )
    try:
        arm = Arm7OSC(chain, _xml_path(args.arm), mode="full", auto_start=False)
        jac, mass = arm._jacobian_and_mass(Q_START)
        minv = np.linalg.inv(mass)
        rng = np.random.default_rng(0)
        worst_dyn = worst_kin = 0.0
        for proj, tag in (
            (arm.dynamic_nullspace_projector(Q_START), "dyn"),
            (arm.nullspace_projector(Q_START), "kin"),
        ):
            for _ in range(200):
                v = proj @ rng.standard_normal(_N)
                v /= max(float(np.linalg.norm(v)), 1e-12)
                a = float(np.linalg.norm(jac @ minv @ v))
                if tag == "dyn":
                    worst_dyn = max(worst_dyn, a)
                else:
                    worst_kin = max(worst_kin, a)
        record(
            "A6 nullspace torque -> tool accel", f"full {worst_dyn:.1e}, kinematic {worst_kin:.1e}", worst_dyn < 1e-6
        )
    finally:
        chain.close()

    # ---- P1b: the hardware defects ------------------------------------------
    print("\nP1b  stable with encoder quantization and polling skew injected")
    for tag, ck, ak in (
        ("shipped gains, ideal", {}, {}),
        ("shipped gains, both defects", {"poll_groups": groups, "encoder_bits": 12}, {}),
        (
            "derated gains, both defects",
            {"poll_groups": groups, "encoder_bits": 12},
            {"kp_cartesian": np.full(6, 10.0), "kd_cartesian": np.full(6, 6.3), "kp_null": 1.0, "kd_null": 0.2},
        ),
    ):
        chain = SimMotorChain(
            _xml_path(args.arm),
            MOTOR_LIST,
            dof_indices=DOF_INDICES,
            control_dt=1 / args.control_hz,
            initial_qpos=np.concatenate([Q_START, [0.0, 0.0]]),
            mechanism_held_dofs=FINGER_DOFS,
            **ck,
        )
        arm = Arm7OSC(chain, _xml_path(args.arm), mode="full", control_hz=args.control_hz, **ak)
        feeder = _CommandFeeder(arm, Q_START)
        try:
            arm.enable_arm()
            feeder.start()
            feeder.ease_to(Q_START, duration=2.5)
            time.sleep(4.0)
            vs = []
            end = time.monotonic() + args.gate_seconds
            while time.monotonic() < end:
                vs.append(float(np.abs(chain.get_true_state()["vel"][:_N]).max()))
                time.sleep(0.004)
            ok = arm.state is ArmState.ACTIVE and float(np.mean(vs)) < 0.05
            detail = f"|dq| {np.mean(vs):.4f}" if arm.state is ArmState.ACTIVE else f"faulted: {arm.fault}"
            record(f"P1b {tag}", detail, ok)
        finally:
            feeder.stop()
            arm.close()
            chain.close()

    n_pass = sum(1 for _, _, ok in results if ok)
    print(f"\n{n_pass}/{len(results)} checks passed")
    failed = [r for r, _, ok in results if not ok]
    if failed:
        print("failed: " + ", ".join(failed))
        print("\nNote: the spec makes A2/A3 conditional on the max stable stiffness A6 measures.")
        print("A gain that cannot reach the reference value revises those thresholds rather than")
        print("failing the work -- but the revision has to be recorded, not assumed.")


def run_viser(args: Args) -> None:
    """Serve the live arm to a browser over HTTP, and let the browser push its elbow.

    The dependable way to *see and touch* this over ssh. Viser renders in the browser and speaks
    WebSocket, so it needs one forwarded TCP port and no X server, no GLX, and no GPU context on
    either end -- the three things that make the MuJoCo viewer awkward on a remote machine.

    Everything shown is the live plant: mesh poses are read from the same ``MjData`` the control
    loop is stepping, under the plant's lock.
    """
    import mujoco
    import viser

    chain, arm = _build(args)
    trajectory: Optional[_CircleTarget] = None
    if args.trajectory == "circle":
        trajectory = _CircleTarget(_xml_path(args.arm), Q_START)
    elif args.trajectory != "hold":
        raise SystemExit(f"--trajectory must be 'hold' or 'circle', got {args.trajectory!r}")

    elbow_id = mujoco.mj_name2id(chain.model, mujoco.mjtObj.mjOBJ_BODY, ELBOW_BODY)

    # Mesh vertices/faces in local geom coordinates, matching ViserControlInterface's approach:
    # upload the geometry once, then move handles every frame.
    mesh_geoms: List[int] = []
    local_verts: Dict[int, np.ndarray] = {}
    local_faces: Dict[int, np.ndarray] = {}
    for gid in range(chain.model.ngeom):
        if chain.model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = chain.model.geom_dataid[gid]
        v0, vn = chain.model.mesh_vertadr[mid], chain.model.mesh_vertnum[mid]
        f0, fn = chain.model.mesh_faceadr[mid], chain.model.mesh_facenum[mid]
        mesh_geoms.append(gid)
        local_verts[gid] = chain.model.mesh_vert[v0 : v0 + vn].copy()
        local_faces[gid] = chain.model.mesh_face[f0 : f0 + fn].copy()

    # Engage before the scene upload, not after. Uploading 100k vertices takes seconds, and an arm
    # left floating for that long drifts hundreds of milliradians -- float mode holds nothing by
    # design. Easing back from there at derated gains is a long move the soft loop cannot track,
    # which lands as an overshoot rather than a return.
    arm.enable_arm()
    feeder = _CommandFeeder(arm, Q_START)
    feeder.hold_here(arm.get_joint_pos())
    feeder.start()
    feeder.ease_to(Q_START, duration=2.0)

    server = viser.ViserServer(port=args.port)
    print(f"[viser] open http://localhost:{args.port} in a browser")
    print(f"[viser] over ssh, forward it first:  ssh -L {args.port}:localhost:{args.port} <host>")

    handles = {
        gid: server.scene.add_mesh_simple(
            f"/robot/geom_{gid}",
            local_verts[gid],
            local_faces[gid],
            color=tuple(int(c * 255) for c in chain.model.geom_rgba[gid][:3]),
            # The grafted wrist mesh has inconsistent triangle winding. Show both sides so
            # its mounting section does not disappear when viewed from a different angle.
            side="double" if chain.model.mesh(chain.model.geom_dataid[gid]).name == "wrist" else "front",
            wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            position=np.zeros(3),
        )
        for gid in mesh_geoms
    }

    with chain.lock:
        ee_ref = chain.get_site_pose("grasp_site")[:3, 3].copy()
        elbow_ref = chain.data.xpos[elbow_id].copy()

    # The reference the marker and the nullspace fraction are measured against. It has to be
    # re-taken whenever OSC engages, because enable_arm() holds wherever the arm *now* is: after a
    # float-and-drift the old reference describes a pose the arm is no longer trying to hold, and
    # the tool would appear to miss the marker through no fault of the controller.
    ref: Dict[str, Any] = {"ee": ee_ref, "q": Q_START.copy(), "pending": False}

    marker = server.scene.add_icosphere("/tool_marker", radius=0.022, color=(255, 60, 60), position=ee_ref)
    tool_frame = server.scene.add_frame("/tool", axes_length=0.07, axes_radius=0.004)

    # Drag handle on the elbow. While "grab" is off it tracks the elbow, so releasing never leaves
    # a stale offset behind; while it is on, the arm is pulled toward wherever it was dragged --
    # the same spring a mouse drag applies in the MuJoCo viewer.
    grab_ctrl = server.scene.add_transform_controls(
        "/elbow_grab", position=elbow_ref, scale=0.16, disable_rotations=True
    )

    @server.on_client_connect
    def _camera(client: viser.ClientHandle) -> None:
        client.camera.position = (0.8, -0.8, 0.3)
        client.camera.look_at = tuple(0.5 * (ee_ref + elbow_ref))

    # Gain sliders scale the gains the arm was *built* with, rather than naming absolute values.
    # That keeps a derated configuration derated: a slider whose range was written for the shipped
    # gains would restore them the moment the loop reads it, undoing the derating silently.
    base = dict(
        kp_null=float(np.max(arm.kp_null)),  # a vector gain only ever comes from trajectory playback
        kd_null=float(np.max(arm.kd_null)),
        kp_cartesian=arm.kp_cartesian.copy(),
        kd_cartesian=arm.kd_cartesian.copy(),
        kp_task=arm.kp_task.copy(),
        kd_task=arm.kd_task.copy(),
    )

    # One task at a time. The panel used to show every control at once, which meant that tuning
    # gravity -- seven factors and a pose -- was buried among OSC gains, fault injection and an
    # elbow spring that do nothing in that state. Each task now owns a folder and only its own
    # folder is visible.
    gui_task = server.gui.add_dropdown("Task", ("gravity", "osc", "faults"), initial_value=args.panel)
    gui_state = server.gui.add_text("State", initial_value="-", disabled=True)
    gui_hint = server.gui.add_text("Message", initial_value="", disabled=True)

    # ---- gravity ----------------------------------------------------------------
    f_grav = server.gui.add_folder("Gravity")
    with f_grav:
        server.gui.add_markdown(
            "Place the arm, then read the residual. **Drift is not the measure here** -- with no "
            "friction in the plant a correctly compensated arm is neutrally stable and coasts "
            "forever, so it drifts whether or not the model is right. The residual torque is what "
            "friction has to absorb on hardware, and it is the quantity that transfers."
        )
        gui_pose = [
            server.gui.add_slider(
                f"q{i + 1}",
                float(np.degrees(chain.model.jnt_range[i, 0])),
                float(np.degrees(chain.model.jnt_range[i, 1])),
                1.0,
                # Open at the home pose, not at Q_START. The first thing anyone does with this
                # panel is check the model against the real arm, and that comparison is only
                # meaningful at a pose they can also see on the hardware -- q = 0, hanging
                # straight down. Opening bent made a correct model look wrong.
                float(np.degrees(GRAVITY_POSES["Hang"][i])),
            )
            for i in range(ARM_DOF)
        ]
        gui_freeze = server.gui.add_checkbox("Hold this pose", True)
        with server.gui.add_folder("Presets"):
            preset_buttons = [(name, server.gui.add_button(name)) for name in GRAVITY_POSES]
        gui_factor = [
            server.gui.add_slider(f"factor j{i + 1}", 0.0, 2.0, 0.01, float(arm._gravity_comp_factor[i]))
            for i in range(ARM_DOF)
        ]
        gui_budget_shoulder = server.gui.add_slider("Budget j1-j4 Nm", 0.0, 2.0, 0.01, 0.30)
        gui_budget_wrist = server.gui.add_slider("Budget j5-j7 Nm", 0.0, 1.0, 0.01, 0.06)
        gui_residual = server.gui.add_text("Residual", initial_value="-", disabled=True)
        gui_worst = server.gui.add_text("Worst joint", initial_value="-", disabled=True)
        btn_sweep = server.gui.add_button("Sweep workspace")
        gui_sweep = server.gui.add_text("Sweep result", initial_value="-", disabled=True)

    # ---- osc --------------------------------------------------------------------
    f_osc = server.gui.add_folder("OSC")
    with f_osc:
        btn_osc = server.gui.add_button("Engage")
        btn_clear = server.gui.add_button("Clear fault")
        btn_resume = server.gui.add_button("Resume")
        btn_home = server.gui.add_button("Return to start pose")
        gui_null_scale = server.gui.add_slider("Nullspace stiffness x", 0.0, 4.0, 0.05, 1.0)
        gui_task_scale = server.gui.add_slider("Task gain x", 0.1, 4.0, 0.05, 1.0)
        gui_grab = server.gui.add_checkbox("Grab elbow", False)
        gui_stiff = server.gui.add_slider("Grab stiffness N/m", 10.0, 400.0, 10.0, 150.0)
        gui_maxf = server.gui.add_slider("Grab max force N", 1.0, 40.0, 1.0, float(PUSH_FORCE_N))
        btn_push = server.gui.add_button("Preset push")
        gui_force = server.gui.add_text("Applied force", initial_value="-", disabled=True)
        gui_tool_err = server.gui.add_text("Tool error", initial_value="-", disabled=True)
        gui_tool_moved = server.gui.add_text("Tool moved", initial_value="-", disabled=True)
        gui_null = server.gui.add_text("Motion in nullspace", initial_value="-", disabled=True)

    # ---- faults -----------------------------------------------------------------
    f_faults = server.gui.add_folder("Faults")
    with f_faults:
        gui_drop_joint = server.gui.add_slider("Joint to silence", 1, ARM_DOF, 1, 5)
        btn_drop = server.gui.add_button("Silence that motor")
        btn_undrop = server.gui.add_button("Restore all motors")
        btn_stop_cmd = server.gui.add_button("Stop commanding")
        gui_fb = server.gui.add_text("Feedback rate", initial_value="-", disabled=True)
        gui_age = server.gui.add_text("Oldest reading", initial_value="-", disabled=True)

    gui_loop = server.gui.add_text("Control loop", initial_value="-", disabled=True)
    gui_joints = server.gui.add_text("q / dq / tau", initial_value="-", disabled=True)

    def _show_task(_: object = None) -> None:
        task = gui_task.value
        f_grav.visible = task == "gravity"
        f_osc.visible = task == "osc"
        f_faults.visible = task == "faults"
        marker.visible = tool_frame.visible = task == "osc"
        grab_ctrl.visible = task == "osc"

    gui_task.on_update(_show_task)

    push_until = {"t": 0.0}

    def _retake_reference() -> None:
        """Pin the marker and the nullspace reference to the pose OSC is now holding."""
        with chain.lock:
            ref["ee"] = chain.get_site_pose("grasp_site")[:3, 3].copy()
        ref["q"] = arm.get_joint_pos().copy()
        marker.position = ref["ee"]
        ref["pending"] = False

    def _hold_where_it_is() -> None:
        """Re-target the stream to the current pose, then re-pin the reference to it.

        Both halves matter after a stop. ``Arm7OSC.resume`` holds wherever the arm now is, so a
        stream still pushing the pre-stop target would immediately command a large step -- the
        lunge that tripped the velocity guard to begin with -- and the arm would stop again.
        """
        feeder.hold_here(arm.get_joint_pos())
        _retake_reference()

    def _guard(fn: "Callable[[], None]") -> "Callable[[Any], None]":
        """Run a GUI action, reporting refusals in the panel instead of raising into viser.

        A callback that raises disappears into viser's own thread, so the button looks dead. The
        controller refuses on purpose (enabling while faulted, resuming when not stopped); the
        refusal is information the operator needs, not a crash.
        """

        def handler(_: object) -> None:
            try:
                fn()
                gui_hint.value = ""
            except RuntimeError as e:
                gui_hint.value = str(e)

        return handler

    def _enable() -> None:
        arm.enable_arm()
        _hold_where_it_is()

    def _resume() -> None:
        arm.resume()
        _hold_where_it_is()

    def _go_home() -> None:
        feeder.ease_to(Q_START, duration=3.0)
        ref["pending"] = True  # re-pin once the ease finishes

    btn_osc.on_click(_guard(_enable))
    btn_clear.on_click(_guard(arm.clear_fault))
    btn_resume.on_click(_guard(_resume))
    btn_home.on_click(_guard(_go_home))
    btn_push.on_click(lambda _: push_until.__setitem__("t", time.monotonic() + PUSH_DURATION_S))

    def _silence() -> None:
        """Stop one motor answering, leaving the rest of the chain healthy.

        The watchdog has to notice this without being confused by ordinary polling, where a healthy
        motor is also silent most ticks. That is why the criterion is keyed to each motor's own
        measured refresh rate rather than to the loop rate.
        """
        idx = int(gui_drop_joint.value) - 1
        chain.set_dropped_frames([idx])
        gui_hint.value = f"joint {idx + 1} silenced -- the watchdog should stop the arm shortly"

    def _restore() -> None:
        chain.set_dropped_frames([])
        gui_hint.value = "all motors answering again -- Clear fault, then Resume"

    def _stop_commanding() -> None:
        """Pause the command stream, as an upstream process dying would.

        Recovery is deliberately manual afterwards: a stalled publisher that restarts must not be
        able to fling the arm at a stale target on its own.
        """
        feeder.pause(COMMAND_TIMEOUT_S + 0.5)
        gui_hint.value = f"command stream paused for {COMMAND_TIMEOUT_S + 0.5:.1f} s"

    btn_drop.on_click(_guard(_silence))
    btn_undrop.on_click(_guard(_restore))
    btn_stop_cmd.on_click(_guard(_stop_commanding))

    def _set_pose(q: np.ndarray) -> None:
        """Place the plant at ``q`` and point the sliders at it."""
        for i in range(ARM_DOF):
            gui_pose[i].value = float(np.degrees(q[i]))
        chain.place(np.concatenate([q, np.zeros(chain.model.nq - ARM_DOF)]))
        arm._q[:] = q
        feeder.hold_here(q)

    for _name, _btn in preset_buttons:
        _btn.on_click(lambda _, n=_name: _set_pose(GRAVITY_POSES[n]))

    def _sweep() -> None:
        """Worst residual per joint over many poses, which one pose cannot reveal.

        The residual is ``(factor - 1) * g(q)``, so it scales with how much gravity the pose asks
        for. Sampling only where the arm hangs reports a residual near zero for almost any factor;
        the workspace maximum is the number that says whether a factor set is usable.
        """
        rng = np.random.default_rng(0)
        lo, hi = chain.model.jnt_range[:ARM_DOF, 0], chain.model.jnt_range[:ARM_DOF, 1]
        factor = np.array([g.value for g in gui_factor])
        worst = np.zeros(ARM_DOF)
        for _ in range(400):
            q = rng.uniform(lo, hi)
            worst = np.maximum(
                worst, np.abs((factor - 1.0) * arm._kdl.compute_inverse_dynamics(q, _Z7, _Z7)[:ARM_DOF])
            )
        budget = np.r_[np.full(4, gui_budget_shoulder.value), np.full(3, gui_budget_wrist.value)]
        gui_sweep.value = "  ".join(
            f"j{i + 1} {worst[i]:.2f}{'!' if worst[i] > budget[i] else ''}" for i in range(ARM_DOF)
        )

    btn_sweep.on_click(_guard(_sweep))
    _show_task()

    deadline = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    try:
        ref["pending"] = True

        while time.monotonic() < deadline:
            if trajectory is not None and not feeder.easing:
                feeder.target = trajectory()

            task = gui_task.value
            if task == "gravity":
                # Gravity compensation only, and the factors live so the residual responds as they
                # move. The arm is placed rather than commanded: a spring-dragged handle can only
                # reach poses the arm settles into, which excludes the raised ones where gravity is
                # largest and the model is actually tested.
                if arm.state is not ArmState.FLOATING:
                    arm.float_arm()
                arm._gravity_comp_factor[:] = [g.value for g in gui_factor]
                q_want = np.radians([g.value for g in gui_pose])
                if gui_freeze.value:
                    chain.place(np.concatenate([q_want, np.zeros(chain.model.nq - ARM_DOF)]))
                    arm._q[:] = q_want

            # Gains are read live so the stiffness trade-off can be felt rather than read about.
            # Damping tracks sqrt(stiffness) to hold the damping ratio roughly constant, which is
            # what keeps the sweep varying one thing.
            ns, ts = float(gui_null_scale.value), float(gui_task_scale.value)
            arm.kp_null = base["kp_null"] * ns
            arm.kd_null = base["kd_null"] * np.sqrt(ns)
            if arm.mode is ControlMode.FULL:
                arm.kp_cartesian = base["kp_cartesian"] * ts
                arm.kd_cartesian = base["kd_cartesian"] * np.sqrt(ts)
            else:
                arm.kp_task = base["kp_task"] * ts
                arm.kd_task = base["kd_task"] * np.sqrt(ts)

            preset = time.monotonic() < push_until["t"]
            with chain.lock:
                elbow_now = chain.data.xpos[elbow_id].copy()
                force = np.zeros(3)
                if gui_grab.value and task == "osc":
                    force = np.asarray(grab_ctrl.position) - elbow_now
                    force *= gui_stiff.value
                    mag = float(np.linalg.norm(force))
                    if mag > gui_maxf.value:
                        force *= gui_maxf.value / mag
                elif preset:
                    force = np.array([0.0, PUSH_FORCE_N, 0.0])
                chain.data.xfrc_applied[:] = 0.0
                chain.data.xfrc_applied[elbow_id, :3] = force

                for gid in mesh_geoms:
                    handles[gid].position = chain.data.geom_xpos[gid].copy()
                    handles[gid].wxyz = _mat3_to_wxyz(chain.data.geom_xmat[gid])
                ee_now = chain.data.site_xpos[arm._site_id].copy()
                tool_frame.position = ee_now
                tool_frame.wxyz = _mat3_to_wxyz(chain.data.site_xmat[arm._site_id])

            if not gui_grab.value:
                grab_ctrl.position = elbow_now

            if task == "osc" and ref["pending"] and not feeder.easing and arm.state is ArmState.ACTIVE:
                _retake_reference()

            obs = arm.get_observations()
            holding = arm.state is ArmState.ACTIVE
            delta = obs["joint_pos"] - ref["q"]
            norm = float(np.linalg.norm(delta))
            frac = _nullspace_fraction(arm, ref["q"], obs["joint_pos"]) if norm > 1e-9 else float("nan")
            stats = arm.loop_stats()

            if task == "gravity":
                need = chain.gravity_torque()[:ARM_DOF]
                resid = obs["joint_eff"] - need
                budget = np.r_[np.full(4, gui_budget_shoulder.value), np.full(3, gui_budget_wrist.value)]
                over = np.abs(resid) > budget
                gui_residual.value = "  ".join(
                    f"j{i + 1} {resid[i]:+.2f}{'!' if over[i] else ''}" for i in range(ARM_DOF)
                )
                w = int(np.argmax(np.abs(resid) / np.maximum(budget, 1e-9)))
                gui_worst.value = (
                    f"j{w + 1}  {resid[w]:+.2f} Nm  vs budget {budget[w]:.2f}  "
                    f"({np.abs(resid[w]) / max(budget[w], 1e-9):.1f}x)"
                )
                if not gui_pose[0].disabled and not gui_freeze.value:
                    for i in range(ARM_DOF):
                        gui_pose[i].value = float(np.degrees(obs["joint_pos"][i]))

            gui_state.value = arm.state.value + (f"   fault: {arm.fault}" if arm.fault else "")
            if arm.fault:
                gui_hint.value = "faulted -- press Clear fault, then Resume"
            elif arm.state is ArmState.DAMPED:
                gui_hint.value = "stopped -- press Resume"
            gui_force.value = f"{np.linalg.norm(force):.1f} N"
            # Only meaningful while OSC is holding a pose; in float there is no target to miss.
            gui_tool_err.value = f"{np.linalg.norm(obs['ee_pos_err']) * 1e3:.2f} mm" if holding else "-"
            gui_tool_moved.value = f"{np.linalg.norm(ee_now - ref['ee']) * 1e3:.2f} mm" if holding else "-"
            gui_null.value = f"{frac * 100:.1f} %" if holding else "-"
            if stats:
                gui_loop.value = (
                    f"{stats['mean_hz']:.0f} Hz   p50 {stats['p50_ms']:.2f}   "
                    f"p99 {stats['p99_ms']:.2f}   max {stats['max_ms']:.2f} ms"
                )
            intervals = arm.feedback_intervals
            rates = np.where(intervals > 0, 1.0 / np.maximum(intervals, 1e-9), 0.0)
            gui_fb.value = "  ".join(f"j{i + 1} {rates[i]:.0f}Hz" for i in range(ARM_DOF))
            gui_age.value = (
                f"{obs['feedback_age'].max() * 1e3:.1f} ms of {FEEDBACK_TIMEOUT_CYCLES:.0f}x each motor's own interval"
            )
            gui_joints.value = "  ".join(
                f"j{i + 1} {obs['joint_pos'][i]:+.3f}/{obs['joint_vel'][i]:+.2f}/{obs['joint_eff'][i]:+.1f}"
                for i in range(ARM_DOF)
            )
            time.sleep(1.0 / args.fps)
    except KeyboardInterrupt:
        pass
    finally:
        feeder.stop()
        final_state, final_fault = arm.state.value, arm.fault
        stats = arm.loop_stats()
        arm.close()
        chain.close()

    print(f"[viser] stopped in state={final_state} fault={final_fault}")
    if stats:
        print(
            f"[viser] loop {stats['mean_hz']:.1f} Hz mean, period p50 {stats['p50_ms']:.3f} ms, "
            f"p99 {stats['p99_ms']:.3f} ms, max {stats['max_ms']:.3f} ms"
        )
        print(
            "  NOTE: the browser update loop shares this process, so these are worse than the\n"
            "  controller's. Quote --headless for timing."
        )


def _mat3_to_wxyz(mat3: np.ndarray) -> np.ndarray:
    """Row-major 3x3 rotation (MuJoCo's ``xmat`` layout) to a ``wxyz`` quaternion."""
    import mujoco

    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(mat3, dtype=float).reshape(9))
    return quat


def run_viewer(args: Args) -> None:
    """Open the passive viewer on the plant so the mouse can push the real arm."""
    import mujoco
    import mujoco.viewer

    if sys.platform == "darwin" and getattr(mujoco.viewer, "_MJPYTHON", None) is None:
        raise SystemExit(
            "On macOS the MuJoCo viewer must be launched by mjpython. Re-run:\n"
            f"    uv run mjpython {' '.join(sys.argv)}"
        )

    chain, arm = _build(args)
    trajectory: Optional[_CircleTarget] = None
    if args.trajectory == "circle":
        trajectory = _CircleTarget(_xml_path(args.arm), Q_START)
    elif args.trajectory != "hold":
        raise SystemExit(f"--trajectory must be 'hold' or 'circle', got {args.trajectory!r}")

    state = {"push_until": 0.0, "quit": False}
    feeder = _CommandFeeder(arm, Q_START)

    def on_key(key: int) -> None:
        if key == ord(" "):
            if arm.state is ArmState.ACTIVE:
                arm.float_arm()
            else:
                arm.enable_arm()
        elif key in (ord("p"), ord("P")):
            state["push_until"] = time.monotonic() + PUSH_DURATION_S
        elif key in (ord("r"), ord("R")):
            arm.resume()
        elif key in (ord("q"), ord("Q")):
            state["quit"] = True

    q_ref = Q_START.copy()
    ee_ref = chain.get_site_pose("grasp_site")[:3, 3].copy()
    deadline = time.monotonic() + args.duration if args.duration > 0 else float("inf")
    last_print = 0.0

    try:
        arm.enable_arm()
        arm.move_joints(Q_START, duration=2.0)
        feeder.start()
        with mujoco.viewer.launch_passive(chain.model, chain.data, key_callback=on_key) as viewer:
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -15.0
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running() and not state["quit"] and time.monotonic() < deadline:
                if trajectory is not None:
                    feeder.target = trajectory()

                # Keyed push, plus whatever the mouse is doing. Both write xfrc_applied, which
                # survives mj_step, so the control loop sees them as an external disturbance.
                pushing = time.monotonic() < state["push_until"]
                with chain.lock:
                    chain.data.xfrc_applied[:] = 0.0
                    if pushing:
                        bid = mujoco.mj_name2id(chain.model, mujoco.mjtObj.mjOBJ_BODY, ELBOW_BODY)
                        chain.data.xfrc_applied[bid, 1] = PUSH_FORCE_N
                    if viewer.perturb.active:
                        mujoco.mjv_applyPerturbForce(chain.model, chain.data, viewer.perturb)
                    viewer.sync()

                now = time.monotonic()
                if now - last_print > 0.25:
                    last_print = now
                    _print_status(arm, chain, q_ref, ee_ref, pushing)
                time.sleep(0.02)
    finally:
        feeder.stop()
        arm.close()
        chain.close()


def _print_status(
    arm: Arm7OSC,
    chain: SimMotorChain,
    q_ref: np.ndarray,
    ee_ref: np.ndarray,
    pushing: bool,
) -> None:
    obs = arm.get_observations()
    q = obs["joint_pos"]
    stats = arm.loop_stats()
    frac = _nullspace_fraction(arm, q_ref, q)
    ee = chain.get_site_pose("grasp_site")[:3, 3]

    lines = [
        "  7-DOF OSC demo   [space] float/OSC   [p] push elbow   [r] resume   [q] quit",
        "  ctrl + right-drag a body to push it",
        "",
        f"  state            {arm.state.value:<12s}{'  PUSHING' if pushing else ''}",
        f"  fault            {arm.fault or '-'}",
        f"  tool error       {np.linalg.norm(obs['ee_pos_err']) * 1e3:8.2f} mm"
        f"   {np.linalg.norm(obs['ee_rot_err']) * 1e3:8.2f} mrad",
        f"  tool moved       {np.linalg.norm(ee - ee_ref) * 1e3:8.2f} mm  from the reference pose",
        f"  in nullspace     {frac * 100:8.1f} %  of the joint displacement",
        "",
        "  joint      q (rad)   dq (rad/s)   tau (Nm)   age (ms)",
    ]
    for i in range(ARM_DOF):
        lines.append(
            f"    {i + 1}     {q[i]:+9.4f}   {obs['joint_vel'][i]:+9.4f}   "
            f"{obs['joint_eff'][i]:+8.2f}   {obs['feedback_age'][i] * 1e3:7.2f}"
        )
    if stats:
        lines += [
            "",
            f"  loop  {stats['mean_hz']:7.1f} Hz mean   p50 {stats['p50_ms']:.3f} ms   "
            f"p99 {stats['p99_ms']:.3f} ms   max {stats['max_ms']:.3f} ms",
        ]
    print("\033[2J\033[H" + "\n".join(lines), flush=True)


def main() -> None:
    args = tyro.cli(Args)
    if args.gate:
        run_gate(args)
    elif args.record:
        run_record(args)
    elif args.headless:
        run_headless(args)
    elif args.viser:
        run_viser(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
