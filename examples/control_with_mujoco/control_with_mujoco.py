"""MuJoCo control interface for i2rt robots (multiprocess).

The robot controller and the MuJoCo viewer each run in their own
subprocess so neither steals GIL time from the other. The main
process is just glue: it spawns both workers, hosts the shared
memory + command queue that they speak through, and shuts them
down together.

Starts in VIS mode, mirroring the robot's joint state (gravity-comp
active on real hardware). Press SPACE in the viewer to toggle into
CONTROL mode; then use the per-joint sliders or double-click the
mocap marker + ctrl+drag for IK. Commands are blocked on self-
collision.

With ``--log`` the table header shows ``[loop XX.X Hz]``, which is
the motor chain's measured ``comm_freq`` (the same number
``dm_driver.py`` reports) instead of the viewer's data-read
cadence.

Usage:
    python examples/control_with_mujoco/control_with_mujoco.py --sim
    python examples/control_with_mujoco/control_with_mujoco.py --sim --log
    python examples/control_with_mujoco/control_with_mujoco.py --arm big_yam --gripper linear_4310 --sim
    python examples/control_with_mujoco/control_with_mujoco.py --arm no_arm --gripper flexible_4310 --sim
    python examples/control_with_mujoco/control_with_mujoco.py --channel can0
"""

import argparse
import logging
import multiprocessing as mp
import os
import queue as _queue
import signal
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Shared memory layout
# ---------------------------------------------------------------------------

_FIXED_FIELDS = (
    ("gripper_pos", 1),
    ("button_states", 2),
    ("loop_hz", 1),
    ("seq", 1),
)
_PER_DOF_FIELDS = ("joint_pos", "joint_vel", "joint_eff", "temp_mos", "temp_rotor")


def _build_layout(n_dofs: int) -> tuple[Dict[str, slice], int]:
    layout: Dict[str, slice] = {}
    offset = 0
    for name in _PER_DOF_FIELDS:
        layout[name] = slice(offset, offset + n_dofs)
        offset += n_dofs
    for name, length in _FIXED_FIELDS:
        layout[name] = slice(offset, offset + length)
        offset += length
    return layout, offset


class SharedState:
    """Numpy views over an ``mp.shared_memory.SharedMemory`` block."""

    def __init__(self, n_dofs: int, shm_name: Optional[str] = None):
        layout, total_floats = _build_layout(n_dofs)
        self._layout = layout
        self._n_dofs = n_dofs
        nbytes = total_floats * np.dtype(np.float64).itemsize
        if shm_name is None:
            self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
            self._owner = True
        else:
            self._shm = shared_memory.SharedMemory(name=shm_name)
            self._owner = False
        self._buf = np.ndarray((total_floats,), dtype=np.float64, buffer=self._shm.buf)
        if self._owner:
            self._buf[:] = np.nan

    @property
    def name(self) -> str:
        return self._shm.name

    def view(self, field: str) -> np.ndarray:
        return self._buf[self._layout[field]]

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Proxy robot (used inside the viewer subprocess)
# ---------------------------------------------------------------------------


class _ProxyJointState:
    """Duck-types the bits of ``MotorChainRobot._joint_state`` that the
    viewer reads (``temp_mos``, ``temp_rotor``)."""

    def __init__(self, shared: SharedState):
        self._shared = shared

    @property
    def temp_mos(self) -> np.ndarray:
        return self._shared.view("temp_mos").copy()

    @property
    def temp_rotor(self) -> np.ndarray:
        return self._shared.view("temp_rotor").copy()


class _ProxyMotorChain:
    """Duck-types ``DMChainCanInterface`` for viewer-side reads."""

    def __init__(self, shared: SharedState, has_teaching_handle: bool):
        self._shared = shared
        self._has_teaching_handle = has_teaching_handle
        self.same_bus_device_driver = object() if has_teaching_handle else None

    @property
    def comm_freq(self) -> float:
        val = float(self._shared.view("loop_hz")[0])
        return 0.0 if np.isnan(val) else val

    def get_same_bus_device_states(self) -> Optional[list[SimpleNamespace]]:
        if not self._has_teaching_handle:
            return None
        buttons = self._shared.view("button_states")
        if np.isnan(buttons[0]):
            return None
        return [SimpleNamespace(io_inputs=[bool(buttons[0]), bool(buttons[1])])]


class ProxyRobot:
    """The viewer subprocess sees this in place of the real robot.

    It implements the subset of the ``Robot`` surface that
    :class:`MujocoControlInterface` consumes: state reads come from shared
    memory, commands go onto a multiprocessing queue that the robot
    subprocess drains.
    """

    def __init__(self, shared: SharedState, cmd_queue: mp.Queue, meta: Dict[str, Any]):
        self._shared = shared
        self._cmd_queue = cmd_queue
        self._meta = meta
        self.xml_path = meta["xml_path"]
        self._gripper_index = meta.get("gripper_index")
        self._n_dofs = int(meta["n_dofs"])
        self.motor_chain = _ProxyMotorChain(shared, bool(meta.get("has_teaching_handle", False)))
        self._joint_state = _ProxyJointState(shared)

    def num_dofs(self) -> int:
        return self._n_dofs

    def get_robot_info(self) -> Dict[str, Any]:
        return {
            "gripper_index": self._gripper_index,
            "gripper_limits": self._meta.get("gripper_limits"),
            "sim": bool(self._meta.get("is_sim", False)),
        }

    def get_joint_pos(self) -> np.ndarray:
        return self._shared.view("joint_pos")[: self._n_dofs].copy()

    def get_observations(self) -> Dict[str, np.ndarray]:
        n = self._n_dofs
        pos = self._shared.view("joint_pos")[:n].copy()
        vel = self._shared.view("joint_vel")[:n].copy()
        eff = self._shared.view("joint_eff")[:n].copy()
        gi = self._gripper_index
        if gi is None:
            return {"joint_pos": pos, "joint_vel": vel, "joint_eff": eff}
        return {
            "joint_pos": pos[:gi],
            "joint_vel": vel[:gi],
            "joint_eff": eff[:gi],
            "gripper_pos": np.array([pos[gi]]),
            "gripper_vel": np.array([vel[gi]]),
            "gripper_eff": np.array([eff[gi]]),
        }

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._cmd_queue.put(("joint_pos", np.asarray(joint_pos, dtype=np.float64).copy()))

    def enter_gravity_comp_idle(self) -> None:
        self._cmd_queue.put(("enter_gravity_comp_idle",))

    def enable_gravity_comp(self) -> None:
        self._cmd_queue.put(("enable_gravity_comp",))

    def disable_gravity_comp(self) -> None:
        self._cmd_queue.put(("disable_gravity_comp",))

    def start_server(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Robot subprocess
# ---------------------------------------------------------------------------


def _apply_command(robot: Any, cmd: tuple) -> None:
    kind = cmd[0]
    if kind == "joint_pos":
        robot.command_joint_pos(cmd[1])
    elif kind == "enter_gravity_comp_idle" and hasattr(robot, "enter_gravity_comp_idle"):
        robot.enter_gravity_comp_idle()
    elif kind == "enable_gravity_comp" and hasattr(robot, "enable_gravity_comp"):
        robot.enable_gravity_comp()
    elif kind == "disable_gravity_comp" and hasattr(robot, "disable_gravity_comp"):
        robot.disable_gravity_comp()


def _robot_worker(
    arm_value: str,
    gripper_value: str,
    channel: str,
    sim: bool,
    use_coulomb_friction: bool,
    meta_queue: mp.Queue,
    cmd_queue: mp.Queue,
    stop_event: Any,
) -> None:
    """Subprocess entry point: owns the real/sim robot and publishes state."""
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType
    from i2rt.utils.utils import RateRecorder

    arm = ArmType.from_string_name(arm_value)
    gripper = GripperType.from_string_name(gripper_value)

    robot = get_yam_robot(
        channel=channel, arm_type=arm, gripper_type=gripper, sim=sim, use_coulomb_friction=use_coulomb_friction
    )
    if sim and hasattr(robot, "start_server"):
        robot.start_server()

    info = robot.get_robot_info() if hasattr(robot, "get_robot_info") else {}
    chain = getattr(robot, "motor_chain", None)
    has_teaching_handle = (
        chain is not None
        and hasattr(chain, "get_same_bus_device_states")
        and hasattr(chain, "same_bus_device_driver")
        and chain.same_bus_device_driver is not None
    )

    n_dofs = int(robot.num_dofs())
    shared = SharedState(n_dofs=n_dofs)  # owner — auto-generated name
    meta: Dict[str, Any] = {
        "n_dofs": n_dofs,
        "xml_path": robot.xml_path,
        "gripper_index": info.get("gripper_index"),
        "gripper_limits": info.get("gripper_limits"),
        "is_sim": sim,
        "has_teaching_handle": has_teaching_handle,
        "shm_name": shared.name,
    }
    meta_queue.put(meta)

    rate = RateRecorder(name="robot_publish", report_interval=1.0)
    target_dt = 0.004  # 250 Hz publish cadence, matching CAN

    try:
        with rate:
            next_tick = time.perf_counter()
            seq = 0.0
            while not stop_event.is_set():
                obs = robot.get_observations()
                joint_pos = obs["joint_pos"]
                joint_vel = obs["joint_vel"]
                joint_eff = obs["joint_eff"]

                pos_buf = shared.view("joint_pos")
                vel_buf = shared.view("joint_vel")
                eff_buf = shared.view("joint_eff")
                n_arm = min(len(joint_pos), n_dofs)
                pos_buf[:n_arm] = joint_pos[:n_arm]
                vel_buf[:n_arm] = joint_vel[:n_arm]
                eff_buf[:n_arm] = joint_eff[:n_arm]

                gi = meta["gripper_index"]
                if gi is not None and "gripper_pos" in obs:
                    pos_buf[gi] = float(obs["gripper_pos"][0])
                    vel_buf[gi] = float(obs.get("gripper_vel", np.zeros(1))[0])
                    eff_buf[gi] = float(obs.get("gripper_eff", np.zeros(1))[0])
                    shared.view("gripper_pos")[0] = pos_buf[gi]

                js = getattr(robot, "_joint_state", None)
                if js is not None:
                    tm = np.asarray(js.temp_mos)
                    tr = np.asarray(js.temp_rotor)
                    n_t = min(len(tm), n_dofs)
                    shared.view("temp_mos")[:n_t] = tm[:n_t]
                    shared.view("temp_rotor")[:n_t] = tr[:n_t]

                if has_teaching_handle:
                    states = chain.get_same_bus_device_states()
                    if states:
                        io_inputs = states[0].io_inputs
                        shared.view("button_states")[0] = float(bool(io_inputs[0]))
                        shared.view("button_states")[1] = float(bool(io_inputs[1]))

                chain_freq = float(chain.comm_freq) if chain is not None and hasattr(chain, "comm_freq") else 0.0
                shared.view("loop_hz")[0] = chain_freq if chain_freq > 0 else float(rate.last_rate)

                seq += 1.0
                shared.view("seq")[0] = seq

                while True:
                    try:
                        cmd = cmd_queue.get_nowait()
                    except _queue.Empty:
                        break
                    _apply_command(robot, cmd)

                rate.track()

                next_tick += target_dt
                sleep_for = next_tick - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if hasattr(robot, "close"):
                robot.close()
        except Exception as e:
            logging.warning("robot.close() failed: %s", e)
        shared.close()


# ---------------------------------------------------------------------------
# Viewer subprocess
# ---------------------------------------------------------------------------


def _viewer_worker(
    gripper_value: str,
    site_arg: Optional[str],
    dt: float,
    log: bool,
    cmd_queue: mp.Queue,
    meta: Dict[str, Any],
    stop_event: Any,
) -> None:
    """Subprocess entry point: runs the MuJoCo viewer against a ProxyRobot."""
    from i2rt.robots.utils import GripperType
    from i2rt.utils.mujoco_control_interface import MujocoControlInterface

    shared = SharedState(n_dofs=int(meta["n_dofs"]), shm_name=meta["shm_name"])
    proxy = ProxyRobot(shared=shared, cmd_queue=cmd_queue, meta=meta)

    if site_arg is not None:
        site = site_arg
    elif gripper_value == GripperType.YAM_TEACHING_HANDLE.value:
        site = "tcp_site"
    else:
        site = "grasp_site"

    iface = MujocoControlInterface.from_robot(proxy, ee_site=site, dt=dt, log=log)
    try:
        iface.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        shared.close()


# ---------------------------------------------------------------------------
# Main (orchestrator)
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    from i2rt.robots.utils import ArmType, GripperType

    arm_choices = [a.value for a in ArmType]
    gripper_choices = [g.value for g in GripperType]
    parser = argparse.ArgumentParser(description="MuJoCo control interface for i2rt robots")
    parser.add_argument("--arm", type=str, default="yam", choices=arm_choices)
    parser.add_argument("--gripper", type=str, default="linear_4310", choices=gripper_choices)
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel")
    parser.add_argument("--sim", action="store_true", help="Use SimRobot")
    parser.add_argument("--dt", type=float, default=0.02, help="Loop timestep (s)")
    parser.add_argument("--site", type=str, default=None, help="EE site name (auto-detected if omitted)")
    parser.add_argument("--log", action="store_true", help="Log joint state and torques each loop iteration")
    parser.add_argument(
        "--friction",
        action="store_true",
        help="Enable Coulomb friction compensation in gravity comp (real hardware only)",
    )
    return parser.parse_args()


def _wait_for_meta(meta_queue: mp.Queue, robot_proc: mp.Process, timeout: float = 60.0) -> Dict[str, Any]:
    """Block until the robot worker publishes its meta dict, or dies."""
    deadline = time.time() + timeout
    while True:
        try:
            return meta_queue.get(timeout=0.5)
        except _queue.Empty:
            if not robot_proc.is_alive():
                raise RuntimeError(
                    f"Robot worker exited (code={robot_proc.exitcode}) before publishing meta"
                ) from None
            if time.time() > deadline:
                raise RuntimeError(f"Timed out after {timeout:.0f}s waiting for robot meta") from None


def main() -> None:
    args = _parse_args()
    from i2rt.robots.utils import ArmType, GripperType

    arm = ArmType.from_string_name(args.arm)
    gripper = GripperType.from_string_name(args.gripper)
    if arm == ArmType.NO_ARM and gripper == GripperType.NO_GRIPPER:
        raise SystemExit("--gripper cannot be 'no_gripper' when --arm is 'no_arm'")

    mp.set_start_method("spawn", force=True)

    cmd_queue: mp.Queue = mp.Queue()
    meta_queue: mp.Queue = mp.Queue(maxsize=1)
    stop_event = mp.Event()

    robot_proc = mp.Process(
        target=_robot_worker,
        name="robot_worker",
        args=(args.arm, args.gripper, args.channel, args.sim, args.friction, meta_queue, cmd_queue, stop_event),
    )
    robot_proc.start()

    meta: Optional[Dict[str, Any]] = None
    try:
        meta = _wait_for_meta(meta_queue, robot_proc)
    except Exception as e:
        print(f"[control] {e}")
        robot_proc.join(timeout=2.0)
        if robot_proc.is_alive():
            robot_proc.terminate()
        raise SystemExit(1) from e

    viewer_proc = mp.Process(
        target=_viewer_worker,
        name="viewer_worker",
        args=(args.gripper, args.site, args.dt, args.log, cmd_queue, meta, stop_event),
    )
    viewer_proc.start()

    try:
        mp.connection.wait([robot_proc.sentinel, viewer_proc.sentinel])
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        for p in (robot_proc, viewer_proc):
            if p.is_alive():
                try:
                    os.kill(p.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        for p in (robot_proc, viewer_proc):
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
            if p.is_alive():
                p.kill()

        try:
            shm = shared_memory.SharedMemory(name=meta["shm_name"])
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        print("[control] Force killing process to close MuJoCo window")
        os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
