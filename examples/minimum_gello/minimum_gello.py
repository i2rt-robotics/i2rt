import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np
import portal
import tyro

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.utils import RateRecorder, override_log_level

DEFAULT_ROBOT_PORT = 11333
_MAX_DOFS = 16
"""Upper bound on robot DOF. The shared pos array is sized to this; the active
slice is exposed through an int Value the worker fills after constructing the robot."""

_WORKER_LOOP_PERIOD_S = 0.002
"""~500 Hz pacing for worker loops whose reads hit cached state (get_joint_pos /
get_info return instantly). Without it these loops peg a CPU core; the actual YAM
hardware/control thread updates at ~250 Hz, well below this cap, so no data is lost."""


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class ClientRobot(Robot):
    """A simple client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        return self._client.get_observations().result()

    def close(self) -> None:
        """Tear down the underlying portal client (background loop thread + socket)."""
        self._client.close()

    def __enter__(self) -> "ClientRobot":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class YAMLeaderRobot:
    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> Tuple[np.ndarray, np.ndarray]:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        if encoder_obs is None:
            # Populated only after the first CAN read in the background driver thread.
            raise RuntimeError("CAN device states not ready yet")
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


# ---------------------------------------------------------------------------
# Args + Resources
# ---------------------------------------------------------------------------


@dataclass
class Args:
    arm: Literal["yam", "yam_pro", "yam_ultra", "big_yam", "no_arm"] = "yam"
    gripper: Literal[
        "crank_4310", "linear_3507", "linear_4310", "flexible_4310", "yam_teaching_handle", "no_gripper"
    ] = "yam_teaching_handle"
    mode: Literal["follower", "leader", "visualizer_local", "visualizer_remote"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT
    can_channel: str = "can0"
    bilateral_kp: float = 0.0
    sim: bool = False
    """Use SimRobot instead of real hardware."""
    ee_mass: Optional[float] = None
    """Override end-effector (link_6) mass in kg for gravity compensation. Defaults to the value in the XML."""


@dataclass
class Resources:
    """Resources accumulated during mode setup; ``cleanup`` drains them on exit.

    Mirrors the ``server_processes`` accumulator pattern from xdof/envs/launch.py:
    every spawn helper appends here so a single ``cleanup(resources)`` call tears
    down the mode regardless of how it exits."""

    processes: List[portal.Process] = field(default_factory=list)
    pos_shared: Optional[portal.SharedArray] = None
    stop_event: Optional[Any] = None


def _spawn_into(processes: List[portal.Process], proc: portal.Process) -> portal.Process:
    """Start a portal.Process and append it to the cleanup pool."""
    proc.start()
    processes.append(proc)
    return proc


# ---------------------------------------------------------------------------
# Background workers. Each runs in its own portal.Process so the foreground
# loop (web IO server, web IO client, or MuJoCo viewer) never blocks on
# hardware or RPC.
# ---------------------------------------------------------------------------


def _yam_polling_worker(
    args: "Args",
    pos_shared: "portal.SharedArray",
    n_dofs_value: Any,
    cmd_queue: Any,
    stop_event: Any,
    rate_name: str,
) -> None:
    """Owns the YAM hardware. Streams joint_pos into ``pos_shared`` at hardware
    rate; if ``cmd_queue`` is provided, drains pending commands onto the YAM.
    Publishes the actual DOF count via ``n_dofs_value`` once the YAM is up."""
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    yam = get_yam_robot(
        channel=args.can_channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        ee_mass=args.ee_mass,
        sim=args.sim,
    )
    n_dofs_value.value = yam.num_dofs()
    rate = RateRecorder(name=rate_name, report_interval=1.0)
    rate.start()
    while not stop_event.is_set():
        if cmd_queue is not None:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                yam.command_joint_pos(cmd)
        try:
            pos = yam.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            rate.track()
        except Exception as e:
            logging.error(f"[{rate_name}] error: {e}")
            time.sleep(0.1)
        time.sleep(_WORKER_LOOP_PERIOD_S)  # pace: get_joint_pos is a cached read, so this loop would otherwise spin


def _rpc_polling_worker(
    server_port: int,
    server_host: str,
    pos_shared: "portal.SharedArray",
    stop_event: Any,
    rate_name: str,
) -> None:
    """Polls a remote follower over portal RPC and caches the latest joint_pos
    in ``pos_shared`` so the foreground viewer reads from memory."""
    override_log_level()
    client = ClientRobot(server_port, host=server_host)
    rate = RateRecorder(name=rate_name, report_interval=1.0)
    rate.start()
    while not stop_event.is_set():
        try:
            pos = client.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            rate.track()
        except Exception as e:
            logging.error(f"[{rate_name}] error: {e}")
            time.sleep(0.1)


def _leader_control_worker(
    args: "Args",
    pos_shared: "portal.SharedArray",
    cmd_queue: Any,
    stop_event: Any,
) -> None:
    """Owns the leader YAM hardware and runs the bilateral control
    loop. Reads the latest follower pos from ``pos_shared`` (written by main's
    foreground web-IO loop) and pushes commands for the follower into
    ``cmd_queue`` (drained by main and forwarded over RPC). Never touches the
    network itself."""
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)
    yam = get_yam_robot(
        channel=args.can_channel,
        arm_type=arm_type,
        gripper_type=gripper_type,
        ee_mass=args.ee_mass,
        sim=args.sim,
    )
    robot = YAMLeaderRobot(yam)
    robot_current_kp = yam._kp

    control_rate = RateRecorder(name="yam-leader control", report_interval=1.0)
    control_rate.start()

    # The encoder device states are populated only after the first CAN read, so get_info() can
    # raise on startup. Retry until the first reading lands before entering the control loop.
    while not stop_event.is_set():
        try:
            current_joint_pos, current_button = robot.get_info()
            break
        except RuntimeError:
            time.sleep(0.05)
    else:
        return  # stop requested before the encoder became ready
    current_follower_joint_pos = pos_shared.array.copy()
    logging.info(f"Current leader joint pos: {current_joint_pos}")
    logging.info(f"Current follower joint pos: {current_follower_joint_pos}")

    def slow_move(joint_pos: np.ndarray, duration: float = 3.0) -> None:
        start_pos = pos_shared.array.copy()
        steps = 100
        dt = duration / steps
        for i in range(steps):
            cmd = joint_pos * i / steps + start_pos * (1 - i / steps)
            cmd_queue.put(cmd)
            time.sleep(dt)

    synchronized = False
    while not stop_event.is_set():
        control_rate.track()
        current_joint_pos, current_button = robot.get_info()
        if current_button[0] > 0.5:
            if not synchronized:
                robot.update_kp_kd(kp=robot_current_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                robot.command_joint_pos(current_joint_pos[:6])
                slow_move(current_joint_pos)
            else:
                logging.info("clear bilateral pd")
                robot.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                current_follower_joint_pos = pos_shared.array.copy()
                robot.command_joint_pos(current_follower_joint_pos[:6])
            synchronized = not synchronized
            while current_button[0] > 0.5:
                time.sleep(0.03)
                current_joint_pos, current_button = robot.get_info()

        current_follower_joint_pos = pos_shared.array.copy()

        if synchronized:
            cmd_queue.put(current_joint_pos)
            # bilateral feedback to the leader, proportional to bilateral_kp
            robot.command_joint_pos(current_follower_joint_pos[:6])

        # Pace the loop: get_info reads cached state, so without this the worker spins a core and
        # competes with the in-process robot_server control thread for the GIL, adding jitter.
        time.sleep(_WORKER_LOOP_PERIOD_S)


# ---------------------------------------------------------------------------
# Foreground helpers and loops
# ---------------------------------------------------------------------------


def _wait_for_dof(n_dofs_value: Any, timeout_s: float = 30.0) -> int:
    """Block until the background worker publishes the active DOF count."""
    deadline = time.time() + timeout_s
    while n_dofs_value.value == 0:
        if time.time() > deadline:
            raise TimeoutError("worker did not publish DOF count within timeout")
        time.sleep(0.05)
    return int(n_dofs_value.value)


def _build_follower_server(
    server_port: int,
    n_dofs_value: Any,
    pos_shared: "portal.SharedArray",
    cmd_queue: Any,
) -> portal.Server:
    """Build the portal.Server that exposes the follower YAM over RPC. Returned
    unstarted; caller does ``server.start()`` to block."""
    io_rate = RateRecorder(name=f"ServerRobot[{server_port}] io", report_interval=1.0)
    io_rate.start()
    io_lock = threading.Lock()

    def _tracked(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            with io_lock:
                io_rate.track()
            return fn(*a, **kw)

        return wrapper

    def _get_pos() -> np.ndarray:
        return pos_shared.array[: n_dofs_value.value].copy()

    def _cmd_pos(joint_pos: np.ndarray) -> None:
        cmd_queue.put(np.asarray(joint_pos))

    def _cmd_state(state: Dict[str, np.ndarray]) -> None:
        key = "position" if "position" in state else "pos"
        cmd_queue.put(np.asarray(state[key]))

    def _get_obs() -> Dict[str, np.ndarray]:
        return {"joint_pos": _get_pos()}

    server = portal.Server(server_port)
    logging.info(f"Robot Server Binding to {server_port}")
    server.bind("num_dofs", _tracked(lambda: n_dofs_value.value))
    server.bind("get_joint_pos", _tracked(_get_pos))
    server.bind("command_joint_pos", _tracked(_cmd_pos))
    server.bind("command_joint_state", _tracked(_cmd_state))
    server.bind("get_observations", _tracked(_get_obs))
    return server


def _run_leader_io_loop(
    client_robot: ClientRobot,
    cmd_queue: Any,
    pos_shared: "portal.SharedArray",
    stop_event: Any,
    io_rate: RateRecorder,
) -> None:
    """Foreground loop for leader mode: drain commands from the bilateral
    control worker onto the follower over RPC, and pull the follower's latest
    joint_pos into ``pos_shared`` so the worker can read it for bilateral
    feedback."""
    while not stop_event.is_set():
        # Drain to the newest command and drop any stale backlog: the control worker can enqueue
        # faster than each blocking RPC round-trip drains, so replaying every queued command would
        # make the follower lag further and further behind.
        cmd = None
        while True:
            try:
                cmd = cmd_queue.get_nowait()
            except queue.Empty:
                break
        try:
            if cmd is not None:
                client_robot.command_joint_pos(cmd)
            pos = client_robot.get_joint_pos()
            pos_shared.array[: len(pos)] = pos
            io_rate.track()
        except Exception as e:
            # Ride out a transient follower/network blip. Without this the exception escapes to
            # run_leader, whose finally: cleanup() calls proc.kill() on the control worker while the
            # leader arm is energized in bilateral PD — an abrupt loss of control of a powered arm.
            logging.error(f"[yam-leader web-port io] error: {e}")
            time.sleep(0.1)


def _run_viewer_loop(
    arm_type: ArmType,
    gripper_type: GripperType,
    get_pos_fn: Callable[[], np.ndarray],
) -> None:
    """Foreground MuJoCo viewer. Pulls joint_pos from ``get_pos_fn`` each frame
    — typically a snapshot of a SharedArray populated by a background worker."""
    xml_path = combine_arm_and_gripper_xml(arm_type, gripper_type)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    dt: float = 0.01
    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        while viewer.is_running():
            step_start = time.time()
            joint_pos = get_pos_fn()
            nq = model.nq
            n = min(len(joint_pos), nq)
            data.qpos[:n] = joint_pos[:n]

            for j in range(model.njnt):
                adr = model.jnt_qposadr[j]
                if adr >= n:
                    continue
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                    lo, hi = model.jnt_range[j]
                    data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)

            for i in range(model.neq):
                if model.eq_type[i] != mujoco.mjtEq.mjEQ_JOINT:
                    continue
                adr1 = model.jnt_qposadr[model.eq_obj1id[i]]
                adr2 = model.jnt_qposadr[model.eq_obj2id[i]]
                coef = model.eq_data[i, :5]
                data.qpos[adr2] = np.polyval(coef[::-1], data.qpos[adr1])

            mujoco.mj_forward(model, data)
            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(resources: Resources) -> None:
    """Tear down everything accumulated into ``resources``. Mirrors
    xdof/envs/launch.py's ``cleanup``: signal workers to stop, kill their
    processes, then release shared memory."""
    logging.info("exiting")
    if resources.stop_event is not None:
        resources.stop_event.set()
    for proc in resources.processes:
        proc.kill()
    if resources.pos_shared is not None:
        resources.pos_shared.close()
    logging.info("Processes terminated.")


# ---------------------------------------------------------------------------
# Per-mode runners. Each owns its own setup → foreground loop → cleanup, the
# same shape as ``main`` in xdof/envs/launch.py: build dependencies, call the
# control loop, wrap in try/finally with a single ``cleanup``.
# ---------------------------------------------------------------------------


def run_follower(args: Args) -> None:
    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )
    spawn = partial(_spawn_into, resources.processes)

    n_dofs_value = portal.mp.Value("i", 0)
    cmd_queue = portal.mp.Queue()

    spawn(
        portal.Process(
            _yam_polling_worker,
            args,
            resources.pos_shared,
            n_dofs_value,
            cmd_queue,
            resources.stop_event,
            "follower yam-hardware",
            name="follower-yam-control",
        )
    )
    n_dofs = _wait_for_dof(n_dofs_value)
    logging.info(f"Follower YAM ready, num_dofs={n_dofs}")

    server = _build_follower_server(args.server_port, n_dofs_value, resources.pos_shared, cmd_queue)

    try:
        server.start()  # blocks
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("follower mode failed")
        raise
    finally:
        cleanup(resources)


def run_leader(args: Args) -> None:
    client_robot = ClientRobot(args.server_port, host=args.server_host)
    initial_follower_pos = client_robot.get_joint_pos()
    logging.info(f"Initial follower joint pos: {initial_follower_pos}")

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(len(initial_follower_pos),), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )
    resources.pos_shared.array[:] = initial_follower_pos
    cmd_queue = portal.mp.Queue()

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _leader_control_worker,
            args,
            resources.pos_shared,
            cmd_queue,
            resources.stop_event,
            name="yam-leader-control",
        )
    )

    io_rate = RateRecorder(name="yam-leader web-port io", report_interval=1.0)
    io_rate.start()
    try:
        _run_leader_io_loop(client_robot, cmd_queue, resources.pos_shared, resources.stop_event, io_rate)
    except KeyboardInterrupt:
        pass
    except Exception:
        logging.exception("leader mode failed")
        raise
    finally:
        cleanup(resources)


def run_visualizer_local(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )
    n_dofs_value = portal.mp.Value("i", 0)

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _yam_polling_worker,
            args,
            resources.pos_shared,
            n_dofs_value,
            None,  # no cmd_queue — viewer doesn't send commands
            resources.stop_event,
            "viz-local yam-read",
            name="viz-local-yam",
        )
    )
    n_dofs = _wait_for_dof(n_dofs_value)
    logging.info(f"Viz YAM ready, num_dofs={n_dofs}")

    def get_pos() -> np.ndarray:
        return resources.pos_shared.array[:n_dofs].copy()

    try:
        _run_viewer_loop(arm_type, gripper_type, get_pos)
    except Exception:
        logging.exception("visualizer_local mode failed")
        raise
    finally:
        cleanup(resources)


def run_visualizer_remote(args: Args) -> None:
    arm_type = ArmType.from_string_name(args.arm)
    gripper_type = GripperType.from_string_name(args.gripper)

    with ClientRobot(args.server_port, host=args.server_host) as bootstrap:
        n_dofs = bootstrap.num_dofs()
    logging.info(f"Remote follower num_dofs={n_dofs}")

    resources = Resources(
        pos_shared=portal.SharedArray(shape=(_MAX_DOFS,), dtype=np.float64),
        stop_event=portal.mp.Event(),
    )

    spawn = partial(_spawn_into, resources.processes)
    spawn(
        portal.Process(
            _rpc_polling_worker,
            args.server_port,
            args.server_host,
            resources.pos_shared,
            resources.stop_event,
            "viz-remote rpc-poll",
            name="viz-remote-rpc",
        )
    )

    def get_pos() -> np.ndarray:
        return resources.pos_shared.array[:n_dofs].copy()

    try:
        _run_viewer_loop(arm_type, gripper_type, get_pos)
    except Exception:
        logging.exception("visualizer_remote mode failed")
        raise
    finally:
        cleanup(resources)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    # Make RateRecorder's logging.info output (and our own logging calls) visible.
    override_log_level()
    arm_type = ArmType.from_string_name(args.arm)

    if arm_type == ArmType.NO_ARM:
        # Every mode here needs an arm model (real YAM for follower/leader, an XML for the
        # visualizers); NO_ARM has no XML path and otherwise crashes deep inside the utils.
        raise ValueError("--arm 'no_arm' is not supported by minimum_gello; all modes require an arm")

    if args.mode == "leader" and args.sim:
        raise ValueError("Leader mode requires real hardware (--sim is not supported)")

    runners: Dict[str, Callable[[Args], None]] = {
        "follower": run_follower,
        "leader": run_leader,
        "visualizer_local": run_visualizer_local,
        "visualizer_remote": run_visualizer_remote,
    }
    runner = runners.get(args.mode)
    if runner is None:
        raise ValueError(f"Invalid mode: {args.mode}")
    runner(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
