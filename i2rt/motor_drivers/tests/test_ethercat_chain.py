"""EtherCATMotorChain and the python-can bus over the EtherCAT-to-CAN board.

The chain runs against a fake master that answers like DM motors, so the startup and shutdown
sequences -- which frames reach the motors, in which order -- can be asserted exactly. The python-can
half runs against the real compiled master in loopback.
"""

import functools
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from i2rt.motor_drivers import ethercat_chain
from i2rt.motor_drivers.ethercat_chain import EtherCATMotorChain, parse_ethercat_channel
from i2rt.motor_drivers.utils import (
    DM_CLEAR_ERR,
    DM_DISABLE,
    DM_ENABLE,
    MotorType,
    float_to_uint,
    unpack_mit_feedback,
)

MOTORS = [(1, MotorType.DM6248), (2, MotorType.DM4340), (3, MotorType.DM4310)]


def _feedback(state: int, can_id: int, pos: float, motor_type: str) -> np.ndarray:
    c = MotorType.get_motor_constants(motor_type)
    p = float_to_uint(pos, c.POSITION_MIN, c.POSITION_MAX, 16)
    v = float_to_uint(0.0, c.VELOCITY_MIN, c.VELOCITY_MAX, 12)
    t = float_to_uint(0.0, c.TORQUE_MIN, c.TORQUE_MAX, 12)
    return np.array(
        [(state << 4) | (can_id & 0xF), p >> 8, p & 0xFF, v >> 4, ((v & 0xF) << 4) | (t >> 8), t & 0xFF, 36, 41],
        dtype=np.uint8,
    )


class FakeBus:
    """Stands in for ``i2rt_ecat.EcatCanBus``: each motor answers every read with DM feedback.

    Tracks each motor's DM state the way the firmware does: enable -> 1, disable or clear-error -> 0,
    and a motor in ``faults`` answers an enable with that fault code instead.
    """

    last: Optional["FakeBus"] = None

    def __init__(self, interface: str, motors: Sequence[Tuple[int, int, int]], release: np.ndarray, **_: object):
        self.ids = [m[2] for m in motors]
        self.n = len(self.ids)
        self.types = [t for _, t in MOTORS[: self.n]]
        self.frames = np.array(release, dtype=np.uint8)
        self.history: List[bytes] = []  # distinct frames latched for motor 0, in order
        self.state = np.zeros(self.n, dtype=int)
        self.pos = np.zeros(self.n)
        self.silent: set = set()
        self.faults: Dict[int, int] = {}
        self.count = np.zeros(self.n, dtype=np.uint64)
        self.closed = False
        FakeBus.last = self

    def set_frames(self, frames: np.ndarray) -> None:
        self.frames = np.array(frames, dtype=np.uint8)
        if not self.history or self.history[-1] != bytes(self.frames[0]):
            self.history.append(bytes(self.frames[0]))
        for i in range(self.n):
            frame = bytes(self.frames[i])
            if frame == DM_ENABLE:
                self.state[i] = self.faults.get(self.ids[i], 1)
            elif frame in (DM_DISABLE, DM_CLEAR_ERR):
                self.state[i] = 0

    def read(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rx = np.zeros((self.n, 8), dtype=np.uint8)
        for i in range(self.n):
            if self.ids[i] in self.silent:
                continue
            self.count[i] += 1
            rx[i] = _feedback(int(self.state[i]), self.ids[i], float(self.pos[i]), self.types[i])
        return rx, np.full(self.n, time.time()), self.count.copy()

    def stats(self) -> Dict[str, int]:
        return {}

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> type:
    monkeypatch.setattr(ethercat_chain, "_bus_class", lambda: FakeBus)
    return FakeBus


def _silent_fake(ids: set, faults: Optional[Dict[int, int]] = None) -> type:
    class Configured(FakeBus):
        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)  # type: ignore[arg-type]
            self.silent = set(ids)
            self.faults = dict(faults or {})

    return Configured


def test_a_silent_motor_is_named_and_nothing_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ethercat_chain, "_bus_class", lambda: _silent_fake({2}))
    with pytest.raises(RuntimeError, match=r"CAN id \[2\].*nothing was enabled.*MASTER_ID == ID"):
        EtherCATMotorChain(MOTORS, interface="eth9")
    bus = FakeBus.last
    assert bus is not None and bus.closed
    assert DM_ENABLE not in bus.history and DM_CLEAR_ERR not in bus.history
    assert bus.history[-1] == DM_DISABLE  # still released and disabled on the way out


def test_startup_clears_then_enables_then_releases(fake: type) -> None:
    chain = EtherCATMotorChain(MOTORS, interface="eth9")
    try:
        bus = FakeBus.last
        assert bus is not None
        i_clear, i_enable = bus.history.index(DM_CLEAR_ERR), bus.history.index(DM_ENABLE)
        assert i_clear < i_enable
        assert bus.history[i_enable + 1] == bytes(chain._release[0])  # release, not a torque command
        assert all(info.error_code == "0x1" for info in chain.read_states())
    finally:
        chain.close()


def test_enable_false_detects_the_motors_and_leaves_them_disabled(fake: type) -> None:
    chain = EtherCATMotorChain(MOTORS, interface="eth9", enable=False)
    try:
        bus = FakeBus.last
        assert bus is not None and DM_ENABLE not in bus.history
        assert all(info.error_code == "0x0" for info in chain.read_states())
    finally:
        chain.close()


def test_a_fault_on_enable_raises_and_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ethercat_chain, "_bus_class", lambda: _silent_fake(set(), faults={3: 0xA}))
    with pytest.raises(RuntimeError, match="motor 3 faulted on enable: over current"):
        EtherCATMotorChain(MOTORS, interface="eth9")
    assert FakeBus.last is not None and FakeBus.last.history[-1] == DM_DISABLE


def test_positions_and_commands_follow_the_can_chains_calibration(fake: type) -> None:
    offset, direction = np.array([0.5, 0.0, -0.25]), np.array([-1.0, 1.0, 1.0])
    chain = EtherCATMotorChain(MOTORS, motor_offset=offset, motor_direction=direction, interface="eth9")
    try:
        bus = FakeBus.last
        assert bus is not None
        bus.pos[:] = [1.5, -0.3, 0.75]
        pos = [info.pos for info in chain.read_states()]
        np.testing.assert_allclose(pos, (bus.pos - offset) * direction, atol=1e-3)

        chain.set_commands(np.zeros(3), pos=np.array([0.2, 0.0, 0.0]), kp=np.array([10.0, 0, 0]), get_state=False)
        c = MotorType.get_motor_constants(MotorType.DM6248)
        sent = bus.frames[0]
        sent_pos = ((int(sent[0]) << 8) | int(sent[1])) * (c.POSITION_MAX - c.POSITION_MIN) / 65535 + c.POSITION_MIN
        assert sent_pos == pytest.approx(0.2 * -1.0 + 0.5, abs=1e-3)  # model -> motor: q * dir + offset
    finally:
        chain.close()


def test_a_motor_that_stops_reporting_enabled_stops_being_stamped(fake: type) -> None:
    chain = EtherCATMotorChain(MOTORS, interface="eth9")
    try:
        bus = FakeBus.last
        assert bus is not None
        first = [info.timestamp for info in chain.read_states()]
        assert chain.running
        bus.state[1] = 0  # joint 2 disables itself (e.g. its firmware timeout)
        time.sleep(0.01)
        second = [info.timestamp for info in chain.read_states()]
        assert second[0] > first[0] and second[2] > first[2]
        assert second[1] == first[1]
        assert chain.read_states()[1].error_code == "0x0"
        assert not chain.running, "MotorChainRobot stops on this flag, as it does for a CAN chain"
    finally:
        chain.close()


def test_get_yam_robot_runs_an_arm_over_an_ecat_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """``channel="ecat:..."`` builds this chain, after the register checks, in place of the CAN one."""
    from i2rt.robots import get_robot
    from i2rt.robots.utils import ArmType, GripperType, _load_arm_config

    hw = _load_arm_config(ArmType.YAM_7DOF_RIGHT)

    class ArmBus(FakeBus):
        def __init__(self, *a: object, **k: object) -> None:
            super().__init__(*a, **k)  # type: ignore[arg-type]
            self.types = [t for _, t in hw.motor_list]

    checked: List[str] = []
    monkeypatch.setattr(ethercat_chain, "_bus_class", lambda: ArmBus)
    monkeypatch.setattr(get_robot, "run_startup_checks", lambda channel, *_, **__: checked.append(channel))
    monkeypatch.setattr(get_robot, "MotorChainRobot", lambda **kw: kw["motor_chain"])
    chain = get_robot.get_yam_robot(
        channel="ecat:eth9:0:2", arm_type=ArmType.YAM_7DOF_RIGHT, gripper_type=GripperType.NO_GRIPPER
    )
    try:
        assert isinstance(chain, EtherCATMotorChain) and chain.running
        assert (chain.interface, chain.slave, chain.channel) == ("eth9", 0, 2)
        assert checked == ["ecat:eth9:0:2"]
        np.testing.assert_array_equal(chain.motor_direction, hw.directions)
    finally:
        chain.close()

    with pytest.raises(ValueError, match="takes an arm only"):
        get_robot.get_yam_robot(channel="ecat:eth9", arm_type=ArmType.NO_ARM, gripper_type=GripperType.LINEAR_4310)


def test_close_releases_then_disables_and_is_idempotent(fake: type) -> None:
    chain = EtherCATMotorChain(MOTORS, interface="eth9")
    bus = FakeBus.last
    assert bus is not None
    chain.set_commands(np.ones(3), get_state=False)
    chain.close()
    chain.close()
    assert bus.closed and bus.history[-2:] == [bytes(chain._release[0]), DM_DISABLE]
    assert not chain.running
    with pytest.raises(RuntimeError, match="closed"):
        chain.set_commands(np.zeros(3))


def test_a_missing_extension_says_how_to_build_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "i2rt_ecat", None)
    with pytest.raises(ImportError, match="uv sync"):
        EtherCATMotorChain(MOTORS, interface="eth9")


def test_an_interface_is_required() -> None:
    with pytest.raises(ValueError, match="interface is required"):
        EtherCATMotorChain(MOTORS)


@pytest.mark.parametrize(
    "channel, parsed",
    [("ecat:enp99s0", ("enp99s0", 0, 1)), ("ecat:enp99s0:1:2", ("enp99s0", 1, 2))],
)
def test_ethercat_channel_names(channel: str, parsed: tuple) -> None:
    assert parse_ethercat_channel(channel) == parsed


def test_a_malformed_ethercat_channel_is_refused() -> None:
    with pytest.raises(ValueError, match="ecat:<nic>"):
        parse_ethercat_channel("ecat:enp99s0:1")


# ---- python-can over the board, on the real master in loopback ------------------------------


@pytest.fixture
def loopback_master(monkeypatch: pytest.MonkeyPatch) -> None:
    i2rt_ecat = pytest.importorskip("i2rt_ecat")
    monkeypatch.setattr(ethercat_chain, "_bus_class", lambda: functools.partial(i2rt_ecat.EcatCanBus, loopback=True))


def test_can_tools_see_every_frame_on_a_shared_board(loopback_master: None) -> None:
    """Two python-can buses on one board both see each frame, as two SocketCAN sockets would -- the
    tools open a register interface beside a motor interface on the same channel."""
    import can

    from i2rt.motor_drivers.can_interface import open_can_bus

    a, b = open_can_bus("ecat:lo"), open_can_bus("ecat:lo")
    try:
        a.send(can.Message(arbitration_id=0x7FF, data=[1, 0, 0x33, 7], is_extended_id=False))
        for bus in (a, b):
            msg = bus.recv(timeout=0.5)
            assert msg is not None and msg.arbitration_id == 0x7FF and list(msg.data) == [1, 0, 0x33, 7]
        with pytest.raises(can.CanError, match="standard-id"):
            a.send(can.Message(arbitration_id=0x123, data=[0], is_extended_id=True))
    finally:
        a.shutdown()
        b.shutdown()


def test_the_register_tool_runs_over_the_board(loopback_master: None) -> None:
    """``dm_motor_registers`` matches a reply by echo, not by id, so in loopback -- where the request
    comes back as its own reply -- a read completes and decodes the (zero) value field."""
    from i2rt.motor_config_tool.dm_motor_registers import _open_bus, read_register

    with _open_bus("ecat:lo") as iface:
        assert read_register(iface, 3, "MST_ID") == 0


def test_dm_tools_answer_on_the_motors_own_id_over_the_board(loopback_master: None) -> None:
    """The board passes only replies on MASTER_ID == ID, so a DM tool asked for ``p16`` -- the CLI
    default -- must still listen there: otherwise ``motor_on`` sends ENABLE, misses the reply and
    raises before its ``motor_off``, leaving the motor enabled. Loopback answers on the frame's own id."""
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface, ReceiveMode

    iface = DMSingleMotorCanInterface(channel="ecat:lo", receive_mode=ReceiveMode.p16)
    try:
        assert iface.receive_mode is ReceiveMode.same
        iface.motor_off(3)  # raises if it waits on 3 + 16
    finally:
        iface.close()


def test_the_mit_feedback_layout_matches_the_fake() -> None:
    fb = unpack_mit_feedback(bytes(_feedback(1, 5, 1.25, MotorType.DM4340)), MotorType.get_motor_constants("DM4340"))
    assert fb.error == 1 and fb.position == pytest.approx(1.25, abs=1e-3)
