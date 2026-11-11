"""``MotorInfo.timestamp`` is when that motor answered, not when the batch was read.

A consumer's missing-frame watchdog has nothing else to key off. It cannot use the payload -- a
motor holding station reports the same position every cycle and is perfectly healthy -- so a
stamp that advances on every read makes a motor that has stopped answering indistinguishable from
one that is answering. :class:`~i2rt.robots.arm7_osc.Arm7OSC` counts a *change* in this field as
one refresh, and with a shared read time it counted a refresh for every motor on every tick,
including a dead one. These tests pin the arrival-time semantics that fix is built on.

The chain is constructed without ``__init__`` because opening a CAN socket is the one thing these
tests must not do; only the two methods under test and the attributes they touch are set up.
"""

from typing import List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd
from i2rt.motor_drivers.utils import FeedbackFrameInfo

MOTOR_LIST = [(0x01, "DM4340"), (0x02, "DM4340"), (0x03, "DM4310")]


def _frame(motor_id: int) -> FeedbackFrameInfo:
    return FeedbackFrameInfo(
        id=motor_id,
        error_code="0x1",
        error_message="",
        position=0.0,
        velocity=0.0,
        torque=0.0,
        temperature_mos=30.0,
        temperature_rotor=30.0,
    )


class _Interface:
    """Answers for every motor, or refuses for the ones named in ``silent``."""

    def __init__(self, silent: Optional[List[int]] = None) -> None:
        self.silent = set(silent or [])

    def set_control(self, motor_id: int, **_: object) -> FeedbackFrameInfo:
        if motor_id in self.silent:
            raise RuntimeError(f"motor {motor_id} did not answer")
        return _frame(motor_id)


def _chain(silent: Optional[List[int]] = None) -> DMChainCanInterface:
    chain = object.__new__(DMChainCanInterface)
    chain.motor_list = MOTOR_LIST
    chain.motor_interface = _Interface(silent)
    chain.motor_offset = np.zeros(len(MOTOR_LIST))
    chain.motor_direction = np.ones(len(MOTOR_LIST))
    chain.absolute_positions = np.zeros(len(MOTOR_LIST))
    chain.state = [_frame(m) for m, _ in MOTOR_LIST]
    chain.state_lock = MagicMock()
    chain.state_lock.__enter__ = MagicMock(return_value=None)
    chain.state_lock.__exit__ = MagicMock(return_value=False)
    chain.channel = "test"
    chain._feedback_time = None
    return chain


def _commands() -> List[MotorCmd]:
    return [MotorCmd(torque=0.0, pos=0.0, vel=0.0, kp=0.0, kd=0.0) for _ in MOTOR_LIST]


def test_timestamps_come_from_arrival_not_from_the_read() -> None:
    """Reading twice without new feedback must not advance any motor's timestamp.

    This is the whole bug in one assertion. The old code took ``time.time()`` inside
    ``read_states``, so two reads of the same stale batch reported two different, ever-newer
    times, and no age ever grew.
    """
    chain = _chain()
    chain._set_commands(_commands())
    first = [info.timestamp for info in chain.read_states()]
    second = [info.timestamp for info in chain.read_states()]
    assert first == second, "timestamp advanced without new feedback"
    assert all(t > 0 for t in first), "no arrival time recorded"


def test_a_silent_motor_stops_advancing_while_the_others_continue() -> None:
    """The fault the watchdog exists for: one motor quiet, the rest of the chain healthy."""
    chain = _chain()
    chain._set_commands(_commands())
    before = [info.timestamp for info in chain.read_states()]

    # Motor 3 goes quiet. _set_commands raises when it reaches that motor, so the two ahead of it
    # are stamped and it is not -- which is what the age comparison below is measuring.
    chain.motor_interface = _Interface(silent=[0x03])
    with pytest.raises(RuntimeError):
        chain._set_commands(_commands())
    after = [info.timestamp for info in chain.read_states()]

    assert after[0] > before[0] and after[1] > before[1], "healthy motors stopped being stamped"
    assert after[2] == before[2], "the silent motor's timestamp advanced anyway"


def test_timestamps_are_old_rather_than_fresh_before_the_first_reply() -> None:
    """An unread chain must look stale, not current: the safe direction for a watchdog."""
    chain = _chain()
    assert all(info.timestamp == 0.0 for info in chain.read_states())
