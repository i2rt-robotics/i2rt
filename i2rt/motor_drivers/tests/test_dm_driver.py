"""Regression tests for DMChainCanInterface control-loop thread startup.

These run without hardware: the CAN interface and the motor bring-up (``_motor_on``)
are mocked, and the control-loop body (``_set_torques_and_update_state``) is replaced
with a lightweight stand-in so we can observe whether/when the loop thread is spawned.
"""

import threading
import time
from typing import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

from i2rt.motor_drivers import dm_driver
from i2rt.motor_drivers.dm_driver import DMChainCanInterface


def _mock_hardware(monkeypatch: pytest.MonkeyPatch, loop_body: Callable[[DMChainCanInterface], None]) -> None:
    """Stub out everything in ``DMChainCanInterface.__init__`` that touches CAN hardware."""
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", MagicMock())

    def fake_motor_on(self: DMChainCanInterface) -> None:
        self.state = [MagicMock(torque=0.0)]
        self.running = True

    monkeypatch.setattr(DMChainCanInterface, "_motor_on", fake_motor_on)
    monkeypatch.setattr(DMChainCanInterface, "_set_torques_and_update_state", loop_body)


def _make_chain(start_thread: bool) -> DMChainCanInterface:
    return DMChainCanInterface(
        motor_list=[(1, "DM4310")],
        motor_offset=np.zeros(1),
        motor_direction=np.ones(1),
        channel="can0",
        start_thread=start_thread,
    )


@pytest.mark.parametrize(
    ("construct_start_thread", "explicit_start_calls", "expect_loop_started"),
    [
        # start_thread=True, no explicit call -> default direct construction used by
        # examples/single_motor, flow_base, and get_gripper_robot. THIS is the pattern the
        # guard bug silently broke: the loop must auto-start exactly once.
        pytest.param(True, 0, True, id="auto_start"),
        # start_thread=True then a redundant explicit start_thread() -> must stay idempotent.
        pytest.param(True, 1, True, id="auto_start+redundant_explicit"),
        # start_thread=False then explicit start_thread() -> the construct-then-start factory
        # pattern used by get_yam_robot and the dm_driver __main__ helper.
        pytest.param(False, 1, True, id="explicit_start"),
        # start_thread=False with two explicit calls -> idempotent (still one thread).
        pytest.param(False, 2, True, id="explicit_start+redundant_explicit"),
        # start_thread=False and never started -> caller opts out, loop must NOT spawn.
        pytest.param(False, 0, False, id="never_started"),
    ],
)
def test_control_loop_started_for_all_call_patterns(
    monkeypatch: pytest.MonkeyPatch,
    construct_start_thread: bool,
    explicit_start_calls: int,
    expect_loop_started: bool,
) -> None:
    """The 250 Hz control loop must spawn exactly once for every pattern that asks for it.

    Regression for the start_thread guard bug: ``__init__`` set ``start_thread_flag = True``
    before calling ``start_thread()``, so the ``if self.start_thread_flag: return`` guard
    short-circuited and the auto-start (``start_thread=True``) pattern never spawned the loop --
    silently, since ``_motor_on`` had already populated state. This matrix exercises every
    construct/explicit-call combination and asserts the loop thread starts where expected,
    exactly once (idempotent), and never where it is not requested.
    """
    spawns: list[int] = []
    lock = threading.Lock()

    def loop_body(self: DMChainCanInterface) -> None:
        with lock:
            spawns.append(1)

    _mock_hardware(monkeypatch, loop_body)

    chain = _make_chain(start_thread=construct_start_thread)
    for _ in range(explicit_start_calls):
        chain.start_thread()

    # Wait for the loop thread to enter loop_body (fast when it spawns), then settle briefly
    # so a bug-induced *second* thread would also register before we count.
    if expect_loop_started:
        for _ in range(100):
            with lock:
                if spawns:
                    break
            time.sleep(0.02)
    time.sleep(0.2)

    with lock:
        spawn_count = len(spawns)

    expected_spawns = 1 if expect_loop_started else 0
    assert spawn_count == expected_spawns, (
        f"expected {expected_spawns} control-loop thread(s), got {spawn_count} "
        f"(construct start_thread={construct_start_thread}, explicit calls={explicit_start_calls})"
    )
    assert chain.start_thread_flag is expect_loop_started
