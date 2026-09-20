import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from i2rt.flow_base.flow_base_controller import VehicleMotorController
from i2rt.flow_base.linear_rail_controller import SingleMotorControlInterface
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd
from i2rt.utils.utils import RateRecorder


def test_can_io_does_not_block_command_updates() -> None:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.running = True
    chain.command_lock, chain.state_lock = threading.RLock(), threading.Lock()
    chain.commands, chain.motor_list = [MotorCmd()], [(1, "fake")]
    chain._rate_recorder, chain._report_interval = RateRecorder(), 30.0
    chain.same_bus_device_driver = None
    chain._update_absolute_positions = lambda _feedback: None
    with ThreadPoolExecutor(1) as pool:

        def motor_io(_commands: list[MotorCmd]) -> list[SimpleNamespace]:
            pool.submit(chain.set_commands, np.ones(1), get_state=False).result(timeout=0.2)
            chain.running = False
            return [SimpleNamespace(error_code="0x1")]

        chain._set_commands = motor_io
        chain._set_torques_and_update_state()
    assert chain.commands[0].torque == 1.0


def test_base_and_rail_updates_preserve_each_other_and_inflight_snapshot() -> None:
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.command_lock = threading.RLock()
    chain.commands = [MotorCmd() for _ in range(3)]
    chain.motor_list = [(index, "fake") for index in range(3)]
    snapshot = chain.commands
    base = VehicleMotorController.__new__(VehicleMotorController)
    base.num_casters, base.motor_interface = 1, chain
    rail = SingleMotorControlInterface(chain, target_motor_idx=2)
    barrier = threading.Barrier(2)
    original = chain.set_commands

    def interleaved_full_update(*args: object, **kwargs: object) -> object:
        # The old read/modify/write paths both read the old list before replacing it.
        barrier.wait(timeout=1.0)
        return original(*args, **kwargs)

    chain.set_commands = interleaved_full_update
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(base.set_velocities, {"steer_vel": [1.0], "drive_vel": [2.0]})
        second = pool.submit(rail.set_velocity, 3.0)
        first.result(timeout=2.0)
        second.result(timeout=2.0)
    assert [command.vel for command in chain.commands] == [1.0, 2.0, 3.0]
    assert [command.vel for command in snapshot] == [0.0, 0.0, 0.0]
