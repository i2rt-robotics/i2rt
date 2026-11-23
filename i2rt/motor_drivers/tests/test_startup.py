from types import SimpleNamespace

import pytest

from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd


def test_startup_uses_zero_torque_instead_of_measured_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from i2rt.motor_drivers import dm_driver

    interface = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(dm_driver, "run_startup_checks", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", lambda **_kwargs: interface)

    def motor_on(chain: DMChainCanInterface) -> None:
        chain.state = [SimpleNamespace(torque=7.0)]
        chain.running = True

    monkeypatch.setattr(DMChainCanInterface, "_motor_on", motor_on)
    chain = DMChainCanInterface([(1, "DM4310")], np.zeros(1), np.ones(1), start_thread=False)
    try:
        assert chain.commands == [MotorCmd()]
    finally:
        chain.close()
