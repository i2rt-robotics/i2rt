"""Applications can control i2rt verbosity without changing their own logging."""

import logging
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any

import can
import pytest

from i2rt.motor_drivers import can_interface, dm_driver
from i2rt.utils.utils import RateRecorder


@pytest.mark.parametrize(
    "module",
    [
        "i2rt.motor_drivers.dm_driver",
        "i2rt.flow_base.flow_base_controller",
        "i2rt.flow_base.linear_rail_controller",
    ],
)
def test_import_does_not_configure_root_logging(module: str) -> None:
    # A fresh process catches import-time basicConfig even when pytest already has handlers.
    code = f"""
import importlib
import logging
root = logging.getLogger()
level, handlers = root.level, root.handlers[:]
importlib.import_module({module!r})
assert root.level == level
assert root.handlers == handlers
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


def test_package_level_suppresses_rate_reports_but_keeps_application_info(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO):
        with RateRecorder("visible report"):
            pass
        assert caplog.records[-1].name == "i2rt.utils.utils"
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="i2rt"), caplog.at_level(logging.INFO):
            with RateRecorder("hidden report"):
                pass
            logging.getLogger("application").info("application remains visible")
        assert caplog.messages == ["application remains visible"]


def test_can_transport_records_have_a_module_name(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: SimpleNamespace(state=can.BusState.ACTIVE))
    with caplog.at_level(logging.INFO):
        can_interface.CanInterface(channel="can_test")
    assert caplog.records[-1].name == "i2rt.motor_drivers.can_interface"


@pytest.mark.parametrize("fail", [False, True], ids=["success", "CAN-failure"])
def test_motor_bringup_never_changes_root_level(fail: bool, caplog: pytest.LogCaptureFixture) -> None:
    levels = []

    def exchange(*args: Any) -> object:
        levels.append(logging.getLogger().level)
        if fail:
            raise RuntimeError("CAN unavailable")
        return object()

    interface = SimpleNamespace(
        _send_message_get_response=exchange,
        parse_recv_message=lambda *args, **kwargs: SimpleNamespace(error_code="0x1"),
    )
    with caplog.at_level(logging.INFO):
        if fail:
            with pytest.raises(RuntimeError, match="CAN unavailable"):
                dm_driver.DMSingleMotorCanInterface.motor_on(interface, 1, "DM4310")
        else:
            dm_driver.DMSingleMotorCanInterface.motor_on(interface, 1, "DM4310")
        assert logging.getLogger().level == logging.INFO
    assert levels == [logging.INFO]


def test_control_loop_errors_use_logging_and_still_propagate(
    caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_commands: Any) -> None:
        raise OSError("CAN unavailable")

    chain = object.__new__(dm_driver.DMChainCanInterface)
    chain.channel = "can_test"
    chain.running = True
    chain.commands = []
    chain.command_lock = threading.RLock()
    chain._rate_recorder = RateRecorder("test")
    chain._set_commands = fail
    with caplog.at_level(logging.ERROR), pytest.raises(OSError, match="CAN unavailable"):
        chain._set_torques_and_update_state()
    assert not chain.running
    assert caplog.records[-1].name == "i2rt.motor_drivers.dm_driver"
    assert "DM Error in control loop" in caplog.messages[-1]
    assert "DM Error in control loop" not in capsys.readouterr().out
