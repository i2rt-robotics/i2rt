"""scripts/set_7dof_motor_registers.py against an in-memory register bank: what it writes, saves and skips."""

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "set_7dof_motor_registers.py"
DECLARED = {1: {"KT_Value": 0.072029, "GREF": 1.0}, 3: {"KT_Value": 0.044631, "GREF": 1.0}}


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, dict[tuple[int, str], float], list[tuple]]:
    """The script with its register I/O on a fake bank: motor 1 already right, motor 3 on the old GREF."""
    spec = importlib.util.spec_from_file_location("set_7dof_motor_registers", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bank = {(1, "KT_Value"): 0.072029, (1, "GREF"): 1.0, (3, "KT_Value"): 0.044631, (3, "GREF"): 0.87}
    calls: list[tuple] = []

    def write(_: Any, motor_id: int, name: str, value: float) -> float:
        calls.append(("write", motor_id, name))
        bank[motor_id, name] = value
        return value

    monkeypatch.setattr(module, "read_register", lambda _, motor_id, name: bank[motor_id, name])
    monkeypatch.setattr(module, "write_register", write)
    monkeypatch.setattr(
        module, "save_register_to_flash", lambda _, motor_id, name: calls.append(("save", motor_id, name))
    )
    return module, bank, calls


def test_only_the_differing_register_is_written_then_saved(script: Any) -> None:
    module, bank, calls = script
    module.set_registers(None, DECLARED, dry_run=False)
    assert calls == [("write", 3, "GREF"), ("save", 3, "GREF")]
    assert bank[3, "GREF"] == 1.0


def test_a_dry_run_writes_and_saves_nothing(script: Any) -> None:
    module, bank, calls = script
    module.set_registers(None, DECLARED, dry_run=True)
    assert calls == []
    assert bank[3, "GREF"] == 0.87


def test_a_write_the_motor_did_not_take_is_never_saved(script: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    module, _, calls = script
    monkeypatch.setattr(module, "write_register", lambda *_: 0.87)  # the motor echoes its old value
    with pytest.raises(SystemExit, match="not saved"):
        module.set_registers(None, DECLARED, dry_run=False)
    assert calls == []
