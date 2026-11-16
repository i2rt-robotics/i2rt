"""Set a 7-DOF arm's motor KT_Value/GREF to what its config's `motor_registers` declares, and save to Flash.

The startup config check refuses to start an arm whose motors do not hold those values; this is the
tool that puts them there. It refuses a chain that does not hold the config's motor types (Gr) before
writing anything, then writes and saves only the registers that differ, one register at a time, and
finishes with the same check `get_yam_robot` runs, read-only. The gripper is not touched: the config
does not declare it. Stop anything else using the bus first.

    python scripts/set_7dof_motor_registers.py --arm yam_7dof_left --channel can_left --dry-run
    python scripts/set_7dof_motor_registers.py --arm yam_7dof_left --channel can_left

A register read only reports RAM, so power-cycle the arm and run with --dry-run again to prove the save.
"""

import logging
from typing import Literal

import tyro

from i2rt.motor_config_tool.dm_motor_registers import (
    REG_BY_NAME,
    read_register,
    save_register_to_flash,
    write_register,
)
from i2rt.motor_config_tool.utils import RawCanInterface
from i2rt.motor_drivers.motor_check import matches, run_startup_checks, verify_motor_types
from i2rt.robots.utils import ArmType, _load_arm_config


def set_registers(iface: RawCanInterface, declared: dict[int, dict[str, float]], dry_run: bool) -> None:
    """Write and save every declared register a motor does not already hold."""
    for motor_id, registers in declared.items():
        for name, want in registers.items():
            spec = REG_BY_NAME[name]
            have = read_register(iface, motor_id, name)
            if matches(spec, have, want):
                print(f"motor {motor_id} {name}: {have:.6g} ok")
                continue
            if dry_run:
                print(f"motor {motor_id} {name}: {have:.6g} -> {want:.6g} (dry run, not written)")
                continue
            # Never save a write that did not land: a save after a discarded write can revert other registers.
            echo = write_register(iface, motor_id, name, want)
            if not matches(spec, echo, want):
                raise SystemExit(f"motor {motor_id} {name}: wrote {want:.6g}, motor echoed {echo:.6g}; not saved")
            save_register_to_flash(iface, motor_id, name)
            print(f"motor {motor_id} {name}: {have:.6g} -> {read_register(iface, motor_id, name):.6g} saved")


def main(arm: Literal["yam_7dof_left", "yam_7dof_right"], channel: str = "can0", dry_run: bool = False) -> None:
    """Set the arm's declared motor registers on `channel`.

    Args:
        arm: Which 7-DOF arm config to read the declared values from.
        channel: SocketCAN channel the arm is on.
        dry_run: Print what would change and write nothing.
    """
    hw = _load_arm_config(ArmType.from_string_name(arm))
    verify_motor_types(channel, hw.motor_list)  # a chain that is not this arm gets no Flash writes
    iface = RawCanInterface(channel=channel, bustype="socketcan", name="set_7dof_motor_registers")
    try:
        set_registers(iface, hw.motor_registers, dry_run)
    finally:
        iface.close()
    if dry_run:
        return
    run_startup_checks(
        channel,
        hw.motor_list,
        check_motor_types=True,
        check_motor_config=True,
        control_mode="MIT",
        repair=False,
        expected_registers=hw.motor_registers,
    )
    print(f"{arm} on {channel}: startup check passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
