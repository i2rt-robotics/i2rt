"""Firmware flashing for the yam_teaching_handle's ioheart encoder over CAN.

Talks to the STM32 bootloader (CAN IDs 0x700 cmd / 0x701+ reply) to upload a
firmware image one 1 KiB page at a time, with per-page and whole-image CRC-32
verification (poly ``0x104C11DB7``). The encoder in the running application is
restarted into bootloader mode automatically via the live encoder REQ channel
(0x50E).

Scope: the leader-arm CAN bus carries the teaching handle's passive encoder
(and only that encoder). ``flash`` auto-detects every encoder it sees on the
bus and flashes them all, so the operator never has to type a device ID — pick
the right SocketCAN channel and the tool does the rest.

CLI: ``python -m i2rt.utils.can_flash FIRMWARE.bin [--channel can0] [--bitrate 1000000]``.

To list devices on the bus, see ``python -m i2rt.utils.encoder_manager list-devices``.
"""

import os
import time
from typing import Annotated, Any

import can
import crcmod
import tyro

MAX_FSIZE = 52 * 1024  # Max file size
FLASH_PG_SIZE = 1024  # Page size in bytes

# Bootloader commands
BL_WBUF = 1
BL_WPAGE = 2
BL_WCRC = 3
BL_PING = 4

# Bootload CAN IDs
CANID_BL_CMD = 0x700
CANID_BL_RPL_BASE = 0x701
CANID_ENCODER_REQ = 0x50E
CANID_ENCODER_REPORT = 0x50F

# Encoder requests
ENCODER_REQ_GET_REPORT = 0x02
ENCODER_REQ_RESTART = 0x0F

ALL_DEVICE = 0xFF
"""All devices for broadcasting requests."""

PAGE_RETRIES = 10

DISCOVERY_TIMEOUT = 30.0  # Max wall-clock seconds to wait for a device to enter bootloader mode


def is_bl_response_id(can_id: int) -> bool:
    """Whether ``can_id`` is in the bootloader reply range (0x701..0x7FF)."""
    return 0 <= can_id - CANID_BL_RPL_BASE <= 254


def canmsg(id: int, data: list[int] | bytes | bytearray) -> can.Message:
    """Create a CAN message from an arbitration ID and a data payload (<= 8 bytes)."""
    if len(data) > 8:
        raise ValueError("invalid data")
    m = can.Message(arbitration_id=id, is_extended_id=False, data=data)
    return m


def bl_cmd(
    bus: can.BusABC,
    device_id: int,
    cmd: int,
    par1: int,
    par2: list[int] | bytes | bytearray,
    timeout: float | None = None,
) -> None:
    """Send a command to the bootloader."""
    # Create data for CAN message
    data = bytearray(8)
    data[0] = device_id
    data[1] = cmd
    data[2:4] = par1.to_bytes(2, "little")
    data[4:8] = par2[::-1]
    bus.send(canmsg(CANID_BL_CMD, data), timeout)


def bl_wait_resp(bus: can.BusABC, device_id: int, expected_cmd: int, timeout: float) -> int | None:
    """Wait for a bootloader response (renamed board_id to device_id)."""
    to_time = time.time() + timeout
    while time.time() < to_time:
        m = bus.recv(max(0, to_time - time.time()))
        if m is not None:
            if (
                (is_bl_response_id(m.arbitration_id))
                and (m.dlc >= 3)
                and (m.data[0] == device_id)
                and (m.data[1] == expected_cmd)
            ):
                return m.data[2]
    return None


def bl_cmd_response(
    bus: can.BusABC,
    device_id: int,
    cmd: int,
    par1: int,
    par2: list[int] | bytes | bytearray,
    timeout_sec: float = 0.05,
    retries: int = 10,
) -> int:
    """Send a command to the bootloader and wait for a response."""
    if retries == 0:
        raise RuntimeError("Did not receive reply from device")
    bl_cmd(bus, device_id, cmd, par1, par2)
    r = bl_wait_resp(bus, device_id, cmd, timeout_sec)
    if r is None:
        return bl_cmd_response(bus, device_id, cmd, par1, par2, timeout_sec, retries - 1)
    if r > 0:
        raise RuntimeError(f"Bootloader command {cmd} error #{r}")
    return r


def bl_wait_for_connection(bus: can.BusABC, device_id: int, timeout_sec: float = 0.1, retries: int = 10) -> bool:
    """Wait for a connection to the bootloader."""
    for _ in range(retries):
        # Ping bootloader
        bl_cmd(bus, device_id, BL_PING, 0, [0] * 4)
        # Wait for a response
        r = bl_wait_resp(bus, device_id, BL_PING, timeout_sec)
        if r is not None:
            return True
    return False


def bl_list_connected_devices(bus: can.BusABC, timeout: float = 0.1) -> set[int]:
    """List all connected bootloader devices."""
    bootloaders: set[int] = set()
    bl_cmd(bus, ALL_DEVICE, BL_PING, 0, [0] * 4)
    to_time = time.time() + timeout
    while time.time() < to_time:
        m = bus.recv(timeout)
        if m is None or m.dlc < 3 or m.data[1] != BL_PING:
            continue
        if not is_bl_response_id(m.arbitration_id):
            continue
        device_id = m.data[0]
        bootloaders.add(device_id)
    return bootloaders


def get_can_bus(channel: str, bitrate: int = 1000000) -> can.BusABC:
    """Open a SocketCAN bus on ``channel`` (e.g. ``can0``)."""
    return can.interface.Bus(interface="socketcan", channel=channel, bitrate=bitrate)


def encoder_list_devices(bus: can.BusABC, timeout: float = 0.1) -> set[int]:
    """List connected encoder devices."""
    req = can.Message(
        arbitration_id=CANID_ENCODER_REQ,
        is_extended_id=False,
        data=[ALL_DEVICE, ENCODER_REQ_GET_REPORT],
    )
    bus.send(req, timeout)
    encoders: set[int] = set()
    to_time = time.time() + timeout
    while time.time() < to_time:
        m = bus.recv(timeout)
        if m is not None:
            if m.arbitration_id == CANID_ENCODER_REPORT and m.dlc > 0:
                encoders.add(m.data[0])
    return encoders


def encoder_restart(bus: can.BusABC, device_id: int, timeout: float | None = None) -> None:
    """Restart the encoder."""
    req = can.Message(
        arbitration_id=CANID_ENCODER_REQ,
        is_extended_id=False,
        data=[device_id, ENCODER_REQ_RESTART],
    )
    bus.send(req, timeout)


def flash_page(
    bus: can.BusABC,
    device_id: int,
    page: int,
    pcrc: Any,
    page_data: dict[int, bytearray],
) -> None:
    """(try to) Flash a single page to the mcu."""
    for w, d in page_data.items():
        if w % 16 == 0:
            print(".", end="")
        # Send data and get response
        bl_cmd_response(bus, device_id, BL_WBUF, w, d)

    bl_cmd_response(bus, device_id, BL_WPAGE, page, pcrc.digest())


def flash_device(
    bus: can.BusABC,
    device_id: int,
    filepath: str,
    firmware_data: bytearray,
    num_pages: int,
) -> None:
    """Core flashing logic for a single device."""
    print(f"Flashing {filepath} {num_pages=} to {device_id=}...")

    # Discover devices and restart encoders if needed
    print("Discovering devices...")
    deadline = time.time() + DISCOVERY_TIMEOUT
    while time.time() < deadline:
        try:
            bootloaders = bl_list_connected_devices(bus)
            if device_id in bootloaders:
                break
        except Exception:
            pass
        try:
            encoders = encoder_list_devices(bus)
            if device_id in encoders:
                print(f"Restarting encoder {device_id}...")
                encoder_restart(bus, device_id)
        except Exception:
            pass
        time.sleep(0.1)
    else:
        raise TimeoutError(f"device {device_id} never entered bootloader mode within {DISCOVERY_TIMEOUT}s")

    # Reset & connect to device with extended timeout for restart
    print(f"Attempting to connect to device with ID {device_id}")
    if not bl_wait_for_connection(bus, device_id, timeout_sec=0.05, retries=10):
        raise RuntimeError("Could not connect to device.")
    print(f"Connected to device {device_id}. Uploading {filepath}")

    # Ability to retry pages that failed
    acrc = crcmod.Crc(0x104C11DB7, initCrc=0xFFFFFFFF, rev=False)  # App CRC
    for p in range(num_pages):
        # Start calculating page CRC
        pcrc = crcmod.Crc(0x104C11DB7, initCrc=0xFFFFFFFF, rev=False)

        # Iterate over each 32-bit word of the page
        # First generate CRCs and data
        page_data: dict[int, bytearray] = {}
        for w in range(FLASH_PG_SIZE // 4):
            # Get next data to send
            a = (p * FLASH_PG_SIZE) + (w * 4)
            d = firmware_data[a : a + 4][::-1]  # take out 4 bytes and reverse them

            # update CRCs
            acrc.update(d)
            pcrc.update(d)

            page_data[w] = d

        page_success = False
        for i in range(PAGE_RETRIES):
            print(f"Flashing page {p}/{num_pages - 1} retry {i + 1} ", end="")
            try:
                flash_page(bus, device_id, p, pcrc, page_data)
                print(" CRC OK")
                page_success = True
                break
            except RuntimeError as e:
                print("Error flashing page: ", e)
        if not page_success:
            raise RuntimeError("Page write failed")

    print(f"Verifying {num_pages} pages, crc={acrc.digest().hex()}...")
    bl_cmd_response(bus, device_id, BL_WCRC, num_pages, list(acrc.digest()))
    time.sleep(0.01)  # bootloader refresh watchdog

    print("Device flashed successfully")
    print("Device will restart automatically after timeout")


def flash(
    filepath: Annotated[str, tyro.conf.Positional],
    channel: str = "can0",
    bitrate: int = 1000000,
    device_id: int | None = None,
) -> None:
    """Flash firmware to every encoder on the bus.

    Discovers all encoders (in either application or bootloader mode), restarts
    any application-mode encoders into the bootloader to silence application
    chatter, then flashes each one in turn. Errors out if no encoders respond.

    Pass ``--device-id`` to restrict the flash to a single device (a safeguard
    against mass-reflashing the wrong bus); by default every discovered encoder
    is flashed.
    """
    file_size = os.path.getsize(filepath)
    if file_size > MAX_FSIZE:
        raise ValueError("File size must be <= 52KB")
    with open(filepath, "rb") as f:
        b = bytearray(f.read())

    pad = (-len(b)) % FLASH_PG_SIZE  # 0 when already page-aligned
    if pad:
        b.extend(bytearray(pad))  # Extend to the next whole page
    num_pages = len(b) // FLASH_PG_SIZE

    with get_can_bus(channel, bitrate) as bus:
        print("Discovering devices...")
        bootloaders = bl_list_connected_devices(bus)
        encoders = encoder_list_devices(bus)

        if device_id is not None:
            encoders = {d for d in encoders if d == device_id}

        for encoder_id in encoders:
            print(f"Restarting encoder {encoder_id} to bootloader mode...")
            encoder_restart(bus, encoder_id)
        if encoders:
            time.sleep(0.1)
            bootloaders = bl_list_connected_devices(bus)

        if device_id is not None:
            # Re-apply the filter after re-discovery so only the requested device is flashed.
            bootloaders = {d for d in bootloaders if d == device_id}

        if not bootloaders:
            raise SystemExit("No encoder devices found on the bus.")

        print(f"Found devices in bootloader mode: {sorted(bootloaders)}")
        failed: list[int] = []
        for dev in sorted(bootloaders):
            print(f"\n--- Flashing device {dev} ---")
            try:
                flash_device(bus, dev, filepath, b, num_pages)
            except Exception as e:
                print(f"Device {dev}: FAILED - {e}")
                failed.append(dev)

        print("\n--- Flash Summary ---")
        successful = [d for d in sorted(bootloaders) if d not in failed]
        print(f"Successfully flashed: {successful}")
        if failed:
            print(f"Failed devices: {sorted(failed)}")
            raise SystemExit(f"{len(failed)} device(s) failed to flash")


if __name__ == "__main__":
    tyro.cli(flash)
