import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, Literal, Union

import can
import click
from can import BusABC, Message
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import BaseModel, Field, field_validator

ALL_DEVICE = 0xFF
"""All devices for broadcasting requests."""


class EncoderConfig(BaseModel):
    """Configuration for encoder settings."""

    adc_freq: int = Field(..., ge=0, le=65535)
    report_freq: int = Field(..., ge=0, le=65535)
    firmware: str = Field(...)

    @field_validator("firmware")
    @classmethod
    def validate_firmware_specifier(cls, v: str) -> str:
        """Validate firmware version specifier."""
        SpecifierSet(v)  # Will raise exception if invalid
        return v


def parse_firmware_version(firmware_input: str) -> Version:
    """Parse firmware version string using packaging.version.Version."""
    if not firmware_input or not firmware_input.strip():
        raise ValueError("Firmware input is empty")

    cleaned_input = SpecifierSet(firmware_input.strip())

    return Version(str(next(iter(cleaned_input)).version))


def check_firmware_version(actual_version: Union[str, Version], expected_specifier: str) -> bool:
    """Check if firmware version matches the expected version specifier."""
    actual = Version(actual_version) if isinstance(actual_version, str) else actual_version
    spec_set = SpecifierSet(expected_specifier)
    return actual in spec_set


class EEPROMField(IntEnum):
    """The EEPROM field."""

    ZPOS_H = 9
    ZPOS_L = 10


@dataclass
class EncoderReport:
    """The encoder report."""

    device: int
    """The device number, uint8."""
    position: int
    """The position in encoder counts, int16."""
    velocity: int
    """The velocity in encoder, int16."""
    inputs: int
    """The inputs, bit by bit,uint8."""


@dataclass
class VersionReply:
    """The version reply."""

    device: int
    """The device number, uint8."""
    major: int
    """The major version, uint8."""
    minor: int
    """The minor version, uint8."""
    patch: int
    """The patch version, uint8."""


class EncoderCanID(IntEnum):
    """The CAN ID for the encoder."""

    REQ = 0x50E
    """The CAN ID for sending a request/reply."""
    REPORT = 0x50F
    """The CAN ID for reporting the encoder status."""
    EVENT = 0x510
    """The CAN ID for reporting the encoder event, e.g. button press."""


class PassiveJointEncoder:
    """The encoder driver."""

    REQ_ZPOS = 0x00
    """Request setting the zero position."""
    REQ_FREQ = 0x01
    """Request setting the report frequency."""
    REQ_REPORT = 0x02
    """Request getting the encoder report."""
    REQ_VERSION = 0x03
    """Request getting the version."""
    REQ_ADC_FREQ = 0x04
    """Request setting the ADC frequency."""
    REQ_DIO_EVENT = 0x05
    """Request toggling the digital IO event report."""
    REQ_READINGS = 0x06
    """Request reading the analog and digital values."""
    REQ_GET_EEPROM = 0x07
    """Request reading EEPROM field by index."""
    REQ_RESTART = 0x0F
    """Restart encoder."""

    def __init__(self, bus: BusABC):
        self.bus = bus
        self.bus.set_filters(
            [
                {"can_id": EncoderCanID.REPORT, "can_mask": 0x7FF},
                {"can_id": EncoderCanID.EVENT, "can_mask": 0x7FF},
                {"can_id": EncoderCanID.REQ, "can_mask": 0x7FF},
            ]
        )

    def reset_zero_position(self, device: int = ALL_DEVICE) -> None:
        """Set current position as the zero position."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        message = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_ZPOS],
            is_extended_id=False,
        )
        self.bus.send(message)

    def set_report_frequency(self, frequency: int, device: int = ALL_DEVICE) -> None:
        """Set the report frequency, 0 for passive mode."""
        assert 0 <= frequency <= 600, "Report frequency value must be between 0 and 600"
        # report frequency must be less than 600 due to the hardware limitation
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        if frequency <= 255:
            message = Message(
                arbitration_id=EncoderCanID.REQ,
                data=[device, self.REQ_FREQ, frequency],
                is_extended_id=False,
            )
        else:
            high_byte = (frequency >> 8) & 0xFF
            low_byte = frequency & 0xFF
            message = Message(
                arbitration_id=EncoderCanID.REQ,
                data=[device, self.REQ_FREQ, high_byte, low_byte],
                is_extended_id=False,
            )
        self.bus.send(message)

    def set_adc_frequency(self, frequency: int, device: int = ALL_DEVICE) -> None:
        """Set the ADC sampling frequency."""
        assert 0 <= frequency <= 601, "ADC frequency value must be between 0 and 601"
        # adc frequency must be less than 601 due to the hardware limitation
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        if frequency <= 255:
            message = Message(
                arbitration_id=EncoderCanID.REQ,
                data=[device, self.REQ_ADC_FREQ, frequency],
                is_extended_id=False,
            )
        else:
            high_byte = (frequency >> 8) & 0xFF
            low_byte = frequency & 0xFF
            message = Message(
                arbitration_id=EncoderCanID.REQ,
                data=[device, self.REQ_ADC_FREQ, high_byte, low_byte],
                is_extended_id=False,
            )
        self.bus.send(message)

    def get_encoder_report(self, device: int = ALL_DEVICE, timeout: float | None = None) -> list[EncoderReport]:
        """Get the encoder report."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        message = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_REPORT],
            is_extended_id=False,
        )
        self.bus.send(message)
        return self.wait_for_report(device, timeout)

    def wait_for(
        self,
        message_id: Literal[EncoderCanID.REPORT, EncoderCanID.EVENT],
        device: int = ALL_DEVICE,
        timeout: float | None = None,
    ) -> list[EncoderReport]:
        """Wait for a report."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        start_time = time.time()
        reports = []
        while True:
            # Adjust timeout for recv based on elapsed time
            remaining_timeout = None
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    break
                remaining_timeout = timeout - elapsed_time

            message = self.bus.recv(timeout=remaining_timeout)
            if message and message.arbitration_id == message_id:
                assert len(message.data) == 6, "Report must be 6 bytes"
                from_device = message.data[0]
                if device not in (from_device, ALL_DEVICE):
                    continue
                position = struct.unpack(">h", message.data[1:3])[0]
                velocity = struct.unpack(">h", message.data[3:5])[0]
                inputs = message.data[5]
                reports.append(EncoderReport(from_device, position, velocity, inputs))
                if device != ALL_DEVICE:
                    break  # Got the specific device report
            elif message is None:
                # This means bus.recv timed out
                break
        return reports

    def wait_for_event(self, device: int = ALL_DEVICE, timeout: float | None = None) -> list[EncoderReport]:
        """Wait for an event."""
        return self.wait_for(EncoderCanID.EVENT, device, timeout)

    def wait_for_report(self, device: int = ALL_DEVICE, timeout: float | None = None) -> list[EncoderReport]:
        """Wait for a report."""
        return self.wait_for(EncoderCanID.REPORT, device, timeout)

    def get_version(self, device: int = ALL_DEVICE, timeout: float | None = None) -> list[VersionReply]:
        """Get the version."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_VERSION],
            is_extended_id=False,
        )
        self.bus.send(req)
        start_time = time.time()
        versions = []
        while True:
            remaining_timeout = None
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    break
                remaining_timeout = timeout - elapsed_time

            message = self.bus.recv(timeout=remaining_timeout)
            if message and message.arbitration_id == EncoderCanID.REQ:
                if len(message.data) != 5:
                    continue
                from_device = message.data[0]

                # If we are looking for a specific device, and the message is from another device, skip it.
                if device not in (from_device, ALL_DEVICE):
                    continue

                cmd = message.data[1]
                if cmd != (self.REQ_VERSION | (1 << 7)):
                    continue

                versions.append(
                    VersionReply(
                        device=from_device,
                        major=message.data[2],
                        minor=message.data[3],
                        patch=message.data[4],
                    )
                )

                # If we are looking for a specific device, we can stop now.
                if device != ALL_DEVICE:
                    break

            if message is None:
                break
        return versions

    def toggle_digital_io_event_report(self, device: int = ALL_DEVICE, io_mask: int = 0) -> None:
        """Toggle the digital IO event report."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= io_mask <= 255, "IO mask must be between 0 and 255"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_DIO_EVENT, io_mask],
            is_extended_id=False,
        )
        self.bus.send(req)

    def get_readings(
        self,
        device: int = ALL_DEVICE,
        analog_index: int = 0,
        timeout: float | None = None,
    ) -> tuple[int, int] | None:
        """Get the readings."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= analog_index <= 3, "Analog index must be between 0 and 3"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_READINGS, analog_index],
            is_extended_id=False,
        )
        self.bus.send(req)
        start_time = time.time()
        while True:
            message = self.bus.recv(timeout=timeout)
            if message and message.arbitration_id == EncoderCanID.REQ:
                if len(message.data) != 5:
                    continue
                from_device = message.data[0]
                if device not in (from_device, ALL_DEVICE):
                    continue
                cmd = message.data[1]
                if cmd != (self.REQ_READINGS | (1 << 7)):
                    continue
                analog_value = struct.unpack(">h", message.data[2:4])[0]
                digital_value = message.data[4]
                return analog_value, digital_value
            if timeout is not None and time.time() - start_time > timeout:
                return None

    def read_eeprom_field(
        self, offset: int, device: int = ALL_DEVICE, timeout: float | None = None
    ) -> dict[int, int] | int | None:
        """Read EEPROM by offset address. Returns byte value or None."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= offset < 29, "Offset must be 0-28"

        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_GET_EEPROM, offset],
            is_extended_id=False,
        )
        self.bus.send(req)

        # Wait for response using existing readings format
        start_time = time.time()
        results = {}
        while True:
            remaining_timeout = None
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    break
                remaining_timeout = timeout - elapsed_time

            message = self.bus.recv(timeout=remaining_timeout)
            if message and message.arbitration_id == EncoderCanID.REQ:
                if len(message.data) != 5:
                    continue
                from_device = message.data[0]
                if device not in (from_device, ALL_DEVICE):
                    continue
                cmd = message.data[1]
                if cmd != (self.REQ_READINGS | (1 << 7)):  # Response uses READINGS format
                    continue
                value = struct.unpack(">h", message.data[2:4])[0] & 0xFF
                if device == ALL_DEVICE:
                    results[from_device] = value
                else:
                    return value

            if message is None:
                break

        if device == ALL_DEVICE:
            return results
        return None

    def read_adc_frequency(self, device: int = ALL_DEVICE, timeout: float | None = None) -> int | None:
        """Read adc frequency from EEPROM"""
        assert 0 <= device <= 255, "Device must be between 0 and 255"

        high_byte = self.read_eeprom_field(27, device, timeout)
        low_byte = self.read_eeprom_field(8, device, timeout)

        if not isinstance(high_byte, int) or not isinstance(low_byte, int):
            return None

        # If high byte is 0xFF (uninitialized), only read low byte as 8-bit value
        if high_byte == 0xFF:
            return low_byte  # Return 8-bit value from low byte

        # Return full 16-bit value
        return (high_byte << 8) | low_byte

    def read_report_frequency(self, device: int = ALL_DEVICE, timeout: float | None = None) -> int | None:
        """Read report frequency from EEPROM"""
        assert 0 <= device <= 255, "Device must be between 0 and 255"

        high_byte = self.read_eeprom_field(28, device, timeout)
        low_byte = self.read_eeprom_field(25, device, timeout)

        if not isinstance(high_byte, int) or not isinstance(low_byte, int):
            return None

        # If high byte is 0xFF (uninitialized), only read low byte as 8-bit value
        if high_byte == 0xFF:
            return low_byte  # Return 8-bit value from low byte

        # Return full 16-bit value
        return (high_byte << 8) | low_byte

    def restart(self, device: int = ALL_DEVICE) -> None:
        """Restart encoder."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_RESTART],
            is_extended_id=False,
        )
        self.bus.send(req)

    @staticmethod
    def validate_encoders(channel: str, expected_config: EncoderConfig) -> Dict[int, Dict[str, Any]]:
        """Validate encoder configuration on a CAN channel."""
        logging.info(f"Validating encoders on {channel}")

        bus = can.interface.Bus(interface="socketcan", channel=channel, bitrate=1000000)
        encoder = PassiveJointEncoder(bus)

        try:
            firmware_versions = encoder.get_version(timeout=1.0)
            logging.info(f"PassiveJointEncoder({channel}) firmware versions: {firmware_versions}")
            if not firmware_versions:
                raise RuntimeError(f"No encoders found on {channel}")

            logging.info(f"Found {len(firmware_versions)} encoders: {[f.device for f in firmware_versions]}")

            all_data = {}
            errors = []

            for firmware_info in firmware_versions:
                device_id = firmware_info.device
                # Read encoder data
                all_data[device_id] = {
                    "version": asdict(firmware_info),
                    "adc_freq": encoder.read_adc_frequency(device=device_id, timeout=0.5),
                    "report_freq": encoder.read_report_frequency(device=device_id, timeout=0.5),
                }

                # Validate firmware version
                actual_version = f"{firmware_info.major}.{firmware_info.minor}.{firmware_info.patch}"
                if not check_firmware_version(actual_version, expected_config.firmware):
                    expected_version = parse_firmware_version(expected_config.firmware)
                    errors.append(f"Encoder {device_id}: Firmware {actual_version} != {expected_version}")

            # Auto-fix frequencies if any encoder needs it (check once after reading all data)
            if any(all_data[device_id]["adc_freq"] != expected_config.adc_freq for device_id in all_data):
                logging.info(f"Auto-fixing ADC frequency to {expected_config.adc_freq}")
                encoder.set_adc_frequency(expected_config.adc_freq)
                time.sleep(0.1)  # Give some time for the setting to take effect

            if any(all_data[device_id]["report_freq"] != expected_config.report_freq for device_id in all_data):
                logging.info(f"Auto-fixing report frequency to {expected_config.report_freq}")
                encoder.set_report_frequency(expected_config.report_freq)

            if errors:
                raise RuntimeError(f"Encoder validation failed for {channel}:\n" + "\n".join(errors))

            logging.info(f"All encoders on {channel} validated ?")
            return all_data

        finally:
            bus.shutdown()


@click.group()
@click.option("--bus", type=str, default="can0", show_default=True)
@click.option("--device", type=int, default=ALL_DEVICE, show_default=True)
@click.option("--bitrate", type=int, default=1000000, show_default=True)
@click.pass_context
def cli(ctx: click.Context, bus: str, device: int, bitrate: int) -> None:
    can_bus = can.interface.Bus(interface="socketcan", channel=bus, bitrate=bitrate)
    encoder = PassiveJointEncoder(can_bus)
    ctx.ensure_object(dict)
    ctx.obj["encoder"] = encoder
    ctx.obj["device"] = device
    ctx.call_on_close(can_bus.shutdown)


@cli.command()
@click.pass_context
def reset_zero_position(ctx: click.Context) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.reset_zero_position(device)


@cli.command()
@click.pass_context
@click.argument("frequency", type=int, default=0)
def set_report_frequency(ctx: click.Context, frequency: int) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.set_report_frequency(frequency, device)


@cli.command()
@click.pass_context
@click.argument("frequency", type=int)
def set_adc_frequency(ctx: click.Context, frequency: int) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.set_adc_frequency(frequency, device)


@cli.command()
@click.option("--timeout", type=float, default=1.0, show_default=True)
@click.pass_context
def get_report(ctx: click.Context, timeout: float | None = None) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    reports: list[EncoderReport] = encoder.get_encoder_report(device, timeout)
    if not reports:
        print("No report")
    else:
        for report in reports:
            print(report)


@cli.command()
@click.option("--timeout", type=float, default=None, show_default=True)
@click.pass_context
def wait_for_event(ctx: click.Context, timeout: float | None = None) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    events: list[EncoderReport] = encoder.wait_for_event(device, timeout)
    if not events:
        print("No event")
    else:
        for event in events:
            print(event)


@cli.command()
@click.option("--timeout", type=float, default=None, show_default=True)
@click.pass_context
def wait_for_report(ctx: click.Context, timeout: float | None = None) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    reports: list[EncoderReport] = encoder.wait_for_report(device, timeout)
    if not reports:
        print("No report")
    else:
        for report in reports:
            print(report)


@cli.command()
@click.option("--timeout", type=float, default=None, show_default=True)
@click.pass_context
def wait_for_event_or_report(ctx: click.Context, timeout: float | None = None) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    executor = ThreadPoolExecutor(max_workers=2)
    event_future = executor.submit(encoder.wait_for_event, device, timeout)
    report_future = executor.submit(encoder.wait_for_report, device, timeout)
    while not event_future.done() or not report_future.done():
        time.sleep(0.1)
    if not event_future.done():
        event_future.cancel()
    if not report_future.done():
        report_future.cancel()


@cli.command()
@click.option("--timeout", type=float, default=1, show_default=True)
@click.pass_context
def get_version(ctx: click.Context, timeout: float) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    version_replies = encoder.get_version(device, timeout)
    if not version_replies:
        print("No version")
    else:
        for version_reply in version_replies:
            print(version_reply)


@cli.command()
@click.argument("io_mask", type=int, required=True)
@click.pass_context
def toggle_digital_io_event(ctx: click.Context, io_mask: int) -> None:
    """Toggle the digital IO events, io_mask is a bit mask of the IOs, 0 to disable all IOs."""
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.toggle_digital_io_event_report(device, io_mask)


@cli.command()
@click.argument("analog_index", type=int, required=True)
@click.pass_context
def get_readings(ctx: click.Context, analog_index: int) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    assert 0 <= analog_index <= 3, "Analog index must be between 0 and 3"
    readings = encoder.get_readings(device, analog_index)
    if readings is None:
        print("No readings")
    else:
        print(f"Analog: 0x{readings[0]:04x}, Digital: 0x{readings[1]:02x}")


@cli.command()
@click.argument("offset", type=int, required=True)
@click.pass_context
def read_eeprom(ctx: click.Context, offset: int) -> None:
    """Read EEPROM field by offset."""
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]

    # EEPROM offset and field name mapping
    EEPROM_FIELDS = {
        0: "magic_h",
        1: "magic_l",
        2: "can_ext",
        3: "can_id_0",
        4: "can_id_1",
        5: "can_id_2",
        6: "can_id_3",
        7: "device",
        8: "adc_freq_l",
        9: "zpos_h",
        10: "zpos_l",
        11: "mpos_h",
        12: "mpos_l",
        13: "mang_h",
        14: "mang_l",
        15: "filters_begin",
        23: "dir",
        24: "threshold_steps",
        25: "report_freq_l",
        26: "dio_report_reverse",
        27: "adc_freq_h",  # ADC frequency high byte (for 16-bit support)
        28: "report_freq_h",  # Report frequency high byte (for 16-bit support)
    }

    field_name = EEPROM_FIELDS.get(offset, "unknown")
    print(f"Reading EEPROM field '{field_name}' at offset {offset}...")
    values = encoder.read_eeprom_field(offset, device, timeout=1.0)

    if values is None:
        print("No response from device")
    elif isinstance(values, dict):
        for dev, val in values.items():
            print(f"Device {dev}: {val} (0x{val:02X})")
    else:
        print(f"Value: {values} (0x{values:02X})")


@cli.command()
@click.pass_context
def read_eeprom_zpos(ctx: click.Context) -> None:
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    zpos_h = encoder.read_eeprom_field(EEPROMField.ZPOS_H, device, timeout=1.0)
    zpos_l = encoder.read_eeprom_field(EEPROMField.ZPOS_L, device, timeout=1.0)
    if zpos_h is None or zpos_l is None:
        print("No response from device")
    elif isinstance(zpos_h, dict) or isinstance(zpos_l, dict):
        print("Unexpected dictionary response for single device")
    else:
        zpos = (zpos_h << 8) | zpos_l
        print(f"ZPOS: 0x{zpos:04x}")


@cli.command()
@click.pass_context
def restart(ctx: click.Context) -> None:
    """Restart the encoder."""
    encoder: PassiveJointEncoder = ctx.obj["encoder"]
    device: int = ctx.obj["device"]
    encoder.restart(device)


if __name__ == "__main__":
    cli()
