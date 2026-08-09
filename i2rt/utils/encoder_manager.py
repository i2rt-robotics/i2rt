"""Encoder driver for ioheart-based passive joint encoders.

Provides ``PassiveJointEncoder`` for live encoder ops (position, velocity,
joystick, GPIO outputs, EEPROM config) over a python-can bus. For firmware
flashing over the same CAN link, see :mod:`i2rt.utils.can_flash`.
"""

import logging
import struct
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Dict, Literal, Optional, Union

import tyro
from can import BusABC, Message
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import BaseModel, Field, field_validator

from i2rt.utils.can_flash import bl_list_connected_devices, encoder_list_devices, get_can_bus

ALL_DEVICE = 0xFF
"""All devices for broadcasting requests."""

_ADC_AUTO_CONFIG_MIN_VERSION = ">=2.4.0"
"""Firmware >= 2.4.0 sets ADC to max speed by default; manual configuration not needed."""


def drain_bus(bus: BusABC) -> int:
    """Non-blocking drain of any frames already buffered on `bus`. Returns frames drained."""
    drained = 0
    while bus.recv(timeout=0) is not None:
        drained += 1
    return drained


def wait_for_quiet_bus(bus: BusABC, quiet_window_s: float = 0.1, timeout_s: float = 3.0) -> int:
    """Block until `bus` is silent for `quiet_window_s`; return frames drained.

    If the bus is still active after `timeout_s`, log a warning and return whatever
    has been drained so far rather than raising — a borderline-jittery station should
    not have its launch killed by this helper itself. The deadline is only checked
    after each receive so a frame arriving right at the boundary still earns one
    full quiet window before we give up.
    """
    deadline = time.time() + timeout_s
    drained = 0
    while True:
        if bus.recv(timeout=quiet_window_s) is None:
            return drained
        drained += 1
        if time.time() >= deadline:
            logging.warning(f"Bus did not quiesce within {timeout_s}s; drained {drained} frames, returning anyway")
            return drained


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
    """EEPROM offsets aligned with ioheart application (main.c eeprom_offset_t)."""

    ZPOS_H = 9
    ZPOS_L = 10
    ADC_FREQ_L = 8
    ADC_FREQ_H = 27
    REPORT_FREQ_L = 25
    REPORT_FREQ_H = 28
    REPORT_JOYSTICK_FREQ_H = 29
    REPORT_JOYSTICK_FREQ_L = 30


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


JOYSTICK_STRUCT_FORMAT = "!B H H H B"
"""Struct format for 0x511: device(u8), analog1(u16), analog2(u16), analog3(u16), inputs(u8)."""


@dataclass
class JoystickReport:
    """Joystick report (0x511). Device in JOYSTICK mode sends this in addition to encoder report (0x50F). Firmware >= v2.3.0."""

    device: int
    """The device number, uint8."""
    analog1: int
    """Analog input 1 (joystick X), uint16."""
    analog2: int
    """Analog input 2 (joystick Y), uint16."""
    analog3: int
    """Analog input 3 (reserved), uint16."""
    inputs: int
    """Digital inputs, bit by bit, uint8."""

    @staticmethod
    def parse(data: bytes) -> "JoystickReport | None":
        """Parse raw CAN data (8 bytes) into a JoystickReport, or None if size mismatches."""
        if len(data) != struct.calcsize(JOYSTICK_STRUCT_FORMAT):
            return None
        device, analog1, analog2, analog3, inputs = struct.unpack(JOYSTICK_STRUCT_FORMAT, data)
        return JoystickReport(device=device, analog1=analog1, analog2=analog2, analog3=analog3, inputs=inputs)


class EncoderCanID(IntEnum):
    """The CAN ID for the encoder."""

    REQ = 0x50E
    """The CAN ID for sending a request/reply."""
    REPORT = 0x50F
    """The CAN ID for reporting the encoder status."""
    EVENT = 0x510
    """The CAN ID for reporting the encoder event, e.g. button press."""
    JOYSTICK = 0x511
    """The CAN ID for joystick report; device set to JOYSTICK sends this in addition to 0x50F (firmware >= v2.3.0)."""


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
    REQ_GPIO_OUTPUT = 0x08
    """Request configuring or driving GPIO outputs."""
    REQ_RESTART = 0x0F
    """Restart encoder."""
    REQ_CONFIG = 0x10
    """Request writing EEPROM config (ioheart). [device, REQ_CONFIG, offset, ...values]."""
    REQ_REPORT_JOYSTICK = 0x11
    """Request getting the joystick report."""

    REPORT_MODE_ENCODER = 0
    """Encoder only: device sends 0x50F/0x510 (default)."""
    REPORT_MODE_JOYSTICK = 1
    """Encoder (0x50F) is always sent; this device also sends joystick report (0x511). Firmware >= v2.3.0."""

    def __init__(self, bus: BusABC):
        self.bus = bus
        self.bus.set_filters(
            [
                {"can_id": EncoderCanID.REPORT, "can_mask": 0x7FF},
                {"can_id": EncoderCanID.EVENT, "can_mask": 0x7FF},
                {"can_id": EncoderCanID.JOYSTICK, "can_mask": 0x7FF},
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

    def set_report_mode(self, mode: int, device: int = ALL_DEVICE) -> None:
        """Set report mode via EEPROM offsets 29-30 (ioheart report_joystick_freq). JOYSTICK=1 enables 0x511."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert mode in (self.REPORT_MODE_ENCODER, self.REPORT_MODE_JOYSTICK), (
            "mode must be REPORT_MODE_ENCODER or REPORT_MODE_JOYSTICK"
        )
        high_byte = 0
        low_byte = 1 if mode == self.REPORT_MODE_JOYSTICK else 0
        self.set_eeprom_configs(EEPROMField.REPORT_JOYSTICK_FREQ_H, [high_byte, low_byte], device)
        time.sleep(0.1)

    def get_report_mode(
        self, device: int = ALL_DEVICE, timeout: Optional[float] = None
    ) -> Optional[Union[int, Dict[int, int]]]:
        """Report mode from EEPROM 29-30 (report_joystick_freq). 0=encoder only, 1=joystick (0x511) also."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        if device == ALL_DEVICE:
            versions = self.get_version(timeout=timeout or 1.0)
            if not versions:
                return None
            result: Dict[int, int] = {}
            for ver in versions:
                freq = self.read_report_joystick_frequency(ver.device, timeout)
                if freq is not None:
                    result[ver.device] = 1 if freq > 0 else 0
            return result if result else None
        freq = self.read_report_joystick_frequency(device, timeout)
        if freq is None:
            return None
        return 1 if freq > 0 else 0

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

    def wait_for_event_or_report(
        self, device: int = ALL_DEVICE, timeout: float | None = None
    ) -> tuple[EncoderCanID, EncoderReport] | None:
        """Return the first EVENT (0x510) or REPORT (0x50F) frame, whichever arrives first.

        Uses a single ``recv`` loop so the two frame types can't race on separate threads sharing
        one socket. Returns ``(arbitration_id, report)`` or ``None`` on timeout.
        """
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        start_time = time.time()
        while True:
            remaining_timeout = None
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    return None
                remaining_timeout = timeout - elapsed_time

            message = self.bus.recv(timeout=remaining_timeout)
            if message is None:
                return None
            if message.arbitration_id not in (EncoderCanID.REPORT, EncoderCanID.EVENT):
                continue
            if len(message.data) != 6:
                continue
            from_device = message.data[0]
            if device not in (from_device, ALL_DEVICE):
                continue
            position = struct.unpack(">h", message.data[1:3])[0]
            velocity = struct.unpack(">h", message.data[3:5])[0]
            inputs = message.data[5]
            return EncoderCanID(message.arbitration_id), EncoderReport(from_device, position, velocity, inputs)

    def wait_for_joystick(self, device: int = ALL_DEVICE, timeout: float | None = None) -> list[JoystickReport]:
        """Wait for a joystick report (0x511). Device set to JOYSTICK also sends 0x50F. Firmware >= v2.3.0."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        if device == ALL_DEVICE and timeout is None:
            # A broadcast wait never breaks on a single device, so without a finite timeout it
            # collects forever on a streaming bus. Require an explicit deadline.
            raise ValueError("wait_for_joystick(ALL_DEVICE) requires a finite timeout")
        start_time = time.time()
        reports: list[JoystickReport] = []
        while True:
            remaining_timeout = None
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    break
                remaining_timeout = timeout - elapsed_time

            message = self.bus.recv(timeout=remaining_timeout)
            if message and message.arbitration_id == EncoderCanID.JOYSTICK:
                report = JoystickReport.parse(bytes(message.data))
                if report is None:
                    continue
                if device not in (report.device, ALL_DEVICE):
                    continue
                reports.append(report)
                if device != ALL_DEVICE:
                    break
            elif message is None:
                break
        return reports

    def get_joystick_report(self, device: int = ALL_DEVICE, timeout: float | None = None) -> list[JoystickReport]:
        """Actively request joystick report (sends REQ_REPORT_JOYSTICK, then waits)."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_REPORT_JOYSTICK],
            is_extended_id=False,
        )
        self.bus.send(req)
        return self.wait_for_joystick(device, timeout)

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

    def set_gpio_output_value(self, value: int, device: int = ALL_DEVICE) -> None:
        """Set DO0-DO3 using a 4-bit mask: bit0->DO0, bit1->DO1, bit2->DO2, bit3->DO3.

        This function forces DO1/DO2 into GPIO output mode, disabling UART.
        """
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= value <= 0x0F, "Value must be between 0 and 0x0F"
        message = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_GPIO_OUTPUT, 1, value],
            is_extended_id=False,
        )
        self.bus.send(message)

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
        """Read EEPROM by offset address. Returns byte value or None. ioheart replies with GET_EEPROM (0x87)."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= offset <= EEPROMField.REPORT_JOYSTICK_FREQ_L, "Offset must be 0-30"

        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_GET_EEPROM, offset],
            is_extended_id=False,
        )
        self.bus.send(req)

        start_time = time.time()
        results: Dict[int, int] = {}
        reply_cmd = self.REQ_GET_EEPROM | (1 << 7)
        while True:
            remaining_timeout = None
            if timeout is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    break
                remaining_timeout = timeout - elapsed_time

            message = self.bus.recv(timeout=remaining_timeout)
            if message and message.arbitration_id == EncoderCanID.REQ:
                if len(message.data) < 3:
                    continue
                from_device = message.data[0]
                if device not in (from_device, ALL_DEVICE):
                    continue
                if message.data[1] != reply_cmd:
                    continue
                value = message.data[2] & 0xFF
                if device == ALL_DEVICE:
                    results[from_device] = value
                else:
                    return value

            if message is None:
                break

        if device == ALL_DEVICE:
            return results
        return None

    def set_eeprom_configs(self, offset: int, values: list[int], device: int = ALL_DEVICE) -> None:
        """Set EEPROM configs."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= offset <= EEPROMField.REPORT_JOYSTICK_FREQ_L, "Offset must be 0-30"
        req = Message(
            arbitration_id=EncoderCanID.REQ,
            data=[device, self.REQ_CONFIG, offset, *values],
            is_extended_id=False,
        )
        self.bus.send(req)

    def read_adc_frequency(self, device: int = ALL_DEVICE, timeout: float | None = None) -> int | None:
        """Read adc frequency from EEPROM"""
        assert 0 <= device <= 255, "Device must be between 0 and 255"

        high_byte = self.read_eeprom_field(EEPROMField.ADC_FREQ_H, device, timeout)
        low_byte = self.read_eeprom_field(EEPROMField.ADC_FREQ_L, device, timeout)

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

        high_byte = self.read_eeprom_field(EEPROMField.REPORT_FREQ_H, device, timeout)
        low_byte = self.read_eeprom_field(EEPROMField.REPORT_FREQ_L, device, timeout)

        if not isinstance(high_byte, int) or not isinstance(low_byte, int):
            return None

        # If high byte is 0xFF (uninitialized), only read low byte as 8-bit value
        if high_byte == 0xFF:
            return low_byte  # Return 8-bit value from low byte

        # Return full 16-bit value
        return (high_byte << 8) | low_byte

    def _read_freq_eeprom_all(self, high_offset: int, low_offset: int, timeout: float | None) -> dict[int, int | None]:
        high_bytes = self.read_eeprom_field(high_offset, ALL_DEVICE, timeout)
        # Replies carry no offset, so a late high-offset reply could otherwise be mis-attributed to
        # the low-offset read below. Drain any stragglers before issuing the second broadcast read.
        drain_bus(self.bus)
        low_bytes = self.read_eeprom_field(low_offset, ALL_DEVICE, timeout)
        assert isinstance(high_bytes, dict) and isinstance(low_bytes, dict)
        result: dict[int, int | None] = {}
        for dev in set(high_bytes) | set(low_bytes):
            h, lo = high_bytes.get(dev), low_bytes.get(dev)
            result[dev] = (
                None if not isinstance(h, int) or not isinstance(lo, int) else (lo if h == 0xFF else (h << 8) | lo)
            )
        return result

    def read_adc_frequency_all(self, timeout: float | None = None) -> dict[int, int | None]:
        """Broadcast ADC-freq EEPROM read for all devices at once.

        Returns {device_id: value} where value is None if the device did not respond.
        Much faster than reading per-device: 2 broadcast reads instead of N*2 reads.
        """
        return self._read_freq_eeprom_all(EEPROMField.ADC_FREQ_H, EEPROMField.ADC_FREQ_L, timeout)

    def read_report_frequency_all(self, timeout: float | None = None) -> dict[int, int | None]:
        """Broadcast report-freq EEPROM read for all devices at once.

        Returns {device_id: value} where value is None if the device did not respond.
        Much faster than reading per-device: 2 broadcast reads instead of N*2 reads.
        """
        return self._read_freq_eeprom_all(EEPROMField.REPORT_FREQ_H, EEPROMField.REPORT_FREQ_L, timeout)

    def read_report_joystick_frequency(self, device: int = ALL_DEVICE, timeout: float | None = None) -> int | None:
        """Read report joystick frequency from EEPROM (offsets 29-30)."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        high_byte = self.read_eeprom_field(EEPROMField.REPORT_JOYSTICK_FREQ_H, device, timeout)
        low_byte = self.read_eeprom_field(EEPROMField.REPORT_JOYSTICK_FREQ_L, device, timeout)
        if not isinstance(high_byte, int) or not isinstance(low_byte, int):
            return None
        return (high_byte << 8) | low_byte

    def set_report_joystick_frequency(self, frequency: int, device: int = ALL_DEVICE) -> None:
        """Set report joystick frequency."""
        assert 0 <= device <= 255, "Device must be between 0 and 255"
        assert 0 <= frequency <= 65535, "Frequency must be between 0 and 65535"
        self.set_eeprom_configs(
            EEPROMField.REPORT_JOYSTICK_FREQ_H,
            [frequency >> 8, frequency & 0xFF],
            device,
        )

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
        """Validate encoder configuration on a CAN channel.

        After a freq change to 0 (passive mode), this function makes a best-effort
        wait for the channel to fall silent on its local CAN socket. If the timeout
        elapses we log a warning and return anyway so a borderline station is not
        blocked from launching. SocketCAN delivers a separate copy of every frame
        to each socket on the channel — callers holding their own bus must still
        drain it after this returns.

        For nonzero target frequencies the encoder keeps streaming, so the quiesce
        step is skipped.
        """
        logging.info(f"Validating encoders on {channel}")

        bus = get_can_bus(channel)
        encoder = PassiveJointEncoder(bus)

        try:
            firmware_versions = encoder.get_version(timeout=1.0)
            if not firmware_versions:
                raise RuntimeError(f"No encoders found on {channel}")

            encoder_summary = ", ".join(f"dev{f.device}:v{f.major}.{f.minor}.{f.patch}" for f in firmware_versions)
            logging.info(
                f"PassiveJointEncoder({channel}): found {len(firmware_versions)} encoders [{encoder_summary}]"
            )

            # Devices with firmware < 2.4.0 need manual ADC configuration.
            # Firmware >= 2.4.0 sets ADC to max speed by default, so we skip read/fix for those.
            devices_needing_adc_fix = [
                fw.device
                for fw in firmware_versions
                if not check_firmware_version(f"{fw.major}.{fw.minor}.{fw.patch}", _ADC_AUTO_CONFIG_MIN_VERSION)
            ]

            # Broadcast EEPROM reads — 2 requests instead of N*2, much faster.
            # Only read ADC frequency if there are legacy firmware devices that need it.
            adc_freq_all = encoder.read_adc_frequency_all(timeout=0.5) if devices_needing_adc_fix else {}
            report_freq_all = encoder.read_report_frequency_all(timeout=0.5)

            all_data = {}
            errors = []

            for firmware_info in firmware_versions:
                device_id = firmware_info.device
                actual_version = f"{firmware_info.major}.{firmware_info.minor}.{firmware_info.patch}"

                # get_report_mode reads joystick EEPROM fields (offsets 29-30) which require firmware >= 2.4.0.
                # Older firmware does not respond to these reads, causing 0.5s timeout per device.
                report_mode = (
                    encoder.get_report_mode(device=device_id, timeout=0.5)
                    if check_firmware_version(actual_version, _ADC_AUTO_CONFIG_MIN_VERSION)
                    else None
                )

                all_data[device_id] = {
                    "version": asdict(firmware_info),
                    "adc_freq": adc_freq_all.get(device_id),
                    "report_freq": report_freq_all.get(device_id),
                    "report_mode": report_mode,
                }

                # Validate firmware version
                if not check_firmware_version(actual_version, expected_config.firmware):
                    expected_version = parse_firmware_version(expected_config.firmware)
                    errors.append(f"Encoder {device_id}: Firmware {actual_version} != {expected_version}")

            # Auto-fix ADC only for legacy firmware devices (firmware < 2.4.0) that actually mismatch.
            # Write per-device so fw >= 2.4.0 devices (which auto-configure ADC) are never touched.
            devices_to_adc_fix = [
                dev for dev in devices_needing_adc_fix if all_data[dev]["adc_freq"] != expected_config.adc_freq
            ]
            if devices_to_adc_fix:
                logging.info(
                    f"Auto-fixing ADC frequency to {expected_config.adc_freq} "
                    f"for legacy firmware devices: {devices_to_adc_fix}"
                )
                for dev in devices_to_adc_fix:
                    encoder.set_adc_frequency(expected_config.adc_freq, device=dev)
                time.sleep(0.1)  # Give some time for the setting to take effect

            freq_mismatch = any(
                all_data[device_id]["report_freq"] != expected_config.report_freq for device_id in all_data
            )
            if freq_mismatch:
                logging.info(f"Auto-fixing report frequency to {expected_config.report_freq}")
                encoder.set_report_frequency(expected_config.report_freq)

            if errors:
                raise RuntimeError(f"Encoder validation failed for {channel}:\n" + "\n".join(errors))

            # set_report_frequency is fire-and-forget; let the firmware EEPROM write commit before
            # the bus is torn down. A zero target goes passive, so we confirm by waiting for the bus
            # to fall silent; a nonzero target keeps streaming and would never quiesce, so fall back
            # to a fixed settle delay (otherwise control reaches bus.shutdown() with no commit window).
            if freq_mismatch:
                if expected_config.report_freq == 0:
                    drained = wait_for_quiet_bus(bus)
                    if drained:
                        logging.info(f"Drained {drained} frames during quiesce wait on {channel}")
                else:
                    time.sleep(0.1)

            logging.info(f"All encoders on {channel} validated")
            return all_data

        finally:
            bus.shutdown()


@contextmanager
def _open_encoder(bus: str, bitrate: int) -> Iterator[PassiveJointEncoder]:
    """Open a SocketCAN bus on ``bus`` and yield a ``PassiveJointEncoder`` bound to it."""
    can_bus = get_can_bus(bus, bitrate)
    try:
        yield PassiveJointEncoder(can_bus)
    finally:
        can_bus.shutdown()


def list_devices(bus: str = "can0", bitrate: int = 1000000) -> None:
    """List connected devices (both encoders and bootloader-mode devices) on the CAN bus."""
    print("Searching for connected devices...")
    can_bus = get_can_bus(bus, bitrate)
    try:
        try:
            bootloaders = bl_list_connected_devices(can_bus)
            print(f"Detected the bootloader devices: {bootloaders}")
        except Exception:
            pass
        try:
            encoders = encoder_list_devices(can_bus)
            print(f"Detected the encoder devices: {encoders}")
        except Exception:
            pass
    finally:
        can_bus.shutdown()


def reset_zero_position(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    restart: bool = False,
) -> None:
    """Set current position as zero position; optionally restart afterwards."""
    with _open_encoder(bus, bitrate) as encoder:
        encoder.reset_zero_position(device)
        if restart:
            time.sleep(1)
            encoder.restart(device)
    print(f"Reset zero position, restarted: {restart}")


def set_report_frequency(
    frequency: int = 0,
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    restart: bool = False,
) -> None:
    """Set the report frequency (0 for passive mode)."""
    with _open_encoder(bus, bitrate) as encoder:
        encoder.set_report_frequency(frequency, device)
        if restart:
            time.sleep(1)
            encoder.restart(device)
    print(f"Report frequency set to {frequency}, restarted: {restart}")


def set_adc_frequency(
    frequency: int,
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Set the ADC sampling frequency."""
    with _open_encoder(bus, bitrate) as encoder:
        encoder.set_adc_frequency(frequency, device)


def set_report_mode(
    mode: Literal["JOYSTICK", "ENCODER"],
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Set report mode. Encoder (0x50F) always sent; JOYSTICK adds 0x511 for device. Firmware >= v2.3.0."""
    mode_val = (
        PassiveJointEncoder.REPORT_MODE_JOYSTICK if mode == "JOYSTICK" else PassiveJointEncoder.REPORT_MODE_ENCODER
    )
    with _open_encoder(bus, bitrate) as encoder:
        encoder.set_report_mode(mode_val, device)


def get_report_mode(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    timeout: float = 1.0,
) -> None:
    """Read report mode from device (EEPROM offset 29). 0=encoder, 1=joystick (0x511)."""
    with _open_encoder(bus, bitrate) as encoder:
        raw = encoder.get_report_mode(device, timeout)
    if raw is None:
        print("No response from device")
        return
    if isinstance(raw, dict):
        for dev, val in raw.items():
            mode_str = "JOYSTICK" if val == PassiveJointEncoder.REPORT_MODE_JOYSTICK else "ENCODER"
            print(f"Device {dev}: {mode_str} ({val})")
    else:
        mode_str = "JOYSTICK" if raw == PassiveJointEncoder.REPORT_MODE_JOYSTICK else "ENCODER"
        print(f"Report mode: {mode_str} ({raw})")


def get_report(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    timeout: float = 1.0,
) -> None:
    """Get encoder report."""
    with _open_encoder(bus, bitrate) as encoder:
        reports: list[EncoderReport] = encoder.get_encoder_report(device, timeout)
    if not reports:
        print("No report")
    else:
        for report in reports:
            print(report)


def wait_for_event(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    timeout: Optional[float] = None,
) -> None:
    """Wait for an event."""
    with _open_encoder(bus, bitrate) as encoder:
        events: list[EncoderReport] = encoder.wait_for_event(device, timeout)
    if not events:
        print("No event")
    else:
        for event in events:
            print(event)


def wait_for_report(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    timeout: Optional[float] = None,
) -> None:
    """Wait for a report."""
    with _open_encoder(bus, bitrate) as encoder:
        reports: list[EncoderReport] = encoder.wait_for_report(device, timeout)
    if not reports:
        print("No report")
    else:
        for report in reports:
            print(report)


def wait_for_event_or_report(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    timeout: Optional[float] = None,
) -> None:
    """Wait for an event or a report (whichever arrives first)."""
    with _open_encoder(bus, bitrate) as encoder:
        result = encoder.wait_for_event_or_report(device, timeout)
    if result is None:
        print("No event or report")
    else:
        kind, report = result
        print(f"{kind.name}: {report}")


def get_version(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    timeout: float = 1.0,
) -> None:
    """Get the firmware version."""
    with _open_encoder(bus, bitrate) as encoder:
        version_replies = encoder.get_version(device, timeout)
    if not version_replies:
        print("No version")
    else:
        for version_reply in version_replies:
            print(version_reply)


def toggle_digital_io_event(
    io_mask: int,
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Toggle digital IO events. ``io_mask`` is a bit mask of the IOs; 0 disables all."""
    with _open_encoder(bus, bitrate) as encoder:
        encoder.toggle_digital_io_event_report(device, io_mask)


def set_gpio_outputs(
    value: int,
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Set DO0-DO3 with a 4-bit mask (0bDCBA).

    Bits map as D->DO3, C->DO2, B->DO1, A->DO0. 0b1111 (15) drives all high,
    0b0000 (0) drives all low. Control of DO1/DO2 triggers an auto-enable.
    """
    assert 0 <= value <= 15, "Value must be between 0 and 15"
    with _open_encoder(bus, bitrate) as encoder:
        encoder.set_gpio_output_value(value, device)


def get_readings(
    analog_index: int,
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Get analog (``analog_index`` 0-3) and digital input readings."""
    assert 0 <= analog_index <= 3, "Analog index must be between 0 and 3"
    with _open_encoder(bus, bitrate) as encoder:
        readings = encoder.get_readings(device, analog_index)
    if readings is None:
        print("No readings")
    else:
        print(f"Analog: 0x{readings[0]:04x}, Digital: 0x{readings[1]:02x}")


_EEPROM_FIELDS: Dict[int, str] = {
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
    27: "adc_freq_h",
    28: "report_freq_h",
    29: "report_joystick_freq_h",
    30: "report_joystick_freq_l",
}


def read_eeprom(
    offset: int,
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Read EEPROM field by offset."""
    field_name = _EEPROM_FIELDS.get(offset, "unknown")
    print(f"Reading EEPROM field '{field_name}' at offset {offset}...")
    with _open_encoder(bus, bitrate) as encoder:
        values = encoder.read_eeprom_field(offset, device, timeout=1.0)
    if values is None:
        print("No response from device")
    elif isinstance(values, dict):
        for dev, val in values.items():
            print(f"Device {dev}: {val} (0x{val:02X})")
    else:
        print(f"Value: {values} (0x{values:02X})")


def set_eeprom(
    offset: int,
    value: int,
    device: int,
    bus: str = "can0",
    bitrate: int = 1000000,
    restart: bool = False,
) -> None:
    """Set an EEPROM byte at ``offset`` to ``value`` on ``device``; optionally restart afterwards.

    ``device`` is required (no ``ALL_DEVICE`` default): EEPROM offsets include CAN/device IDs, so a
    silent broadcast write could mass-reconfigure the whole chain. Pass ``--device 255`` to broadcast
    deliberately.
    """
    with _open_encoder(bus, bitrate) as encoder:
        encoder.set_eeprom_configs(offset, [value], device)
        if restart:
            time.sleep(1)
            encoder.restart(device)
    print(f"EEPROM {offset} set to {value}, restarted: {restart}")


def read_eeprom_zpos(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Read the stored ZPOS (zero-position) from EEPROM."""
    with _open_encoder(bus, bitrate) as encoder:
        zpos_h = encoder.read_eeprom_field(EEPROMField.ZPOS_H, device, timeout=1.0)
        zpos_l = encoder.read_eeprom_field(EEPROMField.ZPOS_L, device, timeout=1.0)
    if zpos_h is None or zpos_l is None:
        print("No response from device")
    elif isinstance(zpos_h, dict) or isinstance(zpos_l, dict):
        print("Unexpected dictionary response for single device")
    else:
        zpos = (zpos_h << 8) | zpos_l
        print(f"ZPOS: 0x{zpos:04x}")


def set_report_joystick_frequency(
    frequency: int,
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    restart: bool = False,
) -> None:
    """Set report joystick frequency."""
    assert 0 <= frequency <= 65535, "Frequency must be between 0 and 65535"
    with _open_encoder(bus, bitrate) as encoder:
        encoder.set_report_joystick_frequency(frequency, device)
        if restart:
            time.sleep(1)
            encoder.restart(device)
    print(f"Report joystick frequency set to {frequency}, restarted: {restart}")


def get_report_joystick_frequency(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Get report joystick frequency."""
    with _open_encoder(bus, bitrate) as encoder:
        frequency = encoder.read_report_joystick_frequency(device)
    if frequency is None:
        print("No response from device")
    else:
        print(f"Report joystick frequency: {frequency}")


def get_report_joystick(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
    timeout: float = 1.0,
) -> None:
    """Actively request a joystick report."""
    with _open_encoder(bus, bitrate) as encoder:
        reports: list[JoystickReport] = encoder.get_joystick_report(device, timeout)
    if not reports:
        print("No response from device")
    else:
        for report in reports:
            print(report)


def restart(
    bus: str = "can0",
    device: int = ALL_DEVICE,
    bitrate: int = 1000000,
) -> None:
    """Restart the encoder."""
    with _open_encoder(bus, bitrate) as encoder:
        encoder.restart(device)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {
            "list-devices": list_devices,
            "reset-zero-position": reset_zero_position,
            "set-report-frequency": set_report_frequency,
            "set-adc-frequency": set_adc_frequency,
            "set-report-mode": set_report_mode,
            "get-report-mode": get_report_mode,
            "get-report": get_report,
            "wait-for-event": wait_for_event,
            "wait-for-report": wait_for_report,
            "wait-for-event-or-report": wait_for_event_or_report,
            "get-version": get_version,
            "toggle-digital-io-event": toggle_digital_io_event,
            "set-gpio-outputs": set_gpio_outputs,
            "get-readings": get_readings,
            "read-eeprom": read_eeprom,
            "set-eeprom": set_eeprom,
            "read-eeprom-zpos": read_eeprom_zpos,
            "set-report-joystick-frequency": set_report_joystick_frequency,
            "get-report-joystick-frequency": get_report_joystick_frequency,
            "get-report-joystick": get_report_joystick,
            "restart": restart,
        }
    )
