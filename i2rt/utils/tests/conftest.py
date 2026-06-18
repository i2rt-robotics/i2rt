"""Shared fixtures for the usb_gpio_driver tests.

The tests run with no hardware: ``serial.Serial`` is monkeypatched with an
in-memory :class:`FakeSerial` whose reads are produced by a ``responder``
callback fed each written frame. A default responder synthesizes protocol
replies from a ``{channel: bool}`` level dict so backend-level tests work
end-to-end; individual tests can install their own responder for boot-frame,
error, and dropped-write scenarios.
"""

from collections.abc import Callable
from typing import Any, Optional

import pytest

Responder = Callable[[bytes], bytes]


class FakeSerial:
    """In-memory stand-in for ``serial.Serial`` driven by a ``responder``.

    ``write(data)`` records the frame and appends ``responder(data)`` to the
    read buffer; ``read(n)`` returns up to ``n`` buffered bytes (``b""`` models a
    timeout); ``reset_input_buffer`` clears it -- so each request/reply
    round-trip is modeled faithfully.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: Optional[float] = None,
        responder: Optional[Responder] = None,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = True
        self.writes: list[bytes] = []
        self._in = bytearray()
        self.responder = responder

    def write(self, data: bytes) -> int:
        data = bytes(data)
        self.writes.append(data)
        if self.responder is not None:
            self._in += self.responder(data)
        return len(data)

    def read(self, n: int) -> bytes:
        n = max(0, int(n))
        out = bytes(self._in[:n])
        del self._in[:n]
        return out

    @property
    def in_waiting(self) -> int:
        return len(self._in)

    def reset_input_buffer(self) -> None:
        self._in.clear()

    def close(self) -> None:
        self.is_open = False


def pin_responder(levels: dict[int, bool], apply_writes: bool = True) -> Responder:
    """Build a responder that synthesizes replies from a ``{channel: bool}`` dict.

    ``apply_writes=False`` makes ``set_pins`` (0x3A) a no-op on ``levels`` (the
    0x2A echo is still returned) so a later read won't reflect the write.
    """

    def respond(data: bytes) -> bytes:
        head = data[0]
        if head == 0x3A:  # SET_PINS: (channel, level) pairs; board echoes 0x2A + the same pairs
            if apply_writes:
                for i in range(1, len(data), 2):
                    levels[data[i]] = bool(data[i + 1])
            return bytes((0x2A,)) + data[1:]
        if head == 0x3F:  # READ_PIN -> 0x2F channel level
            ch = data[1]
            return bytes((0x2F, ch, int(levels.get(ch, False))))
        if head == 0x5B:  # READ_INPUTS -> 0x4B start end pull <levels...>
            start, end, pull = data[1], data[2], data[3]
            body = bytes(int(levels.get(start + i, False)) for i in range(end - start + 1))
            return bytes((0x4B, start, end, pull)) + body
        return b""

    return respond


@pytest.fixture
def fake_serial(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install :class:`FakeSerial` as ``serial.Serial``.

    Returns a mutable ``state`` dict: set ``state["levels"]`` (shared
    ``{channel: bool}``) and/or ``state["responder"]`` *before* the port opens.
    ``state["pin_responder"]`` exposes the default-responder builder.
    """
    state: dict[str, Any] = {"levels": {}, "responder": None, "instances": [], "pin_responder": pin_responder}

    def factory(port: str, baudrate: int = 115200, timeout: Optional[float] = None) -> FakeSerial:
        responder = state["responder"] or pin_responder(state["levels"])
        fs = FakeSerial(port, baudrate, timeout, responder=responder)
        state["instances"].append(fs)
        return fs

    monkeypatch.setattr("serial.Serial", factory)
    return state
