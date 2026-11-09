import threading
from types import SimpleNamespace

import numpy as np
import pytest
from portal.client import Future

from i2rt.flow_base import flow_base_client


@pytest.mark.parametrize("completed_other_request", [False, True])
def test_blocked_rpc_does_not_block_commands_and_close_stops_publisher(
    monkeypatch: pytest.MonkeyPatch, completed_other_request: bool
) -> None:
    entered = threading.Event()
    close_calls = []
    pending, other = Future([False]), Future([False])
    if completed_other_request:
        other.set_result(None)

    def result(timeout: float | None = None) -> None:
        entered.set()
        pending.result(timeout=timeout)

    def close_socket(**_kwargs: object) -> None:
        assert transport.socket.running
        assert not pending.done()
        transport.socket.running = False
        close_calls.append(True)

    transport = SimpleNamespace(
        socket=SimpleNamespace(running=True, close=close_socket),
        futures={0: pending, 1: other},
        connect=lambda **_kwargs: True,
        set_target_velocity=lambda _command: SimpleNamespace(result=result),
    )
    monkeypatch.setattr(flow_base_client.portal, "Client", lambda _address, **_kwargs: transport)
    client = flow_base_client.FlowBaseClient()
    try:
        assert entered.wait(1.0)
        acquired = client._lock.acquire(timeout=0.2)
        if acquired:
            client._lock.release()
        assert acquired
        client.set_target_velocity(np.ones(3))
        client.close()
        assert not client._thread.is_alive()
        assert not transport.socket.running
    finally:
        client.close()
    assert len(close_calls) == 1
    assert pending.done() and other.done()
    assert not transport.futures


def test_failed_connection_closes_transport_before_starting_publisher(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def connect(timeout: float) -> bool:
        assert timeout > 0
        calls.append("connect")
        return False

    transport = SimpleNamespace(
        connect=connect, futures={}, socket=SimpleNamespace(close=lambda **_kwargs: calls.append("close"))
    )
    monkeypatch.setattr(flow_base_client.portal, "Client", lambda _address, **_kwargs: transport)
    with pytest.raises(TimeoutError, match="Could not connect"):
        flow_base_client.FlowBaseClient()
    assert calls == ["connect", "close"]


@pytest.mark.parametrize("error", [TimeoutError, flow_base_client.portal.Disconnected, KeyError, RuntimeError])
def test_publisher_recovers_transient_errors_and_reports_permanent_failure(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    continued, release, latest = threading.Event(), threading.Event(), threading.Event()
    sends, waits = [], []

    def result(timeout: float) -> None:
        waits.append(True)
        if len(waits) == 1 and error is not KeyError:
            raise error
        continued.set()
        if not release.wait(timeout):
            raise TimeoutError

    def send(command: dict) -> SimpleNamespace:
        sends.append(command)
        if error is KeyError and len(sends) == 1:
            try:
                raise flow_base_client.portal.Disconnected
            except flow_base_client.portal.Disconnected:
                raise KeyError("Request removed during disconnect") from None
        if np.any(command["target_velocity"]):
            latest.set()
        return SimpleNamespace(result=result)

    transport = SimpleNamespace(
        connect=lambda **_kwargs: True,
        set_target_velocity=send,
        socket=SimpleNamespace(close=lambda **_kwargs: release.set()),
        futures={},
        connected=True,
    )
    monkeypatch.setattr(flow_base_client.portal, "Client", lambda _address, **_kwargs: transport)
    client = flow_base_client.FlowBaseClient(with_linear_rail=True)
    try:
        if error is RuntimeError:
            client._thread.join(timeout=1.0)
            assert not client._thread.is_alive()
            with pytest.raises(RuntimeError, match="publisher is stopped"):
                client.set_target_velocity(np.ones(4))
            with pytest.raises(RuntimeError, match="publisher is stopped"):
                client.set_linear_rail_velocity(0.1)
        else:
            assert continued.wait(1.0)
            assert len(sends) == (1 if error is TimeoutError else 2)
            client.set_target_velocity(np.ones(4))
            release.set()
            assert latest.wait(1.0)
    finally:
        release.set()
        client.close()
