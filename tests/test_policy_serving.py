"""Policy serving (websocket + msgpack) loopback + action-chunk broker tests."""

from __future__ import annotations

import threading

import numpy as np
import pytest

pytest.importorskip("msgpack_numpy")
pytest.importorskip("websockets")

from yam_policy import (
    ActionChunkBroker,
    AsyncActionChunkBroker,
    WebsocketClientPolicy,
    WebsocketPolicyServer,
)
from yam_policy.policies.dummy import DummyPolicy

from tests._util import free_port, wait_port

HORIZON = 8


def _serve(mode: str = "hold"):
    port = free_port()
    srv = WebsocketPolicyServer(
        DummyPolicy(action_dim=14, action_horizon=HORIZON, mode=mode),
        host="127.0.0.1",
        port=port,
        metadata={"action_horizon": HORIZON},
    )
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    assert wait_port(port), "policy server did not start"
    return port


def test_loopback_metadata_and_chunking():
    port = _serve("hold")
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    assert client.get_server_metadata()["action_horizon"] == HORIZON

    broker = ActionChunkBroker(client, action_horizon=HORIZON)
    state = np.arange(14, dtype=np.float32)
    obs = {"observation/state": state}

    first = broker.infer(obs)["actions"]
    assert first.shape == (14,)
    assert np.allclose(first, state)  # "hold" repeats the state
    # pull more than one horizon -> re-queries the server without error
    for _ in range(2 * HORIZON + 3):
        assert broker.infer(obs)["actions"].shape == (14,)


def test_async_broker_prefetch():
    port = _serve("zeros")
    client = WebsocketClientPolicy(host="127.0.0.1", port=port)
    broker = AsyncActionChunkBroker(client, action_horizon=HORIZON)
    obs = {"observation/state": np.zeros(14, dtype=np.float32)}
    for _ in range(3 * HORIZON):
        a = broker.infer(obs)["actions"]
        assert a.shape == (14,)
    broker.close()
