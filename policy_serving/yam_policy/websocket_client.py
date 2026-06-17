"""Websocket policy client — wire-compatible with openpi's WebsocketClientPolicy.

Connects to a policy server (ours or a real ``openpi`` server), sends an
observation dict, and returns the action dict. Serialization is msgpack-numpy.
Connection is retried every 5 s until the server is up.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional

import msgpack_numpy
import websockets.sync.client
from websockets.exceptions import ConnectionClosed

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


class WebsocketClientPolicy(BasePolicy):
    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}" if port is None else f"ws://{host}:{port}"
        self._api_key = api_key
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> tuple:
        logger.info("Connecting to policy server at %s ...", self._uri)
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logger.info("Connected. Server metadata: %s", metadata)
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                logger.info("Policy server not up yet, retrying in 5 s ...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            # the server sends a traceback as a plain string on error
            raise RuntimeError(f"Policy server error:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._ws.close()
        except ConnectionClosed:
            pass
