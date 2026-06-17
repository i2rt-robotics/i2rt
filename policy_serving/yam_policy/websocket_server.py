"""Websocket policy server — wire-compatible with openpi's WebsocketPolicyServer.

Protocol (per connection):
  1. server sends ``msgpack(metadata)`` immediately on connect
  2. loop: recv ``msgpack(obs)`` -> ``policy.infer(obs)`` -> send ``msgpack(action)``
     (a ``server_timing`` dict with ``infer_ms`` is added to the action)
  3. on exception: send the traceback as a plain string and close

Because the wire format matches openpi exactly, our :class:`WebsocketClientPolicy`
can talk to a real ``openpi`` server and vice-versa.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from typing import Dict, Optional

import msgpack_numpy
import websockets.asyncio.server
from websockets.frames import CloseCode

from .base_policy import BasePolicy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    def __init__(
        self,
        policy: BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Optional[Dict] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        logger.info("Policy server listening on ws://%s:%d", self._host, self._port)
        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port, compression=None, max_size=None
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket) -> None:  # noqa: ANN001
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        prev_total_ms: Optional[float] = None
        while True:
            try:
                t0 = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                t_infer = time.monotonic()
                action = self._policy.infer(obs)
                infer_ms = (time.monotonic() - t_infer) * 1000.0

                action.setdefault("server_timing", {})["infer_ms"] = infer_ms
                if prev_total_ms is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_ms

                await websocket.send(packer.pack(action))
                prev_total_ms = (time.monotonic() - t0) * 1000.0
            except websockets.ConnectionClosed:
                logger.info("Client disconnected")
                break
            except Exception:
                tb = traceback.format_exc()
                logger.error("Inference error:\n%s", tb)
                await websocket.send(tb)  # plain string => client raises RuntimeError
                await websocket.close(code=CloseCode.INTERNAL_ERROR, reason="inference error")
                break
