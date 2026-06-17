"""Shared helpers for the sim/mock smoke tests."""

from __future__ import annotations

import socket
import time


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_port(port: int, host: str = "127.0.0.1", timeout: float = 8.0) -> bool:
    """Block until ``host:port`` accepts a TCP connection (so clients don't hit slow retries)."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection((host, port), 0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False
