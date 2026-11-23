from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, eq=False)
class CameraFrame:
    """One camera frame and the moment it arrived."""

    image: np.ndarray
    """HxWx3 uint8, RGB."""
    timestamp: float
    """``time.time()`` when the frame arrived -- comparable across sensors."""


@runtime_checkable
class Camera(Protocol):
    """A generic camera protocol."""

    @abstractmethod
    def read(self) -> CameraFrame:
        """Get the latest frame.

        Returns:
            CameraFrame: The latest frame and the moment it arrived.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the camera."""
        pass
