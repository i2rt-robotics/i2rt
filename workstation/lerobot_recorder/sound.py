"""Audio cues for the recorder — so the operator gets feedback while watching the
robot, not the screen.

Tone generation is pure/numpy (unit-tested); playback is best-effort via Qt
Multimedia and silently degrades to a no-op if audio isn't available (headless,
missing module, etc.). Distinct pitches per event:

    start   episode recording begins
    success / fail / delete   review or button outcome
    error   camera/robot fault
"""

from __future__ import annotations

import logging
import os
import tempfile
import wave
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# event -> (frequency Hz, duration ms)
CUES: Dict[str, Tuple[int, int]] = {
    "start": (880, 120),
    "success": (1320, 160),
    "fail": (350, 260),
    "delete": (240, 130),
    "error": (160, 400),
}


def tone_pcm(freq: int, ms: int, rate: int = 44100, volume: float = 0.35) -> np.ndarray:
    """A 16-bit mono sine tone with a short fade in/out (avoids clicks). Pure — testable."""
    n = max(int(rate * ms / 1000), 1)
    t = np.arange(n) / rate
    wave_f = np.sin(2 * np.pi * freq * t)
    fade = max(int(rate * 0.005), 1)  # 5 ms ramps
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return (wave_f * env * volume * 32767).astype(np.int16)


def write_wav(path: str, pcm: np.ndarray, rate: int = 44100) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class Cues:
    """Best-effort cue player. ``enabled=False`` or any failure => silent no-op."""

    def __init__(self, enabled: bool = True, rate: int = 44100) -> None:
        self.enabled = enabled
        self._effects: Dict[str, object] = {}
        if not enabled:
            return
        try:
            from PyQt5.QtCore import QUrl
            from PyQt5.QtMultimedia import QSoundEffect

            tmp = tempfile.mkdtemp(prefix="yam_cues_")
            for name, (freq, ms) in CUES.items():
                path = os.path.join(tmp, f"{name}.wav")
                write_wav(path, tone_pcm(freq, ms, rate))
                eff = QSoundEffect()
                eff.setSource(QUrl.fromLocalFile(path))
                self._effects[name] = eff
        except Exception as e:
            logger.info("audio cues disabled (%s)", e)
            self.enabled = False

    def play(self, name: str) -> None:
        if not self.enabled:
            return
        eff = self._effects.get(name)
        if eff is not None:
            try:
                eff.play()
            except Exception:
                pass
