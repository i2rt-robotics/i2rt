"""Audio-cue tone generation (pure) tests."""

from __future__ import annotations

import numpy as np

from workstation.lerobot_recorder.sound import CUES, tone_pcm


def test_tone_pcm_shape_and_range():
    rate = 44100
    pcm = tone_pcm(440, 100, rate=rate)
    assert pcm.dtype == np.int16
    assert pcm.shape[0] == int(rate * 0.1)
    assert np.abs(pcm).max() <= 32767
    assert np.abs(pcm).max() > 0  # non-silent
    # fade-in means the first sample is ~0
    assert abs(int(pcm[0])) < 2000


def test_all_cues_generate():
    for name, (freq, ms) in CUES.items():
        pcm = tone_pcm(freq, ms)
        assert pcm.size > 0, name
