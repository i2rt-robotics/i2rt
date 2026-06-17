"""Operator view composition (pure numpy) tests."""

from __future__ import annotations

import numpy as np

from workstation.lerobot_recorder.views import compose_agentview, overlay


def test_compose_agentview_insets():
    h, w = 120, 160
    images = {
        "agentview": np.zeros((h, w, 3), np.uint8),
        "wrist_left": np.full((48, 64, 3), 200, np.uint8),
        "wrist_right": np.full((48, 64, 3), 100, np.uint8),
    }
    out = compose_agentview(images, agent_key="agentview", inset_frac=0.33)
    assert out.shape == (h, w, 3)
    # bottom-left and bottom-right corners should now contain inset content (non-zero)
    assert out[h - 1, 0:5].max() > 0
    assert out[h - 1, w - 5 : w].max() > 0
    # top stays the (black) agentview
    assert out[0:5, w // 2].max() == 0


def test_overlay_blend():
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.full((20, 20, 3), 200, np.uint8)  # different size -> resized to a
    blended = overlay(a, b, alpha=0.5)
    assert blended.shape == (10, 10, 3)
    assert np.all(blended == 100)  # 0*0.5 + 200*0.5
    assert overlay(None, b) is b  # degenerate inputs pass through
