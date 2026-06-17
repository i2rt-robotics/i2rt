"""Image composition helpers for the operator views (pure numpy, no cv2).

* :func:`compose_agentview` — agentview as the base with the wrist views inset in
  the bottom corners, giving the teleoperator one rich frame.
* :func:`overlay` — alpha-blend two frames (e.g. an episode's first frame with the
  live camera) so the operator can match object placement before a replay.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _resize(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbor resize to (h, w) — dependency-free, fine for previews."""
    if img.shape[0] == h and img.shape[1] == w:
        return img
    yi = np.linspace(0, img.shape[0] - 1, h).astype(int)
    xi = np.linspace(0, img.shape[1] - 1, w).astype(int)
    return img[yi][:, xi]


def compose_agentview(
    images: Dict[str, np.ndarray],
    agent_key: str = "agentview",
    wrist_keys: Tuple[str, ...] = ("wrist_left", "wrist_right"),
    inset_frac: float = 0.33,
) -> Optional[np.ndarray]:
    """Agentview base with each wrist view inset into a bottom corner."""
    base = images.get(agent_key)
    if base is None:
        base = next(iter(images.values()), None)
    if base is None:
        return None
    out = np.ascontiguousarray(base).copy()
    h, w = out.shape[:2]
    ih, iw = max(int(h * inset_frac), 1), max(int(w * inset_frac), 1)
    corners = [(h - ih, 0), (h - ih, w - iw)]  # bottom-left, bottom-right
    for key, (y0, x0) in zip(wrist_keys, corners, strict=False):
        wim = images.get(key)
        if wim is None:
            continue
        small = _resize(np.ascontiguousarray(wim), ih, iw)
        out[y0 : y0 + ih, x0 : x0 + iw] = small
        out[y0 : y0 + ih, x0 : x0 + 2] = 255  # thin border so the inset reads clearly
        out[y0 : y0 + 2, x0 : x0 + iw] = 255
    return out


def overlay(base: Optional[np.ndarray], other: Optional[np.ndarray], alpha: float = 0.5) -> Optional[np.ndarray]:
    """Alpha-blend ``other`` (resized to ``base``) over ``base``."""
    if base is None:
        return other
    if other is None:
        return base
    o = _resize(np.ascontiguousarray(other), base.shape[0], base.shape[1])
    return (base.astype(np.float32) * (1.0 - alpha) + o.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
