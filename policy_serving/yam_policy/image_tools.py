"""Image helpers mirroring ``openpi_client.image_tools``.

Resize-with-pad (preserve aspect ratio, pad to square) and uint8 conversion, so
the client side preprocesses camera frames exactly the way openpi expects
(typically 224x224 uint8). Uses Pillow to avoid heavy deps.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a float image in [0, 1] to uint8 [0, 255]; pass uint8 through."""
    if np.issubdtype(img.dtype, np.floating):
        img = (255.0 * img).clip(0, 255).astype(np.uint8)
    return img.astype(np.uint8)


def resize_with_pad(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize ``image`` (HxWx3) into ``height x width`` keeping aspect ratio.

    The image is scaled to fit and zero-padded (centered) to the target size.
    """
    if image.shape[0] == height and image.shape[1] == width:
        return image

    cur_h, cur_w = image.shape[:2]
    ratio = min(width / cur_w, height / cur_h)
    resized_w, resized_h = round(cur_w * ratio), round(cur_h * ratio)

    pil = Image.fromarray(convert_to_uint8(image))
    pil = pil.resize((resized_w, resized_h), resample=Image.BILINEAR)
    resized = np.asarray(pil)

    out = np.zeros((height, width, resized.shape[2] if resized.ndim == 3 else 1), dtype=np.uint8)
    if resized.ndim == 2:
        resized = resized[..., None]
    top = (height - resized_h) // 2
    left = (width - resized_w) // 2
    out[top : top + resized_h, left : left + resized_w, :] = resized
    return out
