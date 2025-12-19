import math
import numpy as np
from typing import Tuple
from PIL import Image


# Pillow version compatibility for resampling
if hasattr(Image, "Resampling"):
    LANCZOS = Image.Resampling.LANCZOS
elif hasattr(Image, "LANCZOS"):
    LANCZOS = Image.LANCZOS
else:
    LANCZOS = Image.ANTIALIAS


def to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255).astype(np.uint8)


def safe_arccos(x: np.ndarray) -> np.ndarray:
    return np.arccos(np.clip(x, -1.0, 1.0))


def rgb_to_hsi_from_bgr(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = bgr[:, :, 0].astype(np.float32) / 255.0
    g = bgr[:, :, 1].astype(np.float32) / 255.0
    r = bgr[:, :, 2].astype(np.float32) / 255.0

    i = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    denom_i = np.maximum(i, 1e-6)
    s = 1.0 - (min_rgb / denom_i)
    s = np.clip(s, 0.0, 1.0)

    num = (r - g) + (r - b) / 2.0
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    den = np.maximum(den, 1e-6)
    h0 = safe_arccos(num / den) * (180.0 / math.pi)

    h = np.where(b <= g, h0, 360.0 - h0)
    return h, s, i
