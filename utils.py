"""
utils.py – Utility functions, FPS counter, point processing (LUT, Negative), arithmetic demo.

Covers:
  - Exp 1: Point processing (Gamma correction via LUT, Digital negative)
  - Exp 11: Arithmetic overflow demo (saturating cv2.add vs modular numpy +)
"""

import time
import cv2
import numpy as np
import config


# ---------------------------------------------------------------------------
# FPS Counter (Exponential Moving Average)
# ---------------------------------------------------------------------------

class FPSCounter:
    """
    Computes rolling average FPS using an exponential moving average (EMA).
    Prevents jerky UI text updates while preserving responsiveness.
    """

    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self._fps  = 0.0
        self._prev = time.perf_counter()

    def tick(self) -> float:
        now = time.perf_counter()
        dt  = now - self._prev
        self._prev = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = self.alpha * inst + (1.0 - self.alpha) * self._fps
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


# ---------------------------------------------------------------------------
# UI Text Helper with High-Contrast Drop Shadow
# ---------------------------------------------------------------------------

def put_text(img: np.ndarray,
             text: str,
             pos: tuple,
             color: tuple = config.INFO_COLOR,
             scale: float = config.FONT_SCALE,
             thickness: int = config.FONT_THICK) -> None:
    """
    Renders anti-aliased text with a dark shadow outline for readability
    across both light and dark video backgrounds.
    """
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1),
                config.FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y),
                config.FONT, scale, color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Point Processing: Gamma Correction Look-Up Table (Exp 1)
# ---------------------------------------------------------------------------

def build_gamma_lut(gamma: float) -> np.ndarray:
    """
    Generates a 256-element uint8 LUT implementing power-law (gamma) transformation:
        s = c * r^(1 / gamma)
    where r is input intensity normalized to [0, 1].

    - gamma < 1: Compresses highlights, expands shadows (brightens dark rooms).
    - gamma > 1: Expands highlights, compresses shadows (darkens overexposed scenes).

    Using cv2.LUT() provides O(1) table lookup per pixel, dramatically faster
    than computing floating-point powers on millions of pixels per second.
    """
    inv_gamma = 1.0 / max(gamma, 1e-4)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)],
        dtype=np.uint8
    )
    return table


def apply_gamma(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Applies precomputed gamma LUT across all image channels."""
    return cv2.LUT(img, lut)


# ---------------------------------------------------------------------------
# Point Processing: Digital Negative (Exp 1)
# ---------------------------------------------------------------------------

def digital_negative(img: np.ndarray) -> np.ndarray:
    """
    Inverts intensity levels: s = (L - 1) - r = 255 - r.
    Enhances white or grey details embedded in dark regions.
    """
    return cv2.subtract(np.full_like(img, 255), img)


# ---------------------------------------------------------------------------
# Viva Demonstration: Saturating vs. Modular Arithmetic (Exp 11)
# ---------------------------------------------------------------------------

def demo_arithmetic_overflow():
    """
    Demonstrates the difference between saturating arithmetic (cv2.add) and
    wraparound modular arithmetic (NumPy +) on uint8 data.

    Example: 250 + 10
      - OpenCV clamps at 255 (saturation).
      - NumPy wraps modulo 256: (250 + 10) % 256 = 4.

    This illustrates why cv2.add is mandatory for overlaying canvas ink.
    """
    a = np.full((2, 2), 250, dtype=np.uint8)
    b = np.full((2, 2), 10, dtype=np.uint8)

    sat_result = int(cv2.add(a, b)[0, 0])   # saturating -> 255
    wrap_result = int((a + b)[0, 0])        # modular wrap -> 4

    print("=" * 55)
    print("VIVA DEMO: Arithmetic Overflow (uint8: 250 + 10)")
    print(f"  cv2.add  (saturates) : {sat_result}   <-- clamped at 255")
    print(f"  NumPy +  (wraps)     : {wrap_result}     <-- 260 % 256 = 4")
    print("=" * 55)
    return sat_result, wrap_result


# ---------------------------------------------------------------------------
# Geometric Bounding Helpers
# ---------------------------------------------------------------------------

def clamp(val, lo, hi):
    """Clamps a numerical value within [lo, hi]."""
    return max(lo, min(hi, val))


def rect_contains(rect: tuple, point: tuple) -> bool:
    """Returns True if point (px, py) is strictly inside (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = rect
    px, py = point
    return x1 <= px <= x2 and y1 <= py <= y2
