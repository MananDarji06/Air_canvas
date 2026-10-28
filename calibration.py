"""
calibration.py – Click-to-calibrate HSV range using patch histograms.

Covers:
  - Exp 3: Histogram Processing (cv2.calcHist, percentiles, dynamic thresholding)
"""

import os
import json
from typing import Tuple, Optional
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import config


# ---------------------------------------------------------------------------
# Calibration Persistence
# ---------------------------------------------------------------------------

def save_calibration(lower: tuple, upper: tuple, path: str = config.CALIB_FILE) -> None:
    """Saves calibrated HSV lower and upper bounds to JSON."""
    data = {"lower": list(lower), "upper": list(upper)}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Calibration] Saved to {path}: Lower={lower}, Upper={upper}")


def load_calibration(path: str = config.CALIB_FILE) -> Tuple[Optional[tuple], Optional[tuple]]:
    """Loads saved HSV bounds if file exists."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        lower = tuple(data["lower"])
        upper = tuple(data["upper"])
        print(f"[Calibration] Loaded from {path}: Lower={lower}, Upper={upper}")
        return lower, upper
    except Exception as e:
        print(f"[Calibration] Failed to parse {path}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Histogram-Based Bound Estimation (Exp 3)
# ---------------------------------------------------------------------------

def compute_hsv_bounds(hsv_frame: np.ndarray, cx: int, cy: int):
    """
    Extracts a local patch around (cx, cy) and calculates the H, S, V bounds
    based on the 5th and 95th percentiles of the patch histograms.

    Why Percentiles over Min/Max?
      Min/Max are vulnerable to outlier pixels (noise, specular glints).
      Percentiles (5th-95th) exclude extreme 5% tails, isolating the dominant
      spectral cluster of the marker while adding CALIB_MARGIN for tolerance.
    """
    h_img, w_img = hsv_frame.shape[:2]
    p = config.CALIB_PATCH_HALF

    y1 = max(0, cy - p)
    y2 = min(h_img, cy + p + 1)
    x1 = max(0, cx - p)
    x2 = min(w_img, cx + p + 1)

    patch = hsv_frame[y1:y2, x1:x2]
    channels = cv2.split(patch)
    limits = [179, 255, 255]

    lower = []
    upper = []
    for ch, maxval in zip(channels, limits):
        flat = ch.flatten().astype(np.float32)
        lo = int(np.percentile(flat, config.CALIB_PERCENTILE_LO))
        hi = int(np.percentile(flat, config.CALIB_PERCENTILE_HI))

        lo = max(0, lo - config.CALIB_MARGIN)
        hi = min(maxval, hi + config.CALIB_MARGIN)
        lower.append(lo)
        upper.append(hi)

    return tuple(lower), tuple(upper), patch


def show_patch_histograms(patch_hsv: np.ndarray) -> None:
    """
    Renders H, S, and V channel histograms of the calibration sample patch.
    Satisfies Exp 3: Histogram analysis & visualization.
    """
    channel_info = [
        ("Hue Histogram (0-179)", 180, "#e74c3c"),
        ("Saturation Histogram (0-255)", 256, "#2ecc71"),
        ("Value Histogram (0-255)", 256, "#3498db")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    fig.canvas.manager.set_window_title("Air Canvas - Exp 3: Calibration Histograms")
    fig.suptitle("Marker Patch Histograms (Exp 3: Histogram Processing)", fontsize=11, fontweight="bold")

    channels = cv2.split(patch_hsv)
    for ax, ch, (title, bins, color) in zip(axes, channels, channel_info):
        hist = cv2.calcHist([ch], [0], None, [bins], [0, bins]).flatten()
        ax.plot(hist, color=color, lw=1.5)
        ax.fill_between(range(bins), hist, color=color, alpha=0.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Bin Value", fontsize=8)
        ax.set_ylabel("Pixel Count", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


# ---------------------------------------------------------------------------
# Mouse Callback State Machine
# ---------------------------------------------------------------------------

class CalibrationState:
    """Encapsulates mouse click event handling for interactive calibration."""

    def __init__(self):
        self.active   = False
        self.clicked  = False
        self.click_xy = None

    def enable(self):
        self.active   = True
        self.clicked  = False
        self.click_xy = None
        print("[Calibration] Mode ACTIVE. Click on your marker in the video window.")

    def disable(self):
        self.active  = False
        self.clicked = False

    def mouse_callback(self, event, x, y, flags, param):
        if self.active and event == cv2.EVENT_LBUTTONDOWN:
            self.click_xy = (x, y)
            self.clicked  = True


def process_click(state: CalibrationState, hsv_frame: np.ndarray):
    """Computes bounds from recorded click point, persists to JSON, shows plots."""
    cx, cy = state.click_xy
    lower, upper, patch = compute_hsv_bounds(hsv_frame, cx, cy)
    save_calibration(lower, upper)
    show_patch_histograms(patch)

    state.clicked = False
    state.disable()
    return lower, upper
