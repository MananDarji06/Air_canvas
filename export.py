"""
export.py – Canvas export, JPEG quality sweep, PSNR calculation, and compression artifacts report.

Covers:
  - Exp 12: Image Compression & Quality Assessment (PSNR, MSE, DCT Ringing / Blocking artifacts)
"""

import os
import math
from typing import Optional
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import config


# ---------------------------------------------------------------------------
# PSNR & MSE Metrics (Exp 12)
# ---------------------------------------------------------------------------

def compute_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB):
        MSE  = 1 / (3 * M * N) * sum_{i,j,c} (I_orig(i,j,c) - I_comp(i,j,c))^2
        PSNR = 10 * log10(MAX_I^2 / MSE)
    where MAX_I = 255 for 8-bit unsigned images.

    Interpretation:
      - PSNR > 40 dB: Virtually identical, indistinguishable distortion.
      - 30 dB - 40 dB: Good quality, subtle compression degradation.
      - < 30 dB: Poor quality, noticeable blockiness and chromatic fringing.
    """
    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))

    diff = original.astype(np.float64) - compressed.astype(np.float64)
    mse  = np.mean(diff ** 2)

    if mse < 1e-10:
        return float("inf")

    psnr = 10.0 * math.log10((255.0 ** 2) / mse)
    return round(psnr, 2)


# ---------------------------------------------------------------------------
# File I/O Helpers
# ---------------------------------------------------------------------------

def ensure_export_dir() -> str:
    """Ensures export output directory exists."""
    os.makedirs(config.EXPORT_DIR, exist_ok=True)
    return config.EXPORT_DIR


def save_png(canvas_surface: np.ndarray, filename: str = "canvas.png") -> str:
    """Saves canvas as a lossless PNG reference image (DEFLATE compression)."""
    path = os.path.join(ensure_export_dir(), filename)
    cv2.imwrite(path, canvas_surface)
    size_kb = os.path.getsize(path) / 1024.0
    print(f"[Export] Lossless PNG saved: {path} ({size_kb:.1f} KB)")
    return path


def save_jpeg(canvas_surface: np.ndarray, quality: int, filename: Optional[str] = None) -> str:
    """
    Saves canvas as lossy JPEG at a specified quality factor (1-100).
    Uses Discrete Cosine Transform (DCT) block quantization.
    """
    if filename is None:
        filename = f"canvas_q{quality}.jpg"
    path = os.path.join(ensure_export_dir(), filename)
    cv2.imwrite(path, canvas_surface, [cv2.IMWRITE_JPEG_QUALITY, quality])
    size_kb = os.path.getsize(path) / 1024.0
    print(f"[Export] JPEG (Q={quality}) saved: {path} ({size_kb:.1f} KB)")
    return path


# ---------------------------------------------------------------------------
# Compression Quality Sweep & Report (Exp 12)
# ---------------------------------------------------------------------------

def compression_report(canvas_surface: np.ndarray) -> None:
    """
    Executes a comprehensive compression quality sweep across qualities [95, 75, 50, 25, 10]:
      1. Saves lossless PNG ground truth.
      2. Encodes JPEG across quality levels, measuring byte size and PSNR.
      3. Generates and saves a dual-plot Matplotlib figure:
           - File Size (KB) vs. Quality
           - PSNR (dB) vs. Quality
      4. Saves a magnified crop comparison of a stroke edge at Q=10 to showcase
         Gibbs ringing and 8x8 DCT grid blocking artifacts.
    """
    ensure_export_dir()

    # 1. Lossless reference
    png_path = save_png(canvas_surface)
    ref_bgr  = cv2.imread(png_path)

    qualities = config.JPEG_QUALITIES
    sizes_kb  = []
    psnr_vals = []

    print("\n" + "=" * 50)
    print("EXP 12: JPEG Compression Sweep & PSNR Evaluation")
    print("=" * 50)

    # 2. Quality sweep
    for q in qualities:
        jpg_path = save_jpeg(canvas_surface, q)
        jpg_img  = cv2.imread(jpg_path)
        psnr     = compute_psnr(ref_bgr, jpg_img)
        size     = os.path.getsize(jpg_path) / 1024.0

        sizes_kb.append(size)
        psnr_vals.append(psnr if not math.isinf(psnr) else 60.0)
        print(f"  Quality={q:2d} | File Size: {size:6.1f} KB | PSNR: {psnr:5.2f} dB")

    print("=" * 50 + "\n")

    # 3. Two-panel analytical visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.canvas.manager.set_window_title("Air Canvas - Exp 12: Image Compression Analysis")
    fig.suptitle("JPEG Rate-Distortion & Quality Sweep (Exp 12)", fontsize=11, fontweight="bold")

    # File Size vs. Quality
    ax1.plot(qualities, sizes_kb, "o-", color="#2980b9", lw=2, markersize=6)
    ax1.set_xlabel("JPEG Quality Factor", fontsize=9)
    ax1.set_ylabel("Compressed File Size (KB)", fontsize=9)
    ax1.set_title("Storage Footprint vs. Quality", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PSNR vs. Quality
    ax2.plot(qualities, psnr_vals, "s-", color="#c0392b", lw=2, markersize=6)
    ax2.axhline(30.0, color="#7f8c8d", linestyle="--", lw=1.5, label="30 dB Quality Threshold")
    ax2.set_xlabel("JPEG Quality Factor", fontsize=9)
    ax2.set_ylabel("PSNR (dB)", fontsize=9)
    ax2.set_title("Peak Signal-to-Noise Ratio vs. Quality", fontsize=9)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    report_img_path = os.path.join(config.EXPORT_DIR, "compression_report.png")
    plt.savefig(report_img_path, dpi=140)
    print(f"[Export] Analytical report saved to: {report_img_path}")
    plt.show(block=False)
    plt.pause(0.5)

    # 4. Zoomed edge crop demonstrating Gibbs ringing & DCT blocking
    _show_edge_artifacts(ref_bgr, qualities[-1])


def _show_edge_artifacts(ref_img: np.ndarray, low_quality: int) -> None:
    """
    Generates a 4x magnified side-by-side comparison of a stroke boundary
    contrasting PNG with low-quality JPEG to illustrate DCT compression artifacts.
    """
    gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)

    if len(xs) == 0:
        print("[Export] Canvas empty. Skipping artifact crop visualization.")
        return

    # Center crop around average stroke location
    cy = int(np.clip(np.mean(ys), 30, ref_img.shape[0] - 30))
    cx = int(np.clip(np.mean(xs), 30, ref_img.shape[1] - 30))
    rad = 25

    y1, y2 = cy - rad, cy + rad
    x1, x2 = cx - rad, cx + rad

    crop_png = ref_img[y1:y2, x1:x2]
    low_path = os.path.join(config.EXPORT_DIR, f"canvas_q{low_quality}.jpg")
    if not os.path.exists(low_path):
        return

    low_img  = cv2.imread(low_path)
    crop_low = low_img[y1:y2, x1:x2]

    # 4x Nearest-neighbor interpolation reveals discrete 8x8 DCT grid block boundaries
    scale = 4
    zoom_png = cv2.resize(crop_png, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    zoom_low = cv2.resize(crop_low, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    comparison = np.hstack([zoom_png, zoom_low])
    comp_file = os.path.join(config.EXPORT_DIR, "artefact_crop_comparison.png")
    cv2.imwrite(comp_file, comparison)
    print(f"[Export] Zoomed artifact comparison saved: {comp_file}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))
    fig.canvas.manager.set_window_title("Air Canvas - Exp 12: DCT Artifact Inspection")
    fig.suptitle("Stroke Boundary: PNG vs. JPEG Q=10 (DCT Blocking & Ringing)", fontsize=10, fontweight="bold")

    ax1.imshow(cv2.cvtColor(zoom_png, cv2.COLOR_BGR2RGB))
    ax1.set_title("Lossless PNG (Sharp Edges)", fontsize=9)
    ax1.axis("off")

    ax2.imshow(cv2.cvtColor(zoom_low, cv2.COLOR_BGR2RGB))
    ax2.set_title(f"JPEG Q={low_quality} (Ringing & 8x8 Blocks)", fontsize=9)
    ax2.axis("off")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
