"""
trajectory.py – Trajectory smoothing via Moving Average and FFT Low-Pass Filtering.

Covers:
  - Exp 4: Frequency-Domain Processing (1D Discrete Fourier Transform, Brick-wall Low-pass filter)
"""

from typing import Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import config


# ---------------------------------------------------------------------------
# Centroid Trajectory FIFO Buffer
# ---------------------------------------------------------------------------

class TrajectoryBuffer:
    """
    Fixed-capacity ring buffer tracking recent centroid positions.
    Maintains constant sample length for consistent FFT frequency resolution.
    """

    def __init__(self, maxlen: int = config.TRAJ_BUFFER_LEN):
        self.maxlen = maxlen
        self._xs: list = []
        self._ys: list = []

    def push(self, x: int, y: int) -> None:
        self._xs.append(x)
        self._ys.append(y)
        if len(self._xs) > self.maxlen:
            self._xs.pop(0)
            self._ys.pop(0)

    def clear(self) -> None:
        self._xs.clear()
        self._ys.clear()

    @property
    def xs(self) -> np.ndarray:
        return np.array(self._xs, dtype=np.float32)

    @property
    def ys(self) -> np.ndarray:
        return np.array(self._ys, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._xs)

    def last_smoothed(self, use_fft: bool = False, window: int = config.MA_WINDOW) -> Tuple[Optional[int], Optional[int]]:
        """Returns the smoothed coordinate of the latest coordinate sample."""
        if len(self) == 0:
            return None, None

        if use_fft and len(self) >= 6:
            sx = fft_lowpass(self.xs, cutoff_hz=config.FFT_CUTOFF_HZ, fps=config.FPS_ESTIMATE)
            sy = fft_lowpass(self.ys, cutoff_hz=config.FFT_CUTOFF_HZ, fps=config.FPS_ESTIMATE)
        else:
            sx = moving_average(self.xs, window)
            sy = moving_average(self.ys, window)

        return int(round(sx[-1])), int(round(sy[-1]))


# ---------------------------------------------------------------------------
# Time-Domain Smoothing: Moving Average (Exp 2/Exp 4 comparison)
# ---------------------------------------------------------------------------

def moving_average(signal: np.ndarray, window: int = config.MA_WINDOW) -> np.ndarray:
    """
    Computes moving average via 1D discrete convolution:
        y[n] = 1/W * sum_{k=0}^{W-1} x[n-k]
    Equivalent to a uniform boxcar low-pass FIR filter.
    """
    if len(signal) == 0:
        return signal
    w = max(1, min(window, len(signal)))
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(signal, kernel, mode="same")


# ---------------------------------------------------------------------------
# Frequency-Domain Filtering: FFT Brick-Wall Low-Pass (Exp 4)
# ---------------------------------------------------------------------------

def fft_lowpass(signal: np.ndarray,
                cutoff_hz: float = config.FFT_CUTOFF_HZ,
                fps: float = config.FPS_ESTIMATE) -> np.ndarray:
    """
    Applies frequency-domain low-pass filtering via real-valued FFT.

    Pipeline (Exp 4):
      1. np.fft.rfft: Transform 1D discrete temporal signal to frequency spectrum X[k].
      2. Frequency mapping: f_k = k * fps / N for k in [0, N//2].
      3. Brick-wall filtering: Set X[k] = 0 for all f_k > cutoff_hz.
      4. np.fft.irfft: Inverse FFT reconstructing smoothed spatial coordinates.

    Why rfft instead of fft?
      Coordinates x(t) and y(t) are strictly real-valued. Real signals exhibit
      Hermitian symmetry in their Fourier transforms: X[-k] = X*[k].
      np.fft.rfft only computes non-redundant positive frequencies, cutting
      computation and memory in half.
    """
    N = len(signal)
    if N < 4:
        return signal

    spectrum = np.fft.rfft(signal)
    freqs    = np.fft.rfftfreq(N, d=1.0 / fps)

    # Zero all frequency components higher than cutoff frequency
    spectrum[freqs > cutoff_hz] = 0.0

    filtered = np.fft.irfft(spectrum, n=N)
    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Trajectory Diagnostic Visualization (Key 'p')
# ---------------------------------------------------------------------------

def show_trajectory_plot(raw_xs: np.ndarray,
                         cutoff_hz: float = config.FFT_CUTOFF_HZ,
                         fps: float = config.FPS_ESTIMATE) -> None:
    """
    Generates a 2-panel Matplotlib figure comparing time-domain trajectory
    signals against frequency magnitude spectra.
    """
    if len(raw_xs) < 6:
        print("[Trajectory] Insufficient points to plot.")
        return

    ma_xs  = moving_average(raw_xs, config.MA_WINDOW)
    fft_xs = fft_lowpass(raw_xs, cutoff_hz, fps)

    N     = len(raw_xs)
    t     = np.arange(N) / fps
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)
    mag   = np.abs(np.fft.rfft(raw_xs))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5))
    fig.canvas.manager.set_window_title("Air Canvas - Exp 4: Frequency Domain Trajectory Analysis")
    fig.suptitle("Trajectory Filtering (Exp 4: Time vs. Frequency Domain)", fontsize=11, fontweight="bold")

    # Time-Domain Signal Panel
    ax1.plot(t, raw_xs,  label="Raw x(t) [Jittery]", color="#7f8c8d", lw=1.2, alpha=0.7)
    ax1.plot(t, ma_xs,   label=f"Moving Average (W={config.MA_WINDOW})", color="#2980b9", lw=1.8, linestyle="--")
    ax1.plot(t, fft_xs,  label=f"FFT Low-Pass (< {cutoff_hz} Hz)", color="#e74c3c", lw=2.0)
    ax1.set_xlabel("Time (s)", fontsize=9)
    ax1.set_ylabel("Position x (pixels)", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Frequency-Domain Magnitude Spectrum Panel
    ax2.plot(freqs, mag, color="#8e44ad", lw=1.8, marker="o", markersize=4, label="Raw Spectrum |X(f)|")
    ax2.axvline(cutoff_hz, color="#c0392b", linestyle="--", lw=1.5, label=f"Cutoff = {cutoff_hz} Hz")
    ax2.set_xlabel("Frequency (Hz)", fontsize=9)
    ax2.set_ylabel("Magnitude", fontsize=9)
    ax2.set_title("1D Magnitude Spectrum of x(t)", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)
