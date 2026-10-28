"""
shapes.py – Shape recognition, corner detection, and Fourier descriptors.

Covers:
  - Exp 4: Frequency-domain 1D Fourier Descriptors for scale/rotation/translation invariant shape signatures
  - Exp 5: Canny Edge Detection on the rendered stroke
  - Exp 7: Contour analysis, polygon approximation (Douglas-Peucker approxPolyDP), Circularity
  - Exp 10: Shi-Tomasi Corner Detection (cv2.goodFeaturesToTrack)
"""

import math
from typing import Tuple, Dict, Any
import cv2
import numpy as np

import config
from canvas import Canvas, Stroke


# ---------------------------------------------------------------------------
# Stroke Rasterization into Local Mask
# ---------------------------------------------------------------------------

def stroke_to_image(stroke: Stroke, padding: int = 25) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Renders an in-flight hand-drawn stroke into an isolated minimal bounding-box binary image.
    Returns: (binary_img, contour, (offset_x, offset_y))
    """
    pts = np.array(stroke.points, dtype=np.int32)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)

    w = max(int(x_max - x_min + 2 * padding), 20)
    h = max(int(y_max - y_min + 2 * padding), 20)

    img = np.zeros((h, w), dtype=np.uint8)
    local_pts = pts - np.array([x_min - padding, y_min - padding])

    cv2.polylines(img, [local_pts.reshape(-1, 1, 2)], isClosed=False, color=255, thickness=max(stroke.thickness, 2))
    contour = local_pts.reshape(-1, 1, 2).astype(np.int32)
    offset = (int(x_min - padding), int(y_min - padding))
    return img, contour, offset


# ---------------------------------------------------------------------------
# Fourier Descriptors (Exp 4: Frequency Domain Shape Analysis)
# ---------------------------------------------------------------------------

def fourier_descriptors(contour: np.ndarray, N: int = config.FOURIER_N) -> np.ndarray:
    """
    Calculates 1D Elliptic Fourier Descriptors from 2D contour coordinates.

    Mathematical Formulation (Exp 4):
      1. Contour points represented as complex numbers: z(k) = x(k) + j * y(k).
      2. Resampled along perimeter arc-length to N uniformly-spaced samples.
      3. Discrete Fourier Transform: Z[u] = sum_{k=0}^{N-1} z(k) * e^(-j * 2 * pi * u * k / N).
      4. Translation Invariance: Discard DC term Z[0].
      5. Scale Invariance: Normalize all coefficients by magnitude |Z[1]|.
      6. Rotation & Starting-Point Invariance: Take magnitude |Z[u]|.

    The resulting vector |Z[u]| represents a unique invariant frequency signature.
    Low frequencies capture broad geometric shape; higher frequencies capture fine detail.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 4:
        return np.zeros(N // 2)

    # Arc-length parameterization
    diffs   = np.diff(pts, axis=0, prepend=pts[-1:])
    lengths = np.linalg.norm(diffs, axis=1)
    cumlen  = np.cumsum(lengths)
    total   = cumlen[-1]
    if total < 1e-5:
        return np.zeros(N // 2)

    cumlen /= total
    t_samples = np.linspace(0.0, 1.0, N, endpoint=False)

    # Uniform linear resampling
    xs = np.interp(t_samples, cumlen, pts[:, 0])
    ys = np.interp(t_samples, cumlen, pts[:, 1])

    # Complex signal representation
    z = xs + 1j * ys
    F = np.fft.fft(z)

    # Invariances: Remove DC (index 0), normalize by |F[1]|
    F_no_dc = F[1:]
    norm_val = np.abs(F_no_dc[0]) if np.abs(F_no_dc[0]) > 1e-6 else 1.0
    descriptors = np.abs(F_no_dc / norm_val)[:N // 2]
    return descriptors


# ---------------------------------------------------------------------------
# Shape Classifier (Exp 7: Contours & Exp 10: Corner Detection)
# ---------------------------------------------------------------------------

def classify_shape(contour: np.ndarray, stroke_img: np.ndarray) -> Dict[str, Any]:
    """
    Classifies the geometric identity of a hand-drawn stroke combining:
      - Douglas-Peucker Polygon Approximation (approxPolyDP) -> Vertex count
      - Circularity Metric (Isoperimetric Quotient: 4 * pi * Area / Perimeter^2)
      - Shi-Tomasi Eigenvalue Corner Detector (Exp 10)
      - Fourier Descriptor signature cross-check (Exp 4)
    """
    result: Dict[str, Any] = {
        "shape": "unknown",
        "vertices": 0,
        "circularity": 0.0,
        "corners": 0,
        "aspect_ratio": 1.0,
        "descriptors": np.zeros(config.FOURIER_N // 2),
    }

    if contour is None or len(contour) < 4:
        return result

    # 1. Douglas-Peucker Polygon Approximation (Exp 7)
    peri = cv2.arcLength(contour, closed=True)
    if peri < 1e-5:
        return result

    eps = config.APPROX_EPSILON_RATIO * peri
    approx = cv2.approxPolyDP(contour, eps, closed=True)
    n_verts = len(approx)
    result["vertices"] = n_verts

    # 2. Circularity / Compactness: Circle = 1.0, Square = pi/4 ≈ 0.785, Triangle ≈ 0.60
    area = cv2.contourArea(contour)
    circularity = (4.0 * math.pi * area) / (peri * peri) if peri > 0 else 0.0
    result["circularity"] = round(circularity, 3)

    # 3. Aspect Ratio of Bounding Box
    _, _, bw, bh = cv2.boundingRect(approx)
    aspect_ratio = bw / float(bh) if bh > 0 else 1.0
    result["aspect_ratio"] = round(aspect_ratio, 3)

    # 4. Shi-Tomasi Corner Detection (Exp 10)
    corners = cv2.goodFeaturesToTrack(
        stroke_img,
        maxCorners=config.SHITOMASI_MAX_CORNERS,
        qualityLevel=config.SHITOMASI_QUALITY,
        minDistance=config.SHITOMASI_MIN_DISTANCE
    )
    result["corners"] = len(corners) if corners is not None else 0

    # 5. Fourier Descriptors (Exp 4)
    result["descriptors"] = fourier_descriptors(contour)

    # Decision Tree Classification Logic
    if circularity >= config.CIRCULARITY_THRESH or (n_verts >= 8 and circularity >= 0.65):
        result["shape"] = "circle"
    elif n_verts == 3 or result["corners"] == 3:
        result["shape"] = "triangle"
    elif n_verts == 4 or (3 <= result["corners"] <= 5 and circularity < 0.75):
        if 0.80 <= aspect_ratio <= 1.25:
            result["shape"] = "square"
        else:
            result["shape"] = "rectangle"

    return result


# ---------------------------------------------------------------------------
# Recognize and Replace Pipeline (Exp 5: Canny & Clean Geometric Redraw)
# ---------------------------------------------------------------------------

def recognise_and_replace(stroke: Stroke, canvas: Canvas) -> Dict[str, Any]:
    """
    Processes the raw rough stroke:
      1. Renders stroke to local binary mask.
      2. Computes Canny Edge Detection (Exp 5) and displays in a debug window.
      3. Computes Shi-Tomasi corners and Fourier descriptors.
      4. Replaces noisy hand-drawn stroke with a clean anti-aliased primitive.
    """
    if len(stroke.points) < 5:
        return {"shape": "unknown", "message": "Stroke too short."}

    # 1. Rasterize stroke
    img, local_contour, (ox, oy) = stroke_to_image(stroke)

    # 2. Exp 5: Canny Edge Detection on stroke
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    cv2.imshow("Canny - Stroke Edges (Exp 5)", edges)

    # 3. Contour & classification
    contours_found, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_found:
        return {"shape": "unknown", "message": "No contour detected."}

    best_cnt = max(contours_found, key=cv2.contourArea)
    info = classify_shape(best_cnt, img)

    fd = info["descriptors"]
    print(f"[Shape Rec] Detected: {info['shape'].upper()} | Vertices: {info['vertices']} | "
          f"Circularity: {info['circularity']} | Corners: {info['corners']} | "
          f"FD[:4]: {np.round(fd[:4], 3)}")

    # 4. Clean geometric primitive replacement
    _replace_with_primitive(stroke, info["shape"], canvas)
    return info


def _replace_with_primitive(stroke: Stroke, shape: str, canvas: Canvas) -> None:
    """Removes the rough stroke from canvas and repaints the canonical shape."""
    pts = np.array(stroke.points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts.reshape(-1, 1, 2))
    col = stroke.color
    thk = stroke.thickness

    # Remove rough stroke from stroke stack and refresh canvas
    if stroke in canvas.strokes:
        canvas.strokes.remove(stroke)
    canvas.redraw_all_strokes()

    cx, cy = x + w // 2, y + h // 2

    if shape == "circle":
        r = (w + h) // 4
        cv2.circle(canvas.surface, (cx, cy), r, col, thk, lineType=cv2.LINE_AA)
    elif shape in ("square", "rectangle"):
        cv2.rectangle(canvas.surface, (x, y), (x + w, y + h), col, thk, lineType=cv2.LINE_AA)
    elif shape == "triangle":
        tri = np.array([[cx, y], [x, y + h], [x + w, y + h]], dtype=np.int32)
        cv2.polylines(canvas.surface, [tri], isClosed=True, color=col, thickness=thk, lineType=cv2.LINE_AA)
    else:
        # Fallback: keep original polyline if unknown
        cv2.polylines(canvas.surface, [pts.reshape(-1, 1, 2)], isClosed=False, color=col, thickness=thk, lineType=cv2.LINE_AA)

    # Add back the clean stroke reference
    clean_stroke = Stroke(color=col, thickness=thk)
    clean_stroke.points = [(x, y), (x + w, y + h)]
    canvas.strokes.append(clean_stroke)
