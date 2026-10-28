"""
tracker.py – Preprocessing, skin segmentation, hand detection, and fingertip tracking.

Covers:
  - Exp 2: Spatial Filtering (5x5 GaussianBlur for high-frequency noise suppression)
  - Exp 3: Contrast enhancement (CLAHE on V channel of HSV)
  - Exp 6: Morphological Operations (Opening to remove specks, Closing to bridge holes)
  - Exp 7: Contour analysis, Convex Hull, Convexity Defects & Fingertip Localization
  - Exp 10: Angle calculation between convexity defects for finger counting
"""

import math
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np
import config


# ---------------------------------------------------------------------------
# Preprocessing: Blur + Color Conversion + CLAHE (Exp 2 & 3)
# ---------------------------------------------------------------------------

def preprocess(frame: np.ndarray, clahe_on: bool = False) -> np.ndarray:
    """
    Applies Gaussian blurring and converts BGR to HSV color space.
    Optionally applies CLAHE strictly to the V (Value) channel.
    """
    blurred = cv2.GaussianBlur(frame, config.BLUR_KERNEL, sigmaX=0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if clahe_on:
        clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_SIZE
        )
        h, s, v = cv2.split(hsv)
        v_eq = clahe.apply(v)
        hsv = cv2.merge([h, s, v_eq])

    return hsv


# ---------------------------------------------------------------------------
# Skin Color Segmentation (Classical YCrCb + HSV) - Exp 1: Color Space
# ---------------------------------------------------------------------------

def skin_mask(frame: np.ndarray,
              custom_lower: Optional[tuple] = None,
              custom_upper: Optional[tuple] = None) -> np.ndarray:
    """
    Segments human skin using the classical YCrCb color space combined with HSV.

    Why YCrCb for Skin Detection?
      In YCrCb, Y represents luminance (brightness), while Cr (Red-difference)
      and Cb (Blue-difference) represent chrominance. Human skin across diverse
      ethnicities forms an extremely tight elliptical cluster in the Cr-Cb plane:
        133 <= Cr <= 173  and  77 <= Cb <= 127
      This isolates skin from varying ambient brightness.
    """
    # 1. YCrCb segmentation
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb_mask = cv2.inRange(ycrcb,
                             np.array(config.SKIN_YCRCB_LOWER, dtype=np.uint8),
                             np.array(config.SKIN_YCRCB_UPPER, dtype=np.uint8))

    # 2. HSV skin range refinement (eliminates yellowish background noise)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv,
                           np.array(config.SKIN_HSV_LOWER, dtype=np.uint8),
                           np.array(config.SKIN_HSV_UPPER, dtype=np.uint8))

    # Combine both representations for robust segmentation
    combined = cv2.bitwise_and(ycrcb_mask, hsv_mask)

    # If user calibrated custom skin/color bounds, incorporate that mask
    if custom_lower is not None and custom_upper is not None:
        custom = colour_mask(hsv, custom_lower, custom_upper)
        combined = cv2.bitwise_or(combined, custom)

    return combined


# ---------------------------------------------------------------------------
# Color Segmentation & Red Hue Wraparound (Exp 1: Point processing)
# ---------------------------------------------------------------------------

def colour_mask(hsv: np.ndarray, lower: tuple, upper: tuple) -> np.ndarray:
    """
    Creates a binary mask isolating pixels within [lower, upper].
    Handles the 0° / 179° red hue circular wraparound via bitwise_or.
    """
    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)

    if lo[0] <= hi[0]:
        return cv2.inRange(hsv, lo, hi)
    else:
        lo1 = np.array([lo[0], lo[1], lo[2]], dtype=np.uint8)
        hi1 = np.array([179,   hi[1], hi[2]], dtype=np.uint8)
        lo2 = np.array([0,     lo[1], lo[2]], dtype=np.uint8)
        hi2 = np.array([hi[0], hi[1], hi[2]], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lo1, hi1)
        mask2 = cv2.inRange(hsv, lo2, hi2)
        return cv2.bitwise_or(mask1, mask2)


# ---------------------------------------------------------------------------
# Morphological Cleanup (Exp 6: Mathematical Morphology)
# ---------------------------------------------------------------------------

def morphological_cleanup(mask: np.ndarray,
                           kernel_size: int = config.MORPH_KERNEL_INIT) -> np.ndarray:
    """
    Cleans up noise in binary masks using Opening followed by Closing.
    An elliptical structuring element prevents artificial diamond/square distortions.
    """
    k = max(3, kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


# ---------------------------------------------------------------------------
# Hand Contour, Convexity Defects & Fingertip Localization (Exp 7 & 10)
# ---------------------------------------------------------------------------

def find_hand_and_fingertip(clean_mask: np.ndarray):
    """
    Analyzes the segmented hand mask to locate:
      1. Hand contour & area
      2. Palm center (via Euclidean Distance Transform)
      3. Convex Hull & Convexity Defects (Exp 7 & 10)
      4. Number of extended fingers
      5. Coordinates of the active pointing fingertip (fx, fy)

    Returns:
      (fx, fy, palm_center, finger_count, contour, hull_pts, area) or
      (None, None, None, 0, None, None, 0)
    """
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, 0, None, None, 0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < config.MIN_HAND_AREA:
        return None, None, None, 0, None, None, 0

    # 1. Palm Center via Distance Transform (Exp 6/7)
    # Distance transform computes distance from each foreground pixel to the nearest 0.
    # The maximum value corresponds to the center of the largest inscribed circle (the palm center!).
    h, w = clean_mask.shape
    hand_roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(hand_roi_mask, [largest], -1, 255, -1)

    dist_transform = cv2.distanceTransform(hand_roi_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    palm_cx, palm_cy = max_loc
    palm_radius = int(max_val)

    # 2. Convex Hull and Convexity Defects (Exp 7)
    hull_indices = cv2.convexHull(largest, returnPoints=False)
    hull_pts     = cv2.convexHull(largest, returnPoints=True)

    finger_count = 0
    candidate_fingertips: List[Tuple[int, int]] = []

    if hull_indices is not None and len(hull_indices) > 3 and len(largest) > 3:
        try:
            defects = cv2.convexityDefects(largest, hull_indices)
        except cv2.error:
            defects = None

        if defects is not None:
            for s_idx, e_idx, f_idx, d in defects.reshape(-1, 4):
                start = tuple(largest[s_idx][0])
                end   = tuple(largest[e_idx][0])
                far   = tuple(largest[f_idx][0])

                depth = d / 256.0  # OpenCV fixed-point defect depth scale

                # Calculate inter-finger valley angle using law of cosines (Exp 10)
                a = math.dist(start, far)
                b = math.dist(end, far)
                c = math.dist(start, end)

                if a > 0 and b > 0:
                    cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2.0 * a * b)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = math.degrees(math.acos(cos_angle))

                    # True inter-finger valleys have depth > threshold and sharp acute angle (< 85°)
                    if depth > config.DEFECT_MIN_DEPTH and angle_deg < 85.0:
                        finger_count += 1
                        # The start and end points of deep defects are potential fingertips
                        candidate_fingertips.append(start)
                        candidate_fingertips.append(end)

    # Adjust finger count: n defects between outstretched fingers indicates (n + 1) fingers
    if finger_count > 0:
        finger_count += 1
    else:
        # If no deep defects, check if hand is pointing with 1 finger extended vs closed fist
        # Measure ratio of topmost point distance to palm radius
        topmost = tuple(largest[largest[:, :, 1].argmin()][0])
        dist_to_palm = math.dist(topmost, (palm_cx, palm_cy))
        if dist_to_palm > palm_radius * 1.8:
            finger_count = 1  # 1 finger pointing upwards!

    # 3. Fingertip Localization (The Pointing Finger)
    # When pointing, the active fingertip is the point in the contour that is:
    #   - Highest (smallest y coordinate) and furthest away from the palm center
    if candidate_fingertips and finger_count > 1:
        # Filter fingertips that are distinctly above the palm center
        above_palm = [pt for pt in candidate_fingertips if pt[1] < palm_cy]
        if above_palm:
            # Point with highest y-distance above palm (furthest up)
            fingertip = min(above_palm, key=lambda pt: pt[1])
        else:
            fingertip = min(candidate_fingertips, key=lambda pt: pt[1])
    else:
        # For single-finger pointing: the topmost contour point is the fingertip!
        fingertip = tuple(largest[largest[:, :, 1].argmin()][0])

    fx, fy = int(fingertip[0]), int(fingertip[1])
    return fx, fy, (palm_cx, palm_cy), finger_count, largest, hull_pts, area


def draw_hand_overlay(frame: np.ndarray,
                      fx: Optional[int],
                      fy: Optional[int],
                      palm_center: Optional[Tuple[int, int]],
                      finger_count: int,
                      contour: np.ndarray,
                      hull_pts: np.ndarray,
                      is_drawing: bool = True) -> None:
    """
    Renders hand contour, convex hull wireframe, palm anchor, and fingertip cursor.
    """
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (60, 180, 70), 1)

    if hull_pts is not None:
        cv2.drawContours(frame, [hull_pts], -1, (220, 220, 100), 1)

    if palm_center is not None:
        cv2.circle(frame, palm_center, 6, (200, 100, 0), -1)

    if fx is not None and fy is not None:
        # Cursor color: Green for DRAWING mode, Cyan/Blue for HOVER/POINTING mode
        cursor_col = (0, 255, 0) if is_drawing else (255, 200, 0)

        # Concentric pointer target
        cv2.circle(frame, (fx, fy), 5, (0, 0, 255), -1)       # Core dot
        cv2.circle(frame, (fx, fy), 14, cursor_col, 2)        # Outer ring
        # Crosshair hairs
        cv2.line(frame, (fx - 18, fy), (fx - 7, fy), cursor_col, 1)
        cv2.line(frame, (fx + 7, fy), (fx + 18, fy), cursor_col, 1)
        cv2.line(frame, (fx, fy - 18), (fx, fy - 7), cursor_col, 1)
        cv2.line(frame, (fx, fy + 7), (fx, fy + 18), cursor_col, 1)


# ---------------------------------------------------------------------------
# Fallback Marker Detection (Exp 7: Centroid from Color Blobs)
# ---------------------------------------------------------------------------

def find_marker(clean_mask: np.ndarray):
    """Finds the largest colored marker contour and calculates its centroid."""
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < config.MIN_CONTOUR_AREA:
        return None, None, 0, None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None, 0, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy, area, largest


def draw_marker_overlay(frame: np.ndarray,
                        cx: int,
                        cy: int,
                        contour: np.ndarray,
                        ring_color: tuple = (0, 255, 0)) -> None:
    """Visualizes marker boundary contour and center centroid crosshair."""
    if contour is not None:
        cv2.drawContours(frame, [contour], -1, ring_color, 2)
    if cx is not None and cy is not None:
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), 12, ring_color, 2)
