"""
transforms.py – Affine transformations on the canvas layer.

Covers:
  - Exp 9: Geometric Transformations (2x3 Affine Matrix, Rotation, Scaling, Translation, Mirroring)

Mathematical Basis of Affine Transformations:
  An affine transformation maps collinear points to collinear points and preserves parallelism.
  Represented in homogeneous coordinates by a 2x3 matrix M:
      [ x' ]   [ a   b   tx ]   [ x ]
      [ y' ] = [ c   d   ty ] * [ y ]
                                [ 1 ]
  Where [a, b; c, d] handles rotation and scaling, and [tx, ty] specifies translation.
  Applied to the image plane using cv2.warpAffine() with bilinear interpolation.
"""

import cv2
import numpy as np
import config


# ---------------------------------------------------------------------------
# Individual Affine Operations (Exp 9)
# ---------------------------------------------------------------------------

def flip_horizontal(img: np.ndarray) -> np.ndarray:
    """
    Mirrors image across vertical y-axis (flipCode=1).
    Equivalent to affine matrix:
      M = [ [-1, 0, W - 1],
            [ 0, 1, 0     ] ]
    """
    return cv2.flip(img, 1)


def rotate_canvas(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates canvas around its center coordinate (cx, cy) by angle_deg.
    cv2.getRotationMatrix2D builds the 2x3 affine rotation matrix with scale=1.0.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def scale_canvas(img: np.ndarray, factor: float) -> np.ndarray:
    """
    Scales canvas outward or inward relative to center coordinate.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), 0.0, factor)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def translate_canvas(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    Translates the canvas coordinate system by (dx, dy) pixels.
    Matrix M = [ [1, 0, dx],
                 [0, 1, dy] ]
    """
    h, w = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ---------------------------------------------------------------------------
# Cumulative Affine Transformation State Manager
# ---------------------------------------------------------------------------

class TransformState:
    """
    Maintains cumulative rotation, scaling, and translation state.
    Computes a single unified composite 2x3 affine matrix to prevent compounding
    resampling/interpolation blur caused by sequential warpAffine calls.
    """

    def __init__(self):
        self.angle = 0.0
        self.scale = 1.0
        self.tx    = 0
        self.ty    = 0

    def rotate(self, delta_deg: float) -> None:
        self.angle = (self.angle + delta_deg) % 360.0

    def zoom(self, delta: float) -> None:
        self.scale = max(0.1, min(4.0, self.scale + delta))

    def translate(self, dx: int, dy: int) -> None:
        self.tx += dx
        self.ty += dy

    def reset(self) -> None:
        self.angle = 0.0
        self.scale = 1.0
        self.tx    = 0
        self.ty    = 0

    def apply(self, canvas_surface: np.ndarray) -> np.ndarray:
        """
        Applies composite affine matrix [R*S | t] to canvas_surface in one pass.
        """
        if self.angle == 0.0 and self.scale == 1.0 and self.tx == 0 and self.ty == 0:
            return canvas_surface

        h, w = canvas_surface.shape[:2]
        cx, cy = w // 2, h // 2

        # 2x3 Rotation + Scaling matrix
        M = cv2.getRotationMatrix2D((cx, cy), self.angle, self.scale)

        # Incorporate translation offset
        M[0, 2] += self.tx
        M[1, 2] += self.ty

        return cv2.warpAffine(
            canvas_surface, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
