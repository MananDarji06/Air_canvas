"""
canvas.py – Drawing surface management, toolbar ROI palette, and overlay blending.

Covers:
  - Exp 8: NumPy array manipulation as graphics canvas & ROI button bounds
  - Exp 11: Image Arithmetic & Bitwise Logic (Method A: addWeighted, Method B: bitwise masking)
"""

import time
from typing import Optional, List, Tuple
import cv2
import numpy as np

import config
from utils import rect_contains


# ---------------------------------------------------------------------------
# Stroke Representation
# ---------------------------------------------------------------------------

class Stroke:
    """
    Stores an ordered polyline trajectory of (x, y) points plus visual attributes.
    Enables post-processing, shape classification, and clean geometric replacement.
    """

    def __init__(self, color: tuple, thickness: int):
        self.color     = color
        self.thickness = thickness
        self.points: List[Tuple[int, int]] = []

    def add(self, pt: Tuple[int, int]) -> None:
        self.points.append(pt)

    def __len__(self) -> int:
        return len(self.points)


# ---------------------------------------------------------------------------
# Canvas Manager Class
# ---------------------------------------------------------------------------

class Canvas:
    """
    Encapsulates the discrete 2D pixel canvas array, brush state, and toolbar UI.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width  = width

        # Black uint8 NumPy array represents transparent drawing surface
        self.surface: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)

        # Historical stroke registry for shape recognition/redrawing
        self.strokes: List[Stroke] = []
        self.active_stroke: Optional[Stroke] = None

        # Drawing state
        self.color     = config.PALETTE_BUTTONS[0][1]
        self.thickness = config.BRUSH_THICKNESS_INIT
        self.tool      = "draw"

        # Build interactive palette geometry
        self._build_toolbar()

        # Dwell button activation state
        self._dwell_start:   Optional[float] = None
        self._dwell_btn_idx: Optional[int]   = None

    def _build_toolbar(self) -> None:
        """Constructs non-overlapping rectangular bounding boxes across top toolbar."""
        n     = len(config.PALETTE_BUTTONS)
        m     = config.BUTTON_MARGIN
        btn_w = (self.width - (n + 1) * m) // n
        btn_h = config.TOOLBAR_HEIGHT - 2 * m

        self.button_rects: List[Tuple[int, int, int, int]] = []
        for i in range(n):
            x1 = m + i * (btn_w + m)
            y1 = m
            x2 = x1 + btn_w
            y2 = y1 + btn_h
            self.button_rects.append((x1, y1, x2, y2))

    def draw_toolbar(self, frame: np.ndarray, active_idx: Optional[int] = None) -> None:
        """
        Renders the ROI toolbar buttons directly onto the video display frame.
        Exp 8: Image Sub-regions (ROI) and geometric annotations.
        """
        # Toolbar banner background
        cv2.rectangle(frame, (0, 0), (self.width, config.TOOLBAR_HEIGHT), (35, 35, 35), -1)

        for i, ((x1, y1, x2, y2), (label, color)) in enumerate(zip(self.button_rects, config.PALETTE_BUTTONS)):
            fill = color if color is not None else (70, 70, 70)
            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)

            # Highlight border for active/hovered button
            border_col = (255, 255, 255) if i == active_idx else (140, 140, 140)
            border_thk = 3 if i == active_idx else 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_col, border_thk)

            # Label text
            txt_col = (0, 0, 0) if label in ("Yellow",) else (245, 245, 245)
            cv2.putText(frame, label, (x1 + 3, y2 - 7), config.FONT, 0.38, txt_col, 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Interactive Drawing
    # -----------------------------------------------------------------------

    def begin_stroke(self, pt: Tuple[int, int]) -> None:
        """Initiates a continuous stroke trajectory."""
        thk = config.ERASER_THICKNESS if self.tool == "eraser" else self.thickness
        col = (0, 0, 0) if self.tool == "eraser" else self.color
        stroke = Stroke(color=col, thickness=thk)
        stroke.add(pt)
        self.active_stroke = stroke

    def continue_stroke(self, pt: Tuple[int, int]) -> None:
        """Draws a segment between consecutive points to ensure unbroken lines."""
        if self.active_stroke is None:
            self.begin_stroke(pt)
            return

        prev = self.active_stroke.points[-1]
        cv2.line(self.surface, prev, pt, self.active_stroke.color, self.active_stroke.thickness)
        self.active_stroke.add(pt)

    def end_stroke(self) -> Optional[Stroke]:
        """Finalizes the active drawing stroke and commits it to history."""
        if self.active_stroke is None or len(self.active_stroke) < 2:
            self.active_stroke = None
            return None
        self.strokes.append(self.active_stroke)
        completed = self.active_stroke
        self.active_stroke = None
        return completed

    def clear(self) -> None:
        """
        Resets the canvas in-place using NumPy zero-assignment: canvas[:] = 0.
        Fast O(1) buffer reuse without heap re-allocation.
        """
        self.surface[:] = 0
        self.strokes.clear()
        self.active_stroke = None

    def redraw_all_strokes(self) -> None:
        """Repaints all stored historical strokes onto a clean surface."""
        self.surface[:] = 0
        for stroke in self.strokes:
            pts = stroke.points
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(self.surface, a, b, stroke.color, stroke.thickness)

    # -----------------------------------------------------------------------
    # Overlay Blending Pipeline (Exp 11: Arithmetic & Bitwise Operations)
    # -----------------------------------------------------------------------

    def overlay_on_frame(self, frame: np.ndarray, method: str = "B") -> np.ndarray:
        """
        Combines the ink surface with the webcam video frame.

        Method A (cv2.addWeighted):
          Linear combination: I_out = alpha * I_frame + beta * I_canvas + gamma
          Produces semi-transparent (ghosted) strokes.

        Method B (Bitwise masking + cv2.add - DEFAULT):
          1. cv2.cvtColor: Convert canvas to grayscale.
          2. cv2.threshold: Binary mask M where ink exists (canvas > 0).
          3. cv2.bitwise_not: Inverted mask ~M representing background.
          4. cv2.bitwise_and: Zero out webcam pixels where ink will go (frame & ~M).
          5. cv2.bitwise_and: Retain only ink pixels from canvas (canvas & M).
          6. cv2.add: Saturating addition combining background and opaque ink.
          Result: Fully opaque, vivid ink with sharp edges.
        """
        if method == "A":
            # Method A: Linear weighted blend
            out = frame.copy()
            cv2.addWeighted(out, config.OVERLAY_ALPHA, self.surface, config.OVERLAY_BETA, 0, out)
            return out

        else:
            # Method B: Bitwise binary masking (Exp 11)
            gray = cv2.cvtColor(self.surface, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            canvas_fg = cv2.bitwise_and(self.surface, self.surface, mask=mask)

            return cv2.add(frame_bg, canvas_fg)

    # -----------------------------------------------------------------------
    # ROI Palette Dwell Activation (Exp 8)
    # -----------------------------------------------------------------------

    def check_palette(self, pt: Optional[Tuple[int, int]]) -> Optional[int]:
        """
        Evaluates whether the centroid dwells inside a toolbar button for >= BUTTON_DWELL_SEC.
        Prevents accidental triggering when hand sweeps across the top bar.
        """
        if pt is None:
            self._dwell_start   = None
            self._dwell_btn_idx = None
            return None

        px, py = pt
        if py > config.TOOLBAR_HEIGHT:
            self._dwell_start   = None
            self._dwell_btn_idx = None
            return None

        for idx, rect in enumerate(self.button_rects):
            if rect_contains(rect, (px, py)):
                if self._dwell_btn_idx == idx:
                    elapsed = time.time() - self._dwell_start
                    if elapsed >= config.BUTTON_DWELL_SEC:
                        self._dwell_start   = None
                        self._dwell_btn_idx = None
                        return idx
                else:
                    self._dwell_btn_idx = idx
                    self._dwell_start   = time.time()
                return None

        self._dwell_start   = None
        self._dwell_btn_idx = None
        return None

    def activate_button(self, idx: int) -> str:
        """Executes corresponding action for the activated palette button."""
        label, color = config.PALETTE_BUTTONS[idx]

        if label == "Clear":
            self.clear()
            return "Canvas cleared"
        elif label == "Eraser":
            self.tool = "eraser"
            return "Tool: Eraser"
        elif label == "Shape":
            return "shape_trigger"
        else:
            self.color = color
            self.tool  = "draw"
            return f"Color: {label}"
