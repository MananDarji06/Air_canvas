"""
main.py – Real-time Air Canvas with Fingertip Pointing & Hand Gesture Controls.

Classical IPCV Features:
  - Fingertip tracking via YCrCb/HSV Skin Segmentation & Convex Hull/Defects (Exp 7 & 10)
  - Color / tool selection via finger pointing & dwell detection in toolbar ROI (Exp 8)
  - Natural gesture switching:
      * 1 finger pointed (index) -> DRAWING mode
      * 2+ fingers extended (peace sign/open hand) -> HOVER / NAVIGATION mode (move without drawing)
  - Marker cap tracking mode fallback (toggle with key 'm')
  - VideoCapture & mirror flip (Exp 9)
  - Trajectory smoothing via 1D FFT Low-Pass / Moving Average (Exp 4)
  - Overlay blending: Method A (Weighted) or Method B (Bitwise masking) (Exp 11)
  - Point processing: Digital Negative & Gamma LUT (Exp 1)
  - Shape recognition & Canny edges (Exp 4, 5, 7, 10)
  - 2x3 Affine transformations (Exp 9)
  - JPEG rate-distortion & PSNR evaluation report (Exp 12)
"""

import sys
from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np

import config
import utils
import tracker
import calibration
import canvas as canvas_module
import trajectory
import shapes
import export
import transforms


# ---------------------------------------------------------------------------
# Interactive Help Overlay (Key '?')
# ---------------------------------------------------------------------------

HELP_LINES = [
    "AIR CANVAS - FINGERTIP POINTING & GESTURE CONTROLS",
    "--------------------------------------------------",
    "FINGER GESTURES:",
    "  * 1 Finger Pointed  : DRAWING mode (leaves ink trail)",
    "  * 2+ Fingers (Peace): HOVER mode (move without ink)",
    "  * Point at Toolbar  : Hover finger over button (0.5s) to pick color/tool",
    "--------------------------------------------------",
    "KEYBOARD SHORTCUTS:",
    "q / ESC   : Exit and release camera",
    "m         : Toggle Tracking Mode (Hand Fingertip <-> Marker Cap)",
    "d         : Manual override toggle for Draw / Hover mode",
    "+ / -     : Increase / Decrease brush thickness",
    "[ / ]     : Decrease / Increase morphology kernel size",
    "c         : Calibrate skin / marker color (click on finger)",
    "h         : Toggle CLAHE on V channel (Exp 3)",
    "b         : Toggle overlay mode (A: addWeighted / B: Bitwise)",
    "f         : Toggle FFT Low-Pass / Moving Average (Exp 4)",
    "p         : Plot trajectory signals & magnitude spectrum",
    "s         : Shape recognition on last stroke (circle, tri, rect)",
    "e         : Export PNG/JPEG sweep & PSNR report (Exp 12)",
    "v         : Toggle diagnostic debug windows (Exp 1 & 6)",
    "n         : Toggle Digital Negative (Exp 1)",
    ", / .     : Decrease / Increase Gamma LUT (Exp 1)",
    "z / x     : Rotate canvas CCW / CW (Exp 9 Affine)",
    "u / j     : Scale canvas Up / Down (Exp 9 Affine)",
    "w / a     : Translate canvas Up / Left (Exp 9)",
    "r         : Reset canvas affine transformations",
    "?         : Toggle this help screen overlay",
]

HELP_PANEL_W = 510
HELP_PANEL_H = len(HELP_LINES) * 19 + 25


def draw_help_overlay(frame: np.ndarray) -> None:
    """Renders semi-transparent backdrop panel with gesture & keybinding guide."""
    h, w = frame.shape[:2]
    px = max(0, (w - HELP_PANEL_W) // 2)
    py = max(0, (h - HELP_PANEL_H) // 2)

    roi = frame[py:py + HELP_PANEL_H, px:px + HELP_PANEL_W]
    dark_mask = np.zeros_like(roi)
    cv2.addWeighted(roi, 0.25, dark_mask, 0.75, 0, roi)
    frame[py:py + HELP_PANEL_H, px:px + HELP_PANEL_W] = roi

    for idx, line in enumerate(HELP_LINES):
        color = (50, 220, 255) if idx == 0 else config.INFO_COLOR
        if line.startswith("  *"):
            color = (120, 255, 120)
        font_scale = 0.43 if idx > 0 else 0.47
        thickness = 1 if idx > 0 else 2
        utils.put_text(frame, line, (px + 12, py + 22 + idx * 19),
                       color=color, scale=font_scale, thickness=thickness)


# ---------------------------------------------------------------------------
# Assessment HUD Panel (Exp 8: Image Metadata & Properties)
# ---------------------------------------------------------------------------

def draw_info_panel(frame: np.ndarray,
                    fps: float,
                    tracking_mode: str,
                    finger_count: int,
                    mode_str: str,
                    clahe_on: bool,
                    overlay_method: str,
                    gamma: float,
                    fft_on: bool,
                    kernel_size: int,
                    thickness: int,
                    negative_on: bool) -> None:
    """Overlays live image properties, frame dimensions, and pipeline status."""
    H, W, C = frame.shape
    mean_lum = int(np.mean(frame))

    lines = [
        f"FPS: {fps:.1f} | Frame: {W}x{H}x{C} (uint8) | Mean: {mean_lum}",
        f"Tracker: {tracking_mode.upper()} | Fingers: {finger_count} | Mode: {mode_str}",
        f"Overlay: Method {overlay_method} | CLAHE: {'ON' if clahe_on else 'OFF'}",
        f"Smoother: {'FFT Low-Pass (Exp 4)' if fft_on else 'Moving-Avg'} | Gamma: {gamma:.1f}",
        f"Kernel: {kernel_size}x{kernel_size} | Brush: {thickness}px | Negative: {'ON' if negative_on else 'OFF'}",
    ]

    y_start = H - len(lines) * 16 - 8
    for idx, line in enumerate(lines):
        utils.put_text(frame, line, (8, y_start + idx * 16), scale=0.39, color=config.INFO_COLOR)


# ---------------------------------------------------------------------------
# Main Application Loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Air Canvas – Fingertip Pointing & Classical Hand Gesture Recognition")
    print("=" * 70)

    # 1. Camera Initialization
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Unable to open webcam at index {config.CAMERA_INDEX}.")
        print("Please verify camera connections or modify CAMERA_INDEX in config.py.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("[ERROR] Camera connected but unable to retrieve initial video frame.")
        cap.release()
        sys.exit(1)

    H, W = test_frame.shape[:2]
    print(f"[Init] Video Stream Ready: {W}x{H} pixels")

    # 2. State & Module Allocations
    lo_cal, hi_cal = calibration.load_calibration()
    custom_lower = lo_cal
    custom_upper = hi_cal

    fps_counter  = utils.FPSCounter()
    calib_state  = calibration.CalibrationState()
    canvas       = canvas_module.Canvas(H, W)
    traj_buf     = trajectory.TrajectoryBuffer()
    transform_st = transforms.TransformState()
    gamma_lut    = utils.build_gamma_lut(config.GAMMA_INIT)

    state: Dict[str, Any] = {
        "tracking_mode":  config.DEFAULT_TRACKING_MODE, # "hand" or "marker"
        "draw_mode":      True,                         # Auto-updated by finger gestures
        "manual_override": False,                       # If user pressed 'd'
        "clahe_on":       False,
        "overlay_method": "B",
        "fft_on":         False,
        "debug_on":       False,
        "help_on":        False,
        "negative_on":    False,
        "gamma":          config.GAMMA_INIT,
        "kernel_size":    config.MORPH_KERNEL_INIT,
        "thickness":      config.BRUSH_THICKNESS_INIT,
        "lost_frames":    0,
        "finger_count":   0,
        "shape_info":     None,
        "last_stroke":    None,
        "active_btn_idx": None,
    }

    win_name = "Air Canvas - Fingertip Pointing (IPCV Project)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, calib_state.mouse_callback)

    # Viva prompt: Print arithmetic demonstration
    utils.demo_arithmetic_overflow()

    print("[Ready] Point finger into video to draw. Point at top toolbar to select colors/tools.")
    print("        Press '?' for controls guide. Press 'q' or ESC to exit.\n")

    # -----------------------------------------------------------------------
    # Continuous Processing Pipeline
    # -----------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Exp 9: Geometric horizontal mirror flip so interaction mirrors user movements
        frame = cv2.flip(frame, 1)

        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))

        fps = fps_counter.tick()

        # Step 1: Preprocessing -> Blur -> HSV conversion -> (Optional CLAHE) [Exp 2 & 3]
        hsv = tracker.preprocess(frame, clahe_on=state["clahe_on"])

        # Handle mouse calibration click if user clicked on skin or marker
        if calib_state.clicked:
            custom_lower, custom_upper = calibration.process_click(calib_state, hsv)

        # Step 2: Binary Segmentation based on tracking mode
        if state["tracking_mode"] == "hand":
            # Skin Color Segmentation (YCrCb + HSV) - Exp 1
            raw_mask = tracker.skin_mask(frame, custom_lower, custom_upper)
            clean_mask = tracker.morphological_cleanup(raw_mask, kernel_size=state["kernel_size"])

            # Step 3: Fingertip & Hand Detection (Exp 7: Contours & Exp 10: Convexity Defects)
            fx, fy, palm_center, finger_count, hand_cnt, hull_pts, area = tracker.find_hand_and_fingertip(clean_mask)
            state["finger_count"] = finger_count
            pt_x, pt_y = fx, fy
        else:
            # Marker Cap Tracking Mode
            m_lo = custom_lower if custom_lower is not None else config.DEFAULT_HSV_LOWER
            m_hi = custom_upper if custom_upper is not None else config.DEFAULT_HSV_UPPER
            raw_mask = tracker.colour_mask(hsv, m_lo, m_hi)
            clean_mask = tracker.morphological_cleanup(raw_mask, kernel_size=state["kernel_size"])

            cx, cy, area, marker_cnt = tracker.find_marker(clean_mask)
            pt_x, pt_y = cx, cy
            state["finger_count"] = 1 if cx is not None else 0
            hand_cnt, hull_pts, palm_center = None, None, None

        # Step 4: Gesture-Based Drawing / Hover Mode Decision
        if pt_x is not None and pt_y is not None:
            state["lost_frames"] = 0

            # When finger points inside toolbar: strictly HOVER / POINTING (no drawing)
            if pt_y <= config.TOOLBAR_HEIGHT:
                is_currently_drawing = False
            else:
                # Inside drawing canvas:
                # 1 Finger pointing -> DRAWING
                # 2+ Fingers (Peace sign / Open palm) -> HOVERING without ink
                if not state["manual_override"]:
                    if state["tracking_mode"] == "hand":
                        is_currently_drawing = (state["finger_count"] <= 1)
                    else:
                        is_currently_drawing = state["draw_mode"]
                else:
                    is_currently_drawing = state["draw_mode"]

            traj_buf.push(pt_x, pt_y)
        else:
            state["lost_frames"] += 1
            is_currently_drawing = False

        state["draw_mode"] = is_currently_drawing

        # Temporal trajectory smoothing (Exp 4: Moving Avg or FFT Low-Pass)
        sx, sy = traj_buf.last_smoothed(use_fft=state["fft_on"], window=config.MA_WINDOW)

        # Step 5: Palette Hit-Testing & Dwell Activation via Finger Pointing (Exp 8)
        pen_pt = (sx, sy) if sx is not None and sy is not None else None
        activated_btn = canvas.check_palette(pen_pt)
        state["active_btn_idx"] = canvas._dwell_btn_idx

        if activated_btn is not None:
            action_msg = canvas.activate_button(activated_btn)
            if action_msg == "shape_trigger":
                target_stroke = state["last_stroke"] or (canvas.strokes[-1] if canvas.strokes else None)
                if target_stroke is not None:
                    state["shape_info"] = shapes.recognise_and_replace(target_stroke, canvas)
            else:
                state["shape_info"] = None

        # Step 6: Interactive Canvas Drawing
        canvas.thickness = state["thickness"]

        if is_currently_drawing and sx is not None and sy is not None:
            if sy > config.TOOLBAR_HEIGHT:
                if state["lost_frames"] == 0:
                    canvas.continue_stroke((sx, sy))
                else:
                    if canvas.active_stroke is not None:
                        state["last_stroke"] = canvas.end_stroke()
                    canvas.begin_stroke((sx, sy))
        else:
            if state["lost_frames"] > config.PEN_LOST_FRAMES or not is_currently_drawing:
                if canvas.active_stroke is not None:
                    state["last_stroke"] = canvas.end_stroke()
                if state["lost_frames"] > config.PEN_LOST_FRAMES:
                    traj_buf.clear()

        # Step 7: Composition & Overlay Blending (Exp 9 & 11)
        # Apply affine coordinate transformation to visual canvas surface
        transformed_canvas = transform_st.apply(canvas.surface)

        original_surface = canvas.surface
        canvas.surface   = transformed_canvas
        display_frame    = canvas.overlay_on_frame(frame, method=state["overlay_method"])
        canvas.surface   = original_surface

        # Overlay cursor and tracking feedback
        if state["tracking_mode"] == "hand":
            tracker.draw_hand_overlay(display_frame, sx, sy, palm_center,
                                      state["finger_count"], hand_cnt, hull_pts,
                                      is_drawing=is_currently_drawing)
        else:
            tracker.draw_marker_overlay(display_frame, sx, sy, hand_cnt)

        # Render top interactive ROI toolbar
        canvas.draw_toolbar(display_frame, active_idx=state["active_btn_idx"])

        # State notifications & alerts
        if calib_state.active:
            utils.put_text(display_frame, "CALIBRATION ACTIVE: Click on your fingertip or marker in window",
                           (12, config.TOOLBAR_HEIGHT + 24), color=config.WARN_COLOR, scale=0.55, thickness=2)

        mode_tag = "DRAWING (1 Finger)" if is_currently_drawing else "HOVER / POINTING (2+ Fingers)"
        mode_col = (50, 255, 50) if is_currently_drawing else (255, 200, 50)
        utils.put_text(display_frame, f"[{mode_tag}]", (W - 270, config.TOOLBAR_HEIGHT + 22), color=mode_col, scale=0.48)

        if state["shape_info"]:
            sh_info = state["shape_info"]
            utils.put_text(
                display_frame,
                f"Shape: {sh_info.get('shape','').upper()} | Circ: {sh_info.get('circularity',0)} | Verts: {sh_info.get('vertices',0)}",
                (12, config.TOOLBAR_HEIGHT + 44), color=(50, 230, 255), scale=0.48
            )

        # Live metadata assessment HUD
        draw_info_panel(
            display_frame, fps, state["tracking_mode"], state["finger_count"], mode_tag,
            state["clahe_on"], state["overlay_method"],
            state["gamma"], state["fft_on"],
            state["kernel_size"], state["thickness"],
            state["negative_on"]
        )

        # Exp 1: Point processing (Gamma correction LUT)
        display_frame = utils.apply_gamma(display_frame, gamma_lut)

        # Exp 1: Point processing (Digital Negative inversion)
        if state["negative_on"]:
            display_frame = utils.digital_negative(display_frame)

        # Help overlay
        if state["help_on"]:
            draw_help_overlay(display_frame)

        # Step 8: Diagnostic Pipeline Windows (Exp 1 & 6)
        if state["debug_on"]:
            cv2.imshow("(a) Raw Skin/Color Mask (Exp 1)", raw_mask)
            cv2.imshow("(b) Cleaned Mask (Opening+Closing - Exp 6)", clean_mask)
            cv2.imshow("(c) Pure Canvas Layer (Exp 8)", canvas.surface)

        # Step 9: Render Output Window
        cv2.imshow(win_name, display_frame)

        # Step 10: Event & Key Dispatch
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            print("\n[Exit] User quit application.")
            break
        elif key == ord("m"):
            state["tracking_mode"] = "marker" if state["tracking_mode"] == "hand" else "hand"
            print(f"[Mode] Switched Tracker to: {state['tracking_mode'].upper()}")
        elif key == ord("d"):
            state["manual_override"] = True
            state["draw_mode"] = not state["draw_mode"]
            print(f"[Manual] Draw Mode Forced: {'ON' if state['draw_mode'] else 'OFF'}")
        elif key in (ord("+"), ord("=")):
            state["thickness"] = utils.clamp(state["thickness"] + 1, config.BRUSH_THICKNESS_MIN, config.BRUSH_THICKNESS_MAX)
        elif key == ord("-"):
            state["thickness"] = utils.clamp(state["thickness"] - 1, config.BRUSH_THICKNESS_MIN, config.BRUSH_THICKNESS_MAX)
        elif key == ord("]"):
            state["kernel_size"] = utils.clamp(state["kernel_size"] + 2, config.MORPH_KERNEL_MIN, config.MORPH_KERNEL_MAX)
        elif key == ord("["):
            state["kernel_size"] = utils.clamp(state["kernel_size"] - 2, config.MORPH_KERNEL_MIN, config.MORPH_KERNEL_MAX)
        elif key == ord("c"):
            calib_state.enable()
        elif key == ord("h"):
            state["clahe_on"] = not state["clahe_on"]
            print(f"[Toggle] CLAHE: {'ENABLED' if state['clahe_on'] else 'DISABLED'}")
        elif key == ord("b"):
            state["overlay_method"] = "A" if state["overlay_method"] == "B" else "B"
            print(f"[Toggle] Overlay Method: {state['overlay_method']}")
        elif key == ord("f"):
            state["fft_on"] = not state["fft_on"]
            print(f"[Toggle] Trajectory Smoother: {'FFT Low-Pass (Exp 4)' if state['fft_on'] else 'Moving Average'}")
        elif key == ord("p"):
            if len(traj_buf) >= 6:
                trajectory.show_trajectory_plot(traj_buf.xs)
            else:
                print("[Info] Insufficient trajectory points. Draw a stroke first.")
        elif key == ord("s"):
            target_stroke = state["last_stroke"] or (canvas.strokes[-1] if canvas.strokes else None)
            if target_stroke is not None:
                state["shape_info"] = shapes.recognise_and_replace(target_stroke, canvas)
            else:
                print("[Info] No completed stroke available to classify.")
        elif key == ord("e"):
            export.compression_report(canvas.surface)
        elif key == ord("v"):
            state["debug_on"] = not state["debug_on"]
            if not state["debug_on"]:
                for name in ["(a) Raw Skin/Color Mask (Exp 1)", "(b) Cleaned Mask (Opening+Closing - Exp 6)", "(c) Pure Canvas Layer (Exp 8)"]:
                    try:
                        cv2.destroyWindow(name)
                    except Exception:
                        pass
        elif key == ord("n"):
            state["negative_on"] = not state["negative_on"]
            print(f"[Toggle] Digital Negative: {'ENABLED' if state['negative_on'] else 'DISABLED'}")
        elif key == ord(","):
            state["gamma"] = utils.clamp(round(state["gamma"] - config.GAMMA_STEP, 2), config.GAMMA_MIN, config.GAMMA_MAX)
            gamma_lut = utils.build_gamma_lut(state["gamma"])
            print(f"[Setting] Gamma: {state['gamma']}")
        elif key == ord("."):
            state["gamma"] = utils.clamp(round(state["gamma"] + config.GAMMA_STEP, 2), config.GAMMA_MIN, config.GAMMA_MAX)
            gamma_lut = utils.build_gamma_lut(state["gamma"])
            print(f"[Setting] Gamma: {state['gamma']}")
        elif key == ord("?"):
            state["help_on"] = not state["help_on"]
        # Affine Transformations (Exp 9)
        elif key == ord("z"):
            transform_st.rotate(config.ROTATE_STEP_DEG)
        elif key == ord("x"):
            transform_st.rotate(-config.ROTATE_STEP_DEG)
        elif key == ord("u"):
            transform_st.zoom(config.SCALE_STEP)
        elif key == ord("j"):
            transform_st.zoom(-config.SCALE_STEP)
        elif key == ord("w"):
            transform_st.translate(0, -config.TRANSLATE_STEP)
        elif key == ord("a"):
            transform_st.translate(-config.TRANSLATE_STEP, 0)
        elif key == ord("r"):
            transform_st.reset()
            print("[Reset] Canvas affine transformations cleared.")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
