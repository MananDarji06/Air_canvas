"""
config.py – Central configuration for Air Canvas.

All constants and hyperparameters live here. No magic numbers scattered in the code.
Supports both Fingertip Tracking (default) and Colored Marker Tracking.
"""

# ---------------------------------------------------------------------------
# Camera / Frame Settings
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0            # 0 is the default webcam
FRAME_WIDTH  = 640          # Downscaled resolution for real-time 30+ FPS performance
FRAME_HEIGHT = 480

# ---------------------------------------------------------------------------
# Tracking Mode: "hand" (Fingertip) or "marker" (Colored Cap)
# ---------------------------------------------------------------------------
DEFAULT_TRACKING_MODE = "hand"   # "hand" for fingertip pointing, "marker" for colored cap

# ---------------------------------------------------------------------------
# Skin Color Segmentation (YCrCb + HSV) - Classical IPCV
# In YCrCb: Cr (Red-difference chroma), Cb (Blue-difference chroma)
# Skin cluster: Cr in [133, 173], Cb in [77, 127]
# ---------------------------------------------------------------------------
SKIN_YCRCB_LOWER = (0, 133, 77)
SKIN_YCRCB_UPPER = (255, 173, 127)

# HSV Skin backup/refinement bounds
SKIN_HSV_LOWER = (0, 30, 60)
SKIN_HSV_UPPER = (25, 255, 255)

# Colored marker default HSV (Green)
DEFAULT_HSV_LOWER = (35, 80, 80)
DEFAULT_HSV_UPPER = (85, 255, 255)

# Calibration patch half-size (pixels)
CALIB_PATCH_HALF = 12       # ~25x25 patch
CALIB_PERCENTILE_LO = 5     # 5th percentile
CALIB_PERCENTILE_HI = 95    # 95th percentile
CALIB_MARGIN = 12           # Safety margin
CALIB_FILE = "calibration.json"

# ---------------------------------------------------------------------------
# Hand & Fingertip Tracking Parameters (Exp 7: Contours, Convex Hull & Defects)
# ---------------------------------------------------------------------------
MIN_HAND_AREA        = 3000     # Min contour area for hand blob (px^2)
MIN_CONTOUR_AREA     = 400      # Min contour area for marker cap
PEN_LOST_FRAMES      = 10       # Disconnect stroke after lost for N frames
DEFECT_MIN_DEPTH     = 20.0     # Minimum convexity defect depth in pixels (inter-finger valleys)
FINGERTIP_ANGLE_MAX  = 80.0     # Max vertex angle (degrees) to qualify as a finger apex

# ---------------------------------------------------------------------------
# Morphological Kernel Settings (Exp 6)
# ---------------------------------------------------------------------------
MORPH_KERNEL_INIT = 5       # Initial kernel size (must be odd)
MORPH_KERNEL_MIN  = 3
MORPH_KERNEL_MAX  = 21

# ---------------------------------------------------------------------------
# Drawing & Canvas Settings (Exp 8)
# ---------------------------------------------------------------------------
BRUSH_THICKNESS_INIT = 6
BRUSH_THICKNESS_MIN  = 1
BRUSH_THICKNESS_MAX  = 40
ERASER_THICKNESS     = 36   # Wide stroke for erasing

BLUR_KERNEL = (5, 5)        # 5x5 Gaussian blur for spatial noise reduction

# ---------------------------------------------------------------------------
# Trajectory Smoothing (Exp 4: Frequency Domain)
# ---------------------------------------------------------------------------
TRAJ_BUFFER_LEN  = 32       # Centroid ring buffer length
MA_WINDOW        = 7        # Moving average window size
FFT_CUTOFF_HZ    = 5.0      # Low-pass filter cutoff in Hz (rejects high-frequency hand jitter)
FPS_ESTIMATE     = 30.0     # Nominal frame rate for FFT frequency calculation

# ---------------------------------------------------------------------------
# Shape Recognition (Exp 4, 5, 7, 10)
# ---------------------------------------------------------------------------
APPROX_EPSILON_RATIO = 0.02 # Epsilon = 0.02 * perimeter (Douglas-Peucker)
CIRCULARITY_THRESH   = 0.78 # Circularity >= 0.78 indicates circle
FOURIER_N            = 64   # Number of resampled points for Fourier descriptors

# Shi-Tomasi Corner Detector (Exp 10)
SHITOMASI_MAX_CORNERS  = 50
SHITOMASI_QUALITY      = 0.01
SHITOMASI_MIN_DISTANCE = 10

# ---------------------------------------------------------------------------
# Toolbar & ROI Palette Layout (Exp 8)
# ---------------------------------------------------------------------------
TOOLBAR_HEIGHT   = 60       # Height of the top toolbar in pixels
BUTTON_MARGIN    = 4        # Gap between adjacent buttons
BUTTON_DWELL_SEC = 0.5      # Dwell time required to activate a button

# Palette button definitions: (Label, BGR draw-color or None for tools)
PALETTE_BUTTONS = [
    ("Blue",   (255, 120,  30)),
    ("Green",  ( 30, 220,  30)),
    ("Red",    ( 30,  30, 240)),
    ("Yellow", ( 30, 230, 230)),
    ("Purple", (220,  40, 200)),
    ("Eraser", (  0,   0,   0)),
    ("Clear",  None),
    ("Shape",  None),
]

# ---------------------------------------------------------------------------
# Overlay Blending (Exp 11)
# ---------------------------------------------------------------------------
OVERLAY_ALPHA = 0.75        # Frame weight in Method A (addWeighted)
OVERLAY_BETA  = 0.25        # Canvas weight in Method A

# ---------------------------------------------------------------------------
# Export & Compression Assessment (Exp 12)
# ---------------------------------------------------------------------------
EXPORT_DIR     = "exports"
JPEG_QUALITIES = [95, 75, 50, 25, 10]

# ---------------------------------------------------------------------------
# Affine Transformations (Exp 9)
# ---------------------------------------------------------------------------
ROTATE_STEP_DEG = 5.0       # Rotation increment in degrees
SCALE_STEP      = 0.05      # Scale step increment
TRANSLATE_STEP  = 10        # Translation step in pixels

# ---------------------------------------------------------------------------
# Point Processing: Gamma & CLAHE (Exp 1 & 3)
# ---------------------------------------------------------------------------
GAMMA_INIT       = 1.0
GAMMA_STEP       = 0.1
GAMMA_MIN        = 0.1
GAMMA_MAX        = 3.0
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE  = (8, 8)

# ---------------------------------------------------------------------------
# Typography & UI Colors
# ---------------------------------------------------------------------------
FONT       = 0              # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICK = 1
INFO_COLOR = (240, 240, 240)
WARN_COLOR = ( 50, 150, 255)
