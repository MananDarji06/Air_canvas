# Air Canvas – Fingertip Pointing & Classical Computer Vision (IPCV)

A real-time "Air Canvas" application developed in Python for the SEM-5 Image Processing and Computer Vision (IPCV) lab curriculum.

Users draw gestures in the air using **fingertip pointing** and **hand gestures** in front of a webcam, and pick colors/tools by **pointing at the top toolbar**.

> **Classical IPCV Only**: Built strictly with OpenCV, NumPy, Matplotlib, and SciPy. Uses **NO** MediaPipe, deep learning, or external hand-tracking packages.

---

## 1. Quick Setup & Run

Open PowerShell in the `air_canvas` folder:

```powershell
cd C:\Users\User\Desktop\DJSCE\SEM-5\IPCV\Mini_project\air_canvas
pip install -r requirements.txt
python main.py
```

---

## 2. How to Use: Fingertip & Gestures

### ☝️ Drawing Mode (1 Finger Extended)
- Point your **index finger** into the air below the toolbar.
- The green cursor locks onto your fingertip and draws smooth lines on the canvas.

### ✌️ Hover / Navigation Mode (2+ Fingers Extended or Peace Sign)
- Raise 2 fingers (peace sign) or open your palm.
- The cursor turns yellow/cyan: you can reposition your hand across the screen **without drawing any ink**.

### 👆 Color & Tool Picking (Finger Pointing)
- Move your pointing finger into the top toolbar.
- Hover your fingertip inside any button (**Blue**, **Green**, **Red**, **Yellow**, **Eraser**, **Clear**, **Shape**) for **0.5 seconds**.
- The button activates automatically!

### 🎯 Calibration
- If room lighting changes your skin appearance, press **`c`** and left-click on your hand/fingertip in the video window to instantly calibrate the skin color segmenter.

---

## 3. Keyboard Shortcuts

| Key | Description | IPCV Concept Demonstrated |
| :--- | :--- | :--- |
| **`m`** | Toggle tracker: **Hand Fingertip** $\leftrightarrow$ **Marker Cap** | Object detection & color spaces |
| **`d`** | Manual override toggle for Draw / Hover mode | Tracking state override |
| **`c`** | **Click-to-Calibrate** skin or marker color | Exp 3: Histogram analysis & percentiles |
| **`+` / `-`** | Increase / Decrease brush thickness | Geometric rendering |
| **`[` / `]`** | Decrease / Increase morphology kernel size | Exp 6: Elliptical structuring elements |
| **`h`** | Toggle CLAHE on V channel | Exp 3: Contrast Limited Adaptive Hist. Eq. |
| **`b`** | Toggle overlay Method A vs Method B | Exp 11: `addWeighted` vs Bitwise masking |
| **`f`** | Toggle FFT Low-Pass / Moving Average | Exp 4: 1D Discrete Fourier Transform |
| **`p`** | Display Trajectory & Spectrum Plot | Exp 4: Magnitude spectrum $\|X(f)\|$ |
| **`s`** | Recognize shape of last stroke | Exp 5, 7, 10: Douglas-Peucker, Shi-Tomasi |
| **`e`** | Export PNG/JPEG sweep & PSNR Report | Exp 12: Rate-distortion, DCT artifacts |
| **`v`** | Toggle diagnostic pipeline debug windows | Exp 1, 6, 8: Intermediate masks & canvas |
| **`n`** | Toggle Digital Negative | Exp 1: Point processing $s = 255 - r$ |
| **`,` / `.`** | Adjust Gamma LUT (decrease / increase) | Exp 1: Power-law transformation $s = c r^\gamma$ |
| **`z` / `x`** | Rotate canvas CCW / CW | Exp 9: 2x3 Affine rotation matrix |
| **`u` / `j`** | Scale canvas Up / Down | Exp 9: Affine zoom matrix |
| **`w` / `a`** | Translate canvas Up / Left | Exp 9: Affine translation vector |
| **`r`** | Reset canvas affine transformations | Exp 9: Identity matrix |
| **`?`** | On-screen interactive help overlay | Alpha blending HUD |
| **`q` / `ESC`** | Quit application safely | Resource deallocation |

---

## 4. IPCV Lab Syllabus Mapping Table

| Lab Experiment | Topic & Curriculum Title | Implementation in Codebase | Key Functions |
| :--- | :--- | :--- | :--- |
| **Exp 1** | **Point Processing & Color Spaces** | YCrCb + HSV Skin Segmentation; Gamma correction LUT; Digital Negative | `tracker.skin_mask`<br>`utils.build_gamma_lut`<br>`utils.digital_negative` |
| **Exp 2** | **Spatial Filtering** | 5x5 Gaussian spatial smoothing kernel for sensor noise suppression | `tracker.preprocess` (`cv2.GaussianBlur`) |
| **Exp 3** | **Histogram Processing** | CLAHE on V channel; 3-channel patch histograms with 5th/95th percentile bounds | `calibration.compute_hsv_bounds`<br>`calibration.show_patch_histograms`<br>`tracker.preprocess` |
| **Exp 4** | **Frequency Domain** | 1D Real FFT trajectory low-pass filter; Elliptic Fourier descriptors for shape | `trajectory.fft_lowpass`<br>`trajectory.show_trajectory_plot`<br>`shapes.fourier_descriptors` |
| **Exp 5** | **Edge Detection** | Canny multi-stage edge detector (Sobel gradients, non-max suppression, hysteresis) | `shapes.recognise_and_replace` (`cv2.Canny`) |
| **Exp 6** | **Morphological Ops** | Opening (erosion then dilation) and Closing (dilation then erosion) with ellipse SE | `tracker.morphological_cleanup` |
| **Exp 7** | **Object Detection** | External contours, Distance Transform for palm center, Convex Hull, moments | `tracker.find_hand_and_fingertip`<br>`shapes.classify_shape` (`approxPolyDP`) |
| **Exp 8** | **NumPy / OpenCV Array** | Pure NumPy uint8 canvas array, toolbar ROI coordinate bounding checks, metadata HUD | `canvas.Canvas`<br>`canvas.check_palette`<br>`main.draw_info_panel` |
| **Exp 9** | **Geometric Transforms** | Horizontal mirror flip; 2x3 affine rotation, scaling, and translation with bilinear warp | `transforms.TransformState`<br>`cv2.warpAffine`<br>`cv2.getRotationMatrix2D` |
| **Exp 10** | **Corner & Defect Detection** | Convexity Defects between fingers; Shi-Tomasi corner detection ($\min(\lambda_1, \lambda_2)$) | `tracker.find_hand_and_fingertip`<br>`shapes.classify_shape` (`goodFeaturesToTrack`) |
| **Exp 11** | **Arithmetic & Bitwise** | Linear weighted blend vs. bitwise masking + saturating `cv2.add` vs modular NumPy `+` | `canvas.overlay_on_frame`<br>`utils.demo_arithmetic_overflow` |
| **Exp 12** | **Image Compression** | PNG vs JPEG quality sweep (Q=95 to 10), PSNR computation, DCT ringing & blocking artifacts | `export.compression_report`<br>`export.compute_psnr`<br>`export._show_edge_artifacts` |

---

## 5. Fingertip Tracking Pipeline (Classical IPCV)

```
Webcam Frame (640x480)
   │
   ├─► cv2.flip(1) [Horizontal Mirror]
   │
   ├─► cv2.GaussianBlur(5x5) [Noise Suppression]
   │
   ├─► Skin Color Segmentation [Exp 1]:
   │     • YCrCb Space: 133 <= Cr <= 173 and 77 <= Cb <= 127
   │     • HSV Space: 0 <= H <= 25 and 30 <= S <= 255
   │     • Combined Binary Skin Mask
   │
   ├─► Morphological Opening -> Closing (Elliptical SE) [Exp 6]
   │
   ├─► cv2.findContours() -> Hand Contour [Exp 7]
   │     ├─► cv2.distanceTransform() -> Palm Center (Largest inscribed circle)
   │     ├─► cv2.convexHull() & cv2.convexityDefects() [Exp 7 & 10]
   │     ├─► Inter-finger Valley Angles (Law of Cosines) -> Finger Count
   │     └─► Apex Contour Search -> Fingertip Coordinates (fx, fy)
   │
   ├─► Gesture State Machine:
   │     • fx, fy in Top Toolbar -> Pointing/Dwell Selection Mode
   │     • 1 Finger Extended -> DRAWING MODE (ink trail)
   │     • 2+ Fingers Extended -> HOVER MODE (reposition without ink)
   │
   ├─► Trajectory Buffer -> Moving Avg / 1D FFT Low-Pass Filter [Exp 4]
   │
   ├─► Canvas Drawing & Top Toolbar Selection [Exp 8]
   │
   └─► Frame Composition (addWeighted / Bitwise Masking) [Exp 11]
```
