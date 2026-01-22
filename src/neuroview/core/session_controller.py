"""
SessionController (MVP - Vector Tracking)

- Video/camera source
- ROI provided by GUI (no OpenCV windows)
- Tracking via body_tracking.py (Contour-PCA based)
- Behaviors:
    - Freezing: low center motion + low heading variance (windowed)
    - Grooming: stable center + head oscillation (windowed)
- Zones:
    - Open Field: Center vs Periphery (ratio-based inner rectangle)
- Export: CSV/PDF
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List, Tuple
from collections import deque
import time
import math
import csv

import cv2
import numpy as np

from neuroview.core.body_tracking import BodyState, extract_body_state_from_mask

SourceType = Literal["video", "camera"]
ROIType = Tuple[int, int, int, int]  # (x, y, w, h)


# -----------------------------------------------------------------------------
# Behavior configuration (tunable)
# -----------------------------------------------------------------------------

FREEZE_MIN_DURATION_S = 2.0
FREEZE_WINDOW_S = 2.0
FREEZE_CENTER_DISP_PX = 3.0               # max displacement in window
FREEZE_HEADING_VAR_DEG2 = 25.0            # variance threshold (deg^2)

GROOM_MIN_DURATION_S = 1.5
GROOM_WINDOW_S = 1.5
GROOM_CENTER_DISP_PX = 5.0                # stays mostly in place
GROOM_HEAD_OSC_PX = 8.0                   # head moves around while center stays
GROOM_HEADING_VAR_MIN_DEG2 = 40.0         # heading variance typically higher during grooming

MOVEMENT_SPEED_THRESHOLD_PX_S = 3.0

ZONE_CENTER_AREA_RATIO = 0.55


@dataclass
class Metrics:
    elapsed_time_s: float = 0.0
    total_distance_px: float = 0.0
    instantaneous_speed_px_per_s: float = 0.0
    average_speed_px_per_s: float = 0.0

    current_zone: str = "-"
    time_center_s: float = 0.0
    time_periphery_s: float = 0.0

    movement_time_s: float = 0.0
    immobility_time_s: float = 0.0

    freezing_active: bool = False
    grooming_active: bool = False
    freezing_time_s: float = 0.0
    grooming_time_s: float = 0.0

    frames_processed: int = 0


class ZoneModel:
    """
    Simple Open Field zone model: Center vs Periphery based on inner rectangle area ratio.
    """
    def __init__(self, roi: ROIType, center_area_ratio: float = ZONE_CENTER_AREA_RATIO):
        rx, ry, rw, rh = roi
        self.rx, self.ry, self.rw, self.rh = rx, ry, rw, rh
        s = math.sqrt(center_area_ratio)
        cw = rw * s
        ch = rh * s
        self.x1 = (rw - cw) / 2.0
        self.y1 = (rh - ch) / 2.0
        self.x2 = self.x1 + cw
        self.y2 = self.y1 + ch

    def classify(self, x_global: float, y_global: float) -> str:
        lx = x_global - self.rx
        ly = y_global - self.ry
        if self.x1 <= lx <= self.x2 and self.y1 <= ly <= self.y2:
            return "Center"
        return "Periphery"


class SessionController:
    def __init__(self) -> None:
        self.video_fps: float = 30.0
        self.frame_interval_s: float = 1.0 / self.video_fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.source: Optional[SourceType] = None

        self.running: bool = False
        self.paused: bool = False

        self.start_time: Optional[float] = None
        self._last_t: Optional[float] = None

        self.roi: Optional[ROIType] = None
        self.roi_defined: bool = False
        self.zone_model: Optional[ZoneModel] = None

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        self.last_frame: Optional[np.ndarray] = None
        self.metrics = Metrics()

        # Tracking
        self.prev_body: Optional[BodyState] = None
        self.last_body: Optional[BodyState] = None

        # Histories (window buffers)
        self.center_buf: deque[Tuple[Tuple[int, int], float]] = deque()
        self.head_buf: deque[Tuple[Tuple[int, int], float]] = deque()
        self.heading_buf: deque[Tuple[float, float]] = deque()

        self.tracking_records: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # ROI API
    # -------------------------------------------------------------------------
    def set_roi(self, roi: ROIType) -> None:
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise ValueError("Invalid ROI.")
        self.roi = (int(x), int(y), int(w), int(h))
        self.roi_defined = True
        self.zone_model = ZoneModel(self.roi)
        self._reset_runtime_stats()

    def clear_roi(self) -> None:
        self.roi = None
        self.roi_defined = False
        self.zone_model = None
        self._reset_runtime_stats()

    # -------------------------------------------------------------------------
    # Sources
    # -------------------------------------------------------------------------
    def release_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_video(self, path: str) -> None:
        self.release_capture()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 240:
            fps = 30.0
        self.video_fps = float(fps)
        self.frame_interval_s = 1.0 / self.video_fps

        self.cap = cap
        self.source = "video"
        self.reset_for_new_source()

    def open_camera(self, index: int = 0) -> None:
        self.release_capture()
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        self.cap = cap
        self.source = "camera"
        self.video_fps = 30.0
        self.frame_interval_s = 1.0 / self.video_fps
        self.reset_for_new_source()

    def reset_for_new_source(self) -> None:
        self.running = False
        self.paused = False
        self.start_time = None
        self._last_t = None

        self._reset_runtime_stats()

        self.prev_body = None
        self.last_body = None
        self.last_frame = None
        self.tracking_records.clear()

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    def reset_video_position(self) -> None:
        """
        Reset playback to frame 0 (video only) and refresh first frame.
        """
        if self.source != "video" or self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.prev_body = None
        self.last_body = None
        self._reset_runtime_stats()
        self.tracking_records.clear()
        self.read_first_frame()

    def read_first_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        if self.source == "video":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        self.last_frame = frame.copy()
        return frame

    # -------------------------------------------------------------------------
    # Run control
    # -------------------------------------------------------------------------
    def start(self) -> None:
        if self.cap is None:
            raise RuntimeError("No source opened.")
        if not self.roi_defined or self.roi is None:
            raise RuntimeError("ROI not defined.")

        # IMPORTANT: do not call reset_for_new_source() here because it clears ROI.
        self.running = True
        self.paused = False
        self.start_time = time.time()
        self._last_t = self.start_time

        # Reset tracking + stats, keep ROI and source intact
        self._reset_runtime_stats()
        self.prev_body = None
        self.last_body = None
        self.tracking_records.clear()

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        if self.source == "video" and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def stop(self) -> None:
        self.running = False
        self.paused = False

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    # -------------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------------
    def export_csv(self, path: str) -> None:
        if not self.tracking_records:
            raise RuntimeError("No data to export.")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.tracking_records[0].keys()))
            w.writeheader()
            w.writerows(self.tracking_records)

    def export_pdf(self, path: str) -> None:
        if not self.tracking_records:
            raise RuntimeError("No data to export.")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        m = self.metrics
        xs = [r["center_x_px"] for r in self.tracking_records]
        ys = [r["center_y_px"] for r in self.tracking_records]

        with PdfPages(path) as pdf:
            # Summary page
            fig = plt.figure(figsize=(8.5, 6))
            txt = (
                "NeuroView Report\n\n"
                f"Elapsed: {m.elapsed_time_s:.2f}s\n"
                f"Distance: {m.total_distance_px:.2f}px\n"
                f"Avg Speed: {m.average_speed_px_per_s:.2f}px/s\n"
                f"Center time: {m.time_center_s:.2f}s\n"
                f"Periphery time: {m.time_periphery_s:.2f}s\n"
                f"Freezing: {m.freezing_time_s:.2f}s\n"
                f"Grooming: {m.grooming_time_s:.2f}s\n"
            )
            plt.text(0.08, 0.8, txt, fontsize=12, family="monospace")
            plt.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

            # Trajectory
            fig2 = plt.figure(figsize=(7, 7))
            plt.plot(xs, ys, linewidth=1.0)
            plt.gca().invert_yaxis()
            plt.title("Trajectory (center)")
            pdf.savefig(fig2)
            plt.close(fig2)

    # -------------------------------------------------------------------------
    # Internal stats/behavior helpers
    # -------------------------------------------------------------------------
    def _reset_runtime_stats(self) -> None:
        self.metrics = Metrics()
        self.center_buf.clear()
        self.head_buf.clear()
        self.heading_buf.clear()

    def _push_window(self, buf: deque, value, now: float, window_s: float) -> None:
        buf.append((value, now))
        while buf and (now - buf[0][1] > window_s):
            buf.popleft()

    def _window_disp(self, buf: deque) -> float:
        """
        Max distance between current and any point in buffer.
        """
        if len(buf) < 2:
            return 0.0
        curr = np.array(buf[-1][0], dtype=np.float32)
        dmax = 0.0
        for (p, _) in buf:
            d = float(np.linalg.norm(curr - np.array(p, dtype=np.float32)))
            if d > dmax:
                dmax = d
        return dmax

    def _window_var(self, buf: deque) -> float:
        if len(buf) < 5:
            return 0.0
        vals = np.array([v for (v, _) in buf], dtype=np.float32)
        return float(np.var(vals))

    def _update_zone_time(self, zone: str, dt: float) -> None:
        if zone == "Center":
            self.metrics.time_center_s += max(0.0, dt)
        elif zone == "Periphery":
            self.metrics.time_periphery_s += max(0.0, dt)

    # -------------------------------------------------------------------------
    # Main processing
    # -------------------------------------------------------------------------
    def process_next_frame(self) -> Optional[np.ndarray]:
        if not self.running or self.paused or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            if self.source == "video":
                self.running = False
            return None

        self.last_frame = frame.copy()
        self.metrics.frames_processed += 1

        now = time.time()
        if self.start_time is not None:
            self.metrics.elapsed_time_s = now - self.start_time

        dt = max(0.0, now - (self._last_t or now))
        self._last_t = now

        # ROI crop
        if self.roi_defined and self.roi is not None:
            rx, ry, rw, rh = self.roi
        else:
            H, W = frame.shape[:2]
            rx, ry, rw, rh = 0, 0, W, H

        roi_frame = frame[ry:ry + rh, rx:rx + rw]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

        fg = self.bg_subtractor.apply(gray)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # --- Body tracking ---
        body = extract_body_state_from_mask(
            frame_bgr=frame,
            roi_xywh=(rx, ry, rw, rh),
            fg_mask_roi=fg,
            prev_state=self.prev_body,
            now_s=now,
            prev_time_s=(self._last_t - dt) if dt > 0 else None,
        )

        # Clean handling: if no detection -> do not update prev_body, do not append buffers
        # This prevents ghost points and broken behaviors.
        if body is None:
            self.last_body = None
            # We still return raw frame for display, but no pose is drawn by GUI.
            return frame

        self.prev_body = body
        self.last_body = body

        # --- Kinematics ---
        speed = 0.0
        if len(self.center_buf) > 0:
            prev_center = np.array(self.center_buf[-1][0], dtype=np.float32)
            curr_center = np.array(body.center_xy, dtype=np.float32)
            step = float(np.linalg.norm(curr_center - prev_center))
            self.metrics.total_distance_px += step
            if dt > 1e-6:
                speed = step / dt

        self.metrics.instantaneous_speed_px_per_s = float(speed)
        self.metrics.average_speed_px_per_s = (
            self.metrics.total_distance_px / max(1e-6, self.metrics.elapsed_time_s)
        )

        if speed >= MOVEMENT_SPEED_THRESHOLD_PX_S:
            self.metrics.movement_time_s += max(0.0, dt)
        else:
            self.metrics.immobility_time_s += max(0.0, dt)

        # --- Zones ---
        zone = "-"
        if self.zone_model is not None:
            zone = self.zone_model.classify(body.center_xy[0], body.center_xy[1])
        self.metrics.current_zone = zone
        self._update_zone_time(zone, dt)

        # --- Behavior buffers (only when detection is valid) ---
        self._push_window(self.center_buf, body.center_xy, now, max(FREEZE_WINDOW_S, GROOM_WINDOW_S))
        self._push_window(self.head_buf, body.head_xy, now, GROOM_WINDOW_S)
        self._push_window(self.heading_buf, body.heading_deg, now, FREEZE_WINDOW_S)

        # --- FREEZING: low center displacement + low heading variance for >= duration ---
        c_disp = self._window_disp(self.center_buf)
        h_var = self._window_var(self.heading_buf)

        freeze_candidate = (c_disp <= FREEZE_CENTER_DISP_PX) and (h_var <= FREEZE_HEADING_VAR_DEG2)

        if freeze_candidate and self.metrics.elapsed_time_s >= FREEZE_MIN_DURATION_S:
            # Require sustained candidate: simplest method = candidate in window
            # (for MVP). More formal: state machine with timer.
            self.metrics.freezing_active = True
            self.metrics.freezing_time_s += max(0.0, dt)
        else:
            self.metrics.freezing_active = False

        # --- GROOMING: center stable + head oscillation + heading variance above minimum ---
        # This avoids false positives where mouse is just quiet.
        head_disp = self._window_disp(self.head_buf)

        groom_candidate = (
            (c_disp <= GROOM_CENTER_DISP_PX) and
            (head_disp >= GROOM_HEAD_OSC_PX) and
            (h_var >= GROOM_HEADING_VAR_MIN_DEG2) and
            (not self.metrics.freezing_active)
        )

        if groom_candidate and self.metrics.elapsed_time_s >= GROOM_MIN_DURATION_S:
            self.metrics.grooming_active = True
            self.metrics.grooming_time_s += max(0.0, dt)
        else:
            self.metrics.grooming_active = False

        # --- Record ---
        self.tracking_records.append({
            "frame_index": self.metrics.frames_processed,
            "time_s": self.metrics.elapsed_time_s,
            "center_x_px": body.center_xy[0],
            "center_y_px": body.center_xy[1],
            "head_x_px": body.head_xy[0],
            "head_y_px": body.head_xy[1],
            "tail_x_px": body.tail_xy[0],
            "tail_y_px": body.tail_xy[1],
            "heading_deg": body.heading_deg,
            "zone": zone,
            "speed_px_per_s": speed,
            "freezing": int(self.metrics.freezing_active),
            "grooming": int(self.metrics.grooming_active),
            "confidence": float(getattr(body, "confidence", 1.0)),
        })

        return frame
