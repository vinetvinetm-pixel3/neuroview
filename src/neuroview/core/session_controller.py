"""
SessionController (MVP - Vector Tracking)

- Video/camera source
- ROI provided by GUI (no OpenCV windows)
- Tracking via body_tracking.py (Contour-PCA based)
- Behaviors:
    - Freezing: low center motion + low heading variance (windowed + min duration)
    - Grooming: stable center + head oscillation (windowed + min duration)
- Zones:
    - Open Field: Center vs Periphery (ratio-based inner rectangle)
- Export: CSV/PDF

Design:
- Core contains no GUI calls.
- If tracking is lost, we return the raw frame and keep metrics stable.
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

# --- Freezing ---
FREEZE_MIN_DURATION_S = 2.0
FREEZE_WINDOW_S = 2.0
FREEZE_CENTER_DISP_PX = 3.0            # max center displacement inside window
FREEZE_HEADING_VAR_DEG2 = 25.0         # variance threshold (deg^2)

# --- Grooming ---
GROOM_MIN_DURATION_S = 1.5
GROOM_WINDOW_S = 1.5
GROOM_CENTER_DISP_PX = 5.0             # center stays mostly in place
GROOM_HEAD_OSC_PX = 8.0                # head moves around while center stays
GROOM_HEADING_VAR_MIN_DEG2 = 40.0      # heading variance usually higher during grooming

# --- Movement classification ---
MOVEMENT_SPEED_THRESHOLD_PX_S = 3.0

# --- Zones ---
ZONE_CENTER_AREA_RATIO = 0.55

# --- Tracking robustness ---
# If body tracking returns None for too many consecutive frames, drop continuity lock
# so reacquisition is fast when the animal reappears.
MAX_LOST_FRAMES_BEFORE_RESET = 10


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
    Simple Open Field zone model:
    - Center: inner rectangle whose AREA is center_area_ratio of ROI area.
    - Periphery: rest.
    """
    def __init__(self, roi: ROIType, center_area_ratio: float = ZONE_CENTER_AREA_RATIO):
        rx, ry, rw, rh = roi
        self.rx, self.ry, self.rw, self.rh = int(rx), int(ry), int(rw), int(rh)

        s = math.sqrt(float(center_area_ratio))
        cw = self.rw * s
        ch = self.rh * s
        self.x1 = (self.rw - cw) / 2.0
        self.y1 = (self.rh - ch) / 2.0
        self.x2 = self.x1 + cw
        self.y2 = self.y1 + ch

    def classify(self, x_global: float, y_global: float) -> str:
        lx = float(x_global) - float(self.rx)
        ly = float(y_global) - float(self.ry)
        if self.x1 <= lx <= self.x2 and self.y1 <= ly <= self.y2:
            return "Center"
        return "Periphery"


class SessionController:
    def __init__(self) -> None:
        # FPS handling
        self.video_fps: float = 30.0
        self.frame_interval_s: float = 1.0 / self.video_fps

        # Capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.source: Optional[SourceType] = None

        # Run state
        self.running: bool = False
        self.paused: bool = False

        # Timing
        self.start_time: Optional[float] = None
        self._last_t: Optional[float] = None

        # ROI / Zones
        self.roi: Optional[ROIType] = None
        self.roi_defined: bool = False
        self.zone_model: Optional[ZoneModel] = None

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        # Last frame & metrics
        self.last_frame: Optional[np.ndarray] = None
        self.metrics = Metrics()

        # Tracking (single animal)
        self.prev_body: Optional[BodyState] = None
        self.last_body: Optional[BodyState] = None
        self._lost_frames: int = 0

        # Histories (sliding windows)
        self.center_buf: deque[Tuple[Tuple[int, int], float]] = deque()
        self.head_buf: deque[Tuple[Tuple[int, int], float]] = deque()
        self.heading_buf: deque[Tuple[float, float]] = deque()

        # Behavior state machines (durations)
        self._freeze_candidate_start: Optional[float] = None
        self._groom_candidate_start: Optional[float] = None

        # Export data
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
        self._reset_runtime_stats(keep_time=False)

    def clear_roi(self) -> None:
        self.roi = None
        self.roi_defined = False
        self.zone_model = None
        self._reset_runtime_stats(keep_time=False)

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
        # Do not clear ROI here (GUI controls ROI)
        self.running = False
        self.paused = False
        self.start_time = None
        self._last_t = None

        self._reset_runtime_stats(keep_time=False)

        self.prev_body = None
        self.last_body = None
        self.last_frame = None
        self.tracking_records.clear()
        self._lost_frames = 0

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

        # Reset runtime (keep ROI)
        self.prev_body = None
        self.last_body = None
        self.tracking_records.clear()
        self._lost_frames = 0

        self._reset_runtime_stats(keep_time=False)

        # Reset bg model
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

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

        self.running = True
        self.paused = False
        self.start_time = time.time()
        self._last_t = self.start_time

        # Reset runtime, keep ROI and source
        self._reset_runtime_stats(keep_time=False)
        self.prev_body = None
        self.last_body = None
        self.tracking_records.clear()
        self._lost_frames = 0

        # Fresh bg model
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
            # Summary
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
    # Internal helpers
    # -------------------------------------------------------------------------
    def _reset_runtime_stats(self, keep_time: bool = False) -> None:
        """
        Reset metrics + buffers. If keep_time=True, preserves elapsed time counters.
        """
        prev_elapsed = self.metrics.elapsed_time_s if keep_time else 0.0

        self.metrics = Metrics()
        self.metrics.elapsed_time_s = prev_elapsed

        self.center_buf.clear()
        self.head_buf.clear()
        self.heading_buf.clear()

        self._freeze_candidate_start = None
        self._groom_candidate_start = None

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

    def _behavior_freezing_update(self, candidate: bool, now: float, dt: float) -> None:
        """
        Freezing state machine:
        - Candidate must be sustained for FREEZE_MIN_DURATION_S.
        - Once active, accumulates time while candidate remains true.
        """
        if not candidate:
            self.metrics.freezing_active = False
            self._freeze_candidate_start = None
            return

        # candidate is true
        if self._freeze_candidate_start is None:
            self._freeze_candidate_start = now
            self.metrics.freezing_active = False
            return

        if (now - self._freeze_candidate_start) >= FREEZE_MIN_DURATION_S:
            self.metrics.freezing_active = True
            self.metrics.freezing_time_s += max(0.0, dt)
        else:
            self.metrics.freezing_active = False

    def _behavior_grooming_update(self, candidate: bool, now: float, dt: float) -> None:
        """
        Grooming state machine:
        - Candidate must be sustained for GROOM_MIN_DURATION_S.
        - Once active, accumulates time while candidate remains true.
        - Mutually exclusive with freezing.
        """
        if self.metrics.freezing_active:
            self.metrics.grooming_active = False
            self._groom_candidate_start = None
            return

        if not candidate:
            self.metrics.grooming_active = False
            self._groom_candidate_start = None
            return

        # candidate is true
        if self._groom_candidate_start is None:
            self._groom_candidate_start = now
            self.metrics.grooming_active = False
            return

        if (now - self._groom_candidate_start) >= GROOM_MIN_DURATION_S:
            self.metrics.grooming_active = True
            self.metrics.grooming_time_s += max(0.0, dt)
        else:
            self.metrics.grooming_active = False

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

        # Foreground mask
        fg = self.bg_subtractor.apply(gray)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        # --- Body tracking ---
        body = extract_body_state_from_mask(
            frame_bgr=frame,
            roi_xywh=(rx, ry, rw, rh),
            fg_mask_roi=fg,
            prev_state=self.prev_body,
        )

        # If no detection: do not update buffers/records and avoid ghost points.
        if body is None:
            self.last_body = None
            self._lost_frames += 1

            # After several lost frames, drop continuity lock for fast reacquisition.
            if self._lost_frames >= MAX_LOST_FRAMES_BEFORE_RESET:
                self.prev_body = None

            # Behaviors should not accumulate while we are blind.
            self.metrics.freezing_active = False
            self.metrics.grooming_active = False
            self._freeze_candidate_start = None
            self._groom_candidate_start = None

            return frame

        # Tracking recovered
        self._lost_frames = 0
        self.prev_body = body
        self.last_body = body

        # --- Kinematics / speed ---
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

        # --- Update windows (valid tracking only) ---
        self._push_window(self.center_buf, body.center_xy, now, max(FREEZE_WINDOW_S, GROOM_WINDOW_S))
        self._push_window(self.head_buf, body.head_xy, now, GROOM_WINDOW_S)
        self._push_window(self.heading_buf, body.heading_deg, now, FREEZE_WINDOW_S)

        # --- Window statistics ---
        c_disp = self._window_disp(self.center_buf)
        h_var = self._window_var(self.heading_buf)
        head_disp = self._window_disp(self.head_buf)

        # --- Freezing candidate ---
        freeze_candidate = (c_disp <= FREEZE_CENTER_DISP_PX) and (h_var <= FREEZE_HEADING_VAR_DEG2)
        self._behavior_freezing_update(freeze_candidate, now, dt)

        # --- Grooming candidate ---
        # Key idea: stable center + head oscillation + enough heading variability,
        # and not freezing.
        groom_candidate = (
            (c_disp <= GROOM_CENTER_DISP_PX) and
            (head_disp >= GROOM_HEAD_OSC_PX) and
            (h_var >= GROOM_HEADING_VAR_MIN_DEG2) and
            (not self.metrics.freezing_active)
        )
        self._behavior_grooming_update(groom_candidate, now, dt)

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
        })

        return frame
