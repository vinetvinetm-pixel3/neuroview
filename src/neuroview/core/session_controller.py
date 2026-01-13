"""
SessionController

This module contains the core runtime state and processing logic for the session.
It reuses the legacy centroid/contrast tracking logic (MOG2 + contours + centroid).

GUI must call:
- open_video/open_camera
- start/stop/toggle_pause
- process_next_frame() -> returns a frame (for display) + updates metrics/history

Later, we will replace the centroid logic with pose estimation and keep the same API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List, Tuple

import time
import math
import os
import csv
import datetime

import cv2
import numpy as np


SourceType = Literal["video", "camera"]
PointT = Tuple[int, int]


@dataclass
class Metrics:
    run_time_s: float = 0.0
    total_distance_px: float = 0.0
    inst_speed_px_s: float = 0.0
    current_zone: str = "-"
    frames_processed: int = 0


class SessionController:
    """
    Legacy-compatible controller (centroid tracking).
    """

    def __init__(self) -> None:
        # FPS handling
        self.video_fps: float = 30.0
        self.frame_interval_s: float = 1.0 / self.video_fps

        # Visual tail
        self.tail_seconds: float = 3.0
        self.point_stride: int = 6
        self.point_radius: int = 3

        # Heatmap
        self.heatmap: Optional[np.ndarray] = None
        self.heatmap_scale: int = 2

        # Zones
        self.zone_history: List[Tuple[str, float]] = []

        # Video capture
        self.cap: Optional[cv2.VideoCapture] = None
        self.source: Optional[SourceType] = None

        # State
        self.running: bool = False
        self.paused: bool = False
        self.start_time: Optional[float] = None

        # ROI
        self.roi: Optional[Tuple[int, int, int, int]] = None
        self.roi_defined: bool = False

        # Histories
        self.points_history: List[Tuple[int, int, float]] = []   # (x, y, t)
        self.tracking_records: List[Dict[str, Any]] = []

        # Background subtractor (legacy)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        # Last frame (for exports)
        self.last_frame: Optional[np.ndarray] = None

        # Live metrics
        self.metrics = Metrics()

    # -------------------------
    # Video sources
    # -------------------------
    def release_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def open_video(self, path: str) -> None:
        self.release_capture()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        self.video_fps = float(fps)
        self.frame_interval_s = 1.0 / self.video_fps

        self.cap = cap
        self.source = "video"
        self.reset_for_new_source()

    def open_camera(self, index: int = 0) -> None:
        self.release_capture()
        cap = cv2.VideoCapture(index)  # macOS: no CAP_DSHOW
        if not cap.isOpened():
            raise RuntimeError("Could not open camera.")
        self.cap = cap
        self.source = "camera"
        self.video_fps = 30.0
        self.frame_interval_s = 1.0 / self.video_fps
        self.reset_for_new_source()

    def reset_for_new_source(self) -> None:
        self.metrics = Metrics()
        self.points_history.clear()
        self.zone_history.clear()
        self.heatmap = None
        self.roi_defined = False
        self.roi = None
        self.tracking_records.clear()
        self.last_frame = None

    # -------------------------
    # ROI (legacy OpenCV ROI picker)
    # -------------------------
    def define_roi_opencv(self) -> bool:
        """
        Blocks UI while OpenCV ROI window is open.
        For MVP it's OK. Later we'll implement ROI selection directly in the GUI.
        """
        if self.cap is None:
            return False

        ret, frame = self.cap.read()
        if not ret:
            return False

        win_name = "Select ROI - Press ENTER to confirm / ESC to cancel"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        r = cv2.selectROI(win_name, frame, False, False)
        cv2.destroyWindow(win_name)

        x, y, w, h = r
        if w == 0 or h == 0:
            return False

        self.roi = (int(x), int(y), int(w), int(h))
        self.roi_defined = True

        # If video, rewind so we don't lose frames
        if self.source == "video" and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        return True

    # -------------------------
    # Run control
    # -------------------------
    def start(self) -> None:
        if self.cap is None:
            raise RuntimeError("No source opened.")
        if not self.roi_defined:
            ok = self.define_roi_opencv()
            if not ok:
                raise RuntimeError("ROI not defined.")

        self.running = True
        self.paused = False
        self.start_time = time.time()

        self.metrics = Metrics()
        self.points_history.clear()
        self.zone_history.clear()
        self.heatmap = None
        self.tracking_records.clear()
        self.last_frame = None

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

    def stop(self) -> None:
        self.running = False

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    # -------------------------
    # Core processing (reused from your process_loop)
    # -------------------------
    def compute_zone(self, x_roi: int, y_roi: int, w: int, h: int) -> str:
        col = w / 3.0
        if x_roi < col:
            return "Zona 1"
        elif x_roi < 2 * col:
            return "Zona 2"
        return "Zona 3"

    def process_next_frame(self) -> Optional[np.ndarray]:
        """
        Reads one frame and updates all state.
        Returns a BGR frame with overlays (for GUI display).
        If video ended, returns None and stops.
        """
        if not self.running or self.paused or self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            if self.source == "video":
                self.running = False
            return None

        self.last_frame = frame.copy()
        self.metrics.frames_processed += 1

        if self.start_time is not None:
            self.metrics.run_time_s = time.time() - self.start_time

        h_total, w_total = frame.shape[:2]

        # ROI
        if self.roi_defined and self.roi is not None:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 2)
            roi_frame = frame[ry:ry + rh, rx:rx + rw]
        else:
            rx, ry, rw, rh = 0, 0, w_total, h_total
            roi_frame = frame

        # Foreground mask (legacy)
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        fg = self.bg_subtractor.apply(gray)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 100:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cx_global = rx + cx
                    cy_global = ry + cy

                    t_now = time.time()
                    self.points_history.append((cx_global, cy_global, t_now))

                    # Distance + speed
                    if len(self.points_history) >= 2:
                        (x1, y1, t1), (x2, y2, t2) = self.points_history[-2:]
                        dist = float(np.hypot(x2 - x1, y2 - y1))
                        self.metrics.total_distance_px += dist
                        if (t2 - t1) > 0:
                            self.metrics.inst_speed_px_s = dist / (t2 - t1)

                    # Zone based on ROI-local coordinates (same as your code)
                    zona = self.compute_zone(cx, cy, rw, rh)
                    self.metrics.current_zone = zona

                    # Heatmap
                    if self.heatmap is None:
                        self.heatmap = np.zeros(
                            (rh // self.heatmap_scale, rw // self.heatmap_scale),
                            dtype=np.float32
                        )
                    hx = cx // self.heatmap_scale
                    hy = cy // self.heatmap_scale
                    if 0 <= hy < self.heatmap.shape[0] and 0 <= hx < self.heatmap.shape[1]:
                        self.heatmap[hy, hx] += 1.0

                    # Record
                    elapsed_frame = self.metrics.run_time_s
                    self.tracking_records.append({
                        "frame": self.metrics.frames_processed,
                        "time_s": elapsed_frame,
                        "x": cx_global,
                        "y": cy_global,
                        "zona": zona,
                        "distancia_total_px": self.metrics.total_distance_px,
                        "velocidad_px_s": self.metrics.inst_speed_px_s
                    })

                    # Draw current point
                    cv2.circle(frame, (cx_global, cy_global), 6, (0, 255, 0), -1)

                    # Draw tail points (last N seconds)
                    cutoff = time.time() - self.tail_seconds
                    for idx, (px, py, tt) in enumerate(self.points_history):
                        if tt < cutoff:
                            continue
                        if idx % self.point_stride != 0:
                            continue
                        cv2.circle(frame, (px, py), self.point_radius, (0, 255, 255), -1)

        return frame
