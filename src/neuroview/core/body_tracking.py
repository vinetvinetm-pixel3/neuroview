"""
Body Tracking Module for NeuroView (MVP)
=======================================

Goal:
- Robust, geometry-based single-animal body tracking from a binary foreground mask.
- Outputs 3 keypoints: Nose (Head), Body (Center), Tail Base (Tail) as vectors.
- Prevents "ghost points" and prevents head/tail flipping using a strict hierarchy:
    Priority 1 (Movement): head leads motion (velocity projection)
    Priority 2 (Morphology): if still, head is thicker end / tail thinner end
    Priority 3 (Continuity): avoid 180-degree flips frame-to-frame

Critical Fix:
- Adaptive anti-teleport:
    * If animal is moving, do NOT freeze head/tail to previous positions.
    * Only clamp when detection is poor or jump is implausible while nearly still.

Hard constraint:
- Head and tail markers lie on the animal contour (after smoothing we snap to contour).

Dependencies: numpy, opencv-python
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MIN_BODY_AREA_PX2 = 150.0
MORPH_KERNEL_SIZE = 5

# EMA (adaptive): higher alpha = more responsive (less lag)
EMA_ALPHA_STILL = 0.35
EMA_ALPHA_MOVING = 0.70

# Motion threshold (ROI-local px per frame) to consider "moving"
MOTION_CENTER_STEP_PX = 2.0

# Anti-teleport configuration (ROI-local pixels)
# If nearly still and endpoint jumps too far -> keep previous (avoid corners/reflections)
MAX_JUMP_WHEN_STILL_PX = 30.0

# If moving, allow larger corrections (so points don't "stick")
MAX_JUMP_WHEN_MOVING_PX = 120.0

# Continuity: prevent head/tail flip unless strongly supported
FLIP_DOT_THRESHOLD = -0.25  # if heading dot < this, it's ~180deg flip


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Point = Tuple[int, int]
ROIType = Tuple[int, int, int, int]  # (x, y, w, h)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class BodyState:
    """
    All coordinates are GLOBAL frame coordinates.
    """
    center_xy: Point
    head_xy: Point
    tail_xy: Point
    heading_deg: float
    area_px2: float
    length_px: float
    width_px: float
    contour_global: Optional[np.ndarray] = None
    id: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-9:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _angle_deg(v: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(v[1]), float(v[0]))))


def _smooth(prev: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * new + (1.0 - alpha) * prev


def _snap_to_contour(point_xy: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """
    Snap point to the nearest contour point (brute force).
    Contour shape: (N,1,2) or (N,2).
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if pts.size == 0:
        return point_xy
    diffs = pts - point_xy
    d2 = np.sum(diffs * diffs, axis=1)
    return pts[int(np.argmin(d2))]


def _local_width_score(dist_transform: np.ndarray, p_xy: np.ndarray) -> float:
    x = int(round(float(p_xy[0])))
    y = int(round(float(p_xy[1])))
    h, w = dist_transform.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0.0
    return float(dist_transform[y, x])


def _min_rect_dims(contour: np.ndarray) -> tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), _ = rect
    return float(max(w, h)), float(min(w, h))


def _pca_axis_and_extremes(contour: np.ndarray, center_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA axis from contour points; extremes are min/max projection onto axis.
    Returns: axis_unit (2,), p_min (2,), p_max (2,) in ROI-local coordinates.
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        axis = np.array([1.0, 0.0], dtype=np.float32)
        return axis, center_roi.copy(), center_roi.copy()

    mean = pts.mean(axis=0, keepdims=True)
    X = pts - mean
    C = np.cov(X.T)
    if C.ndim != 2:
        axis = np.array([1.0, 0.0], dtype=np.float32)
        return axis, pts[0], pts[-1]

    eigvals, eigvecs = np.linalg.eig(C)
    idx = int(np.argmax(np.real(eigvals)))
    axis = np.real(eigvecs[:, idx]).astype(np.float32)
    axis = _unit(axis)

    proj = (pts - center_roi) @ axis  # (N,)
    p_min = pts[int(np.argmin(proj))]
    p_max = pts[int(np.argmax(proj))]
    return axis, p_min, p_max


def _decide_head_tail(
    center_roi: np.ndarray,
    p_min: np.ndarray,
    p_max: np.ndarray,
    cleaned_mask_255: np.ndarray,
    prev_state: Optional[BodyState],
    rx: int,
    ry: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hierarchy:
    1) Movement: head leads velocity (center delta)
    2) Morphology: thinner end => tail (thickness heuristic)
    3) Continuity: prevent 180deg flip unless justified
    """
    # Default candidates
    c1 = p_min.astype(np.float32)
    c2 = p_max.astype(np.float32)

    # Compute velocity (ROI-local) from prev center -> current center
    v = None
    moving = False
    if prev_state is not None:
        prev_c = np.array([prev_state.center_xy[0] - rx, prev_state.center_xy[1] - ry], dtype=np.float32)
        v = center_roi - prev_c
        moving = float(np.linalg.norm(v)) >= MOTION_CENTER_STEP_PX

    if moving and v is not None:
        v_u = _unit(v)
        # pick endpoint that has higher dot with velocity direction (leads motion)
        d1 = float(np.dot(_unit(c1 - center_roi), v_u))
        d2 = float(np.dot(_unit(c2 - center_roi), v_u))
        if d1 >= d2:
            head, tail = c1, c2
        else:
            head, tail = c2, c1
        return head, tail

    # Not moving: thickness heuristic
    dist = cv2.distanceTransform((cleaned_mask_255 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    # sample slightly inward from each endpoint
    s1 = c1 + _unit(center_roi - c1) * 5.0
    s2 = c2 + _unit(center_roi - c2) * 5.0
    th1 = _local_width_score(dist, s1)
    th2 = _local_width_score(dist, s2)

    # Tail is usually thinner
    if th1 < th2:
        tail, head = c1, c2
    else:
        tail, head = c2, c1

    # Continuity: prevent sudden flip if prev heading exists
    if prev_state is not None:
        prev_h = np.array([prev_state.head_xy[0] - rx, prev_state.head_xy[1] - ry], dtype=np.float32)
        prev_t = np.array([prev_state.tail_xy[0] - rx, prev_state.tail_xy[1] - ry], dtype=np.float32)
        prev_dir = _unit(prev_h - prev_t)
        new_dir = _unit(head - tail)
        dot = float(np.dot(prev_dir, new_dir))
        # if strong flip (~180deg), keep previous assignment (swap)
        if dot < FLIP_DOT_THRESHOLD:
            head, tail = tail, head

    return head, tail


def _apply_anti_teleport(
    head: np.ndarray,
    tail: np.ndarray,
    center_roi: np.ndarray,
    prev_state: Optional[BodyState],
    rx: int,
    ry: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    If nearly still and the new head/tail jump too far, keep previous points.
    If moving, allow much larger jumps so points don't "stick" behind.
    Returns (head, tail, moving)
    """
    if prev_state is None:
        return head, tail, False

    prev_c = np.array([prev_state.center_xy[0] - rx, prev_state.center_xy[1] - ry], dtype=np.float32)
    prev_h = np.array([prev_state.head_xy[0] - rx, prev_state.head_xy[1] - ry], dtype=np.float32)
    prev_t = np.array([prev_state.tail_xy[0] - rx, prev_state.tail_xy[1] - ry], dtype=np.float32)

    v = center_roi - prev_c
    moving = float(np.linalg.norm(v)) >= MOTION_CENTER_STEP_PX

    max_jump = MAX_JUMP_WHEN_MOVING_PX if moving else MAX_JUMP_WHEN_STILL_PX

    if float(np.linalg.norm(head - prev_h)) > max_jump:
        head = prev_h
    if float(np.linalg.norm(tail - prev_t)) > max_jump:
        tail = prev_t

    return head, tail, moving


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_body_state(
    roi_xywh: ROIType,
    fg_mask_roi_255: np.ndarray,
    prev_state: Optional[BodyState],
) -> Optional[BodyState]:
    rx, ry, rw, rh = roi_xywh

    if fg_mask_roi_255 is None or fg_mask_roi_255.size == 0:
        return None

    # --- Clean mask ---
    cleaned = cv2.medianBlur(fg_mask_roi_255, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < MIN_BODY_AREA_PX2:
        return None

    # --- Center (moments) ---
    M = cv2.moments(contour)
    if M.get("m00", 0.0) <= 1e-9:
        x, y, w, h = cv2.boundingRect(contour)
        cx_roi = x + w / 2.0
        cy_roi = y + h / 2.0
    else:
        cx_roi = float(M["m10"] / M["m00"])
        cy_roi = float(M["m01"] / M["m00"])

    center_roi = np.array([cx_roi, cy_roi], dtype=np.float32)

    # --- PCA axis + extremes (ON contour) ---
    _, p_min, p_max = _pca_axis_and_extremes(contour, center_roi)

    # --- Head/Tail decision ---
    head_roi, tail_roi = _decide_head_tail(center_roi, p_min, p_max, cleaned, prev_state, rx, ry)

    # --- Anti-teleport (adaptive) ---
    head_roi, tail_roi, moving = _apply_anti_teleport(head_roi, tail_roi, center_roi, prev_state, rx, ry)

    # --- Adaptive smoothing (so it follows immediately when moving) ---
    alpha = EMA_ALPHA_MOVING if moving else EMA_ALPHA_STILL
    if prev_state is not None:
        prev_c = np.array([prev_state.center_xy[0] - rx, prev_state.center_xy[1] - ry], dtype=np.float32)
        prev_h = np.array([prev_state.head_xy[0] - rx, prev_state.head_xy[1] - ry], dtype=np.float32)
        prev_t = np.array([prev_state.tail_xy[0] - rx, prev_state.tail_xy[1] - ry], dtype=np.float32)

        center_roi = _smooth(prev_c, center_roi, alpha * 0.6)  # center slightly smoother
        head_roi = _smooth(prev_h, head_roi, alpha)
        tail_roi = _smooth(prev_t, tail_roi, alpha)

    # --- Snap head/tail to contour after smoothing (hard constraint) ---
    head_roi = _snap_to_contour(head_roi, contour)
    tail_roi = _snap_to_contour(tail_roi, contour)

    # Heading tail->head
    heading_vec = head_roi - tail_roi
    heading_deg = _angle_deg(heading_vec)

    length_px, width_px = _min_rect_dims(contour)

    # --- Convert to global ---
    center_g = (int(round(rx + float(center_roi[0]))), int(round(ry + float(center_roi[1]))))
    head_g = (int(round(rx + float(head_roi[0]))), int(round(ry + float(head_roi[1]))))
    tail_g = (int(round(rx + float(tail_roi[0]))), int(round(ry + float(tail_roi[1]))))

    contour_global = contour.copy()
    contour_global[:, :, 0] += rx
    contour_global[:, :, 1] += ry

    return BodyState(
        center_xy=center_g,
        head_xy=head_g,
        tail_xy=tail_g,
        heading_deg=heading_deg,
        area_px2=area,
        length_px=length_px,
        width_px=width_px,
        contour_global=contour_global,
        id=0
    )


# ---------------------------------------------------------------------------
# Backwards-Compatible API (used by SessionController)
# ---------------------------------------------------------------------------

def extract_body_state_from_mask(
    *,
    frame_bgr: Optional[np.ndarray] = None,   # kept for compatibility (not used here)
    roi_xywh: ROIType,
    fg_mask_roi: np.ndarray,
    prev_state: Optional[BodyState],
) -> Optional[BodyState]:
    """
    Public API used by SessionController.
    Must return None cleanly when no animal is detected.
    """
    return extract_body_state(
        roi_xywh=roi_xywh,
        fg_mask_roi_255=fg_mask_roi,
        prev_state=prev_state,
    )
