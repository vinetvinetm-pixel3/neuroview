"""
Body Tracking Module for NeuroView
==================================

Goal:
- Robust single-animal body pose vectors: Nose (Head), Body (Center), Tail base (Tail)
- Output must be stable (no teleports), Any-maze-like, and NEVER create "ghost points".
- If the animal is not detected reliably -> return None (cleanly).

Key ideas:
- Contour-based tracking (no deep learning required for MVP).
- Major axis via PCA on contour points.
- Endpoints = extremes along PCA axis (guaranteed to be ON contour).
- Head vs Tail resolution priority:
    1) Motion: head is endpoint aligned with velocity (when moving).
    2) Morphology: head is endpoint in thinner region (thickness heuristic).
    3) Continuity: prevent 180° flips / sudden swaps.
- Anti-teleport "local anchor": if candidate head/tail jumps > MAX_JUMP_PX vs previous,
  clamp to previous (and reduce confidence).

Public API (HARD CONSTRAINT):
- BodyState dataclass
- extract_body_state_from_mask(...) callable from SessionController
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import numpy as np
import cv2


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

MIN_BODY_AREA_PX2 = 180.0
MORPH_KERNEL_SIZE = 5

# Smoothing (EMA) for stable points
EMA_ALPHA = 0.35

# Anti-teleport anchor
MAX_JUMP_PX = 30.0

# Motion threshold: if center moves faster than this, use velocity rule
MOTION_SPEED_THRESHOLD_PX_PER_S = 5.0

# Continuity: prevent sudden flip unless strongly supported
CONTINUITY_MARGIN_PX = 20.0

Point = Tuple[int, int]
ROIType = Tuple[int, int, int, int]  # (x, y, w, h)


# -----------------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------------

@dataclass
class BodyState:
    """
    All coordinates are GLOBAL (frame coordinates).
    """
    center_xy: Point
    head_xy: Point   # "nose"/head end
    tail_xy: Point   # tail base end
    heading_deg: float  # tail -> head
    area_px2: float
    length_px: float
    width_px: float
    confidence: float = 1.0
    contour_global: Optional[np.ndarray] = None  # Nx1x2 (optional)
    id: int = 0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _angle_deg(v: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(v[1]), float(v[0]))))


def _smooth(prev: np.ndarray, new: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * new + (1.0 - alpha) * prev


def _local_width_score(dist_transform: np.ndarray, p_xy: np.ndarray) -> float:
    """
    Distance transform returns distance to background.
    Bigger means thicker (more interior).
    """
    x = int(round(float(p_xy[0])))
    y = int(round(float(p_xy[1])))
    h, w = dist_transform.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0.0
    return float(dist_transform[y, x])


def _pca_axis(contour_xy: np.ndarray) -> np.ndarray:
    """
    PCA major axis from Nx2 float32 points.
    """
    mu = contour_xy.mean(axis=0, keepdims=True)
    X = contour_xy - mu
    C = np.cov(X.T)
    if C.shape != (2, 2):
        return np.array([1.0, 0.0], dtype=np.float32)
    eigvals, eigvecs = np.linalg.eig(C)
    idx = int(np.argmax(np.real(eigvals)))
    axis = np.real(eigvecs[:, idx]).astype(np.float32)
    return _unit(axis)


def _contour_extremes_along_axis(contour_xy: np.ndarray, center_xy: np.ndarray, axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project contour points onto axis relative to center, pick min/max.
    Both returned points lie on contour.
    """
    vecs = contour_xy - center_xy
    proj = vecs @ axis  # Nx
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    return contour_xy[i_min].copy(), contour_xy[i_max].copy()


def _apply_local_anchor(prev_pt: np.ndarray, new_pt: np.ndarray, max_jump_px: float) -> Tuple[np.ndarray, float]:
    """
    If new point jumps too far from prev, keep prev and reduce confidence.
    Returns (pt, confidence_factor).
    """
    d = float(np.linalg.norm(new_pt - prev_pt))
    if d > max_jump_px:
        return prev_pt.copy(), 0.35
    return new_pt, 1.0


def _min_area_rect_dims(contour: np.ndarray) -> Tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), _ = rect
    length_px = float(max(w, h))
    width_px = float(min(w, h))
    return length_px, width_px


# -----------------------------------------------------------------------------
# Core extraction
# -----------------------------------------------------------------------------

def extract_body_state(
    roi_xywh: ROIType,
    fg_mask_roi_255: np.ndarray,
    prev_state: Optional[BodyState],
    *,
    now_s: Optional[float] = None,
    prev_time_s: Optional[float] = None,
) -> Optional[BodyState]:
    """
    Compute BodyState from ROI-local foreground mask (0/255).
    Returns None when detection is not reliable.
    """
    rx, ry, rw, rh = roi_xywh

    if fg_mask_roi_255 is None or fg_mask_roi_255.size == 0:
        return None

    # --- Clean mask ---
    mask = fg_mask_roi_255.copy()
    mask = cv2.medianBlur(mask, 5)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # --- Contour ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < MIN_BODY_AREA_PX2:
        return None

    # --- Center (ROI-local) ---
    M = cv2.moments(contour)
    if float(M.get("m00", 0.0)) < 1e-9:
        x, y, w, h = cv2.boundingRect(contour)
        center_roi = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
    else:
        center_roi = np.array([float(M["m10"]) / float(M["m00"]), float(M["m01"]) / float(M["m00"])], dtype=np.float32)

    # Contour points Nx2
    pts = contour.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 10:
        return None

    # --- PCA axis and extremes on contour ---
    axis = _pca_axis(pts)
    p1_roi, p2_roi = _contour_extremes_along_axis(pts, center_roi, axis)

    # --- Thickness heuristic (for "quiet" cases) ---
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    # Sample a few pixels inward from each endpoint to measure thickness
    d1 = _unit(center_roi - p1_roi)
    d2 = _unit(center_roi - p2_roi)
    s1 = p1_roi + d1 * 6.0
    s2 = p2_roi + d2 * 6.0
    th1 = _local_width_score(dist, s1)
    th2 = _local_width_score(dist, s2)

    # Default: tail is thinner, head is thicker (often true; can fail on extreme poses)
    # We'll override with motion rule when moving.
    if th1 < th2:
        tail_cand = p1_roi
        head_cand = p2_roi
    else:
        tail_cand = p2_roi
        head_cand = p1_roi

    confidence = 1.0

    # --- Motion rule (Priority 1) ---
    # If animal is moving, head is the endpoint aligned with velocity direction.
    if prev_state is not None and now_s is not None and prev_time_s is not None:
        dt = max(1e-6, float(now_s - prev_time_s))
        prev_center_roi = np.array([prev_state.center_xy[0] - rx, prev_state.center_xy[1] - ry], dtype=np.float32)
        v = (center_roi - prev_center_roi) / dt
        speed = float(np.linalg.norm(v))

        if speed >= MOTION_SPEED_THRESHOLD_PX_PER_S:
            vhat = _unit(v)
            # Compare which endpoint is more aligned with velocity from center
            e1 = _unit(p1_roi - center_roi)
            e2 = _unit(p2_roi - center_roi)
            score1 = float(e1 @ vhat)
            score2 = float(e2 @ vhat)
            # Head = endpoint with higher alignment to velocity
            if score1 > score2:
                head_cand, tail_cand = p1_roi, p2_roi
            else:
                head_cand, tail_cand = p2_roi, p1_roi

    # --- Continuity rule (Priority 3) ---
    # Avoid sudden head swap unless clearly better.
    if prev_state is not None:
        prev_head_roi = np.array([prev_state.head_xy[0] - rx, prev_state.head_xy[1] - ry], dtype=np.float32)
        prev_tail_roi = np.array([prev_state.tail_xy[0] - rx, prev_state.tail_xy[1] - ry], dtype=np.float32)

        d_head_to_p1 = float(np.linalg.norm(p1_roi - prev_head_roi))
        d_head_to_p2 = float(np.linalg.norm(p2_roi - prev_head_roi))

        # If one endpoint is significantly closer to previous head, enforce it.
        if d_head_to_p1 + CONTINUITY_MARGIN_PX < d_head_to_p2:
            head_cand, tail_cand = p1_roi, p2_roi
        elif d_head_to_p2 + CONTINUITY_MARGIN_PX < d_head_to_p1:
            head_cand, tail_cand = p2_roi, p1_roi

        # Anti-teleport local anchor (HARD FIX)
        head_cand, c1 = _apply_local_anchor(prev_head_roi, head_cand, MAX_JUMP_PX)
        tail_cand, c2 = _apply_local_anchor(prev_tail_roi, tail_cand, MAX_JUMP_PX)
        confidence *= min(c1, c2)

        # Smooth AFTER anchoring
        head_cand = _smooth(prev_head_roi, head_cand, EMA_ALPHA)
        tail_cand = _smooth(prev_tail_roi, tail_cand, EMA_ALPHA)
        center_roi = _smooth(prev_center_roi, center_roi, EMA_ALPHA)

    # --- Build outputs (GLOBAL coords) ---
    center_g = (int(round(rx + float(center_roi[0]))), int(round(ry + float(center_roi[1]))))
    head_g = (int(round(rx + float(head_cand[0]))), int(round(ry + float(head_cand[1]))))
    tail_g = (int(round(rx + float(tail_cand[0]))), int(round(ry + float(tail_cand[1]))))

    heading_vec = np.array([head_g[0] - tail_g[0], head_g[1] - tail_g[1]], dtype=np.float32)
    heading_deg = _angle_deg(heading_vec)

    length_px, width_px = _min_area_rect_dims(contour)

    contour_global = contour.copy()
    contour_global[:, :, 0] += int(rx)
    contour_global[:, :, 1] += int(ry)

    # NOTE: we do NOT force points to contour here to avoid wrong snapping when mask is noisy.
    # We rely on local anchor + smoothing. UI will fade if confidence is low.

    return BodyState(
        center_xy=center_g,
        head_xy=head_g,
        tail_xy=tail_g,
        heading_deg=heading_deg,
        area_px2=area,
        length_px=length_px,
        width_px=width_px,
        confidence=float(max(0.0, min(confidence, 1.0))),
        contour_global=contour_global,
        id=0,
    )


# -----------------------------------------------------------------------------
# Backwards-compatible public API
# -----------------------------------------------------------------------------

def extract_body_state_from_mask(
    *,
    frame_bgr: Optional[np.ndarray] = None,  # kept for compatibility; unused in MVP geometry tracker
    roi_xywh: ROIType,
    fg_mask_roi: np.ndarray,
    prev_state: Optional[BodyState],
    now_s: Optional[float] = None,
    prev_time_s: Optional[float] = None,
) -> Optional[BodyState]:
    """
    Compatibility wrapper used by SessionController.

    IMPORTANT:
    - Must return None cleanly when no animal is detected (prevents ghost points).
    """
    return extract_body_state(
        roi_xywh=roi_xywh,
        fg_mask_roi_255=fg_mask_roi,
        prev_state=prev_state,
        now_s=now_s,
        prev_time_s=prev_time_s,
    )
