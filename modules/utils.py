"""
Drawing Helpers and Access Log Utilities
=========================================
All draw_* functions modify a BGR frame in-place and return nothing.
"""

import cv2
import numpy as np

# ── BGR colour palette ────────────────────────────────────────────────────────
GREEN  = (0, 200, 0)
RED    = (0, 0, 220)
GOLD   = (0, 215, 255)
ORANGE = (0, 140, 255)
YELLOW = (0, 220, 220)
GREY   = (160, 160, 160)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)

LANDMARK_PALETTE = {
    "left_eye":      (0, 255, 0),
    "right_eye":     (0, 255, 0),
    "left_eyebrow":  (0, 200, 255),
    "right_eyebrow": (0, 200, 255),
    "nose_bridge":   (255, 200, 0),
    "nose_tip":      (255, 200, 0),
    "top_lip":       (0, 120, 255),
    "bottom_lip":    (0, 120, 255),
    "chin":          (180, 180, 180),
}


# ── Face / RBAC ───────────────────────────────────────────────────────────────

def draw_rbac_result(frame_bgr: np.ndarray, face: dict, decision: dict) -> None:
    """
    Draw bounding box, identity label, and action badge for one face.

    face:     dict with top/right/bottom/left keys (from face_recognizer)
    decision: dict returned by rbac_engine.make_decision()
    """
    top, right, bottom, left = face["top"], face["right"], face["bottom"], face["left"]
    color   = decision["color"]
    label   = decision["label"]
    action  = decision["action"]
    thick   = 3 if decision.get("role") == "admin" else 2

    cv2.rectangle(frame_bgr, (left, top), (right, bottom), color, thick)

    # Label background pill
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    y_label = top - 10 if top > 28 else bottom + 20
    cv2.rectangle(frame_bgr,
                  (left, y_label - th - 4),
                  (left + tw + 6, y_label + 3),
                  color, -1)
    cv2.putText(frame_bgr, label, (left + 3, y_label),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, BLACK, 1, cv2.LINE_AA)

    # Full-width alert banner for blacklisted users
    if action == "ALERT":
        h = frame_bgr.shape[0]
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, h - 50), (frame_bgr.shape[1], h), (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)
        cv2.putText(frame_bgr, "!! SECURITY ALERT — BLACKLISTED USER !!",
                    (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def draw_landmarks(frame_bgr: np.ndarray, landmarks: dict) -> None:
    """Draw 68-point facial landmark dots and connecting polylines."""
    for feature, pts in landmarks.items():
        color = LANDMARK_PALETTE.get(feature, WHITE)
        arr   = np.array(pts, dtype=np.int32)
        for pt in pts:
            cv2.circle(frame_bgr, (int(pt[0]), int(pt[1])), 2, color, -1)
        if len(pts) >= 2:
            closed = feature in ("left_eye", "right_eye")
            cv2.polylines(frame_bgr, [arr.reshape(-1, 1, 2)], closed, color, 1)


def draw_spoof_badge(frame_bgr: np.ndarray, face: dict, spoof: dict) -> None:
    """Overlay a small ✓ LIVE or ✗ SPOOF badge below the face box."""
    x = face["left"]
    y = face["bottom"] + 14
    if spoof.get("is_live", True):
        cv2.putText(frame_bgr, "LIVE", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame_bgr, "SPOOF?", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, RED, 1, cv2.LINE_AA)


# ── Person tracking ───────────────────────────────────────────────────────────

def draw_person_tracks(frame_bgr: np.ndarray,
                       persons: list[dict], tailgating: bool) -> None:
    color = RED if tailgating else ORANGE
    for p in persons:
        cv2.rectangle(frame_bgr, (p["x1"], p["y1"]), (p["x2"], p["y2"]), color, 2)
        cv2.putText(frame_bgr, f"ID {p['track_id']}",
                    (p["x1"], p["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)

    if tailgating:
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (frame_bgr.shape[1], 44), (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)
        cv2.putText(frame_bgr, "!! TAILGATING ALERT !!",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 3, cv2.LINE_AA)


# ── Background modelling ──────────────────────────────────────────────────────

def draw_motion_regions(frame_bgr: np.ndarray, regions: list) -> None:
    for (x, y, w, h) in regions:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), YELLOW, 1)
    if regions:
        cv2.putText(frame_bgr,
                    f"Motion ({len(regions)} region{'s' if len(regions) != 1 else ''})",
                    (8, frame_bgr.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, YELLOW, 2)


# ── Status stamp ──────────────────────────────────────────────────────────────

def stamp_status(frame_bgr: np.ndarray, text: str, color=GREEN) -> None:
    """Right-aligned status string in the top-right corner."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x = frame_bgr.shape[1] - tw - 8
    cv2.putText(frame_bgr, text, (x, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def stamp_ear(frame_bgr: np.ndarray, ear: float,
              blink_count: int, liveness_ok: bool) -> None:
    """Live-camera EAR / blink indicator in the bottom-left corner."""
    h = frame_bgr.shape[0]
    color = GREEN if liveness_ok else YELLOW
    cv2.putText(frame_bgr,
                f"EAR:{ear:.2f}  Blinks:{blink_count}  "
                f"{'LIVE ✓' if liveness_ok else 'Waiting for blink…'}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
