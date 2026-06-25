"""
Background Modelling and Image Enhancement Module
==================================================
MOG2 — Mixture of Gaussians v2 (background subtraction):
  - Maintains a per-pixel statistical model built from the recent frame history.
  - Pixels whose intensity deviates significantly from the model are labelled
    foreground (likely moving objects).
  - Shadow pixels (intensity drop without colour shift) are detected separately
    by MOG2 (value = 127 in raw mask) and removed in post-processing so they
    don't produce spurious motion regions.

CLAHE — Contrast Limited Adaptive Histogram Equalization (image enhancement):
  - Applied only to the L (luminance) channel of the LAB colour space so
    colour information is preserved.
  - The tile grid localises equalization to 8×8 image patches, improving
    contrast in dark corners without over-brightening already-lit areas.
  - clipLimit caps the contrast amplification to prevent noise blow-up in
    very uniform regions.
  - Applied before face recognition to improve detection under uneven or
    low-light conditions.
"""

import cv2
import numpy as np

# Singleton subtractor — avoids re-initialising the background model on every call,
# which would discard the learned background and restart from scratch.
_subtractor = None


def _get_subtractor():
    """Return the shared MOG2 instance, creating it on first use."""
    global _subtractor
    if _subtractor is None:
        _subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,        # number of recent frames used to build the background model
            varThreshold=40,    # pixel variance required to be classified as foreground
            detectShadows=True, # separate shadow detection (marked as 127 in raw mask)
        )
    return _subtractor


def reset_background_model():
    """
    Discard the current background model.
    Call this before starting a new, unrelated video so stale background
    statistics from the previous clip don't contaminate motion detection.
    """
    global _subtractor
    _subtractor = None


# ── Image enhancement ─────────────────────────────────────────────────────────

def apply_clahe(frame_bgr):
    """
    Apply CLAHE to the luminance channel and return an enhanced BGR image.

    Improves local contrast without shifting hue or saturation, which makes
    face embeddings more stable under poor or uneven lighting.

    Args:
        frame_bgr: H×W×3 uint8 NumPy array in BGR order.

    Returns:
        Enhanced BGR image of the same shape and dtype.
    """
    # Convert to LAB so we can equalize only the lightness channel.
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # clipLimit=2.0 limits contrast amplification to reduce noise in flat regions.
    # tileGridSize=(8, 8) performs localized equalization across an 8×8 grid of tiles.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)

    enhanced_lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ── Binary morphological operations ──────────────────────────────────────────

def apply_morphology(mask: np.ndarray) -> np.ndarray:
    """
    Apply binary morphological operations to a foreground mask.

    Two-stage pipeline:
      1. Morphological closing (dilation then erosion with an elliptic kernel)
         fills small holes and gaps inside a detected moving object so that
         a person's torso is not split into disconnected blobs.
      2. Dilation (two iterations) expands the remaining blobs outward, merging
         nearby foreground regions that belong to the same physical object.

    Args:
        mask: Binary uint8 mask (0 / 255) from MOG2 threshold.

    Returns:
        Cleaned binary mask of the same shape and dtype.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.dilate(mask, kernel, iterations=2)
    return mask


# ── Motion detection ──────────────────────────────────────────────────────────

def detect_motion(frame_bgr, min_contour_area=2000):
    """
    Apply MOG2 background subtraction and return bounding boxes of moving objects.

    Args:
        frame_bgr:        H×W×3 uint8 NumPy array in BGR order.
        min_contour_area: Contours smaller than this (in pixels²) are ignored
                          to suppress camera noise and tiny background flicker.

    Returns:
        fg_mask  (np.ndarray): binary foreground mask (uint8, 0 or 255).
        regions  (list[tuple]): list of (x, y, w, h) bounding rects for each
                                detected moving region.
    """
    sub     = _get_subtractor()
    raw_mask = sub.apply(frame_bgr)

    # Threshold at 200 to discard shadow pixels (value=127) and keep only
    # confirmed foreground pixels (value=255).
    _, fg_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

    fg_mask = apply_morphology(fg_mask)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [
        cv2.boundingRect(cnt)
        for cnt in contours
        if cv2.contourArea(cnt) > min_contour_area
    ]

    return fg_mask, regions
