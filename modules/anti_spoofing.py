"""
Anti-Spoofing Module
====================

Three complementary liveness checks:

A. Eye Aspect Ratio (EAR) — blink detection  (Soukupová & Čech, 2016)
   The EAR is the ratio of the vertical eye opening to the horizontal eye width:
       EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 · ||p1-p4|| )
   where p1…p6 are the six eye-landmark points in order.
   A live face blinks naturally: EAR briefly drops below ~0.25 then recovers.
   A printed photo or static replay has constant EAR — no blinks are detected.
   Only meaningful in video / live-camera mode (requires multiple frames).

B. LBP Texture Variance — real face vs printed image
   Real skin has rich 3-D micro-texture (pores, hair, subsurface scattering).
   Printed paper has uniform, flat ink-dot texture.
   The variance of the LBP histogram distinguishes them: high variance → real.

C. Laplacian Sharpness — blur / depth-of-field check
   A printed photo photographed by a second camera loses high-frequency detail
   through the extra capture step.  The variance of the Laplacian of the face
   chip is a fast proxy for this: low variance → blurry / flat → possible spoof.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern


EAR_BLINK_THRESHOLD  = 0.25    # EAR below this → blink
BLINKS_REQUIRED      = 1       # blinks needed to pass liveness in live mode
LBP_VARIANCE_THRESH  = 0.0015  # LBP histogram variance below this → suspicious
LAPLACIAN_THRESH     = 60.0    # Laplacian variance below this → too flat


# ── EAR ──────────────────────────────────────────────────────────────────────

def _eye_aspect_ratio(eye_pts: list) -> float:
    """Compute EAR for one eye given its 6 landmark points as (x, y) tuples."""
    pts = np.array(eye_pts, dtype=np.float32)
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-6))


def compute_ear(landmarks: dict) -> tuple[float, bool]:
    """
    Compute average EAR for both eyes.
    Returns (avg_ear, is_blinking).
    """
    if "left_eye" not in landmarks or "right_eye" not in landmarks:
        return 1.0, False
    l = _eye_aspect_ratio(landmarks["left_eye"])
    r = _eye_aspect_ratio(landmarks["right_eye"])
    avg = (l + r) / 2.0
    return avg, avg < EAR_BLINK_THRESHOLD


class BlinkTracker:
    """
    Stateful blink counter for live video streams.

    Counts rising-edge transitions: blinking → not blinking.
    Call update() on every frame with the current landmarks dict.
    Check .passed to know whether liveness is confirmed.
    """

    def __init__(self):
        self.blink_count   = 0
        self._was_blinking = False
        self.last_ear      = 1.0

    def update(self, landmarks: dict) -> tuple[float, bool]:
        ear, blinking = compute_ear(landmarks)
        self.last_ear = ear
        # Rising edge: was blinking last frame, not blinking this frame → completed blink
        if self._was_blinking and not blinking:
            self.blink_count += 1
        self._was_blinking = blinking
        return ear, blinking

    @property
    def passed(self) -> bool:
        return self.blink_count >= BLINKS_REQUIRED

    def reset(self):
        self.blink_count   = 0
        self._was_blinking = False


# ── Texture check ─────────────────────────────────────────────────────────────

def check_texture(gray_chip: np.ndarray) -> tuple[float, bool]:
    """
    LBP histogram variance as a real/printed face discriminator.
    Returns (variance, is_likely_real).
    """
    lbp  = local_binary_pattern(gray_chip, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, density=True)
    var  = float(np.var(hist))
    return var, var > LBP_VARIANCE_THRESH


# ── Laplacian sharpness ───────────────────────────────────────────────────────

def check_sharpness(gray_chip: np.ndarray) -> tuple[float, bool]:
    """
    Laplacian variance as a sharpness / anti-blur check.
    Returns (score, is_sharp_enough).
    """
    lap   = cv2.Laplacian(gray_chip, cv2.CV_64F)
    score = float(lap.var())
    return score, score > LAPLACIAN_THRESH


# ── Combined static check ─────────────────────────────────────────────────────

def static_spoof_check(gray_chip: np.ndarray, landmarks: dict = None) -> dict:
    """
    Run texture and sharpness checks on a single face chip.
    (EAR blink check is handled separately by BlinkTracker in video/live mode.)

    Returns a dict with per-check results and an overall is_live flag.
    """
    tex_var,  tex_ok  = check_texture(gray_chip)
    lap_score, lap_ok = check_sharpness(gray_chip)

    ear = None
    if landmarks:
        ear, _ = compute_ear(landmarks)

    is_live = tex_ok and lap_ok

    return {
        "texture_variance": round(tex_var,  6),
        "texture_ok":       tex_ok,
        "laplacian_score":  round(lap_score, 2),
        "laplacian_ok":     lap_ok,
        "ear":              round(ear, 3) if ear is not None else None,
        "is_live":          is_live,
    }
