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


# ── Temporal motion check ─────────────────────────────────────────────────────

def temporal_spoof_check(
    prev_gray: "np.ndarray | None",
    curr_gray: np.ndarray,
    cache: dict,
    security_mode: str = "normal",
) -> dict:
    """
    Frame-to-frame pixel-difference check for live vs. replay/static attacks.
    A real face has natural micro-motion; a still replay has near-zero difference.
    Updates cache['prev_gray_face'] for the next call.
    """
    result = {"motion_score": 0.0, "motion_ok": True, "suspicious_static": False}

    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        cache["prev_gray_face"] = curr_gray.copy()
        return result

    diff = cv2.absdiff(prev_gray, curr_gray)
    motion_score = float(np.mean(diff))
    result["motion_score"] = round(motion_score, 3)

    # Below this level the face is suspiciously static (printed / paused replay)
    static_thresh = {"strict": 1.5, "normal": 1.0, "relaxed": 0.5}.get(security_mode, 1.0)
    result["suspicious_static"] = motion_score < static_thresh
    result["motion_ok"] = not result["suspicious_static"]

    cache["prev_gray_face"] = curr_gray.copy()
    return result


# ── Replay / screen artifact heuristic ───────────────────────────────────────

def replay_artifact_check(gray_face: np.ndarray) -> dict:
    """
    DCT frequency-domain check for screen/print artifacts.
    Replay attacks via a screen may have unnaturally low or periodic
    high-frequency content compared with real skin texture.
    """
    f32 = np.float32(gray_face)
    dct = cv2.dct(f32)
    h, w = dct.shape

    total_energy = float(np.sum(np.abs(dct))) + 1e-6
    low_energy   = float(np.sum(np.abs(dct[:h // 4, :w // 4])))
    high_energy  = float(np.sum(np.abs(dct[h // 4:, w // 4:])))

    freq_ratio = high_energy / (low_energy + 1e-6)
    low_ratio  = low_energy / total_energy

    # Natural face: moderate freq_ratio; flat/screen dominated by low-freq DC
    freq_ok = 0.02 < freq_ratio < 20.0 and low_ratio < 0.97

    return {
        "freq_ratio": round(freq_ratio, 3),
        "low_ratio":  round(low_ratio,  3),
        "freq_ok":    freq_ok,
    }


# ── Spoof score aggregation ───────────────────────────────────────────────────

def aggregate_spoof_result(
    static_result:   dict,
    blink_passed:    bool,
    temporal_result: "dict | None" = None,
    replay_result:   "dict | None" = None,
    security_mode:   str = "normal",
) -> dict:
    """
    Combine all spoof signals (static, blink, temporal, replay) into one verdict.

    strict:  blink + texture + laplacian + temporal motion + frequency all required
    normal:  blink + texture + laplacian required; temporal/replay used to downgrade
    relaxed: blink required; static checks advisory; temporal only for full statics
    """
    tex_ok      = static_result.get("texture_ok",   True)
    lap_ok      = static_result.get("laplacian_ok", True)
    susp_static = (temporal_result or {}).get("suspicious_static", False)
    motion_ok   = (temporal_result or {}).get("motion_ok",   True)
    freq_ok     = (replay_result   or {}).get("freq_ok",     True)

    if security_mode == "strict":
        is_live = blink_passed and tex_ok and lap_ok and motion_ok and freq_ok
    elif security_mode == "relaxed":
        is_live = blink_passed
        if susp_static and not motion_ok:   # even relaxed catches completely static frames
            is_live = False
    else:  # normal
        is_live = blink_passed and tex_ok and lap_ok
        if is_live and susp_static:         # temporal check downgrades suspicious statics
            is_live = False

    if is_live:
        status = "LIVE"
    elif not blink_passed:
        status = "WAITING FOR BLINK"
    else:
        status = "SPOOF SUSPECTED"

    return {
        "is_live":           is_live,
        "status":            status,
        "texture_ok":        tex_ok,
        "laplacian_ok":      lap_ok,
        "motion_ok":         motion_ok,
        "freq_ok":           freq_ok,
        "blink_passed":      blink_passed,
        "suspicious_static": susp_static,
        "texture_variance":  static_result.get("texture_variance", 0.0),
        "laplacian_score":   static_result.get("laplacian_score",  0.0),
        "ear":               static_result.get("ear"),
    }
