"""
Classical Computer Vision Feature Extractor
============================================

Four sequential stages applied to every detected face:

1. Face Alignment
   Uses eye-centre landmarks to compute the tilt angle and applies an affine
   rotation so both eyes are level. Alignment is critical for LBP and HOG
   consistency: a tilted face changes the gradient orientations and shifts
   LBP patterns, degrading both descriptors significantly.

2. Canny Edge Detection  (Canny, 1986)
   Gaussian smoothing → Sobel gradients → non-maximum suppression →
   hysteresis double-threshold.  The resulting edge map captures the
   structural outline of facial features (eyes, nose, jaw) independent of
   skin tone or lighting intensity.

3. Local Binary Pattern — LBP  (Ojala et al., 2002)
   For each pixel, compare against P neighbours on a circle of radius R.
   Neighbours brighter than centre → 1; darker → 0.  The circular binary
   code is the pixel's LBP value.  The histogram of codes over the face
   encodes micro-texture and is used both for recognition and anti-spoofing
   (printed photos exhibit characteristically flatter LBP distributions).

4. Histogram of Oriented Gradients — HOG  (Dalal & Triggs, 2005)
   Divide image into small cells; bin gradient orientations weighted by
   magnitude; normalise across overlapping blocks (L2-Hys).  Captures
   shape structure robustly against small geometric distortions.

The three feature vectors are concatenated into a single descriptor that
complements the deep 128-d face_recognition embedding used for matching.
"""

import cv2
import numpy as np
import face_recognition
from skimage.feature import local_binary_pattern
from skimage.feature import hog as sk_hog

FACE_CHIP_SIZE = (64, 64)   # standard size for all classical descriptors
LBP_RADIUS     = 1
LBP_POINTS     = 8          # uniform LBP → P+2 = 10 bins


# ── Face alignment ────────────────────────────────────────────────────────────

def align_face(frame_bgr: np.ndarray, face_location: tuple):
    """
    Rotate the frame so the line connecting both eye centres is horizontal.

    Args:
        frame_bgr:     Full input frame in BGR colour order.
        face_location: (top, right, bottom, left) from face_recognition.

    Returns:
        aligned_frame (np.ndarray): rotated copy of the full frame.
        gray_chip     (np.ndarray): 64×64 grayscale face crop from aligned frame.
                                    None if the crop is empty.
        landmarks     (dict):       raw landmark dict for downstream drawing.
    """
    top, right, bottom, left = face_location
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    lm_list = face_recognition.face_landmarks(rgb, [face_location])
    landmarks = lm_list[0] if lm_list else {}

    aligned_frame = frame_bgr.copy()

    if "left_eye" in landmarks and "right_eye" in landmarks:
        l_eye = np.mean(landmarks["left_eye"],  axis=0)
        r_eye = np.mean(landmarks["right_eye"], axis=0)

        # Angle between eye centres
        angle = np.degrees(np.arctan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))

        mid = ((l_eye + r_eye) / 2).astype(int)
        h, w = frame_bgr.shape[:2]
        M = cv2.getRotationMatrix2D((int(mid[0]), int(mid[1])), angle, 1.0)
        aligned_frame = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR)

    # Crop and normalise face chip
    face_roi = aligned_frame[top:bottom, left:right]
    if face_roi.size == 0:
        return aligned_frame, None, landmarks

    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    # Histogram equalisation before descriptors for lighting robustness
    gray = cv2.equalizeHist(gray)
    gray_chip = cv2.resize(gray, FACE_CHIP_SIZE, interpolation=cv2.INTER_AREA)

    return aligned_frame, gray_chip, landmarks


# ── Classical descriptors ─────────────────────────────────────────────────────

def extract_canny(gray_chip: np.ndarray):
    """
    Apply Canny edge detector to a 64×64 grayscale face chip.

    Returns:
        edge_map   (np.ndarray uint8):  binary edge image for visualisation.
        feat_vec   (np.ndarray float32): flattened, normalised edge map.
    """
    blurred  = cv2.GaussianBlur(gray_chip, (3, 3), sigmaX=0)
    edge_map = cv2.Canny(blurred, threshold1=30, threshold2=80)
    feat_vec = edge_map.flatten().astype(np.float32) / 255.0
    return edge_map, feat_vec


def extract_lbp(gray_chip: np.ndarray):
    """
    Compute uniform LBP histogram over a 64×64 face chip.

    Returns:
        lbp_image  (np.ndarray float32): per-pixel LBP code image.
        hist_norm  (np.ndarray float32): normalised histogram (LBP_POINTS+2 bins).
    """
    lbp_img  = local_binary_pattern(gray_chip, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    n_bins   = LBP_POINTS + 2
    hist, _  = np.histogram(lbp_img.ravel(), bins=n_bins, range=(0, n_bins))
    hist_f   = hist.astype(np.float32)
    hist_f  /= hist_f.sum() + 1e-7
    return lbp_img.astype(np.float32), hist_f


def extract_hog(gray_chip: np.ndarray):
    """
    Compute HOG descriptor and visualisation for a 64×64 face chip.

    Returns:
        hog_vis  (np.ndarray uint8):   gradient magnitude image for display.
        hog_vec  (np.ndarray float32): HOG feature vector.
    """
    hog_vec, hog_img = sk_hog(
        gray_chip,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    # Normalise visualisation to uint8
    hog_vis = cv2.normalize(hog_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return hog_vis, hog_vec.astype(np.float32)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def extract_all(frame_bgr: np.ndarray, face_location: tuple) -> dict | None:
    """
    Run the complete classical feature extraction pipeline for one face.

    Returns a dict with everything needed for display and matching, or None
    if no valid face chip could be extracted.

    Keys:
        aligned_frame  — rotation-corrected full frame (BGR)
        gray_chip      — 64×64 normalised grayscale face
        canny_map      — Canny edge image (uint8)
        lbp_image      — LBP-coded image (float32, for visualisation)
        hog_vis        — HOG gradient image (uint8)
        feature_vector — concatenated [Canny + LBP + HOG] descriptor
        landmarks      — raw face_recognition landmark dict
    """
    aligned_frame, gray_chip, landmarks = align_face(frame_bgr, face_location)
    if gray_chip is None:
        return None

    canny_map, canny_vec = extract_canny(gray_chip)
    lbp_image, lbp_vec   = extract_lbp(gray_chip)
    hog_vis,   hog_vec   = extract_hog(gray_chip)

    feature_vector = np.concatenate([canny_vec, lbp_vec, hog_vec])

    return {
        "aligned_frame":  aligned_frame,
        "gray_chip":      gray_chip,
        "canny_map":      canny_map,
        "lbp_image":      lbp_image,
        "hog_vis":        hog_vis,
        "feature_vector": feature_vector,
        "landmarks":      landmarks,
    }
