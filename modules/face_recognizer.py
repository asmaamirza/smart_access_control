"""
Face Recognition Module
=======================

Detection:  face_recognition.face_locations() — HOG-based detector (fast CPU).
Embeddings: face_recognition.face_encodings() — dlib ResNet, 128-d vector.
Matching:   Euclidean distance against all stored encodings in the SQLite DB.
            Lower distance = more similar.  Typical threshold: 0.45–0.65.

No separate training step is required.  Adding or removing a user takes
effect immediately because the DB IS the learned reference set.

Enrolment helpers also live here so the registration pipeline stays in one place.
"""

import os
import numpy as np
import face_recognition as fr

from modules.database import (
    get_all_face_encodings, get_blacklist_encodings,
    get_user_by_username, get_user_face_encoding, update_face_encoding,
)

# Distance at which confidence normalisation reaches 0 %
_NORM_DIST = 0.80

# Per-mode maximum acceptable distance (maps to the RBAC threshold in rbac_engine)
_DIST_THRESH = {
    "strict":  0.50,
    "normal":  0.60,
    "relaxed": 0.70,
}


# ── Recognition ───────────────────────────────────────────────────────────────

def recognize_faces(rgb_image: np.ndarray, security_mode: str = "normal") -> list[dict]:
    """
    Detect and match every face in an RGB image against the database.

    Returns list of dicts:
        top, right, bottom, left  — face bounding box
        match                     — user dict from DB, or None
        confidence                — float in [0, 1]
    """
    dist_thresh = _DIST_THRESH.get(security_mode, 0.55)

    face_locations = fr.face_locations(rgb_image, model="hog")
    if not face_locations:
        return []

    face_encs = fr.face_encodings(rgb_image, face_locations)
    db_users  = get_all_face_encodings()

    results = []
    for (top, right, bottom, left), enc in zip(face_locations, face_encs):
        match      = None
        confidence = 0.0

        if db_users:
            known = [u["encoding"] for u in db_users]
            dists = fr.face_distance(known, enc)
            best  = int(np.argmin(dists))
            dist  = float(dists[best])

            # Confidence: linearly mapped so dist=0 → 100%, dist=_NORM_DIST → 0%
            confidence = max(0.0, 1.0 - dist / _NORM_DIST)

            if dist <= dist_thresh:
                match = db_users[best]

        results.append({
            "top": top, "right": right, "bottom": bottom, "left": left,
            "match": match, "confidence": confidence,
        })

    return results


# ── Enrolment helpers ─────────────────────────────────────────────────────────

def extract_average_encoding(image_paths: list[str]) -> np.ndarray | None:
    """
    Extract one averaged 128-d face encoding from a list of image files.

    Averages all valid per-image encodings so a single representative vector
    captures the person across different lighting / pose conditions.
    Returns None if no faces could be extracted.
    """
    encodings = []
    for path in image_paths:
        try:
            img  = fr.load_image_file(path)
            encs = fr.face_encodings(img)
            if encs:
                encodings.append(encs[0])
        except Exception:
            continue

    if not encodings:
        return None
    return np.mean(encodings, axis=0)


def extract_encoding_from_array(rgb_image: np.ndarray) -> np.ndarray | None:
    """Extract a 128-d encoding from an RGB numpy array. Returns None if no face found."""
    encs = fr.face_encodings(rgb_image)
    return encs[0] if encs else None


def recognize_with_blacklist_check(rgb_image: np.ndarray,
                                   security_mode: str = "normal") -> list[dict]:
    """
    Detect faces and match them — blacklist is checked FIRST for every face.

    If a face matches the blacklist it is flagged immediately; normal user
    matching is skipped for that face entirely.

    Returns list of dicts (same shape as recognize_faces, plus):
        blacklisted          — True if face matched blacklist
        blacklist_entry      — {id, name, threat_reason} or None
        blacklist_confidence — float
    """
    dist_thresh = _DIST_THRESH.get(security_mode, 0.60)

    face_locations = fr.face_locations(rgb_image, model="hog")
    if not face_locations:
        return []

    face_encs        = fr.face_encodings(rgb_image, face_locations)
    blacklist_entries = get_blacklist_encodings()
    db_users          = get_all_face_encodings()

    results = []
    for (top, right, bottom, left), enc in zip(face_locations, face_encs):

        # ── Priority 1: blacklist check ───────────────────────────────────────
        is_bl    = False
        bl_entry = None
        bl_conf  = 0.0

        if blacklist_entries:
            bl_known = [e["encoding"] for e in blacklist_entries]
            bl_dists = fr.face_distance(bl_known, enc)
            bl_best  = int(np.argmin(bl_dists))
            bl_dist  = float(bl_dists[bl_best])
            bl_conf  = max(0.0, 1.0 - bl_dist / _NORM_DIST)
            # Blacklist always uses strict threshold (0.50) regardless of security mode
            if bl_dist <= 0.50:
                is_bl    = True
                bl_entry = blacklist_entries[bl_best]

        if is_bl:
            results.append({
                "top": top, "right": right, "bottom": bottom, "left": left,
                "match": None, "confidence": 0.0,
                "blacklisted": True,
                "blacklist_entry": bl_entry,
                "blacklist_confidence": bl_conf,
            })
            continue   # do NOT compare against authorized users

        # ── Priority 2: normal authorized-user matching ───────────────────────
        match      = None
        confidence = 0.0

        if db_users:
            known = [u["encoding"] for u in db_users]
            dists = fr.face_distance(known, enc)
            best  = int(np.argmin(dists))
            dist  = float(dists[best])
            confidence = max(0.0, 1.0 - dist / _NORM_DIST)
            if dist <= dist_thresh:
                match = db_users[best]

        results.append({
            "top": top, "right": right, "bottom": bottom, "left": left,
            "match": match, "confidence": confidence,
            "blacklisted": False,
            "blacklist_entry": None,
            "blacklist_confidence": 0.0,
        })

    return results


def check_blacklist(face_enc: np.ndarray,
                    dist_thresh: float = 0.50) -> tuple[bool, dict | None, float]:
    """
    Check a single pre-computed face encoding against all blacklist entries.
    Returns (is_blacklisted, matched_entry, confidence).
    Used by the Security Terminal pipeline.
    """
    entries = get_blacklist_encodings()
    if not entries:
        return False, None, 0.0

    bl_known = [e["encoding"] for e in entries]
    dists    = fr.face_distance(bl_known, face_enc)
    best     = int(np.argmin(dists))
    dist     = float(dists[best])
    conf     = max(0.0, 1.0 - dist / _NORM_DIST)

    if dist <= dist_thresh:
        return True, entries[best], conf
    return False, None, conf


def verify_claimed_user(face_enc: np.ndarray, username: str,
                        security_mode: str = "normal") -> tuple[dict | None, float]:
    """
    Verify face_enc matches the encoding stored for the claimed username.

    This is a targeted 1-vs-1 match: much more efficient than comparing
    against all enrolled users, and eliminates cross-user false positives.

    Returns (user_dict, confidence) — user_dict is None if the face does not
    match the claimed user within the current security mode threshold.
    """
    user = get_user_by_username(username)
    if user is None:
        return None, 0.0

    stored_enc = get_user_face_encoding(username)
    if stored_enc is None:
        return None, 0.0

    dist_thresh = _DIST_THRESH.get(security_mode, 0.60)
    dist        = float(fr.face_distance([stored_enc], face_enc)[0])
    confidence  = max(0.0, 1.0 - dist / _NORM_DIST)

    if dist <= dist_thresh:
        return {"username": user["username"], "name": user["name"], "role": user["role"]}, confidence
    return None, confidence


def enroll_from_folder(username: str, folder_path: str) -> tuple[bool, str]:
    """
    Extract and store an averaged face encoding from all images in a folder.
    Assumes the user record already exists in the DB.
    """
    if not os.path.isdir(folder_path):
        return False, f"Folder not found: {folder_path}"

    paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not paths:
        return False, "No image files found in folder."

    enc = extract_average_encoding(paths)
    if enc is None:
        return False, f"No faces detected in any of the {len(paths)} image(s)."

    update_face_encoding(username, enc)
    return True, f"Enrolled from {len(paths)} image(s)."
