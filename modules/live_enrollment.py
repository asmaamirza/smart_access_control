"""
Live Camera Enrollment
=======================

Replaces the old "upload 5 images → average encoding / train KNN" flow with a
real-time, webcam-driven enrollment pipeline:

    look straight → turn left → turn right → look up → look down

For every pose the user is guided through, we pull live frames, run them
through the SAME quality gates already used elsewhere in the system —

    face_recognition.face_locations()      (face must be present)
    static_spoof_check()                   (anti-spoofing: texture + sharpness)
    a cheap landmark-based pose heuristic   (confirms the requested pose)

— and only keep frames that pass all three. Once a pose has enough good
samples, we move to the next pose. When every pose is done we average every
collected 128-d face_recognition encoding into one robust identity vector and
write it straight into the DB via `update_face_encoding()` / `create_user()`.

──────────────────────────────────────────────────────────────────────────────
WHY ACCEPTED FRAMES ARE ALSO SAVED TO known_faces/ (read before removing this)
──────────────────────────────────────────────────────────────────────────────
The 128-d ResNet encoding (averaged, stored in the DB) is and remains the
ONLY thing that drives access decisions anywhere in this app — recognition
logic in face_recognizer.py is untouched and does not read known_faces/.

However, modules/knn_engine.py trains a *separate* classifier on classical
Canny+LBP+HOG feature vectors, and its own docstring states this exists to
satisfy the project requirement of having at least one model trained on
custom data. That feature extraction (feature_extractor.py: Canny edges, LBP
texture, HOG gradients) operates on raw pixels — there is no embedding-based
substitute for it, so train_knn() genuinely needs JPEG files in
known_faces/<username>/ to have anything to train on.

So: every frame that passes all three quality gates below gets saved as a
JPEG into known_faces/<username>/ in addition to having its encoding
collected. This is a deliberate, narrow exception to "nothing touches disk"
— these are frames the system already judged good enough to enroll with, the
write only happens for accepted samples (not raw unfiltered footage), and it
exists solely so the classical-CV/KNN system the rubric requires has
something real to train on. If that requirement goes away, this save step
(and the train_knn() call in finalize()) can be deleted with no effect on
recognition — the encoding path doesn't depend on it.

State machine
-------------
This module is UI-framework agnostic: it exposes a single `EnrollmentSession`
class that the Streamlit page drives frame-by-frame. All the camera loop /
st.rerun() plumbing stays in app.py, consistent with how `_run_pipeline_live`
and `_run_terminal_pipeline` are already structured there.

──────────────────────────────────────────────────────────────────────────────
KNOWN LIMITATION — "look down" pose (read before adjusting thresholds again)
──────────────────────────────────────────────────────────────────────────────
dlib's HOG frontal-face detector (used by face_recognition.face_locations)
degrades fastest on downward pitch: the brow ridge self-shadows the eye
region the detector relies on most, so a moderate "chin down" tilt already
returns zero detections — there is often no tilt angle that is simultaneously
"enough to register as down" and "still detected at all." This is a detector
limitation, not a threshold-tuning problem; raising/lowering
PITCH_TILT_RATIO alone cannot fix it because the failure happens upstream of
the ratio check, in face_locations() itself returning nothing.

This version mitigates it in two ways:
  1. Pitch now uses chin-vs-eye-line landmarks instead of the nose tip — the
     nose tip barely moves vertically for a given pitch angle (it mostly
     moves toward/away from camera), while the chin point sweeps much
     further, so the *signal* is bigger before the detector gives out.
  2. PITCH_TILT_RATIO is now asymmetric and tuned lower than the yaw
     threshold, and the down case has its own (lower) threshold than up,
     since "down" both produces a smaller geometric signal AND is the pose
     most likely to be lost by the detector — needing the threshold tripped
     on a smaller, earlier tilt.
If "down" still won't register at all (status flips straight from
"pose_mismatch" to "no_face" with nothing detected in between), that's the
detector failing outright — see the UI message change below, which now
distinguishes "tilt down a bit more" from "tilt down less, you're losing
detection" so the person isn't stuck guessing which direction to adjust.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import face_recognition as fr

from modules.anti_spoofing import static_spoof_check
from modules.database import create_user, update_face_encoding
from modules.knn_engine import train_knn

KNOWN_FACES_DIR = "database/known_faces"

# ── Pose sequence ─────────────────────────────────────────────────────────────
# Each pose: (key, instruction shown to the user, validator name)
POSE_SEQUENCE = [
    ("straight", "Look straight at the camera"),
    ("left",     "Turn your head to the LEFT"),
    ("right",    "Turn your head to the RIGHT"),
    ("up",       "Tilt your head UP"),
    ("down",     "Tilt your head DOWN — just slightly, a little goes a long way"),
]

# Samples required per pose before advancing
SAMPLES_PER_POSE = 6

# Minimum frames between accepted captures (avoids grabbing near-duplicate
# frames every tick of the camera loop)
MIN_CAPTURE_GAP_FRAMES = 3

# ── Pose heuristic thresholds ─────────────────────────────────────────────────
# These operate on landmark offsets relative to the eye midpoint / inter-eye
# distance, normalised so they're roughly scale-invariant. This is a
# lightweight stand-in for full head-pose estimation (no solvePnP / 3D model
# in this codebase) — adequate to confirm gross pose, not a precise
# yaw/pitch measurement.
YAW_TURN_RATIO       = 0.18   # nose-tip x-offset from eye midpoint / inter_eye_dist
PITCH_UP_RATIO        = 0.12  # chin-to-eye-line vertical ratio decrease, "up"
PITCH_DOWN_RATIO      = 0.08  # chin-to-eye-line vertical ratio increase, "down"
# DOWN gets a lower bar than UP on purpose: the detector loses the face
# sooner on downward tilt, so we must accept the pose at a smaller, earlier
# angle than we do for "up" — there's a narrower window where it's both
# detected AND distinguishable from straight.


@dataclass
class PoseProgress:
    key: str
    instruction: str
    collected: int = 0
    target: int = SAMPLES_PER_POSE
    done: bool = False


@dataclass
class EnrollmentSession:
    """
    Drives one user through the full pose sequence and accumulates encodings.

    Usage (per camera frame, from the Streamlit page):

        result = session.process_frame(frame_bgr)
        # result.status tells you what to render:
        #   "no_face" | "spoof_rejected" | "pose_mismatch" |
        #   "captured" | "pose_complete" | "enrollment_complete"
    """
    username: str
    name: str
    role: str

    poses: list[PoseProgress] = field(default_factory=lambda: [
        PoseProgress(key=k, instruction=instr) for k, instr in POSE_SEQUENCE
    ])
    current_pose_idx: int = 0
    encodings: list = field(default_factory=list)
    saved_image_paths: list = field(default_factory=list)  # mirrors `encodings` 1:1
    _frame_n: int = 0
    _last_capture_frame: int = -999
    _straight_baseline_chin_ratio: float | None = None
    _consecutive_no_face: int = 0

    # ── Public state ───────────────────────────────────────────────────────
    @property
    def current_pose(self) -> PoseProgress | None:
        if self.current_pose_idx >= len(self.poses):
            return None
        return self.poses[self.current_pose_idx]

    @property
    def is_complete(self) -> bool:
        return self.current_pose_idx >= len(self.poses)

    @property
    def total_collected(self) -> int:
        return len(self.encodings)

    @property
    def total_target(self) -> int:
        return sum(p.target for p in self.poses)

    # ── Frame processing ─────────────────────────────────────────────────────
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Run one frame through detection → spoof-check → pose-check → capture.

        Returns a dict:
            status:   "no_face" | "blurry" | "spoof_rejected" |
                      "pose_mismatch" | "captured" | "pose_complete" |
                      "enrollment_complete" | "cooldown"
            face_box: {top,right,bottom,left} or None
            message:  human-readable status line
            pose:     current PoseProgress (or None if already complete)
        """
        self._frame_n += 1
        pose = self.current_pose

        if pose is None:
            return self._result("enrollment_complete", None,
                                "All poses captured — ready to finalise.")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locs = fr.face_locations(rgb, model="hog")
        if not locs:
            self._consecutive_no_face += 1
            # If we were *just* getting pose_mismatch on this pose and now we
            # have lost detection entirely, the person likely over-rotated —
            # most relevant for "down", where the detector gives out fast.
            if pose.key == "down" and self._consecutive_no_face <= 5 and self._frame_n > 1:
                return self._result(
                    "no_face", None,
                    "Lost the face — that's too far. Ease off the tilt, just a small dip.",
                    pose,
                )
            return self._result("no_face", None, "No face detected — center your face.", pose)

        self._consecutive_no_face = 0

        # Use the largest detected face (closest / most prominent)
        loc = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
        top, right, bottom, left = loc
        face_box = {"top": top, "right": right, "bottom": bottom, "left": left}

        # Respect a short cooldown so we don't grab near-identical consecutive frames
        if self._frame_n - self._last_capture_frame < MIN_CAPTURE_GAP_FRAMES:
            return self._result("cooldown", face_box, "Hold the pose…", pose)

        # ── Quality gate 1: anti-spoofing (texture + sharpness) ──────────────
        lm_list = fr.face_landmarks(rgb, [loc])
        lm = lm_list[0] if lm_list else {}

        roi = frame_bgr[top:bottom, left:right]
        if roi.size == 0:
            return self._result("no_face", face_box, "Face crop invalid — reposition.", pose)
        gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))

        spoof = static_spoof_check(gray, lm)
        if not spoof.get("is_live", True):
            return self._result("spoof_rejected", face_box,
                                f"Liveness check failed ({spoof.get('status', 'SPOOF SUSPECTED')}). "
                                "Use a real camera feed, good lighting.", pose)
        if not spoof.get("laplacian_ok", True):
            return self._result("blurry", face_box,
                                "Image too blurry — hold still.", pose)

        # ── Quality gate 2: pose confirmation ────────────────────────────────
        matched, hint = self._pose_matches(pose.key, lm)
        if not matched:
            return self._result("pose_mismatch", face_box, hint or pose.instruction, pose)

        # ── Capture ───────────────────────────────────────────────────────────
        encs = fr.face_encodings(rgb, [loc])
        if not encs:
            return self._result("no_face", face_box, "Could not encode face — try again.", pose)

        self.encodings.append(encs[0])
        pose.collected += 1
        self._last_capture_frame = self._frame_n

        # Save the accepted crop to disk too — purely so knn_engine.train_knn()
        # has real images to train its classical-feature classifier on (see
        # module docstring). This does NOT feed recognition; the encoding
        # appended above is the only thing that drives access decisions.
        saved_path = self._save_sample_image(frame_bgr, face_box, pose.key, pose.collected)
        self.saved_image_paths.append(saved_path)

        if pose.collected >= pose.target:
            pose.done = True
            self.current_pose_idx += 1
            if self.current_pose_idx >= len(self.poses):
                return self._result("enrollment_complete", face_box,
                                    "All poses captured — ready to finalise.")
            next_pose = self.poses[self.current_pose_idx]
            return self._result("pose_complete", face_box,
                                f"'{pose.instruction}' done. Next: {next_pose.instruction}", pose)

        return self._result("captured", face_box,
                            f"Captured {pose.collected}/{pose.target} for '{pose.instruction}'.", pose)

    def _save_sample_image(self, frame_bgr: np.ndarray, face_box: dict,
                           pose_key: str, sample_n: int) -> str | None:
        """
        Save the accepted face crop to known_faces/<username>/<pose>_<NN>.jpg.

        Crops with a small margin (not just the tight box) so train_knn()'s
        own face_locations() + extract_all() pipeline has enough context to
        re-detect and align the face, same as it would for any other image
        in that folder. Returns the saved path, or None if the write failed
        (failure here must never break enrollment — it only weakens the
        diagnostic KNN's training set, not recognition).
        """
        try:
            user_dir = os.path.join(KNOWN_FACES_DIR, self.username)
            os.makedirs(user_dir, exist_ok=True)

            h, w = frame_bgr.shape[:2]
            margin = int(0.25 * max(face_box["bottom"] - face_box["top"],
                                    face_box["right"] - face_box["left"]))
            top    = max(face_box["top"] - margin, 0)
            bottom = min(face_box["bottom"] + margin, h)
            left   = max(face_box["left"] - margin, 0)
            right  = min(face_box["right"] + margin, w)
            crop = frame_bgr[top:bottom, left:right]
            if crop.size == 0:
                return None

            filename = f"{pose_key}_{sample_n:02d}.jpg"
            path = os.path.join(user_dir, filename)
            ok = cv2.imwrite(path, crop)
            return path if ok else None
        except Exception:
            return None

    # ── Pose heuristic ────────────────────────────────────────────────────────
    def _pose_matches(self, pose_key: str, landmarks: dict) -> tuple[bool, str | None]:
        """
        Lightweight landmark-based pose check (no solvePnP head-pose model in
        this codebase).

        Yaw (left/right) uses nose-tip x-offset from the eye midpoint — this
        is a strong, early signal for left/right turns.

        Pitch (up/down) uses the CHIN point's vertical distance below the eye
        line, not the nose tip. The nose tip moves mostly toward/away from
        the camera as the head pitches (small vertical pixel delta), while
        the chin sweeps a much larger vertical arc for the same rotation —
        a bigger, earlier signal, which matters because "down" in particular
        loses face detection quickly if you over-rotate.

        Returns (matched, hint_message_or_None). hint is only populated for
        the "down" pose mismatch, where the direction to adjust is genuinely
        ambiguous from the message alone otherwise.
        """
        chin = landmarks.get("chin")
        l_eye = landmarks.get("left_eye")
        r_eye = landmarks.get("right_eye")
        nose = landmarks.get("nose_tip")

        if pose_key == "straight":
            # Straight just requires a detected, reasonably centred face —
            # already guaranteed by detection succeeding. Always accept and
            # use it to set the pitch baseline for up/down comparisons.
            if chin and l_eye and r_eye:
                l_eye_c = np.mean(l_eye, axis=0)
                r_eye_c = np.mean(r_eye, axis=0)
                eye_mid = (l_eye_c + r_eye_c) / 2.0
                inter_eye = max(np.linalg.norm(r_eye_c - l_eye_c), 1e-5)
                # Chin tip is the bottom-most point of the jaw outline
                chin_tip = max(chin, key=lambda p: p[1])
                self._straight_baseline_chin_ratio = (chin_tip[1] - eye_mid[1]) / inter_eye
            return True, None

        if not (l_eye and r_eye and nose):
            return False, None

        l_eye_c = np.mean(l_eye, axis=0)
        r_eye_c = np.mean(r_eye, axis=0)
        nose_c = np.mean(nose, axis=0)
        eye_mid = (l_eye_c + r_eye_c) / 2.0
        inter_eye = max(np.linalg.norm(r_eye_c - l_eye_c), 1e-5)

        if pose_key in ("left", "right"):
            yaw_ratio = (nose_c[0] - eye_mid[0]) / inter_eye
            if pose_key == "left":
                # Image-mirror note: in a selfie-style feed, "turn head left"
                # (the user's left) moves the nose toward the camera's right
                # in the unflipped frame — i.e. positive x offset.
                return yaw_ratio > YAW_TURN_RATIO, None
            return yaw_ratio < -YAW_TURN_RATIO, None

        # ── up / down: chin-line pitch signal ────────────────────────────────
        if not chin:
            return False, None
        chin_tip = max(chin, key=lambda p: p[1])
        chin_ratio = (chin_tip[1] - eye_mid[1]) / inter_eye
        baseline = self._straight_baseline_chin_ratio or chin_ratio

        if pose_key == "up":
            # Tilting up brings the chin closer to the eye line (ratio drops)
            return (baseline - chin_ratio) > PITCH_UP_RATIO, None

        if pose_key == "down":
            # Tilting down pushes the chin further below the eye line
            delta = chin_ratio - baseline
            if delta > PITCH_DOWN_RATIO:
                return True, None
            return False, "Tilt your head down just a little more…"

        return False, None

    def _result(self, status: str, face_box, message: str, pose: PoseProgress | None = None) -> dict:
        return {
            "status": status,
            "face_box": face_box,
            "message": message,
            "pose": pose if pose is not None else self.current_pose,
            "total_collected": self.total_collected,
            "total_target": self.total_target,
        }

    # ── Finalisation ──────────────────────────────────────────────────────────
    def finalize(self, password: str | None = None) -> tuple[bool, str]:
        """
        Average all collected encodings into one robust identity vector and
        persist it. If `password` is given and the user doesn't exist yet,
        creates the user record too; otherwise just updates the encoding for
        an existing username (re-enrollment).

        After a successful save, retrains the diagnostic KNN classifier
        (modules/knn_engine.py) on the images this session just wrote to
        known_faces/ — this is the only consumer of those saved JPEGs;
        recognition itself never reads them. A KNN retrain failure is
        reported but does not flip the overall result to failure: the
        encoding is the part that matters for access control and it already
        succeeded by the time train_knn() runs.

        Returns (success, message).
        """
        if not self.encodings:
            return False, "No valid face samples were collected."
        if len(self.encodings) < self.total_target:
            return False, (
                f"Only {len(self.encodings)}/{self.total_target} samples collected — "
                "finish all poses before finalising."
            )

        final_encoding = np.mean(np.stack(self.encodings, axis=0), axis=0)

        if password is not None:
            ok, msg = create_user(self.name, self.username, password, self.role, final_encoding)
        else:
            update_face_encoding(self.username, final_encoding)
            ok, msg = True, f"Face encoding updated for '{self.username}' from {len(self.encodings)} samples."

        if not ok:
            return ok, msg

        knn_ok, knn_msg = train_knn()
        if knn_ok:
            msg += f" KNN classifier retrained ({knn_msg})."
        else:
            msg += f" (Note: KNN retrain skipped — {knn_msg})"

        return True, msg

    def reset_current_pose(self) -> None:
        """Allow the UI to let the user redo the in-progress pose from scratch."""
        pose = self.current_pose
        if pose is None:
            return
        if pose.collected:
            # Drop the encodings AND the saved image files captured for this
            # pose (the most recent N of each) so a "redo" doesn't leave
            # orphaned training images on disk with no matching encoding.
            self.encodings = self.encodings[: len(self.encodings) - pose.collected]
            stale_paths = self.saved_image_paths[len(self.saved_image_paths) - pose.collected:]
            for p in stale_paths:
                if p:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            self.saved_image_paths = self.saved_image_paths[: len(self.saved_image_paths) - pose.collected]
        pose.collected = 0
        pose.done = False