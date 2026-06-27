"""
Smart Access Control System — CSCI435 Project
Entry point: streamlit run app.py
"""

import os
import shutil
import secrets
import tempfile
import time

import cv2
import face_recognition
import numpy as np
import streamlit as st

from modules.anti_spoofing    import (
    BlinkTracker, static_spoof_check,
    temporal_spoof_check, replay_artifact_check, aggregate_spoof_result,
)
from modules.background_model import apply_clahe, detect_motion, reset_background_model
from modules.database         import (
    blacklist_count, clear_access_log, create_blacklist_entry, create_user,
    delete_blacklist_entry, delete_user, get_access_log, get_all_blacklist_entries,
    get_all_face_encodings, get_all_users, get_log_stats, init_db, log_access_event,
    update_password, update_role, update_face_encoding, user_count, verify_credentials,
)
from modules.face_recognizer  import (
    check_blacklist, extract_average_encoding, recognize_with_blacklist_check,
    verify_claimed_user,
)
from modules.feature_extractor import extract_all
from modules.knn_engine        import knn_info, knn_is_ready, predict_knn, train_knn
from modules.person_tracker    import detect_and_track, reset_tracker
from modules.rbac_engine       import make_blacklist_decision, make_decision, ROLE_COLORS
from modules.utils             import (
    draw_landmarks, draw_motion_regions, draw_person_tracks,
    draw_rbac_result, draw_scanning_bar, draw_spoof_badge, stamp_ear, stamp_status,
)

KNOWN_FACES_DIR = "known_faces"

# ── Stability / stillness gate constants ──────────────────────────────────────
STABILITY_MOVEMENT_THRESHOLD = 12    # max center-pixel displacement between frames
STABILITY_REQUIRED_FRAMES    = 3     # consecutive stable frames before proceeding
STABILITY_SCALE_THRESHOLD    = 0.15  # max fractional change in face box height

# ── Security Terminal frame-interval constants ─────────────────────────────────
TERMINAL_FACE_DETECT_INTERVAL = 2    # HOG face detection every N frames (pre-scan)
TERMINAL_SCAN_FRAMES          = 25   # scanning animation frames before backend checks

# ── Test sample directories ───────────────────────────────────────────────────
TEST_DIR          = "test"
TEST_IMAGE_DIR    = os.path.join(TEST_DIR, "image_detection")
TEST_SPOOF_DIR    = os.path.join(TEST_DIR, "spoof_samples")
TEST_TAILGATE_DIR = os.path.join(TEST_DIR, "tailgating_samples")

_SPOOF_FALLBACK = {
    "is_live": True, "texture_ok": True, "laplacian_ok": True,
    "texture_variance": 0.0, "laplacian_score": 0.0, "ear": None,
    "status": "LIVE",
}

# Run live detection on a downsampled copy — 4x fewer pixels => ~4x faster HOG
_PROC_SCALE  = 0.5
_CLAHE_EVERY = 4   # reuse enhanced frame for this many frames before recomputing

# ── Bootstrap ─────────────────────────────────────────────────────────────────
init_db()
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
for _d in (TEST_IMAGE_DIR, TEST_SPOOF_DIR, TEST_TAILGATE_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Access Control",
    page_icon=":material/lock:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='padding:0.4rem 0 0.1rem 0'>"
    "<span style='font-size:1.25rem;font-weight:700;line-height:1.3'>Smart Access Control</span><br>"
    "<span style='font-size:0.85rem;color:gray'>CSCI435 — Computer Vision Project</span>"
    "</div>"
    "<hr style='margin:0.5rem 0 0.4rem 0;border:none;border-top:1px solid #e0e0e0'>",
    unsafe_allow_html=True,
)

_NAV_ITEMS = [
    ("Home",              ":material/home:"),
    ("Security Terminal", ":material/fingerprint:"),
    ("Register User",     ":material/person_add:"),
    ("Identify: Image",   ":material/image_search:"),
    ("Live Camera",       ":material/videocam:"),
    ("Admin Panel",       ":material/admin_panel_settings:"),
    ("Access Log",        ":material/assignment:"),
]

if "page" not in st.session_state:
    st.session_state["page"] = "Home"

for _pg_name, _pg_icon in _NAV_ITEMS:
    _is_active = st.session_state["page"] == _pg_name
    if st.sidebar.button(
        _pg_name,
        icon=_pg_icon,
        use_container_width=True,
        key=f"nav_{_pg_name}",
        type="primary" if _is_active else "secondary",
    ):
        st.session_state["page"] = _pg_name
        st.rerun()

page = st.session_state["page"]

st.sidebar.divider()
n  = user_count()
bl = blacklist_count()
st.sidebar.metric("Enrolled users", n)
st.sidebar.metric("Blacklisted",    bl)
if n == 0:
    st.sidebar.warning("No users enrolled. Go to Register User first.")

st.sidebar.divider()
st.sidebar.markdown("**System health**")
_dot = lambda ok, label: st.sidebar.markdown(
    f"<span style='color:{'#22cc44' if ok else '#cc2222'};font-size:1.1rem'>●</span> {label}",
    unsafe_allow_html=True,
)
try:
    user_count(); _db_ok = True
except Exception:
    _db_ok = False
_dot(_db_ok, "Database")
_dot(knn_is_ready(), f"KNN model {'ready' if knn_is_ready() else 'not trained'}")
_cam_live = st.session_state.get("cam_running", False)
_dot(_cam_live, f"Camera {'active' if _cam_live else 'idle'}")


# ═══════════════════════════════════════════════════════════════════════════════
# STABILITY / STILLNESS GATE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_face_stability(
    prev_box: "dict | None",
    curr_box: dict,
    movement_threshold: float = STABILITY_MOVEMENT_THRESHOLD,
    scale_threshold:    float = STABILITY_SCALE_THRESHOLD,
) -> bool:
    """
    Return True when the face has not moved significantly since the last frame.
    Uses bounding-box centre displacement and height change as stability signals.
    """
    if prev_box is None:
        return False

    prev_cx = (prev_box["left"] + prev_box["right"]) / 2.0
    prev_cy = (prev_box["top"]  + prev_box["bottom"]) / 2.0
    curr_cx = (curr_box["left"] + curr_box["right"]) / 2.0
    curr_cy = (curr_box["top"]  + curr_box["bottom"]) / 2.0
    displacement = ((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2) ** 0.5

    prev_h = max(prev_box["bottom"] - prev_box["top"], 1)
    curr_h = curr_box["bottom"] - curr_box["top"]
    scale_change = abs(curr_h - prev_h) / prev_h

    return displacement < movement_threshold and scale_change < scale_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED CORE — anti-spoofing + RBAC decisions for a list of detected faces
# Used by both image pipeline and live pipeline to eliminate duplication.
# ═══════════════════════════════════════════════════════════════════════════════
def _decide_faces(faces: list, lm_list: list, frame_bgr: np.ndarray,
                  security_mode: str,
                  blink_tracker=None,
                  feat_list: list | None = None) -> tuple[list, list]:
    """
    Run anti-spoofing + RBAC for every detected face.
    Returns (decisions, spoofs) — one entry per face, in the same order.
    Does NOT draw or log; callers handle that differently.

    blink_tracker: if supplied (live mode), EAR is tracked and liveness gate applied.
    feat_list:     if supplied (image mode), pre-extracted gray chips are preferred.
    """
    decisions: list = []
    spoofs:    list = []

    for i, (face, lm) in enumerate(zip(faces, lm_list)):
        if face.get("blacklisted"):
            decisions.append(make_blacklist_decision(face["blacklist_entry"]))
            spoofs.append(dict(_SPOOF_FALLBACK))
            continue

        # Update blink tracker before spoof check so .passed reflects this frame
        if blink_tracker is not None and lm:
            blink_tracker.update(lm)

        # Prefer pre-extracted aligned chip; fall back to cropping the full frame
        gray = None
        if feat_list and i < len(feat_list) and feat_list[i]:
            gray = feat_list[i].get("gray_chip")
        if gray is None:
            roi = frame_bgr[face["top"]:face["bottom"], face["left"]:face["right"]]
            if roi.size > 0:
                gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))

        spoof = static_spoof_check(gray, lm) if gray is not None else dict(_SPOOF_FALLBACK)
        if blink_tracker is not None:
            spoof["is_live"] = spoof["is_live"] and blink_tracker.passed
        spoofs.append(spoof)

        decisions.append(make_decision(face["match"], face["confidence"],
                                       spoof["is_live"], security_mode))

    return decisions, spoofs


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE — full identification on one BGR frame  (image mode)
# ═══════════════════════════════════════════════════════════════════════════════
def _run_pipeline(frame_bgr: np.ndarray,
                  security_mode: str,
                  show_landmarks: bool,
                  source: str,
                  pipeline_ui=None):
    """
    Execute all pipeline stages on a single frame.
    pipeline_ui: optional object with a .write(msg) method for stage logging.
    Returns (annotated_frame, faces, decisions, spoof_results, features_list, persons, tailgating)
    """
    def _log(msg):
        if pipeline_ui:
            pipeline_ui.write(msg)

    result     = frame_bgr.copy()
    faces      = []
    decisions  = []
    spoofs     = []
    feat_list  = []
    persons    = []
    tailgating = False

    _log("Enhancing image (CLAHE)…")
    enhanced = apply_clahe(frame_bgr)

    _log("Detecting motion (MOG2 + morphological ops)…")
    fg_mask, motion_regions = detect_motion(frame_bgr)
    draw_motion_regions(result, motion_regions)

    _log("Detecting faces (HOG)…")
    rgb       = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    face_locs = face_recognition.face_locations(rgb, model="hog")

    _log(f"Extracting landmarks ({len(face_locs)} face(s) found)…")
    all_lm = face_recognition.face_landmarks(rgb, face_locs) if face_locs else []

    _log("Extracting classical features (Canny / LBP / HOG)…")
    for loc in face_locs:
        feat_list.append(extract_all(frame_bgr, loc))

    _log("Checking blacklist + matching faces against database…")
    faces = recognize_with_blacklist_check(rgb, security_mode)

    _log("Anti-spoofing + RBAC decision…")
    pad_lm = all_lm if all_lm else [{}] * len(faces)
    decisions, spoofs = _decide_faces(faces, pad_lm, frame_bgr, security_mode,
                                      feat_list=feat_list)
    for face, decision, spoof, lm in zip(faces, decisions, spoofs, pad_lm):
        draw_rbac_result(result, face, decision)
        if not face.get("blacklisted"):
            draw_spoof_badge(result, face, spoof)
            if show_landmarks and lm:
                draw_landmarks(result, lm)

    _log("Tracking persons (YOLOv8 + ByteTrack)…")
    persons, tailgating = detect_and_track(result)
    draw_person_tracks(result, persons, tailgating)

    any_granted = any(d["action"] == "ALLOW" for d in decisions)
    stamp_status(result,
                 "ACCESS GRANTED" if any_granted else "ACCESS DENIED",
                 (0, 200, 0) if any_granted else (0, 0, 220))

    _log("Logging events…")
    for face, decision in zip(faces, decisions):
        log_access_event(
            detected_name=decision.get("name", "Unknown"),
            username=decision.get("username", ""),
            role=decision.get("role", "unknown"),
            confidence=face["confidence"],
            action=decision["action"],
            reason=decision.get("reason", ""),
            tailgating=tailgating,
            source=source,
        )
    if not faces:
        log_access_event("No face", "", "unknown", 0.0, "DENY",
                         "No face detected", tailgating, source)

    return result, faces, decisions, spoofs, feat_list, persons, tailgating, fg_mask


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMISED LIVE PIPELINE — tiered frame processing with caching
#
# Every frame:       downscale + MOG2 motion detection
# Every _CLAHE_EVERY frames: CLAHE enhancement (reuse cached result otherwise)
# Every face_every frames:   face recog + landmarks + spoof (when motion exists)
# Every yolo_every frames:   YOLO person / tailgating detection
# ═══════════════════════════════════════════════════════════════════════════════
def _run_pipeline_live(frame_bgr: np.ndarray,
                       security_mode: str,
                       show_landmarks: bool,
                       blink_tracker: BlinkTracker,
                       cache: dict) -> tuple:
    """
    Returns (annotated_frame, faces, decisions, spoofs, persons, tailgating, stage_log)
    stage_log entries: (name, "computed" | "cached" | "skipped")
    """
    frame_n    = cache.get("frame_n",    0)
    face_every = cache.get("face_every", 3)
    yolo_every = cache.get("yolo_every", 5)

    result    = frame_bgr.copy()
    stage_log = []

    # ── Every frame: downscale + MOG2 ────────────────────────────────────────
    S     = 1.0 / _PROC_SCALE
    small = cv2.resize(frame_bgr, (0, 0), fx=_PROC_SCALE, fy=_PROC_SCALE)

    _, motion_regions_s = detect_motion(small)
    motion_regions = [(int(x*S), int(y*S), int(w*S), int(h*S))
                      for x, y, w, h in motion_regions_s]
    draw_motion_regions(result, motion_regions)
    stage_log.append(("Motion detection", "computed"))

    has_motion = bool(motion_regions_s)
    cache_warm = "faces" in cache
    has_tracked_face = cache_warm and bool(cache.get("faces"))

    # ── CLAHE — cached, recomputed every _CLAHE_EVERY frames ─────────────────
    if frame_n % _CLAHE_EVERY == 0 or "enhanced_s" not in cache:
        enhanced_s = apply_clahe(small)
        cache["enhanced_s"] = enhanced_s
        stage_log.append(("Image enhancement", "computed"))
    else:
        enhanced_s = cache["enhanced_s"]
        stage_log.append(("Image enhancement", "cached"))

    # ── SLOW-1: face recognition — every N frames when motion present ─────────
    # Motion gating is only a perf optimization for an empty scene — once we're
    # actively tracking a face, keep updating on cadence even if they're still.
    run_face = (frame_n % face_every == 0) and (has_motion or not cache_warm or has_tracked_face)

    if run_face:
        rgb_s   = cv2.cvtColor(enhanced_s, cv2.COLOR_BGR2RGB)
        faces_s = recognize_with_blacklist_check(rgb_s, security_mode)

        faces = [{**f, "top": int(f["top"]*S), "right": int(f["right"]*S),
                  "bottom": int(f["bottom"]*S), "left": int(f["left"]*S)}
                 for f in faces_s]

        face_locs_s = [(f["top"], f["right"], f["bottom"], f["left"]) for f in faces_s]
        all_lm_s    = face_recognition.face_landmarks(rgb_s, face_locs_s) if faces_s else []
        all_lm      = [{k: [(int(x*S), int(y*S)) for x, y in pts]
                        for k, pts in lm.items()} for lm in all_lm_s]

        pad_lm_s = all_lm_s if all_lm_s else [{}] * len(faces)
        decisions, spoofs = _decide_faces(faces, pad_lm_s, frame_bgr, security_mode,
                                          blink_tracker=blink_tracker)

        # ── Stability tracking (first face box) ───────────────────────────────
        # in _run_pipeline_live's stability block
        if faces:
            curr_box = faces[0]
            is_stable_frame = compute_face_stability(cache.get("last_face_box"), curr_box)
            cache["last_face_box"] = curr_box
            cache["stable_count"] = (cache.get("stable_count", 0) + 1) if is_stable_frame \
                                    else max(0, cache.get("stable_count", 0) - 1)
            cache["is_stable"] = cache["stable_count"] >= STABILITY_REQUIRED_FRAMES
        else:
            cache["stable_count"] = 0
            cache["is_stable"]    = False
            cache.pop("last_face_box", None)

        # ── Temporal + replay spoof enhancement (first face only) ─────────────
        if faces and spoofs and not faces[0].get("blacklisted"):
            roi = frame_bgr[faces[0]["top"]:faces[0]["bottom"],
                            faces[0]["left"]:faces[0]["right"]]
            if roi.size > 0:
                curr_gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))
                t_result  = temporal_spoof_check(
                    cache.get("prev_gray_face"), curr_gray, cache, security_mode
                )
                r_result  = replay_artifact_check(curr_gray)
                agg       = aggregate_spoof_result(
                    spoofs[0], blink_tracker.passed, t_result, r_result, security_mode
                )
                spoofs[0] = agg
                # Override ALLOW→DENY when temporal/replay catches a spoof
                if not agg["is_live"] and decisions[0]["action"] == "ALLOW":
                    decisions[0] = make_decision(
                        faces[0].get("match"), faces[0].get("confidence", 0.0),
                        False, security_mode,
                    )

        # Live Camera is testing-only — access events are NOT logged here.
        # Logging happens only in Security Terminal (source="terminal") and
        # Identify Image (source="image").

        cache.update({"faces": faces, "decisions": decisions,
                      "spoofs": spoofs, "all_lm": all_lm,
                      "ear": blink_tracker.last_ear})
        stage_log += [
            ("Face recognition", "computed"),
            ("Anti-spoofing",    "computed"),
            ("RBAC decision",    "computed"),
        ]
    else:
        faces     = cache.get("faces",     [])
        decisions = cache.get("decisions", [])
        spoofs    = cache.get("spoofs",    [])
        all_lm    = cache.get("all_lm",    [])
        label = "cached" if cache_warm else "skipped"
        stage_log += [
            ("Face recognition", label),
            ("Anti-spoofing",    label),
            ("RBAC decision",    label),
        ]

    # ── SLOW-2: YOLO person tracking ──────────────────────────────────────────
    if frame_n % yolo_every == 0:
        persons, tailgating = detect_and_track(result)
        cache["persons"]    = persons
        cache["tailgating"] = tailgating
        stage_log.append(("Person tracking", "computed"))
    else:
        persons    = cache.get("persons",    [])
        tailgating = cache.get("tailgating", False)
        stage_log.append(("Person tracking", "cached"))

    # ── Draw overlays ──────────────────────────────────────────────────────────
    lm_pad = all_lm if all_lm else [{}] * len(faces)
    for face, decision, lm in zip(faces, decisions, lm_pad):
        draw_rbac_result(result, face, decision)
        if show_landmarks and lm:
            draw_landmarks(result, lm)
    draw_person_tracks(result, persons, tailgating)

    # ── Status stamp — stability hint takes priority ───────────────────────────
    is_stable  = cache.get("is_stable", False)
    any_granted = any(d["action"] == "ALLOW" for d in decisions)
    if faces and not is_stable:
        stamp_status(result, "HOLD STILL", (0, 200, 255))
    elif any_granted:
        stamp_status(result, "ACCESS GRANTED", (0, 200, 0))
    else:
        stamp_status(result, "ACCESS DENIED",  (0, 0, 220))

    stamp_ear(result, cache.get("ear", 1.0), blink_tracker.blink_count, blink_tracker.passed)

    return result, faces, decisions, spoofs, persons, tailgating, stage_log


# ── Verdict card (HTML only — Streamlit icon syntax does not work in raw HTML) ─
def _render_verdict(faces: list, decisions: list) -> str:
    any_granted = any(d["action"] == "ALLOW" for d in decisions)
    pd_dec      = decisions[0] if decisions else None

    if not faces:
        return (
            "<div style='background:#1e1e1e;padding:20px;border-radius:10px;"
            "text-align:center;border:2px solid #444'>"
            "<p style='font-size:2rem;margin:0'>—</p>"
            "<h3 style='color:#888;margin:6px 0'>NO FACE DETECTED</h3>"
            "</div>"
        )
    if any_granted:
        label = pd_dec.get("label", "") if pd_dec else ""
        return (
            "<div style='background:#0d2b0d;padding:20px;border-radius:10px;"
            "text-align:center;border:2px solid #00bb00'>"
            "<p style='font-size:2rem;color:#00dd00;margin:0'>PASS</p>"
            "<h3 style='color:#00dd00;margin:6px 0'>ACCESS GRANTED</h3>"
            f"<p style='color:#99ee99;margin:0;font-size:0.85rem'>{label}</p>"
            "</div>"
        )

    action = pd_dec["action"] if pd_dec else "DENY"
    label  = pd_dec.get("label", "No authorised face") if pd_dec else "No authorised face"
    title  = "SECURITY ALERT" if action == "ALERT" else "ACCESS DENIED"
    return (
        "<div style='background:#2b0d0d;padding:20px;border-radius:10px;"
        "text-align:center;border:2px solid #cc0000'>"
        "<p style='font-size:2rem;color:#ee2222;margin:0'>FAIL</p>"
        f"<h3 style='color:#ee2222;margin:6px 0'>{title}</h3>"
        f"<p style='color:#ffaaaa;margin:0;font-size:0.85rem'>{label}</p>"
        "</div>"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE CAMERA FRAGMENT
# Defined at module level so Streamlit tracks it across reruns.
# st.rerun() inside a fragment reruns only the fragment, not the full script.
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _live_camera_loop():
    if not st.session_state.get("cam_running", False):
        st.info("Press **Start** to begin live access control.")
        return

    hq       = st.session_state.get("live_hq",   True)
    sec_mode = "normal" if hq else "relaxed"
    show_lm  = st.session_state.get("live_lm",   True)
    face_ev  = st.session_state.get("live_skip", 3)
    yolo_ev  = st.session_state.get("live_yolo", 5)

    if "cam_cap" not in st.session_state:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            st.error("Cannot open webcam. Check camera permissions.")
            st.session_state["cam_running"] = False
            return
        st.session_state["cam_cap"] = cap

    cap        = st.session_state["cam_cap"]
    tracker    = st.session_state["blink_tracker"]
    frame_n    = st.session_state.get("cam_frame", 0)
    live_cache = st.session_state.get("live_cache", {})

    live_cache["frame_n"]    = frame_n
    live_cache["face_every"] = face_ev
    live_cache["yolo_every"] = yolo_ev

    t_frame_start = time.time()

    # Drain stale frames before reading the freshest one
    cap.grab()
    cap.grab()
    ret, frame = cap.read()

    if not ret:
        st.warning("Failed to read frame from webcam.")
        st.session_state["cam_frame"] = frame_n + 1
        time.sleep(0.03)
        st.rerun()
        return

    result, faces, decisions, spoofs, _, tailgating, stage_log = \
        _run_pipeline_live(frame, sec_mode, show_lm, tracker, live_cache)

    # ── ALERT flash — full-page red pulse when a blacklisted face is detected ──
    if any(d["action"] == "ALERT" for d in decisions):
        st.markdown(
            "<style>@keyframes alert-flash{"
            "0%,100%{background:rgba(200,0,0,0.28)}50%{background:rgba(200,0,0,0.04)}}"
            "</style>"
            "<div style='position:fixed;top:0;left:0;width:100vw;height:100vh;"
            "pointer-events:none;z-index:9999;"
            "animation:alert-flash 0.45s ease-in-out infinite'></div>",
            unsafe_allow_html=True,
        )

    # ── FPS calculation (rolling average over last 10 frames) ─────────────────
    elapsed = time.time() - t_frame_start
    fps_raw = 1.0 / elapsed if elapsed > 0 else 0.0
    fps_history = st.session_state.get("fps_history", [])
    fps_history.append(fps_raw)
    if len(fps_history) > 10:
        fps_history = fps_history[-10:]
    st.session_state["fps_history"] = fps_history
    fps_avg = sum(fps_history) / len(fps_history)

    st.session_state["live_cache"] = live_cache
    st.session_state["cam_frame"]  = frame_n + 1

    col_feed, col_panel = st.columns([3, 2])

    with col_feed:
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                 use_container_width=True,
                 caption=f"Frame #{frame_n}  |  {fps_avg:.1f} FPS  |  {elapsed*1000:.0f} ms/frame")

    with col_panel:
        st.markdown(":material/monitoring: **System Status**")
        st.divider()

        # FPS + performance row
        fps_color = "green" if fps_avg >= 10 else "orange" if fps_avg >= 5 else "red"
        st.markdown(
            f":material/speed: **Performance:** "
            f"<span style='color:{fps_color};font-weight:700'>{fps_avg:.1f} FPS</span> "
            f"· {elapsed*1000:.0f} ms/frame",
            unsafe_allow_html=True,
        )

        # Current stage
        active = next((n for n, s in reversed(stage_log) if s == "computed"), "Idle")
        st.markdown(f":material/bolt: **Stage:** `{active}`")

        # User identity
        if faces:
            match = faces[0].get("match")
            if match:
                uname = decisions[0].get("name", match.get("name", "?")) if decisions else "?"
                conf  = faces[0].get("confidence", 0.0)
                st.markdown(f":material/manage_accounts: **User:** `{uname}` ({conf:.0%})")
            else:
                st.markdown(":material/no_accounts: **User:** `Unknown`")
        else:
            st.markdown(":material/no_accounts: **User:** `No face detected`")

        # Stability
        is_stable = live_cache.get("is_stable", False)
        if faces and not is_stable:
            st.info(":material/timer: **Stability:** Hold still for scan…")
        elif faces:
            st.markdown(":material/check_circle: **Stability:** Face stable")
        else:
            st.markdown(":material/radio_button_unchecked: **Stability:** No face")

        # Liveness / spoof
        if spoofs:
            sp      = spoofs[0]
            sp_stat = sp.get("status", "LIVE" if sp.get("is_live", True) else "SPOOF SUSPECTED")
            if sp.get("is_live", True):
                st.markdown(f":material/visibility: **Liveness:** `{sp_stat}`")
            else:
                st.markdown(f":material/visibility_off: **Liveness:** `{sp_stat}`")
            if sp.get("ear") is not None:
                st.caption(
                    f"EAR: {sp['ear']:.3f}  ·  Blinks: {tracker.blink_count}  ·  "
                    f"{'✓ Liveness OK' if tracker.passed else 'Waiting for blink…'}"
                )
            if sp.get("suspicious_static"):
                st.caption(":material/warning: Motion check: suspiciously static — possible replay")
        else:
            st.markdown(":material/visibility: **Liveness:** `—`")

        st.divider()

        # Verdict card
        st.markdown(_render_verdict(faces, decisions), unsafe_allow_html=True)

        if tailgating:
            st.warning(":material/group: **Tailgating** — multiple persons detected in frame!")
        if not hq:
            st.caption(":material/info: Low quality camera mode — anti-spoofing accuracy reduced.")

        st.divider()

        # Pipeline stage progress
        st.markdown(":material/timeline: **Pipeline**")
        computed = sum(1 for _, s in stage_log if s == "computed")
        total    = len(stage_log)
        st.progress(computed / total if total else 0,
                    text=f"{computed}/{total} stages computed this frame")

        _STATUS_ICON = {
            "computed": ":material/check_circle:",
            "cached":   ":material/radio_button_unchecked:",
            "skipped":  ":material/block:",
        }
        for stage_name, stage_status in stage_log:
            icon = _STATUS_ICON.get(stage_status, ":material/radio_button_unchecked:")
            st.markdown(f"{icon} {stage_name} `{stage_status}`")

    time.sleep(0.03)
    st.rerun()   # fragment-scoped: sidebar and other pages are untouched


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL PIPELINE — state-machine MFA verification for Security Terminal
#
# State flow:
#   credentials → waiting_for_face → hold_still → blink_required →
#   scanning_animation → backend_checks → (tailgating | spoof | blacklist |
#                                          identity_mismatch | complete)
#
# Heavy operations (YOLO, full spoof, encoding, identity) only run after the
# user is stable AND has blinked, keeping every pre-scan frame lightweight.
# ═══════════════════════════════════════════════════════════════════════════════
def _run_terminal_pipeline(frame_bgr: np.ndarray,
                           claimed_username: str,
                           security_mode: str,
                           blink_tracker,
                           cache: dict) -> tuple:
    """
    Returns (annotated_frame, decision_or_None, spoof_or_None, tailgating, status_msg)

    status_msg values:
      'NO FACE' | 'HOLD STILL' | 'BLINK REQUIRED' | 'SCANNING' |
      'TAILGATING' | 'SPOOF' | 'BLACKLISTED' | 'COMPLETE'
    """
    frame_n = cache.get("frame_n", 0)
    result  = frame_bgr.copy()
    S       = 1.0 / _PROC_SCALE

    # ══ Phase A: Scanning animation ════════════════════════════════════════════
    # Runs for TERMINAL_SCAN_FRAMES frames after blink is confirmed.
    # No heavy processing — just draw the animation over the cached frame state.
    if cache.get("scan_started") and cache.get("scan_frame", 0) < TERMINAL_SCAN_FRAMES:
        sf = cache.get("scan_frame", 0)
        draw_scanning_bar(result, sf, TERMINAL_SCAN_FRAMES, cache.get("last_face_box"))
        draw_person_tracks(result, cache.get("persons", []), cache.get("tailgating", False))
        stamp_status(result, "SCANNING...", (0, 255, 120))
        cache.update({"scan_frame": sf + 1, "frame_n": frame_n + 1,
                      "pipeline_stage": "scanning_animation"})
        return result, None, None, cache.get("tailgating", False), "SCANNING"

    # ══ Phase B: Backend checks (runs once after animation completes) ══════════
    if cache.get("scan_started") and cache.get("scan_frame", 0) >= TERMINAL_SCAN_FRAMES:
        face_box = cache.get("last_face_box")
        cache["pipeline_stage"] = "backend_checks"

        # B1 — YOLO tailgating (single run)
        if not cache.get("yolo_done"):
            persons, tailgating = detect_and_track(result)
            cache.update({"persons": persons, "tailgating": tailgating, "yolo_done": True})
        persons    = cache.get("persons",    [])
        tailgating = cache.get("tailgating", False)
        draw_person_tracks(result, persons, tailgating)

        if tailgating:
            tg = {"action": "DENY", "role": "unknown", "color": ROLE_COLORS["unknown"],
                  "label": "Tailgating — multiple persons in frame",
                  "reason": "Multiple persons detected in access zone",
                  "name": "", "username": ""}
            if face_box:
                draw_rbac_result(result, face_box, tg)
            stamp_status(result, "TAILGATING ALERT", (0, 0, 220))
            cache.update({"frame_n": frame_n + 1, "pipeline_stage": "tailgating_alert"})
            return result, None, None, True, "TAILGATING"

        # B2 — Anti-spoofing (static + temporal + replay)
        lm_s = cache.get("last_lm_s", {})
        gray = None
        if face_box:
            roi = frame_bgr[face_box["top"]:face_box["bottom"],
                            face_box["left"]:face_box["right"]]
            if roi.size > 0:
                gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))
        spoof = static_spoof_check(gray, lm_s) if gray is not None else dict(_SPOOF_FALLBACK)
        if gray is not None:
            spoof = aggregate_spoof_result(
                spoof, True,
                temporal_spoof_check(cache.get("prev_gray_face"), gray, cache, security_mode),
                replay_artifact_check(gray),
                security_mode,
            )
        if not spoof.get("is_live", True):
            sd = {"action": "DENY", "role": "spoof", "color": ROLE_COLORS["spoof"],
                  "label": f"Anti-spoofing failed — {spoof.get('status', 'SPOOF SUSPECTED')}",
                  "reason": "Possible replay/print attack", "name": "", "username": ""}
            if face_box:
                draw_rbac_result(result, face_box, sd)
            stamp_status(result, "SPOOF SUSPECTED", (0, 80, 255))
            cache.update({"frame_n": frame_n + 1, "pipeline_stage": "spoof_rejected"})
            return result, sd, spoof, tailgating, "SPOOF"

        # B3 — Blacklist check + identity verification
        enc_s = cache.get("last_enc_s")
        if enc_s is not None:
            is_bl, bl_entry, _ = check_blacklist(enc_s)
            if is_bl:
                dec = make_blacklist_decision(bl_entry)
                if face_box:
                    draw_rbac_result(result, face_box, dec)
                stamp_status(result, "SECURITY ALERT", (0, 0, 220))
                cache.update({"frame_n": frame_n + 1, "pipeline_stage": "blacklist_alert"})
                return result, dec, spoof, tailgating, "BLACKLISTED"

            match, conf = verify_claimed_user(enc_s, claimed_username, security_mode)
            cache.update({"last_match": match, "last_confidence": conf if match else 0.0})
            if match:
                dec = make_decision(match, conf, True, security_mode)
            else:
                dec = {"action": "DENY", "role": "unknown", "color": ROLE_COLORS["unknown"],
                       "label": f"Identity not verified ({conf:.0%})",
                       "reason": "Face does not match claimed user", "name": "", "username": ""}
        else:
            dec = {"action": "DENY", "role": "unknown", "color": ROLE_COLORS["unknown"],
                   "label": "No face encoding available",
                   "reason": "Feature extraction failed", "name": "", "username": ""}

        if face_box:
            draw_rbac_result(result, face_box, dec)
        granted = dec["action"] == "ALLOW"
        stamp_status(result,
                     "ACCESS GRANTED" if granted else "ACCESS DENIED",
                     (0, 200, 0) if granted else (0, 0, 220))
        cache.update({"frame_n": frame_n + 1, "pipeline_stage": "complete"})
        return result, dec, spoof, tailgating, "COMPLETE"

    # ══ Phase C: Pre-scan — lightweight per-frame processing ══════════════════
    # Goal: minimise lag before the scan starts.
    #   • HOG face detection: every TERMINAL_FACE_DETECT_INTERVAL frames
    #   • Landmarks + EAR:    every frame using cached face locations (cheap)
    #   • Face encoding:      once, when face first becomes stable
    #   • YOLO:               deferred to Phase B (post-blink)
    cache["pipeline_stage"] = "pre_scan"
    small      = cv2.resize(frame_bgr, (0, 0), fx=_PROC_SCALE, fy=_PROC_SCALE)
    enhanced_s = apply_clahe(small)
    rgb_s      = cv2.cvtColor(enhanced_s, cv2.COLOR_BGR2RGB)

    run_face = (frame_n % TERMINAL_FACE_DETECT_INTERVAL == 0) or ("last_face_locs_s" not in cache)
    if run_face:
        face_locs_s = face_recognition.face_locations(rgb_s, model="hog")
        cache["last_face_locs_s"] = face_locs_s
    else:
        face_locs_s = cache.get("last_face_locs_s", [])

    if not face_locs_s:
        cache.pop("last_face_box", None)
        cache.update({"stable_count": 0, "is_stable": False,
                      "pipeline_stage": "waiting_for_face", "frame_n": frame_n + 1})
        stamp_status(result, "NO FACE DETECTED", (160, 160, 160))
        return result, None, None, False, "NO FACE"

    # Update stability on face-detect frames
    if run_face:
        loc_s    = face_locs_s[0]
        face_box = {"top":    int(loc_s[0] * S), "right": int(loc_s[1] * S),
                    "bottom": int(loc_s[2] * S), "left":  int(loc_s[3] * S)}
        ok = compute_face_stability(cache.get("last_face_box"), face_box)
        cache["last_face_box"] = face_box
        cache["stable_count"] = (cache.get("stable_count", 0) + 1) if ok \
                                else max(0, cache.get("stable_count", 0) - 1)
        cache["is_stable"] = cache["stable_count"] >= STABILITY_REQUIRED_FRAMES
    else:
        face_box = cache.get("last_face_box")
        if face_box is None:
            cache.update({"frame_n": frame_n + 1})
            stamp_status(result, "NO FACE DETECTED", (160, 160, 160))
            return result, None, None, False, "NO FACE"

    is_stable = cache.get("is_stable", False)

    # Landmarks + EAR every frame (cheap — landmarks on cached small-frame locations)
    if is_stable:
        all_lm_s = face_recognition.face_landmarks(rgb_s, face_locs_s)
        if all_lm_s:
            cache["last_lm_s"] = all_lm_s[0]
            blink_tracker.update(all_lm_s[0])
        # Cache encoding once for backend phase (only on face-detect frames)
        if "last_enc_s" not in cache and run_face:
            enc_list = face_recognition.face_encodings(rgb_s, face_locs_s)
            if enc_list:
                cache["last_enc_s"] = enc_list[0]

    # Draw appropriate status overlay
    if not is_stable:
        sc  = cache.get("stable_count", 0)
        pen = {"action": "DENY", "role": "unknown", "color": ROLE_COLORS["unknown"],
               "label": f"Hold still… ({sc}/{STABILITY_REQUIRED_FRAMES})",
               "reason": "Stabilising face detection", "name": "", "username": ""}
        draw_rbac_result(result, face_box, pen)
        stamp_status(result, "HOLD STILL", (0, 200, 255))
        cache.update({"frame_n": frame_n + 1, "pipeline_stage": "hold_still"})
        return result, None, None, False, "HOLD STILL"

    if not blink_tracker.passed:
        pen = {"action": "DENY", "role": "unknown", "color": ROLE_COLORS["unknown"],
               "label": "Blink once to confirm liveness",
               "reason": "Waiting for blink", "name": "", "username": ""}
        draw_rbac_result(result, face_box, pen)
        stamp_ear(result, blink_tracker.last_ear, blink_tracker.blink_count, blink_tracker.passed)
        stamp_status(result, "BLINK ONCE", (0, 200, 255))
        cache.update({"frame_n": frame_n + 1, "pipeline_stage": "blink_required"})
        return result, None, None, False, "BLINK REQUIRED"

    # Blink confirmed — ensure encoding is captured before animation starts
    if "last_enc_s" not in cache:
        enc_list = face_recognition.face_encodings(rgb_s, face_locs_s)
        if enc_list:
            cache["last_enc_s"] = enc_list[0]

    cache.update({"scan_started": True, "scan_frame": 0,
                  "pipeline_stage": "scanning_animation"})
    draw_scanning_bar(result, 0, TERMINAL_SCAN_FRAMES, face_box)
    cache["frame_n"] = frame_n + 1
    return result, None, None, False, "SCANNING"


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown("# :material/home: Smart Access Control")
    st.markdown("""
A complete computer-vision RBAC pipeline demonstrating both classical and
modern CV techniques integrated into a single, deployable web application.

---
### Vision Pipeline

| Stage | Technique | Purpose |
|---|---|---|
| Image Enhancement | CLAHE (LAB luminance) | Normalise lighting before detection |
| Background Separation | MOG2 Gaussian Mixture | Isolate moving foreground |
| Face Detection | HOG + SVM (dlib) | Locate faces in frame |
| Landmark Detection | 68-point dlib shape predictor | Eye/nose/mouth positions |
| Face Alignment | Affine rotation via eye centres | Canonical pose for descriptors |
| Classical Features | Canny · LBP · HOG | Structural & texture descriptors |
| Face Recognition | 128-d ResNet embedding (dlib) | Identity matching |
| Anti-Spoofing | EAR blink · LBP variance · Laplacian | Liveness verification |
| Person Tracking | YOLOv8n detection | Multi-person, tailgating detection |
| RBAC Decision | Role-based engine | Allow / Deny / Alert |

---
### Roles
| Role | Decision | Box colour |
|---|---|---|
| **Admin** | ALLOW | Gold |
| **Authorized** | ALLOW | Green |
| **Blacklisted** | ALERT + DENY | Red |
| **Unknown** | DENY | Grey |

---
### Quick Start
1. **Register User** — enrol people with face photos and assign roles
2. **Identify: Image** — upload a photo and run the full pipeline
3. **Live Camera** — real-time webcam feed with EAR blink anti-spoofing
4. **Admin Panel** — manage users and roles
5. **Access Log** — timestamped audit trail
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY TERMINAL
# Two-factor flow: credentials (something you know) → face scan (something you are)
# Blacklist is checked FIRST on every scanned frame.
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Security Terminal":
    st.markdown("# :material/fingerprint: Security Terminal")
    st.caption("Two-factor authentication — credentials verified first, then face scan.")

    if "terminal_state" not in st.session_state:
        st.session_state["terminal_state"] = "credentials"

    t_state = st.session_state["terminal_state"]

    # ── Step 1: Credentials ───────────────────────────────────────────────────
    if t_state == "credentials":
        st.markdown("#### Step 1 of 2 — Enter credentials")
        col_cred, _ = st.columns([1, 1])
        with col_cred:
            with st.form("terminal_login"):
                t_user = st.text_input("Username", placeholder="your username")
                t_pass = st.text_input("Password", type="password")
                submitted = st.form_submit_button(
                    "Authenticate", type="primary",
                    icon=":material/lock_open:",
                    use_container_width=True,
                )

            if submitted:
                user_rec = verify_credentials(t_user, t_pass)
                if user_rec and user_rec["role"] in ("admin", "authorized"):
                    st.session_state.update({
                        "terminal_state":        "face_scan",
                        "terminal_username":     t_user,
                        "terminal_user_data":    user_rec,
                        "terminal_sec_mode":     "normal",
                        "terminal_blink":        BlinkTracker(),
                        "terminal_frame_n":      0,
                        "terminal_cache":        {},
                    })
                    reset_background_model()
                    st.rerun()
                else:
                    st.error("Invalid credentials or account not authorized.")

        st.info(
            "Blacklisted individuals attempting to use stolen credentials "
            "will be flagged at the face-scan step."
        )

    # ── Step 2: Face scan ─────────────────────────────────────────────────────
    elif t_state == "face_scan":
        user_data        = st.session_state.get("terminal_user_data", {})
        claimed_username = st.session_state.get("terminal_username", "")
        sec_mode         = st.session_state.get("terminal_sec_mode", "normal")
        blink_tracker    = st.session_state["terminal_blink"]
        t_cache          = st.session_state.get("terminal_cache", {})

        st.markdown("#### Step 2 of 2 — Face verification")
        st.success(
            f"Credentials accepted for **{user_data.get('name', claimed_username)}**. "
            "Look directly at the camera and blink once to confirm liveness."
        )

        col_ctrl, _ = st.columns([2, 3])
        with col_ctrl:
            hq = st.toggle(
                "High quality camera", value=True, key="terminal_hq",
                help="ON = normal anti-spoofing. OFF = relaxed thresholds (less accurate).",
            )
            sec_mode = "normal" if hq else "relaxed"
            st.session_state["terminal_sec_mode"] = sec_mode
            if not hq:
                st.warning(
                    "Low camera quality mode — anti-spoofing accuracy may be reduced. "
                    "False failures possible with poor lighting."
                )
            if st.button("Cancel", icon=":material/cancel:", use_container_width=True):
                cap_ref = st.session_state.pop("terminal_cap", None)
                if cap_ref:
                    cap_ref.release()
                st.session_state["terminal_state"] = "credentials"
                st.rerun()

        st.divider()

        # Open webcam once and keep it in session_state
        if "terminal_cap" not in st.session_state:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
                st.session_state["terminal_state"] = "credentials"
                st.stop()
            st.session_state["terminal_cap"] = cap

        cap = st.session_state["terminal_cap"]
        cap.grab(); cap.grab()
        ret, frame = cap.read()

        if ret:
            annotated, decision, spoof, term_tailgating, term_status = _run_terminal_pipeline(
                frame, claimed_username, sec_mode, blink_tracker, t_cache
            )
            st.session_state["terminal_cache"] = t_cache

            # ── Layout: camera feed + pipeline status panel ───────────────────
            col_feed, col_panel = st.columns([3, 2])
            with col_feed:
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                         use_container_width=True)

            with col_panel:
                st.markdown("#### :material/monitoring: Pipeline Status")
                pipeline_stage = t_cache.get("pipeline_stage", "pre_scan")

                # Map current stage → human label + hint
                _STAGE_MAP = {
                    "waiting_for_face":   ("Waiting for face",       "Position your face in frame"),
                    "hold_still":         ("Hold still check",        "Please stop moving"),
                    "blink_required":     ("Blink liveness check",    "Blink once to confirm liveness"),
                    "scanning_animation": ("Scanning animation",      "Hold still — scan in progress"),
                    "backend_checks":     ("Backend checks",          "Running security verification…"),
                    "tailgating_alert":   ("Tailgating detection",    "Multiple persons detected — denied"),
                    "spoof_rejected":     ("Anti-spoofing check",     "Spoof suspected — denied"),
                    "blacklist_alert":    ("Blacklist check",         "Blacklisted user detected — alert"),
                    "complete":           ("Final decision",          "Authentication complete"),
                }
                _stage_label, _stage_hint = _STAGE_MAP.get(pipeline_stage, ("Processing…", ""))
                st.info(f":material/bolt: **Current process:** {_stage_label}")
                if _stage_hint:
                    st.caption(_stage_hint)

                # Stage pipeline with status icons
                _SI = {
                    "passed":  (":material/check_circle:",           "green"),
                    "running": (":material/sync:",                   "#1e90ff"),
                    "waiting": (":material/radio_button_unchecked:", "gray"),
                    "failed":  (":material/cancel:",                 "#cc2222"),
                    "skipped": (":material/block:",                  "gray"),
                }

                def _row(label: str, st_key: str) -> None:
                    icon, color = _SI[st_key]
                    st.markdown(
                        f"{icon} <span style='color:{color}'>{label}</span> "
                        f"`{st_key}`",
                        unsafe_allow_html=True,
                    )

                def _stage_statuses(ps: str, dec: "dict | None") -> list:
                    p = "passed"; w = "waiting"; r = "running"; f = "failed"; s = "skipped"
                    base = [p, w, w, w, w, w, w, w, w, w]
                    m = {
                        "waiting_for_face":   [p, r, w, w, w, w, w, w, w, w],
                        "hold_still":         [p, p, r, w, w, w, w, w, w, w],
                        "blink_required":     [p, p, p, r, w, w, w, w, w, w],
                        "scanning_animation": [p, p, p, p, r, w, w, w, w, w],
                        "backend_checks":     [p, p, p, p, p, r, w, w, w, w],
                        "tailgating_alert":   [p, p, p, p, p, f, s, s, s, f],
                        "spoof_rejected":     [p, p, p, p, p, p, f, s, s, f],
                        "blacklist_alert":    [p, p, p, p, p, p, p, f, s, f],
                        "complete": (
                            [p, p, p, p, p, p, p, p, p, p]
                            if dec and dec.get("action") == "ALLOW"
                            else [p, p, p, p, p, p, p, p, f, f]
                        ),
                    }
                    return m.get(ps, base)

                statuses = _stage_statuses(pipeline_stage, decision)
                stage_names = [
                    "Credentials verified",
                    "Waiting for face",
                    "Face stability check",
                    "Blink liveness check",
                    "Scanning animation",
                    "Tailgating detection",
                    "Anti-spoofing check",
                    "Blacklist check",
                    "Identity verification",
                    "Final decision",
                ]
                st.divider()
                for sname, skey in zip(stage_names, statuses):
                    _row(sname, skey)

                # Spoof detail when relevant
                if spoof and not spoof.get("is_live", True):
                    st.divider()
                    st.error(f":material/block: {spoof.get('status', 'SPOOF SUSPECTED')}")
                    if spoof.get("suspicious_static"):
                        st.caption("Motion check failed — possible replay attack")

                if term_tailgating:
                    st.error(":material/group: **Tailgating** — multiple persons detected!")

            # ── Definitive outcomes: log and transition to result ─────────────
            if term_status in ("COMPLETE", "BLACKLISTED", "SPOOF") and decision:
                log_access_event(
                    detected_name=decision.get("name", user_data.get("name", "")),
                    username=claimed_username,
                    role=decision.get("role", "unknown"),
                    confidence=t_cache.get("last_confidence", 0.0),
                    action=decision["action"],
                    reason=decision.get("reason", ""),
                    tailgating=term_tailgating,
                    source="terminal",
                )
                cap.release()
                st.session_state.pop("terminal_cap", None)
                st.session_state["terminal_state"]  = "result"
                st.session_state["terminal_result"] = decision
                st.rerun()

            if term_status == "TAILGATING":
                tg_decision = {
                    "action": "DENY",
                    "label":  "Tailgating detected",
                    "reason": "Multiple persons detected in access zone",
                    "role":   "unknown", "color": ROLE_COLORS["unknown"],
                    "name":   "", "username": "",
                }
                log_access_event(
                    detected_name=user_data.get("name", claimed_username),
                    username=claimed_username,
                    role="unknown",
                    confidence=0.0,
                    action="DENY",
                    reason="Tailgating detected",
                    tailgating=True,
                    source="terminal",
                )
                cap.release()
                st.session_state.pop("terminal_cap", None)
                st.session_state["terminal_state"]  = "result"
                st.session_state["terminal_result"] = tg_decision
                st.rerun()

        st.session_state["terminal_frame_n"] = st.session_state.get("terminal_frame_n", 0) + 1
        time.sleep(0.05)
        st.rerun()

    # ── Step 3: Result ────────────────────────────────────────────────────────
    elif t_state == "result":
        decision  = st.session_state.get("terminal_result", {})
        action    = decision.get("action", "DENY")
        user_data = st.session_state.get("terminal_user_data", {})

        st.markdown("#### Authentication Result")

        if action == "ALLOW":
            st.markdown(
                "<div style='background:#0d2b0d;padding:30px;border-radius:12px;"
                "text-align:center;border:2px solid #00bb00'>"
                "<p style='font-size:3rem;color:#00dd00;margin:0'>✓ ACCESS GRANTED</p>"
                f"<h3 style='color:#00dd00;margin:10px 0'>{decision.get('label','')}</h3>"
                f"<p style='color:#99ee99;margin:0'>{decision.get('reason','')}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif action == "ALERT":
            st.markdown(
                "<div style='background:#2b0000;padding:30px;border-radius:12px;"
                "text-align:center;border:3px solid #cc0000'>"
                "<p style='font-size:3rem;color:#ff2222;margin:0'>⚠ SECURITY ALERT</p>"
                f"<h3 style='color:#ff2222;margin:10px 0'>{decision.get('label','')}</h3>"
                f"<p style='color:#ffaaaa;margin:0'>{decision.get('reason','')}</p>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#2b0d0d;padding:30px;border-radius:12px;"
                "text-align:center;border:2px solid #cc0000'>"
                "<p style='font-size:3rem;color:#ee2222;margin:0'>✗ ACCESS DENIED</p>"
                f"<h3 style='color:#ee2222;margin:10px 0'>{decision.get('label','')}</h3>"
                f"<p style='color:#ffaaaa;margin:0'>{decision.get('reason','')}</p>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()
        if st.button("New Session", icon=":material/refresh:", type="primary"):
            for k in ("terminal_state", "terminal_username", "terminal_user_data",
                      "terminal_blink", "terminal_cache", "terminal_result",
                      "terminal_frame_n", "terminal_sec_mode"):
                st.session_state.pop(k, None)
            cap_ref = st.session_state.pop("terminal_cap", None)
            if cap_ref:
                cap_ref.release()
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTER USER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Register User":
    st.markdown("# :material/person_add: Register User")

    if "reg_n" not in st.session_state:
        st.session_state.reg_n = 0
    n = st.session_state.reg_n

    tab_manual, tab_blacklist, tab_bulk = st.tabs(
        ["Enroll Authorized User", "Blacklist Entry", "Bulk Import — Pins Dataset"]
    )

    # ── Tab 1: Enroll authorized / admin user ─────────────────────────────────
    with tab_manual:
        st.subheader("Create a new authorized user account")
        col_form, col_info = st.columns([3, 2], gap="large")

        with col_form:
            name      = st.text_input("Full name",  placeholder="e.g. Alice Smith",  key=f"reg_name_{n}")
            username  = st.text_input("Username",   placeholder="e.g. alice",          key=f"reg_uname_{n}")
            password  = st.text_input("Password",   type="password",
                                      placeholder="Min 8 chars, upper, lower, special", key=f"reg_pw_{n}")
            role      = st.selectbox("Role", ["authorized", "admin"],                  key=f"reg_role_{n}")
            photos    = st.file_uploader(
                "Face photos (3–5 recommended)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"reg_photos_{n}",
            )

            # ── Photo preview with face detection ─────────────────────────────
            if photos:
                st.markdown("**Preview — face detection per photo:**")
                prev_cols = st.columns(min(len(photos), 5))
                any_missing = False
                for _pi, _uf in enumerate(photos[:5]):
                    _raw  = np.frombuffer(_uf.read(), np.uint8)
                    _uf.seek(0)
                    _bgr  = cv2.imdecode(_raw, cv2.IMREAD_COLOR)
                    if _bgr is None:
                        continue
                    _rgb  = cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB)
                    _locs = face_recognition.face_locations(_rgb, model="hog")
                    _prev = _bgr.copy()
                    for (_t, _r, _b, _l) in _locs:
                        cv2.rectangle(_prev, (_l, _t), (_r, _b), (0, 200, 0), 2)
                    if not _locs:
                        any_missing = True
                    prev_cols[_pi].image(
                        cv2.cvtColor(_prev, cv2.COLOR_BGR2RGB),
                        caption=f"{'Face OK' if _locs else 'No face!'} ({len(_locs)})",
                        use_container_width=True,
                    )
                if any_missing:
                    st.warning("Some photos have no detected face — they will be skipped.")

            # ── Password strength check ───────────────────────────────────────
            pw_errors = []
            if password:
                if len(password) < 8:
                    pw_errors.append("at least 8 characters")
                if not any(c.islower() for c in password):
                    pw_errors.append("a lowercase letter")
                if not any(c.isupper() for c in password):
                    pw_errors.append("an uppercase letter")
                if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in password):
                    pw_errors.append("a special character (!@#$%…)")

                if pw_errors:
                    st.error("Password must contain: " + ", ".join(pw_errors))
                else:
                    st.success("Password strength: OK")

            pw_valid = password and not pw_errors
            ready = bool(name and username and pw_valid and photos)
            if st.button("Register", type="primary", icon=":material/how_to_reg:",
                         disabled=not ready):

                with st.status("Running enrolment pipeline…", expanded=True) as status:
                    prog = st.progress(0.0)

                    st.write("Saving face images…")
                    person_dir = os.path.join(KNOWN_FACES_DIR, username.strip())
                    os.makedirs(person_dir, exist_ok=True)
                    saved_paths = []
                    for uf in photos:
                        dst = os.path.join(person_dir, uf.name)
                        with open(dst, "wb") as f:
                            f.write(uf.read())
                        saved_paths.append(dst)
                    prog.progress(0.20)
                    st.write(f"Saved {len(saved_paths)} image(s)")

                    st.write("Detecting faces…")
                    enc = extract_average_encoding(saved_paths)
                    prog.progress(0.50)
                    if enc is None:
                        st.error("No faces detected. Try clearer images.")
                        status.update(label="Enrolment failed", state="error")
                        st.stop()
                    st.write("Faces detected and encoded")

                    st.write("Extracting classical features (Canny, LBP, HOG)…")
                    sample_bgr = cv2.imdecode(
                        np.frombuffer(open(saved_paths[0], "rb").read(), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    sample_rgb  = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
                    locs        = face_recognition.face_locations(sample_rgb, model="hog")
                    sample_feat = extract_all(sample_bgr, locs[0]) if locs else None
                    prog.progress(0.75)
                    st.write("Classical descriptors extracted")

                    st.write("Storing in database…")
                    ok, msg = create_user(name.strip(), username.strip(), password, role, enc)
                    prog.progress(1.0)

                    _knn_ok  = False
                    _knn_msg = ""
                    if ok:
                        st.write("User created successfully.")
                        st.write("Updating KNN classifier with new face data…")
                        _knn_ok, _knn_msg = train_knn()
                        if _knn_ok:
                            st.write(f"KNN updated: {_knn_msg}")
                        else:
                            st.write(f"KNN retraining warning: {_knn_msg}")
                        prog.progress(1.0)
                        status.update(label="Enrolment complete!", state="complete")
                    else:
                        st.error(msg)
                        status.update(label="Enrolment failed", state="error")
                        st.stop()

                st.success(msg)
                if not _knn_ok and ok:
                    st.warning(
                        f"KNN retraining failed ({_knn_msg}). "
                        "Use **Admin Panel → Retrain KNN Model** to update manually."
                    )

                if sample_feat:
                    st.subheader("Classical feature outputs")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.image(sample_feat["gray_chip"],  caption="Aligned face (64x64)", clamp=True)
                    c2.image(sample_feat["canny_map"],  caption="Canny edge map",       clamp=True)
                    lbp_vis = cv2.normalize(sample_feat["lbp_image"], None, 0, 255,
                                            cv2.NORM_MINMAX).astype(np.uint8)
                    c3.image(lbp_vis, caption="LBP texture pattern", clamp=True)
                    c4.image(sample_feat["hog_vis"], caption="HOG gradient image", clamp=True)

        with col_info:
            if st.button("Clear everything", icon=":material/refresh:", type="secondary"):
                st.session_state.reg_n += 1
                st.rerun()
            st.markdown("""
**Role descriptions**

| Role | Access |
|---|---|
| `authorized` | Standard access — green box |
| `admin` | Full access — gold box |

**Tips for good enrolment**
- Use 3–5 photos with varied lighting
- Face clearly visible and front-facing
- Avoid sunglasses or heavy obstructions

> Threat individuals are enrolled separately
> via the **Blacklist Entry** tab — they do not
> get a username or system credentials.
            """)

    # ── Tab 2: Blacklist entry (no credentials required) ─────────────────────
    with tab_blacklist:
        st.subheader("Add a threat individual to the blacklist")
        st.caption(
            "Blacklisted identities are stored separately from authorized users. "
            "They have no system credentials and cannot authenticate. "
            "Their face encodings are checked first in every pipeline."
        )

        col_bl, col_bl_info = st.columns([3, 2], gap="large")
        with col_bl:
            bl_name   = st.text_input("Name / Label",
                                      placeholder="e.g. John Doe or Unknown Male #1",
                                      key=f"bl_name_{n}")
            bl_reason = st.selectbox("Threat reason",
                                     ["Trespassing", "Banned employee",
                                      "Flagged intruder", "Court order",
                                      "Other security threat"],
                                     key=f"bl_reason_{n}")
            bl_notes  = st.text_area("Additional notes (optional)",
                                     placeholder="e.g. Attempted break-in on 2026-01-15",
                                     key=f"bl_notes_{n}")
            bl_photos = st.file_uploader(
                "Face photos (3–5 recommended)",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                key=f"bl_photos_{n}",
            )

            bl_ready = bool(bl_name and bl_reason and bl_photos)
            if st.button("Add to Blacklist", type="primary",
                         icon=":material/block:", disabled=not bl_ready):
                with st.status("Processing blacklist entry…", expanded=True) as bl_status:
                    bl_prog = st.progress(0.0)

                    saved_bl_paths = []
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        for uf in bl_photos:
                            dst = os.path.join(tmp_dir, uf.name)
                            with open(dst, "wb") as f:
                                f.write(uf.read())
                            saved_bl_paths.append(dst)
                        bl_prog.progress(0.30)
                        st.write(f"Saved {len(saved_bl_paths)} image(s) temporarily")

                        st.write("Detecting and encoding face…")
                        bl_enc = extract_average_encoding(saved_bl_paths)
                        bl_prog.progress(0.70)

                    if bl_enc is None:
                        st.error("No faces detected. Use clearer, front-facing photos.")
                        bl_status.update(label="Failed — no face detected", state="error")
                        st.stop()

                    st.write("Storing in blacklist database…")
                    bl_ok, bl_msg = create_blacklist_entry(
                        bl_name, bl_reason, bl_notes, bl_enc
                    )
                    bl_prog.progress(1.0)

                    if bl_ok:
                        bl_status.update(label="Blacklist entry added!", state="complete")
                        st.success(bl_msg)
                    else:
                        bl_status.update(label="Failed", state="error")
                        st.error(bl_msg)

        with col_bl_info:
            st.markdown("""
**Who belongs here?**
- Trespassers or intruders caught on camera
- Former employees banned from premises
- Individuals with court-issued restrictions
- Any flagged security threat

**How it works**
1. Face encoding stored in the `blacklist` table
2. Every pipeline checks blacklist **first**
3. On match: immediate **ALERT** — no further processing
4. Blacklisted persons cannot authenticate via the Security Terminal
            """)

    with tab_bulk:
        st.subheader("Bulk Import — Pins Face Recognition (Kaggle)")
        st.markdown("Dataset: **Pins Face Recognition** by herbi4rtz. "
                    "Paste the path to the extracted `105_classes_pins_dataset` folder.")

        dataset_path = st.text_input(
            "Dataset root folder",
            placeholder=r"C:\Users\Dell\smart_access_control\archive\105_classes_pins_dataset",
            key=f"bulk_path_{n}",
        )

        if dataset_path and os.path.isdir(dataset_path):
            pins_dirs = sorted(
                d for d in os.listdir(dataset_path)
                if d.startswith("pins_") and os.path.isdir(os.path.join(dataset_path, d))
            )
            if not pins_dirs:
                st.warning("No `pins_*` folders found.")
            else:
                all_names = [d[5:] for d in pins_dirs]
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    selected = st.multiselect(
                        f"Select people ({len(all_names)} available)",
                        all_names, default=all_names[:6],
                    )
                with col_b:
                    max_imgs    = st.slider("Images / person", 3, 15, 5)
                    bulk_role   = st.selectbox("Assign role",
                                               ["authorized", "admin"],
                                               key="bulk_role")
                    auto_enroll = st.checkbox("Auto-extract encodings", value=True)

                if st.button("Import", type="primary",
                             icon=":material/upload:", disabled=not selected):
                    prog = st.progress(0.0)
                    imported, skipped = 0, 0

                    for i, name in enumerate(selected):
                        src = os.path.join(dataset_path, f"pins_{name}")
                        dst = os.path.join(KNOWN_FACES_DIR, name)
                        os.makedirs(dst, exist_ok=True)

                        imgs = sorted(
                            f for f in os.listdir(src)
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))
                        )[:max_imgs]
                        for img in imgs:
                            shutil.copy2(os.path.join(src, img), os.path.join(dst, img))

                        uname = name.lower().replace(" ", "_")
                        pw    = secrets.token_hex(8)

                        enc = extract_average_encoding(
                            [os.path.join(dst, f) for f in imgs]
                        ) if auto_enroll else None

                        ok, _ = create_user(name, uname, pw, bulk_role, enc)
                        if ok:
                            imported += 1
                        else:
                            if enc is not None:
                                update_face_encoding(uname, enc)
                            skipped += 1

                        prog.progress((i + 1) / len(selected))

                    st.success(
                        f"Imported {imported} new user(s), updated {skipped} existing."
                    )
                    knn_ok, knn_msg = train_knn()
                    if knn_ok:
                        st.info(f"KNN retrained: {knn_msg}")
                    st.rerun()

        elif dataset_path:
            st.error("Path does not exist.")


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTIFY: IMAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Identify: Image":
    st.markdown("# :material/image_search: Identify: Image")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    col_opts, _ = st.columns([2, 3])
    with col_opts:
        hq             = st.toggle("High quality camera", value=True,
                                   help="ON = normal thresholds. OFF = relaxed anti-spoofing.")
        sec_mode       = "normal" if hq else "relaxed"
        show_landmarks = st.checkbox("Show facial landmarks", value=True)
        if not hq:
            st.caption(":material/warning: Low quality mode — anti-spoofing thresholds relaxed.")

    # ── Test samples panel ────────────────────────────────────────────────────
    with st.expander(":material/folder: Test samples", expanded=False):
        _img_files = sorted(
            f for f in os.listdir(TEST_IMAGE_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ) if os.path.isdir(TEST_IMAGE_DIR) else []
        _spoof_files = sorted(
            f for f in os.listdir(TEST_SPOOF_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ) if os.path.isdir(TEST_SPOOF_DIR) else []
        _tail_files = sorted(
            f for f in os.listdir(TEST_TAILGATE_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ) if os.path.isdir(TEST_TAILGATE_DIR) else []

        if not (_img_files or _spoof_files or _tail_files):
            st.info(
                f"No test samples found. Add images to:\n"
                f"- `{TEST_IMAGE_DIR}/` — general detection samples\n"
                f"- `{TEST_SPOOF_DIR}/` — spoof/replay samples\n"
                f"- `{TEST_TAILGATE_DIR}/` — tailgating samples"
            )
        else:
            if _img_files:
                st.markdown("**Image detection samples**")
                _cols = st.columns(min(len(_img_files), 4))
                for _i, _fname in enumerate(_img_files[:8]):
                    _img = cv2.imread(os.path.join(TEST_IMAGE_DIR, _fname))
                    if _img is not None:
                        _cols[_i % 4].image(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB),
                                            caption=_fname[:30], use_container_width=True)
            if _spoof_files:
                st.markdown("**Spoof / replay samples**")
                _cols = st.columns(min(len(_spoof_files), 4))
                for _i, _fname in enumerate(_spoof_files[:8]):
                    _img = cv2.imread(os.path.join(TEST_SPOOF_DIR, _fname))
                    if _img is not None:
                        _cols[_i % 4].image(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB),
                                            caption=_fname[:30], use_container_width=True)
            if _tail_files:
                st.markdown("**Tailgating samples**")
                _cols = st.columns(min(len(_tail_files), 4))
                for _i, _fname in enumerate(_tail_files[:8]):
                    _img = cv2.imread(os.path.join(TEST_TAILGATE_DIR, _fname))
                    if _img is not None:
                        _cols[_i % 4].image(cv2.cvtColor(_img, cv2.COLOR_BGR2RGB),
                                            caption=_fname[:30], use_container_width=True)

    if uploaded and st.button("Run Full Pipeline", type="primary",
                              icon=":material/play_arrow:"):
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col_orig, col_result = st.columns(2)
        col_orig.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                       caption="Input", use_container_width=True)

        with st.status("Running identification pipeline…", expanded=True) as status:
            _STAGE_COUNT = 8
            prog = st.progress(0.0)

            class _Writer:
                def __init__(self): self._i = 0
                def write(self, msg):
                    st.write(msg)
                    self._i += 1
                    prog.progress(min(self._i / _STAGE_COUNT, 1.0))

            writer = _Writer()
            result, faces, decisions, spoofs, feat_list, persons, tailgating, fg_mask = \
                _run_pipeline(frame_bgr, sec_mode, show_landmarks, "image", writer)
            status.update(label="Pipeline complete!", state="complete")

        col_result.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                         caption="Analysis result", use_container_width=True)

        st.divider()
        if not faces:
            st.warning("No faces detected in the image.")
        else:
            st.subheader("Access decisions")
            for face, decision, spoof in zip(faces, decisions, spoofs):
                action = decision["action"]
                if action == "ALLOW":
                    st.success(f"ALLOW — {decision['label']} — {decision['reason']}")
                elif action == "ALERT":
                    st.error(f"SECURITY ALERT — {decision['label']} — {decision['reason']}")
                else:
                    st.error(f"DENY — {decision['label']} — {decision['reason']}")


        # ── Background Model / Foreground Mask ────────────────────────────────
        st.divider()
        with st.expander(":material/layers: Background model — foreground mask", expanded=False):
            st.caption(
                "MOG2 foreground mask after binary morphological operations "
                "(closing + dilation). White pixels = detected moving foreground."
            )
            st.image(fg_mask, caption="MOG2 foreground mask (post-morphology)",
                     use_container_width=True)

        # ── KNN Secondary Verification ─────────────────────────────────────────
        if feat_list and any(f is not None for f in feat_list):
            st.divider()
            st.subheader(":material/model_training: KNN Secondary Verification")
            st.caption(
                "Classical KNN classifier (Canny + LBP + HOG features) — diagnostic only. "
                "The final access decision is based on the primary ResNet/RBAC pipeline. "
                "KNN output is shown for comparison and is **never** used to override the decision."
            )

            if not knn_is_ready():
                st.warning(
                    "KNN model not trained yet. Go to Admin Panel and click "
                    "**Retrain KNN Model**, or enrol users via Register User."
                )
            else:
                info = knn_info()
                st.caption(
                    f"Model: KNN k={info['k']} · "
                    f"{info['n_samples']} training samples · "
                    f"{info['n_classes']} enrolled users · "
                    f"confidence threshold: {info['threshold']:.0%}"
                )
                if info.get("low_sample_warning"):
                    st.warning(
                        ":material/warning: One or more users have fewer than 3 training images — "
                        "KNN predictions may be unreliable. Enrol more photos per user."
                    )
                for i, feats in enumerate(feat_list):
                    if feats is None:
                        continue
                    knn_user, knn_conf = predict_knn(feats["feature_vector"])
                    # knn_user is None when confidence is below KNN_CONFIDENCE_THRESHOLD
                    dlib_user = decisions[i].get("username", "") if i < len(decisions) else ""

                    col_k1, col_k2, col_k3 = st.columns(3)

                    if knn_user is None:
                        col_k1.metric(f"Face {i+1} — KNN (diagnostic)", "Unknown")
                        col_k2.metric("KNN confidence (raw)", f"{knn_conf:.0%}")
                        col_k3.caption(
                            ":material/info: Confidence below threshold — "
                            "prediction rejected, result not used."
                        )
                    else:
                        col_k1.metric(f"Face {i+1} — KNN (diagnostic)", knn_user)
                        col_k2.metric("KNN confidence", f"{knn_conf:.0%}")
                        if dlib_user and knn_user == dlib_user:
                            col_k3.success(":material/check_circle: KNN agrees with primary recognizer")
                        elif dlib_user:
                            col_k3.warning(
                                f":material/warning: KNN mismatch — ResNet says `{dlib_user}`. "
                                "KNN result ignored for final decision."
                            )
                        else:
                            col_k3.info(
                                ":material/info: ResNet found no identity match — "
                                "KNN result is diagnostic only."
                            )

        if tailgating:
            st.error("Tailgating detected — multiple persons in frame!")

        if feat_list and any(f is not None for f in feat_list):
            st.divider()
            st.subheader("Intermediate CV outputs")
            for i, feats in enumerate(feat_list):
                if feats is None:
                    continue
                st.markdown(f"**Face {i + 1}**")
                c1, c2, c3, c4 = st.columns(4)
                c1.image(feats["gray_chip"], caption="Aligned (64x64)", clamp=True)
                c2.image(feats["canny_map"], caption="Canny edges",     clamp=True)
                lbp_vis = cv2.normalize(feats["lbp_image"], None, 0, 255,
                                        cv2.NORM_MINMAX).astype(np.uint8)
                c3.image(lbp_vis,           caption="LBP texture",      clamp=True)
                c4.image(feats["hog_vis"],  caption="HOG gradients",    clamp=True)



# ═══════════════════════════════════════════════════════════════════════════════
# LIVE CAMERA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Live Camera":
    st.markdown("# :material/videocam: Live Camera — Testing Mode")
    st.caption(
        "Testing live camera feed. Blink once to confirm liveness. "
        "**Events from this page are not recorded in the access log.**"
    )
    st.info(
        ":material/science: This page is for testing and development only. "
        "Detection results are displayed visually but are **not** written to the access log. "
        "Use the **Security Terminal** for authenticated access events.",
        icon=":material/info:",
    )

    cam_running = st.session_state.get("cam_running", False)

    # Row 1 — Start / Stop / mode / landmarks
    c1, c2, c3, c4 = st.columns([1, 1, 2, 1])
    start = c1.button(
        "Start", type="primary", icon=":material/play_arrow:",
        key="live_start", use_container_width=True, disabled=cam_running,
    )
    stop = c2.button(
        "Stop", icon=":material/stop:",
        key="live_stop", use_container_width=True, disabled=not cam_running,
    )
    c3.toggle(
        "High quality camera", value=True, key="live_hq",
        help="ON: normal anti-spoofing.  OFF: relaxed thresholds (less accurate).",
    )
    c4.checkbox("Show landmarks", value=True, key="live_lm")

    # Row 2 — performance sliders
    s1, s2 = st.columns(2)
    s1.slider("Face recog. interval (frames)", 1, 10, 3, key="live_skip",
              help="Higher = faster feed, less frequent recognition")
    s2.slider("Tracking interval (frames)", 2, 15, 5, key="live_yolo",
              help="Higher = faster feed, less frequent person tracking")

    if not st.session_state.get("live_hq", True):
        st.warning(
            ":material/warning: Low camera quality mode — anti-spoofing accuracy "
            "is reduced. False failures may occur under poor lighting or low resolution."
        )

    st.divider()

    if start:
        st.session_state.update({
            "cam_running":   True,
            "cam_frame":     0,
            "blink_tracker": BlinkTracker(),
            "live_cache":    {},
        })
        reset_tracker()
        reset_background_model()
        st.rerun()   # force a clean full-page rerun so the fragment starts on first click

    if stop:
        st.session_state["cam_running"] = False
        cap_ref = st.session_state.pop("cam_cap", None)
        if cap_ref is not None:
            cap_ref.release()

    _live_camera_loop()


# ═══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Admin Panel":
    st.markdown("# :material/admin_panel_settings: Admin Panel")

    users = get_all_users()
    stats = get_log_stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total users",     len(users))
    c2.metric("Access granted",  stats["ALLOW"])
    c3.metric("Access denied",   stats["DENY"])
    c4.metric("Security alerts", stats["ALERT"])
    st.divider()

    # ── KNN Model Management ──────────────────────────────────────────────────
    st.subheader(":material/model_training: KNN Classifier")
    info = knn_info()
    km1, km2, km3, km4 = st.columns([2, 2, 2, 2])
    km1.metric("Status",   "Ready" if info["ready"] else "Not trained")
    km2.metric("Samples",  info["n_samples"])
    km3.metric("Classes",  info["n_classes"])
    km4.metric("k (neighbours)", info["k"] if info["ready"] else "—")

    if st.button(
        "Retrain KNN Model", icon=":material/model_training:",
        type="primary" if not info["ready"] else "secondary",
        help="Re-scans known_faces/ and retrains the KNN on all enrolled images.",
    ):
        with st.spinner("Scanning face images and training KNN…"):
            knn_ok, knn_msg = train_knn()
        if knn_ok:
            st.success(knn_msg)
            st.rerun()
        else:
            st.error(knn_msg)
    st.divider()

    st.subheader("Manage users")
    if not users:
        st.info("No users enrolled yet.")
    else:
        ROLE_BADGE = {
            "admin":      ":material/shield: Admin",
            "authorized": ":material/check_circle: Authorized",
        }
        for u in users:
            col_name, col_role, col_change, col_del = st.columns(
                [3, 2, 3, 1.5], vertical_alignment="center"
            )
            col_name.markdown(
                f"**{u['name']}** `{u['username']}`  \n"
                f"<small style='color:gray'>{u['created_at']}</small>",
                unsafe_allow_html=True,
            )
            col_role.markdown(ROLE_BADGE.get(u["role"], u["role"]))

            new_role = col_change.selectbox(
                "Role", ["authorized", "admin"],
                index=["authorized", "admin"].index(u["role"])
                      if u["role"] in ("authorized", "admin") else 0,
                key=f"role_{u['username']}",
                label_visibility="collapsed",
            )
            if new_role != u["role"]:
                update_role(u["username"], new_role)
                st.rerun()

            if col_del.button(
                "Remove", key=f"del_{u['username']}",
                icon=":material/delete:", help="Remove user",
                use_container_width=True,
            ):
                delete_user(u["username"])
                folder = os.path.join(KNOWN_FACES_DIR, u["username"])
                if os.path.isdir(folder):
                    shutil.rmtree(folder, ignore_errors=True)
                st.rerun()

    st.divider()

    # ── Blacklist management ──────────────────────────────────────────────────
    st.subheader(":material/block: Blacklist Management")
    bl_entries = get_all_blacklist_entries()
    st.caption(
        f"{len(bl_entries)} blacklisted individual(s) — "
        "their face encodings are checked first in every pipeline."
    )

    if not bl_entries:
        st.info("No blacklist entries. Add threat individuals via Register User → Blacklist Entry.")
    else:
        for entry in bl_entries:
            bc1, bc2, bc3, bc4 = st.columns([3, 3, 2, 1.5], vertical_alignment="center")
            bc1.markdown(
                f"**{entry['name']}**  \n"
                f"<small style='color:gray'>{entry['created_at']}</small>",
                unsafe_allow_html=True,
            )
            bc2.markdown(
                f":material/warning: `{entry['threat_reason']}`  \n"
                f"<small style='color:gray'>{entry.get('notes','') or '—'}</small>",
                unsafe_allow_html=True,
            )
            bc3.markdown(":material/block: **BLACKLISTED**")
            if bc4.button("Remove", key=f"bl_del_{entry['id']}",
                          icon=":material/delete:", use_container_width=True):
                delete_blacklist_entry(entry["id"])
                st.rerun()

    st.divider()

    # ── Password change ───────────────────────────────────────────────────────
    st.subheader(":material/key: Change Password")
    if not users:
        st.info("No users enrolled yet.")
    else:
        with st.form("change_pw_form"):
            pw_col1, pw_col2 = st.columns([2, 3])
            pw_uname   = pw_col1.selectbox("User", [u["username"] for u in users],
                                           format_func=lambda u: f"{u}  ({next((x['name'] for x in users if x['username']==u), '')})")
            new_pw     = pw_col2.text_input("New password", type="password",
                                            placeholder="Min 8 chars, upper, lower, special")
            confirm_pw = pw_col2.text_input("Confirm password", type="password")
            pw_submit  = st.form_submit_button("Update password",
                                               type="primary",
                                               icon=":material/lock_reset:")

        if pw_submit:
            _pw_errors = []
            if len(new_pw) < 8:
                _pw_errors.append("at least 8 characters")
            if not any(c.islower() for c in new_pw):
                _pw_errors.append("a lowercase letter")
            if not any(c.isupper() for c in new_pw):
                _pw_errors.append("an uppercase letter")
            if not any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in new_pw):
                _pw_errors.append("a special character")
            if new_pw != confirm_pw:
                st.error("Passwords do not match.")
            elif _pw_errors:
                st.error("Password must contain: " + ", ".join(_pw_errors))
            else:
                update_password(pw_uname, new_pw)
                st.success(f"Password updated for `{pw_uname}`.")

    st.divider()

    # ── Benchmark ─────────────────────────────────────────────────────────────
    st.subheader(":material/analytics: Recognition Benchmark")
    st.caption(
        "Runs face recognition on every image in known_faces/ and reports "
        "top-1 accuracy, average confidence, and per-frame latency. "
        "Use these numbers in the Experiments and Results section of the report."
    )

    if st.button("Run Benchmark", icon=":material/play_arrow:",
                 disabled=not knn_is_ready()):
        if not knn_is_ready():
            st.warning("Train the KNN model first.")
        else:
            # Load all enrolled encodings once — avoids repeated DB queries
            enrolled = get_all_face_encodings()   # [{username, name, role, encoding}]

            total = correct_resnet = correct_knn = 0
            lat_sum = 0.0
            rows = []

            scan_dir = KNOWN_FACES_DIR
            for username in sorted(os.listdir(scan_dir)):
                user_dir = os.path.join(scan_dir, username)
                if not os.path.isdir(user_dir):
                    continue
                imgs = sorted(
                    f for f in os.listdir(user_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                for img_name in imgs:
                    path = os.path.join(user_dir, img_name)
                    try:
                        raw = open(path, "rb").read()
                        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                        if bgr is None:
                            continue
                        t0  = time.time()
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        locs = face_recognition.face_locations(rgb, model="hog")
                        if not locs:
                            continue
                        encs = face_recognition.face_encodings(rgb, locs)
                        if not encs:
                            continue

                        # ResNet nearest-neighbour match against enrolled encodings
                        best_match = None
                        best_conf  = 0.0
                        for u in enrolled:
                            dist = float(np.linalg.norm(
                                np.array(u["encoding"]) - np.array(encs[0])
                            ))
                            conf = max(0.0, 1.0 - dist)
                            if conf > best_conf:
                                best_conf  = conf
                                best_match = u["username"]

                        lat       = (time.time() - t0) * 1000
                        resnet_ok = (best_match == username)

                        # KNN match
                        feats    = extract_all(bgr, locs[0])
                        knn_ok   = False
                        knn_pred = "—"
                        if feats:
                            knn_pred, _ = predict_knn(feats["feature_vector"])
                            knn_ok = (knn_pred == username)

                        total          += 1
                        correct_resnet += int(resnet_ok)
                        correct_knn    += int(knn_ok)
                        lat_sum        += lat
                        rows.append({
                            "User (GT)":      username,
                            "File":           img_name,
                            "ResNet pred":    best_match or "—",
                            "ResNet ✓/✗":    "✓" if resnet_ok else "✗",
                            "KNN pred":       knn_pred or "—",
                            "KNN ✓/✗":       "✓" if knn_ok else "✗",
                            "Confidence":     f"{best_conf:.1%}",
                            "Latency (ms)":   f"{lat:.1f}",
                        })
                    except Exception:
                        continue

            if total == 0:
                st.warning("No images with detectable faces found in known_faces/.")
            else:
                import pandas as pd
                acc_r = correct_resnet / total
                acc_k = correct_knn    / total
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Images tested",         total)
                b2.metric("ResNet top-1 accuracy", f"{acc_r:.1%}")
                b3.metric("KNN top-1 accuracy",    f"{acc_k:.1%}")
                b4.metric("Avg latency / frame",   f"{lat_sum / total:.1f} ms")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)


# ═══════════════════════════════════════════════════════════════════════════════
# ACCESS LOG
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Access Log":
    st.markdown("# :material/assignment: Access Log")

    df    = get_access_log()
    stats = get_log_stats()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total events",   stats["ALLOW"] + stats["DENY"] + stats["ALERT"])
    c2.metric("Access granted", stats["ALLOW"])
    c3.metric("Alerts",         stats["ALERT"])
    st.divider()

    if df.empty:
        st.info("No events logged yet. Run an analysis first.")
    else:
        col_f1, col_f2 = st.columns(2)
        action_filter = col_f1.multiselect(
            "Filter by action", ["ALLOW", "DENY", "ALERT"],
            default=["ALLOW", "DENY", "ALERT"],
        )
        source_filter = col_f2.multiselect(
            "Filter by source", ["image", "terminal", "live"],
            default=["image", "terminal"],
            help="'live' events are from the Testing page and are no longer logged by default.",
        )

        mask = df["action"].isin(action_filter) & df["source"].isin(source_filter)
        st.dataframe(df[mask].reset_index(drop=True),
                     use_container_width=True, height=420)

        st.divider()
        if st.button("Clear log", type="secondary", icon=":material/delete_sweep:"):
            clear_access_log()
            st.success("Log cleared.")
            st.rerun()
            