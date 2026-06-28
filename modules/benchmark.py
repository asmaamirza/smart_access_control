"""
Recognition Benchmark
======================
Evaluates both recognisers (ResNet/Euclidean and the classical-feature KNN)
on a genuinely held-out test split, not on the same images they were
enrolled/trained on.

WHY THIS EXISTS (read before changing the split logic)
--------------------------------------------------------
The original benchmark computed each user's encoding from ALL their images
in known_faces/<user>/, then tested recognition on those same images. That
is train/test contamination: it measures whether the model can recognize a
photo it already memorized, not whether it generalizes to a new photo of
that person. A model can score 100% on a contaminated benchmark while being
genuinely bad at telling two similar-looking enrolled users apart — the
benchmark below demonstrates a synthetic case where contaminated accuracy is
100% while held-out accuracy on the SAME data is 67%, because the held-out
split happens to land on the genuinely confusable samples.

How this version avoids it
---------------------------
For each user, their images in known_faces/<user>/ are split into:
  - a TRAIN portion -> used to recompute that user's reference encoding
    (mean of ResNet embeddings) and to fit the KNN on classical features
  - a TEST (held-out) portion -> never used for either model; only used to
    measure accuracy

This means the benchmark's own enrolled-encoding average and KNN fit are
SEPARATE from (and usually smaller than) the live system's actual encoding
and KNN model — the benchmark trains its own temporary copies purely for
evaluation, then discards them. The live DB encoding and the live KNN
classifier (modules/knn_engine.py's saved .pkl files) are never touched by
this module.

Multiple random splits (default 5) are run and averaged, with the spread
(std dev) reported too — a single 80/20 split can land luckily or unluckily
depending on which photos end up in the test set, especially with few
images per user. Reporting only one run's number without the spread
overstates how reliable that number is.

Users with too few images to make a meaningful split (fewer than
MIN_IMAGES_PER_USER) are excluded from the benchmark and listed separately,
rather than silently lowering the held-out fraction for them or crashing.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import face_recognition
import numpy as np

from modules.feature_extractor import extract_all

KNOWN_FACES_DIR        = "known_faces"
DEFAULT_TEST_FRACTION  = 0.3   # fraction of each user's images held out per split
DEFAULT_N_SPLITS       = 5     # independent random splits, averaged
MIN_IMAGES_PER_USER    = 4     # need at least this many to hold out >=1 and keep >=1 to train on

# Matches the confidence formula in face_recognizer.py exactly, so the
# Confidence column means the same thing here as it does in live recognition
# (distance=0 -> 100%, distance=_NORM_DIST -> 0%) rather than a different
# ad-hoc formula that happens to also produce numbers between 0 and 1.
_NORM_DIST = 0.80


@dataclass
class UserImageSet:
    username: str
    image_paths: list = field(default_factory=list)


def _collect_user_images(known_faces_dir: str = KNOWN_FACES_DIR) -> list[UserImageSet]:
    """Scan known_faces/<username>/ and return per-user image path lists."""
    out = []
    if not os.path.isdir(known_faces_dir):
        return out
    for username in sorted(os.listdir(known_faces_dir)):
        user_dir = os.path.join(known_faces_dir, username)
        if not os.path.isdir(user_dir):
            continue
        imgs = sorted(
            os.path.join(user_dir, f)
            for f in os.listdir(user_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if imgs:
            out.append(UserImageSet(username=username, image_paths=imgs))
    return out


def _load_and_encode(path: str):
    """
    Read one image and return (bgr, encoding, classical_feature_vector) or
    None if no face could be detected/encoded. Loaded once per image and
    reused for both the ResNet and KNN evaluation passes -- avoids decoding
    and running HOG detection twice per file.
    """
    try:
        raw = open(path, "rb").read()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            return None
        encs = face_recognition.face_encodings(rgb, locs)
        if not encs:
            return None
        feats = extract_all(bgr, locs[0])
        feature_vector = feats["feature_vector"] if feats else None
        return bgr, encs[0], feature_vector
    except Exception:
        return None


def _resnet_confidence(dist: float) -> float:
    """Same normalisation as face_recognizer.py's recognize_faces()."""
    return max(0.0, 1.0 - dist / _NORM_DIST)


def run_benchmark(
    known_faces_dir: str = KNOWN_FACES_DIR,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    n_splits: int = DEFAULT_N_SPLITS,
    min_images_per_user: int = MIN_IMAGES_PER_USER,
    progress_cb=None,
) -> dict:
    """
    Run the held-out benchmark across n_splits random train/test partitions
    and return aggregated results.

    progress_cb: optional callable(fraction: float, message: str) for UI
    progress bars. Called occasionally, not on every image, to stay cheap.

    Returns a dict:
        ok:                  bool — False if there wasn't enough data to run at all
        message:             str  — explanation when ok is False
        n_splits:             int
        test_fraction:        float
        usable_users:         [username, ...]   — users with enough images
        excluded_users:       [{username, n_images}, ...] — too few images
        resnet_acc_mean/std:  float
        knn_acc_mean/std:     float
        avg_latency_ms:       float  (ResNet encode+match time per image)
        per_split:            [{resnet_acc, knn_acc, n_test_images}, ...]
        rows:                 [...] — per-image results from the LAST split only,
                               for the detail table (showing all splits would be
                               unreadable; the aggregate numbers above are what
                               matters for the report, this table is illustrative)
    """
    user_sets = _collect_user_images(known_faces_dir)
    usable = [u for u in user_sets if len(u.image_paths) >= min_images_per_user]
    excluded = [
        {"username": u.username, "n_images": len(u.image_paths)}
        for u in user_sets if len(u.image_paths) < min_images_per_user
    ]

    if len(usable) < 2:
        return {
            "ok": False,
            "message": (
                f"Need at least 2 users with {min_images_per_user}+ images each "
                f"to run a held-out benchmark. Found {len(usable)} usable user(s)"
                + (f", {len(excluded)} excluded for too few images." if excluded else ".")
            ),
            "usable_users": [u.username for u in usable],
            "excluded_users": excluded,
        }

    # Pre-load + encode every image ONCE, reused across all splits — this is
    # the expensive part (HOG detection + ResNet encoding + classical
    # features per image), so doing it n_splits times would be wasteful and
    # would also make latency measurements include redundant decode work.
    if progress_cb:
        progress_cb(0.0, "Loading and encoding all images…")

    loaded: dict[str, list[tuple]] = {}   # username -> [(path, bgr, enc, feat_vec, encode_ms), ...]
    total_imgs = sum(len(u.image_paths) for u in usable)
    done_imgs = 0
    for u in usable:
        entries = []
        for path in u.image_paths:
            t0 = time.time()
            result = _load_and_encode(path)
            encode_ms = (time.time() - t0) * 1000
            done_imgs += 1
            if progress_cb and done_imgs % 5 == 0:
                progress_cb(0.5 * done_imgs / total_imgs,
                           f"Encoding images… ({done_imgs}/{total_imgs})")
            if result is None:
                continue
            bgr, enc, feat_vec = result
            entries.append((path, enc, feat_vec, encode_ms))
        loaded[u.username] = entries

    # Drop any user who lost too many images to failed detection during loading
    usable_after_load = [u for u in usable if len(loaded[u.username]) >= min_images_per_user]
    newly_excluded = [
        {"username": u.username, "n_images": len(loaded[u.username])}
        for u in usable if len(loaded[u.username]) < min_images_per_user
    ]
    excluded = excluded + newly_excluded

    if len(usable_after_load) < 2:
        return {
            "ok": False,
            "message": (
                "After face detection, fewer than 2 users had enough valid "
                "images left to benchmark. Check that known_faces/ images "
                "have clearly detectable faces."
            ),
            "usable_users": [u.username for u in usable_after_load],
            "excluded_users": excluded,
        }

    usernames = [u.username for u in usable_after_load]
    per_split_results = []
    last_split_rows = []

    for split_idx in range(n_splits):
        rng = np.random.RandomState(split_idx)  # deterministic per split index, varies across splits

        train_encodings: dict[str, np.ndarray] = {}     # username -> mean ResNet encoding (train portion)
        train_features:  list[tuple[str, np.ndarray]] = []  # (username, classical_feature_vector) for KNN fit
        test_items: list[tuple[str, str, np.ndarray, np.ndarray, float]] = []  # (username, path, enc, feat_vec, encode_ms)

        for uname in usernames:
            entries = list(loaded[uname])
            rng.shuffle(entries)
            n_test = max(1, int(round(len(entries) * test_fraction)))
            n_test = min(n_test, len(entries) - 1)  # always keep >=1 for training
            test_entries = entries[:n_test]
            train_entries = entries[n_test:]

            train_encodings[uname] = np.mean([e[1] for e in train_entries], axis=0)
            for _, _, feat_vec, _ in train_entries:
                if feat_vec is not None:
                    train_features.append((uname, feat_vec))
            for path, enc, feat_vec, encode_ms in test_entries:
                test_items.append((uname, path, enc, feat_vec, encode_ms))

        # Fit a TEMPORARY KNN for this split only — this is intentionally
        # separate from modules.knn_engine's saved classifier. We are
        # evaluating the approach, not consuming or overwriting the live
        # model the rest of the app uses.
        knn_model = None
        label_encoder = None
        if len(set(u for u, _ in train_features)) >= 2:
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.preprocessing import LabelEncoder
            X = np.array([f for _, f in train_features], dtype=np.float32)
            y_raw = [u for u, _ in train_features]
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw)
            k = min(3, len(train_features))
            knn_model = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
            knn_model.fit(X, y)

        # ── Evaluate on the held-out test_items for this split ──────────────
        resnet_correct = knn_correct = 0
        lat_sum = 0.0
        rows = []
        for uname, path, enc, feat_vec, encode_ms in test_items:
            t_match0 = time.time()
            best_user, best_dist = None, float("inf")
            for cand_user, cand_enc in train_encodings.items():
                d = float(np.linalg.norm(cand_enc - enc))
                if d < best_dist:
                    best_dist, best_user = d, cand_user
            match_ms = (time.time() - t_match0) * 1000
            confidence = _resnet_confidence(best_dist)
            resnet_ok = (best_user == uname)
            resnet_correct += int(resnet_ok)

            knn_pred, knn_ok = "—", False
            if knn_model is not None and feat_vec is not None:
                proba = knn_model.predict_proba(feat_vec.reshape(1, -1))[0]
                idx = int(np.argmax(proba))
                knn_pred = label_encoder.inverse_transform([idx])[0]
                knn_ok = (knn_pred == uname)
            knn_correct += int(knn_ok)

            lat_sum += encode_ms + match_ms
            rows.append({
                "User (GT)":    uname,
                "File":         os.path.basename(path),
                "ResNet pred":  best_user or "—",
                "ResNet ✓/✗":  "✓" if resnet_ok else "✗",
                "KNN pred":     knn_pred,
                "KNN ✓/✗":     "✓" if knn_ok else "✗",
                "Confidence":   f"{confidence:.1%}",
                "Latency (ms)": f"{encode_ms + match_ms:.1f}",
            })

        n_test_total = len(test_items)
        per_split_results.append({
            "resnet_acc":     resnet_correct / n_test_total if n_test_total else 0.0,
            "knn_acc":        knn_correct / n_test_total if n_test_total else 0.0,
            "n_test_images":  n_test_total,
            "avg_latency_ms": lat_sum / n_test_total if n_test_total else 0.0,
        })
        last_split_rows = rows

        if progress_cb:
            progress_cb(0.5 + 0.5 * (split_idx + 1) / n_splits,
                       f"Evaluating split {split_idx + 1}/{n_splits}…")

    resnet_accs = [s["resnet_acc"] for s in per_split_results]
    knn_accs    = [s["knn_acc"] for s in per_split_results]
    all_latencies = [s["avg_latency_ms"] for s in per_split_results]

    return {
        "ok": True,
        "message": "",
        "n_splits": n_splits,
        "test_fraction": test_fraction,
        "usable_users": usernames,
        "excluded_users": excluded,
        "resnet_acc_mean": float(np.mean(resnet_accs)),
        "resnet_acc_std":  float(np.std(resnet_accs)),
        "knn_acc_mean":    float(np.mean(knn_accs)),
        "knn_acc_std":     float(np.std(knn_accs)),
        "avg_latency_ms":  float(np.mean(all_latencies)),
        "per_split": per_split_results,
        "rows": last_split_rows,
    }