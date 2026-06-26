"""
KNN Face Classifier
===================
Secondary recogniser trained on classical feature vectors:
    Canny edge map  +  LBP histogram  +  HOG descriptor  (concatenated)

This model is trained on the group's own enrolled face images (custom data),
directly satisfying the project requirement for at least one model trained or
fine-tuned on custom data.

Training  : scan known_faces/<username>/ dirs, extract feature vectors per
            image, fit KNeighborsClassifier, serialise to models/.
Inference : load pkl, return predicted username + softmax-like confidence.
"""

import os
import pickle

import cv2
import face_recognition
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from modules.feature_extractor import extract_all

MODELS_DIR      = "models"
KNN_PATH        = os.path.join(MODELS_DIR, "knn_classifier.pkl")
ENCODER_PATH    = os.path.join(MODELS_DIR, "label_encoder.pkl")
KNOWN_FACES_DIR = "known_faces"

# Minimum confidence for a KNN prediction to be considered reliable.
# Predictions below this threshold are returned as (None, raw_conf) so callers
# display "Unknown / Low confidence" instead of a potentially wrong username.
KNN_CONFIDENCE_THRESHOLD = 0.45


# ── Training ──────────────────────────────────────────────────────────────────

def train_knn(known_faces_dir: str = KNOWN_FACES_DIR) -> tuple[bool, str]:
    """
    Scan known_faces/<username>/ directories, extract classical feature vectors
    from every face image, fit a KNN on the resulting labelled dataset, and
    serialise to models/knn_classifier.pkl and models/label_encoder.pkl.

    Returns (success, message).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    vectors: list[np.ndarray] = []
    labels:  list[str]        = []

    if not os.path.isdir(known_faces_dir):
        return False, "known_faces/ directory not found."

    for username in sorted(os.listdir(known_faces_dir)):
        user_dir = os.path.join(known_faces_dir, username)
        if not os.path.isdir(user_dir):
            continue

        imgs = sorted(
            f for f in os.listdir(user_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not imgs:
            continue

        for img_name in imgs:
            path = os.path.join(user_dir, img_name)
            try:
                raw = open(path, "rb").read()
                bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb, model="hog")
                if not locs:
                    continue
                feats = extract_all(bgr, locs[0])
                if feats is None:
                    continue
                vectors.append(feats["feature_vector"])
                labels.append(username)
            except Exception:
                continue

    n_classes = len(set(labels))
    if n_classes < 2:
        return False, (
            f"Need at least 2 users with face images to train the KNN. "
            f"Found {n_classes}. Enrol more users first."
        )

    X    = np.array(vectors, dtype=np.float32)
    y    = np.array(labels)
    le   = LabelEncoder()
    y_enc = le.fit_transform(y)

    k   = min(3, len(vectors))
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="distance")
    knn.fit(X, y_enc)

    with open(KNN_PATH,     "wb") as f:
        pickle.dump(knn, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    return True, (
        f"KNN trained on {len(vectors)} samples across {n_classes} users "
        f"(k={k}, features: Canny+LBP+HOG)."
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_knn(feature_vector: np.ndarray) -> tuple[str | None, float]:
    """
    Classify a face by its classical feature vector.

    Returns (predicted_username, confidence) or (None, 0.0) if the model has
    not been trained yet or the vector is incompatible.
    """
    if not knn_is_ready():
        return None, 0.0

    try:
        with open(KNN_PATH,     "rb") as f:
            knn = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            le  = pickle.load(f)

        X     = feature_vector.reshape(1, -1)
        proba = knn.predict_proba(X)[0]
        idx   = int(np.argmax(proba))
        conf  = float(proba[idx])
        name  = le.inverse_transform([idx])[0]

        # Reject low-confidence predictions to avoid confidently wrong output.
        # Return None so callers display "Unknown" rather than a wrong username.
        if conf < KNN_CONFIDENCE_THRESHOLD:
            return None, conf
        return name, conf
    except Exception:
        return None, 0.0


def knn_is_ready() -> bool:
    """True when both pkl files exist (model has been trained at least once)."""
    return os.path.exists(KNN_PATH) and os.path.exists(ENCODER_PATH)


def knn_info() -> dict:
    """Return basic stats about the saved model for display in the UI."""
    if not knn_is_ready():
        return {"ready": False, "n_samples": 0, "n_classes": 0, "k": 0,
                "threshold": KNN_CONFIDENCE_THRESHOLD, "low_sample_warning": False}
    try:
        with open(KNN_PATH,     "rb") as f:
            knn = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            le  = pickle.load(f)
        y_enc          = knn._y
        n_samples      = len(knn._fit_X)
        n_classes      = len(le.classes_)
        counts_per_cls = [int(np.sum(y_enc == i)) for i in range(n_classes)]
        low_sample     = any(c < 3 for c in counts_per_cls)
        return {
            "ready":              True,
            "n_samples":          n_samples,
            "n_classes":          n_classes,
            "k":                  knn.n_neighbors,
            "threshold":          KNN_CONFIDENCE_THRESHOLD,
            "low_sample_warning": low_sample,
            "samples_per_class":  counts_per_cls,
        }
    except Exception:
        return {"ready": False, "n_samples": 0, "n_classes": 0, "k": 0,
                "threshold": KNN_CONFIDENCE_THRESHOLD, "low_sample_warning": False}