# Smart Access Control System
**CSCI435 — Computer Vision Algorithms and Systems**  
University of Wollongong in Dubai

A deployable Streamlit web application that implements enterprise-grade physical access control using **11 integrated computer vision techniques**, a two-factor authentication terminal, anti-spoofing liveness detection, a threat blacklist, and a role-based access decision engine.

---

## Vision Capabilities

| # | Capability | Implementation |
|---|---|---|
| 1 | Image enhancement | CLAHE on CIE-LAB luminance channel — normalises uneven lighting before detection |
| 2 | Edge detection | Canny on aligned 64×64 face chip — structural descriptor in KNN feature vector |
| 3 | Keypoint detection | dlib 68-point facial landmark predictor — eyes, nose, mouth positions |
| 4 | Object detection | YOLOv8n-seg (COCO-pretrained) — person detection for tailgating |
| 5 | Image segmentation | YOLOv8n-seg instance masks — per-pixel person segmentation drawn as overlay |
| 6 | Object recognition | dlib ResNet-34 128-d face embedding + Euclidean distance matching |
| 7 | Face detection / recognition | HOG+SVM detector + ResNet identity matching |
| 8 | Video processing | Frame-by-frame webcam loop (`@st.fragment`) and uploaded video file processing |
| 9 | Change detection & background modelling | MOG2 Gaussian Mixture Model foreground/background separation |
| 10 | Object tracking | YOLOv8n-seg person detection; tailgating flagged when `len(detections) > 1` |
| 11 | Binary morphological operations | Named `apply_morphology()` — closing then dilation on MOG2 foreground mask |

---

## Requirements

- **Python 3.10 – 3.13**
- A webcam (for Live Camera and Security Terminal)
- ~2 GB free disk space (dlib models + YOLOv8 weights)
- Windows, macOS, or Linux

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/asmaamirza/smart_access_control.git
cd smart_access_control
```

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Windows — dlib build failure?**  
> If `face-recognition` fails to install, first run:
> ```bash
> pip install cmake
> pip install dlib
> ```
> If that still fails, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the **Desktop development with C++** workload selected, then retry.

> **Conda alternative (easiest on Windows):**
> ```bash
> conda create -n access_control python=3.10
> conda activate access_control
> conda install -c conda-forge dlib
> pip install -r requirements.txt
> ```

---

## Running the App

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Pages and Usage

### Home
Overview of the full vision pipeline, all roles, and quick-start guide.

### Security Terminal *(two-factor authentication)*
1. Enter your **username** and **password** — credentials are verified first.
2. If valid, the webcam opens and waits for a **stable face** (hold still for 3 consecutive frames).
3. While stable, **YOLO tailgating check** runs every 5 frames — if more than one person is detected, access is immediately denied and logged with `tailgating=True`.
4. **Blacklist check** runs first on the detected face encoding — a blacklisted individual using stolen credentials is caught here.
5. **Blink once** to confirm liveness (EAR blink detection).
6. **Anti-spoofing** runs: static texture (LBP) + sharpness (Laplacian) + temporal motion check + DCT replay artifact check.
7. **1-vs-1 face verification** against the claimed username's stored encoding.
8. Result: **ACCESS GRANTED** (green) or **SECURITY ALERT / ACCESS DENIED** (red).

> **Camera quality toggle:** ON = normal anti-spoofing thresholds. OFF = relaxed thresholds with a warning banner (use for low-resolution or poor-lighting cameras).

### Register User
Three tabs:

| Tab | Purpose |
|---|---|
| **Enroll Authorized User** | Create an `admin` or `authorized` account with username, password (min 8 chars, upper+lower+special), and 3–5 face photos |
| **Blacklist Entry** | Add a threat individual by name, threat reason, and face photos — no credentials required |
| **Bulk Import** | Import multiple users from the Pins Face Recognition dataset |

After enrolment the KNN classifier is automatically retrained on all enrolled face images.

### Identify: Image
Upload a JPG/PNG and run the full pipeline in one click:
- CLAHE → HOG detection → ResNet matching → anti-spoofing → RBAC decision → YOLOv8n-seg tailgating
- Shows annotated result, access decision, KNN secondary verification, and intermediate CV outputs (Canny edge map, LBP texture, HOG gradients)
- Expandable **Background model** panel showing the MOG2 foreground mask after morphological operations
- Expandable **Test samples** panel — browse images from `test/image_detection/`, `test/spoof_samples/`, and `test/tailgating_samples/` directly in the UI
- **Camera quality toggle:** replaces the old security mode slider — maps to normal (ON) or relaxed (OFF) thresholds

### Identify: Video *(third input modality)*
Upload a video file (MP4, AVI, MOV, MKV) and run the full pipeline on sampled frames:
- Configurable frame stride (process every N frames) for speed control
- Displays summary stats: frames granted / denied / tailgating count
- Shows a grid of up to 8 annotated sample frames with per-frame verdict labels

### Live Camera
Real-time webcam feed with full pipeline:
- **Start works on first click** — camera initialises cleanly via forced rerun
- **High quality camera toggle** replaces old security mode slider — ON = normal thresholds, OFF = relaxed with warning banner
- Frame-skip sliders for face recognition interval and YOLO interval
- **Stability gate** — shows "HOLD STILL" overlay until face is stable for 3 consecutive frames before running recognition
- Live FPS counter, blink counter, spoof status (`LIVE` / `WAITING FOR BLINK` / `SPOOF SUSPECTED`), and pipeline stage log (`computed` / `cached` / `skipped`)
- Tailgating alert when more than one person is detected
- CLAHE cached every 4 frames; face recognition and YOLO run at configurable intervals to reduce per-frame CPU load

### Admin Panel
- View system stats (total users, ALLOW/DENY/ALERT counts)
- Retrain the KNN classifier manually
- Change user roles or remove users
- View and remove blacklist entries
- **Benchmark** — runs ResNet + KNN recognition on every enrolled image and reports top-1 accuracy, average confidence, and per-frame latency

### Access Log
Searchable, filterable table of every access event with timestamp, name, role, confidence, action, and source (image / video / live / terminal).

---

## Camera Quality Toggle

The security mode slider has been replaced with a **High quality camera** toggle present on Live Camera, Security Terminal, and Identify Image pages.

| Toggle | Security Mode | Anti-Spoofing Behaviour |
|---|---|---|
| ON (default) | Normal | LBP + Laplacian + blink + temporal motion + DCT replay all active |
| OFF | Relaxed | Blink required; static checks advisory; warning banner shown |

The underlying face recognition distance thresholds are unchanged:

| Mode | Max Distance | Min Confidence |
|---|---|---|
| Normal | 0.60 | 25% |
| Relaxed | 0.70 | 12% |

---

## Pipeline Order

### Live Camera / Identify Image
```
Frame → CLAHE (cached every 4 frames in live mode)
      → MOG2 background subtraction → morphological close+dilate → motion regions
      → Stability gate — skip recognition if face is moving (live mode only)
      → HOG face detection → 68-point landmark extraction
      → BLACKLIST CHECK FIRST (0.50 threshold, always strict)
         ↳ Match → ALERT + stop
         ↳ No match → ResNet full-DB search
      → Anti-spoofing:
         • EAR blink (BlinkTracker)
         • LBP texture variance
         • Laplacian sharpness
         • Temporal motion check (frame-to-frame pixel diff)
         • DCT replay artifact check
         • aggregate_spoof_result() → final LIVE / SPOOF SUSPECTED verdict
      → RBAC decision (ALLOW / DENY / ALERT)
      → YOLOv8n person detection + tailgating check (every 5 frames in live mode)
      → Log to SQLite
```

### Security Terminal
```
Credentials (username + password) → verify_credentials()
      → Webcam open → wait for stable face (3 consecutive stable frames)
      → YOLO tailgating check (every 5 frames)
         ↳ Tailgating → DENY + log tailgating=True + stop
      → BLACKLIST CHECK FIRST
         ↳ Match → ALERT + stop
      → Blink liveness gate
         ↳ Not passed → "BLINK ONCE" + continue scanning
      → Full spoof check (static + temporal + DCT)
         ↳ Fail → DENY + stop
      → ResNet 1-vs-1 verify against claimed username
      → RBAC decision → log → result page
```

---

## Project Structure

```
smart_access_control/
├── app.py                      # All eight pages and pipeline orchestration
├── requirements.txt
├── modules/
│   ├── anti_spoofing.py        # EAR blink, LBP texture, Laplacian, temporal, DCT replay checks
│   ├── background_model.py     # CLAHE + MOG2 + apply_morphology() (closing + dilation)
│   ├── database.py             # SQLite CRUD — users, blacklist, access_log tables
│   ├── face_recognizer.py      # HOG detect + ResNet embed + blacklist check + 1-vs-1 verify
│   ├── feature_extractor.py    # Face alignment, Canny edge, LBP texture, HOG → feature vector
│   ├── knn_engine.py           # KNN train/predict on Canny+LBP+HOG features (custom data)
│   ├── person_tracker.py       # YOLOv8n-seg — person detection + instance segmentation masks
│   ├── rbac_engine.py          # ALLOW / DENY / ALERT role-based decisions
│   └── utils.py                # OpenCV annotation helpers (draws seg masks, landmarks, badges)
├── database/
│   └── access_control.db       # SQLite database (auto-created on first run)
├── known_faces/<username>/     # Enrolled face images (auto-created on enrolment)
├── models/
│   ├── knn_classifier.pkl      # Trained KNN (auto-generated after enrolment)
│   └── label_encoder.pkl       # Label encoder (auto-generated after enrolment)
└── test/
    ├── image_detection/        # Sample images for Identify: Image testing
    ├── spoof_samples/          # Sample images for anti-spoofing testing
    └── tailgating_samples/     # Sample images for tailgating detection testing
```

> `database/`, `known_faces/`, and `models/` are created automatically. You do not need to create them manually.  
> YOLOv8n-seg weights (`yolov8n-seg.pt`) are downloaded automatically by Ultralytics on first use.

---

## Models Used

| Model | Type | Source | Purpose |
|---|---|---|---|
| dlib HOG+SVM | Pre-trained | `face_recognition_models` | Face bounding box detection |
| dlib ResNet-34 | Pre-trained | `face_recognition_models` | 128-d face embedding (LFW 99.38%) |
| dlib shape predictor | Pre-trained | `face_recognition_models` | 68 facial landmark keypoints |
| KNN Classifier | **Trained on custom data** | scikit-learn, trained at runtime | Secondary identity verification (Canny+LBP+HOG features) |
| YOLOv8n-seg | Pre-trained (COCO) | Ultralytics | Person detection + instance segmentation; tailgating = `len > 1` |
| MOG2 | Statistical model | OpenCV | Background subtraction + motion gating |

The **KNN classifier** is the project's custom-trained model: it is fitted from scratch on Canny+LBP+HOG feature vectors extracted from each enrolled user's face images. It retrains automatically on every new enrolment.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.32 | Web interface and camera fragment |
| `opencv-python` | ≥4.9 | CLAHE, MOG2, morphology, Canny, drawing |
| `face-recognition` | ≥1.3 | dlib HOG detector + ResNet embeddings |
| `numpy` | ≥1.24 | Array operations |
| `scikit-learn` | ≥1.4 | KNN classifier |
| `scikit-image` | ≥0.21 | LBP texture descriptor, HOG visualisation |
| `ultralytics` | ≥8.1 | YOLOv8n-seg detection + segmentation |
| `Pillow` | ≥10.2 | Image decode for uploads |
| `pandas` | ≥2.1 | Access log and benchmark results display |
