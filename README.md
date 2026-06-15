# Smart Access Control System
**CSCI435 — Computer Vision Algorithms and Systems**  
University of Wollongong in Dubai

A deployable Streamlit web application that implements enterprise-grade physical access control using **10 integrated computer vision techniques**, a two-factor authentication terminal, anti-spoofing liveness detection, a threat blacklist, and a role-based access decision engine.

---

## Vision Capabilities

| # | Capability | Implementation |
|---|---|---|
| 1 | Image enhancement | CLAHE on CIE-LAB luminance channel — normalises uneven lighting before detection |
| 2 | Edge detection | Canny on aligned 64×64 face chip — part of KNN feature vector |
| 3 | Keypoint detection | dlib 68-point facial landmark predictor — eyes, nose, mouth |
| 4 | Object detection | YOLOv8n (COCO-pretrained) — person detection for tailgating |
| 5 | Object recognition | dlib ResNet-34 128-d face embedding + Euclidean distance matching |
| 6 | Face detection / recognition | HOG+SVM detector + ResNet identity matching |
| 7 | Video processing | Frame-by-frame webcam loop with `@st.fragment`; motion-gated recognition |
| 8 | Change detection & background modelling | MOG2 Gaussian Mixture Model foreground/background separation |
| 9 | Object tracking | ByteTrack multi-person tracker with persistent track IDs; tailgating detection |
| 10 | Binary morphological operations | Dilate + erode on MOG2 foreground mask to remove noise |

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
2. If valid, the webcam activates for **face verification** against your enrolled encoding (1-vs-1 match, not a database scan).
3. **Blink once** to confirm liveness (EAR blink detection).
4. System runs the blacklist check on your face first — a blacklisted individual using stolen credentials is caught here.
5. Result: **ACCESS GRANTED** (green) or **SECURITY ALERT / ACCESS DENIED** (red).

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
- CLAHE → HOG detection → ResNet matching → anti-spoofing → RBAC decision
- Shows annotated result image, access decision, KNN secondary verification, and intermediate CV outputs (Canny edge map, LBP texture, HOG gradients)

### Live Camera
Real-time webcam feed with full pipeline:
- Start/Stop controls, security mode slider (strict / normal / relaxed), landmark toggle
- Frame-skip sliders for face recognition and YOLO intervals
- Live FPS counter, blink counter, and pipeline stage indicator
- Tailgating alert when more than one person is tracked simultaneously

### Admin Panel
- View system stats (total users, ALLOW/DENY/ALERT counts)
- Retrain the KNN classifier manually
- Change user roles or remove users
- View and remove blacklist entries

### Access Log
Searchable, filterable table of every access event with timestamp, name, role, confidence, action, and source (image / live / terminal).

---

## Security Modes

| Mode | Max Distance | Min Confidence | Use Case |
|---|---|---|---|
| Strict | 0.50 | 40% | High-security environments |
| Normal | 0.60 | 25% | Standard operation |
| Relaxed | 0.70 | 12% | Demo / low-risk |

---

## Pipeline Order (all modes)

```
Frame → CLAHE → MOG2 motion gate → HOG face detect
      → BLACKLIST CHECK FIRST (0.50 threshold, always strict)
         ↳ Match → ALERT + stop
         ↳ No match → ResNet 1-vs-1 verify (Terminal) or full-DB search (Live/Image)
      → Anti-spoofing (EAR blink + LBP texture + Laplacian sharpness)
      → RBAC decision (ALLOW / DENY / ALERT)
      → YOLOv8n + ByteTrack tailgating check
      → Log to SQLite
```

---

## Project Structure

```
smart_access_control/
├── app.py                      # All seven pages and pipeline orchestration
├── requirements.txt
├── yolov8n.pt                  # YOLOv8n weights (COCO-pretrained, ~6 MB)
├── modules/
│   ├── anti_spoofing.py        # EAR blink detection, LBP texture, Laplacian sharpness
│   ├── background_model.py     # CLAHE + MOG2 + morphological cleanup
│   ├── database.py             # SQLite CRUD — users, blacklist, access_log tables
│   ├── face_recognizer.py      # HOG detect + ResNet embed + blacklist check + 1-vs-1 verify
│   ├── feature_extractor.py    # Face alignment, Canny, LBP, HOG → feature vector
│   ├── knn_engine.py           # KNN train/predict on Canny+LBP+HOG features (custom data)
│   ├── person_tracker.py       # YOLOv8n + ByteTrack tailgating detection
│   ├── rbac_engine.py          # ALLOW / DENY / ALERT role-based decisions
│   └── utils.py                # OpenCV annotation helpers
├── database/
│   └── access_control.db       # SQLite database (auto-created on first run)
├── known_faces/<username>/     # Enrolled face images (auto-created on enrolment)
└── models/
    ├── knn_classifier.pkl      # Trained KNN (auto-generated after enrolment)
    └── label_encoder.pkl       # Label encoder (auto-generated after enrolment)
```

> `database/`, `known_faces/`, and `models/` are created automatically. You do not need to create them manually.

---

## Models Used

| Model | Type | Source | Purpose |
|---|---|---|---|
| dlib HOG+SVM | Pre-trained | `face_recognition_models` | Face bounding box detection |
| dlib ResNet-34 | Pre-trained | `face_recognition_models` | 128-d face embedding (LFW 99.38%) |
| dlib shape predictor | Pre-trained | `face_recognition_models` | 68 facial landmark keypoints |
| KNN Classifier | **Trained on custom data** | scikit-learn, trained at runtime | Secondary identity verification (Canny+LBP+HOG features) |
| YOLOv8n | Pre-trained (COCO) | Ultralytics | Person detection for tailgating |
| MOG2 | Statistical model | OpenCV | Background subtraction, motion gating |

The **KNN classifier** is the project's custom-trained model: it is fitted from scratch on Canny+LBP+HOG feature vectors extracted from each enrolled user's face images. It retrains automatically on every new enrolment.

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.32 | Web interface and camera fragment |
| `opencv-python` | ≥4.9 | CLAHE, MOG2, morphology, drawing |
| `face-recognition` | ≥1.3 | dlib HOG detector + ResNet embeddings |
| `numpy` | ≥1.24 | Array operations |
| `scikit-learn` | ≥1.4 | KNN classifier |
| `scikit-image` | ≥0.21 | LBP texture descriptor |
| `ultralytics` | ≥8.1 | YOLOv8n + ByteTrack |
| `Pillow` | ≥10.2 | Image decode for uploads |
| `pandas` | ≥2.1 | Access log display |
