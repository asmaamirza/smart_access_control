# Smart Access Control System
**CSCI435 — Computer Vision Algorithms and Systems**

A Streamlit web application that performs intelligent door/entry access control using five computer vision techniques.

---

## Vision Capabilities

| # | Capability | Implementation |
|---|---|---|
| 1 | Face Detection | HOG detector via `face_recognition` (dlib ResNet) |
| 2 | Face Recognition | 128-d face embeddings + **KNN classifier trained on custom data** |
| 3 | Person Tracking | YOLOv8n + ByteTrack multi-object tracker |
| 4 | Background Modelling | MOG2 Gaussian Mixture Model (live camera feed) |
| 5 | Image Enhancement | CLAHE on LAB luminance channel |

---

## Requirements

- **Python 3.10 – 3.13**
- A webcam (for Live Camera mode)
- ~2 GB free disk space (PyTorch + dlib + YOLOv8 weights)

---

## Installation

### Option A — pip (recommended, works on Windows/macOS/Linux)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt
```

> **Windows note:** If `dlib` (pulled in by `face-recognition`) fails to build,
> install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
> (select the **C++ Desktop Development** workload) then run `pip install cmake dlib`
> before installing the rest.

### Option B — Conda (easiest on Windows if Option A fails)

```bash
conda create -n access_control python=3.10
conda activate access_control
conda install -c conda-forge dlib
pip install -r requirements.txt
```

---

## Running the App

```bash
# Make sure your virtual environment is active, then:
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Usage

### 1. Enroll Faces
- Go to **Enroll Faces** in the sidebar.
- Enter a person's name and upload 3–5 clear, front-facing photos.
- Repeat for every authorised person.
- Click **Train KNN Classifier** — this extracts face embeddings and fits the classifier.

### 2. Analyze an Image
- Go to **Analyze Image** and upload a JPG/PNG.
- Click **Run Analysis** to see face recognition, person tracking, and tailgating detection.

### 3. Live Camera
- Go to **Live Camera** and click **START**.
- Allow browser camera access when prompted.
- Face boxes and ACCESS GRANTED / DENIED decisions appear as overlays in real time.
- Adjust the **Recognition rate** slider to balance responsiveness vs CPU usage.

### 4. View the Access Log
- Go to **Access Log** to see all timestamped recognition events.

---

## Project Structure

```
smart_access_control/
├── app.py                      # Streamlit application (all pages)
├── requirements.txt            # Python dependencies
├── README.md
├── modules/
│   ├── face_recognizer.py      # Face detection + KNN recognition
│   ├── person_tracker.py       # YOLOv8 + ByteTrack person tracking
│   ├── background_model.py     # MOG2 background subtraction + CLAHE
│   └── utils.py                # Drawing helpers + CSV access log
├── known_faces/                # Enrolled face images — created at runtime
├── models/                     # Saved KNN classifier — created after training
└── logs/                       # access_log.csv — created at runtime
```

> `known_faces/`, `models/`, and `logs/` are created automatically the first time
> you enroll a face or run an analysis. You do not need to create them manually.

---

## Technical Notes

**Pre-trained model:** dlib's ResNet (via `face_recognition`) extracts 128-dimensional face embeddings. This model is bundled with the `face-recognition-models` package and does not require a separate download.

**Custom-trained model:** A `KNeighborsClassifier` (sklearn) is fitted on the embeddings extracted from enrolled photos. This is trained entirely on user-provided custom data and saved to `models/knn_classifier.pkl`.

**Access decision (double-gate):** A face is only identified as a known person when *both* the KNN confidence score ≥ 0.75 **and** the raw embedding distance to the nearest training sample ≤ 0.45. This prevents the classifier from approving out-of-distribution faces.

**Tailgating detection:** triggered when YOLOv8 tracks more than one person in a single frame simultaneously.

**CLAHE:** applied to the luminance channel (LAB colour space) before recognition to improve detection accuracy under uneven or low lighting.

**YOLOv8 weights:** `yolov8n.pt` (~6 MB) is downloaded automatically by `ultralytics` on first run. An internet connection is required for this one-time download.

**Live camera (WebRTC):** uses a Google STUN server for ICE negotiation. Works on local networks and direct connections. For cross-network deployments a TURN server would be needed.

---

## Dependencies

| Library | Purpose |
|---|---|
| `face_recognition` | Face detection and 128-d embeddings (wraps dlib) |
| `scikit-learn` | KNN classifier |
| `ultralytics` | YOLOv8 person detection and ByteTrack tracking |
| `opencv-python` | MOG2, CLAHE, image I/O, frame drawing |
| `streamlit` | Web interface |
| `streamlit-webrtc` | Real-time WebRTC camera stream in the browser |
| `av` | PyAV — video frame decoding/encoding for streamlit-webrtc |
| `pandas` | Access log CSV handling |
