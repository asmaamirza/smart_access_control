"""
Person Detection and Multi-Object Tracking Module
===================================================
Uses YOLOv8n (nano variant) for real-time person detection and ByteTrack for
maintaining consistent track IDs across frames.

ByteTrack improves on SORT by also associating low-confidence detections using
IoU overlap, which helps keep IDs stable through partial occlusions and brief
disappearances.

Tailgating is defined as more than one person appearing in the same frame —
a simple but effective heuristic for an access-control entry point where only
one person should pass at a time.
"""

from ultralytics import YOLO

# Singleton model — loaded once and reused across all calls to avoid the
# multi-second startup cost on every frame.
_model = None


def _get_model():
    """Return the shared YOLOv8n instance, initialising it on first call."""
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def reset_tracker():
    """
    Discard the current model instance and reload it fresh.
    Clears ByteTrack's internal state (track IDs, kalman filters) — call this
    before processing a new, unrelated video clip.
    """
    global _model
    _model = YOLO("yolov8n.pt")


def detect_and_track(frame_bgr):
    """
    Detect and track persons in a single BGR frame.

    Args:
        frame_bgr: H×W×3 uint8 NumPy array in BGR order (OpenCV convention).

    Returns:
        persons    (list[dict]): one entry per tracked person with keys
                                 {track_id, x1, y1, x2, y2}.
        tailgating (bool):       True when more than one person is in the frame.
    """
    model = _get_model()

    results = model.track(
        frame_bgr,
        persist=True,          # keep ByteTrack state between calls for stable IDs
        classes=[0],           # COCO class 0 = "person"; ignore all other classes
        conf=0.4,              # minimum detection confidence to enter tracking
        iou=0.5,               # IoU threshold for NMS suppression of duplicate boxes
        verbose=False,         # suppress per-frame console output
        tracker="bytetrack.yaml",
    )

    persons = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        # boxes.id is None when ByteTrack hasn't assigned IDs yet (first frame).
        if boxes.id is not None:
            for box, tid in zip(boxes.xyxy.cpu().numpy(), boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                persons.append({
                    "track_id": int(tid),
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                })

    # Flag tailgating whenever more than one distinct person is tracked simultaneously.
    tailgating = len(persons) > 1
    return persons, tailgating
