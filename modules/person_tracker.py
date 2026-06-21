"""
Person Detection Module
=======================
Uses YOLOv8n (nano variant) for real-time person detection.
Tailgating is flagged when more than one person appears in the same frame —
a simple but effective heuristic for an access-control entry point where only
one person should pass at a time.
"""

from ultralytics import YOLO

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def reset_tracker():
    """No-op kept for API compatibility — no tracker state to reset."""
    pass


def detect_and_track(frame_bgr):
    """
    Detect persons in a single BGR frame using YOLOv8n.

    Returns:
        persons    (list[dict]): one entry per detected person with keys
                                 {x1, y1, x2, y2}.
        tailgating (bool):       True when more than one person is in the frame.
    """
    model = _get_model()

    results = model(
        frame_bgr,
        classes=[0],     # COCO class 0 = "person"
        conf=0.4,        # minimum detection confidence
        iou=0.5,         # NMS IoU threshold
        verbose=False,
    )

    persons = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            persons.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    tailgating = len(persons) > 1
    return persons, tailgating
