"""
Person Detection and Segmentation Module
=========================================
Uses YOLOv8n-seg (nano segmentation variant) for real-time person detection
with instance segmentation masks.  Segmentation is an additional computer
vision capability on top of detection: the model produces a per-pixel mask
for each detected person rather than just a bounding box.

Tailgating is flagged when more than one person appears in the same frame —
a simple but effective heuristic for an access-control entry point where only
one person should pass at a time.
"""

import numpy as np
from ultralytics import YOLO

_model = None


def _get_model():
    global _model
    if _model is None:
        # yolov8n-seg adds instance segmentation on top of detection;
        # auto-downloaded by Ultralytics on first use (same as yolov8n.pt).
        _model = YOLO("yolov8n-seg.pt")
    return _model


def reset_tracker():
    """No-op kept for API compatibility — no tracker state to reset."""
    pass


def detect_and_track(frame_bgr):
    """
    Detect and segment persons in a single BGR frame using YOLOv8n-seg.

    Returns:
        persons    (list[dict]): one entry per detected person with keys
                                 {x1, y1, x2, y2} and optionally
                                 {mask_poly} — an (N, 2) int32 array of
                                 polygon vertices in image coordinates.
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
        seg_masks = results[0].masks   # None when no detections
        for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            person = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            # Attach polygon mask when available (xy gives coords in original
            # image space — no manual resize needed).
            if seg_masks is not None and i < len(seg_masks.xy):
                poly = seg_masks.xy[i]
                if len(poly) > 0:
                    person["mask_poly"] = poly.astype(np.int32)
            persons.append(person)

    tailgating = len(persons) > 1
    return persons, tailgating
