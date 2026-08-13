"""Geometry helpers for image detections.

Detection dictionaries in this project use OpenCV's image-shape convention:
``size`` is always ``(height, width)``.
"""


def detection_to_box(detection):
    """Convert a detection to an ``(x1, y1, x2, y2)`` box."""
    x, y = detection["position"]
    height, width = detection["size"]
    return (x, y, x + width, y + height)


def detection_center(detection):
    """Return the center point of a detection."""
    x1, y1, x2, y2 = detection_to_box(detection)
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def intersection_area(box1, box2):
    """Return the intersection area of two ``(x1, y1, x2, y2)`` boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def get_iou(box1, box2):
    """Calculate intersection-over-union for two boxes."""
    inter_area = intersection_area(box1, box2)
    if inter_area == 0:
        return 0.0

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def nms(detections, iou_threshold=0.3):
    """Suppress overlapping SQDIFF detections.

    OpenCV ``TM_SQDIFF_NORMED`` scores are distances, so lower scores are
    better. The best (lowest-score) detection is retained first.
    """
    remaining = sorted(detections, key=lambda item: item["score"])
    kept = []

    while remaining:
        best = remaining.pop(0)
        best_box = detection_to_box(best)
        kept.append(best)
        remaining = [
            candidate
            for candidate in remaining
            if get_iou(best_box, detection_to_box(candidate)) < iou_threshold
        ]

    return kept


def suppress_nearby_same_class(detections, center_distance=18):
    """Keep the best SQDIFF hit for one nearby same-class sprite.

    Different animation templates can describe the same monster with boxes
    whose sizes differ enough that IoU-based NMS keeps both.  Their centers,
    however, remain close.  Suppress only same-class centers so adjacent
    monsters of different classes are never merged.
    """
    distance = max(0, float(center_distance))
    distance_sq = distance * distance
    kept = []

    for candidate in sorted(detections, key=lambda item: item["score"]):
        candidate_center = detection_center(candidate)
        is_duplicate = any(
            candidate.get("name") == existing.get("name")
            and (
                (candidate_center[0] - detection_center(existing)[0]) ** 2
                + (candidate_center[1] - detection_center(existing)[1]) ** 2
            ) <= distance_sq
            for existing in kept
        )
        if not is_duplicate:
            kept.append(candidate)

    return kept
