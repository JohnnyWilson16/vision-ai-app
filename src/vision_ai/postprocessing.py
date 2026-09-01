"""
Post-processing, Non-Maximum Suppression (NMS), and Box Filtering utilities.

Cleans up cluttered, overlapping candidate bounding boxes from raw Vision-Language Models.

Author: Johnny Wilson Dougherty
"""

from typing import List, Tuple, Dict, Any, Optional


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union_area = areaA + areaB - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def suppress_overlapping_bboxes(
    bboxes: List[List[float]],
    labels: Optional[List[str]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    iou_threshold: float = 0.55,
    min_area_ratio: float = 0.005,
    max_boxes: int = 12,
) -> Tuple[List[List[float]], List[str]]:
    """
    Apply Non-Maximum Suppression and area filtering to candidate bounding boxes.

    Parameters:
        bboxes: List of [xmin, ymin, xmax, ymax]
        labels: Optional list of labels
        image_size: Optional (width, height) of image
        iou_threshold: Max allowed overlap before suppression (0.1 = strict, 0.9 = loose)
        min_area_ratio: Minimum box area relative to image (0.005 = 0.5% minimum area)
        max_boxes: Maximum number of top regions to keep

    Returns:
        Filtered (bboxes, labels)
    """
    if not bboxes:
        return [], []

    clean_labels = list(labels) if labels else ["" for _ in bboxes]
    while len(clean_labels) < len(bboxes):
        clean_labels.append("")

    total_img_area = (image_size[0] * image_size[1]) if image_size and image_size[0] and image_size[1] else None

    # Filter invalid and microscopic noise boxes
    valid_candidates = []
    for idx, box in enumerate(bboxes):
        if len(box) == 4 and all(v is not None for v in box):
            w = max(0.0, box[2] - box[0])
            h = max(0.0, box[3] - box[1])
            area = w * h
            
            # Skip degenerated or tiny 4x4 pixel noise
            if w < 10 or h < 10 or area < 100:
                continue

            # Skip sub-threshold tiny noise
            if total_img_area and (area / total_img_area) < min_area_ratio:
                continue

            valid_candidates.append((box, clean_labels[idx], area))

    if not valid_candidates:
        return [], []

    # Sort by region saliency/area (largest most prominent regions first)
    valid_candidates.sort(key=lambda item: item[2], reverse=True)

    kept_boxes: List[List[float]] = []
    kept_labels: List[str] = []

    for box, label, _ in valid_candidates:
        overlap = False
        for kept in kept_boxes:
            if calculate_iou(box, kept) > iou_threshold:
                overlap = True
                break

        if not overlap:
            kept_boxes.append(box)
            kept_labels.append(label)
            if len(kept_boxes) >= max_boxes:
                break

    return kept_boxes, kept_labels


def filter_prediction_clutter(
    prediction: Dict[str, Any],
    image_size: Optional[Tuple[int, int]] = None,
    iou_threshold: float = 0.55,
    min_area_ratio: float = 0.005,
    max_boxes: int = 12,
) -> Dict[str, Any]:
    """
    Clean up a Florence-2 prediction dictionary by suppressing overlapping boxes.
    """
    cleaned = {}
    for task_key, data in prediction.items():
        if isinstance(data, dict) and "bboxes" in data:
            bboxes = data.get("bboxes", [])
            labels = data.get("labels", [])
            
            filtered_boxes, filtered_labels = suppress_overlapping_bboxes(
                bboxes=bboxes,
                labels=labels,
                image_size=image_size,
                iou_threshold=iou_threshold,
                min_area_ratio=min_area_ratio,
                max_boxes=max_boxes,
            )
            
            cleaned[task_key] = {
                **data,
                "bboxes": filtered_boxes,
                "labels": filtered_labels,
            }
        else:
            cleaned[task_key] = data

    return cleaned
