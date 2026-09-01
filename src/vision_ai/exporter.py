"""
Export and serialization utilities for Vision AI structured outputs.

Author: Johnny Wilson Dougherty
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from PIL import Image


def format_detection_records(prediction: Dict[str, Any], image_size: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """
    Flatten prediction bounding boxes, region proposals, and OCR quad-boxes into tabular records.
    """
    records = []
    for task_key, data in prediction.items():
        if not isinstance(data, dict):
            continue

        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])
        quad_boxes = data.get("quad_boxes", []) or data.get("polygons", [])

        is_proposal = task_key in ("<REGION_PROPOSAL>", "REGION_PROPOSAL")

        # 1. Standard 4-point bounding boxes (Object Detection, Dense Captions, Region Proposals)
        if bboxes:
            for idx, box in enumerate(bboxes):
                raw_label = labels[idx] if idx < len(labels) else ""
                if not raw_label or not str(raw_label).strip():
                    label = f"Region Proposal {idx+1}" if is_proposal else f"Object {idx+1}"
                else:
                    label = str(raw_label).strip()

                rec = {
                    "index": idx + 1,
                    "label": label,
                    "xmin": round(box[0], 1) if len(box) > 0 and box[0] is not None else None,
                    "ymin": round(box[1], 1) if len(box) > 1 and box[1] is not None else None,
                    "xmax": round(box[2], 1) if len(box) > 2 and box[2] is not None else None,
                    "ymax": round(box[3], 1) if len(box) > 3 and box[3] is not None else None,
                }
                if len(box) == 4 and all(v is not None for v in box):
                    rec["width"] = round(box[2] - box[0], 1)
                    rec["height"] = round(box[3] - box[1], 1)
                    rec["area"] = round(rec["width"] * rec["height"], 1)
                records.append(rec)

        # 2. Quad / polygon boxes (e.g. from OCR with Region)
        elif quad_boxes:
            for idx, quad in enumerate(quad_boxes):
                raw_label = labels[idx] if idx < len(labels) else ""
                label = str(raw_label).strip() if raw_label else f"Text Line {idx+1}"

                # Handle [x1, y1, x2, y2, x3, y3, x4, y4] or list of (x, y) pairs
                if len(quad) == 8:
                    xs = [quad[0], quad[2], quad[4], quad[6]]
                    ys = [quad[1], quad[3], quad[5], quad[7]]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                elif isinstance(quad[0], (list, tuple)):
                    xs = [p[0] for p in quad]
                    ys = [p[1] for p in quad]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                else:
                    xmin, ymin, xmax, ymax = None, None, None, None

                rec = {
                    "index": idx + 1,
                    "label": label,
                    "xmin": round(xmin, 1) if xmin is not None else None,
                    "ymin": round(ymin, 1) if ymin is not None else None,
                    "xmax": round(xmax, 1) if xmax is not None else None,
                    "ymax": round(ymax, 1) if ymax is not None else None,
                }
                if xmin is not None and ymin is not None and xmax is not None and ymax is not None:
                    rec["width"] = round(xmax - xmin, 1)
                    rec["height"] = round(ymax - ymin, 1)
                    rec["area"] = round(rec["width"] * rec["height"], 1)
                records.append(rec)

    return records


def export_results_json(
    prediction: Dict[str, Any],
    task_name: str,
    image_source: str = "",
    image_size: Optional[tuple] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Format and optionally save Vision AI results as a standard JSON report.
    """
    payload = {
        "metadata": {
            "app": "Vision AI Application (Florence-2)",
            "author": "Johnny Wilson Dougherty",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task_name,
            "source": str(image_source),
            "image_size": {
                "width": image_size[0] if image_size else None,
                "height": image_size[1] if image_size else None,
            } if image_size else None,
        },
        "raw_prediction": prediction,
        "entities": format_detection_records(prediction, image_size=image_size),
    }

    json_str = json.dumps(payload, indent=2)

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json_str, encoding="utf-8")

    return json_str
