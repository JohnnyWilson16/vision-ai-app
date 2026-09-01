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
    """Flatten prediction bounding boxes and labels into tabular records."""
    records = []
    for task_key, data in prediction.items():
        if not isinstance(data, dict):
            continue
        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])

        for idx, box in enumerate(bboxes):
            label = labels[idx] if idx < len(labels) else f"item_{idx+1}"
            rec = {
                "index": idx + 1,
                "label": label,
                "xmin": box[0] if len(box) > 0 else None,
                "ymin": box[1] if len(box) > 1 else None,
                "xmax": box[2] if len(box) > 2 else None,
                "ymax": box[3] if len(box) > 3 else None,
            }
            if len(box) == 4:
                rec["width"] = box[2] - box[0]
                rec["height"] = box[3] - box[1]
                rec["area"] = rec["width"] * rec["height"]
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
