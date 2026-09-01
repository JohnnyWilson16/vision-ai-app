"""
Unit tests for visualizer annotations and result export functions.

Author: Johnny Wilson Dougherty
"""

import json
import pytest
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision_ai.visualizer import annotate_image
from vision_ai.exporter import export_results_json, format_detection_records


def test_annotate_image_with_bboxes():
    img = Image.new("RGB", (300, 300), color=(200, 200, 200))
    mock_pred = {
        "<OD>": {
            "bboxes": [
                [10, 10, 100, 100],
                [120, 50, 250, 200],
            ],
            "labels": ["car", "wheel"],
        }
    }

    annotated = annotate_image(img, mock_pred)
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (300, 300)
    assert annotated.mode == "RGB"


def test_format_detection_records():
    mock_pred = {
        "<OD>": {
            "bboxes": [[20, 30, 80, 110]],
            "labels": ["person"],
        }
    }
    records = format_detection_records(mock_pred, image_size=(200, 200))
    assert len(records) == 1
    assert records[0]["label"] == "person"
    assert records[0]["xmin"] == 20
    assert records[0]["ymin"] == 30
    assert records[0]["xmax"] == 80
    assert records[0]["ymax"] == 110
    assert records[0]["width"] == 60
    assert records[0]["height"] == 80
    assert records[0]["area"] == 4800


def test_export_results_json(tmp_path):
    mock_pred = {
        "<CAPTION>": "a modern red car parked in a driveway"
    }
    out_file = tmp_path / "report.json"
    json_str = export_results_json(
        prediction=mock_pred,
        task_name="General Caption",
        image_source="test_car.jpg",
        image_size=(640, 480),
        output_path=out_file,
    )

    data = json.loads(json_str)
    assert data["metadata"]["author"] == "Johnny Wilson Dougherty"
    assert data["metadata"]["task"] == "General Caption"
    assert data["metadata"]["image_size"] == {"width": 640, "height": 480}
    assert out_file.exists()
