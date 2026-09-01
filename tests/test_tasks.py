"""
Unit tests for Florence-2 Vision Tasks.

Author: Johnny Wilson Dougherty
"""

import pytest
import sys
from pathlib import Path

# Add src/ to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision_ai.tasks import VisionTask, TASK_METADATA, get_task_by_name


def test_vision_task_enum_values():
    assert VisionTask.OBJECT_DETECTION.value == "<OD>"
    assert VisionTask.CAPTION.value == "<CAPTION>"
    assert VisionTask.DETAILED_CAPTION.value == "<DETAILED_CAPTION>"
    assert VisionTask.MORE_DETAILED_CAPTION.value == "<MORE_DETAILED_CAPTION>"
    assert VisionTask.OCR.value == "<OCR>"
    assert VisionTask.OCR_WITH_REGION.value == "<OCR_WITH_REGION>"
    assert VisionTask.DENSE_REGION_CAPTION.value == "<DENSE_REGION_CAPTION>"
    assert VisionTask.REGION_PROPOSAL.value == "<REGION_PROPOSAL>"


def test_task_metadata_completeness():
    for task in VisionTask:
        if task in TASK_METADATA:
            meta = TASK_METADATA[task]
            assert "name" in meta
            assert "prompt" in meta
            assert "has_boxes" in meta
            assert "description" in meta
            assert meta["prompt"] == task.value


def test_get_task_by_name_resolution():
    assert get_task_by_name("<OD>") == VisionTask.OBJECT_DETECTION
    assert get_task_by_name("OD") == VisionTask.OBJECT_DETECTION
    assert get_task_by_name("detect") == VisionTask.OBJECT_DETECTION
    assert get_task_by_name("Object Detection") == VisionTask.OBJECT_DETECTION
    assert get_task_by_name("CAPTION") == VisionTask.CAPTION
    assert get_task_by_name("ocr") == VisionTask.OCR
    assert get_task_by_name("ocr_region") == VisionTask.OCR_WITH_REGION
    assert get_task_by_name("dense") == VisionTask.DENSE_REGION_CAPTION


def test_get_task_by_name_invalid():
    with pytest.raises(ValueError):
        get_task_by_name("NON_EXISTENT_TASK_123")
