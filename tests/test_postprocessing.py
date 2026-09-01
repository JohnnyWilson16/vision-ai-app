"""
Unit tests for postprocessing and Non-Maximum Suppression (NMS).

Author: Johnny Wilson Dougherty
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision_ai.postprocessing import (
    calculate_iou,
    suppress_overlapping_bboxes,
    filter_prediction_clutter,
)


def test_calculate_iou_disjoint():
    b1 = [0, 0, 10, 10]
    b2 = [20, 20, 30, 30]
    assert calculate_iou(b1, b2) == 0.0


def test_calculate_iou_identical():
    b1 = [10, 10, 50, 50]
    b2 = [10, 10, 50, 50]
    assert pytest.approx(calculate_iou(b1, b2), 0.01) == 1.0


def test_calculate_iou_partial_overlap():
    b1 = [0, 0, 20, 20]  # area = 400
    b2 = [10, 0, 30, 20] # area = 400, intersection = 10*20 = 200, union = 600, iou = 0.333
    assert pytest.approx(calculate_iou(b1, b2), 0.01) == 0.333


def test_suppress_overlapping_bboxes():
    # 3 duplicate overlapping boxes and 1 separate box
    bboxes = [
        [100, 100, 300, 300],
        [105, 105, 305, 305], # heavy overlap with box 0
        [102, 98, 298, 302],  # heavy overlap with box 0
        [400, 400, 500, 500], # distinct separate box
    ]
    labels = ["obj1", "obj1_dup", "obj1_dup2", "obj2"]

    kept_b, kept_l = suppress_overlapping_bboxes(
        bboxes=bboxes,
        labels=labels,
        iou_threshold=0.5,
        max_boxes=10,
    )
    assert len(kept_b) == 2
    assert len(kept_l) == 2


def test_filter_prediction_clutter_capping():
    bboxes = [[i * 10, i * 10, i * 10 + 50, i * 10 + 50] for i in range(25)]
    labels = [f"item_{i}" for i in range(25)]

    raw_pred = {"<REGION_PROPOSAL>": {"bboxes": bboxes, "labels": labels}}
    filtered = filter_prediction_clutter(raw_pred, max_boxes=8, iou_threshold=0.5)

    assert len(filtered["<REGION_PROPOSAL>"]["bboxes"]) <= 8
