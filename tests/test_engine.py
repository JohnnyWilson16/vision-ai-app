"""
Unit tests for VisionAIEngine configuration and pipeline initialization.

Author: Johnny Wilson Dougherty
"""

import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision_ai.config import VisionAIConfig, get_default_device, get_default_dtype
from vision_ai.engine import VisionAIEngine
from vision_ai.tasks import VisionTask


def test_config_initialization():
    cfg = VisionAIConfig(model_id="florence-community/Florence-2-base", num_beams=4)
    assert cfg.model_id == "florence-community/Florence-2-base"
    assert cfg.num_beams == 4
    assert cfg.device in ("cuda", "mps", "cpu")
    assert cfg.torch_dtype in (torch.float16, torch.float32)


def test_device_and_dtype_selection():
    dev = get_default_device()
    assert dev in ("cuda", "mps", "cpu")
    dtype = get_default_dtype("cpu")
    assert dtype == torch.float32
    dtype_cuda = get_default_dtype("cuda")
    assert dtype_cuda == torch.float16


def test_engine_mock_run_task():
    cfg = VisionAIConfig(model_id="mock/florence-2", device="cpu")
    engine = VisionAIEngine(cfg)

    # Mock model and processor
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")

    mock_inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "pixel_values": torch.zeros((1, 3, 224, 224)),
    }
    mock_processor.return_value = mock_inputs
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
    mock_processor.batch_decode.return_value = ["<OD>car<loc_10><loc_10><loc_100><loc_100>"]
    mock_processor.post_process_generation.return_value = {
        "<OD>": {"bboxes": [[10, 10, 100, 100]], "labels": ["car"]}
    }

    engine.processor = mock_processor
    engine.model = mock_model
    engine._is_loaded = True

    dummy_img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    res = engine.run_task(dummy_img, task=VisionTask.OBJECT_DETECTION, render_annotation=True)

    assert res["task"] == VisionTask.OBJECT_DETECTION
    assert "<OD>" in res["parsed_answer"]
    assert res["parsed_answer"]["<OD>"]["labels"] == ["car"]
    assert res["annotated_image"] is not None
    assert res["latency_ms"] >= 0
