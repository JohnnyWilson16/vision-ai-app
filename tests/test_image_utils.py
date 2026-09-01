"""
Unit tests for image loading and preprocessing utilities.

Author: Johnny Wilson Dougherty
"""

import io
import pytest
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vision_ai.image_utils import prepare_image_rgb, load_image


def test_prepare_image_rgb_already_rgb():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = prepare_image_rgb(img)
    assert result.mode == "RGB"
    assert result.size == (100, 100)


def test_prepare_image_rgb_from_rgba():
    img = Image.new("RGBA", (120, 80), color=(0, 255, 0, 128))
    result = prepare_image_rgb(img)
    assert result.mode == "RGB"
    assert result.size == (120, 80)


def test_prepare_image_rgb_from_grayscale():
    img = Image.new("L", (50, 50), color=128)
    result = prepare_image_rgb(img)
    assert result.mode == "RGB"
    assert result.size == (50, 50)


def test_load_image_from_bytes():
    img = Image.new("RGB", (64, 64), color=(0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    loaded = load_image(raw_bytes)
    assert isinstance(loaded, Image.Image)
    assert loaded.mode == "RGB"
    assert loaded.size == (64, 64)


def test_load_image_from_local_file(tmp_path):
    img_path = tmp_path / "test_sample.png"
    img = Image.new("RGBA", (80, 60), color=(100, 100, 100, 255))
    img.save(img_path)

    loaded = load_image(img_path)
    assert isinstance(loaded, Image.Image)
    assert loaded.mode == "RGB"
    assert loaded.size == (80, 60)


def test_load_image_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_image("/non/existent/path/image_12345.jpg")


def test_load_file_like_object_with_seek():
    img = Image.new("RGB", (70, 70), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    # Simulate an already-read buffer
    buf.read()
    
    loaded = load_image(buf)
    assert isinstance(loaded, Image.Image)
    assert loaded.size == (70, 70)


def test_load_bundled_samples():
    samples_dir = Path(__file__).parent.parent / "assets" / "samples"
    assert samples_dir.exists()
    for sample_file in samples_dir.glob("*.*"):
        loaded = load_image(sample_file)
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size[0] > 0 and loaded.size[1] > 0

