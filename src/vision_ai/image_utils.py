"""
Image processing and ingestion utilities.

Handles safe RGB conversion, remote URL fetching, local file loading,
and EXIF orientation normalization.

Author: Johnny Wilson Dougherty
"""

import io
from pathlib import Path
from typing import Union, BinaryIO
from PIL import Image, ImageOps
import requests


def prepare_image_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure the image is in RGB format and correctly oriented.
    
    Prevents shape mismatch errors in Vision-Language Models caused by
    transparency alpha channels (RGBA) or grayscale channels (L/LA).
    """
    # Auto-rotate based on EXIF tag if present
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if image.mode == "RGB":
        return image

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        # Composite transparent image against a solid white background
        alpha_image = image.convert("RGBA")
        background = Image.new("RGB", alpha_image.size, (255, 255, 255))
        background.paste(alpha_image, mask=alpha_image.split()[3])
        return background

    return image.convert("RGB")


def load_image(
    source: Union[str, Path, bytes, BinaryIO, Image.Image]
) -> Image.Image:
    """
    Load an image from various source types and return a verified RGB PIL Image.
    
    Supports:
    - Web URLs (http:// or https://)
    - Local filesystem paths (str or Path)
    - In-memory bytes or file-like streams
    - Existing PIL Image instances
    """
    if isinstance(source, Image.Image):
        return prepare_image_rgb(source)

    if isinstance(source, (bytes, bytearray)):
        stream = io.BytesIO(source)
        img = Image.open(stream)
        return prepare_image_rgb(img)

    if hasattr(source, "read"):  # File-like object (e.g., Streamlit UploadedFile)
        img = Image.open(source)
        return prepare_image_rgb(img)

    if isinstance(source, (str, Path)):
        source_str = str(source).strip()
        if source_str.startswith(("http://", "https://")):
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(source_str, headers=headers, stream=True, timeout=20)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            return prepare_image_rgb(img)
        else:
            file_path = Path(source_str)
            if not file_path.exists():
                raise FileNotFoundError(f"Image file not found at path: {file_path}")
            img = Image.open(file_path)
            return prepare_image_rgb(img)

    raise ValueError(f"Unsupported image source type: {type(source)}")
