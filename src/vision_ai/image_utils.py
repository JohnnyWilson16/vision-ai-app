"""
Image processing and ingestion utilities.

Handles safe RGB conversion, remote URL fetching, local file loading,
and EXIF orientation normalization.

Author: Johnny Wilson Dougherty
"""

import io
from pathlib import Path
from typing import Union, BinaryIO
from PIL import Image, ImageOps, UnidentifiedImageError
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
    source: Union[str, Path, bytes, bytearray, BinaryIO, Image.Image]
) -> Image.Image:
    """
    Load an image from various source types and return a verified RGB PIL Image.
    
    Supports:
    - Web URLs (http:// or https://)
    - Local filesystem paths (str or Path)
    - In-memory bytes or file-like streams (e.g., Streamlit UploadedFile)
    - Existing PIL Image instances
    """
    if isinstance(source, Image.Image):
        return prepare_image_rgb(source)

    # In-memory raw bytes
    if isinstance(source, (bytes, bytearray)):
        if len(source) == 0:
            raise ValueError("Provided image byte buffer is empty.")
        try:
            img = Image.open(io.BytesIO(source))
            return prepare_image_rgb(img)
        except UnidentifiedImageError as e:
            raise ValueError(f"Cannot identify image from provided byte stream: {e}")

    # File-like object (e.g., Streamlit UploadedFile, io.BytesIO)
    if hasattr(source, "read") or hasattr(source, "getvalue"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            if hasattr(source, "getvalue"):
                data = source.getvalue()
            else:
                data = source.read()
            
            if not data:
                raise ValueError("Uploaded file contains zero bytes.")
            img = Image.open(io.BytesIO(data))
            return prepare_image_rgb(img)
        except UnidentifiedImageError as e:
            raise ValueError(f"Uploaded file is not a valid image format (PNG, JPG, WEBP expected): {e}")

    # File path or remote URL string
    if isinstance(source, (str, Path)):
        source_str = str(source).strip()
        if source_str.startswith(("http://", "https://")):
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            try:
                response = requests.get(source_str, headers=headers, stream=True, timeout=20)
                response.raise_for_status()
            except requests.RequestException as e:
                raise ValueError(f"Failed to fetch remote image from '{source_str}': {e}")

            content = response.content
            if len(content) == 0:
                raise ValueError(f"Remote URL '{source_str}' returned an empty response.")

            try:
                img = Image.open(io.BytesIO(content))
                return prepare_image_rgb(img)
            except UnidentifiedImageError:
                content_type = response.headers.get("Content-Type", "unknown")
                raise ValueError(
                    f"URL returned non-image content (Content-Type: '{content_type}'). "
                    "Please ensure the link points directly to an image file (PNG, JPG, JPEG, WEBP)."
                )
        else:
            file_path = Path(source_str)
            if not file_path.exists():
                raise FileNotFoundError(f"Local image file not found at path: {file_path}")
            try:
                img = Image.open(file_path)
                return prepare_image_rgb(img)
            except UnidentifiedImageError as e:
                raise ValueError(f"File at '{file_path}' is not a valid image: {e}")

    raise ValueError(f"Unsupported image source type: {type(source)}")
