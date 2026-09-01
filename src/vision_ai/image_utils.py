"""
Image processing and ingestion utilities.

Handles safe RGB conversion, remote URL fetching, local file loading,
automatic HTML og:image extraction, and EXIF orientation normalization.

Author: Johnny Wilson Dougherty
"""

import io
import re
import urllib.parse
from pathlib import Path
from typing import Union, BinaryIO, Optional
from PIL import Image, ImageOps, UnidentifiedImageError
import requests


def prepare_image_rgb(image: Image.Image) -> Image.Image:
    """
    Ensure the image is in RGB format and correctly oriented.
    
    Prevents shape mismatch errors in Vision-Language Models caused by
    transparency alpha channels (RGBA) or grayscale channels (L/LA).
    """
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    if image.mode == "RGB":
        return image

    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        alpha_image = image.convert("RGBA")
        background = Image.new("RGB", alpha_image.size, (255, 255, 255))
        background.paste(alpha_image, mask=alpha_image.split()[3])
        return background

    return image.convert("RGB")


def normalize_image_url(url: str) -> str:
    """Rewrite common image hosting URLs to direct raw asset endpoints."""
    clean_url = url.strip()
    
    # GitHub blob URL -> raw content URL
    github_match = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$", clean_url)
    if github_match:
        user, repo, branch, file_path = github_match.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"

    # Dropbox share link -> direct download
    if "dropbox.com" in clean_url and "dl=0" in clean_url:
        return clean_url.replace("dl=0", "raw=1")

    # Google Drive view link -> direct download stream
    gdrive_match = re.search(r"drive\.google\.com/file/d/([^/&?]+)", clean_url)
    if gdrive_match:
        file_id = gdrive_match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    return clean_url


def extract_image_url_from_html(html_text: str, base_url: str) -> Optional[str]:
    """Inspect an HTML document for OpenGraph, Twitter card, or prominent image tags."""
    # 1. OpenGraph Image tag
    og_patterns = [
        r'<meta\s+[^>]*property=["\']og:image["\']\s+[^>]*content=["\']([^"\']+)["\']',
        r'<meta\s+[^>]*content=["\']([^"\']+)["\']\s+[^>]*property=["\']og:image["\']',
    ]
    for pattern in og_patterns:
        match = re.search(pattern, html_text, re.IGNORECASE)
        if match:
            return urllib.parse.urljoin(base_url, match.group(1))

    # 2. Twitter Image tag
    tw_patterns = [
        r'<meta\s+[^>]*name=["\']twitter:image["\']\s+[^>]*content=["\']([^"\']+)["\']',
        r'<meta\s+[^>]*content=["\']([^"\']+)["\']\s+[^>]*name=["\']twitter:image["\']',
    ]
    for pattern in tw_patterns:
        match = re.search(pattern, html_text, re.IGNORECASE)
        if match:
            return urllib.parse.urljoin(base_url, match.group(1))

    # 3. First prominent <img> tag pointing to common image extensions
    img_matches = re.findall(r'<img\s+[^>]*src=["\']([^"\']+\.(?:png|jpg|jpeg|webp)(?:\?[^"\']*)?)["\']', html_text, re.IGNORECASE)
    for candidate in img_matches:
        cand_lower = candidate.lower()
        if not candidate.startswith("data:") and "icon" not in cand_lower and "avatar" not in cand_lower and "logo" not in cand_lower:
            return urllib.parse.urljoin(base_url, candidate)

    return None


def fetch_image_from_url(url: str, max_redirects: int = 2) -> Image.Image:
    """Download an image from a URL, with automatic HTML preview resolution."""
    normalized_url = normalize_image_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    try:
        response = requests.get(normalized_url, headers=headers, stream=True, timeout=20)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch remote URL: {e}")

    content = response.content
    if len(content) == 0:
        raise ValueError(f"Remote URL returned an empty response.")

    # Attempt direct PIL decode
    try:
        img = Image.open(io.BytesIO(content))
        return prepare_image_rgb(img)
    except UnidentifiedImageError:
        # Check if the response was an HTML page with an embedded image
        content_type = response.headers.get("Content-Type", "").lower()
        if ("text/html" in content_type or b"<html" in content[:200].lower()) and max_redirects > 0:
            try:
                html_text = content.decode("utf-8", errors="ignore")
                extracted_url = extract_image_url_from_html(html_text, base_url=normalized_url)
                if extracted_url and extracted_url != normalized_url:
                    return fetch_image_from_url(extracted_url, max_redirects=max_redirects - 1)
            except Exception:
                pass

        raise ValueError(
            f"The provided URL returned a webpage instead of an image file (Content-Type: '{content_type}'). "
            "Please provide a direct link to an image file ending in .jpg, .png, or .webp."
        )


def load_image(
    source: Union[str, Path, bytes, bytearray, BinaryIO, Image.Image]
) -> Image.Image:
    """
    Load an image from various source types and return a verified RGB PIL Image.
    
    Supports:
    - Web URLs (direct image links or webpages with og:image metadata)
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
            return fetch_image_from_url(source_str)
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
