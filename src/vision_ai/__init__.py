"""
Vision AI Application
=====================
A unified Vision-Language Model framework powered by Microsoft's Florence-2.

Author: Johnny Wilson Dougherty
GitHub: https://github.com/JohnnyWilson16
License: MIT
"""

from .config import VisionAIConfig
from .engine import VisionAIEngine
from .tasks import VisionTask
from .image_utils import load_image, prepare_image_rgb
from .visualizer import annotate_image
from .exporter import export_results_json

__all__ = [
    "VisionAIConfig",
    "VisionAIEngine",
    "VisionTask",
    "load_image",
    "prepare_image_rgb",
    "annotate_image",
    "export_results_json",
]
__version__ = "0.1.0"
__author__ = "Johnny Wilson Dougherty"
