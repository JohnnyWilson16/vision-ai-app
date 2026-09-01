"""
Configuration module for Vision AI Engine.

Author: Johnny Wilson Dougherty
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import torch


def get_default_device() -> str:
    """Determine the optimal hardware accelerator for PyTorch inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_dtype(device: str) -> torch.dtype:
    """Select the optimal floating-point precision based on hardware device."""
    if device in ("cuda", "mps"):
        return torch.float16
    return torch.float32


@dataclass
class VisionAIConfig:
    """Configuration settings for Florence-2 Vision AI model and inference."""

    model_id: str = "florence-community/Florence-2-base"
    device: str = field(default_factory=get_default_device)
    torch_dtype: Optional[torch.dtype] = None
    num_beams: int = 3
    max_new_tokens: int = 1024
    early_stopping: bool = False
    cache_dir: Optional[str] = field(
        default_factory=lambda: os.environ.get("TRANSFORMERS_CACHE")
    )
    trust_remote_code: bool = True

    def __post_init__(self):
        if self.torch_dtype is None:
            self.torch_dtype = get_default_dtype(self.device)
