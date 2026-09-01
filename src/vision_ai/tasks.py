"""
Task definitions and prompt mappings for Florence-2 Vision-Language Model.

Author: Johnny Wilson Dougherty
"""

from enum import Enum
from typing import Dict, Any


class VisionTask(str, Enum):
    """Supported Vision-Language Tasks in Florence-2."""

    OBJECT_DETECTION = "<OD>"
    CAPTION = "<CAPTION>"
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    OCR = "<OCR>"
    OCR_WITH_REGION = "<OCR_WITH_REGION>"
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


TASK_METADATA: Dict[VisionTask, Dict[str, Any]] = {
    VisionTask.OBJECT_DETECTION: {
        "name": "Object Detection",
        "prompt": "<OD>",
        "has_boxes": True,
        "description": "Locates common objects in the image and outputs labels with coordinate bounding boxes [xmin, ymin, xmax, ymax].",
    },
    VisionTask.CAPTION: {
        "name": "General Caption",
        "prompt": "<CAPTION>",
        "has_boxes": False,
        "description": "Generates a concise single-sentence caption describing the whole image scene.",
    },
    VisionTask.DETAILED_CAPTION: {
        "name": "Detailed Caption",
        "prompt": "<DETAILED_CAPTION>",
        "has_boxes": False,
        "description": "Generates an enriched, paragraph-length descriptive caption covering entities and context.",
    },
    VisionTask.MORE_DETAILED_CAPTION: {
        "name": "More Detailed Caption",
        "prompt": "<MORE_DETAILED_CAPTION>",
        "has_boxes": False,
        "description": "Provides an exhaustive, high-fidelity scene analysis including background, lighting, and relationships.",
    },
    VisionTask.OCR: {
        "name": "Optical Character Recognition (OCR)",
        "prompt": "<OCR>",
        "has_boxes": False,
        "description": "Extracts raw text and typographic content present within the image.",
    },
    VisionTask.OCR_WITH_REGION: {
        "name": "OCR with Region Bounding Boxes",
        "prompt": "<OCR_WITH_REGION>",
        "has_boxes": True,
        "description": "Extracts text content along with localized polygon/box regions for each text span.",
    },
    VisionTask.DENSE_REGION_CAPTION: {
        "name": "Dense Region Captioning",
        "prompt": "<DENSE_REGION_CAPTION>",
        "has_boxes": True,
        "description": "Detects specific localized visual sub-regions and generates captions for each region.",
    },
    VisionTask.REGION_PROPOSAL: {
        "name": "Region Proposals",
        "prompt": "<REGION_PROPOSAL>",
        "has_boxes": True,
        "description": "Proposes bounding boxes for prominent objects or visual elements across the image.",
    },
    VisionTask.CAPTION_TO_PHRASE_GROUNDING: {
        "name": "Caption to Phrase Grounding",
        "prompt": "<CAPTION_TO_PHRASE_GROUNDING>",
        "has_boxes": True,
        "description": "Locates and bounds specific text phrases within the visual image (requires text query).",
    },
    VisionTask.OPEN_VOCABULARY_DETECTION: {
        "name": "Open Vocabulary Detection",
        "prompt": "<OPEN_VOCABULARY_DETECTION>",
        "has_boxes": True,
        "description": "Detects specific user-specified open-vocabulary target concepts with bounding boxes.",
    },
}


def get_task_by_name(name_or_prompt: str) -> VisionTask:
    """Resolve a user-friendly name, alias, or prompt tag to a VisionTask enum."""
    clean = name_or_prompt.strip().upper()
    for task in VisionTask:
        if task.value == clean or task.name == clean:
            return task
        if task in TASK_METADATA and TASK_METADATA[task]["name"].upper() == clean:
            return task
    # Check aliases
    aliases = {
        "OD": VisionTask.OBJECT_DETECTION,
        "DETECT": VisionTask.OBJECT_DETECTION,
        "DETECTION": VisionTask.OBJECT_DETECTION,
        "CAPTION": VisionTask.CAPTION,
        "DETAILED": VisionTask.DETAILED_CAPTION,
        "MORE_DETAILED": VisionTask.MORE_DETAILED_CAPTION,
        "OCR": VisionTask.OCR,
        "OCR_REGION": VisionTask.OCR_WITH_REGION,
        "DENSE": VisionTask.DENSE_REGION_CAPTION,
        "PROPOSAL": VisionTask.REGION_PROPOSAL,
        "GROUNDING": VisionTask.CAPTION_TO_PHRASE_GROUNDING,
    }
    if clean in aliases:
        return aliases[clean]
    raise ValueError(f"Unknown Vision Task identifier: '{name_or_prompt}'")

