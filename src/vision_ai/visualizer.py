"""
Visualization utilities for Florence-2 Vision AI outputs.

Renders high-contrast bounding boxes, labeled badges, and polygon regions
on images using PIL and Matplotlib.

Author: Johnny Wilson Dougherty
"""

import io
from typing import Dict, Any, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Distinct, high-contrast palette for multi-class entity visualization
DISTINCT_COLORS = [
    "#FF3B30",  # Red
    "#34C759",  # Green
    "#007AFF",  # Blue
    "#AF52DE",  # Purple
    "#FF9500",  # Orange
    "#5856D6",  # Indigo
    "#00C7BE",  # Teal
    "#FF2D55",  # Pink
    "#FFCC00",  # Yellow
    "#30B0C7",  # Cyan
    "#A2845E",  # Brown
    "#8E8E93",  # Gray
]


def _get_color_for_label(label: str, color_map: Dict[str, str]) -> str:
    """Return a consistent HEX color for a given label string."""
    if label not in color_map:
        idx = len(color_map) % len(DISTINCT_COLORS)
        color_map[label] = DISTINCT_COLORS[idx]
    return color_map[label]


def annotate_image(
    image: Image.Image,
    prediction: Dict[str, Any],
    line_width: int = 3,
    font_size: int = 14,
    show_labels: bool = True,
) -> Image.Image:
    """
    Draw bounding boxes and labels directly onto a copy of the input PIL Image.

    Supports:
    - Standard Florence-2 Object Detection: {"<OD>": {"bboxes": [...], "labels": [...]}}
    - Dense Region Captions: {"<DENSE_REGION_CAPTION>": {"bboxes": [...], "labels": [...]}}
    - Region Proposals: {"<REGION_PROPOSAL>": {"bboxes": [...], "labels": [...]}}
    - OCR with Region: {"<OCR_WITH_REGION>": {"quad_boxes": [...], "labels": [...]}}
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    color_map: Dict[str, str] = {}

    # Attempt to load a default TrueType font or fallback to default
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            font = ImageFont.load_default()

    # Iterate through keys in the prediction dictionary
    for task_key, data in prediction.items():
        if not isinstance(data, dict):
            continue

        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])
        quad_boxes = data.get("quad_boxes", []) or data.get("polygons", [])
        is_proposal = task_key in ("<REGION_PROPOSAL>", "REGION_PROPOSAL")

        # Handle 4-point bounding boxes [xmin, ymin, xmax, ymax]
        if bboxes:
            for idx, box in enumerate(bboxes):
                if len(box) == 4 and all(v is not None for v in box):
                    xmin, ymin, xmax, ymax = box
                    raw_label = labels[idx] if idx < len(labels) else ""
                    if not raw_label or not str(raw_label).strip():
                        label = f"Region {idx+1}" if is_proposal else f"Object {idx+1}"
                    else:
                        label = str(raw_label).strip()

                    color = _get_color_for_label(label, color_map)

                    # Draw rectangle outline
                    draw.rectangle(
                        [(xmin, ymin), (xmax, ymax)],
                        outline=color,
                        width=line_width,
                    )

                    if show_labels and label:
                        # Compute text bounding box for label badge
                        bbox_text = draw.textbbox((xmin, ymin), label, font=font)
                        text_w = bbox_text[2] - bbox_text[0] + 8
                        text_h = bbox_text[3] - bbox_text[1] + 6

                        badge_y0 = max(0, ymin - text_h)
                        badge_y1 = badge_y0 + text_h
                        badge_x1 = xmin + text_w

                        # Draw background pill/rectangle
                        draw.rectangle(
                            [(xmin, badge_y0), (badge_x1, badge_y1)],
                            fill=color,
                        )
                        # Draw label text in white
                        draw.text(
                            (xmin + 4, badge_y0 + 2),
                            label,
                            fill="#FFFFFF",
                            font=font,
                        )

        # Handle quad / polygon boxes (OCR with Region)
        if quad_boxes:
            for idx, quad in enumerate(quad_boxes):
                raw_label = labels[idx] if idx < len(labels) else ""
                label = str(raw_label).strip() if raw_label else f"Text {idx+1}"
                color = _get_color_for_label(label, color_map)
                
                # quad might be [x1, y1, x2, y2, x3, y3, x4, y4] or list of pairs
                points: List[Tuple[float, float]] = []
                if len(quad) == 8:
                    points = [(quad[i], quad[i+1]) for i in range(0, 8, 2)]
                elif isinstance(quad[0], (list, tuple)):
                    points = [(p[0], p[1]) for p in quad]

                if points:
                    draw.polygon(points, outline=color, width=line_width)
                    if show_labels and label:
                        first_pt = points[0]
                        draw.text((first_pt[0], max(0, first_pt[1] - 14)), label, fill=color, font=font)

    return annotated


def plot_annotated_matplotlib(
    image: Image.Image,
    prediction: Dict[str, Any],
    figsize: Tuple[int, int] = (10, 8),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Render predictions using Matplotlib for rich notebook visualization.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")

    color_map: Dict[str, str] = {}

    for task_key, data in prediction.items():
        if not isinstance(data, dict):
            continue

        bboxes = data.get("bboxes", [])
        labels = data.get("labels", [])
        is_proposal = task_key in ("<REGION_PROPOSAL>", "REGION_PROPOSAL")

        for idx, box in enumerate(bboxes):
            if len(box) == 4:
                xmin, ymin, xmax, ymax = box
                w = xmax - xmin
                h = ymax - ymin
                raw_label = labels[idx] if idx < len(labels) else ""
                if not raw_label or not str(raw_label).strip():
                    label = f"Region {idx+1}" if is_proposal else f"Entity {idx+1}"
                else:
                    label = str(raw_label).strip()

                color = _get_color_for_label(label, color_map)

                rect = patches.Rectangle(
                    (xmin, ymin),
                    w,
                    h,
                    linewidth=2.5,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    xmin,
                    max(0, ymin - 5),
                    label,
                    color="white",
                    fontsize=10,
                    weight="bold",
                    bbox=dict(facecolor=color, alpha=0.85, edgecolor="none", pad=2),
                )

    if title:
        plt.title(title, fontsize=14, weight="bold")
    plt.tight_layout()
    return fig
