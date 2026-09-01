# Florence-2 Vision AI

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=JohnnyWilson16/vision-ai-app&branch=main&mainModule=app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)

An open-source Python framework and interactive workspace for multi-task computer vision using Microsoft's **Florence-2** Vision-Language Model (VLM).

Florence-2 treats diverse vision tasks—including object detection, optical character recognition (OCR), dense region captioning, and multi-level scene descriptions—as a unified sequence-to-sequence problem. This project provides a production-grade inference engine, a command-line interface, a Streamlit evaluation dashboard, and an automated test suite.

---


## Why Florence-2?

Most computer vision applications chain together multiple fragmented models: a YOLO or Faster-RCNN model for bounding boxes, an OCR engine like Tesseract or PaddleOCR for text, and a multimodal model or CNN for classification.

Florence-2 unifies these tasks within a single architecture:

1. **DaViT Vision Backbone**: Converts image patches into hierarchical multi-scale visual tokens.
2. **Task Prompt Conditioning**: Text prompts (e.g., `<OD>`, `<OCR>`, `<DENSE_REGION_CAPTION>`) instruct the model on the requested output representation.
3. **Sequence-to-Sequence Transformer**: Jointly decodes text and discrete spatial location tokens (`<loc_0>` to `<loc_999>`), which are mapped back to pixel coordinates.

This design enables local zero-shot image analysis without relying on proprietary, paid cloud vision APIs.

---

## Key Capabilities

- **Unified Multi-Task Pipeline**: Run Object Detection, OCR with region polygons, Dense Region Captioning, Region Proposals, and multi-granularity scene captions through a single model instance.
- **Hardware-Aware Acceleration**: Automatic device placement supporting Apple Silicon (`mps`), NVIDIA CUDA (`cuda`), and CPU fallback with float16 / float32 precision management.
- **Safe Image Ingestion**: Automated handling of image alpha channels (RGBA, LA, palette transparency) to prevent tensor shape mismatches, paired with EXIF orientation auto-rotation.
- **Structured Outputs & Visualization**: Formats raw coordinate tokens into structured JSON, tabular data, and high-contrast bounding box overlays.
- **Multiple Interfaces**: Command-line tool (`run_vision_ai.py`), interactive web workspace (`app.py`), and a step-by-step Jupyter Notebook.

---

## Supported Florence-2 Tasks

| Task Prompt | Mode | Description | Output Structure |
|---|---|---|---|
| `<OD>` | Object Detection | Detects common objects with localized bounding boxes | Boxes `[xmin, ymin, xmax, ymax]` + Labels |
| `<DENSE_REGION_CAPTION>` | Dense Region Captioning | Identifies localized sub-regions with descriptive phrases | Regional Bounding Boxes + Text Labels |
| `<OCR>` | Plain Text OCR | Extracts text present within documents, signage, or images | Plain text string |
| `<OCR_WITH_REGION>` | Localized OCR | Extracts text alongside spatial coordinate polygons | Text strings + Quad polygon coordinates |
| `<CAPTION>` | Concise Caption | Generates a single-sentence high-level summary | Short text description |
| `<DETAILED_CAPTION>` | Detailed Caption | Generates an expanded descriptive paragraph | Medium text description |
| `<MORE_DETAILED_CAPTION>` | Deep Scene Analysis | Provides an exhaustive breakdown of entities and context | Comprehensive scene description |
| `<REGION_PROPOSAL>` | Region Proposals | Identifies candidate bounding regions for salient objects | Candidate bounding boxes |

---

## Project Structure

```
vision-ai-app/
├── README.md                           # Documentation and technical guide
├── LICENSE                             # MIT License
├── pyproject.toml                      # Build metadata & dependency declarations
├── requirements.txt                    # Project dependencies
├── run_vision_ai.py                    # CLI entrypoint runner
├── app.py                              # Streamlit web dashboard
├── Vision_AI_Florence2_Walkthrough.ipynb # Step-by-step notebook tutorial
├── src/
│   └── vision_ai/
│       ├── __init__.py                 # Package exports
│       ├── config.py                   # Engine configuration & accelerator detection
│       ├── tasks.py                    # Task prompt catalog & alias mapping
│       ├── image_utils.py              # URL fetching, local loading & RGBA conversion
│       ├── visualizer.py               # PIL & Matplotlib bounding box renderers
│       ├── exporter.py                 # JSON & tabular record formatting
│       ├── engine.py                   # Model lifecycle & inference pipeline
│       └── cli.py                      # CLI argument parsing & command dispatcher
└── tests/
    ├── test_engine.py                  # Config & mock pipeline tests
    ├── test_image_utils.py             # Image loading & format validation tests
    ├── test_tasks.py                   # Task prompt enum & alias resolution tests
    └── test_visualizer_exporter.py     # Annotation drawing & export verification
```

---

## Installation

### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0+ (CUDA, MPS, or CPU build)

### Setup

```bash
git clone https://github.com/JohnnyWilson16/vision-ai-app.git
cd vision-ai-app

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Interactive Web Application

Launch the Streamlit dashboard to evaluate images and inspect bounding box coordinates:

```bash
streamlit run app.py
```

Features:
- Select from built-in benchmark images, upload local files, or fetch remote URLs.
- Switch dynamically across all Florence-2 task modes.
- Side-by-side visual comparison with labeled bounding boxes.
- Download annotated images (PNG) and structured JSON reports.

### 2. Command-Line Interface (CLI)

Run inference directly from the terminal:

```bash
# Object Detection on an image URL
python run_vision_ai.py \
  --image "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" \
  --task OD \
  --save-image car_detected.png \
  --save-json car_results.json

# Dense Region Captioning on a local image
python run_vision_ai.py \
  --image path/to/image.jpg \
  --task DENSE \
  --save-image dense_output.png

# Extract OCR text from a document
python run_vision_ai.py \
  --image path/to/document.png \
  --task OCR
```

#### CLI Options
```
--image, -i      Path to local image file or remote image URL (Required)
--task, -t       Vision task: OD, CAPTION, DETAILED, MORE_DETAILED, OCR, OCR_REGION, DENSE, PROPOSAL (Default: OD)
--text           Optional prompt text (for phrase grounding or conditional queries)
--model, -m      Hugging Face model ID (Default: florence-community/Florence-2-base)
--device, -d     Target device: 'cpu', 'cuda', 'mps' (Default: auto-detected)
--beams, -b      Beam search width (Default: 3)
--save-image     Output path to save annotated image (PNG/JPG)
--save-json      Output path to export structured JSON prediction
```

### 3. Python API

```python
from vision_ai import VisionAIEngine, VisionAIConfig, VisionTask, load_image

# Initialize configuration (auto-detects CUDA / MPS / CPU)
config = VisionAIConfig(
    model_id="florence-community/Florence-2-base",
    num_beams=3
)
engine = VisionAIEngine(config)

# Run Object Detection
result = engine.run_task(
    image="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    task=VisionTask.OBJECT_DETECTION,
    render_annotation=True
)

print("Parsed Answer:", result["parsed_answer"])
print(f"Latency: {result['latency_ms']} ms")

# Save annotated image
if result["annotated_image"]:
    result["annotated_image"].save("detection_output.png")
```

---

## Example Output

Given a sample vehicle image, running `<OD>` returns normalized pixel bounding boxes and category labels:

```json
{
  "<OD>": {
    "bboxes": [
      [34, 160, 597, 371],
      [272, 241, 303, 247],
      [454, 276, 553, 370],
      [96, 280, 198, 371]
    ],
    "labels": [
      "car",
      "door handle",
      "wheel",
      "wheel"
    ]
  }
}
```

The rendering module maps these coordinates to colored bounding boxes with contrasting category badges.

---

## Testing & Quality Assurance

The test suite covers task resolution, image format sanitization (RGBA $\rightarrow$ RGB conversion), coordinate transformations, and engine inference mocking:

```bash
pytest -v tests/
```

Test coverage includes:
- `tests/test_tasks.py`: Validates task prompts, metadata consistency, and alias resolution.
- `tests/test_image_utils.py`: Validates RGB conversion, alpha channel flattening, and EXIF orientation.
- `tests/test_visualizer_exporter.py`: Validates bounding box rendering and JSON/CSV serialization.
- `tests/test_engine.py`: Validates configuration defaults and mock inference execution.

---

## Hardware & Deployment Considerations

- **Apple Silicon (MPS)**: Supported natively via PyTorch `mps` backend with `float16` precision.
- **NVIDIA GPU (CUDA)**: Supports `device_map="auto"` and `float16` precision for fast inference.
- **CPU Fallback**: Uses standard `float32` precision when no GPU accelerator is detected.
- **Model Checkpoints**: Defaults to `florence-community/Florence-2-base` (~0.23B parameters) for balanced memory efficiency and speed. Can be configured to `microsoft/Florence-2-large` (~0.77B parameters) for complex document understanding.

---

## Limitations & Known Constraints

- **Small Text in Dense Documents**: For very high-density multi-page PDFs or low-resolution scans, specialized document OCR models or high-resolution tile cropping may yield higher fidelity.
- **Flash Attention**: Flash Attention is optional; the engine falls back to standard PyTorch SDPA (Scaled Dot-Product Attention) when `flash-attn` is not installed.
- **Trust Remote Code**: Florence-2 custom architecture requires `trust_remote_code=True` when loading from Hugging Face.

---

## Responsible Use

Florence-2 is a general-purpose vision-language model trained on web-scale datasets. When deploying in production contexts:
- Verify model outputs before taking automated physical or administrative actions.
- Be conscious of domain-specific biases in object classification and caption generation.
- Ensure compliance with local privacy regulations when processing images containing identifiable individuals or sensitive documents.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Author

**Johnny Wilson Dougherty**  
- GitHub: [@JohnnyWilson16](https://github.com/JohnnyWilson16)  
- LinkedIn: [Johnny Wilson Dougherty](https://www.linkedin.com/in/johnny-wilson-dougherty-81693b292)  
- Email: `johnnydougherty09@gmail.com`
