"""
Vision AI Web Application
=========================
Interactive multi-task Vision-Language interface powered by Microsoft Florence-2.

Author: Johnny Wilson Dougherty
GitHub: https://github.com/JohnnyWilson16
Email: johnnydougherty09@gmail.com
"""

import gc
import io
import json
import sys
from pathlib import Path
from PIL import Image
import streamlit as st


# Add src/ to path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from vision_ai.config import VisionAIConfig, get_default_device
from vision_ai.engine import VisionAIEngine
from vision_ai.tasks import VisionTask, TASK_METADATA
from vision_ai.image_utils import load_image
from vision_ai.exporter import export_results_json, format_detection_records

# Page configuration
st.set_page_config(
    page_title="Vision AI Workspace",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLES_DIR = Path(__file__).parent / "assets" / "samples"

SAMPLE_IMAGES = {
    "Automobile / Street Scene": str(SAMPLES_DIR / "car.jpg") if (SAMPLES_DIR / "car.jpg").exists() else "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
    "Animals & Pets": str(SAMPLES_DIR / "cat.jpg") if (SAMPLES_DIR / "cat.jpg").exists() else "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg",
    "Printed Text & Document Receipt": str(SAMPLES_DIR / "document.png") if (SAMPLES_DIR / "document.png").exists() else "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg",
}



@st.cache_resource(show_spinner=False)
def get_engine(model_id: str, device: str, beams: int) -> VisionAIEngine:
    """Initialize and cache the Florence-2 model engine."""
    config = VisionAIConfig(
        model_id=model_id,
        device=device,
        num_beams=beams,
    )
    engine = VisionAIEngine(config)
    engine.load_model()
    return engine


def main():
    # Sidebar
    st.sidebar.markdown("### Model Configuration")
    
    available_models = [
        "florence-community/Florence-2-base",
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large",
    ]
    selected_model = st.sidebar.selectbox("Model Checkpoint", available_models, index=0)

    default_dev = get_default_device()
    device_options = ["mps", "cuda", "cpu"]
    default_index = device_options.index(default_dev) if default_dev in device_options else 2
    selected_device = st.sidebar.selectbox("Compute Accelerator", device_options, index=default_index)

    with st.sidebar.expander("Inference & Box Filtering", expanded=False):
        beam_count = st.slider("Beam Search Width", min_value=1, max_value=3, value=1, help="1 = Greedy search (fastest, lowest memory).")
        max_tokens = st.slider("Max Output Tokens", min_value=128, max_value=1024, value=512, step=128)
        st.markdown("---")
        enable_nms = st.checkbox("Declutter Overlapping Boxes (NMS)", value=True, help="Removes redundant duplicate boxes and microscopic noise.")
        max_regions = st.slider("Max Displayed Regions / Boxes", min_value=1, max_value=25, value=10, help="Caps the number of candidate proposals or boxes displayed.")
        iou_thresh = st.slider("Overlap IoU Threshold", min_value=0.1, max_value=0.9, value=0.50, step=0.05, help="Lower values aggressively suppress overlapping boxes.")


    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Author:** Johnny Wilson Dougherty  
        **Repository:** [JohnnyWilson16/vision-ai-app](https://github.com/JohnnyWilson16/vision-ai-app)  
        **Architecture:** Microsoft Florence-2 (Sequence-to-Sequence VLM)
        """
    )

    # Main Header
    st.markdown("## Vision AI Workspace")
    st.markdown(
        "Multi-task computer vision and scene understanding powered by open-source Vision-Language Models. "
        "Select a task, provide an image, and run inference locally without external API dependencies."
    )
    st.markdown("---")

    # Layout: Controls & Input
    col_task, col_source = st.columns([1, 1], gap="medium")

    with col_task:
        st.markdown("#### 1. Select Vision Task")
        task_options = [
            ("Object Detection (<OD>)", VisionTask.OBJECT_DETECTION),
            ("Dense Region Captioning (<DENSE_REGION_CAPTION>)", VisionTask.DENSE_REGION_CAPTION),
            ("OCR with Bounding Boxes (<OCR_WITH_REGION>)", VisionTask.OCR_WITH_REGION),
            ("Plain Text OCR (<OCR>)", VisionTask.OCR),
            ("Detailed Scene Caption (<DETAILED_CAPTION>)", VisionTask.DETAILED_CAPTION),
            ("Exhaustive Scene Analysis (<MORE_DETAILED_CAPTION>)", VisionTask.MORE_DETAILED_CAPTION),
            ("Concise Caption (<CAPTION>)", VisionTask.CAPTION),
            ("Region Proposals (<REGION_PROPOSAL>)", VisionTask.REGION_PROPOSAL),
        ]
        task_labels = [opt[0] for opt in task_options]
        selected_task_label = st.selectbox("Task Mode", task_labels, index=0, label_visibility="collapsed")
        selected_task = next(opt[1] for opt in task_options if opt[0] == selected_task_label)

        task_info = TASK_METADATA[selected_task]
        st.caption(f"**Prompt Tag:** `{task_info['prompt']}` — {task_info['description']}")

    with col_source:
        st.markdown("#### 2. Provide Input Image")
        source_mode = st.radio(
            "Input Mode",
            ["Preset Sample", "Upload File", "Remote URL"],
            horizontal=True,
            label_visibility="collapsed",
        )

        image_source = None
        if source_mode == "Preset Sample":
            preset_choice = st.selectbox("Choose sample", list(SAMPLE_IMAGES.keys()))
            image_source = SAMPLE_IMAGES[preset_choice]
        elif source_mode == "Upload File":
            uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
            if uploaded:
                image_source = uploaded
        else:
            url_str = st.text_input(
                "Image URL",
                placeholder="https://example.com/image.jpg",
                label_visibility="collapsed",
            )
            st.caption("Paste a direct image URL (`.jpg`, `.png`, `.webp`), Google search link, or webpage with a featured image.")
            if url_str:
                image_source = url_str.strip()

    # Load Image
    if not image_source:
        st.info("Select a preset sample, upload an image file, or enter an image URL to begin.")
        return

    try:
        pil_image = load_image(image_source)
    except Exception as e:
        st.error(f"Unable to load image: {e}")
        st.info("💡 **Tip**: To get a direct image link, right-click any image on the web and choose **'Copy Image Address'** (the link usually ends in `.jpg`, `.png`, or `.webp`).")
        return

    st.markdown("---")

    # Run Trigger
    if st.button("Run Inference", type="primary", use_container_width=True):
        with st.spinner("Processing image with Florence-2..."):
            try:
                engine = get_engine(selected_model, selected_device, beam_count)
                result = engine.run_task(
                    image=pil_image,
                    task=selected_task,
                    num_beams=beam_count,
                    max_new_tokens=max_tokens,
                    filter_clutter=enable_nms,
                    max_boxes=max_regions,
                    iou_threshold=iou_thresh,
                    render_annotation=True,
                )
                st.session_state["inference_result"] = result
                gc.collect()
            except Exception as e:
                st.error(f"Inference execution failed: {e}")
                return


    # Render Results

    if "inference_result" in st.session_state:
        result = st.session_state["inference_result"]
        parsed = result["parsed_answer"]
        has_annotations = result.get("annotated_image") is not None

        # Metrics Bar
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Task", result["task_name"])
        with m2:
            st.metric("Inference Latency", f"{result['latency_ms']} ms")
        with m3:
            st.metric("Image Resolution", f"{result['image_size'][0]} × {result['image_size'][1]} px")

        st.markdown("#### Visual Output")
        img_col1, img_col2 = st.columns(2, gap="medium")

        with img_col1:
            st.markdown("**Original Image**")
            try:
                st.image(result["image"], use_container_width=True)
            except TypeError:
                st.image(result["image"])

        with img_col2:
            st.markdown("**Model Output**")
            if has_annotations:
                try:
                    st.image(result["annotated_image"], use_container_width=True)
                except TypeError:
                    st.image(result["annotated_image"])
            else:
                # Text-only task output
                text_content = parsed.get(result["task"].value, result["raw_text"])
                st.text_area("Generated Output", value=str(text_content), height=240)

        # Tabular and Structured Records
        entities = format_detection_records(parsed, image_size=result["image_size"])
        if entities:
            st.markdown("#### Detected Entities & Coordinates")
            st.dataframe(entities, use_container_width=True)

        with st.expander("Raw Prediction JSON", expanded=False):
            st.json(parsed)

        # Export Actions
        st.markdown("#### Export Results")
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            if has_annotations:
                buf = io.BytesIO()
                result["annotated_image"].save(buf, format="PNG")
                st.download_button(
                    label="Download Annotated Image (PNG)",
                    data=buf.getvalue(),
                    file_name="annotated_prediction.png",
                    mime="image/png",
                    use_container_width=True,
                )

        with d_col2:
            json_report = export_results_json(
                prediction=parsed,
                task_name=result["task_name"],
                image_size=result["image_size"],
            )
            st.download_button(
                label="Download Structured JSON Report",
                data=json_report,
                file_name="prediction_report.json",
                mime="application/json",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
