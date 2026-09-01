"""
Core Vision AI Engine powered by Microsoft Florence-2.

Encapsulates model initialization, memory management, multi-device routing,
beam-search inference, and structured coordinate decoding.

Author: Johnny Wilson Dougherty
"""

import time
import logging
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
from PIL import Image
import torch

from .config import VisionAIConfig
from .tasks import VisionTask, TASK_METADATA, get_task_by_name
from .image_utils import load_image, prepare_image_rgb
from .visualizer import annotate_image
from .postprocessing import filter_prediction_clutter

logger = logging.getLogger("vision_ai.engine")



class VisionAIEngine:
    """
    Unified Vision-Language inference engine using Florence-2.
    """

    def __init__(self, config: Optional[VisionAIConfig] = None):
        self.config = config or VisionAIConfig()
        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self):
        """
        Load the Florence-2 model and AutoProcessor into memory with hardware mapping.
        """
        if self._is_loaded:
            return

        logger.info(f"Loading Florence-2 model: {self.config.model_id} on {self.config.device}...")
        
        # Import dynamically to support diverse transformers installations
        try:
            from transformers import Florence2ForConditionalGeneration, AutoProcessor
            model_cls = Florence2ForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoProcessor
            model_cls = AutoModelForCausalLM

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir,
        )

        # Load model with device placement and dtype
        load_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "cache_dir": self.config.cache_dir,
        }

        if self.config.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.config.torch_dtype

        # Multi-device mapping
        if self.config.device == "cuda" and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            self.model = model_cls.from_pretrained(self.config.model_id, **load_kwargs)
        else:
            self.model = model_cls.from_pretrained(self.config.model_id, **load_kwargs)
            target_device = torch.device(self.config.device)
            self.model.to(target_device)

        self.model.eval()
        self._is_loaded = True
        logger.info("Florence-2 model and processor initialized successfully.")

    def run_task(
        self,
        image: Union[str, Path, Image.Image],
        task: Union[str, VisionTask] = VisionTask.OBJECT_DETECTION,
        text_input: Optional[str] = None,
        num_beams: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        filter_clutter: bool = True,
        max_boxes: int = 12,
        iou_threshold: float = 0.55,
        min_area_ratio: float = 0.005,
        render_annotation: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a Vision AI task on an image.

        Parameters:
            image: Image source (local path, URL, or PIL Image)
            task: Task enum or string alias ('<OD>', '<OCR>', '<CAPTION>', etc.)
            text_input: Optional text input for grounding/prompting
            num_beams: Override default beam count for generation
            max_new_tokens: Override max token length
            filter_clutter: Whether to apply NMS and area filtering to reduce duplicate/microscopic boxes
            max_boxes: Maximum number of bounding boxes to keep after NMS
            iou_threshold: IoU overlap threshold for suppression (0.1 = strict, 0.9 = loose)
            min_area_ratio: Minimum box area relative to image (e.g. 0.005 = 0.5%)
            render_annotation: Whether to draw bounding boxes on the result image

        Returns:
            Dictionary containing task result, structured dict, and images.
        """
        if not self._is_loaded:
            self.load_model()

        # Resolve task enum
        if isinstance(task, str):
            task_enum = get_task_by_name(task)
        else:
            task_enum = task

        task_prompt = task_enum.value
        full_prompt = f"{task_prompt} {text_input}" if text_input else task_prompt

        # Load and prepare image
        pil_image = load_image(image)
        w, h = pil_image.size

        start_time = time.perf_counter()

        # Tokenize and process multi-modal inputs
        inputs = self.processor(
            text=full_prompt,
            images=pil_image,
            return_tensors="pt",
        )

        # Move tensors to the model device and match dtype for pixel_values
        device = self.model.device
        prepared_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k == "pixel_values" and self.config.torch_dtype in (torch.float16, torch.bfloat16):
                    prepared_inputs[k] = v.to(device, dtype=self.config.torch_dtype)
                else:
                    prepared_inputs[k] = v.to(device)
            else:
                prepared_inputs[k] = v

        # Generate sequence tokens via beam search
        beams = num_beams if num_beams is not None else self.config.num_beams
        max_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=prepared_inputs["input_ids"],
                pixel_values=prepared_inputs["pixel_values"],
                max_new_tokens=max_tokens,
                num_beams=beams,
                early_stopping=self.config.early_stopping,
            )

        # Decode generated token IDs
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        # Post-process into structured coordinates / dict
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(w, h),
        )

        # Apply Non-Maximum Suppression / box decluttering if enabled
        if filter_clutter and TASK_METADATA.get(task_enum, {}).get("has_boxes", False):
            parsed_answer = filter_prediction_clutter(
                prediction=parsed_answer,
                image_size=(w, h),
                iou_threshold=iou_threshold,
                min_area_ratio=min_area_ratio,
                max_boxes=max_boxes,
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        # Render visual bounding boxes if requested and applicable
        annotated_img = None
        if render_annotation and TASK_METADATA.get(task_enum, {}).get("has_boxes", False):
            annotated_img = annotate_image(pil_image, parsed_answer)


        return {
            "task": task_enum,
            "task_name": TASK_METADATA[task_enum]["name"],
            "prompt": full_prompt,
            "raw_text": generated_text,
            "parsed_answer": parsed_answer,
            "image": pil_image,
            "annotated_image": annotated_img,
            "latency_ms": round(latency_ms, 2),
            "image_size": (w, h),
        }
