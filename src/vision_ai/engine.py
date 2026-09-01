"""
Core Vision AI Engine powered by Microsoft Florence-2.

Optimized for memory-efficient multi-device execution and fast inference.

Author: Johnny Wilson Dougherty
"""

import gc
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
        Load the Florence-2 model and AutoProcessor into memory with low memory footprint.
        """
        if self._is_loaded:
            return

        # Restrict CPU thread creation to prevent stack memory bloat
        if self.config.device == "cpu":
            try:
                torch.set_num_threads(2)
            except Exception:
                pass

        logger.info(f"Loading Florence-2 model: {self.config.model_id} on {self.config.device}...")
        
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

        load_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
            "cache_dir": self.config.cache_dir,
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }

        if self.config.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.config.torch_dtype

        if self.config.device == "cuda" and torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
            self.model = model_cls.from_pretrained(self.config.model_id, **load_kwargs)
        else:
            self.model = model_cls.from_pretrained(self.config.model_id, **load_kwargs)
            target_device = torch.device(self.config.device)
            self.model.to(target_device)

        self.model.eval()
        self._is_loaded = True
        gc.collect()
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
        Execute a Vision AI task on an image with memory optimization.
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
        orig_w, orig_h = pil_image.size

        # Downsample if image exceeds max dimension to avoid OOM memory spikes
        max_dim = self.config.max_image_dim
        scale_ratio = 1.0
        proc_image = pil_image
        if max(orig_w, orig_h) > max_dim:
            scale_ratio = max_dim / max(orig_w, orig_h)
            new_w = int(orig_w * scale_ratio)
            new_h = int(orig_h * scale_ratio)
            proc_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        proc_w, proc_h = proc_image.size
        start_time = time.perf_counter()

        # Tokenize multi-modal inputs
        inputs = self.processor(
            text=full_prompt,
            images=proc_image,
            return_tensors="pt",
        )

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

        # Decode generated text
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        # Post-process into structured coordinates based on original image size
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(orig_w, orig_h),
        )

        # Apply Non-Maximum Suppression / box decluttering if enabled
        if filter_clutter and TASK_METADATA.get(task_enum, {}).get("has_boxes", False):
            parsed_answer = filter_prediction_clutter(
                prediction=parsed_answer,
                image_size=(orig_w, orig_h),
                iou_threshold=iou_threshold,
                min_area_ratio=min_area_ratio,
                max_boxes=max_boxes,
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0

        # Render visual bounding boxes on original image
        annotated_img = None
        if render_annotation and TASK_METADATA.get(task_enum, {}).get("has_boxes", False):
            annotated_img = annotate_image(pil_image, parsed_answer)

        # Explicit garbage cleanup to keep memory inside limits
        del inputs
        del prepared_inputs
        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {
            "task": task_enum,
            "task_name": TASK_METADATA[task_enum]["name"],
            "prompt": full_prompt,
            "raw_text": generated_text,
            "parsed_answer": parsed_answer,
            "image": pil_image,
            "annotated_image": annotated_img,
            "latency_ms": round(latency_ms, 2),
            "image_size": (orig_w, orig_h),
        }
