"""
Command-Line Interface (CLI) for Vision AI Application.

Author: Johnny Wilson Dougherty
"""

import argparse
import sys
from pathlib import Path

from .config import VisionAIConfig, get_default_device
from .engine import VisionAIEngine
from .tasks import VisionTask, TASK_METADATA
from .exporter import export_results_json


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-ai",
        description="Vision AI Application powered by Florence-2 (Author: Johnny Wilson Dougherty)",
    )

    parser.add_argument(
        "--image",
        "-i",
        required=True,
        type=str,
        help="Path to local image file or remote image URL.",
    )
    parser.add_argument(
        "--task",
        "-t",
        default="OD",
        type=str,
        help="Vision task: OD (Object Detection), CAPTION, DETAILED, MORE_DETAILED, OCR, OCR_REGION, DENSE (Dense Caption), PROPOSAL. Default: OD.",
    )
    parser.add_argument(
        "--text",
        default=None,
        type=str,
        help="Optional text prompt (used for phrase grounding or custom queries).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="florence-community/Florence-2-base",
        type=str,
        help="Hugging Face model ID (default: florence-community/Florence-2-base).",
    )
    parser.add_argument(
        "--device",
        "-d",
        default=None,
        type=str,
        help="Device to use: 'cpu', 'cuda', 'mps' (default: auto-detected).",
    )
    parser.add_argument(
        "--beams",
        "-b",
        default=3,
        type=int,
        help="Beam search count (default: 3).",
    )
    parser.add_argument(
        "--save-image",
        default=None,
        type=str,
        help="Optional file path to save the annotated output image.",
    )
    parser.add_argument(
        "--save-json",
        default=None,
        type=str,
        help="Optional file path to export structured JSON results.",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    device = args.device or get_default_device()
    config = VisionAIConfig(
        model_id=args.model,
        device=device,
        num_beams=args.beams,
    )

    print("=" * 60)
    print("   Vision AI Application (Florence-2 VLM Engine)")
    print("   Author: Johnny Wilson Dougherty")
    print(f"   Model: {config.model_id} | Device: {config.device}")
    print("=" * 60)

    try:
        engine = VisionAIEngine(config)
        print(f"[*] Processing image: {args.image}")
        print(f"[*] Task: {args.task}")

        result = engine.run_task(
            image=args.image,
            task=args.task,
            text_input=args.text,
            render_annotation=True,
        )

        print("\n" + "-" * 40)
        print(f"[+] Task: {result['task_name']} ({result['task'].value})")
        print(f"[+] Inference Latency: {result['latency_ms']} ms")
        print("-" * 40)
        print("[+] Parsed Output:")
        print(result["parsed_answer"])

        # Save annotated image if requested
        if args.save_image and result.get("annotated_image"):
            out_img_path = Path(args.save_image)
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            result["annotated_image"].save(out_img_path)
            print(f"[+] Annotated image saved to: {out_img_path}")

        # Save JSON if requested
        if args.save_json:
            export_results_json(
                prediction=result["parsed_answer"],
                task_name=result["task_name"],
                image_source=args.image,
                image_size=result["image_size"],
                output_path=args.save_json,
            )
            print(f"[+] Structured JSON saved to: {args.save_json}")

    except Exception as e:
        print(f"\n[!] Error during execution: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
