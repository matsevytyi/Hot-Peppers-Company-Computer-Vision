"""Run model inference on video and save annotated output."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import cv2
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from pipelines.contracts import EvalConfig  # noqa: E402
from pipelines.dependencies import assert_mamba_runtime_support  # noqa: E402
from pipelines.evaluation import _normalize_detection_outputs  # noqa: E402
from pipelines.inference_loader import load_model_from_eval_entry  # noqa: E402
from pipelines.training import resolve_device  # noqa: E402
from pipelines.video_inference import (  # noqa: E402
    draw_detections,
    format_router_overlay,
    infer_domain_names,
    load_class_names,
    pick_eval_model_entry,
    prepare_frame_tensor,
    rescale_boxes_xyxy,
)
from pipelines.yolo_ops import decode_predictions  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--video-input", type=str, required=True)
    parser.add_argument("--video-output", type=str, required=True)
    parser.add_argument("--classes-config", type=str, default="configs/classes/common_8.yaml")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--conf-threshold", type=float, default=-1.0)
    parser.add_argument("--nms-iou", type=float, default=-1.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def main() -> None:
    args = parse_args()
    assert_mamba_runtime_support()

    cfg_path = _resolve_path(args.config)
    cfg = EvalConfig.from_yaml(cfg_path).eval
    device_preference = args.device or str(cfg.get("device", "cuda"))
    device = resolve_device(device_preference)

    image_size = int(args.image_size) if args.image_size > 0 else int(
        (cfg.get("datasets") or [{}])[0].get("image_size", 640)
    )
    conf_threshold = float(args.conf_threshold) if args.conf_threshold >= 0 else float(cfg.get("conf_threshold", 0.25))
    nms_iou = float(args.nms_iou) if args.nms_iou >= 0 else float(cfg.get("nms_iou", 0.5))
    max_frames = int(args.max_frames) if args.max_frames > 0 else None

    model_entry = pick_eval_model_entry(cfg.get("models", []), args.model_name)
    model, section = load_model_from_eval_entry(model_cfg_dict=model_entry, repo_root=REPO_ROOT, device=str(device))
    model = model.to(device).eval()

    classes_path = _resolve_path(args.classes_config)
    class_names = load_class_names(classes_path)

    input_path = _resolve_path(args.video_input)
    output_path = _resolve_path(args.video_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(cap.get(cv2.CAP_PROP_FPS))
    output_fps = input_fps if input_fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    print(f"Running inference with model='{args.model_name}' on {input_path}")
    print(
        f"Settings: device={device}, image_size={image_size}, "
        f"conf_threshold={conf_threshold}, nms_iou={nms_iou}"
    )

    processed = 0
    started = time.perf_counter()

    try:
        with torch.no_grad():
            while True:
                if max_frames is not None and processed >= max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                batch = prepare_frame_tensor(frame, image_size=image_size, device=device)
                model_outputs = model(batch)
                outputs, router_probs = _normalize_detection_outputs(model_outputs)
                predictions = decode_predictions(
                    outputs,
                    num_classes=section.num_classes,
                    image_size=image_size,
                    conf_threshold=conf_threshold,
                    nms_iou=nms_iou,
                )
                pred = predictions[0]
                boxes = rescale_boxes_xyxy(
                    pred["boxes"],
                    from_width=image_size,
                    from_height=image_size,
                    to_width=frame_width,
                    to_height=frame_height,
                )

                num_domains = int(router_probs.shape[1]) if torch.is_tensor(router_probs) and router_probs.ndim == 2 else 0
                domain_names = infer_domain_names(model=model, model_cfg_dict=model_entry, num_domains=num_domains)
                router_text = format_router_overlay(router_probs, domain_names)

                draw_detections(
                    frame,
                    boxes_xyxy=boxes,
                    scores=pred["scores"],
                    labels=pred["labels"],
                    class_names=class_names,
                    router_text=router_text,
                )
                writer.write(frame)
                processed += 1

                if processed % 100 == 0:
                    elapsed = max(time.perf_counter() - started, 1e-8)
                    print(f"Processed {processed} frames ({processed / elapsed:.2f} FPS)")
    finally:
        cap.release()
        writer.release()

    elapsed = max(time.perf_counter() - started, 1e-8)
    fps = processed / elapsed if processed > 0 else 0.0
    print(f"Done. Frames: {processed}, elapsed: {elapsed:.2f}s, avg FPS: {fps:.2f}")
    print(f"Saved annotated video: {output_path}")


if __name__ == "__main__":
    main()
