"""Evaluation helpers for base and LoRA checkpoints."""

from __future__ import annotations

import time
from typing import Dict, List

import torch

from .training import resolve_device
from .yolo_ops import decode_predictions, targets_to_abs_xyxy


def _build_map_metric():
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        return MeanAveragePrecision(iou_type="bbox")
    except Exception:
        return None


def _fallback_precision_recall(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"]
        target_boxes = target["boxes"]
        if pred_boxes.numel() == 0 and target_boxes.numel() == 0:
            continue
        if pred_boxes.numel() == 0:
            fn += int(target_boxes.shape[0])
            continue
        if target_boxes.numel() == 0:
            fp += int(pred_boxes.shape[0])
            continue

        ious = box_iou(pred_boxes, target_boxes)
        matched_targets = set()
        for i in range(ious.shape[0]):
            best_iou, best_idx = ious[i].max(dim=0)
            if best_iou.item() >= iou_thresh and int(best_idx.item()) not in matched_targets:
                tp += 1
                matched_targets.add(int(best_idx.item()))
            else:
                fp += 1
        fn += int(target_boxes.shape[0] - len(matched_targets))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision@50": precision, "recall@50": recall, "f1@50": f1}


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def evaluate_model(
    *,
    model: torch.nn.Module,
    dataloader,
    num_classes: int,
    image_size: int,
    device_preference: str = "cuda",
    conf_threshold: float = 0.25,
    nms_iou: float = 0.5,
    max_batches: int | None = None,
) -> Dict[str, float]:
    device = resolve_device(device_preference)
    model = model.to(device)
    model.eval()

    map_metric = _build_map_metric()
    all_preds: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    total_images = 0
    total_infer_time = 0.0

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            total_infer_time += end - start
            total_images += images.shape[0]

            predictions = decode_predictions(
                outputs,
                num_classes=num_classes,
                image_size=image_size,
                conf_threshold=conf_threshold,
                nms_iou=nms_iou,
            )
            converted_targets = targets_to_abs_xyxy(targets, image_size=image_size)

            all_preds.extend([{k: v.detach().cpu() for k, v in p.items()} for p in predictions])
            all_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in converted_targets])

            if map_metric is not None:
                map_metric.update(all_preds[-len(predictions) :], all_targets[-len(converted_targets) :])

    metrics: Dict[str, float] = {}
    fps = total_images / max(total_infer_time, 1e-8)
    metrics["fps"] = fps

    if map_metric is not None:
        computed = map_metric.compute()
        metrics["map_50"] = float(computed.get("map_50", torch.tensor(0.0)).item())
        metrics["map_50_95"] = float(computed.get("map", torch.tensor(0.0)).item())
        map_per_class = computed.get("map_per_class")
        if map_per_class is not None and map_per_class.numel() > 0:
            for idx, value in enumerate(map_per_class):
                metrics[f"class_{idx}_ap"] = float(value.item())
    else:
        metrics.update(_fallback_precision_recall(all_preds, all_targets))
        metrics["map_50"] = metrics["precision@50"]
        metrics["map_50_95"] = 0.0

    return metrics
