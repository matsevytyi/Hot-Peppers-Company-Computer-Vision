"""Evaluation helpers for base and LoRA checkpoints."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .power_monitor import NullPowerMonitor, build_power_monitor
from .training import resolve_device
from .yolo_ops import decode_predictions, targets_to_abs_xyxy


@dataclass
class EvaluationResult:
    metrics: Dict[str, float | int | str | bool | None]
    telemetry: List[Dict[str, float | int | str | bool | None]]


def _build_map_metric():
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision(
            iou_type="bbox",
            # Keep AP@[1,10,300] aligned with decode_predictions(max_detections=300)
            # to avoid dropping detections above 100 by the metric layer.
            max_detection_thresholds=[1, 10, 300],
        )
        if hasattr(metric, "warn_on_many_detections"):
            metric.warn_on_many_detections = False
        return metric
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


def evaluate_model_detailed(
    *,
    model: torch.nn.Module,
    dataloader,
    num_classes: int,
    image_size: int,
    device_preference: str = "cuda",
    conf_threshold: float = 0.25,
    nms_iou: float = 0.5,
    max_batches: int | None = None,
    power_enabled: bool = True,
    power_backend: str = "auto",
    power_gpu_index: int = 0,
    power_poll_interval_ms: int = 100,
    warmup_batches: int = 20,
    collect_telemetry: bool = True,
) -> EvaluationResult:
    device = resolve_device(device_preference)
    model = model.to(device)
    model.eval()

    map_metric = _build_map_metric()
    all_preds: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    warmup_batches = max(0, int(warmup_batches))
    poll_interval_s = max(0.001, float(power_poll_interval_ms) / 1000.0)

    monitor = build_power_monitor(enabled=power_enabled, backend=power_backend, gpu_index=power_gpu_index)
    try:
        monitor.start()
    except Exception as exc:
        monitor = NullPowerMonitor(reason=f"start_failed:{type(exc).__name__}")

    telemetry: List[Dict[str, float | int | str | bool | None]] = []
    measured_frames = 0
    measured_batches = 0
    measured_infer_time_s = 0.0
    total_energy_j = 0.0
    observed_power_samples = 0

    try:
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                images = images.to(device, non_blocking=True)
                targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

                is_measured_batch = batch_idx >= warmup_batches
                batch_power_samples: List[float] = []
                stop_event: Optional[threading.Event] = None
                sampler_thread: Optional[threading.Thread] = None

                if is_measured_batch and monitor.available:
                    stop_event = threading.Event()

                    def _poll_power() -> None:
                        while not stop_event.is_set():
                            power_w = monitor.read_power_w()
                            if power_w is not None and power_w > 0:
                                batch_power_samples.append(float(power_w))
                            stop_event.wait(poll_interval_s)

                    sampler_thread = threading.Thread(target=_poll_power, daemon=True)
                    sampler_thread.start()

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                outputs = model(images)
                predictions = decode_predictions(
                    outputs,
                    num_classes=num_classes,
                    image_size=image_size,
                    conf_threshold=conf_threshold,
                    nms_iou=nms_iou,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                end = time.perf_counter()

                if stop_event is not None:
                    stop_event.set()
                if sampler_thread is not None:
                    sampler_thread.join(timeout=1.0)

                if is_measured_batch and monitor.available and not batch_power_samples:
                    fallback_sample = monitor.read_power_w()
                    if fallback_sample is not None and fallback_sample > 0:
                        batch_power_samples.append(float(fallback_sample))

                infer_time_s = end - start
                batch_frames = int(images.shape[0])
                converted_targets = targets_to_abs_xyxy(targets, image_size=image_size)

                all_preds.extend([{k: v.detach().cpu() for k, v in p.items()} for p in predictions])
                all_targets.extend([{k: v.detach().cpu() for k, v in t.items()} for t in converted_targets])

                if map_metric is not None:
                    map_metric.update(all_preds[-len(predictions) :], all_targets[-len(converted_targets) :])

                if is_measured_batch:
                    measured_batches += 1
                    measured_frames += batch_frames
                    measured_infer_time_s += infer_time_s

                    batch_avg_power_w: Optional[float] = None
                    batch_energy_j: Optional[float] = None
                    batch_fpw: Optional[float] = None
                    if batch_power_samples:
                        observed_power_samples += len(batch_power_samples)
                        batch_avg_power_w = sum(batch_power_samples) / len(batch_power_samples)
                        batch_energy_j = batch_avg_power_w * infer_time_s
                        total_energy_j += batch_energy_j
                        if batch_energy_j > 0:
                            batch_fpw = batch_frames / batch_energy_j

                    if collect_telemetry:
                        telemetry.append(
                            {
                                "batch_idx": int(batch_idx),
                                "num_frames": batch_frames,
                                "infer_time_ms": infer_time_s * 1000.0,
                                "fps_batch": batch_frames / max(infer_time_s, 1e-8),
                                "avg_power_w_batch": batch_avg_power_w,
                                "energy_j_batch": batch_energy_j,
                                "frames_per_watt_batch": batch_fpw,
                                "power_backend": monitor.backend_name,
                            }
                        )
    finally:
        monitor.stop()

    metrics: Dict[str, float | int | str | bool | None] = {}

    fps = measured_frames / max(measured_infer_time_s, 1e-8) if measured_frames > 0 else 0.0
    metrics["fps"] = fps

    if map_metric is not None:
        computed = map_metric.compute()
        metrics["map_50"] = float(computed.get("map_50", torch.tensor(0.0)).item())
        metrics["map_50_95"] = float(computed.get("map", torch.tensor(0.0)).item())
        map_per_class = computed.get("map_per_class")
        if map_per_class is not None:
            # Depending on torchmetrics settings/version, `map_per_class` can be
            # absent, a scalar tensor, or a 1D tensor. Only iterate when 1D+.
            if torch.is_tensor(map_per_class):
                if map_per_class.ndim > 0 and map_per_class.numel() > 0:
                    for idx, value in enumerate(map_per_class):
                        metrics[f"class_{idx}_ap"] = float(value.item())
            elif isinstance(map_per_class, (list, tuple)):
                for idx, value in enumerate(map_per_class):
                    metrics[f"class_{idx}_ap"] = float(value)
    else:
        metrics.update(_fallback_precision_recall(all_preds, all_targets))
        metrics["map_50"] = float(metrics["precision@50"])
        metrics["map_50_95"] = 0.0

    power_available = observed_power_samples > 0
    avg_power_w: Optional[float] = None
    frames_per_watt: Optional[float] = None
    energy_per_frame_j: Optional[float] = None

    if power_available and total_energy_j > 0 and measured_infer_time_s > 0 and measured_frames > 0:
        avg_power_w = total_energy_j / measured_infer_time_s
        frames_per_watt = measured_frames / total_energy_j
        energy_per_frame_j = total_energy_j / measured_frames

    metrics["avg_power_w"] = avg_power_w
    metrics["frames_per_watt"] = frames_per_watt
    metrics["energy_per_frame_j"] = energy_per_frame_j
    metrics["total_energy_j"] = total_energy_j if power_available else None
    metrics["power_backend"] = monitor.backend_name
    metrics["power_available"] = power_available
    metrics["warmup_batches"] = warmup_batches
    metrics["measured_batches"] = measured_batches
    metrics["measured_frames"] = measured_frames

    if isinstance(monitor, NullPowerMonitor):
        metrics["power_reason"] = monitor.reason

    return EvaluationResult(metrics=metrics, telemetry=telemetry)


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
    """Backward-compatible wrapper that returns only legacy eval metrics."""
    result = evaluate_model_detailed(
        model=model,
        dataloader=dataloader,
        num_classes=num_classes,
        image_size=image_size,
        device_preference=device_preference,
        conf_threshold=conf_threshold,
        nms_iou=nms_iou,
        max_batches=max_batches,
        power_enabled=False,
        power_backend="off",
        warmup_batches=0,
        collect_telemetry=False,
    )

    metrics: Dict[str, float] = {}
    base_keys = {"fps", "map_50", "map_50_95", "precision@50", "recall@50", "f1@50"}
    for key, value in result.metrics.items():
        is_class_ap = key.startswith("class_") and key.endswith("_ap")
        if not (key in base_keys or is_class_ap):
            continue
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            metrics[key] = float(value)

    return metrics
