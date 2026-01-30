"""Evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch


def calculate_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Calculate IoU for boxes in (x, y, w, h) format."""
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1, min=0
    )

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area

    return inter_area / (union_area + 1e-7)


def calculate_ap(
    pred_boxes: torch.Tensor,
    pred_confs: torch.Tensor,
    target_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> float:
    """Calculate Average Precision (11-point interpolation)."""
    if len(pred_boxes) == 0:
        return 0.0

    sorted_idx = torch.argsort(pred_confs, descending=True)
    pred_boxes = pred_boxes[sorted_idx]
    pred_confs = pred_confs[sorted_idx]

    tp = []
    fp = []
    matched = set()

    for pred_box in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for j, target_box in enumerate(target_boxes):
            if j in matched:
                continue
            iou = calculate_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0))
            if iou.item() > best_iou:
                best_iou = iou.item()
                best_idx = j

        if best_iou >= iou_threshold:
            tp.append(1)
            fp.append(0)
            matched.add(best_idx)
        else:
            tp.append(0)
            fp.append(1)

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (len(target_boxes) + 1e-7)

    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0.0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11

    return float(ap)


class DetectionMetrics:
    """Track detection metrics during training."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        self.all_ious = []
        self.all_precisions = []
        self.all_recalls = []

    def update(self, predictions: dict, targets: torch.Tensor):
        pred_bbox = predictions["bbox"]
        pred_conf = predictions["confidence"]
        target_bbox = targets[:, :4]
        target_conf = targets[:, 4]

        ious = calculate_iou(pred_bbox, target_bbox)
        self.all_ious.extend(ious.detach().cpu().numpy())

        for i in range(len(pred_bbox)):
            iou = ious[i].item()
            is_correct = int(iou >= self.iou_threshold and pred_conf[i] > 0.5)
            is_positive = int(target_conf[i] > 0.5)

            if is_positive:
                self.all_precisions.append(1 if is_correct else 0)
                self.all_recalls.append(1 if is_correct else 0)

    def compute(self):
        if len(self.all_ious) == 0:
            return {"mean_iou": 0.0, "precision": 0.0, "recall": 0.0}

        return {
            "mean_iou": float(np.mean(self.all_ious)),
            "precision": float(np.mean(self.all_precisions)) if self.all_precisions else 0.0,
            "recall": float(np.mean(self.all_recalls)) if self.all_recalls else 0.0,
        }
