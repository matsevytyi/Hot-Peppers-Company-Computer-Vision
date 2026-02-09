"""Losses and post-processing for the Mamba-Vision YOLO-like head."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms


def _scale_choice(area: float, num_scales: int) -> int:
    if num_scales <= 1:
        return 0
    if area < 0.02:
        return 0
    if num_scales == 2:
        return 1
    if area < 0.10:
        return 1
    return 2


class MultiScaleYoloLoss(nn.Module):
    """Minimal anchor-free multi-scale loss for pilot/full runs."""

    def __init__(
        self,
        num_classes: int,
        obj_weight: float = 1.0,
        box_weight: float = 5.0,
        cls_weight: float = 1.0,
        scale_weights: List[float] | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.scale_weights = scale_weights or [1.0, 1.0, 1.0]
        self.obj_criterion = nn.BCEWithLogitsLoss(reduction="mean")
        self.cls_criterion = nn.BCEWithLogitsLoss(reduction="mean")

    def _decode_boxes(self, pred: torch.Tensor) -> torch.Tensor:
        b, _, h, w = pred.shape
        device = pred.device
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        tx = pred[:, 0]
        ty = pred[:, 1]
        tw = pred[:, 2]
        th = pred[:, 3]
        cx = (torch.sigmoid(tx) + grid_x) / max(w, 1)
        cy = (torch.sigmoid(ty) + grid_y) / max(h, 1)
        bw = torch.sigmoid(tw)
        bh = torch.sigmoid(th)
        return torch.stack((cx, cy, bw, bh), dim=1)

    def forward(self, outputs: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not outputs:
            raise ValueError("Model returned no output scales")

        total_obj = torch.tensor(0.0, device=outputs[0].device)
        total_box = torch.tensor(0.0, device=outputs[0].device)
        total_cls = torch.tensor(0.0, device=outputs[0].device)

        num_scales = len(outputs)
        for scale_idx, out in enumerate(outputs):
            batch, channels, height, width = out.shape
            expected_channels = 5 + self.num_classes
            if channels != expected_channels:
                raise RuntimeError(
                    f"Unexpected channel count for scale {scale_idx}: got {channels}, expected {expected_channels}"
                )

            obj_target = torch.zeros((batch, height, width), device=out.device)
            box_target = torch.zeros((batch, 4, height, width), device=out.device)
            cls_target = torch.zeros((batch, self.num_classes, height, width), device=out.device)
            pos_mask = torch.zeros((batch, height, width), dtype=torch.bool, device=out.device)

            for b in range(batch):
                boxes = targets[b]["boxes"].to(out.device)
                labels = targets[b]["labels"].to(out.device)
                for gt_idx in range(boxes.shape[0]):
                    cx, cy, bw, bh = boxes[gt_idx]
                    preferred_scale = _scale_choice(float((bw * bh).item()), num_scales)
                    if preferred_scale != scale_idx:
                        continue
                    gx = int(torch.clamp(cx * width, min=0, max=width - 1).item())
                    gy = int(torch.clamp(cy * height, min=0, max=height - 1).item())
                    if pos_mask[b, gy, gx]:
                        continue

                    pos_mask[b, gy, gx] = True
                    obj_target[b, gy, gx] = 1.0
                    box_target[b, :, gy, gx] = boxes[gt_idx]
                    cls_target[b, labels[gt_idx], gy, gx] = 1.0

            obj_loss = self.obj_criterion(out[:, 4], obj_target)
            pred_boxes = self._decode_boxes(out)
            if pos_mask.any():
                pred_pos = pred_boxes.permute(0, 2, 3, 1)[pos_mask]
                target_pos = box_target.permute(0, 2, 3, 1)[pos_mask]
                box_loss = F.smooth_l1_loss(pred_pos, target_pos, reduction="mean")

                pred_cls = out[:, 5:].permute(0, 2, 3, 1)[pos_mask]
                target_cls_pos = cls_target.permute(0, 2, 3, 1)[pos_mask]
                cls_loss = self.cls_criterion(pred_cls, target_cls_pos)
            else:
                box_loss = torch.tensor(0.0, device=out.device)
                cls_loss = torch.tensor(0.0, device=out.device)

            weight = self.scale_weights[min(scale_idx, len(self.scale_weights) - 1)]
            total_obj = total_obj + obj_loss * weight
            total_box = total_box + box_loss * weight
            total_cls = total_cls + cls_loss * weight

        total = self.obj_weight * total_obj + self.box_weight * total_box + self.cls_weight * total_cls
        return {
            "loss": total,
            "obj_loss": total_obj.detach(),
            "box_loss": total_box.detach(),
            "cls_loss": total_cls.detach(),
        }


def _xywh_to_xyxy_abs(boxes_xywh: torch.Tensor, image_size: int) -> torch.Tensor:
    cx = boxes_xywh[:, 0] * image_size
    cy = boxes_xywh[:, 1] * image_size
    w = boxes_xywh[:, 2] * image_size
    h = boxes_xywh[:, 3] * image_size
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return torch.stack((x1, y1, x2, y2), dim=1)


def decode_predictions(
    outputs: List[torch.Tensor],
    *,
    num_classes: int,
    image_size: int,
    conf_threshold: float = 0.25,
    nms_iou: float = 0.5,
    max_detections: int = 300,
) -> List[Dict[str, torch.Tensor]]:
    """Decode per-scale outputs into absolute xyxy detections."""
    if not outputs:
        return []

    batch = outputs[0].shape[0]
    decoded: List[Dict[str, torch.Tensor]] = []
    for b in range(batch):
        all_boxes: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        for out in outputs:
            _, channels, height, width = out.shape
            if channels != 5 + num_classes:
                raise RuntimeError(
                    f"Channel mismatch in decode: got {channels}, expected {5 + num_classes}"
                )

            sample = out[b]
            tx = sample[0]
            ty = sample[1]
            tw = sample[2]
            th = sample[3]
            obj = torch.sigmoid(sample[4])
            cls = torch.sigmoid(sample[5 : 5 + num_classes])

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=sample.device),
                torch.arange(width, device=sample.device),
                indexing="ij",
            )

            cx = (torch.sigmoid(tx) + grid_x) / max(width, 1)
            cy = (torch.sigmoid(ty) + grid_y) / max(height, 1)
            bw = torch.sigmoid(tw)
            bh = torch.sigmoid(th)

            boxes = torch.stack((cx, cy, bw, bh), dim=-1).reshape(-1, 4)
            boxes = _xywh_to_xyxy_abs(boxes, image_size=image_size)

            obj_flat = obj.reshape(-1, 1)
            cls_flat = cls.permute(1, 2, 0).reshape(-1, num_classes)
            scores = obj_flat * cls_flat

            score_vals, label_ids = scores.max(dim=1)
            keep = score_vals > conf_threshold
            if keep.any():
                all_boxes.append(boxes[keep])
                all_scores.append(score_vals[keep])
                all_labels.append(label_ids[keep])

        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)
            keep_idx = batched_nms(boxes, scores, labels, nms_iou)
            keep_idx = keep_idx[:max_detections]
            decoded.append(
                {
                    "boxes": boxes[keep_idx],
                    "scores": scores[keep_idx],
                    "labels": labels[keep_idx],
                }
            )
        else:
            device = outputs[0].device
            decoded.append(
                {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros((0,), device=device),
                    "labels": torch.zeros((0,), dtype=torch.long, device=device),
                }
            )

    return decoded


def targets_to_abs_xyxy(targets: List[Dict[str, torch.Tensor]], image_size: int) -> List[Dict[str, torch.Tensor]]:
    converted: List[Dict[str, torch.Tensor]] = []
    for target in targets:
        boxes = target["boxes"]
        if boxes.numel() == 0:
            converted.append(
                {
                    "boxes": torch.zeros((0, 4), device=boxes.device),
                    "labels": target["labels"],
                }
            )
            continue
        converted.append(
            {
                "boxes": _xywh_to_xyxy_abs(boxes, image_size=image_size),
                "labels": target["labels"],
            }
        )
    return converted
