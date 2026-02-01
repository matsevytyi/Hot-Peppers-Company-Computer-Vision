"""Utility functions including loss functions."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """Combined loss for bbox regression and confidence with numerical stability."""

    def __init__(self, bbox_weight: float = 1.0, conf_weight: float = 1.0, pos_weight: float = 1.0):
        super().__init__()
        self.bbox_weight = bbox_weight
        self.conf_weight = conf_weight
        self.pos_weight = pos_weight  # Weight for positive (object present) samples

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        predictions = predictions.contiguous()
        targets = targets.contiguous()
        pred_bbox = predictions[..., :4]
        pred_conf = predictions[..., 4]

        target_bbox = targets[..., :4]
        target_conf = targets[..., 4]

        # Clamp predictions for numerical stability
        pred_conf = torch.clamp(pred_conf, -10, 10)
        
        # Only apply bbox loss where objects are present (target_conf > 0.5)
        object_mask = target_conf > 0.5
        if object_mask.sum() > 0:
            bbox_loss = F.smooth_l1_loss(
                pred_bbox[object_mask], 
                target_bbox[object_mask], 
                reduction="mean"
            )
        else:
            bbox_loss = torch.tensor(0.0, device=predictions.device)

        # Confidence loss with pos_weight for class imbalance
        conf_loss = F.binary_cross_entropy_with_logits(
            pred_conf, 
            target_conf, 
            pos_weight=torch.tensor([self.pos_weight], device=predictions.device),
            reduction="mean"
        )

        total_loss = self.bbox_weight * bbox_loss + self.conf_weight * conf_loss
        
        # Ensure no NaN values
        if torch.isnan(total_loss):
            total_loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)

        return {
            "loss": total_loss,
            "bbox_loss": bbox_loss.detach() if not torch.isnan(bbox_loss) else torch.tensor(0.0),
            "conf_loss": conf_loss.detach() if not torch.isnan(conf_loss) else torch.tensor(0.0),
        }


def giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """Generalized IoU loss for [N, 4] boxes in (x, y, w, h) format."""
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

    iou = inter_area / (union_area + 1e-7)

    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)

    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    return 1 - giou.mean()
