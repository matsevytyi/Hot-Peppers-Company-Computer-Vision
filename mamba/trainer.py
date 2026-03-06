"""PyTorch Lightning training module."""
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from .metrics import DetectionMetrics
from .model import MambaUAVDetector
from .utils import DetectionLoss


class MambaDetectorModule(pl.LightningModule):
    """Lightning module for Mamba UAV Detector."""

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config

        self.model = MambaUAVDetector(
            backbone=config.model.backbone,
            d_model=config.model.d_model,
            d_state=config.model.d_state,
            mamba_layers=config.model.mamba_layers,
            output_last_only=config.model.output_last_only,
            freeze_backbone=config.model.freeze_backbone,
            pretrained=config.model.pretrained,
            mamba_type=config.model.mamba_type,
        )

        self.criterion = DetectionLoss(
            bbox_weight=config.training.bbox_loss_weight,
            conf_weight=config.training.conf_loss_weight,
        )

        self.train_metrics = DetectionMetrics()
        self.val_metrics = DetectionMetrics()

    def forward(self, x):
        return self.model(x)

    def _select_targets(self, predictions: torch.Tensor, targets: torch.Tensor):
        if self.config.model.output_last_only and targets.dim() == 3:
            return targets[:, -1, :]
        return targets

    def _metrics_inputs(self, predictions: torch.Tensor, targets: torch.Tensor):
        if predictions.dim() == 3:
            predictions = predictions[:, -1, :]
        if targets.dim() == 3:
            targets = targets[:, -1, :]
        pred_dict = {"bbox": predictions[:, :4], "confidence": predictions[:, 4]}
        return pred_dict, targets

    def training_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)

        targets_used = self._select_targets(predictions, targets)
        loss_dict = self.criterion(predictions, targets_used)

        self.log("train_loss", loss_dict["loss"], prog_bar=True)
        self.log("train_bbox_loss", loss_dict["bbox_loss"])
        self.log("train_conf_loss", loss_dict["conf_loss"])

        pred_dict, metric_targets = self._metrics_inputs(predictions, targets)
        self.train_metrics.update(pred_dict, metric_targets)

        return loss_dict["loss"]

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        for key, value in metrics.items():
            self.log(f"train_{key}", value)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)

        targets_used = self._select_targets(predictions, targets)
        loss_dict = self.criterion(predictions, targets_used)

        self.log("val_loss", loss_dict["loss"], prog_bar=True)
        self.log("val_bbox_loss", loss_dict["bbox_loss"])
        self.log("val_conf_loss", loss_dict["conf_loss"])

        pred_dict, metric_targets = self._metrics_inputs(predictions, targets)
        self.val_metrics.update(pred_dict, metric_targets)

        return loss_dict["loss"]

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for key, value in metrics.items():
            self.log(f"val_{key}", value, prog_bar=True)
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )

        if self.config.training.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.lr * 0.01,
            )
        elif self.config.training.lr_scheduler == "step":
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
