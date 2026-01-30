"""Complete Mamba UAV Detector."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .backbone import create_backbone
from .detection_head import DetectionHead
from .mamba_block import create_mamba_block


class MambaUAVDetector(nn.Module):
    """Fixed-wing UAV detector with Mamba temporal modeling."""

    def __init__(
        self,
        backbone: str = "mobilevit_s",
        d_model: int = 256,
        d_state: int = 16,
        mamba_layers: int = 4,
        output_last_only: bool = True,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.output_last_only = output_last_only

        self.backbone = create_backbone(
            backbone,
            pretrained=pretrained,
            freeze=freeze_backbone,
        )
        backbone_channels = self.backbone.out_channels

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_proj = nn.Linear(backbone_channels, d_model)

        self.mamba_layers = nn.ModuleList(
            [create_mamba_block(d_model=d_model, d_state=d_state) for _ in range(mamba_layers)]
        )

        self.detection_head = DetectionHead(in_features=d_model, hidden_dim=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, T, 3, H, W]

        Returns:
            [B, 5] if output_last_only else [B, T, 5]
        """
        bsz, seq_len, channels, height, width = x.shape
        x = x.reshape(bsz * seq_len, channels, height, width)

        features = self.backbone(x)
        if features.dim() == 4:
            features = self.pool(features).flatten(1)

        features = self.feature_proj(features)
        features = features.view(bsz, seq_len, -1)

        for mamba in self.mamba_layers:
            features = mamba(features)

        if self.output_last_only:
            return self.detection_head(features[:, -1, :])
        return self.detection_head(features)

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference with post-processing."""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)

        if self.output_last_only:
            return {
                "bbox": predictions[:, :4],
                "confidence": predictions[:, 4],
            }
        return {
            "bbox": predictions[:, :, :4],
            "confidence": predictions[:, :, 4],
        }
