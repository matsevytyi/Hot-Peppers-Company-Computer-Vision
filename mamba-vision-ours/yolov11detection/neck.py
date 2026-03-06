import torch
import torch.nn as nn
from torch import Tensor
from typing import List


class YOLONeck(nn.Module):
    """Feature Pyramid Network (FPN) style neck for multi-scale detection."""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        """
        Args:
            in_channels: List of input channel dimensions from backbone
            out_channels: Output channel dimension for all levels
        """
        super().__init__()
        self.out_channels = out_channels
        
        # Store dimensions
        self.in_channels = in_channels
        
        # Lateral layers (1x1 convs to reduce channels)
        # one for each input stage
        self.lateral_convs = nn.ModuleList()
        for in_c in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_c, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Smooth layers (3x3 convs to reduce aliasing)
        # 3 output scales
        self.smooth_convs = nn.ModuleList()
        for _ in range(3):  # 3 output scales
            self.smooth_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from backbone stages 3 & 4
                     [stage3 (320, 14, 14), stage4 (640, 7, 7)]
        
        Returns:
            List of 3 feature pyramids at scales: 28x28, 14x14, 7x7
        """
        # features = [stage3 (dim=320, 14x14), stage4 (dim=640, 7x7)]
        assert len(features) == 2, f"Expected 2 features, got {len(features)} with shape of {[f.shape for f in features]}"
        
        # lateral convolutions to both stages
        stage3_lateral = self.lateral_convs[0](features[0])  # (256, 14, 14)
        stage4_lateral = self.lateral_convs[1](features[1])  # (256, 7, 7)
        
        # Top-down path: upsample stage4 and add to stage3
        stage4_upsampled = torch.nn.functional.interpolate(
            stage4_lateral,
            size=stage3_lateral.shape[-2:],
            mode='nearest'
        )
        stage3_fused = stage4_upsampled + stage3_lateral  # (256, 14, 14)
        
        # smooth convolutions
        stage3_smooth = self.smooth_convs[1](stage3_fused)  # (256, 14, 14)
        stage4_smooth = self.smooth_convs[2](stage4_lateral)  # (256, 7, 7)
        
        # additional fine scale by upsampling stage3
        stage3_upsampled = torch.nn.functional.interpolate(
            stage3_smooth,
            scale_factor=2,
            mode='nearest'
        )  # (256, 28, 28)
        stage3_fine = self.smooth_convs[0](stage3_upsampled)  # (256, 28, 28)
        
        # 3 scales: [small_objects, medium_objects, large_objects]
        return [
            stage3_fine,    # 28x28 (small objects, stride=8)
            stage3_smooth,  # 14x14 (medium objects, stride=16)
            stage4_smooth   # 7x7 (large objects, stride=32)
        ]