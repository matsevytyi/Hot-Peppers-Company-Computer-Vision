import torch
import torch.nn as nn
from typing import List

class YOLOv11Head(nn.Module):
    """YOLOv11 Detection Head with 3-scale outputs."""
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 80,
        num_scales: int = 3,
    ):
        """
        Args:
            in_channels: Number of input channels from neck
            num_classes: Number of object classes
            num_scales: Number of detection scales (default: 3)
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_scales = num_scales
        
        # Detection heads for each scale
        self.detection_heads = nn.ModuleList()
        for _ in range(num_scales):
            head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                # Output: (x, y, w, h, confidence, class_probs)
                nn.Conv2d(
                    in_channels,
                    num_classes + 5,  # 5 = x, y, w, h, confidence
                    kernel_size=1,
                    padding=0
                )
            )
            self.detection_heads.append(head)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from neck
        
        Returns:
            List of detection outputs for each scale
        """
        outputs = []
        for i, feature in enumerate(features[:self.num_scales]):
            output = self.detection_heads[i](feature)
            outputs.append(output)
        return outputs
    
    def _initialize_head_weights(self):
        """Initialize detection head with Xavier normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)