"""Initial model: YOLOv11 Detection Head with Mamba-Vision Backbone."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import List
import sys

from yolov11detection import YOLOv11Head
from yolov11detection import YOLONeck


# submodule import playaround
_current_dir = Path(__file__).resolve().parent
_reengine_dir = _current_dir.parent / "MambaVisionReengineering"
if str(_reengine_dir) not in sys.path:
    sys.path.insert(0, str(_reengine_dir))

try:
    # official approach
    from MambaVisionReengineering.mambavision import create_model 
except ImportError:
    raise ImportError("Check if 'mambavision' package/folder is in MambaVisionReengineering")

class MambaVisionOurs(nn.Module):
    """MambaVision-based detector with frozen backbone and YOLOv11 detection head."""
    
    def __init__(
        self, 
        model_type="mamba_vision_T", 
        num_output_classes=80, 
        pretrained=True,
        checkpoint_path=''
    ):
        """
        Args:
            model_type – which checkpoint you want to use
            num_output_classes – number of classes to be used on training stage
            pretrained, checkpoint_path – path to saved checkpoint if pretrained=True
        """
        super().__init__()
        self.backbone = create_model(
            model_type, 
            pretrained=pretrained, 
            num_classes=0,
            checkpoint_path=checkpoint_path
        )
        
        # MambaVision-T Stage dims: [80, 160, 320, 640]
        self.backbone_dims = [80, 160, 320, 640]
        
        # Detection Neck (FPN/PAN style)
        # Using stages 2, 3, 4 (160, 320, 640 channels) for multi-scale detection
        self.neck = YOLONeck(in_channels=self.backbone_dims[1:], out_channels=256)
        
        # Detection Head
        self.head = YOLOv11Head(in_channels=256, num_classes=num_output_classes)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through MambaYOLO.
        
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            List of detection outputs at 3 scales
        """
        # in paper mamba vision returns (avg_pool_out, list_of_stage_features)
        # when num_classes=0 or when used as a feature extractor
        _, features = self.backbone(x)
        
        # features[0]=Stage1, features[1]=Stage2, features[2]=Stage3, features[3]=Stage4
        # pass the last N stages to the YOLO neck
        neck_out = self.neck(features[1:]) 
        
        head_out = self.head(neck_out)
        
        return head_out
    

# utils
def check_shapes(model: MambaVisionOurs, input_tensor: torch.Tensor):
    """Check dimensions through backbone, neck, and head."""
    model.eval()
    
    with torch.no_grad():
        # Backbone
        print("\n=== BACKBONE ===")
        backbone_out, features = model.backbone(input_tensor)
        print(f"Backbone output (pooled logits/features): {backbone_out.shape if backbone_out is not None else None}")
        for i, f in enumerate(features):
            print(f"Feature {i} shape: {f.shape}")
        
        # Neck
        print("\n=== NECK ===")
        neck_out = model.neck(features[1:])  # stages 2,3,4
        for i, n in enumerate(neck_out):
            print(f"Neck output scale {i} shape: {n.shape}")
        
        # Head
        print("\n=== HEAD ===")
        head_out = model.head(neck_out)
        for i, h in enumerate(head_out):
            print(f"Head output scale {i} shape: {h.shape}")
        
        # test match channel dim of neck vs head input
        print("\n=== CHECKS ===")
        for i, n in enumerate(neck_out):
            assert n.shape[1] == model.head.in_channels, \
                f"Neck output channels {n.shape[1]} != head expected {model.head.in_channels}"
        print("Channel dimensions between neck and head match ")
    

if __name__ == "__main__":

    print("Creating MambaYOLO model...")
    model = MambaVisionOurs(
        model_type="mamba_vision_T",
        num_output_classes=80,
        pretrained=True
    )
    
    check_shapes(model)
    
    print("\nModel Summary:")
    summary = model.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Number of detection scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Scale {i + 1}: {out.shape}")
