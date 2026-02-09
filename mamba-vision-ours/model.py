"""Initial model: YOLOv11 Detection Head with Mamba-Vision Backbone."""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn

try:
    from .yolov11detection import YOLOv11Head
    from .yolov11detection import YOLONeck
except ImportError:
    from yolov11detection import YOLOv11Head
    from yolov11detection import YOLONeck

try:
    # official approach
    # mambavision installed as 'cd to mambavision' and 'pip install -e MambaVisionReengineering'
    from mambavision import create_model 
    torch.serialization.add_safe_globals([argparse.Namespace])
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    local_pkg_root = repo_root / "MambaVisionReengineering"
    if str(local_pkg_root) not in sys.path:
        sys.path.insert(0, str(local_pkg_root))
    try:
        from mambavision import create_model
        torch.serialization.add_safe_globals([argparse.Namespace])
    except ImportError as exc:
        raise ImportError(
            "Could not import 'mambavision'. Initialize submodule and install it "
            "(`git submodule update --init --recursive` + `pip install -e MambaVisionReengineering`)."
        ) from exc

class MambaVisionOurs(nn.Module):
    """MambaVision-based detector with frozen backbone and YOLOv11 detection head."""
    
    def __init__(
        self, 
        device="cuda",
        model_type="mamba_vision_T2", 
        num_output_classes=80, 
        pretrained=True,
        checkpoint_path="mambavision_tiny2_1k.pth.tar"
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
            #checkpoint_path=checkpoint_path
        )

        self.device=device
        
        # MambaVision-T Stage dims: [80, 160, 320, 640]
        self.backbone_dims = [80, 160, 320, 640]

        if pretrained:

            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get('model', checkpoint)

            # remove head weights
            backbone_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
            print(f"Loaded pretrained backbone tensors: {len(backbone_state_dict)}")

            missing, unexpected = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        
        # Detection Neck (FPN/PAN style)
        # Using stages 3, 4 (320, 640 channels) for multi-scale detection
        self.neck = YOLONeck(in_channels=self.backbone_dims[2:], out_channels=256)
        
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
        _, stage_features = self.backbone.forward_features(x)
        #print(np.shape(stage_features))
        
        # features[0]=Stage1, features[1]=Stage2, features[2]=Stage3, features[3]=Stage4
        # pass the last N stages to the YOLO neck
        neck_out = self.neck(stage_features[2:4]) 
        
        head_out = self.head(neck_out)
        
        return head_out
    

# utils
def check_shapes(model: MambaVisionOurs, input_tensor: torch.Tensor):
    """Check dimensions through backbone, neck, and head."""
    model.eval()
    
    with torch.no_grad():
        # Backbone
        print("\n=== BACKBONE ===")
        backbone_out, features = model.backbone.forward_features(input_tensor)
        print(f"Backbone output (pooled logits/features): {backbone_out.shape if backbone_out is not None else None}")
        for i, f in enumerate(features):
            print(f"Feature {i} shape: {f.shape}")
        
        # Neck
        print("\n=== NECK ===")
        neck_out = model.neck(features[2:4])  # stages 3,4
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

    # pretrained=False to check shapes or forward pass or head/neck integration
    # pretrained=True to run actual code, requires GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = MambaVisionOurs(
        model_type="mamba_vision_T2",
        device=device,
        num_output_classes=80,
        pretrained=False,
    ).to(device)

    
    print("\nModel Summary:")
    try:
        from torchinfo import summary

        summary(model.backbone, input_size=(20, 3, 224, 224))
    except Exception as exc:
        print(f"Skipping torchinfo summary: {exc}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(20, 3, 224, 224).to(device)

    check_shapes(model, x)
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Number of detection scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Scale {i + 1}: {out.shape}")
