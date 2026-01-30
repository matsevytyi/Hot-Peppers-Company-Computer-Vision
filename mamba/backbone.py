"""Feature extraction backbones."""
import torch.nn as nn
import timm


class MobileViTBackbone(nn.Module):
    """MobileViT backbone for lightweight feature extraction."""

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        self.model = timm.create_model("mobilevit_s", pretrained=pretrained, num_classes=0)
        self.out_channels = self.model.num_features

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model.forward_features(x)


class ResNetBackbone(nn.Module):
    """ResNet50 backbone."""

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        self.model = timm.create_model("resnet50", pretrained=pretrained, num_classes=0)
        self.out_channels = self.model.num_features

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model.forward_features(x)


def create_backbone(name: str = "mobilevit_s", **kwargs) -> nn.Module:
    """Factory function for backbones."""
    name_lower = name.lower()
    if "mobilevit" in name_lower:
        return MobileViTBackbone(**kwargs)
    if "resnet" in name_lower:
        return ResNetBackbone(**kwargs)
    raise ValueError(f"Unknown backbone: {name}")
