from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# simple MLP for fast domain routing
class RouterMLP(nn.Module):
    def __init__(self, hidden: int = 64, num_domains: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden), 
            nn.BatchNorm1d(hidden), # for stable training with stats
            nn.ReLU(), 
            
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            
            nn.Linear(hidden, num_domains)
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        
        stats = _image_stats(images)
        
        logits = self.net(stats)
        probs = F.softmax(logits, dim=1)
        return probs
    
# computes quick image stats
def _image_stats(self, images: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # images: (B,3,H,W), on-device
    x = images.float()
    if x.max() > 2.0:
        x = x / 255.0

    # brightness (mean over C,H,W)
    brightness = x.mean(dim=(1, 2, 3))

    # contrast (std over C,H,W)
    contrast = x.std(dim=(1, 2, 3))

    # saturation: (max-min)/(max+eps) per pixel, then mean
    mx = x.max(dim=1)[0]
    mi = x.min(dim=1)[0]
    saturation = ((mx - mi) / (mx + self.eps)).mean(dim=(1, 2))

    # high-frequency energy: Laplacian variance per image
    B, C, H, W = x.shape
    device = x.device
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=device, dtype=x.dtype)
    kernel = kernel.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    pad = 1
    hf = F.conv2d(x.reshape(B * C, 1, H, W), kernel, padding=pad, groups=C)
    hf = hf.reshape(B, C, H, W)
    hf_energy = hf.var(dim=(1, 2, 3))

    stats = torch.stack([brightness, contrast, hf_energy, saturation], dim=1)
    return stats