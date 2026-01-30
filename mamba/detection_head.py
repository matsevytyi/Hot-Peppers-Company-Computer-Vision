"""Detection head for bounding box prediction."""
import torch.nn as nn


class DetectionHead(nn.Module):
    """Simple detection head: [x, y, w, h, confidence]."""

    def __init__(self, in_features: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(x)
