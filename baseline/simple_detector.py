"""Placeholder baseline detector (to be implemented)."""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleDetector(nn.Module):
    """A minimal baseline stub for future implementation."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dummy(x)
