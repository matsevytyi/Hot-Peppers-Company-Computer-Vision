"""Mamba state-space block with LSTM fallback."""
import warnings

import torch.nn as nn

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except Exception:
    MAMBA_AVAILABLE = False
    warnings.warn("mamba-ssm not available, using LSTM fallback", RuntimeWarning)


class MambaBlockPlaceholder(nn.Module):
    """LSTM-based placeholder for Mamba (for Mac development)."""

    def __init__(self, d_model: int = 256, d_state: int = 16, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.norm(out + x)


def create_mamba_block(d_model: int = 256, d_state: int = 16, **kwargs):
    """Factory function to create Mamba or LSTM block."""
    if MAMBA_AVAILABLE:
        return Mamba(d_model=d_model, d_state=d_state, **kwargs)
    return MambaBlockPlaceholder(d_model=d_model, d_state=d_state, **kwargs)
