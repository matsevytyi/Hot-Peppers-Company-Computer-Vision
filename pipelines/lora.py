"""LoRA utilities for backbone-only adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class LoRALinear(nn.Module):
    """LoRA wrapper for a linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora = F.linear(self.dropout(x), self.lora_A)
        lora = F.linear(lora, self.lora_B)
        return base_out + lora * self.scaling

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def _matches_target_rule(module_name: str, target_rule: str) -> bool:
    lower = module_name.lower()
    if target_rule == "all_linear_except_head":
        blocked_tokens = ("head", "classifier", "fc_out", "logits")
        return not any(token in lower for token in blocked_tokens)
    return True


def inject_lora_modules(
    root: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    target_rule: str = "all_linear_except_head",
) -> List[str]:
    """Recursively replace selected nn.Linear modules by LoRALinear wrappers."""
    replaced: List[str] = []

    def _inject(module: nn.Module, prefix: str) -> None:
        for child_name, child in list(module.named_children()):
            fq_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and _matches_target_rule(fq_name, target_rule):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(fq_name)
                continue
            _inject(child, fq_name)

    _inject(root, "")
    return replaced


def freeze_module(module: nn.Module, freeze: bool) -> None:
    for param in module.parameters():
        param.requires_grad = not freeze


def configure_lora_training(
    model: nn.Module,
    *,
    freeze_neck: bool = True,
    freeze_head: bool = True,
) -> None:
    """Keep only LoRA params trainable and freeze optional top modules."""
    for name, param in model.named_parameters():
        param.requires_grad = ("lora_A" in name) or ("lora_B" in name)

    if freeze_neck and hasattr(model, "neck"):
        freeze_module(model.neck, True)
    if freeze_head and hasattr(model, "head"):
        freeze_module(model.head, True)


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if "lora_A" in key or "lora_B" in key:
            state[key] = value.detach().cpu()
    return state


def save_lora_adapters(model: nn.Module, output_path: str, metadata: Dict[str, str] | None = None) -> None:
    state = lora_state_dict(model)
    if not state:
        raise RuntimeError("No LoRA parameters found to save")
    save_file(state, output_path, metadata=metadata or {})


def load_lora_adapters(model: nn.Module, adapter_path: str, strict: bool = False) -> Tuple[List[str], List[str]]:
    state = load_file(adapter_path)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected


def collect_trainable_parameter_summary(model: nn.Module) -> Dict[str, int]:
    trainable = 0
    frozen = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable += param.numel()
        else:
            frozen += param.numel()
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
