"""LoRA utilities for backbone-only adaptation.

This file contains the minimal set of LoRA helpers needed by the model for
deployment and by the training pipelines. Training pipelines will be updated
to import these helpers from here via a small shim so existing notebooks keep
working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


class LoRALinear(nn.Module):
    """Multi-Adapter LoRA wrapper for a linear layer."""

    def __init__(self, base: nn.Linear, domains: List[str], rank: int = 8, alpha: int = 16, dropout: float = 0.05):
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

        self.domains = domains
        
        # Create a dictionary of parameters for EACH domain
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        
        for domain in domains:
            A = nn.Parameter(torch.zeros(rank, base.in_features))
            B = nn.Parameter(torch.zeros(base.out_features, rank))
            nn.init.kaiming_uniform_(A, a=5 ** 0.5)
            nn.init.zeros_(B)
            self.lora_A[domain] = A
            self.lora_B[domain] = B

    def forward(self, x: torch.Tensor, domain_idx: int = None, domain_probs: torch.Tensor = None) -> torch.Tensor:
        base_out = self.base(x)
        
        # If no routing is provided, just return base (shouldn't happen in MoE)
        if domain_idx is None and domain_probs is None:
            return base_out
            
        dropped_x = self.dropout(x)
        
        # HARD ROUTING: If a specific domain index is given
        if domain_idx is not None:
            domain = self.domains[domain_idx]
            lora = F.linear(dropped_x, self.lora_A[domain])
            lora = F.linear(lora, self.lora_B[domain])
            return base_out + lora * self.scaling
            
        # SOFT ROUTING: If probabilities are given (optional, for soft-MoE)
        if domain_probs is not None:
            lora_out = 0
            for i, domain in enumerate(self.domains):
                prob = domain_probs[i]
                if prob > 0:
                    l = F.linear(dropped_x, self.lora_A[domain])
                    l = F.linear(l, self.lora_B[domain])
                    lora_out += l * prob
            return base_out + lora_out * self.scaling

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
    # allow explicit lists/paths passed as sequence-like strings handled by caller
    return True


def inject_lora_modules(
    root: nn.Module,
    domains: List[str],
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
                setattr(module, child_name, LoRALinear(child, domains=domains, rank=rank, alpha=alpha, dropout=dropout))
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
    """Keep only LoRA params trainable and optionally freeze neck/head."""
    lo_ra_found = False
    for name, param in model.named_parameters():
        if ("lora_A" in name) or ("lora_B" in name):
            lo_ra_found = True
            param.requires_grad = True
        else:
            if lo_ra_found:
                param.requires_grad = False

    if not lo_ra_found:
        # nothing to train; leave flags unchanged for callers that want to handle it
        print("configure_lora_training: no LoRA parameters found; grads left untouched")
        return

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
