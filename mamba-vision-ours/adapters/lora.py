"""LoRA utilities for backbone-only adaptation.

This file contains the minimal set of LoRA helpers needed by the model for
deployment and by the training pipelines. Training pipelines will be updated
to import these helpers from here via a small shim so existing notebooks keep
working.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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

    def _resolve_domain_idx(self, domain_idx: int | None, domain_probs: torch.Tensor | None) -> int | None:
        if domain_idx is not None:
            return int(domain_idx)
        active = getattr(self, "active_domain_idx", None)
        if active is not None:
            return int(active)
        if domain_probs is None and len(self.domains) == 1:
            return 0
        return None

    def forward(self, x: torch.Tensor, domain_idx: int = None, domain_probs: torch.Tensor = None) -> torch.Tensor:
        base_out = self.base(x)

        resolved_domain_idx = self._resolve_domain_idx(domain_idx, domain_probs)
        if resolved_domain_idx is not None:
            if resolved_domain_idx < 0 or resolved_domain_idx >= len(self.domains):
                raise IndexError(
                    f"Domain index {resolved_domain_idx} is out of range for {len(self.domains)} domains"
                )
            domain = self.domains[resolved_domain_idx]
            dropped_x = self.dropout(x)
            lora = F.linear(dropped_x, self.lora_A[domain])
            lora = F.linear(lora, self.lora_B[domain])
            return base_out + lora * self.scaling

        if domain_probs is not None:
            if domain_probs.dim() != 1 or domain_probs.shape[0] != len(self.domains):
                raise ValueError(
                    "Soft routing expects domain_probs to be a 1D tensor "
                    f"with shape [{len(self.domains)}], got {tuple(domain_probs.shape)}"
                )
            dropped_x = self.dropout(x)
            lora_out = torch.zeros_like(base_out)
            for i, domain in enumerate(self.domains):
                prob = domain_probs[i]
                if prob > 0:
                    l = F.linear(dropped_x, self.lora_A[domain])
                    l = F.linear(l, self.lora_B[domain])
                    lora_out += l * prob
            return base_out + lora_out * self.scaling

        if not getattr(self, "_warned_missing_domain", False):
            print(
                "LoRALinear: no domain was selected in multi-domain mode; returning base output only."
            )
            self._warned_missing_domain = True
        return base_out

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
    """Keep LoRA params trainable and optionally unfreeze neck/head."""
    for param in model.parameters():
        param.requires_grad = False

    lora_param_count = 0
    for name, param in model.named_parameters():
        if ("lora_A" in name) or ("lora_B" in name):
            param.requires_grad = True
            lora_param_count += 1

    if lora_param_count == 0:
        raise RuntimeError(
            "configure_lora_training: no LoRA parameters found. "
            "Check LoRA injection and target_rule before training."
        )

    if hasattr(model, "neck"):
        freeze_module(model.neck, freeze_neck)
    if hasattr(model, "head"):
        freeze_module(model.head, freeze_head)


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
