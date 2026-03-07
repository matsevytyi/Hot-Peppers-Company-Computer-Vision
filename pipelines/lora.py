"""Backward-compatible LoRA shim for pipelines and notebooks.

The canonical LoRA implementation now lives at:
`mamba-vision-ours/adapters/lora.py`

This module keeps the historic `pipelines.lora` API stable.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from typing import List, Sequence

_HERE = Path(__file__).resolve()
_IMPL_PATH = _HERE.parent.parent / "mamba-vision-ours" / "adapters" / "lora.py"
_DEFAULT_DOMAIN = "default"


def _load_impl_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("mamba_vision_ours_lora", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LoRA implementation from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_impl = _load_impl_module(_IMPL_PATH)

LoRALinear = getattr(_impl, "LoRALinear")
_inject_impl = getattr(_impl, "inject_lora_modules")
configure_lora_training = getattr(_impl, "configure_lora_training")
lora_state_dict = getattr(_impl, "lora_state_dict")
save_lora_adapters = getattr(_impl, "save_lora_adapters")
load_lora_adapters = getattr(_impl, "load_lora_adapters")
collect_trainable_parameter_summary = getattr(_impl, "collect_trainable_parameter_summary")


def inject_lora_modules(
    root,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_rule="all_linear_except_head",
    domains: Sequence[str] | None = None,
) -> List[str]:
    """Inject LoRA wrappers into selected linear layers.

    `pipelines` training/eval historically operated in a single-adapter mode.
    We keep that behavior by defaulting to one synthetic domain.
    """
    resolved_domains = list(domains) if domains is not None else [_DEFAULT_DOMAIN]
    return _inject_impl(
        root,
        domains=resolved_domains,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_rule=target_rule,
    )

__all__ = [
    "LoRALinear",
    "inject_lora_modules",
    "configure_lora_training",
    "lora_state_dict",
    "save_lora_adapters",
    "load_lora_adapters",
    "collect_trainable_parameter_summary",
]
