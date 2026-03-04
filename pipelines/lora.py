"""Shim: delegate LoRA utilities to mamba-vision-ours/lora.py.

This module preserves the original public API so existing notebooks and
training scripts keep working. It loads the implementation from the
`mamba-vision-ours/lora.py` file at runtime and re-exports its symbols.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path
from typing import Any

# Resolve the implementation file relative to the repo layout. We assume this
# file sits in <repo>/pipelines/lora.py so the implementation is at
# <repo>/mamba-vision-ours/lora.py.
_HERE = Path(__file__).resolve()
_IMPL_PATH = _HERE.parent.parent / "mamba-vision-ours" / "lora.py"


def _load_impl_module(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("mamba_vision_ours_lora", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load LoRA implementation from: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_impl = _load_impl_module(_IMPL_PATH)

# Re-export commonly used symbols so the public API remains unchanged
LoRALinear = getattr(_impl, "LoRALinear")
inject_lora_modules = getattr(_impl, "inject_lora_modules")
configure_lora_training = getattr(_impl, "configure_lora_training")
lora_state_dict = getattr(_impl, "lora_state_dict")
save_lora_adapters = getattr(_impl, "save_lora_adapters")
load_lora_adapters = getattr(_impl, "load_lora_adapters")
collect_trainable_parameter_summary = getattr(_impl, "collect_trainable_parameter_summary")

__all__ = [
    "LoRALinear",
    "inject_lora_modules",
    "configure_lora_training",
    "lora_state_dict",
    "save_lora_adapters",
    "load_lora_adapters",
    "collect_trainable_parameter_summary",
]
