"""Load the canonical Mamba-Vision detector and optional MoE wrapper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Type

from .contracts import ModelSection


def _load_module_from_file(module_name: str, model_file: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not create import spec for: {model_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_mamba_vision_class(model_file: str | Path) -> Type:
    path = Path(model_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Mamba-Vision model file not found: {path}")

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module = _load_module_from_file("mamba_vision_ours_runtime", path)
    if not hasattr(module, "MambaVisionOurs"):
        raise AttributeError(f"{path} does not export MambaVisionOurs")
    return module.MambaVisionOurs

def load_moe_wrapper_class(model_file: str | Path) -> Type:
    """Load MoEMambaVision from moe_model.py next to base_model.py."""
    path = Path(model_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"MOE wrapper for Mamba-Vision model file not found: {path}")

    module = _load_module_from_file("mamba_vision_moe_runtime", path)
    if not hasattr(module, "MoEMambaVision"):
        raise AttributeError(f"{path} does not export MoEMambaVision")
    return module.MoEMambaVision

def create_model_from_config(model_cfg: ModelSection, device: str):
    model_cls = load_mamba_vision_class(model_cfg.model_file)
    kwargs = {
        "device": device,
        "model_type": model_cfg.backbone,
        "num_output_classes": model_cfg.num_classes,
        "pretrained": model_cfg.pretrained,
    }
    if model_cfg.checkpoint_path:
        kwargs["checkpoint_path"] = model_cfg.checkpoint_path
        
    base_model = model_cls(**kwargs)

    # MoE
    if model_cfg.moe_adapters:  # already a dict from contracts.ModelSection
        print("MoE configuration detected. Wrapping base model...")
        moe_cls = load_moe_wrapper_class(model_cfg.moe_model_file)
        adapter_paths = dict(model_cfg.moe_adapters)

        model = moe_cls(
            base_model=base_model,
            adapter_paths=adapter_paths,
            target_rule=getattr(model_cfg, "lora_target_rule", "all_linear_except_head")
            # router_weights_path=model_cfg.moe_router_weights
        )
        return model

    print("Base (no-MoE) config detected. Proceeding with base model")
    return base_model
