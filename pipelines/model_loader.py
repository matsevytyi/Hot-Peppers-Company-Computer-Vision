"""Load the canonical Mamba-Vision detector from mamba-vision-ours/model.py."""

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
    model = model_cls(**kwargs)
    return model
