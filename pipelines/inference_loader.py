"""Shared model loading helpers for eval and inference CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch.nn as nn

from .contracts import ModelSection
from .lora import inject_lora_modules, load_lora_adapters
from .model_loader import create_model_from_config
from .training import load_checkpoint


def _resolve_repo_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path


def load_model_from_eval_entry(*, model_cfg_dict: Dict, repo_root: Path, device: str) -> Tuple[nn.Module, ModelSection]:
    """Build and load a model from one entry in eval.shared_eval.yaml."""
    section = ModelSection.from_dict(model_cfg_dict["model"])
    section.model_file = str(_resolve_repo_path(section.model_file, repo_root))
    section.moe_model_file = str(_resolve_repo_path(section.moe_model_file, repo_root))
    model = create_model_from_config(section, device=device)

    # Backward-compatible precedence: top-level eval model entry, then nested model section.
    base_checkpoint = model_cfg_dict.get("base_checkpoint") or section.base_checkpoint
    if base_checkpoint:
        base_path = _resolve_repo_path(str(base_checkpoint), repo_root)
        load_checkpoint(base_path, model)

    lora_adapter = model_cfg_dict.get("lora_adapter")
    if lora_adapter:
        lora_path = _resolve_repo_path(str(lora_adapter), repo_root)
        lora_cfg = model_cfg_dict.get("lora", {})
        inject_lora_modules(
            model.backbone,
            rank=int(lora_cfg.get("rank", 8)),
            alpha=int(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.05)),
            target_rule=str(lora_cfg.get("target_rule", "all_linear_except_head")),
        )
        missing, unexpected = load_lora_adapters(model, str(lora_path), strict=False)
        print(f"Loaded LoRA adapter: {lora_path} (missing={len(missing)}, unexpected={len(unexpected)})")

    return model, section
