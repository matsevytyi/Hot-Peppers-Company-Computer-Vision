"""Train base or LoRA Mamba-Vision pipeline from YAML config."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from pipelines.coco_dataset import build_dataloader  # noqa: E402
from pipelines.contracts import DatasetManifest, TrainConfig  # noqa: E402
from pipelines.dependencies import assert_required_packages  # noqa: E402
from pipelines.lora import (  # noqa: E402
    collect_trainable_parameter_summary,
    configure_lora_training,
    inject_lora_modules,
    save_lora_adapters,
)
from pipelines.model_loader import create_model_from_config  # noqa: E402
from pipelines.training import fit_model, load_checkpoint, resolve_device  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    assert_required_packages(["torch", "torchvision", "yaml", "safetensors"])

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    cfg = TrainConfig.from_yaml(config_path)
    cfg.model.model_file = str((REPO_ROOT / cfg.model.model_file).resolve())
    ckpt_output = Path(cfg.ckpt.output_path)
    if not ckpt_output.is_absolute():
        ckpt_output = (REPO_ROOT / ckpt_output).resolve()
    cfg.ckpt.output_path = str(ckpt_output)

    device = resolve_device(cfg.train.device)
    model = create_model_from_config(cfg.model, device=str(device))

    if cfg.model.base_checkpoint:
        base_ckpt = Path(cfg.model.base_checkpoint)
        if not base_ckpt.is_absolute():
            base_ckpt = (REPO_ROOT / base_ckpt).resolve()
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Missing base checkpoint: {base_ckpt}")
        load_checkpoint(base_ckpt, model)
        print(f"Loaded base checkpoint: {base_ckpt}")

    if cfg.lora is not None:
        replaced = inject_lora_modules(
            model.backbone,
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            dropout=cfg.lora.dropout,
            target_rule=cfg.lora.target_rule,
        )
        configure_lora_training(
            model,
            freeze_neck=cfg.freeze.neck,
            freeze_head=cfg.freeze.head,
        )
        print(f"Injected LoRA into {len(replaced)} linear layers")
    else:
        if cfg.freeze.backbone_base:
            for param in model.backbone.parameters():
                param.requires_grad = False
        if cfg.freeze.neck and hasattr(model, "neck"):
            for param in model.neck.parameters():
                param.requires_grad = False
        if cfg.freeze.head and hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad = False

    summary = collect_trainable_parameter_summary(model)
    print("Parameter summary:", summary)

    manifest_train_path = Path(cfg.data["manifest_train"])
    if not manifest_train_path.is_absolute():
        manifest_train_path = (REPO_ROOT / manifest_train_path).resolve()
    manifest_train = DatasetManifest.from_json(manifest_train_path)

    manifest_val_path = Path(cfg.data["manifest_val"])
    if not manifest_val_path.is_absolute():
        manifest_val_path = (REPO_ROOT / manifest_val_path).resolve()
    manifest_val = DatasetManifest.from_json(manifest_val_path)

    train_loader = build_dataloader(
        manifest_train,
        image_size=cfg.train.image_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        max_samples=cfg.data.get("max_samples_train"),
    )
    val_loader = build_dataloader(
        manifest_val,
        image_size=cfg.train.image_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        max_samples=cfg.data.get("max_samples_val"),
    )

    history = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=cfg.train,
        ckpt_cfg=cfg.ckpt,
        num_classes=cfg.model.num_classes,
        run_mode=args.run_mode,
    )

    print("Training completed")
    print("Last train loss:", history["train"][-1].loss if history["train"] else None)
    print("Last val loss:", history["val"][-1].loss if history["val"] else None)

    if cfg.lora is not None:
        output_path = cfg.data.get("lora_output_path")
        if not output_path:
            raise RuntimeError("LoRA run requires `data.lora_output_path` in config")
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_lora_adapters(
            model,
            output_path=str(output_path),
            metadata={"run_name": cfg.run_name, "base_checkpoint": cfg.model.base_checkpoint},
        )
        print(f"Saved LoRA adapters: {output_path}")


if __name__ == "__main__":
    main()
