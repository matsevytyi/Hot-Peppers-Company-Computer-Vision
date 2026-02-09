"""Training loop utilities with pilot/full run modes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .contracts import CkptConfig, TrainSection
from .yolo_ops import MultiScaleYoloLoss


@dataclass
class EpochMetrics:
    loss: float
    obj_loss: float
    box_loss: float
    cls_loss: float


def resolve_device(device_preference: str) -> torch.device:
    pref = device_preference.lower()
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _checkpoint_payload(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: Dict[str, float]):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }


def save_checkpoint(path: str | Path, payload: Dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _create_scheduler(optimizer: torch.optim.Optimizer, train_cfg: TrainSection):
    if train_cfg.scheduler.lower() == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(1, train_cfg.epochs), eta_min=train_cfg.lr * 0.01)
    return None


def _autocast_context(device: torch.device, precision: str):
    use_amp = device.type == "cuda" and precision.lower() in {"fp16", "bf16"}
    if not use_amp:
        return torch.autocast("cpu", enabled=False)
    dtype = torch.float16 if precision.lower() == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def _run_epoch(
    *,
    model: torch.nn.Module,
    loader,
    criterion: MultiScaleYoloLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    precision: str,
    grad_clip_norm: float,
    max_steps: Optional[int],
    is_train: bool,
) -> EpochMetrics:
    if is_train:
        model.train()
    else:
        model.eval()

    running = {"loss": 0.0, "obj_loss": 0.0, "box_loss": 0.0, "cls_loss": 0.0}
    num_steps = 0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and precision.lower() == "fp16"))

    iterator = tqdm(loader, total=min(len(loader), max_steps) if max_steps else len(loader), leave=False)
    for step, (images, targets) in enumerate(iterator):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with _autocast_context(device, precision):
                outputs = model(images)
                losses = criterion(outputs, targets)
                loss_value = losses["loss"]

            if is_train and optimizer is not None:
                if scaler.is_enabled():
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    optimizer.step()

        num_steps += 1
        for key in running:
            running[key] += float(losses[key].detach().cpu())

        iterator.set_description(
            f"{'train' if is_train else 'val'} loss={running['loss'] / max(1, num_steps):.4f}"
        )

    if num_steps == 0:
        return EpochMetrics(loss=math.inf, obj_loss=math.inf, box_loss=math.inf, cls_loss=math.inf)

    return EpochMetrics(
        loss=running["loss"] / num_steps,
        obj_loss=running["obj_loss"] / num_steps,
        box_loss=running["box_loss"] / num_steps,
        cls_loss=running["cls_loss"] / num_steps,
    )


def fit_model(
    *,
    model: torch.nn.Module,
    train_loader,
    val_loader,
    train_cfg: TrainSection,
    ckpt_cfg: CkptConfig,
    num_classes: int,
    run_mode: str = "full",
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, List[EpochMetrics]]:
    """Run pilot/full training and persist top-k/last checkpoints."""
    device = resolve_device(train_cfg.device)
    model = model.to(device)

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
    scheduler = _create_scheduler(optimizer, train_cfg)
    criterion = MultiScaleYoloLoss(num_classes=num_classes)

    if run_mode not in {"pilot", "full"}:
        raise ValueError("run_mode must be one of: pilot, full")

    total_epochs = 1 if run_mode == "pilot" else train_cfg.epochs
    train_steps = train_cfg.pilot_steps if run_mode == "pilot" else None
    val_steps = train_cfg.pilot_val_steps if run_mode == "pilot" else None

    history = {"train": [], "val": []}
    best_paths: List[tuple[float, Path]] = []
    output_path = Path(ckpt_cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(total_epochs):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            precision=train_cfg.precision,
            grad_clip_norm=train_cfg.grad_clip_norm,
            max_steps=train_steps,
            is_train=True,
        )
        val_metrics = _run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            precision=train_cfg.precision,
            grad_clip_norm=train_cfg.grad_clip_norm,
            max_steps=val_steps,
            is_train=False,
        )
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if scheduler is not None:
            scheduler.step()

        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": train_metrics.loss, "val_loss": val_metrics.loss},
        )

        epoch_path = output_path.with_name(f"{output_path.stem}_epoch{epoch + 1:03d}.ckpt")
        save_checkpoint(epoch_path, payload)
        best_paths.append((val_metrics.loss, epoch_path))
        best_paths.sort(key=lambda item: item[0])

        while len(best_paths) > ckpt_cfg.save_top_k:
            _, drop_path = best_paths.pop()
            if drop_path.exists():
                drop_path.unlink()

        if ckpt_cfg.save_last:
            save_checkpoint(output_path, payload)

    return history
