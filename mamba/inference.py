"""Inference utilities for the Mamba UAV detector."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from .config import Config
from .trainer import MambaDetectorModule


def _preprocess_image(image_path: str, img_size: int) -> torch.Tensor:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)


def load_model(checkpoint_path: str, config: Config | None = None, device: str = "cpu"):
    """Load a trained model from a Lightning checkpoint."""
    if config is None:
        config = Config()
    model = MambaDetectorModule.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model.to(device)
    return model


def run_sequence_inference(
    model: MambaDetectorModule,
    image_paths: List[str],
    img_size: int = 640,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference on a sequence of images.

    Returns:
        bbox: [B, 4] or [B, T, 4]
        confidence: [B] or [B, T]
    """
    frames = [_preprocess_image(p, img_size) for p in image_paths]
    sequence = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model.model.predict(sequence)

    return pred["bbox"], pred["confidence"]


def load_sequence_from_dir(
    image_dir: str, suffix: str = ".jpg", limit: int | None = None
) -> List[str]:
    """Load a sorted list of image paths from a directory."""
    paths = sorted([p for p in Path(image_dir).iterdir() if p.suffix.lower() == suffix])
    if limit is not None:
        paths = paths[:limit]
    return [str(p) for p in paths]
