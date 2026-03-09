"""Utilities for frame-level video inference and rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import torch
import yaml


def load_class_names(classes_config_path: Path) -> List[str]:
    with open(classes_config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    class_list = payload.get("class_list", [])
    if not isinstance(class_list, list) or not class_list:
        raise ValueError(f"`class_list` is missing or empty in {classes_config_path}")
    return [str(name) for name in class_list]


def rescale_boxes_xyxy(
    boxes_xyxy: torch.Tensor,
    *,
    from_width: int,
    from_height: int,
    to_width: int,
    to_height: int,
) -> torch.Tensor:
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy

    scale_x = float(to_width) / float(max(from_width, 1))
    scale_y = float(to_height) / float(max(from_height, 1))
    scaled = boxes_xyxy.clone()
    scaled[:, 0] = scaled[:, 0] * scale_x
    scaled[:, 2] = scaled[:, 2] * scale_x
    scaled[:, 1] = scaled[:, 1] * scale_y
    scaled[:, 3] = scaled[:, 3] * scale_y
    return scaled


def infer_domain_names(*, model: object, model_cfg_dict: Dict, num_domains: int) -> List[str]:
    domains_attr = getattr(model, "domains", None)
    if isinstance(domains_attr, list) and len(domains_attr) >= num_domains:
        return [str(name) for name in domains_attr[:num_domains]]

    model_payload = model_cfg_dict.get("model", {})
    adapters = model_payload.get("moe_adapters", {})
    if isinstance(adapters, dict) and adapters:
        names = [str(name) for name in adapters.keys()]
        if len(names) >= num_domains:
            return names[:num_domains]

    return [f"domain_{idx}" for idx in range(num_domains)]


def format_router_overlay(router_probs: Optional[torch.Tensor], domain_names: Sequence[str]) -> Optional[str]:
    if not torch.is_tensor(router_probs):
        return None
    if router_probs.ndim != 2 or router_probs.shape[0] == 0:
        return None

    probs = router_probs[0].detach().float().cpu()
    if probs.numel() == 0:
        return None

    top_idx = int(probs.argmax().item())
    conf = float(probs[top_idx].item())
    domain_name = domain_names[top_idx] if top_idx < len(domain_names) else f"domain_{top_idx}"
    return f"router: {domain_name} ({conf:.2f})"


def _color_for_label(label_id: int) -> tuple[int, int, int]:
    base = int(label_id) + 1
    return ((37 * base) % 255, (17 * base) % 255, (29 * base) % 255)


def draw_detections(
    frame_bgr,
    *,
    boxes_xyxy: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: Sequence[str],
    router_text: Optional[str] = None,
) -> None:
    h, w = frame_bgr.shape[:2]
    boxes = boxes_xyxy.detach().cpu()
    score_vals = scores.detach().cpu()
    label_vals = labels.detach().cpu()

    for idx in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[idx].tolist()
        x1 = int(max(0, min(w - 1, round(x1))))
        y1 = int(max(0, min(h - 1, round(y1))))
        x2 = int(max(0, min(w - 1, round(x2))))
        y2 = int(max(0, min(h - 1, round(y2))))
        label_id = int(label_vals[idx].item())
        score = float(score_vals[idx].item())
        cls_name = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        color = _color_for_label(label_id)

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        text = f"{cls_name}: {score:.2f}"
        cv2.putText(
            frame_bgr,
            text,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    if router_text:
        cv2.putText(
            frame_bgr,
            router_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def prepare_frame_tensor(frame_bgr, *, image_size: int, device: torch.device) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(frame_rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0).to(device)


def pick_eval_model_entry(models: Iterable[Dict], model_name: str) -> Dict:
    for entry in models:
        if str(entry.get("name", "")) == model_name:
            return dict(entry)
    raise KeyError(f"Model '{model_name}' not found in eval.models")
