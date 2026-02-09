"""Typed contracts for configs and exported dataset manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML at {path} must contain a top-level mapping")
    return payload


def _as_path(path: str | Path) -> str:
    return str(Path(path).resolve())


@dataclass
class DatasetManifest:
    dataset_name: str
    source: str
    split: str
    class_list: List[str]
    class_map: Dict[str, int]
    root_dir: str
    images_dir: str
    labels_or_annotations: str
    num_images: int
    num_instances: int
    created_at: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_name=str(payload["dataset_name"]),
            source=str(payload["source"]),
            split=str(payload["split"]),
            class_list=list(payload["class_list"]),
            class_map={str(k): int(v) for k, v in dict(payload["class_map"]).items()},
            root_dir=str(payload["root_dir"]),
            images_dir=str(payload["images_dir"]),
            labels_or_annotations=str(payload["labels_or_annotations"]),
            num_images=int(payload["num_images"]),
            num_instances=int(payload["num_instances"]),
            created_at=str(payload["created_at"]),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetManifest":
        import json

        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "source": self.source,
            "split": self.split,
            "class_list": self.class_list,
            "class_map": self.class_map,
            "root_dir": self.root_dir,
            "images_dir": self.images_dir,
            "labels_or_annotations": self.labels_or_annotations,
            "num_images": self.num_images,
            "num_instances": self.num_instances,
            "created_at": self.created_at,
        }

    def save_json(self, path: str | Path) -> None:
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def create(
        cls,
        *,
        dataset_name: str,
        source: str,
        split: str,
        class_list: List[str],
        class_map: Dict[str, int],
        root_dir: str | Path,
        images_dir: str | Path,
        labels_or_annotations: str | Path,
        num_images: int,
        num_instances: int,
    ) -> "DatasetManifest":
        return cls(
            dataset_name=dataset_name,
            source=source,
            split=split,
            class_list=class_list,
            class_map=class_map,
            root_dir=_as_path(root_dir),
            images_dir=_as_path(images_dir),
            labels_or_annotations=_as_path(labels_or_annotations),
            num_images=num_images,
            num_instances=num_instances,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


@dataclass
class CkptConfig:
    output_path: str
    save_top_k: int = 3
    save_last: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CkptConfig":
        return cls(
            output_path=str(payload["output_path"]),
            save_top_k=int(payload.get("save_top_k", 3)),
            save_last=bool(payload.get("save_last", True)),
        )


@dataclass
class TrainSection:
    epochs: int = 30
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    pilot_steps: int = 5
    pilot_val_steps: int = 2
    precision: str = "fp16"
    device: str = "cuda"
    image_size: int = 640
    grad_clip_norm: float = 1.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainSection":
        return cls(
            epochs=int(payload.get("epochs", 30)),
            batch_size=int(payload.get("batch_size", 8)),
            num_workers=int(payload.get("num_workers", 4)),
            lr=float(payload.get("lr", 1e-4)),
            weight_decay=float(payload.get("weight_decay", 1e-4)),
            scheduler=str(payload.get("scheduler", "cosine")),
            pilot_steps=int(payload.get("pilot_steps", 5)),
            pilot_val_steps=int(payload.get("pilot_val_steps", 2)),
            precision=str(payload.get("precision", "fp16")),
            device=str(payload.get("device", "cuda")),
            image_size=int(payload.get("image_size", 640)),
            grad_clip_norm=float(payload.get("grad_clip_norm", 1.0)),
        )


@dataclass
class ModelSection:
    backbone: str = "mamba_vision_T2"
    num_classes: int = 8
    pretrained: bool = True
    checkpoint_path: str = ""
    base_checkpoint: str = ""
    model_file: str = "mamba-vision-ours/model.py"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelSection":
        return cls(
            backbone=str(payload.get("backbone", "mamba_vision_T2")),
            num_classes=int(payload.get("num_classes", 8)),
            pretrained=bool(payload.get("pretrained", True)),
            checkpoint_path=str(payload.get("checkpoint_path", "")),
            base_checkpoint=str(payload.get("base_checkpoint", "")),
            model_file=str(payload.get("model_file", "mamba-vision-ours/model.py")),
        )


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_rule: str = "all_linear_except_head"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LoRAConfig":
        return cls(
            rank=int(payload.get("rank", 8)),
            alpha=int(payload.get("alpha", 16)),
            dropout=float(payload.get("dropout", 0.05)),
            target_rule=str(payload.get("target_rule", "all_linear_except_head")),
        )


@dataclass
class FreezeConfig:
    backbone_base: bool = False
    neck: bool = False
    head: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FreezeConfig":
        return cls(
            backbone_base=bool(payload.get("backbone_base", False)),
            neck=bool(payload.get("neck", False)),
            head=bool(payload.get("head", False)),
        )


@dataclass
class TrainConfig:
    run_name: str
    data: Dict[str, Any]
    model: ModelSection = field(default_factory=ModelSection)
    train: TrainSection = field(default_factory=TrainSection)
    ckpt: CkptConfig = field(default_factory=lambda: CkptConfig(output_path="checkpoints/model.ckpt"))
    lora: Optional[LoRAConfig] = None
    freeze: FreezeConfig = field(default_factory=FreezeConfig)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainConfig":
        run_name = str(payload.get("run_name", "unnamed_run"))
        data = dict(payload.get("data", {}))
        model = ModelSection.from_dict(dict(payload.get("model", {})))
        train = TrainSection.from_dict(dict(payload.get("train", {})))
        ckpt = CkptConfig.from_dict(dict(payload.get("ckpt", {"output_path": "checkpoints/model.ckpt"})))
        lora_payload = payload.get("lora")
        lora = LoRAConfig.from_dict(dict(lora_payload)) if isinstance(lora_payload, dict) else None
        freeze = FreezeConfig.from_dict(dict(payload.get("freeze", {})))
        return cls(run_name=run_name, data=data, model=model, train=train, ckpt=ckpt, lora=lora, freeze=freeze)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(_load_yaml(path))


@dataclass
class EvalConfig:
    eval: Dict[str, Any]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvalConfig":
        return cls(eval=dict(payload.get("eval", {})))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EvalConfig":
        return cls.from_dict(_load_yaml(path))
