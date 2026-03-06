"""Dataset and dataloader helpers for COCO-style detection exports."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .constants import canonicalize_label
from .contracts import DatasetManifest


@dataclass
class SampleRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    boxes: List[List[float]]
    labels: List[int]


class COCODetectionDataset(Dataset):
    """Simple COCO JSON parser that emits normalized boxes for YOLO-style losses."""

    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        class_map: Dict[str, int],
        image_size: int = 640,
        max_samples: Optional[int] = None,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.class_map = class_map
        self.image_size = image_size
        self.samples = self._load_records(max_samples=max_samples)

    def _load_records(self, max_samples: Optional[int]) -> List[SampleRecord]:
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        categories = {
            int(cat["id"]): canonicalize_label(str(cat["name"]))
            for cat in payload.get("categories", [])
        }

        anns_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ann in payload.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)

        records: List[SampleRecord] = []
        for img in payload.get("images", []):
            image_id = int(img["id"])
            file_name = str(img["file_name"])
            width = int(img["width"])
            height = int(img["height"])

            boxes: List[List[float]] = []
            labels: List[int] = []
            for ann in anns_by_image.get(image_id, []):
                if int(ann.get("iscrowd", 0)) == 1:
                    continue
                category_id = int(ann["category_id"])
                class_name = categories.get(category_id)
                if class_name is None or class_name not in self.class_map:
                    continue

                x, y, w, h = [float(v) for v in ann["bbox"]]
                cx = (x + w / 2.0) / width
                cy = (y + h / 2.0) / height
                wn = w / width
                hn = h / height

                boxes.append(
                    [
                        max(0.0, min(1.0, cx)),
                        max(0.0, min(1.0, cy)),
                        max(0.001, min(1.0, wn)),
                        max(0.001, min(1.0, hn)),
                    ]
                )
                labels.append(self.class_map[class_name])

            records.append(
                SampleRecord(
                    image_id=image_id,
                    file_name=file_name,
                    width=width,
                    height=height,
                    boxes=boxes,
                    labels=labels,
                )
            )

        if max_samples is not None:
            records = records[: max(0, int(max_samples))]
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        record = self.samples[index]
        image_path = self.images_dir / record.file_name
        image = Image.open(image_path).convert("RGB")
        image = TF.resize(image, [self.image_size, self.image_size], antialias=True)
        image_tensor = TF.to_tensor(image)

        boxes = torch.tensor(record.boxes, dtype=torch.float32) if record.boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = (
            torch.tensor(record.labels, dtype=torch.long)
            if record.labels
            else torch.zeros((0,), dtype=torch.long)
        )

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([record.image_id], dtype=torch.long),
            "orig_size": torch.tensor([record.height, record.width], dtype=torch.long),
        }
        return image_tensor, target


def collate_detection_batch(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


def build_dataloader(
    manifest: DatasetManifest,
    *,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_samples: Optional[int] = None,
) -> DataLoader:
    dataset = COCODetectionDataset(
        images_dir=manifest.images_dir,
        annotations_path=manifest.labels_or_annotations,
        class_map=manifest.class_map,
        image_size=image_size,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_detection_batch,
    )
