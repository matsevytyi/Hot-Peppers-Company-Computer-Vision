"""Sequence dataset for Mamba model."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from shared.parser import extract_bboxes, parse_voc_xml
from shared.transforms import get_train_transforms, get_val_transforms


class MMFWUAVSequenceDataset(Dataset):
    """Dataset that returns sequences of frames for temporal modeling."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        sensor_type: str = "Zoom",
        view: str = "Top_Down",
        sequence_length: int = 10,
        stride: int = 5,
        img_size: int = 640,
        transform=None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.sensor_type = sensor_type
        self.view = view
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_size = img_size

        split_file = self.data_root.parent / "splits" / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}. "
                "Run scripts/prepare_data.py to generate splits."
            )
        with open(split_file, "r", encoding="utf-8") as f:
            self.split_data = json.load(f)

        self.sequences = self._create_sequences()

        if transform is None:
            self.transform = (
                get_train_transforms(img_size) if split == "train" else get_val_transforms(img_size)
            )
        else:
            self.transform = transform

    def _create_sequences(self) -> List[List[Dict]]:
        """Group frames into temporal sequences."""
        sequences: List[List[Dict]] = []

        for uav_type in self.split_data.get("uav_types", []):
            img_dir = self.data_root / uav_type / self.view / f"{self.sensor_type}_Imgs"
            ann_dir = self.data_root / uav_type / self.view / f"{self.sensor_type}_Anns"

            if not img_dir.exists():
                continue

            frames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])

            for i in range(0, len(frames) - self.sequence_length + 1, self.stride):
                seq_frames = frames[i : i + self.sequence_length]
                sequence: List[Dict] = []

                for frame_name in seq_frames:
                    img_path = img_dir / frame_name
                    ann_path = ann_dir / f"{Path(frame_name).stem}.xml"

                    if ann_path.exists():
                        sequence.append(
                            {
                                "image_path": str(img_path),
                                "annotation_path": str(ann_path),
                                "uav_type": uav_type,
                            }
                        )

                if len(sequence) == self.sequence_length:
                    sequences.append(sequence)

        return sequences

    def __len__(self) -> int:  # noqa: D401
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return images [T, C, H, W] and targets [T, 5]."""
        sequence = self.sequences[idx]

        images: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        for frame_data in sequence:
            img = cv2.imread(frame_data["image_path"])
            if img is None:
                raise FileNotFoundError(f"Image not found: {frame_data['image_path']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            annotation = parse_voc_xml(frame_data["annotation_path"])
            bboxes = extract_bboxes(annotation)

            if bboxes:
                bbox = bboxes[0]
                bbox_list = [[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]]
                class_labels = [1]
            else:
                bbox_list = []
                class_labels = []

            transformed = self.transform(
                image=img,
                bboxes=bbox_list,
                class_labels=class_labels,
            )

            img_tensor = transformed["image"]

            if transformed["bboxes"]:
                bbox_t = transformed["bboxes"][0]
                x_center = (bbox_t[0] + bbox_t[2]) / 2 / self.img_size
                y_center = (bbox_t[1] + bbox_t[3]) / 2 / self.img_size
                width = (bbox_t[2] - bbox_t[0]) / self.img_size
                height = (bbox_t[3] - bbox_t[1]) / self.img_size
                target = torch.tensor(
                    [x_center, y_center, width, height, 1.0], dtype=torch.float32
                )
            else:
                target = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.0], dtype=torch.float32)

            images.append(img_tensor)
            targets.append(target)

        images_tensor = torch.stack(images)
        targets_tensor = torch.stack(targets)
        return images_tensor, targets_tensor


def create_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    sequence_length: int = 10,
    **kwargs,
):
    """Create train/val/test dataloaders."""
    train_dataset = MMFWUAVSequenceDataset(
        data_root=data_root,
        split="train",
        sequence_length=sequence_length,
        **kwargs,
    )

    val_dataset = MMFWUAVSequenceDataset(
        data_root=data_root,
        split="val",
        sequence_length=sequence_length,
        **kwargs,
    )

    test_dataset = MMFWUAVSequenceDataset(
        data_root=data_root,
        split="test",
        sequence_length=sequence_length,
        **kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
