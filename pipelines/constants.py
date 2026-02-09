"""Shared constants for the COCO+BDD+ACDC pipelines."""

from __future__ import annotations

from typing import Dict

COMMON_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic_light",
]

COMMON_CLASS_TO_INDEX = {name: idx for idx, name in enumerate(COMMON_CLASSES)}

LABEL_ALIASES: Dict[str, str] = {
    "traffic light": "traffic_light",
    "traffic-light": "traffic_light",
    "traffic_light": "traffic_light",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "cyclist": "bicycle",
}


def canonicalize_label(label: str) -> str:
    """Map raw dataset label names to the canonical shared class names."""
    normalized = label.strip().lower().replace("/", "_")
    return LABEL_ALIASES.get(normalized, normalized)
