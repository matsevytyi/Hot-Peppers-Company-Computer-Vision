"""FiftyOne-first data ingestion, curation, export, and manifest generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .constants import COMMON_CLASSES, COMMON_CLASS_TO_INDEX, canonicalize_label
from .contracts import DatasetManifest
from .dependencies import assert_required_packages


def _require_fiftyone():
    assert_required_packages(["fiftyone"])
    import fiftyone as fo
    import fiftyone.zoo as foz

    return fo, foz


def _select_detection_field(dataset, preferred_fields: Iterable[str] | None = None) -> str:
    preferred_fields = list(preferred_fields or ["ground_truth", "detections"])
    schema = dataset.get_field_schema()
    for field_name in preferred_fields:
        if field_name in schema:
            field_type_name = schema[field_name].__name__.lower()
            if "detections" in field_type_name:
                return field_name
    for field_name, field_type in schema.items():
        if "detections" in field_type.__name__.lower():
            return field_name
    raise RuntimeError(
        f"No Detections field found in dataset {dataset.name}. Available fields: {sorted(schema.keys())}"
    )


def load_zoo_split(
    *,
    zoo_name: str,
    dataset_name: str,
    split: str,
    classes: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    persist: bool = True,
):
    """Load one split from FiftyOne Zoo."""
    fo, foz = _require_fiftyone()
    zoo_kwargs = {
        "split": split,
        "label_types": ["detections"],
        "max_samples": max_samples,
        "persistent": persist,
        "dataset_name": dataset_name,
    }
    if classes:
        zoo_kwargs["classes"] = classes
    return foz.load_zoo_dataset(zoo_name, **zoo_kwargs)


def import_local_dataset(
    *,
    dataset_name: str,
    dataset_dir: str,
    dataset_type_name: str = "COCODetectionDataset",
    persist: bool = True,
):
    """Load local datasets into FiftyOne (used for ACDC fallback)."""
    fo, _ = _require_fiftyone()
    dataset_type = getattr(fo.types, dataset_type_name)
    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=dataset_type,
            name=dataset_name,
            persistent=persist,
        )
    return dataset


def _extract_time_of_day(sample) -> Optional[str]:
    candidates = []
    for field_name in ("timeofday", "time_of_day", "period"):
        if field_name in sample:
            candidates.append(sample[field_name])

    attrs = sample.get("attributes", None)
    if isinstance(attrs, dict):
        for field_name in ("timeofday", "time_of_day", "period"):
            if field_name in attrs:
                candidates.append(attrs[field_name])

    for item in candidates:
        if item is None:
            continue
        value = str(item).strip().lower()
        if value:
            return value
    return None


def filter_samples_by_time_of_day(dataset, allowed_values: List[str]):
    """Keep samples where a sample-level time-of-day matches allowed values."""
    allowed = {v.strip().lower() for v in allowed_values}
    keep_ids = []
    for sample in dataset.iter_samples(progress=True):
        time_of_day = _extract_time_of_day(sample)
        if time_of_day in allowed:
            keep_ids.append(sample.id)
    if not keep_ids:
        raise RuntimeError(f"No samples matched time_of_day in {allowed}")
    return dataset.select(keep_ids)


def normalize_and_filter_classes(dataset, *, label_field: str, keep_classes: List[str]):
    """Normalize labels and drop detections outside the shared class list."""
    keep_set = set(keep_classes)
    for sample in dataset.iter_samples(progress=True):
        labels = sample[label_field]
        if labels is None:
            continue

        filtered = []
        for det in labels.detections:
            canonical = canonicalize_label(det.label)
            if canonical in keep_set:
                det.label = canonical
                filtered.append(det)
        labels.detections = filtered
        sample[label_field] = labels
        sample.save()
    return dataset


def export_coco_with_manifest(
    *,
    dataset,
    export_dir: str,
    split_name: str,
    source: str,
    manifest_path: str,
    preferred_label_fields: Iterable[str] | None = None,
    class_list: Optional[List[str]] = None,
) -> DatasetManifest:
    fo, _ = _require_fiftyone()

    class_list = class_list or list(COMMON_CLASSES)
    label_field = _select_detection_field(dataset, preferred_fields=preferred_label_fields)
    export_path = Path(export_dir).resolve()
    export_path.mkdir(parents=True, exist_ok=True)

    dataset.export(
        export_dir=str(export_path),
        dataset_type=fo.types.COCODetectionDataset,
        label_field=label_field,
        classes=class_list,
    )

    annotations_file = export_path / "labels.json"
    images_dir = export_path / "data"
    if not annotations_file.exists():
        raise FileNotFoundError(f"COCO export missing labels.json at: {annotations_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"COCO export missing image folder at: {images_dir}")

    with open(annotations_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    num_images = len(payload.get("images", []))
    num_instances = len(payload.get("annotations", []))

    manifest = DatasetManifest.create(
        dataset_name=dataset.name,
        source=source,
        split=split_name,
        class_list=class_list,
        class_map=COMMON_CLASS_TO_INDEX,
        root_dir=export_path,
        images_dir=images_dir,
        labels_or_annotations=annotations_file,
        num_images=num_images,
        num_instances=num_instances,
    )
    manifest.save_json(manifest_path)
    return manifest


def prepare_zoo_split_export(
    *,
    zoo_name: str,
    split: str,
    dataset_name: str,
    export_dir: str,
    manifest_path: str,
    source: str = "fiftyone_zoo",
    max_samples: Optional[int] = None,
    time_of_day: Optional[List[str]] = None,
    local_dataset_dir: Optional[str] = None,
    local_dataset_type: str = "COCODetectionDataset",
) -> DatasetManifest:
    """High-level helper used by notebooks for COCO/BDD split export."""
    try:
        dataset = load_zoo_split(
            zoo_name=zoo_name,
            dataset_name=dataset_name,
            split=split,
            classes=COMMON_CLASSES,
            max_samples=max_samples,
            persist=True,
        )
    except Exception as exc:
        if not local_dataset_dir:
            raise RuntimeError(
                f"Failed to load zoo dataset '{zoo_name}' split '{split}'. "
                "Provide `local_dataset_dir` for fallback import."
            ) from exc
        dataset = import_local_dataset(
            dataset_name=dataset_name,
            dataset_dir=local_dataset_dir,
            dataset_type_name=local_dataset_type,
            persist=True,
        )

    label_field = _select_detection_field(dataset)
    dataset = normalize_and_filter_classes(dataset, label_field=label_field, keep_classes=COMMON_CLASSES)
    if time_of_day:
        dataset = filter_samples_by_time_of_day(dataset, time_of_day)
    return export_coco_with_manifest(
        dataset=dataset,
        export_dir=export_dir,
        split_name=split,
        source=source,
        manifest_path=manifest_path,
        preferred_label_fields=[label_field],
    )
