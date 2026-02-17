"""FiftyOne-first data ingestion, curation, export, and manifest generation."""

from __future__ import annotations

import ast
import json
import re
import socket
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .constants import COMMON_CLASSES, COMMON_CLASS_TO_INDEX, canonicalize_label
from .contracts import DatasetManifest
from .dependencies import assert_required_packages


KNOWN_ZOO_ERROR_PATTERNS = (
    "name resolution",
    "max retries exceeded",
    "connectionerror",
    "failed to download",
    "unable to get",
    "unsupported split",
    "dataset not found",
    "must provide a `source_dir`",
    "download the source files for bdd100k dataset manually",
)

RETRIABLE_ZOO_ERROR_PATTERNS = (
    "name resolution",
    "max retries exceeded",
    "connectionerror",
    "failed to download",
    "unable to get",
    "temporarily unavailable",
    "timed out",
    "timeout",
)


def _require_fiftyone():
    assert_required_packages(["fiftyone"])
    import fiftyone as fo
    import fiftyone.zoo as foz

    return fo, foz


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None:
        ident = id(cur)
        if ident in seen:
            break
        seen.add(ident)
        yield cur
        nxt = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        cur = nxt


def _exception_chain_text(exc: BaseException) -> str:
    return " | ".join(f"{type(item).__name__}: {item}" for item in _iter_exception_chain(exc)).lower()


def _is_known_zoo_load_error(exc: BaseException) -> bool:
    chain = list(_iter_exception_chain(exc))
    chain_text = _exception_chain_text(exc)

    for item in chain:
        module = type(item).__module__
        name = type(item).__name__
        if module.startswith("requests.exceptions") and name.endswith("RequestException"):
            return True
        if module.startswith("requests.exceptions"):
            return True
        if module.startswith("urllib3.exceptions") and name.endswith("HTTPError"):
            return True
        if module.startswith("urllib3.exceptions"):
            return True
        if module.startswith("eta.core.web") and name == "WebSessionError":
            return True
        if isinstance(item, (socket.gaierror, TimeoutError)):
            return True
        if "requests.exceptions" in module or "urllib3.exceptions" in module:
            return True

    if re.search(r"dataset.+not found", chain_text):
        return True
    return any(pattern in chain_text for pattern in KNOWN_ZOO_ERROR_PATTERNS)


def _is_retriable_zoo_error(exc: BaseException) -> bool:
    chain = list(_iter_exception_chain(exc))
    chain_text = _exception_chain_text(exc)

    for item in chain:
        module = type(item).__module__
        if module.startswith("requests.exceptions") or module.startswith("urllib3.exceptions"):
            return True
        if module.startswith("eta.core.web") and type(item).__name__ == "WebSessionError":
            return True
        if isinstance(item, (socket.gaierror, TimeoutError)):
            return True

    return any(pattern in chain_text for pattern in RETRIABLE_ZOO_ERROR_PATTERNS)


def _select_detection_field(dataset, preferred_fields: Iterable[str] | None = None) -> str:
    def _is_detections(field_spec) -> bool:
        doc_type = getattr(field_spec, "document_type", None) or getattr(
            field_spec, "embedded_doc_type", None
        )
        if doc_type is not None:
            doc_name = getattr(doc_type, "__name__", type(doc_type).__name__)
            return "detections" in str(doc_name).lower()
        field_name = getattr(field_spec, "__name__", type(field_spec).__name__)
        return "detections" in str(field_name).lower()

    preferred_fields = list(preferred_fields or ["ground_truth", "detections"])
    schema = dataset.get_field_schema()
    for field_name in preferred_fields:
        if field_name in schema:
            if _is_detections(schema[field_name]):
                return field_name
    for field_name, field_type in schema.items():
        if _is_detections(field_type):
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
    retries: int = 2,
    source_dir: Optional[str] = None,
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
    if source_dir:
        zoo_kwargs["source_dir"] = source_dir
    # In restricted runtimes (e.g., some containerized SSH sessions), thread pools
    # may fail to initialize; serial downloads are slower but more robust.
    if zoo_name.startswith("coco"):
        zoo_kwargs["num_workers"] = 1

    max_attempts = max(1, int(retries) + 1)

    def _load_once() -> object:
        if classes:
            class_candidates: List[List[str]] = []
            with_spaces = [c.replace("_", " ") for c in classes]
            if with_spaces != classes:
                class_candidates.append(with_spaces)
            class_candidates.append(list(classes))

            unsupported_error: Exception | None = None
            for candidate in class_candidates:
                zoo_kwargs["classes"] = candidate
                try:
                    return foz.load_zoo_dataset(zoo_name, **zoo_kwargs)
                except Exception as exc:
                    if "unsupported classes" in str(exc).lower():
                        unsupported_error = exc
                        continue
                    raise
            if unsupported_error is not None:
                raise unsupported_error
        return foz.load_zoo_dataset(zoo_name, **zoo_kwargs)

    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return _load_once()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= max_attempts:
                raise
            if not _is_retriable_zoo_error(exc):
                raise
            time.sleep(min(2.0, 0.5 * (attempt + 1)))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unexpected state: zoo split load did not return a dataset")


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


def _normalize_acdc_split_name(split: str) -> str:
    value = (split or "").strip().lower()
    if value in {"val", "validation"}:
        return "val"
    if value in {"train", "test"}:
        return value
    return value


def _resolve_acdc_detection_json_path(dataset_dir: str, split: str) -> Path:
    split_name = _normalize_acdc_split_name(split)
    json_by_split = {
        "train": "instancesonly_train_gt_detection.json",
        "val": "instancesonly_val_gt_detection.json",
        "test": "instancesonly_test_image_info.json",
    }
    if split_name not in json_by_split:
        raise ValueError(
            "Unsupported ACDC split for local detection import: "
            f"'{split}'. Expected one of train/val/test."
        )

    root = Path(dataset_dir).resolve()
    candidates = [
        root,
        root / "gt_detection",
        root / "gt_detection_trainval" / "gt_detection",
        root.parent / "gt_detection",
    ]
    if root.name == "gt_detection_trainval":
        candidates.append(root / "gt_detection")
    if root.name == "gt_detection":
        candidates.append(root)

    target_name = json_by_split[split_name]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        target = candidate / target_name
        if target.exists():
            return target

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"ACDC detection annotations not found for split '{split_name}'. "
        f"Expected file '{target_name}' under one of: {searched}"
    )


def _resolve_acdc_rgb_root(dataset_dir: str) -> Path:
    root = Path(dataset_dir).resolve()
    parents = [root]
    if root.parent != root:
        parents.append(root.parent)
    if root.parent.parent != root.parent:
        parents.append(root.parent.parent)

    candidates = []
    for base in parents:
        candidates.extend(
            [
                base / "rgb_anon",
                base / "rgb_anon_trainvaltest" / "rgb_anon",
            ]
        )

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "ACDC RGB root not found. Expected one of the following directories: "
        f"{searched}"
    )


def _is_acdc_detection_layout(dataset_dir: str, split: str) -> bool:
    try:
        _resolve_acdc_detection_json_path(dataset_dir, split)
        _resolve_acdc_rgb_root(dataset_dir)
        return True
    except Exception:
        return False


def import_acdc_detection_dataset(
    *,
    dataset_name: str,
    dataset_dir: str,
    split: str,
    persist: bool = True,
):
    """Import local ACDC detection annotations + rgb_anon images in COCO format."""
    fo, _ = _require_fiftyone()
    labels_path = _resolve_acdc_detection_json_path(dataset_dir, split)
    data_path = _resolve_acdc_rgb_root(dataset_dir)

    dataset_type = fo.types.COCODetectionDataset
    if dataset_name in fo.list_datasets():
        existing = fo.load_dataset(dataset_name)
        try:
            _select_detection_field(existing, preferred_fields=["ground_truth", "detections"])
            return existing
        except Exception:
            try:
                existing.delete()
            except Exception:
                if hasattr(fo, "delete_dataset"):
                    fo.delete_dataset(dataset_name)

    return fo.Dataset.from_dir(
        dataset_type=dataset_type,
        data_path=str(data_path),
        labels_path=str(labels_path),
        name=dataset_name,
        persistent=persist,
    )


def _normalize_bdd_split_name(split: str) -> str:
    value = (split or "").strip().lower()
    if value in {"val", "validation"}:
        return "val"
    if value == "train":
        return "train"
    if value == "test":
        return "test"
    return value


def _is_bdd100k_dataset_ninja_layout(dataset_dir: str, split: str) -> bool:
    split_name = _normalize_bdd_split_name(split)
    root = Path(dataset_dir).resolve()
    return (root / split_name / "img").is_dir() and (root / split_name / "ann").is_dir()


def _extract_bdd_tag_value(tags: object, tag_name: str) -> Optional[str]:
    if not isinstance(tags, list):
        return None
    wanted = tag_name.strip().lower()
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        name = str(tag.get("name", "")).strip().lower()
        if name != wanted:
            continue
        value = tag.get("value")
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None
    return None


def _build_bdd_ninja_detection(fo, obj: dict, width: int, height: int):
    if not isinstance(obj, dict):
        return None
    if str(obj.get("geometryType", "")).strip().lower() != "rectangle":
        return None
    points = obj.get("points")
    if not isinstance(points, dict):
        return None
    exterior = points.get("exterior")
    if not isinstance(exterior, list) or len(exterior) < 2:
        return None
    try:
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        x1f = float(x1)
        y1f = float(y1)
        x2f = float(x2)
        y2f = float(y2)
    except (TypeError, ValueError):
        return None

    x_min = min(x1f, x2f)
    y_min = min(y1f, y2f)
    x_max = max(x1f, x2f)
    y_max = max(y1f, y2f)
    box_w = max(0.0, x_max - x_min)
    box_h = max(0.0, y_max - y_min)
    if width <= 0 or height <= 0 or box_w <= 0.0 or box_h <= 0.0:
        return None

    label = str(obj.get("classTitle", "")).strip()
    if not label:
        return None

    detection = fo.Detection(
        label=label,
        bounding_box=[x_min / width, y_min / height, box_w / width, box_h / height],
    )

    raw_tags = obj.get("tags")
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if not isinstance(tag, dict):
                continue
            if str(tag.get("name", "")).strip().lower() != "attributes":
                continue
            raw_value = tag.get("value")
            if raw_value is None:
                continue
            try:
                attrs = ast.literal_eval(raw_value) if isinstance(raw_value, str) else raw_value
            except (SyntaxError, ValueError):
                attrs = None
            if isinstance(attrs, dict):
                for key, value in attrs.items():
                    if key:
                        detection[str(key)] = value

    return detection


def import_bdd100k_dataset_ninja(
    *,
    dataset_name: str,
    dataset_dir: str,
    split: str,
    persist: bool = True,
    max_samples: Optional[int] = None,
):
    """Import dataset-ninja BDD format: `<split>/img` + `<split>/ann/*.jpg.json`."""
    fo, _ = _require_fiftyone()
    split_name = _normalize_bdd_split_name(split)
    root = Path(dataset_dir).resolve()
    image_dir = root / split_name / "img"
    ann_dir = root / split_name / "ann"

    if not image_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset-ninja BDD layout not found for split '{split_name}' under '{root}'. "
            "Expected directories: <root>/<split>/img and <root>/<split>/ann"
        )

    if dataset_name in fo.list_datasets():
        existing = fo.load_dataset(dataset_name)
        try:
            _select_detection_field(existing, preferred_fields=["ground_truth", "detections"])
            return existing
        except Exception:
            # Rebuild stale/partial datasets created by failed earlier runs
            # (for example, datasets without a Detections field).
            try:
                existing.delete()
            except Exception:
                if hasattr(fo, "delete_dataset"):
                    fo.delete_dataset(dataset_name)

    dataset = fo.Dataset(name=dataset_name, persistent=persist)
    ann_files = sorted(ann_dir.glob("*.json"))
    if max_samples is not None:
        ann_files = ann_files[: max(0, int(max_samples))]

    samples = []
    for ann_path in ann_files:
        image_name = ann_path.name[:-5] if ann_path.name.endswith(".json") else ann_path.name
        image_path = image_dir / image_name
        if not image_path.exists():
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        size = payload.get("size") if isinstance(payload, dict) else None
        if not isinstance(size, dict):
            continue
        try:
            width = int(size.get("width") or 0)
            height = int(size.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue

        objects = payload.get("objects") if isinstance(payload, dict) else None
        detections = []
        if isinstance(objects, list):
            for obj in objects:
                det = _build_bdd_ninja_detection(fo, obj, width, height)
                if det is not None:
                    detections.append(det)

        sample = fo.Sample(filepath=str(image_path))
        sample["ground_truth"] = fo.Detections(detections=detections)

        time_of_day = _extract_bdd_tag_value(payload.get("tags"), "timeofday")
        if time_of_day:
            sample["timeofday"] = time_of_day

        samples.append(sample)
        if len(samples) >= 512:
            dataset.add_samples(samples)
            samples = []

    if samples:
        dataset.add_samples(samples)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No valid samples were imported from '{root}' split '{split_name}'. "
            "Check that annotation JSON files match image names."
        )

    return dataset


def _extract_time_of_day(sample) -> Optional[str]:
    candidates = []
    for field_name in ("timeofday", "time_of_day", "period"):
        if field_name in sample:
            candidates.append(sample[field_name])

    attrs = sample["attributes"] if "attributes" in sample else None
    if attrs is not None:
        if isinstance(attrs, dict):
            attrs_dict = attrs
        elif hasattr(attrs, "to_dict"):
            attrs_dict = attrs.to_dict()
        else:
            attrs_dict = None

        if isinstance(attrs_dict, dict):
            for field_name in ("timeofday", "time_of_day", "period"):
                if field_name in attrs_dict:
                    candidates.append(attrs_dict[field_name])

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
    fallback_policy: str = "strict",
    zoo_load_retries: int = 2,
) -> DatasetManifest:
    """High-level helper used by notebooks for COCO/BDD split export."""
    policy = (fallback_policy or "strict").strip().lower()
    if policy not in {"strict", "permissive", "off"}:
        raise ValueError("fallback_policy must be one of: strict, permissive, off")

    zoo_source_dir = local_dataset_dir if zoo_name == "bdd100k" and local_dataset_dir else None

    try:
        dataset = load_zoo_split(
            zoo_name=zoo_name,
            dataset_name=dataset_name,
            split=split,
            classes=COMMON_CLASSES,
            max_samples=max_samples,
            persist=True,
            retries=zoo_load_retries,
            source_dir=zoo_source_dir,
        )
    except Exception as exc:
        known_error = _is_known_zoo_load_error(exc)
        retriable_error = _is_retriable_zoo_error(exc)
        has_local = bool(local_dataset_dir)
        is_bdd = zoo_name == "bdd100k"
        is_acdc = zoo_name == "acdc"

        if not has_local:
            raise RuntimeError(
                f"Failed to load zoo dataset '{zoo_name}' split '{split}'. "
                f"policy={policy}, known_error={known_error}, retriable_error={retriable_error}, "
                f"local_dataset_dir_present={has_local}. "
                f"Root cause: {type(exc).__name__}: {exc}. "
                "Provide `local_dataset_dir` for fallback import."
            ) from exc

        if policy == "off":
            raise RuntimeError(
                f"Failed to load zoo dataset '{zoo_name}' split '{split}' with fallback disabled. "
                f"policy={policy}, known_error={known_error}, retriable_error={retriable_error}, "
                f"local_dataset_dir_present={has_local}. "
                f"Root cause: {type(exc).__name__}: {exc}."
            ) from exc

        if is_bdd and has_local and _is_bdd100k_dataset_ninja_layout(local_dataset_dir, split):
            dataset = import_bdd100k_dataset_ninja(
                dataset_name=dataset_name,
                dataset_dir=local_dataset_dir,
                split=split,
                persist=True,
                max_samples=max_samples,
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

        if is_acdc and has_local and _is_acdc_detection_layout(local_dataset_dir, split):
            if policy == "permissive" or known_error:
                dataset = import_acdc_detection_dataset(
                    dataset_name=dataset_name,
                    dataset_dir=local_dataset_dir,
                    split=split,
                    persist=True,
                )
            else:
                raise RuntimeError(
                    f"Failed to load zoo dataset '{zoo_name}' split '{split}'. "
                    "Fallback blocked by strict policy for unknown error. "
                    f"policy={policy}, known_error={known_error}, retriable_error={retriable_error}, "
                    f"local_dataset_dir_present={has_local}. "
                    f"Root cause: {type(exc).__name__}: {exc}."
                ) from exc

        if policy == "strict" and not known_error:
            raise RuntimeError(
                f"Failed to load zoo dataset '{zoo_name}' split '{split}'. "
                "Fallback blocked by strict policy for unknown error. "
                f"policy={policy}, known_error={known_error}, retriable_error={retriable_error}, "
                f"local_dataset_dir_present={has_local}. "
                f"Root cause: {type(exc).__name__}: {exc}."
            ) from exc

        # BDD100K local paths are expected to be passed as FiftyOne `source_dir`.
        # Falling back to COCODetectionDataset import on the same path is usually incorrect
        # and produces misleading errors like missing `<path>/data`.
        if is_bdd:
            raise RuntimeError(
                f"Failed to load zoo dataset '{zoo_name}' split '{split}' from source_dir '{local_dataset_dir}'. "
                "For BDD100K, local files must be in the original BDD100K layout expected by FiftyOne "
                "(for example labels and images under the official structure), not a COCO `data/ + labels.json` export root. "
                f"policy={policy}, known_error={known_error}, retriable_error={retriable_error}. "
                f"Root cause: {type(exc).__name__}: {exc}."
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
