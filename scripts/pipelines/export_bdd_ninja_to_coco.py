"""Fast BDD dataset-ninja to COCO export with day/night filtering.

This script avoids FiftyOne DB/services and writes:
- COCO labels at `<export_dir>/labels.json`
- `data` symlink at `<export_dir>/data` -> original BDD image folder
- Dataset manifest JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from pipelines.constants import COMMON_CLASSES, COMMON_CLASS_TO_INDEX, canonicalize_label  # noqa: E402
from pipelines.contracts import DatasetManifest  # noqa: E402


EXPORT_PROFILES = {
    "bdd_day_train": {
        "split_dir": "train",
        "manifest_split": "train",
        "time_of_day": {"daytime"},
        "export_dir": "data/exports/bdd_day/train",
        "manifest_path": "configs/manifests/bdd_day_train.json",
    },
    "bdd_day_val": {
        "split_dir": "val",
        "manifest_split": "validation",
        "time_of_day": {"daytime"},
        "export_dir": "data/exports/bdd_day/val",
        "manifest_path": "configs/manifests/bdd_day_val.json",
    },
    "bdd_night_train": {
        "split_dir": "train",
        "manifest_split": "train",
        "time_of_day": {"night"},
        "export_dir": "data/exports/bdd_night/train",
        "manifest_path": "configs/manifests/bdd_night_train.json",
    },
    "bdd_night_val": {
        "split_dir": "val",
        "manifest_split": "validation",
        "time_of_day": {"night"},
        "export_dir": "data/exports/bdd_night/val",
        "manifest_path": "configs/manifests/bdd_night_val.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/teamspace/studios/this_studio/datasets/bdd100k:-images-100k",
        help="BDD dataset-ninja root with train/val/test folders",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["bdd_day_val", "bdd_night_train", "bdd_night_val"],
        choices=sorted(EXPORT_PROFILES.keys()),
        help="Which exports to build",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing labels/manifest and replace mismatched data symlink",
    )
    return parser.parse_args()


def _extract_time_of_day(payload: dict) -> Optional[str]:
    tags = payload.get("tags")
    if not isinstance(tags, list):
        return None
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        if str(tag.get("name", "")).strip().lower() != "timeofday":
            continue
        value = tag.get("value")
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None
    return None


def _bbox_from_rectangle(obj: dict, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
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

    x_min = max(0.0, min(x1f, x2f))
    y_min = max(0.0, min(y1f, y2f))
    x_max = min(float(width), max(x1f, x2f))
    y_max = min(float(height), max(y1f, y2f))
    box_w = x_max - x_min
    box_h = y_max - y_min
    if box_w <= 0.0 or box_h <= 0.0:
        return None
    return x_min, y_min, box_w, box_h


def _build_coco_payload(
    *,
    image_dir: Path,
    ann_dir: Path,
    allowed_time_of_day: Iterable[str],
) -> Tuple[Dict[str, object], int, int]:
    allowed = {v.strip().lower() for v in allowed_time_of_day}

    categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(COMMON_CLASSES)]
    images: List[Dict[str, object]] = []
    annotations: List[Dict[str, object]] = []

    image_id = 1
    ann_id = 1
    skipped_missing_image = 0
    skipped_bad_json = 0

    for ann_path in sorted(ann_dir.glob("*.json")):
        try:
            payload = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            skipped_bad_json += 1
            continue

        time_of_day = _extract_time_of_day(payload if isinstance(payload, dict) else {})
        if time_of_day not in allowed:
            continue

        image_name = ann_path.name[:-5] if ann_path.name.endswith(".json") else ann_path.name
        image_path = image_dir / image_name
        if not image_path.exists():
            skipped_missing_image += 1
            continue

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

        images.append({"id": image_id, "file_name": image_name, "width": width, "height": height})

        objects = payload.get("objects") if isinstance(payload, dict) else None
        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                raw_label = str(obj.get("classTitle", "")).strip()
                if not raw_label:
                    continue
                canonical = canonicalize_label(raw_label)
                class_idx = COMMON_CLASS_TO_INDEX.get(canonical)
                if class_idx is None:
                    continue

                bbox = _bbox_from_rectangle(obj, width, height)
                if bbox is None:
                    continue
                x, y, w, h = bbox
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": class_idx + 1,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

        image_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    return coco, skipped_missing_image, skipped_bad_json


def _ensure_data_symlink(link_path: Path, target_dir: Path, *, force: bool) -> None:
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target_dir.resolve():
            return
        if not force:
            raise FileExistsError(
                f"{link_path} already exists and does not match {target_dir}. "
                "Use --force to replace it."
            )
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            raise IsADirectoryError(
                f"{link_path} is a directory. Remove it manually or export to a new location."
            )
    link_path.symlink_to(target_dir, target_is_directory=True)


def _export_profile(*, dataset_root: Path, target_name: str, force: bool) -> None:
    profile = EXPORT_PROFILES[target_name]
    split_dir = profile["split_dir"]
    image_dir = dataset_root / split_dir / "img"
    ann_dir = dataset_root / split_dir / "ann"
    if not image_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(
            f"Expected dataset-ninja layout at {image_dir} and {ann_dir}"
        )

    export_dir = (REPO_ROOT / profile["export_dir"]).resolve()
    manifest_path = (REPO_ROOT / profile["manifest_path"]).resolve()
    labels_path = export_dir / "labels.json"
    data_symlink = export_dir / "data"

    if labels_path.exists() and not force:
        raise FileExistsError(f"{labels_path} exists. Use --force to overwrite.")

    export_dir.mkdir(parents=True, exist_ok=True)
    coco_payload, skipped_missing, skipped_bad_json = _build_coco_payload(
        image_dir=image_dir,
        ann_dir=ann_dir,
        allowed_time_of_day=profile["time_of_day"],
    )
    labels_path.write_text(json.dumps(coco_payload), encoding="utf-8")
    _ensure_data_symlink(data_symlink, image_dir, force=force)

    manifest = DatasetManifest.create(
        dataset_name=target_name,
        source="bdd_ninja_local",
        split=profile["manifest_split"],
        class_list=list(COMMON_CLASSES),
        class_map=dict(COMMON_CLASS_TO_INDEX),
        root_dir=export_dir,
        images_dir=data_symlink,
        labels_or_annotations=labels_path,
        num_images=len(coco_payload["images"]),
        num_instances=len(coco_payload["annotations"]),
    )
    manifest.save_json(manifest_path)

    print(f"[ok] {target_name}")
    print(f"  labels: {labels_path}")
    print(f"  images: {data_symlink} -> {image_dir}")
    print(f"  manifest: {manifest_path}")
    print(
        f"  stats: images={manifest.num_images}, instances={manifest.num_instances}, "
        f"skipped_missing_image={skipped_missing}, skipped_bad_json={skipped_bad_json}"
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    for target_name in args.targets:
        _export_profile(dataset_root=dataset_root, target_name=target_name, force=args.force)


if __name__ == "__main__":
    main()
