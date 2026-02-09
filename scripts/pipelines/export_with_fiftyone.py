"""Export COCO-style data with FiftyOne and generate DatasetManifest JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from pipelines.fiftyone_data import prepare_zoo_split_export  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo_name", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--export_dir", type=str, required=True)
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--source", type=str, default="fiftyone_zoo")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--time_of_day", type=str, nargs="*", default=None)
    parser.add_argument("--local_dataset_dir", type=str, default=None)
    parser.add_argument("--local_dataset_type", type=str, default="COCODetectionDataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir)
    if not export_dir.is_absolute():
        export_dir = (REPO_ROOT / export_dir).resolve()
    manifest_path = Path(args.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = (REPO_ROOT / manifest_path).resolve()

    manifest = prepare_zoo_split_export(
        zoo_name=args.zoo_name,
        split=args.split,
        dataset_name=args.dataset_name,
        export_dir=str(export_dir),
        manifest_path=str(manifest_path),
        source=args.source,
        max_samples=args.max_samples,
        time_of_day=args.time_of_day,
        local_dataset_dir=args.local_dataset_dir,
        local_dataset_type=args.local_dataset_type,
    )
    print("✅ Export completed")
    print(f"Manifest: {manifest_path}")
    print(f"Images: {manifest.num_images}")
    print(f"Instances: {manifest.num_instances}")


if __name__ == "__main__":
    main()
