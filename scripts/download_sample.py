"""Create a small sample subset from a local MMFW-UAV dataset copy."""
import argparse
import json
import random
import shutil
from pathlib import Path


def create_sample(
    raw_dir: str,
    output_dir: str,
    uav_types=None,
    view: str = "Top_Down",
    sensor: str = "Zoom",
    limit_per_type: int = 200,
    seed: int = 42,
):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if uav_types is None:
        uav_types = [p.name for p in raw_dir.iterdir() if p.is_dir() and "UAV" in p.name]

    random.seed(seed)
    metadata = {
        "view": view,
        "sensor": sensor,
        "uav_types": [],
        "counts": {},
    }

    for uav_type in sorted(uav_types):
        img_dir = raw_dir / uav_type / view / f"{sensor}_Imgs"
        ann_dir = raw_dir / uav_type / view / f"{sensor}_Anns"

        if not img_dir.exists():
            print(f"Skipping missing: {img_dir}")
            continue

        frames = sorted([p for p in img_dir.iterdir() if p.suffix.lower() == ".jpg"])
        if not frames:
            print(f"No frames found in {img_dir}")
            continue

        sample = frames
        if limit_per_type is not None and len(frames) > limit_per_type:
            sample = random.sample(frames, limit_per_type)

        out_img_dir = output_dir / uav_type / view / f"{sensor}_Imgs"
        out_ann_dir = output_dir / uav_type / view / f"{sensor}_Anns"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_ann_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img_path in sample:
            ann_path = ann_dir / f"{img_path.stem}.xml"
            if not ann_path.exists():
                continue
            shutil.copy2(img_path, out_img_dir / img_path.name)
            shutil.copy2(ann_path, out_ann_dir / ann_path.name)
            copied += 1

        metadata["uav_types"].append(uav_type)
        metadata["counts"][uav_type] = copied
        print(f"{uav_type}: copied {copied} frames")

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to MMFW-UAV raw dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/MMFW-UAV/sample",
        help="Output sample directory",
    )
    parser.add_argument("--uav_types", nargs="*", default=None, help="Specific UAV types to include")
    parser.add_argument("--view", type=str, default="Top_Down")
    parser.add_argument("--sensor", type=str, default="Zoom")
    parser.add_argument("--limit_per_type", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    create_sample(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        uav_types=args.uav_types,
        view=args.view,
        sensor=args.sensor,
        limit_per_type=args.limit_per_type,
        seed=args.seed,
    )
