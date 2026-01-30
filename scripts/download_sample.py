"""Create a small sample subset from a local MMFW-UAV dataset copy."""
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def _parse_frame_id(path: Path) -> Tuple[str, int | None]:
    stem = path.stem
    if "_" in stem:
        prefix, idx_str = stem.rsplit("_", 1)
        if idx_str.isdigit():
            return prefix, int(idx_str)
    return stem, None


def _contiguous_runs(frames: List[Path]) -> List[List[Path]]:
    grouped: dict[str, list[tuple[int | None, Path]]] = {}
    for path in frames:
        prefix, idx = _parse_frame_id(path)
        grouped.setdefault(prefix, []).append((idx, path))

    runs: List[List[Path]] = []
    for items in grouped.values():
        items.sort(key=lambda x: x[0] if x[0] is not None else x[1].name)
        if items and items[0][0] is None:
            runs.append([p for _, p in items])
            continue

        current = [items[0]]
        for prev, curr in zip(items, items[1:]):
            prev_idx = prev[0]
            curr_idx = curr[0]
            if prev_idx is not None and curr_idx is not None and curr_idx == prev_idx + 1:
                current.append(curr)
            else:
                runs.append([p for _, p in current])
                current = [curr]
        runs.append([p for _, p in current])

    return runs


def _select_contiguous_frames(
    runs: List[List[Path]],
    max_frames: int | None,
    clip_length: int,
    seed: int,
) -> List[Path]:
    if max_frames is None:
        return [p for run in runs for p in run]

    rng = random.Random(seed)
    candidates: List[List[Path]] = []
    for run in runs:
        if len(run) < clip_length:
            continue
        for start in range(0, len(run) - clip_length + 1, clip_length):
            candidates.append(run[start : start + clip_length])

    rng.shuffle(candidates)
    selected: List[Path] = []
    for clip in candidates:
        for frame in clip:
            if len(selected) >= max_frames:
                return selected
            selected.append(frame)

    if len(selected) < max_frames:
        leftovers = [p for run in runs for p in run if p not in selected]
        for frame in leftovers:
            if len(selected) >= max_frames:
                break
            selected.append(frame)

    return selected


def create_sample(
    raw_dir: str,
    output_dir: str,
    uav_types=None,
    view: str = "Top_Down",
    sensor: str = "Zoom",
    limit_per_type: int = 200,
    clip_length: int = 10,
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
        "clip_length": clip_length,
        "limit_per_type": limit_per_type,
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

        ann_stems = {p.stem for p in ann_dir.glob("*.xml")}
        frames = [p for p in frames if p.stem in ann_stems]
        if not frames:
            print(f"No annotated frames found in {img_dir}")
            continue

        runs = _contiguous_runs(frames)
        sample = _select_contiguous_frames(runs, limit_per_type, clip_length, seed)

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
    parser.add_argument("--clip_length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    create_sample(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        uav_types=args.uav_types,
        view=args.view,
        sensor=args.sensor,
        limit_per_type=args.limit_per_type,
        clip_length=args.clip_length,
        seed=args.seed,
    )
