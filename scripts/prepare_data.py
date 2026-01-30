"""Prepare MMFW-UAV dataset: create splits and preprocess."""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def _split_counts(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if total <= 1:
        return total, 0, 0

    n_train = max(1, int(total * train_ratio))
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    if n_test == 0 and total > 1:
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
        n_test = total - n_train - n_val

    if n_test < 0:
        n_val = max(0, total - n_train)
        n_test = total - n_train - n_val

    return n_train, n_val, n_test


def _discover_groups(data_root: Path) -> Tuple[str, Dict[str, List[Path]]]:
    top_uavs = [d for d in data_root.iterdir() if d.is_dir() and "UAV" in d.name]
    if top_uavs:
        groups = {d.name: [d.relative_to(data_root)] for d in sorted(top_uavs, key=lambda p: p.name)}
        return "uav", groups

    groups: Dict[str, List[Path]] = {}
    for part_dir in sorted([d for d in data_root.iterdir() if d.is_dir()], key=lambda p: p.name):
        uavs = [d for d in part_dir.iterdir() if d.is_dir() and "UAV" in d.name]
        if uavs:
            groups[part_dir.name] = [u.relative_to(data_root) for u in sorted(uavs, key=lambda p: p.name)]

    if not groups:
        raise RuntimeError(f"No UAV type directories found in {data_root}")

    return "part", groups


def create_splits(data_root, output_dir=None, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Create train/val/test splits by dataset group.

    If the root contains UAV directories, split by UAV type.
    If the root contains parts with UAV subfolders, split by part.
    Guarantees at least one group in train; val/test may be empty for tiny datasets.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir) if output_dir else data_root / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    group_by, groups = _discover_groups(data_root)

    random.seed(seed)
    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    total = len(group_keys)
    n_train, n_val, n_test = _split_counts(total, train_ratio, val_ratio)

    train_groups = group_keys[:n_train]
    val_groups = group_keys[n_train : n_train + n_val]
    test_groups = group_keys[n_train + n_val :]

    def _flatten(group_list: List[str]) -> List[str]:
        items: List[str] = []
        for group in group_list:
            items.extend([p.as_posix() for p in groups[group]])
        return items

    train_items = _flatten(train_groups)
    val_items = _flatten(val_groups)
    test_items = _flatten(test_groups)

    if not val_items:
        print("⚠️  Validation split is empty (small dataset).")
    if not test_items:
        print("⚠️  Test split is empty (small dataset).")

    splits = {
        "train": {"items": train_items, "meta": {"group_by": group_by, "groups": train_groups}},
        "val": {"items": val_items, "meta": {"group_by": group_by, "groups": val_groups}},
        "test": {"items": test_items, "meta": {"group_by": group_by, "groups": test_groups}},
    }

    for split_name, split_data in splits.items():
        with open(output_dir / f"{split_name}.json", "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2)

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to MMFW-UAV dataset root (e.g., data/MMFW-UAV/raw)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for splits (default: <data_root>/splits)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    splits = create_splits(args.data_root, args.output_dir, args.train_ratio, args.val_ratio, args.seed)

    print("✅ Splits saved:")
    for name, split in splits.items():
        print(f"  {name}: {len(split['uav_types'])} UAV types")
