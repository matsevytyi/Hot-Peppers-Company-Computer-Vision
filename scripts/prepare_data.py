"""Prepare MMFW-UAV dataset: create splits and preprocess."""
import argparse
import json
import random
from pathlib import Path


def create_splits(data_root, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """Create train/val/test splits by UAV type."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    uav_types = [d.name for d in data_root.iterdir() if d.is_dir() and "UAV" in d.name]
    uav_types = sorted(uav_types)

    if not uav_types:
        raise RuntimeError(f"No UAV type directories found in {data_root}")

    random.seed(seed)
    random.shuffle(uav_types)

    n_train = int(len(uav_types) * train_ratio)
    n_val = int(len(uav_types) * val_ratio)

    train_uavs = uav_types[:n_train]
    val_uavs = uav_types[n_train : n_train + n_val]
    test_uavs = uav_types[n_train + n_val :]

    splits = {
        "train": {"uav_types": train_uavs},
        "val": {"uav_types": val_uavs},
        "test": {"uav_types": test_uavs},
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
        default="data/MMFW-UAV/splits",
        help="Output directory for splits",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    splits = create_splits(args.data_root, args.output_dir, args.train_ratio, args.val_ratio, args.seed)

    print("✅ Splits saved:")
    for name, split in splits.items():
        print(f"  {name}: {len(split['uav_types'])} UAV types")
