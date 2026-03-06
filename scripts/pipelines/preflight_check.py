"""Preflight checks for Mamba-Vision pipeline environment."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    required = ["torch", "torchvision", "yaml", "einops", "safetensors"]
    optional = ["fiftyone", "mambavision", "wandb", "pycocotools"]

    print("=== Required packages ===")
    missing_required = []
    for package in required:
        ok = has_package(package)
        print(f"{package:15s} {'OK' if ok else 'MISSING'}")
        if not ok:
            missing_required.append(package)

    print("\n=== Optional packages ===")
    for package in optional:
        ok = has_package(package)
        print(f"{package:15s} {'OK' if ok else 'MISSING'}")

    print("\n=== Paths ===")
    repo_root = Path(__file__).resolve().parents[2]
    print("Repo:", repo_root)
    print("MambaVision file:", repo_root / "mamba-vision-ours/model.py")
    print("Submodule dir:", repo_root / "MambaVisionReengineering")

    if missing_required:
        raise SystemExit(f"Missing required packages: {', '.join(missing_required)}")

    print("\nPreflight check passed.")


if __name__ == "__main__":
    main()
