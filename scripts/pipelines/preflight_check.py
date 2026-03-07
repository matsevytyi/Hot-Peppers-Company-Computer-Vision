"""Preflight checks for Mamba-Vision pipeline environment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from pipelines.dependencies import mamba_runtime_support_report


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
    repo_root = REPO_ROOT
    print("Repo:", repo_root)
    print("MambaVision file:", repo_root / "mamba-vision-ours/base_model.py")
    print("Submodule dir:", repo_root / "MambaVisionReengineering")

    print("\n=== Mamba runtime support ===")
    runtime = mamba_runtime_support_report()
    print("System:", runtime["system"])
    print("CUDA available:", runtime["cuda_available"])
    print("mamba_ssm installed:", runtime["has_mamba_ssm"])
    if runtime["supported"]:
        print("Mamba train/eval runtime: SUPPORTED")
    else:
        print("Mamba train/eval runtime: NOT SUPPORTED")
        print("Reason(s):", "; ".join(runtime["reasons"]))
        print("Note: on macOS without CUDA, run only data-export/manifest steps.")

    if missing_required:
        raise SystemExit(f"Missing required packages: {', '.join(missing_required)}")

    print("\nPreflight check passed.")


if __name__ == "__main__":
    main()
