"""Dependency checks used by notebooks and pipeline scripts."""

from __future__ import annotations

import importlib.util
import platform
from typing import Dict, Iterable


def package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def check_packages(packages: Iterable[str]) -> Dict[str, bool]:
    return {pkg: package_available(pkg) for pkg in packages}


def assert_required_packages(packages: Iterable[str]) -> None:
    missing = [pkg for pkg in packages if not package_available(pkg)]
    if missing:
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(missing)
            + ". Install dependencies before running this notebook."
        )


def mamba_runtime_support_report() -> Dict[str, object]:
    """Return runtime support status for Mamba-based train/eval flows."""
    system = platform.system()
    has_mamba_ssm = package_available("mamba_ssm")

    cuda_available = False
    torch_error = ""
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive guard
        torch_error = f"{type(exc).__name__}: {exc}"

    reasons = []
    if not has_mamba_ssm:
        reasons.append("Missing package: mamba_ssm")
    if not cuda_available:
        reasons.append("CUDA runtime not available. Current Mamba selective_scan path expects CUDA.")
    if system == "Darwin" and not cuda_available:
        reasons.append("macOS detected without CUDA. Current Mamba selective_scan path is CUDA-only.")

    supported = len(reasons) == 0
    return {
        "supported": supported,
        "system": system,
        "has_mamba_ssm": has_mamba_ssm,
        "cuda_available": cuda_available,
        "torch_error": torch_error,
        "reasons": reasons,
    }


def assert_mamba_runtime_support() -> None:
    report = mamba_runtime_support_report()
    if report["supported"]:
        return
    reasons = "; ".join(report["reasons"]) if report["reasons"] else "Unknown runtime incompatibility"
    raise RuntimeError(
        "Mamba runtime is not supported in this environment. "
        f"System={report['system']}, CUDA={report['cuda_available']}, "
        f"mamba_ssm={report['has_mamba_ssm']}. "
        f"Reasons: {reasons}. "
        "Use Linux + NVIDIA CUDA runtime for Mamba train/eval, or run only data-export steps on macOS."
    )
