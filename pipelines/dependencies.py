"""Dependency checks used by notebooks and pipeline scripts."""

from __future__ import annotations

import importlib.util
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
