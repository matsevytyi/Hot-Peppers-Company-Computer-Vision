"""Shared dataset helpers."""
from pathlib import Path
from typing import List


def list_uav_types(data_root: str) -> List[str]:
    """Return sorted UAV type directory names under the data root."""
    root = Path(data_root)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def list_frames(image_dir: str, suffix: str = ".jpg") -> List[str]:
    """Return sorted frame filenames in a directory."""
    image_path = Path(image_dir)
    return sorted([p.name for p in image_path.iterdir() if p.suffix.lower() == suffix])
