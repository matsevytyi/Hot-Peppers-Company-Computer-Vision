"""Sequence dataset for Mamba model with variable image size support."""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve
from urllib.parse import unquote

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from tqdm import tqdm

from shared.parser import extract_bboxes, parse_voc_xml


# =============================================================================
# Dataset Download URLs
# =============================================================================

DATASET_URLS: Dict[str, List[str]] = {
    "A": [
        "https://download.scidb.cn/download?fileId=9282ae0baf2816c72bfff8164d735c83&path=/V4/Fixed-wing-UAV-A.zip&fileName=Fixed-wing-UAV-A.zip",
        "https://download.scidb.cn/download?fileId=540ca62c76a19725f728dc1c3caf2fa6&path=/V4/Fixed-wing-UAV-A'.zip&fileName=Fixed-wing-UAV-A'.zip",
    ],
    "B": [
        "https://download.scidb.cn/download?fileId=3e4f6ea1c00086277d3cfa3394760c10&path=/V4/Fixed-wing-UAV-B.zip&fileName=Fixed-wing-UAV-B.zip",
        "https://download.scidb.cn/download?fileId=4d9c87d634036c1025626187a9cd39b4&path=/V4/Fixed-wing-UAV-B'.zip&fileName=Fixed-wing-UAV-B'.zip",
    ],
    "C": [
        "https://download.scidb.cn/download?fileId=d83105594fcda6c1289736dd9c1399be&path=/V4/Fixed-wing-UAV-C.zip&fileName=Fixed-wing-UAV-C.zip",
        "https://download.scidb.cn/download?fileId=2ee85e05571f3a3b8a36ad5ce334816c&path=/V4/Fixed-wing-UAV-C'.zip&fileName=Fixed-wing-UAV-C'.zip",
    ],
    "D": [
        "https://download.scidb.cn/download?fileId=8ab3c0618dbc1a8593b6b9a877984fb8&path=/V4/Fixed-wing-UAV-D.zip&fileName=Fixed-wing-UAV-D.zip",
        "https://download.scidb.cn/download?fileId=b3c7263efcd06b0b59f05bf20e5cdd94&path=/V4/Fixed-wing-UAV-D'.zip&fileName=Fixed-wing-UAV-D'.zip",
    ],
    "E": [
        "https://download.scidb.cn/download?fileId=7d4017b5b04aff2862959726bab02a50&path=/V4/Fixed-wing-UAV-E.zip&fileName=Fixed-wing-UAV-E.zip",
        "https://download.scidb.cn/download?fileId=5045a615b5ca84d3d834c1f4174c644f&path=/V4/Fixed-wing-UAV-E'.zip&fileName=Fixed-wing-UAV-E'.zip",
    ],
    "F": [
        "https://download.scidb.cn/download?fileId=685fe3af3ddacffa08eaee41b5665ffd&path=/V4/Fixed-wing-UAV-F.zip&fileName=Fixed-wing-UAV-F.zip",
        "https://download.scidb.cn/download?fileId=9439953f6b87ee79261c614f42f033ef&path=/V4/Fixed-wing-UAV-F'.zip&fileName=Fixed-wing-UAV-F'.zip",
    ],
    "JSON": [
        "https://download.scidb.cn/download?fileId=9b4a1af53569adc3389e5e6659d0dfd7&path=/V4/JSON.zip&fileName=JSON.zip",
    ],
}


# =============================================================================
# Download Utilities
# =============================================================================


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> Path:
    """Download a file with progress bar.
    
    Handles SSL certificate issues common on macOS.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try using requests first (better SSL handling)
    try:
        import requests
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=output_path.name,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return output_path
    
    except ImportError:
        pass  # Fall back to urllib
    except Exception as e:
        print(f"⚠️ requests failed: {e}, trying urllib with SSL workaround...")
    
    # Fallback: urllib with SSL context that doesn't verify (for macOS)
    import ssl
    import urllib.request
    
    # Create unverified SSL context (needed for some macOS Python installations)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Install the opener globally
    https_handler = urllib.request.HTTPSHandler(context=ssl_context)
    opener = urllib.request.build_opener(https_handler)
    
    with opener.open(url) as response:
        total_size = int(response.headers.get('Content-Length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=output_path.name,
        ) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    
    return output_path



def download_uav_part(
    uav_type: str,
    output_dir: str = "data/MMFW-UAV/raw",
    keep_zip: bool = False,
) -> Path:
    """Download and extract one UAV type.
    
    Args:
        uav_type: One of 'A', 'B', 'C', 'D', 'E', 'F', or 'JSON'
        output_dir: Directory to extract files to
        keep_zip: Whether to keep ZIP files after extraction
        
    Returns:
        Path to the extracted UAV directory
    """
    if uav_type not in DATASET_URLS:
        raise ValueError(f"Unknown UAV type: {uav_type}. Must be one of {list(DATASET_URLS.keys())}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    uav_dir = output_path / f"Fixed-wing-UAV-{uav_type}"
    if uav_dir.exists() and any(uav_dir.iterdir()):
        print(f"✅ UAV-{uav_type} already exists at: {uav_dir}")
        print(f"   Skipping download. Delete folder to re-download.")
        return uav_dir
    
    urls = DATASET_URLS[uav_type]
    extracted_dirs = []
    
    for url in urls:
        # Extract filename from URL
        filename = unquote(url.split("fileName=")[-1])
        zip_path = output_path / filename
        
        # Download if not exists
        if not zip_path.exists():
            print(f"📥 Downloading {filename}...")
            download_file(url, zip_path)
        else:
            print(f"✅ {filename} already exists, skipping download")
        
        # Extract
        print(f"📦 Extracting {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_path)
        
        # Track extracted directory
        # Handle both "Fixed-wing-UAV-A" and "Fixed-wing-UAV-A'" naming
        base_name = filename.replace(".zip", "").replace("'", "")
        extracted_dirs.append(output_path / base_name)
        
        # Clean up ZIP if requested
        if not keep_zip:
            zip_path.unlink()
            print(f"🗑️ Deleted {filename}")
    
    # Return path to main UAV directory
    print(f"✅ UAV-{uav_type} ready at: {uav_dir}")
    return uav_dir


def cleanup_uav_part(uav_type: str, data_dir: str = "data/MMFW-UAV/raw") -> None:
    """Delete downloaded UAV data to free space.
    
    Args:
        uav_type: One of 'A', 'B', 'C', 'D', 'E', 'F'
        data_dir: Directory containing the UAV data
    """
    uav_dir = Path(data_dir) / f"Fixed-wing-UAV-{uav_type}"
    
    if uav_dir.exists():
        shutil.rmtree(uav_dir)
        print(f"🗑️ Deleted UAV-{uav_type} data, freed space")
    else:
        print(f"⚠️ UAV-{uav_type} directory not found at {uav_dir}")


def get_available_uav_types(data_dir: str = "data/MMFW-UAV/raw") -> List[str]:
    """Get list of UAV types currently downloaded."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    available = []
    for uav_type in ["A", "B", "C", "D", "E", "F"]:
        if (data_path / f"Fixed-wing-UAV-{uav_type}").exists():
            available.append(uav_type)
    return available


# =============================================================================
# Image Processing with Variable Size Support
# =============================================================================


def letterbox_resize(
    image: np.ndarray,
    target_size: int,
    bboxes: List[Dict],
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, List[Dict], dict]:
    """Resize image with letterbox padding, preserving aspect ratio.
    
    Args:
        image: Input image (H, W, C)
        target_size: Target size for both dimensions
        bboxes: List of bboxes in format {'xmin', 'ymin', 'xmax', 'ymax', 'class'}
        color: Padding color
        
    Returns:
        Resized image, scaled bboxes, and scale info dict
    """
    h, w = image.shape[:2]
    
    # Calculate scale
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    
    # Calculate padding offsets (center the image)
    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    
    # Place resized image on padded canvas
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    
    # Scale bboxes
    scaled_bboxes = []
    for bbox in bboxes:
        scaled_bbox = {
            'xmin': int(bbox['xmin'] * scale + pad_left),
            'ymin': int(bbox['ymin'] * scale + pad_top),
            'xmax': int(bbox['xmax'] * scale + pad_left),
            'ymax': int(bbox['ymax'] * scale + pad_top),
            'class': bbox.get('class', 'uav'),
        }
        scaled_bboxes.append(scaled_bbox)
    
    scale_info = {
        'scale': scale,
        'pad_top': pad_top,
        'pad_left': pad_left,
        'orig_h': h,
        'orig_w': w,
    }
    
    return padded, scaled_bboxes, scale_info


# =============================================================================
# Transforms
# =============================================================================


def get_train_transforms(img_size: int = 640) -> A.ReplayCompose:
    """Get training augmentation transforms."""
    return A.ReplayCompose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Blur(blur_limit=3, p=0.1),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


def get_val_transforms(img_size: int = 640) -> A.ReplayCompose:
    """Get validation transforms (no augmentation)."""
    return A.ReplayCompose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )


# =============================================================================
# Dataset Class
# =============================================================================


class MMFWUAVSequenceDataset(Dataset):
    """Dataset that returns sequences of frames for temporal modeling.
    
    Handles variable image sizes via letterbox resizing.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        sensor_type: str = "Zoom",
        view: str = "Top_Down",
        sequence_length: int = 10,
        stride: int = 5,
        img_size: int = 640,
        transform=None,
        uav_types: Optional[List[str]] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    ):
        """Initialize dataset.
        
        Args:
            data_root: Root directory containing UAV folders
            split: One of 'train', 'val', 'test'
            sensor_type: 'Zoom', 'Wide', or 'Infrared'
            view: 'Top_Down', 'Horizontal', or 'Bottom_Up'
            sequence_length: Number of frames per sequence
            stride: Step between sequence starts
            img_size: Target image size after letterbox resize
            transform: Albumentations transform (must be ReplayCompose)
            uav_types: Optional list of UAV types to include (e.g., ['A', 'B'])
            split_ratios: Tuple of (train, val, test) ratios for deterministic split
        """
        self.data_root = Path(data_root)
        self.split = split
        self.sensor_type = sensor_type
        self.view = view
        self.sequence_length = sequence_length
        self.stride = stride
        self.img_size = img_size
        self.uav_types = uav_types
        self.split_ratios = split_ratios

        # Try to load split file, or scan directories directly
        self.split_items = self._get_split_items()

        # Create all sequences first
        all_sequences = self._create_sequences()
        
        # Apply train/val/test split
        self.sequences = self._apply_split(all_sequences)

        # Setup transforms
        if transform is None:
            self.transform = (
                get_train_transforms(img_size) if split == "train" else get_val_transforms(img_size)
            )
        else:
            self.transform = transform
            
        if not isinstance(self.transform, A.ReplayCompose):
            raise ValueError(
                "Transform must be an albumentations.ReplayCompose to keep sequence augmentations consistent."
            )
    
    def _apply_split(self, all_sequences: List[List[Dict]]) -> List[List[Dict]]:
        """Apply train/val/test split to sequences using deterministic ordering.
        
        Uses sequence order (not random): first 70% train, next 15% val, last 15% test.
        """
        n_total = len(all_sequences)
        train_ratio, val_ratio, _ = self.split_ratios
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if self.split == "train":
            return all_sequences[:n_train]
        elif self.split == "val":
            return all_sequences[n_train:n_train + n_val]
        else:  # test
            return all_sequences[n_train + n_val:]



    def _get_split_items(self) -> List[Path]:
        """Get UAV directories to include based on split or direct scanning."""
        # If specific UAV types are requested, use those
        if self.uav_types:
            items = []
            for uav_type in self.uav_types:
                uav_dir = self.data_root / f"Fixed-wing-UAV-{uav_type}"
                if uav_dir.exists():
                    items.append(uav_dir)
            return items
        
        # Try to load split file
        split_candidates = [
            self.data_root / "splits" / f"{self.split}.json",
            self.data_root.parent / "splits" / f"{self.split}.json",
        ]
        
        for split_file in split_candidates:
            if split_file.exists():
                with open(split_file, "r", encoding="utf-8") as f:
                    split_data = json.load(f)
                
                if "items" in split_data:
                    return [self.data_root / item for item in split_data["items"]]
                elif "uav_types" in split_data:
                    return [self.data_root / item for item in split_data["uav_types"]]
        
        # Fallback: scan directory for UAV folders
        items = []
        for child in sorted(self.data_root.iterdir()):
            if child.is_dir() and "UAV" in child.name:
                items.append(child)
        
        if not items:
            raise FileNotFoundError(
                f"No UAV directories found in {self.data_root}. "
                "Download data first using download_uav_part()."
            )
        
        return items

    @staticmethod
    def _parse_frame_id(frame_name: str) -> Tuple[str, Optional[int]]:
        """Parse frame name into prefix and numeric ID."""
        stem = Path(frame_name).stem
        if "_" in stem:
            prefix, idx_str = stem.rsplit("_", 1)
            if idx_str.isdigit():
                return prefix, int(idx_str)
        return stem, None

    def _create_sequences(self) -> List[List[Dict]]:
        """Group frames into temporal sequences."""
        sequences: List[List[Dict]] = []

        for uav_dir in self.split_items:
            uav_type = uav_dir.name
            img_dir = uav_dir / self.view / f"{self.sensor_type}_Imgs"
            ann_dir = uav_dir / self.view / f"{self.sensor_type}_Anns"

            if not img_dir.exists() or not ann_dir.exists():
                continue

            # Get all frames
            #frames = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))])

            frames = sorted([
                f for f in os.listdir(img_dir) 
                if f.lower().endswith((".jpg", ".png")) 
                and not f.lower().__contains__("._")  # metadata files (manually checked they are not imgs) 
            ])

            print("grabbed frames, ", len(frames), frames[0])

            
            if not frames:
                continue
                
            ann_stems = {p.stem for p in ann_dir.glob("*.xml")}

            # Group frames by prefix for continuous sequences
            grouped: Dict[str, List[Tuple[Optional[int], str]]] = defaultdict(list)
            for frame_name in frames:
                prefix, idx = self._parse_frame_id(frame_name)
                grouped[prefix].append((idx, frame_name))

            # Create sequences from each group
            for items in grouped.values():
                items.sort(key=lambda x: x[0] if x[0] is not None else 0)
                
                # Find consecutive runs
                if items and items[0][0] is None:
                    runs = [items]
                else:
                    runs: List[List[Tuple[Optional[int], str]]] = []
                    current = [items[0]]
                    for prev, curr in zip(items, items[1:]):
                        prev_idx, curr_idx = prev[0], curr[0]
                        if prev_idx is not None and curr_idx is not None and curr_idx == prev_idx + 1:
                            current.append(curr)
                        else:
                            runs.append(current)
                            current = [curr]
                    runs.append(current)

                # Create sequences from runs
                for run in runs:
                    run_frames = [name for _, name in run]
                    valid = [Path(name).stem in ann_stems for name in run_frames]
                    
                    for i in range(0, len(run_frames) - self.sequence_length + 1, self.stride):
                        if not all(valid[i:i + self.sequence_length]):
                            continue
                            
                        sequence: List[Dict] = []
                        for frame_name in run_frames[i:i + self.sequence_length]:
                            img_path = img_dir / frame_name
                            ann_path = ann_dir / f"{Path(frame_name).stem}.xml"
                            sequence.append({
                                "image_path": str(img_path),
                                "annotation_path": str(ann_path),
                                "uav_type": uav_type,
                            })
                        sequences.append(sequence)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return images [T, C, H, W] and targets [T, 5]."""
        sequence = self.sequences[idx]

        # Load all frames and annotations
        images_np: List[np.ndarray] = []
        bboxes_list: List[List[List[float]]] = []
        labels_list: List[List[int]] = []

        for frame_data in sequence:
            # Load image
            img = cv2.imread(frame_data["image_path"])
            if img is None:
                #raise FileNotFoundError(f"Image not found: {frame_data['image_path']}")
                print(f"Image not found: {frame_data['image_path']}")
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Parse annotation
            annotation = parse_voc_xml(frame_data["annotation_path"])
            bboxes = extract_bboxes(annotation)

            # Letterbox resize to target size (handles variable image sizes)
            if bboxes:
                img, bboxes, _ = letterbox_resize(img, self.img_size, bboxes)
                bbox = bboxes[0]  # Use first bbox
                bbox_list = [[bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]]]
                class_labels = [1]
            else:
                img, _, _ = letterbox_resize(img, self.img_size, [])
                bbox_list = []
                class_labels = []

            images_np.append(img)
            bboxes_list.append(bbox_list)
            labels_list.append(class_labels)

        # Apply transforms (consistent across sequence using ReplayCompose)
        transformed_frames = []
        first = self.transform(
            image=images_np[0],
            bboxes=bboxes_list[0],
            class_labels=labels_list[0],
        )
        replay = first["replay"]
        transformed_frames.append(first)
        
        for img, bbox_list, class_labels in zip(images_np[1:], bboxes_list[1:], labels_list[1:]):
            transformed = A.ReplayCompose.replay(
                replay,
                image=img,
                bboxes=bbox_list,
                class_labels=class_labels,
            )
            transformed_frames.append(transformed)

        # Convert to tensors
        images: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []
        
        for transformed in transformed_frames:
            img_tensor = transformed["image"]

            if transformed["bboxes"]:
                bbox_t = transformed["bboxes"][0]
                # Convert to normalized center format [x_center, y_center, w, h, conf]
                x_center = (bbox_t[0] + bbox_t[2]) / 2 / self.img_size
                y_center = (bbox_t[1] + bbox_t[3]) / 2 / self.img_size
                width_n = (bbox_t[2] - bbox_t[0]) / self.img_size
                height_n = (bbox_t[3] - bbox_t[1]) / self.img_size
                target = torch.tensor([x_center, y_center, width_n, height_n, 1.0], dtype=torch.float32)
            else:
                target = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.0], dtype=torch.float32)

            images.append(img_tensor)
            targets.append(target)

        images_tensor = torch.stack(images)
        targets_tensor = torch.stack(targets)
        return images_tensor, targets_tensor


def create_dataloaders(
    data_root: str,
    batch_size: int = 4,
    num_workers: int = 4,
    sequence_length: int = 10,
    uav_types: Optional[List[str]] = None,
    **kwargs,
):
    """Create train/val/test dataloaders.
    
    Args:
        data_root: Root directory containing UAV folders
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        sequence_length: Frames per sequence
        uav_types: Optional list of UAV types to include
        **kwargs: Additional arguments for dataset
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = MMFWUAVSequenceDataset(
        data_root=data_root,
        split="train",
        sequence_length=sequence_length,
        uav_types=uav_types,
        **kwargs,
    )

    val_dataset = MMFWUAVSequenceDataset(
        data_root=data_root,
        split="val",
        sequence_length=sequence_length,
        uav_types=uav_types,
        **kwargs,
    )

    test_dataset = MMFWUAVSequenceDataset(
        data_root=data_root,
        split="test",
        sequence_length=sequence_length,
        uav_types=uav_types,
        **kwargs,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
