"""Visualization utilities."""
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_bbox(image: np.ndarray, bbox: Dict, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draw a bounding box on an image (in-place)."""
    cv2.rectangle(
        image,
        (bbox["xmin"], bbox["ymin"]),
        (bbox["xmax"], bbox["ymax"]),
        color,
        thickness,
    )
    label = bbox.get("class", "UAV")
    cv2.putText(
        image,
        label,
        (bbox["xmin"], max(0, bbox["ymin"] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )
    return image


def visualize_sample(
    images: List[np.ndarray],
    bboxes: List[List[Dict]],
    titles: Optional[List[str]] = None,
    figsize=(15, 5),
):
    """Visualize multiple images with bounding boxes."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for i, (img, bbox_list) in enumerate(zip(images, bboxes)):
        img_copy = img.copy()
        for bbox in bbox_list:
            img_copy = draw_bbox(img_copy, bbox)

        axes[i].imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        axes[i].axis("off")
        if titles:
            axes[i].set_title(titles[i])

    plt.tight_layout()
    return fig
