"""Parse Pascal VOC XML annotations."""
from typing import Dict, List

import numpy as np
import xmltodict


def parse_voc_xml(xml_path: str) -> Dict:
    """Parse a VOC XML file and return the annotation dict."""
    with open(xml_path, "r", encoding="utf-8") as f:
        data = xmltodict.parse(f.read())
    return data["annotation"]


def extract_bboxes(annotation: Dict) -> List[Dict]:
    """Extract all bounding boxes from a VOC annotation."""
    objects = annotation.get("object", [])
    if not isinstance(objects, list):
        objects = [objects]

    bboxes: List[Dict] = []
    for obj in objects:
        bbox = obj["bndbox"]
        bboxes.append(
            {
                "class": obj["name"],
                "xmin": int(bbox["xmin"]),
                "ymin": int(bbox["ymin"]),
                "xmax": int(bbox["xmax"]),
                "ymax": int(bbox["ymax"]),
            }
        )
    return bboxes


def bbox_to_yolo(bbox: Dict, img_w: int, img_h: int) -> np.ndarray:
    """Convert a VOC bbox to YOLO format [x_center, y_center, w, h] normalized."""
    x_center = (bbox["xmin"] + bbox["xmax"]) / 2 / img_w
    y_center = (bbox["ymin"] + bbox["ymax"]) / 2 / img_h
    width = (bbox["xmax"] - bbox["xmin"]) / img_w
    height = (bbox["ymax"] - bbox["ymin"]) / img_h
    return np.array([x_center, y_center, width, height], dtype=np.float32)
