"""Utilities for converting YOLO labels and building training prompts.

These helpers mirror the lab pseudocode described in GUIDE.md and COLAB_GUIDE.md.
The functions are intentionally lightweight and framework-agnostic so they can run
outside the full GPU training environment.
"""

from __future__ import annotations

import json
from typing import Iterable, Dict, Any


def yolo_to_box(label: str, yolo: Iterable[float]) -> Dict[str, Any]:
    """Convert normalized YOLO box to a 0-1000 grounding box."""
    xc, yc, w, h = (float(v) for v in yolo)
    x1 = max(0.0, min(1000.0, (xc - w / 2.0) * 1000.0))
    y1 = max(0.0, min(1000.0, (yc - h / 2.0) * 1000.0))
    x2 = max(0.0, min(1000.0, (xc + w / 2.0) * 1000.0))
    y2 = max(0.0, min(1000.0, (yc + h / 2.0) * 1000.0))
    return {
        "label": label,
        "bbox_2d": [round(x1), round(y1), round(x2), round(y2)],
    }


def objects_to_answer(objects: Iterable[Dict[str, Any]]) -> str:
    """Serialize a list of object annotations in stable reading order."""
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda box: (box["bbox_2d"][1], box["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":")) if boxes else "[]"
