"""Reward function used for the optional GRPO ablation in E5."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from scripts.score import parse_output


def reward(completion: str, gt_boxes: Iterable[Dict[str, Any]]) -> float:
    """Compute a stable IoU-based reward for grounded detection outputs."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    gt = list(gt_boxes)
    if not gt and not pred:
        return 1.0
    if not gt:
        return 0.1
    if not pred:
        return 0.0

    total_iou = 0.0
    matched = 0
    for item in pred:
        label = item["label"]
        best = 0.0
        for gt_box in gt:
            if gt_box["label"] != label:
                continue
            a = gt_box["bbox"]
            b = item["bbox"]
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best:
                best = iou
        if best >= 0.5:
            matched += 1
            total_iou += best
    denom = max(len(gt), len(pred), 1)
    return 0.2 + 0.8 * (total_iou / denom)
