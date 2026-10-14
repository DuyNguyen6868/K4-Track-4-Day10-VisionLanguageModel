"""E5: reward IoU kiểm chứng được (Python thuần, test không cần GPU)."""
from scripts.score import iou_xyxy, parse_output


def iou_reward(completion, gt_boxes):
    """gt_boxes: list[{"label", "bbox": (x1,y1,x2,y2)}] cùng thang với completion."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0                                     # JSON hỏng
    if not gt_boxes and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < 0.5:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)   # box bịa bị phạt qua mẫu số
