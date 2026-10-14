"""Nhãn, prompt và đọc split (không phụ thuộc torch)."""
import json
import random
from pathlib import Path

from PIL import Image

DATA = Path("data")
CLASSES = ("pothole", "crack", "manhole")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(name, limit=None, fraction=1.0, seed=42):
    rows = read_jsonl(DATA / "splits" / f"{name}.jsonl")
    if fraction < 1.0:
        rows = random.Random(seed).sample(rows, int(len(rows) * fraction))
    return rows[:limit] if limit else rows


def load_image(entry):
    return Image.open(DATA / entry["image_path"]).convert("RGB")


def make_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        scale = ("Coordinates are integers on a 0-1000 scale relative to the image width and height "
                 "((0,0) = top-left, (1000,1000) = bottom-right).")
    else:
        scale = (f"Coordinates are integer pixels of the {width}x{height} image "
                 f"(x in 0-{width}, y in 0-{height}).")
    return ("Detect all road damage in this dashcam image. Classes: pothole, crack, manhole. "
            'Answer with ONLY a JSON list, no other text: [{"label": "<class>", "bbox_2d": [x1, y1, x2, y2]}]. '
            f"{scale} Answer [] if there is no damage.")


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    xc, yc, w, h = yolo

    def clamp(v, hi):
        return int(min(max(round(v), 0), hi))

    x1, y1 = clamp((xc - w / 2) * sx, sx), clamp((yc - h / 2) * sy, sy)
    x2, y2 = clamp((xc + w / 2) * sx, sx), clamp((yc + h / 2) * sy, sy)
    # box suy biến (x2 <= x1) bị score.py coi là JSON hỏng, nên ép tối thiểu 1 đơn vị
    return {"label": label, "bbox_2d": [x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)]}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))  # trên→dưới, trái→phải
    return json.dumps(boxes, separators=(",", ":"))
