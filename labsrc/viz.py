"""Vẽ box đúng (xanh) và box dự đoán (đỏ) để làm ảnh minh hoạ cho báo cáo."""
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output

from .data import DATA, read_jsonl


def image_errors(entry, raw, coord="norm1000"):
    parsed = parse_output(raw)
    pred = _to_norm1000(parsed, coord, entry["width"], entry["height"]) if parsed else []
    per, _ = _match(_gt_boxes(entry), pred)
    return (sum(v[0] for v in per.values()), sum(v[1] for v in per.values()),
            sum(v[2] for v in per.values()))          # tp, fp, fn


def draw(entry, raw, path, title, coord="norm1000", zoom=2):
    img = Image.open(DATA / entry["image_path"]).convert("RGB")
    img = img.resize((img.width * zoom, img.height * zoom))
    d = ImageDraw.Draw(img)
    for g in _gt_boxes(entry):
        x1, y1, x2, y2 = g["bbox"]
        d.rectangle([x1 * img.width / 1000, y1 * img.height / 1000,
                     x2 * img.width / 1000, y2 * img.height / 1000], outline=(0, 200, 0), width=3)
        d.text((x1 * img.width / 1000 + 3, y1 * img.height / 1000 + 2), g["label"], fill=(0, 255, 0))
    for p in parse_output(raw) or []:
        x1, y1, x2, y2 = p["bbox"]
        sx, sy = (img.width / 1000, img.height / 1000) if coord == "norm1000" else (zoom, zoom)
        d.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=(230, 0, 0), width=2)
        d.text((x1 * sx + 3, y2 * sy - 12), p["label"], fill=(255, 80, 80))
    d.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    d.text((4, 2), title, fill=(255, 255, 255))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_figures(pred_path, split="test", coord="norm1000", out_dir="report/figures", name="e1", k=3):
    entries = {e["image_id"]: e for e in read_jsonl(DATA / "splits" / f"{split}.jsonl")}
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    scored = [(image_errors(entries[i], preds[i], coord), i) for i in preds]
    good = [i for (tp, fp, fn), i in scored if tp > 0 and fp == 0 and fn == 0][:k]
    bad = [i for (tp, fp, fn), i in sorted(scored, key=lambda s: -(s[0][1] + s[0][2]))][:k]
    for tag, ids in (("right", good), ("wrong", bad)):
        for i in ids:
            tp, fp, fn = image_errors(entries[i], preds[i], coord)
            draw(entries[i], preds[i], f"{out_dir}/{name}_{tag}_{i}.png",
                 f"{name} {tag} | TP={tp} FP={fp} FN={fn} | green=GT, red=pred", coord)
    print(f"đã lưu {len(good)} ảnh đúng + {len(bad)} ảnh sai vào {out_dir}/")
