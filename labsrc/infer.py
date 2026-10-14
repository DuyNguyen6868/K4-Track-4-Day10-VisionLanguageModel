"""Bước 3/7: suy luận. E0: bỏ --adapter. Ví dụ: python -m labsrc.infer --adapter work/adapters/e1 --out work/e1.jsonl"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel

from .data import load_split, make_prompt
from .modeling import load_model, load_projector_state, run_inference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--coord", choices=["norm1000", "pixel"])    # mặc định: lấy từ run.json của adapter
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    adapter = Path(args.adapter) if args.adapter else None
    coord = args.coord
    if adapter and not coord and (adapter / "run.json").exists():
        coord = json.loads((adapter / "run.json").read_text(encoding="utf-8")).get("coord")
    coord = coord or "norm1000"
    print(f"coord={coord}")

    model, processor = load_model()
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
        extra = adapter / "extra_trainable.pt"
        if extra.exists():
            n = load_projector_state(model, torch.load(extra, map_location="cpu"))
            print(f"đã nạp lại {n} tensor projector từ {extra}")
    run_inference(model, processor, load_split(args.split, args.limit), make_prompt(coord), args.out,
                  batch_size=args.bs, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
