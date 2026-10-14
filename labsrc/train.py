"""Bước 6: vòng huấn luyện. Ví dụ: python -m labsrc.train --config configs/e1.yaml --out work/adapters/e1"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from .data import load_image, load_split, make_prompt, objects_to_answer
from .modeling import apply_setup, collate, encode_example, load_model, projector_state


class SFTData(Dataset):
    def __init__(self, entries, processor, coord):
        self.entries, self.processor, self.coord = entries, processor, coord
        self.prompt = make_prompt(coord)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        answer = objects_to_answer(e["objects"], self.coord, e["width"], e["height"])
        return encode_example(self.processor, load_image(e), self.prompt, answer)


def forever(loader):
    while True:
        yield from loader


def save(model, out, cfg):
    model.save_pretrained(out)                       # chỉ lưu adapter LoRA
    if cfg["train_projector"]:                       # projector nằm NGOÀI adapter → lưu riêng
        torch.save(projector_state(model), out / "extra_trainable.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int)                  # E4: đổi rank (alpha = 2*r)
    ap.add_argument("--train-fraction", type=float, default=1.0)   # E4: đổi kích thước dữ liệu
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])   # E4: đổi thang
    ap.add_argument("--max-steps", type=int)             # chỉ để chạy thử nhanh
    ap.add_argument("--batch-size", type=int)            # chỉ dùng khi hết VRAM (giữ batch*accum = 8)
    ap.add_argument("--grad-accum", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.rank:
        cfg["lora"]["r"], cfg["lora"]["alpha"] = args.rank, 2 * args.rank
    cfg["batch_size"] = args.batch_size or cfg["batch_size"]
    cfg["grad_accum"] = args.grad_accum or cfg["grad_accum"]
    max_steps = args.max_steps or cfg["max_steps"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    model, processor = load_model()
    model, n_matched = apply_setup(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    print(f"{cfg['name']}: {n_matched} module LoRA, {n_trainable:,} tham số huấn luyện")

    entries = load_split("train", fraction=args.train_fraction, seed=cfg["seed"])
    data = SFTData(entries, processor, args.coord)
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(data, batch_size=cfg["batch_size"], shuffle=True, drop_last=True, generator=gen,
                        collate_fn=lambda b: collate(b, processor.tokenizer.pad_token_id))
    print(f"{len(entries)} ảnh train, coord={args.coord}")

    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    warm = max(1, int(0.05 * max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm))))
    scaler = torch.amp.GradScaler("cuda")

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0, losses, batches = time.time(), [], forever(loader)
    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(cfg["grad_accum"]):
            batch = {k: v.to("cuda") for k, v in next(batches).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / cfg["grad_accum"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: nguy cơ tràn số fp16")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        losses.append(step_loss)
        if step % 10 == 0 or step == 1:
            print(f"step {step}/{max_steps} loss={step_loss:.4f} lr={sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / 60:.1f}ph vram={torch.cuda.max_memory_allocated() / 2**30:.2f}GB",
                  flush=True)
        if step % 100 == 0:
            save(model, out, cfg)

    save(model, out, cfg)
    run = {"name": cfg["name"], "trainable_params": n_trainable, "matched_modules": n_matched,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1), "coord": args.coord,
           "train_images": len(entries), "rank": cfg["lora"]["r"], "max_steps": max_steps,
           "batch_size": cfg["batch_size"], "grad_accum": cfg["grad_accum"], "losses": losses}
    (out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    print({k: v for k, v in run.items() if k != "losses"})


if __name__ == "__main__":
    main()
