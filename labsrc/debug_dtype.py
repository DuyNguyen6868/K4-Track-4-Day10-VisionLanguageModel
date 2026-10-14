"""Chẩn đoán 'Attempting to unscale FP16 gradients' và đếm tham số huấn luyện theo nhóm."""
import collections

import torch
import yaml

from .data import load_image, load_split, make_prompt, objects_to_answer
from .modeling import PROJ_KEYS, apply_setup, collate, encode_example, load_model

cfg = yaml.safe_load(open("configs/e3.yaml", encoding="utf-8"))
model, processor = load_model()
model, n_matched = apply_setup(model, cfg)


def group(name):
    if "lora_" in name:
        return "lora_vit" if ".visual." in name else "lora_llm"
    return next((k for k in PROJ_KEYS if k in name), "khác")


train, frozen = collections.defaultdict(lambda: [0, collections.Counter()]), collections.Counter()
for name, p in model.named_parameters():
    g = group(name)
    if p.requires_grad:
        train[g][0] += p.numel()
        train[g][1][str(p.dtype)] += 1
    elif g in PROJ_KEYS:
        frozen[g] += p.numel()
print("--- tham số ĐANG huấn luyện theo nhóm (số lượng, {dtype: số tensor}) ---")
for g, (num, c) in train.items():
    print(f"{g:30s} {num:>13,} {dict(c)}")
print("--- tham số projector còn BỊ ĐÓNG BĂNG:", dict(frozen))

e = load_split("train")[0]
ans = objects_to_answer(e["objects"], width=e["width"], height=e["height"])
batch = collate([encode_example(processor, load_image(e), make_prompt(), ans)], processor.tokenizer.pad_token_id)
batch = {k: v.to("cuda") for k, v in batch.items()}
model.train()
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(**batch).loss
loss.backward()
bad = [(n, str(p.dtype), str(p.grad.dtype)) for n, p in model.named_parameters()
       if p.grad is not None and p.grad.dtype != torch.float32]
print("tham số có grad không phải fp32:", len(bad))
for x in bad[:10]:
    print("  ", x)
nograd = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
print("requires_grad nhưng không có grad:", len(nograd), nograd[:5])
