"""Bước 5 (không cần GPU, không tải trọng số): đếm tham số từng khối và kiểm tra regex LoRA."""
import re

import torch
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

from .modeling import LLM_RE, MODEL_ID, VIT_RE

cfg = AutoConfig.from_pretrained(MODEL_ID)
with torch.device("meta"):                      # dựng khung model rỗng, không tốn RAM/VRAM
    m = Qwen3VLForConditionalGeneration._from_config(cfg)


def count(mod):
    return sum(p.numel() for p in mod.parameters())


vis = m.model.visual
print(f"tổng              : {count(m):>15,}")
print(f"language_model    : {count(m.model.language_model):>15,}")
print(f"visual (ViT+merger): {count(vis):>14,}")
print(f"  visual.blocks   : {count(vis.blocks):>15,}")
print(f"  visual.merger   : {count(vis.merger):>15,}")
print(f"  deepstack_merger: {count(vis.deepstack_merger_list):>15,}")
names = [n for n, _ in m.named_modules()]
print("module khớp LLM_RE:", sum(bool(re.fullmatch(LLM_RE, n)) for n in names), "(mong đợi 252)")
print("module khớp VIT_RE:", sum(bool(re.fullmatch(VIT_RE, n)) for n in names), "(mong đợi 96)")
