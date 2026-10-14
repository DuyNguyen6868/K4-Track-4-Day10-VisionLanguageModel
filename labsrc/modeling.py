"""Nạp model 4-bit, mã hoá ví dụ + mask loss, setup LoRA, suy luận."""
import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from .data import load_image

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROJ_KEYS = ("visual.merger", "visual.deepstack_merger_list")
LLM_RE = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_RE = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"


# "visual" một mình KHÔNG đủ trên transformers 5.19: merger/deepstack vẫn bị nén 4-bit (đã đo),
# nên liệt kê cả tên đầy đủ có tiền tố "model."
SKIP_QUANT = ["visual", "model.visual", "visual.merger", "visual.deepstack_merger_list", "lm_head"]


def assert_vit_not_quantized(model):
    import bitsandbytes as bnb
    bad = [n for n, m in model.named_modules()
           if isinstance(m, bnb.nn.Linear4bit) and "visual" in n.split(".")]
    assert not bad, f"{len(bad)} Linear trong ViT vẫn bị lượng tử hoá 4-bit, ví dụ: {bad[:3]}"


def load_model():
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_QUANT)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    assert_vit_not_quantized(model)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor


def user_msg(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


# ---------- Bước 4: ví dụ huấn luyện + mask loss ----------
def encode_example(processor, image, prompt, answer):
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    user = user_msg(image, prompt)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    assert torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]), "prompt không là tiền tố"
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = -100                       # chỉ học phần trả lời
    out = dict(full_enc)
    out["labels"] = labels
    return out


def collate(batch, pad_id):
    out = {}
    for k in batch[0]:
        if k in ("pixel_values", "image_grid_thw"):          # nối, KHÔNG pad
            out[k] = torch.cat([b[k] for b in batch], 0)
        else:                                                # pad phải đến cùng độ dài
            fill = {"input_ids": pad_id, "labels": -100}.get(k, 0)
            width = max(b[k].shape[1] for b in batch)
            out[k] = torch.cat([torch.nn.functional.pad(b[k], (0, width - b[k].shape[1]), value=fill)
                                for b in batch], 0)
    return out


# ---------- Bước 5: setup LoRA / đóng băng ----------
def apply_setup(model, cfg):
    for p in model.parameters():
        p.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    regex = LLM_RE if cfg["lora"]["targets"] == "llm" else f"{LLM_RE}|{VIT_RE}"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    lora = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                      lora_dropout=cfg["lora"]["dropout"], target_modules=regex)
    model = get_peft_model(model, lora)
    if cfg["train_projector"]:
        for n, p in model.named_parameters():
            if any(k in n for k in PROJ_KEYS):
                p.data = p.data.float()
                p.requires_grad = True
    for p in model.parameters():               # GradScaler cần gradient fp32
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model, len(matched)


def projector_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters()
            if any(k in n for k in PROJ_KEYS)}


def load_projector_state(model, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for n, v in state.items():
            params[n].copy_(v)                 # fp32 → fp16 (KeyError nếu tên không khớp)
    return len(state)


# ---------- Bước 3/7: suy luận ----------
def run_inference(model, processor, entries, prompt, out_path, batch_size=4, max_new_tokens=256):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():                      # chạy tiếp được sau khi Colab ngắt
        done = {json.loads(ln)["image_id"] for ln in out_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()}
    todo = [e for e in entries if e["image_id"] not in done]
    model.eval()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            messages = [[user_msg(load_image(e), prompt)] for e in chunk]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to("cuda")
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for e, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{len(done) + i + len(chunk)}/{len(entries)}", end="\r")
    print()
