"""Bước 4 (không cần GPU): kiểm tra số token ảnh và mask loss trên một ví dụ."""
from transformers import AutoProcessor

from .data import load_image, load_split, make_prompt, objects_to_answer
from .modeling import MODEL_ID, encode_example

proc = AutoProcessor.from_pretrained(MODEL_ID)
e = next(x for x in load_split("train") if len(x["objects"]) >= 2)
answer = objects_to_answer(e["objects"], width=e["width"], height=e["height"])
ex = encode_example(proc, load_image(e), make_prompt(), answer)
ids, labels = ex["input_ids"][0], ex["labels"][0]
img_id = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
print("các khoá:", list(ex))
print("token ảnh:", int((ids == img_id).sum()), "(mong đợi 220)")
print("tổng token:", len(ids), "| token có loss:", int((labels != -100).sum()))
print("phần có loss:", repr(proc.tokenizer.decode(ids[labels != -100])))
print("đáp án      :", answer)
assert (labels[ids == img_id] == -100).all(), "token ảnh không được có loss"
