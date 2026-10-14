# Hướng dẫn chi tiết hoàn thành Lab Day 10 trên Google Colab

> Bổ sung cho [`README.md`](README.md) và [`GUIDE.md`](GUIDE.md): GUIDE.md cho **pseudo code và câu hỏi**, file này cho **thao tác từng ô (cell) trên Colab T4**, kèm mã tham chiếu chạy theo đúng thứ tự E0 → E4.
>
> **Đọc trước khi dùng:**
> - Mã tham chiếu bên dưới viết theo pseudo code của GUIDE.md và tên module Qwen3-VL trong đề. Phần thuần Python (nhãn YOLO→JSON, vẽ box, reward) đã được chạy thử trên máy; **phần cần GPU (nạp model, huấn luyện, suy luận) chưa được chạy trên T4 thật**. Nếu một API lệch phiên bản (`transformers==5.19.0`), xem mục 11 và đọc thông báo lỗi.
> - Đề yêu cầu **mã, số liệu và báo cáo là của bạn**, và giảng viên có thể gọi vấn đáp 5 phút. Hãy đọc hiểu từng hàm (mỗi hàm có chú thích ứng với một bước trong GUIDE), chạy, rồi tự trả lời `REPORT.md` bằng số của chính bạn. Đừng chép nguyên văn phần "gợi ý" vào báo cáo.
> - **Không sửa** `scripts/` và `tests/` (trừ 10 điểm). Mã của bạn đặt ở thư mục mới `labsrc/` và `my_tests/`.

## Mục lục

0. [Bản đồ bài lab](#0-bản-đồ-bài-lab)
1. [Chuẩn bị một lần (ngoài Colab)](#1-chuẩn-bị-một-lần-ngoài-colab)
2. [Mở session Colab: môi trường + dữ liệu](#2-mở-session-colab-môi-trường--dữ-liệu)
3. [Bước 1: kiểm tra dữ liệu](#3-bước-1-kiểm-tra-dữ-liệu)
4. [Bước 2: nhãn YOLO → JSON thang 0–1000](#4-bước-2-nhãn-yolo--json-thang-01000)
5. [Bước 3: E0 zero-shot](#5-bước-3-e0-zero-shot)
6. [Bước 4–6: mã huấn luyện và chạy thử](#6-bước-46-mã-huấn-luyện-và-chạy-thử)
7. [Bước 7: E1, E2, E3](#7-bước-7-e1-e2-e3)
8. [Bước 8: E4 ablation](#8-bước-8-e4-ablation)
9. [Bước 9: E5 tuỳ chọn](#9-bước-9-e5-tuỳ-chọn-10-điểm)
10. [Ảnh minh hoạ, bảng tổng hợp và cách tìm số cho 10 câu báo cáo](#10-ảnh-minh-hoạ-bảng-tổng-hợp-và-số-liệu-cho-10-câu-báo-cáo)
11. [Nộp bài](#11-nộp-bài) và [Sự cố thường gặp](#12-sự-cố-thường-gặp)

---

## 0. Bản đồ bài lab

**Bài toán:** ảnh dashcam 640×360 → JSON `[{"label": ..., "bbox_2d": [x1,y1,x2,y2]}]` thang 0–1000. Finetune Qwen3-VL-4B bằng QLoRA, so sánh 4 cách đóng băng/mở khối.

| Mã | Làm gì | Chạy lệnh | Sinh file |
|---|---|---|---|
| E0 | zero-shot | `labsrc.infer` | `report/evidence/e0.json` |
| E1 | LoRA LLM | `labsrc.train` → `labsrc.infer` | `e1.json` |
| E2 | + projector | như trên | `e2.json` |
| E3 | + LoRA ViT | như trên | `e3.json` |
| E4 | ablation 1 biến | như trên | `e4_<biến>_<giá trị>.json` (cả hai giá trị) |
| E5 | GRPO (tuỳ chọn) | | +10 điểm |

**Ước lượng thời gian** (T4 free; README nói 20–30 phút/lần train, hãy **ghi thời gian thật** vào báo cáo):

| Việc | Ước lượng |
|---|---|
| Cài đặt + dữ liệu + pytest | 10–15 phút |
| E0 (311 ảnh test, tối đa 256 token/ảnh) | 15–25 phút |
| Mỗi lần train E1/E2/E3 (300 bước) | 20–30 phút (theo README) |
| Mỗi lần suy luận test sau train | 10–15 phút |
| E4 (1 lần train + 1 lần suy luận nếu chọn `data_size`) | 30–45 phút |
| Viết báo cáo, vẽ ảnh | 40–60 phút |

Tổng ≈ 4–5 giờ; Colab free có hạn mức GPU theo ngày, nên **chia làm 2 buổi** và lưu mọi thứ lên Drive (mục 2).

**Kế hoạch commit (cần ≥ 5, mỗi mốc một commit):**

1. `chore: môi trường + kiểm tra dữ liệu` (mục 2–3)
2. `feat: nhãn JSON 0-1000 + E0 zero-shot` (mục 4–5)
3. `feat: mã huấn luyện/suy luận + E1`
4. `feat: E2, E3`
5. `feat: E4 ablation`
6. `docs: ảnh minh hoạ + REPORT.md`

**Cấu trúc mã bạn sẽ có:**

```
labsrc/data.py          nhãn, prompt, đọc split           (bước 2)
labsrc/modeling.py      nạp model, mask loss, LoRA, suy luận (bước 3-5, 7)
labsrc/train.py         vòng huấn luyện                      (bước 6)
labsrc/infer.py         suy luận + nạp lại projector         (bước 3, 7)
labsrc/viz.py           vẽ ảnh minh hoạ
labsrc/count_params.py  đếm tham số, kiểm tra regex LoRA (không cần GPU)
labsrc/check_encode.py  kiểm tra token ảnh + mask loss (không cần GPU)
labsrc/reward.py        reward IoU (E5)
my_tests/test_reward.py test cho reward (E5)
```

---

## 1. Chuẩn bị một lần (ngoài Colab)

**1.1 Repo GitHub công khai.** Trên máy bạn (thư mục lab hiện có sẵn git), tạo repo **public** trống trên GitHub rồi:

```bash
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

**1.2 Khoá Kaggle** (tải dữ liệu): kaggle.com → *Settings* → *API* → tạo token. Bạn cần `KAGGLE_USERNAME` và `KAGGLE_KEY`. Không có tài khoản Kaggle: tải zip dataset bằng tay, upload lên Drive và dùng `--zip <đường dẫn>` (xem mục 12).

**1.3 Token GitHub** để push từ Colab: GitHub → *Settings* → *Developer settings* → *Fine-grained tokens* → chỉ chọn **repo này**, quyền *Contents: Read and write*.

**1.4 Colab Secrets** (biểu tượng chìa khoá ở thanh bên trái của Colab): thêm `KAGGLE_USERNAME`, `KAGGLE_KEY`, `GITHUB_TOKEN`, và (tuỳ chọn) `HF_TOKEN`; bật công tắc **Notebook access** cho từng khoá. **Không dán khoá vào ô mã, không commit `.env`.**

**1.5 Mở notebook.** Mở `lab_day10.ipynb` từ GitHub trong Colab (File → Open notebook → tab GitHub), hoặc upload lên Colab. *Runtime → Change runtime type → **T4 GPU***. Giữ nguyên các tiêu đề `## 0.` … `## 8.` của notebook (test của đề kiểm tra chúng). Bạn dán các ô trong hướng dẫn này vào đúng mục tương ứng của notebook.

---

## 2. Mở session Colab: môi trường + dữ liệu

> Colab free **xoá toàn bộ `/content` khi ngắt session**. Vì vậy: repo clone mỗi lần, còn `work/` (dự đoán, adapter) được nối sang **Google Drive** để không mất.

**Ô 2.1: GPU, Drive, repo** (chạy mỗi session; sửa `GH_USER`, `GH_REPO`, tên, email)

```python
import os
from google.colab import drive, userdata

drive.mount('/content/drive')
GH_USER, GH_REPO = "<user-github>", "<ten-repo>"
os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")
os.environ["GITHUB_TOKEN"] = userdata.get("GITHUB_TOKEN")

!nvidia-smi -L
%cd /content
!test -d {GH_REPO} || git clone https://github.com/{GH_USER}/{GH_REPO}.git
%cd /content/{GH_REPO}
!git config user.name "<Họ tên>"
!git config user.email "<email-github>"
# token chỉ nằm trong .git/config của runtime Colab (không commit); đừng in ra
# GH_USER và GH_REPO phải đã được gán ở trên, nếu không remote sẽ chứa chữ {GH_USER}
!git remote set-url origin https://$GITHUB_TOKEN@github.com/{GH_USER}/{GH_REPO}.git
!git pull origin main
# work/ -> Drive: adapter và dự đoán sống sót qua các lần ngắt session
!mkdir -p /content/drive/MyDrive/lab_day10/work && ln -sfn /content/drive/MyDrive/lab_day10/work work
# `work/` trong .gitignore chỉ khớp THƯ MỤC, không khớp symlink `work` → phải loại riêng, nếu không `git add -A` sẽ commit symlink
!grep -qx work .git/info/exclude || echo work >> .git/info/exclude
!ls -la work | head -3
```

`nvidia-smi -L` phải in ra `Tesla T4`. Nếu không, đổi lại runtime.

**Ô 2.2: cài thư viện, tải và chia dữ liệu, chạy test của đề**

```python
!pip -q install -r requirements-colab.txt
!pip -q uninstall -y torchao      # Colab cài sẵn torchao cũ (0.10) làm peft 0.21 báo ImportError; bài lab không dùng
!python scripts/prepare_data.py --kaggle --work /content/raw
!python -m pytest -q
```

Kết quả mong đợi:
- `prepare_data.py` in `dataset_version` và số ảnh/box từng split: **train 1.422 · val 276 · test 311**. Ghi chú trong `tests/test_configs_and_docs.py` cho biết `dataset_version` đo trên dữ liệu thật là `d10-21e4070301`; nếu bạn ra mã khác, dữ liệu tải về không phải bản chuẩn.
- `pytest` toàn bộ pass.
- `--work /content/raw` đặt bản tải thô ngoài Drive (tránh ghi vài trăm MB vào Drive).

> **Đừng `import torch`/`transformers` trong kernel của notebook.** Mọi việc nặng chạy bằng `!python -m labsrc.…` (tiến trình riêng, tự giải phóng VRAM khi xong). Cách này tránh phải restart runtime sau `pip install` và tránh tràn VRAM giữa các thí nghiệm.

**Ô 2.3: hàm tiện ích để commit/push** (dán một lần mỗi session)

```python
import subprocess
def push(msg):
    for cmd in (["git", "add", "-A"], ["git", "commit", "-m", msg], ["git", "push", "origin", "HEAD:main"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout + r.stderr)
```

Trước khi `push`, kiểm tra `!git status --short`: không được có `data/`, `work/`, `.env`, `*.safetensors`, `*.bin`, `*.pt` (đã nằm trong `.gitignore`). `work` (symlink sang Drive) được loại bằng `.git/info/exclude` ở ô 2.1; nếu `git status` vẫn hiện `work`, chạy lại dòng `echo work >> .git/info/exclude`.

---

## 3. Bước 1: kiểm tra dữ liệu

```python
import json
from PIL import Image

def load_split(name):          # bước 2 sẽ có bản chính thức trong labsrc/data.py
    with open(f"data/splits/{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_image(e):
    return Image.open(f"data/{e['image_path']}").convert("RGB")

meta = json.load(open("data/splits/meta.json"))
print("dataset_version:", meta["dataset_version"])
sess = {s: {e["session"] for e in load_split(s)} for s in ("train", "val", "test")}
print({s: len(v) for s, v in sess.items()}, "phiên mỗi split")
assert not (sess["train"] & sess["val"] or sess["train"] & sess["test"] or sess["val"] & sess["test"]), "rò rỉ phiên!"
for s, c in meta["counts"].items():
    print(s, c["images"], "ảnh", c["boxes"])
```

Vẽ 6 ảnh có box (ô này dùng lại `load_split`/`load_image` vừa định nghĩa ở trên, nên chạy ô trên trước):

```python
import matplotlib.pyplot as plt
from PIL import ImageDraw
rows = load_split("train")[::200][:6]
fig, ax = plt.subplots(2, 3, figsize=(15, 6))
for a, e in zip(ax.ravel(), rows):
    im = load_image(e); d = ImageDraw.Draw(im)
    for o in e["objects"]:
        xc, yc, w, h = o["yolo"]
        d.rectangle([(xc-w/2)*640, (yc-h/2)*360, (xc+w/2)*640, (yc+h/2)*360], outline="lime", width=2)
        d.text(((xc-w/2)*640+2, (yc-h/2)*360+2), o["label"], fill="yellow")
    a.imshow(im); a.set_title(e["session"]); a.axis("off")
plt.show()
```

**Ghi lại cho Câu 6:** số ảnh và số box từng lớp ở mỗi split (in ở trên), cùng với việc không phiên nào xuất hiện ở hai split. Lưu ý val có rất ít pothole (66 box).

Commit: `push("chore: môi trường + kiểm tra dữ liệu")`

---

## 4. Bước 2: nhãn YOLO → JSON thang 0–1000

Tạo gói `labsrc` và file `data.py`:

```python
!mkdir -p labsrc my_tests report/figures && touch labsrc/__init__.py
```

```python
%%writefile labsrc/data.py
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
```

Điểm cần hiểu (GUIDE bước 2):
- YOLO là `(tâm x, tâm y, rộng, cao)` trong [0,1]; Qwen3-VL cần `(x1,y1,x2,y2)` thang 0–1000 nên `x1 = (xc - w/2) * 1000`, ...
- Sắp box theo `(y1, x1)` để thứ tự đáp án **ổn định**, model dễ học.
- Prompt nói rõ "thang 0–1000": thang là một phần của hợp đồng giữa nhãn, prompt và bộ chấm.
- Tham số `coord="pixel"` có sẵn cho ablation E4 `coord_scale`.

**Kiểm tra (đề yêu cầu: quy ngược về pixel và vẽ lại, phải trùng box gốc):**

```python
from labsrc.data import *
from scripts.score import _gt_boxes, parse_output

bad = 0
for e in load_split("train"):
    ans = objects_to_answer(e["objects"], width=e["width"], height=e["height"])
    gt = sorted(v for g in _gt_boxes(e) for v in g["bbox"])
    pr = sorted(v for p in parse_output(ans) for v in p["bbox"])
    bad += len(gt) != len(pr) or max((abs(a - b) for a, b in zip(gt, pr)), default=0) > 1.5
print("ảnh lệch > 1.5 đơn vị:", bad, "/ 1422 (chỉ nên là vài ảnh có box chạm mép)")

e = next(x for x in load_split("train") if len(x["objects"]) >= 2)
print(objects_to_answer(e["objects"], width=e["width"], height=e["height"]))
```

Phần "vẽ lại bằng JSON và so với box gốc" làm ở ô 5.0 (sau khi có `viz.py`).

**Câu hỏi GUIDE:** vì sao Qwen3-VL dùng thang 0–1000 chứ không dùng pixel tuyệt đối? Gợi ý hướng nghĩ: kích thước ảnh đầu vào bị processor resize (ảnh 640×360 thành lưới 640×352), nên toạ độ pixel gốc không còn ý nghĩa ổn định; thang tương đối thì độc lập độ phân giải.

---

## 5. Bước 3: E0 zero-shot

**Ô 5.0: dán mã nền.** `modeling.py` chứa mọi thứ cần model (nạp 4-bit, mask loss, LoRA, suy luận); giải thích chi tiết ở mục 6.

```python
%%writefile labsrc/modeling.py
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
```

`infer.py` (bước 3 và 7): nạp model, nạp adapter (nếu có), nạp lại projector (nếu có), suy luận. Đọc `coord` từ `run.json` của adapter nên E4 `pixel` không bị nhầm prompt. E0 chạy **không** có `--adapter`.

```python
%%writefile labsrc/infer.py
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
```

`viz.py` vẽ box đúng (xanh) và box dự đoán (đỏ), dùng cho kiểm tra nhãn ở đây và ảnh minh hoạ ở mục 10.

```python
%%writefile labsrc/viz.py
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
```

`count_params.py` đếm tham số từng khối trên khung model rỗng (kiểu `meta`, không tải trọng số, không cần GPU):

```python
%%writefile labsrc/count_params.py
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
```

Kiểm tra nhãn quy ngược về pixel và vẽ lại (xanh phải trùng đỏ):

```python
from labsrc.viz import draw
from IPython.display import Image as Show
e = next(x for x in load_split("train") if len(x["objects"]) >= 2)
ans = objects_to_answer(e["objects"], width=e["width"], height=e["height"])
draw(e, ans, "work/roundtrip.png", "roundtrip")
Show("work/roundtrip.png")
```

**Ô 5.1: kiểm tra cây model.** Xác nhận số tham số từng khối (Câu 1) và regex LoRA đủ 252 + 96 module:

```python
!python -m labsrc.count_params
```

Ghi lại các con số in ra cho Câu 1. Nếu số khớp LLM/ViT khác 252/96: dừng, đọc tên module thật (xem mục 12).

**Ô 5.2: gỡ lỗi nhanh trên 40 ảnh val** (lần đầu tải model ≈ 8 GB từ Hugging Face, mất vài phút)

```python
!python -m labsrc.infer --split val --limit 40 --out work/dbg_e0_val40.jsonl
```

Xem 5 output thô:

```python
import json
for ln in open("work/dbg_e0_val40.jsonl").readlines()[:5]:
    r = json.loads(ln); print(r["image_id"], "->", repr(r["raw"][:300]), "\n")
```

Chấm riêng 40 ảnh (tạo split val nhỏ; `score.py` chấm mọi ảnh trong split nên nếu không thu nhỏ split thì các ảnh còn lại bị tính là thiếu):

```python
import os, shutil
os.makedirs("work/val40", exist_ok=True)
with open("work/val40/val.jsonl", "w", encoding="utf-8") as f:
    for e in load_split("val", 40): f.write(json.dumps(e, ensure_ascii=False) + "\n")
shutil.copy("data/splits/meta.json", "work/val40/meta.json")
!python scripts/score.py --split val --splits-dir work/val40 --pred work/dbg_e0_val40.jsonl --name dbg --out work/dbg_e0.json
```

Dùng đúng mẹo này (đổi `--pred`) cho mọi lần gỡ lỗi sau; **file chính thức luôn là `--split test` ghi vào `report/evidence/`**.

**Ô 5.3: E0 chính thức trên test** (311 ảnh, ≈ 15–25 phút; tự chạy tiếp nếu bị ngắt giữa chừng vì `run_inference` bỏ qua ảnh đã có trong file)

```python
!python -m labsrc.infer --out work/e0.jsonl
!python scripts/score.py --pred work/e0.jsonl --name E0 --out report/evidence/e0.json
```

**Ghi lại cho Câu 2:** `parse_valid_rate`; hai lỗi phổ biến nhất (sai thang, bịa box, sai lớp, JSON hỏng, bọc ```` ```json ````, chữ thừa); và **một output thô** làm ví dụ. Dùng `ln` ở ô trên để xem thêm output.

Commit: `push("feat: nhãn JSON 0-1000 + E0 zero-shot")`

---

## 6. Bước 4–6: mã huấn luyện và chạy thử

**`labsrc/modeling.py` (đã dán ở ô 5.0)** gồm: nạp model 4-bit (bước 3), `encode_example`/`collate` với mask loss (bước 4), `apply_setup` cho E1–E3 (bước 5), lưu/nạp projector, và `run_inference`.

**Vì sao viết như vậy (đây là phần vấn đáp hay hỏi):**
- `llm_int8_skip_modules=["visual","lm_head"]`: ViT (kèm projector) và `lm_head` ở fp16. Nếu ViT bị lượng tử hoá 4-bit thì trọng số merger là trọng số nén (Params4bit), optimizer không cập nhật được đầy đủ (Câu 5).
- `encode_example`: mã hoá hai lần (có/không có câu trả lời); phần prompt là tiền tố, nên `labels[:, :n] = -100` che mọi thứ trừ đáp án (+ `<|im_end|>`). `assert` bắt trường hợp tiền tố không khớp (Câu 4).
- `collate`: `pixel_values` và `image_grid_thw` **nối**, không pad; mọi tensor còn lại pad phải (labels đệm −100).
- `apply_setup`: đóng băng hết → bật gradient checkpointing → LoRA bằng **regex** trên tên module (`re.fullmatch`, tên bắt đầu bằng `model.language_model…` / `model.visual…`) → với E2/E3 mở khối `visual.merger` + `visual.deepstack_merger_list` và **ép float32**.
- Float32 vì `GradScaler.unscale_` từ chối gradient fp16 (lỗi "Attempting to unscale FP16 gradients"), nên mọi tham số huấn luyện phải là fp32 (Câu 5).
- Projector **không nằm trong adapter**: `train.py` lưu riêng `extra_trainable.pt`, `infer.py` nạp lại. Quên bước này thì E2/E3 ≈ E1 (mục 12).

**`labsrc/train.py`** (bước 6): AdamW lr 2e-4, warmup 5% rồi cosine, GradScaler, autocast fp16, clip 1.0, dừng khi loss không hữu hạn, lưu mỗi 100 bước và ghi `run.json`.

```python
%%writefile labsrc/train.py
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
```

### Ô 6.1: kiểm tra số token ảnh và mask loss (không cần GPU)

```python
%%writefile labsrc/check_encode.py
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
```

```python
!python -m labsrc.check_encode
```

Phải thấy: **token ảnh = 220**, và "phần có loss" **đúng bằng đáp án JSON** (cộng `<|im_end|>`). Không đúng: sửa trước khi huấn luyện (Câu 4).

### Ô 6.2: chạy thử 3 bước trên cấu hình nặng nhất (E3)

Chạy thử E3 trước, vì nếu E3 vừa VRAM thì E1/E2 chắc chắn vừa. Mục đích: bắt lỗi sớm thay vì mất 25 phút mới biết.

```python
!python -m labsrc.train --config configs/e3.yaml --out work/adapters/smoke --max-steps 3
```

Mong đợi: `348 module LoRA`, loss hữu hạn, không OOM, `run.json` và `extra_trainable.pt` được ghi vào `work/adapters/smoke`. Sau đó kiểm tra suy luận qua adapter vừa lưu (phải in "đã nạp lại … tensor projector"):

```python
!python -m labsrc.infer --adapter work/adapters/smoke --split val --limit 8 --out work/dbg_smoke.jsonl
```

Xoá bản thử: `!rm -rf work/adapters/smoke work/dbg_smoke.jsonl`.

### Ô 6.3: chẩn đoán dtype và lượng tử hoá (chạy khi gặp `Attempting to unscale FP16 gradients` hoặc số tham số lạ)

```python
%%writefile labsrc/debug_dtype.py
"""Chẩn đoán dtype/lượng tử hoá của ViT + projector, và lỗi 'Attempting to unscale FP16 gradients'."""
import collections

import torch
import yaml

from .data import load_image, load_split, make_prompt, objects_to_answer
from .modeling import PROJ_KEYS, apply_setup, collate, encode_example, load_model

cfg = yaml.safe_load(open("configs/e3.yaml", encoding="utf-8"))
model, processor = load_model()          # assert_vit_not_quantized chạy trong load_model
inner = model.model
for label, mod in (("visual.blocks[0].attn.qkv", inner.visual.blocks[0].attn.qkv),
                   ("visual.merger.linear_fc1", inner.visual.merger.linear_fc1),
                   ("language_model q_proj", inner.language_model.layers[0].self_attn.q_proj),
                   ("lm_head", model.lm_head)):
    print(f"{label:28s} {type(mod).__name__:12s} weight={mod.weight.dtype} numel={mod.weight.numel():,}")
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
print("tổng:", sum(v[0] for v in train.values()), "| projector còn bị đóng băng:", dict(frozen))

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
```

```python
!python -m labsrc.debug_dtype
```

Kết quả đúng: `qkv`, `merger.linear_fc1` và `lm_head` đều là `Linear` (không phải `Linear4bit`) với `weight=torch.float16`; `visual.merger.linear_fc1` có `numel=16,777,216`; E3 có tổng ≈ 148,6M tham số huấn luyện (LoRA LLM 33,0M + LoRA ViT 6,3M + merger 27,3M + deepstack ≈ 82M); "tham số có grad không phải fp32: 0".

**Sự cố đã gặp:** `llm_int8_skip_modules=["visual","lm_head"]` (đúng như GUIDE) **không** loại ViT/projector khỏi lượng tử hoá trên `transformers==5.19.0`: `merger` chỉ còn một nửa số phần tử (trọng số nén 4-bit), trọng số không có gradient, và bitsandbytes tự cast bias về fp16 nên GradScaler báo "Attempting to unscale FP16 gradients". `load_model` đã dùng `SKIP_QUANT` (có thêm `model.visual`, `visual.merger`, `visual.deepstack_merger_list`) và `assert_vit_not_quantized` để bắt lỗi này ngay lúc nạp. **Mọi kết quả E0 chạy bằng bản `load_model` cũ phải chạy lại** (xoá `work/e0.jsonl`, `work/dbg_e0_val40.jsonl`).

Commit: `push("feat: mã huấn luyện/suy luận")`

---

## 7. Bước 7: E1, E2, E3

Hàm dựng lệnh chấm (lấy `trainable_params`, `peak_vram_gb`, `train_minutes` từ `run.json` để **khớp bằng chứng với báo cáo**):

```python
import json
def score_cmd(name, key):
    r = json.load(open(f"work/adapters/{key}/run.json"))
    meta = " ".join(f"--meta {k}={r[k]}" for k in ("trainable_params", "peak_vram_gb", "train_minutes"))
    return (f"python scripts/score.py --pred work/{key}.jsonl --name {name} "
            f"--out report/evidence/{key}.json {meta}")
```

Với **mỗi** setup, chạy lần lượt 3 ô (ví dụ E1; E2 và E3 đổi `e1` → `e2`/`e3`):

```python
# (a) huấn luyện ≈ 20–30 phút; giữ tab Colab mở
!PYTORCH_ALLOC_CONF=expandable_segments:True python -m labsrc.train --config configs/e1.yaml --out work/adapters/e1 --batch-size 1 --grad-accum 8
```
```python
# (b) suy luận test (≈ 10–15 phút)
!python -m labsrc.infer --adapter work/adapters/e1 --out work/e1.jsonl
```
```python
# (c) chấm
cmd = score_cmd("E1", "e1")
!{cmd}
```

**Kiểm tra sau mỗi lần:**
- Log huấn luyện: loss giảm đều, hữu hạn. In đường cong nếu muốn: `json.load(open("work/adapters/e1/run.json"))["losses"]`.
- E1: `parse_valid_rate` ≥ 0,95 và F1 hơn E0 ≥ 0,10 (đề lấy hai điều kiện này làm điểm tự động). Nếu không: kiểm tra mask loss (mục 6.1) và thang toạ độ.
- E2/E3: log suy luận có dòng "đã nạp lại N tensor projector". Nếu F1 của E2/E3 **giống hệt** E1 thì nhiều khả năng projector chưa được nạp.
- `trainable_params` của E2 lớn hơn E1 khoảng **109M**; E3 có **348** module LoRA.

**Dùng `--batch-size 1 --grad-accum 8` cho MỌI lần train (E1–E4)**: với batch 2 ảnh, T4 bị `CUDA out of memory` ngay ở phép tính loss khi gặp ảnh có nhiều box (đáp án JSON dài, logits 151.936 từ × độ dài chuỗi × fp32). Số ảnh mỗi bước tối ưu vẫn là 8 như cấu hình gốc. Phải dùng **cùng** giá trị cho cả bốn setup để so sánh công bằng, và ghi rõ trong báo cáo (`run.json` đã lưu `batch_size`, `grad_accum`). **Đừng sửa `configs/*.yaml`**: test của đề kiểm tra các config chỉ khác nhau đúng biến thí nghiệm. `PYTORCH_ALLOC_CONF=expandable_segments:True` giảm phân mảnh bộ nhớ.

Chạy dài, nên chạy nền và theo dõi log để sống sót khi trình duyệt chập chờn:

```python
!PYTORCH_ALLOC_CONF=expandable_segments:True nohup python -m labsrc.train --config configs/e1.yaml --out work/adapters/e1 --batch-size 1 --grad-accum 8 > work/train_e1.log 2>&1 &
!tail -n 5 work/train_e1.log     # chạy lại lệnh này để xem tiến độ
```

Sau E1, E2, E3 bạn có `report/evidence/e0–e3.json`. Commit riêng: `push("feat: E1")`, `push("feat: E2, E3")`.

Adapter và dự đoán đã nằm trên Drive nhờ symlink `work/`.

> **Chọn "setup tốt nhất" cho E4:** so F1 và **CI95** của E1–E3. Nếu E2/E3 chỉ hơn E1 trong phạm vi nhiễu (CI95 chồng nhau mạnh), chọn setup rẻ hơn rồi giải thích: đó chính là nội dung Câu 7 và Câu 8. Lưu ý trung thực: chọn theo F1 trên test là hơi "nhìn trộm"; hãy nói rõ điều này trong báo cáo.

---

## 8. Bước 8: E4 ablation

Chọn **một** biến, đổi **đúng một thứ** so với setup tốt nhất (gọi là `eK`, `K` ∈ {1,2,3}), và có **cả hai giá trị** trong `report/evidence/`. Giá trị trùng setup gốc thì **dùng lại dự đoán của `eK`** (không phải train lại), nhưng vẫn phải chấm ra một file `e4_*.json` riêng.

| Biến | Giá trị | Phải train lại | Chi phí thêm |
|---|---|---|---|
| `data_size` (khuyên chọn) | 0,1 và 1,0 | chỉ 0,1 | 1 lần train + 1 lần suy luận |
| `rank` | 4 và 64 (alpha = 2·r; r=16 là gốc) | cả hai | 2 lần train + 2 lần suy luận |
| `coord_scale` | `norm1000` và `pixel` | chỉ `pixel` | 1 lần train + 1 lần suy luận (đổi cả nhãn, prompt, và chấm `--coord pixel`) |

**Ví dụ `data_size`** (giả sử setup tốt nhất là E2):

```python
# giá trị 0.1: train lại với 10% ảnh train (≈ 142 ảnh, vẫn 300 bước)
!PYTORCH_ALLOC_CONF=expandable_segments:True python -m labsrc.train --config configs/e2.yaml --out work/adapters/e4_data_size_0.1 --train-fraction 0.1 --batch-size 1 --grad-accum 8
!python -m labsrc.infer --adapter work/adapters/e4_data_size_0.1 --out work/e4_data_size_0.1.jsonl
```
```python
import json
best = "e2"      # đổi thành setup có F1 cao nhất trong E1-E3 (kết quả thật của bạn)

def meta_args(key):
    r = json.load(open(f"work/adapters/{key}/run.json"))
    return " ".join(f"--meta {k}={r[k]}" for k in ("trainable_params", "peak_vram_gb", "train_minutes"))

# giá trị 0.1: dự đoán và adapter vừa train
meta = meta_args("e4_data_size_0.1")
!python scripts/score.py --pred work/e4_data_size_0.1.jsonl --name E4_data_0.1 --out report/evidence/e4_data_size_0.1.json {meta}

# giá trị 1.0 chính là setup gốc: dùng lại dự đoán của nó (phải trùng config đã dùng để train bản 0.1)
!cp work/{best}.jsonl work/e4_data_size_1.0.jsonl
meta = meta_args(best)
!python scripts/score.py --pred work/e4_data_size_1.0.jsonl --name E4_data_1.0 --out report/evidence/e4_data_size_1.0.json {meta}
```

(Trong một ô Colab được phép trộn dòng Python với dòng bắt đầu bằng `!`. Nếu setup tốt nhất là E1 hoặc E3, đổi `e2` thành `e1`/`e3` và `--config` tương ứng.)

Ví dụ `rank`: thêm `--rank 4` / `--rank 64` và đặt tên `e4_rank_4`, `e4_rank_64`. Ví dụ `coord_scale`: `--coord pixel` khi train (infer tự đọc từ `run.json`), và khi chấm thêm `--coord pixel`:

```python
!python scripts/score.py --pred work/e4_coord_pixel.jsonl --coord pixel --name E4_pixel --out report/evidence/e4_coord_scale_pixel.json {meta}
```

**Câu 10 & Câu 8:** nhìn **CI95**. Hai giá trị chỉ "khác nhau thật" nếu khoảng tin cậy không chồng nhau nhiều (test chỉ 311 ảnh).

Commit: `push("feat: E4 ablation")`

---

## 9. Bước 9: E5 tuỳ chọn (+10 điểm)

Tách thành 3 phần điểm theo RUBRIC: hàm reward đúng + test (4), chạy GRPO có log (3), phân tích reward hacking + so với SFT kèm CI95 (3). Làm sau cùng, khi E0–E4 đã xong và đã commit.

**Reward (Python thuần, không cần GPU)** và **test** (chạy trước khi động đến GRPO):

```python
%%writefile labsrc/reward.py
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
```

```python
%%writefile my_tests/test_reward.py
import json

import pytest

from labsrc.reward import iou_reward

GT = [{"label": "pothole", "bbox": (100, 100, 300, 300)}]


def ans(*boxes):
    return json.dumps([{"label": l, "bbox_2d": list(b)} for l, b in boxes])


def test_perfect():
    assert iou_reward(ans(("pothole", (100, 100, 300, 300))), GT) == pytest.approx(1.0)


def test_broken_json():
    assert iou_reward("không phải json", GT) == 0.0


def test_wrong_class_only_gets_validity_bonus():
    assert iou_reward(ans(("crack", (100, 100, 300, 300))), GT) == pytest.approx(0.2)


def test_extra_box_is_penalised():
    r = iou_reward(ans(("pothole", (100, 100, 300, 300)), ("pothole", (600, 600, 700, 700))), GT)
    assert r == pytest.approx(0.2 + 0.8 * 1.0 / 2)


def test_empty_vs_empty():
    assert iou_reward("[]", []) == 1.0


def test_empty_answer_when_gt_exists():
    assert iou_reward("[]", GT) == pytest.approx(0.2)


def test_low_iou_not_matched():
    assert iou_reward(ans(("pothole", (250, 250, 500, 500))), GT) == pytest.approx(0.2)
```

```python
!python -m pytest -q my_tests
```

**Chạy GRPO** (`configs/e5.yaml`: `num_generations=4`, `max_completion_length=128`, ~60 bước, `lr=5e-6`, `beta=0`, `use_vllm=False`):
- Khởi tạo từ **adapter tốt nhất** (nạp bằng `PeftModel.from_pretrained(model, path, is_trainable=True)`); nếu là E2/E3, nạp lại cả projector.
- Dùng `trl.GRPOTrainer` bản `1.9.2`. **Đọc docs đúng phiên bản này** (cách truyền ảnh vào dataset VLM, chữ ký `reward_funcs`, cột dataset được chuyển thành `kwargs` cho hàm reward). Mình chưa kiểm chứng API này nên không đưa mã mẫu.
- Hàm reward bọc `iou_reward`: lấy `completion` (có thể là chuỗi hoặc danh sách message `[{"role":"assistant","content":...}]`, hãy xử lý cả hai) và `gt_boxes` quy về thang 0–1000 như `scripts.score._gt_boxes`.
- Suy luận trên test như E1, chấm `--name E5 --out report/evidence/e5.json`.
- **Reward hacking:** đếm tỉ lệ output `[]` và box khổng lồ trước/sau GRPO; so F1 với SFT kèm CI95.

---

## 10. Ảnh minh hoạ, bảng tổng hợp và số liệu cho 10 câu báo cáo

**Ảnh minh hoạ (RUBRIC B: ≥ 6 ảnh đúng/sai có chú thích, hiển thị cả box dự đoán và box đúng).** Dùng `labsrc/viz.py` đã dán ở ô 5.0:

```python
from labsrc.viz import make_figures
make_figures("work/e3.jsonl", name="e3")             # 3 ảnh đúng + 3 ảnh sai nhất → report/figures/
make_figures("work/e0.jsonl", name="e0", k=2)        # tuỳ chọn: lỗi của zero-shot (Câu 2)
```

Xem ảnh trong Colab rồi **chèn vào `REPORT.md`** bằng `![](report/figures/e3_wrong_<id>.png)` kèm 1–2 câu chú thích ("model nhầm crack thành pothole vì ..."). Ảnh chỉ có giá trị khi bạn chú thích được vì sao đúng/sai.

**Bảng tổng hợp** (copy vào `REPORT.md`, các cột khớp mẫu):

```python
import glob, json
pc = ("pothole", "crack", "manhole")
print("| Mã | Tham số | VRAM (GB) | Phút | F1 | CI95 | FP/ảnh | " + " | ".join(f"F1 {c}" for c in pc) + " |")
for p in sorted(glob.glob("report/evidence/e*.json")):
    d = json.load(open(p)); m = d.get("meta", {}); lo, hi = d["f1_ci95"]
    print(f"| {d['name']} | {m.get('trainable_params', 0):,} | {m.get('peak_vram_gb', '')} | {m.get('train_minutes', '')} | "
          f"{d['micro']['f1']:.3f} | {lo:.3f}–{hi:.3f} | {d['fp_per_image']:.2f} | "
          + " | ".join(f"{d['per_class'][c]['f1']:.3f}" for c in pc) + " |")
```

**Số liệu cho từng câu** (bạn tự viết lời giải thích bằng số của mình):

| Câu | Số liệu / thao tác lấy |
|---|---|
| 1 Kiến trúc | Output `labsrc.count_params` (ViT, merger, deepstack, LLM). E1: LoRA trên 252 module LLM; E2: + `visual.merger` & `deepstack_merger_list` (≈ 109M); E3: + 96 module ViT (tổng 348) |
| 2 E0 | `parse_valid_rate` trong `e0.json`; output thô ở mục 5; ảnh `make_figures(... e0 ...)` |
| 3 Thang 0–1000 | Chấm lại **cùng file dự đoán** của E1 như thể là pixel, **ghi ra `work/`, không ghi vào `evidence/`**: `!python scripts/score.py --pred work/e1.jsonl --coord pixel --name tmp --out work/e1_as_pixel.json`. F1 sụp vì toạ độ 0–1000 bị chia lại cho 640/360. Có thể thêm số liệu từ E4 `coord_scale` nếu chọn |
| 4 Mask loss | Kết quả `labsrc.check_encode` (số token ảnh 220, phần có loss). Giải thích: token ảnh và prompt là *điều kiện*, không phải thứ cần sinh; tính loss lên đó làm model học "chép" prompt, loss giảm giả và tốn gradient vào token không cần học. Tuỳ chọn: bỏ mask chạy 100 bước để có số thật |
| 5 4-bit / float32 | Chính `llm_int8_skip_modules` ở `load_model`; lý do float32 ở mục 6; `peak_vram_gb` của E2 so với E1 |
| 6 Split | `meta.json`: bảng số ảnh/box ở mục 3. Giải thích rò rỉ: frame liền kề gần như trùng nhau nên test "đã thấy" lúc train |
| 7 So E1/E2/E3 | Bảng tổng hợp: F1, CI95, `peak_vram_gb`, `train_minutes`. So chênh lệch F1 với độ rộng CI95 |
| 8 CI trên 311 ảnh | `hi - lo` của từng file. Kết luận nào còn đứng vững (E1 vs E0) và nào không (E2 vs E3) |
| 9 Lỗi theo lớp | F1 từng lớp trong bảng; kích thước box trung vị theo lớp (ô dưới); `make_figures` |
| 10 E4 | Hai file `e4_*.json`, CI95 của chúng, và biến bạn đổi |

Kích thước box trung vị theo lớp trên tập test (Câu 9):

```python
import statistics as st
from labsrc.data import load_split
areas = {}
for e in load_split("test"):
    for o in e["objects"]:
        areas.setdefault(o["label"], []).append(o["yolo"][2] * o["yolo"][3])
for c, v in areas.items():
    print(f"{c:8s} n={len(v):4d} diện tích trung vị = {st.median(v):.4%} khung hình")
```

Mục **"Nhìn lại"** của `REPORT.md`: viết một điều bạn thật sự chưa hiểu hoặc một lỗi mà công cụ chấm chưa bắt được (ví dụ: ảnh không có box nào, box chạm mép, lớp ít mẫu).

---

## 11. Nộp bài

1. Điền **toàn bộ** `REPORT.md` (không còn chuỗi `<điền`, kể cả ở "Họ tên", `dataset_version`, "Nhìn lại"; check tự động sẽ báo FAIL).
2. Kiểm tra bằng chứng:
   ```python
   !python scripts/check_submission.py
   ```
   Không còn **FAIL**. Chạy trong Colab (nơi có `data/splits/meta.json`) để mục "khớp meta.json" là PASS, không phải WARN.
3. Notebook: *File → Save a copy in GitHub*, hoặc tải `.ipynb` về và commit. Xoá mọi khoá nếu bạn lỡ in ra; kiểm tra bằng cách tìm `KAGGLE_KEY` trong file.
4. Kiểm tra repo sạch:
   ```python
   !git ls-files | grep -E "^(data|work)/|\.env$|\.(safetensors|bin|pt|ckpt)$" || echo "sạch"
   !git log --oneline | wc -l       # cần ≥ 5
   ```
5. `push("docs: báo cáo + ảnh minh hoạ")`, rồi mở repo trên GitHub ở cửa sổ ẩn danh để chắc rằng nó **công khai** và có đủ: `report/evidence/e0–e3.json`, ít nhất một `e4*.json`, `REPORT.md`, `report/figures/*.png`, notebook, `labsrc/`.
6. Không sửa `scripts/`, `tests/` (giảng viên ghi đè bằng bản gốc, sửa bị trừ 10 điểm). Không chép số liệu/văn bản giữa các bài (0 điểm cho cả hai bên).

**Đối chiếu RUBRIC trước khi nộp:** `e0–e3.json` đủ, `split=test`, `missing_predictions=0` (12) · `meta` E1–E3 là số dương (6) · `parse_valid_rate` E1 ≥ 0,95 (4) · F1(E1) − F1(E0) ≥ 0,10 (6) · có `e4*.json` (6) · `check_submission` không FAIL (6) · bảng + ≥ 6 ảnh + E4 sạch + nhật ký tài nguyên khớp `meta` (20) · 10 câu (30) · ≥ 5 commit & repo sạch (10).

---

## 12. Sự cố thường gặp

| Triệu chứng | Nguyên nhân thường gặp | Xử lý |
|---|---|---|
| `ImportError ... requires the Torchvision library` | thiếu `torchvision` cùng bản torch của Colab | `!pip -q install torchvision` (không ghim bản khác torch), chạy lại |
| `kaggle` 401/403 | secret sai, chưa bật Notebook access | kiểm tra `KAGGLE_USERNAME`/`KAGGLE_KEY`; hoặc tải zip bằng tay rồi `!python scripts/prepare_data.py --zip /content/drive/MyDrive/<file>.zip` |
| Mất session giữa chừng | Colab free ngắt | chạy lại ô 2.1–2.2; adapter/dự đoán còn trên Drive; `labsrc.infer` chạy tiếp được; huấn luyện phải chạy lại từ đầu (mã tham chiếu lưu mỗi 100 bước nhưng không có chế độ resume) |
| Hết hạn mức GPU | giới hạn của Colab free | chờ hoặc dùng tài khoản khác; lưu Drive trước |
| `CUDA out of memory` (thường ở phép tính loss) | batch 2 ảnh gặp ảnh nhiều box → chuỗi dài, logits fp32 rất lớn | `--batch-size 1 --grad-accum 8` và `PYTORCH_ALLOC_CONF=expandable_segments:True` (đã có sẵn trong các lệnh train); khởi động lại runtime nếu VRAM còn bị giữ; nếu batch 1 vẫn OOM, hãy báo để đổi cách tính loss chỉ trên vị trí đáp án |
| Loss `nan`/`inf` | tràn số fp16 | `train.py` dừng và báo; thử `--batch-size 1 --grad-accum 8`, kiểm tra mask |
| `Attempting to unscale FP16 gradients` | có tham số huấn luyện còn fp16 | kiểm tra `apply_setup` ép fp32 đúng tên (`visual.merger`, `visual.deepstack_merger_list`) |
| `ImportError: Found an incompatible version of torchao` (khi `get_peft_model`) | Colab có sẵn `torchao` cũ, `peft` yêu cầu ≥ 0.16 | `!pip -q uninstall -y torchao` (không cần restart vì mọi bước chạy trong tiến trình riêng) |
| `Target modules ... not found` hoặc số module ≠ 252/348 | tên module khác tên trong GUIDE | `!python -m labsrc.count_params`; in `[n for n,_ in model.named_modules()]` rồi chỉnh `LLM_RE`/`VIT_RE` |
| `KeyError` khi nạp projector | tên tham số lúc lưu ≠ lúc nạp | cả hai dùng `model.named_parameters()` của **PeftModel**; đừng lưu từ model chưa bọc peft |
| F1 E2/E3 = E1 chính xác | quên nạp lại projector | tìm dòng "đã nạp lại N tensor projector" trong log suy luận; kiểm tra `extra_trainable.pt` có trong thư mục adapter |
| `parse_valid_rate` thấp | model lặp vô hạn / bị cắt | tăng `--max-new-tokens`; kiểm tra token kết thúc `<\|im_end\|>` nằm trong nhãn (`check_encode`) |
| F1 ≈ 0 mà JSON hợp lệ | sai thang toạ độ | xem lại bước 2; chấm đúng `--coord` |
| Lỗi `apply_chat_template` về `content` | định dạng message khác phiên bản | in `processor.apply_chat_template(..., tokenize=False)` để xem chuỗi; thử `content` của assistant là chuỗi thay vì danh sách |
| Cảnh báo `generation flags` (temperature, top_p) | `generation_config` của Qwen có sẵn | bỏ qua, vì `do_sample=False` |
