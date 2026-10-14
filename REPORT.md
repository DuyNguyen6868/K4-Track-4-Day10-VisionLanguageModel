Họ tên: Nguyễn Duy
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

Test = 311 ảnh, 583 box (pothole 121, crack 299, manhole 163). E4 (bonus ablation) không thực hiện.

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | zero-shot | 0 | – | – | 0.079 | 0.055–0.108 | 0.85 | 0.159 | 0.024 | 0.142 |
| E1 | LoRA LLM | 33,030,144 | 6.45 | 80.0 | 0.254 | 0.216–0.289 | 1.20 | 0.138 | 0.180 | 0.528 |
| E2 | + projector | 142,135,296 | 6.75 | 91.1 | 0.338 | 0.294–0.384 | 0.95 | 0.223 | 0.271 | 0.569 |
| E3 | + LoRA ViT | 148,426,752 | 8.30 | 52.2 | 0.411 | 0.368–0.456 | 0.95 | 0.296 | 0.336 | 0.657 |

Thêm: `parse_valid_rate` E0 = 0.910, E1–E3 = 1.000. Mean IoU của các box khớp: E0 0.616, E1 0.681, E2 0.672, E3 0.697. Precision / recall: E0 0.117 / 0.060, E1 0.271 / 0.238, E2 0.377 / 0.307, E3 0.435 / 0.389.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Tôi chưa lưu lại output của `labsrc.count_params` nên không có số đo trực tiếp cho tổng tham số ViT và LLM (theo model card: ViT cỡ 0.4B, LLM cỡ 4B; đây là số xấp xỉ, không phải số tôi đo). Các số tôi suy ra chính xác từ `meta.trainable_params` trong bằng chứng:
- **E1** (LoRA r=16 trên LLM, 252 module): 33,030,144 tham số huấn luyện. ViT và projector đóng băng.
- **E2**: 142,135,296 − 33,030,144 = **109,105,152** tham số, chính là projector (`visual.merger` + `visual.deepstack_merger_list`) được mở đầy đủ, khớp với ≈109M của đề.
- **E3**: 148,426,752 − 142,135,296 = **6,291,456** tham số là LoRA trên ViT (96 module, 65,536 tham số/module), cộng thêm vào E2.
Vậy E1 cập nhật LLM (LoRA); E2 = E1 + projector đầy đủ; E3 = E2 + LoRA trên ViT. Toàn bộ vẫn trên nền 4-bit NF4, trừ ViT và `lm_head`.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` của E0 = **0.910** (≈9% câu trả lời không parse được thành JSON hợp lệ; E1–E3 đều đạt 1.000). Từ số liệu chấm, hai vấn đề rõ nhất của model gốc:
1. **Sai lớp / bịa box ở lớp crack**: crack có 193 dự đoán nhưng chỉ 6 đúng (precision 0.031, FP 187), còn 293/299 box crack thật bị bỏ sót (recall 0.020, F1 0.024).
2. **Recall rất thấp ở mọi lớp**: tổng 35 TP trên 583 box thật (recall 0.060). Khi model đã khớp được một box thì định vị không tệ (mean IoU 0.616), nên vấn đề chủ yếu là nhận diện đúng đối tượng/lớp chứ không phải sai thang.
⚠ **Thiếu**: tôi chưa chép lại một output thô cụ thể từ `work/e0.jsonl` (nằm trên Drive), nên chưa dẫn được ví dụ và chưa phân loại chính xác lỗi parse (bọc ```` ```json ````, chữ thừa, JSON hỏng). Cần lấy từ file `work/e0.jsonl`.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Bản chất: model sinh toạ độ nguyên trên thang 0–1000 so với kích thước ảnh. Ảnh là 640×360 nên nếu đọc số đó như pixel thì x bị coi như ≤1000 trên chiều rộng 640 và y ≤1000 trên chiều cao 360, tức box bị phóng sai tỷ lệ và lệch xa box thật, IoU rơi dưới 0.5 nên F1 sụp mạnh (parse_valid_rate vẫn không đổi vì JSON vẫn hợp lệ). Bằng chứng gián tiếp từ bài làm: chấm E0–E3 với `coord = norm1000` cho mean IoU 0.62–0.70 trên các box khớp, tức output của model đúng ở thang 0–1000.
⚠ **Thiếu**: tôi chưa chạy lệnh chấm đảo thang (`python scripts/score.py --pred work/e1.jsonl --coord pixel --name tmp --out work/e1_as_pixel.json`) nên chưa có con số F1 cụ thể; cần chạy trên Colab và điền số vào đây.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Prompt và token ảnh (220 token vision cho mỗi ảnh 640×360) là *điều kiện đầu vào*, không phải thứ model cần sinh ra; chỉ phần JSON đáp án (cộng `<|im_end|>`) mới là mục tiêu. `encode_example` che mọi thứ trước đáp án bằng nhãn −100. Nếu tính loss cả prompt và ảnh: (1) phần lớn gradient dồn vào việc dự đoán lại prompt cố định và token ảnh, nên loss trông thấp giả tạo mà không cải thiện đầu ra; (2) tín hiệu cho phần toạ độ bị pha loãng; (3) model bị kéo sang việc "chép" prompt. Kết quả thật khi chỉ mask đáp án: `parse_valid_rate` đi từ 0.910 (E0) lên 1.000 (E1–E3), tức model học đúng định dạng JSON. Tôi **không** chạy thử ablation bỏ mask nên không có số F1 so sánh trực tiếp cho trường hợp này.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Khi ViT bị lượng tử 4-bit, trọng số của merger là trọng số nén (`Params4bit`), optimizer không cập nhật đầy đủ được. Cách xử lý: đặt `llm_int8_skip_modules=["visual", "lm_head"]` để ViT (kèm projector) và `lm_head` giữ fp16, rồi trong `apply_setup` mở `visual.merger` + `visual.deepstack_merger_list` và **ép float32**. Float32 là bắt buộc vì `GradScaler.unscale_` từ chối gradient fp16 (lỗi "Attempting to unscale FP16 gradients"), nên mọi tham số huấn luyện phải ở fp32. Chi phí đo được: E2 huấn luyện thêm 109,105,152 tham số projector so với E1, VRAM đỉnh tăng từ 6.45 lên 6.75 GB (+0.30 GB).

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Dữ liệu là frame cắt từ video dashcam, nên hai frame liền kề gần như trùng nhau. Chia ngẫu nhiên sẽ đặt các frame gần giống nhau ở cả train và test (rò rỉ), model "đã thấy" ảnh test lúc train và F1 cao giả. `prepare_data.py` gom ảnh theo phiên quay (ngày + giờ trong tên file) và gán cả phiên vào một split. Số liệu từ `prepare_data.py` / README (`dataset_version` d10-21e4070301):

| Split | Ảnh | pothole | crack | manhole |
|---|---:|---:|---:|---:|
| train | 1,422 | 1,074 | 1,750 | 691 |
| val | 276 | 66 | 470 | 103 |
| test | 311 | 121 | 299 | 163 |

Số box test khớp với bằng chứng chấm (TP+FN: pothole 121, crack 299, manhole 163). Phân bố lớp không đều giữa các split (val chỉ có 66 pothole), nên phải đọc F1 từng lớp chứ không chỉ F1 gộp.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
| | F1 | CI95 | VRAM (GB) | Phút |
|---|---:|---|---:|---:|
| E1 | 0.254 | 0.216–0.289 | 6.45 | 80.0 |
| E2 | 0.338 | 0.294–0.384 | 6.75 | 91.1 |
| E3 | 0.411 | 0.368–0.456 | 8.30 | 52.2 |

- **E1 → E2 (mở projector)**: +0.085 F1, chỉ tốn +0.30 GB VRAM. CI95 gần như không chồng nhau (cận trên E1 0.289 so với cận dưới E2 0.294), nên mức tăng khá đáng tin. Rất đáng.
- **E2 → E3 (LoRA ViT)**: +0.072 F1, tốn +1.55 GB VRAM. CI95 **chồng nhau** (0.368–0.384), nên chưa kết luận chắc E3 hơn E2 chỉ từ khoảng tin cậy trên 311 ảnh.
- **E1 → E3**: +0.157 F1, CI95 tách hẳn nhau, nên E3 chắc chắn tốt hơn E1.
- Thời gian: E3 (52.2 phút) ngắn hơn E1 (80.0) và E2 (91.1) dù huấn luyện nhiều tham số hơn, điều này không hợp lý nếu cùng điều kiện; tôi cho rằng thời gian giữa các lần chạy không so sánh trực tiếp được (khác phiên/runtime Colab), nên không rút kết luận về chi phí thời gian.
Kết luận: E3 cho F1 cao nhất và VRAM 8.3 GB vẫn vừa T4; E2 là điểm cân bằng nếu cần tiết kiệm.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Độ rộng CI95 của F1: E0 0.053, E1 0.073, E2 0.090, E3 0.088 (nửa độ rộng khoảng ±0.03 đến ±0.045). Kết luận còn đứng vững: E1 hơn E0 (0.254 so với 0.079, chênh 0.175, CI không chạm nhau), và E3 hơn E1 (chênh 0.157, CI tách nhau). Kết luận không đứng vững: E2 so với E3 (chênh 0.072 nhưng CI chồng nhau 0.368–0.384), và E1 so với E2 chỉ ở mức biên (khoảng cách giữa hai CI chỉ 0.005). Lý do khoảng rộng: chỉ 311 ảnh/583 box, và các box tương quan theo phiên quay nên số mẫu độc lập thực tế còn ít hơn. Vì thế chênh lệch dưới khoảng 0.07–0.09 giữa hai setup không nên được coi là khác biệt thật.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Sau khi fine-tune, lớp kém nhất là **pothole** ở cả E1/E2/E3 (F1 0.138 / 0.223 / 0.296), tiếp theo là crack (0.180 / 0.271 / 0.336), còn manhole tốt nhất (0.528 / 0.569 / 0.657). Với E3, pothole chỉ bắt được 30/121 box (recall 0.248, FP 52). Crack có 104 TP nhưng tới 216 FP (precision 0.325). Số mẫu **không** giải thích được thứ hạng: train có 1,074 pothole nhiều hơn manhole 691 nhưng manhole vẫn tốt hơn nhiều, nên nguyên nhân nhiều khả năng nằm ở hình dạng và kích thước box (README nêu box nhỏ, diện tích trung vị chỉ ≈0.7% khung hình) và ngữ nghĩa mơ hồ (ổ gà nhỏ dễ nhầm với vết nứt/vùng tối; crack dài, mảnh nên IoU ≥ 0.5 khó đạt). Đây là giả thuyết, chưa kiểm chứng bằng kích thước box theo lớp.
⚠ **Thiếu**: chưa có ≥ 6 ảnh minh hoạ dự đoán đúng/sai có box dự đoán và box thật (cần chạy `make_figures` trên Colab, lưu vào `report/`), và chưa có số kích thước box trung vị theo lớp.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Tôi **không thực hiện E4** (ablation là phần bonus), nên không có kết quả hay kết luận nào ở đây, và `report/evidence/` không có file `e4*.json`.

## Nhìn lại

E0–E3 đã chạy và chấm trên toàn bộ 311 ảnh test cùng `dataset_version` d10-21e4070301. Fine-tuning giúp rõ rệt: F1 từ 0.079 (zero-shot) lên 0.254 (E1), 0.338 (E2) và 0.411 (E3), `parse_valid_rate` từ 0.910 lên 1.000. Mở projector là bước hiệu quả về chi phí nhất; LoRA trên ViT cho F1 cao nhất nhưng chưa tách khỏi E2 theo CI95. Giới hạn: test nhỏ nên CI rộng, thời gian huấn luyện giữa các setup không so sánh được, và còn thiếu E4, ví dụ output thô (Câu 2), phép đo đảo thang (Câu 3) và ảnh minh hoạ (Câu 9).
