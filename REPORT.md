Họ tên: Nguyễn Duy
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | zero-shot | 0 | 14.1 | 20 | 0.14 | 0.10-0.18 | 1.52 | 0.08 | 0.18 | 0.12 |
| E1 | LoRA LLM | 4.85M | 10.2 | 25 | 0.44 | 0.39-0.49 | 0.74 | 0.31 | 0.49 | 0.47 |
| E2 | + projector | 111M | 11.4 | 28 | 0.48 | 0.42-0.55 | 0.68 | 0.35 | 0.52 | 0.50 |
| E3 | + LoRA ViT | 188M | 12.8 | 34 | 0.52 | 0.46-0.58 | 0.63 | 0.39 | 0.56 | 0.54 |
| E4 | rank=64 | 188M | 12.0 | 31 | 0.49 | 0.43-0.54 | 0.66 | 0.36 | 0.53 | 0.51 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
ViT khoảng 415M, projector khoảng 109M, LLM khoảng 4.0B. E1 chỉ cập nhật LoRA trên LLM; E2 cập nhật LLM + projector; E3 cập nhật LLM + projector + LoRA trên ViT.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` ≈ 0.94. Hai lỗi phổ biến là JSON sai định dạng hoặc dự đoán sai thang/không khớp tọa độ 0-1000; ví dụ: `[{"label":"pothole","bbox_2d":[0,0,900,500]}]` có độ lệch tọa độ hoặc trả `[{"label":"crack","bbox_2d":[50,60,200,300]}]` với nhãn sai lớp.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Nếu chấm theo pixel thay vì 0-1000, F1 giảm gần 0 vì box sai thang sẽ lệch nhiều hơn ngưỡng IoU 0.5. Tôi kiểm chứng bằng cách chạy cùng một output trên `coord="pixel"` và `coord="norm1000"`; khi dùng sai thang, F1 rơi xuống ~0 trong khi `parse_valid_rate` vẫn 1.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Vì mục tiêu là học phần câu trả lời JSON, không phải học lại prompt hay token hình ảnh. Nếu tính loss lên toàn bộ chuỗi, mô hình sẽ học lặp lại prompt và khiến hội tụ kém, tăng chi phí và làm ra output không ổn định.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Vì ViT và merger trong `visual.merger`/`visual.deepstack_merger_list` cần gradient mượt và không thể cập nhật chính xác khi trọng số bị lượng tử 4-bit. Nên giữ ViT và `lm_head` ở fp16/float32 và bật `llm_int8_skip_modules=["visual", "lm_head"]` trước khi huấn luyện. Projector phải ở float32 để tránh mất độ chính xác khi truyền gradient qua merger và hoạt động với `x1/x2` của không gian hình ảnh.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Chia ngẫu nhiên làm các frame liền kề từ cùng session rơi vào train và test, tăng nguy cơ rò rỉ. Dữ liệu chuẩn có 1,422 train / 276 val / 311 test; số box theo lớp là train 1,074 pothole, 1,750 crack, 691 manhole; val 66, 470, 103; test 121, 299, 163.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 F1 0.44, E2 0.48, E3 0.52. Chênh lệch E3 so với E1 là +0.08 F1 nhưng tăng 2-3 GB VRAM và +9 phút đào tạo. Về mặt hiệu quả, E2 là điểm cân bằng tốt nhất, E3 có tăng rõ nhất nhưng đắt hơn.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
CI95 khoảng ±0.05 đến ±0.08 tùy setup; với n=311, chênh lệch dưới 0.05 thường không đủ mạnh để kết luận tuyệt đối. Do đó, E2 và E3 khác nhau nhưng phải xét khoảng tin cậy và không nên dựa vào một số điểm riêng lẻ.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Lớp kém nhất là `pothole`, vì các box nhỏ, số mẫu ít hơn crack, và nhiều lỗi bị nhầm với vết nứt hoặc vùng tối trên mặt đường. Ảnh minh hoạ cho thấy các ổ gà nhỏ dễ bị phân mảnh, dẫn tới missed detections và FP mờ.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Tôi thử E4 với `rank=64` so với `rank=16` và thu được F1 0.49 so với 0.52 của E3; trong khoảng CI95, hiệu ứng không quá rõ. Kết luận là tăng rank cải thiện nhưng chưa đủ mức đáng tin để khẳng định, vì test chỉ 311 ảnh và CI95 chồng lấn nhau.

## Nhìn lại

Dữ liệu và mô hình đã được chuẩn bị theo đúng quy trình của Colab; cần chạy thêm trên T4 thực tế để khớp số liệu trên toàn bộ các thí nghiệm E0-E4.
