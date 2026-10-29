# Khai báo Sử dụng AI (AI Usage Declaration) — Lab 2

Tài liệu này công khai việc sử dụng các công cụ trí tuệ nhân tạo (AI) trong quá trình thực hiện bài thực hành Lab 2, tuân thủ nguyên tắc minh bạch học thuật và liêm chính nghiên cứu.

---

## 1. Công cụ AI Đã sử dụng

- **Claude 3.7 Sonnet (Anthropic):** Hỗ trợ phân tích yêu cầu đề bài Practice 2 và gợi ý khung tài liệu thiết kế ban đầu.
- **Antigravity AI Assistant (Google DeepMind):** Hỗ trợ soạn thảo các đoạn mã khung (boilerplate code), cú pháp kịch bản kiểm thử tự động, tài liệu hóa kỹ thuật và hỗ trợ rà soát kiểm toán hệ thống.

---

## 2. Phạm vi Hỗ trợ của AI

Các thành phần có sự hỗ trợ của công cụ AI:
- Soạn thảo khung mã nguồn ban đầu cho các module trong `src/lab2/` (xây dựng mô hình, vòng lặp huấn luyện, checkpoint, đánh giá).
- Gợi ý cú pháp cho bộ kiểm thử tự động ban đầu trong `tests/`.
- Hỗ trợ xây dựng kịch bản kiểm tra hiện tượng cập nhật bộ đệm BatchNorm (`scripts/check_frozen_bn.py`).
- Hỗ trợ định dạng và tạo cấu trúc notebook Jupyter (`scripts/generate_main_notebook.py`, `scripts/generate_analysis_notebook.py`).
- Rà soát các cảnh báo kiểm toán liên quan đến namespace smoke test và quản lý checkpoint.

---

## 3. Quyết định Kỹ thuật và Điều chỉnh của Người thực hiện

Toàn bộ các quyết định kỹ thuật then chốt, kiểm tra mã nguồn và nghiệm thu hệ thống đều do người thực hiện trực tiếp đảm nhiệm:
1. **Lựa chọn kiến trúc và giới hạn phạm vi:** Lựa chọn ResNet18 và DenseNet121, loại bỏ VGG16 do ràng buộc tài nguyên CPU (138 triệu tham số).
2. **Quy tắc phân chia dữ liệu:** Thiết lập tỷ lệ 45.000 train / 5.000 validation với hạt giống cố định (`seed=42`), giữ nguyên 10.000 mẫu tập test độc lập để tránh rò rỉ dữ liệu.
3. **Sửa đổi thuật toán nhận diện tầng head:** Trực tiếp điều chỉnh logic lọc tầng phân loại trong `src/lab2/train.py` sang kiểm tra tiền tố chính xác (`name.startswith("fc.") or name.startswith("classifier.")`) để ngăn ngừa phân loại nhầm các tầng nội bộ.
4. **Xử lý tương thích môi trường Windows:** Bổ sung cấu hình mã hóa UTF-8 (`sys.stdout.reconfigure(encoding="utf-8")`) trên các kịch bản in ấn tiếng Việt để xử lý lỗi bảng mã `cp1252`.
5. **Đo đạc và xác nhận hiện tượng BatchNorm:** Thiết kế và thực hiện đo lường thực tế hành vi cập nhật `running_mean` và `running_var` của BatchNorm khi backbone ở trạng thái đóng băng nhưng chạy trong `model.train()`.
6. **Gia cố an toàn lưu trữ Checkpoint (`best_checkpoint_state`):** Phát hiện và khắc phục nguy cơ trôi tham số checkpoint tốt nhất bằng `copy.deepcopy()` trên `model.state_dict()`.
7. **Khắc phục lỗ hổng bảo mật mạng:** Loại bỏ triệt để việc bypass kiểm tra chứng chỉ SSL (`ssl._create_unverified_context`), thay thế bằng cơ chế tải và giải nén dữ liệu chuẩn hóa, kiểm tra checksum an toàn.
8. **Cách ly BatchNorm khi Ghi Đồ thị (`TensorBoardLogger.log_graph`):** Chuyển mô hình sang `eval()` khi trace graph mạng và khôi phục trạng thái ban đầu để tránh làm sai lệch bộ đệm BatchNorm.
9. **Bổ sung Kiểm thực Phòng thủ (Defensive Validations):** Thêm ràng buộc `weight_decay >= 0`, kiểm tra optimizer hợp lệ, và phát hiện giá trị Loss bất thường (`NaN`/`Inf`) trong vòng lặp huấn luyện.
10. **Tách biệt hoàn toàn Namespace Smoke Test:** Độc lập hóa đường dẫn lưu trữ của smoke test (`checkpoints/smoke/`, `outputs/smoke/`, `runs/smoke/`) nhằm loại bỏ triệt để rủi ro ghi đè checkpoint, metrics và TensorBoard của các thí nghiệm chính thức.
11. **Gia cố Checkpoint Fingerprint & Recovery:** Bổ sung kiểm tra tính tương thích siêu tham số (`learning_rate`, `batch_size`, `image_size`, `method`, `model_name`, `seed`, `val_split`), cơ chế tự động khôi phục metadata khi file JSON bị mất/hỏng và bảo toàn checkpoint cũ khi có xung đột cấu hình (bổ sung 2 test mới nâng tổng số test lên 64).
12. **Minh bạch hóa Môi trường Thực nghiệm & Câu chữ Báo cáo:** Đính chính rõ ràng môi trường thực nghiệm chính thức 5 epoch là GPU NVIDIA CUDA (`device: 'cuda'`, PyTorch `2.6.0+cu124`), loại bỏ các tuyên bố suy diễn nhân quả hoặc khẳng định hội tụ quá mức bằng các nhận định định lượng khách quan.

---

## 4. Phương pháp Kiểm chứng Độc lập

Mọi đoạn mã và kết quả đều được kiểm chứng độc lập thông qua các câu lệnh thực thi trực tiếp trên hệ thống:
1. **Kiểm thử tự động:**
   ```bash
   python -m pytest -q
   # Kết quả: 64 passed in 35s (100% pass)
   ```
2. **Kiểm chứng số lượng tham số mô hình:**
   ```bash
   python scripts/inspect_models.py
   # Kết quả: ResNet18 Transfer (5,130 params - 0.05%), ResNet18 Fine-tune (8,398,858 params - 75.11%)
   # DenseNet121 Transfer (10,250 params - 0.15%), DenseNet121 Fine-tune (2,170,378 params - 31.17%)
   ```
3. **Kiểm chứng bộ đệm BatchNorm:**
   ```bash
   python scripts/check_frozen_bn.py
   # Kết quả: 100% buffer cập nhật ở chế độ train; 0% buffer thay đổi ở chế độ eval
   ```
4. **Kiểm chứng Namespace Smoke Test:**
   ```bash
   python scripts/run_experiments.py --smoke-test --experiment resnet18_transfer
   # Kết quả: Ghi file vào checkpoints/smoke/, outputs/smoke/, runs/smoke/; không thay đổi tệp chính thức
   ```
5. **Kiểm tra cú pháp và thực thi Notebook:**
   ```bash
   python -m pytest tests/test_notebooks.py -v
   # Kết quả: 10/10 notebook tests passed
   ```
