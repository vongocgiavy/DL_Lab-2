# FINAL CLEANUP & TEXT SANITIZATION AUDIT — DL_Lab2_Nhom1

**Ngày kiểm kê:** 2026-10-27  
**Môi trường:** Windows 11, Python 3.14.7, PyTorch 2.14.1+cpu  
**Nhánh:** `main` (đồng bộ với `origin/main`)

---

## A. Kiểm kê Repository (File Inventory)

### A1. Toàn bộ file được Git theo dõi (git ls-files)

| File | Kích thước | Mục đích | Trạng thái |
|---|---|---|---|
| `.gitignore` | 474 B | Loại trừ `__pycache__`, `*.pth`, `data/`, `runs/`, `*.tar.gz` | Giữ |
| `README.md` | ~21 KB | Tài liệu hướng dẫn kỹ thuật | Giữ |
| `requirements.txt` | 204 B | Danh sách phụ thuộc Python | Giữ |
| `configs/config.yaml` | 2,552 B | Cấu hình 6 thí nghiệm (4 core + 2 hparam) | Giữ |
| `docs/AI_Usage.md` | ~3 KB | Khai báo sử dụng AI minh bạch theo chuẩn học thuật | Giữ |
| `docs/experiment_plan.md` | 7,507 B | Kế hoạch thực nghiệm và biến kiểm soát | Giữ |
| `docs/progress_status.md` | ~5 KB | Báo cáo trạng thái kỹ thuật hệ thống | Giữ |
| `docs/project_description.md` | 10,574 B | Mô tả dự án và đối chiếu yêu cầu Practice 2 | Giữ |
| `docs/project_design.md` | 13,051 B | Tài liệu thiết kế kỹ thuật và module interfaces | Giữ |
| `docs/results.md` | 12,933 B | Bảng kết quả thực nghiệm và phân tích lỗi | Giữ |
| `docs/figures/*.png` (3 files) | 155–282 KB | Sơ đồ quy trình, kiến trúc và biểu đồ tham số | Giữ |
| `notebooks/Lab2_Practice2_Main.ipynb` | ~700 KB | Notebook thực hành chính (31 cells) | Giữ |
| `notebooks/Lab2_Results_Analysis.ipynb` | ~23 KB | Notebook phân tích kết quả định lượng | Giữ |
| `checkpoints/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `data/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `outputs/figures/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `outputs/metrics/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `runs/.gitkeep` | 17 B | Giữ thư mục trong Git | Giữ |
| `scripts/check_frozen_bn.py` | 5,697 B | Đo lường hành vi BatchNorm đóng băng | Giữ |
| `scripts/evaluate_checkpoints.py` | 2,805 B | Chạy đánh giá checkpoint | Giữ |
| `scripts/generate_analysis_notebook.py` | ~17 KB | Generator cho notebook phân tích | Giữ |
| `scripts/generate_main_notebook.py` | ~26 KB | Generator cho notebook chính | Giữ |
| `scripts/generate_diagrams.py` | ~5 KB | Script tái lập 3 hình sơ đồ trong `docs/figures/` | Giữ |
| `scripts/inspect_models.py` | 2,478 B | Đếm và hiển thị tham số mô hình | Giữ |
| `scripts/make_results_table.py` | 8,860 B | Tổng hợp bảng kết quả từ JSON | Giữ |
| `scripts/run_experiments.py` | 6,317 B | Entry-point chạy thực nghiệm | Giữ |
| `src/lab2/*.py` (8 files) | 2–9 KB | Module mã nguồn nòng cốt | Giữ |
| `tests/*.py` (10 files) | 1–7 KB | Bộ 51 unit & integration tests | Giữ |

### A2. File Cục bộ (Không được Git theo dõi — đúng theo .gitignore)

| Đường dẫn | Kích thước | Trạng thái | Lý do |
|---|---|---|---|
| `checkpoints/*.pth` | 29–112 MB | **Không commit** | `.gitignore` loại trừ `*.pth`; checkpoint cục bộ |
| `data/cifar-10-python.tar.gz` | 170 MB | **Không commit** | Dataset gốc; `.gitignore` loại trừ |
| `data/cifar-10-batches-py/` | ~175 MB | **Không commit** | Dữ liệu huấn luyện đã giải nén |
| `runs/**/*.tfevents.*` | 0.4–1.6 MB | **Không commit** | TensorBoard logs; `.gitignore` loại trừ |
| `__pycache__/` | Biến thiên | **Không commit** | File bytecode sinh tự động |
| `.pytest_cache/` | Nhỏ | **Không commit** | Cache pytest |

---

## B. Danh mục Hình minh họa Đã bổ sung trong Notebook Chính

| File hình | Vị trí trong notebook | Ý nghĩa minh họa |
|---|---|---|
| `docs/figures/pipeline_workflow.png` | Sau header/TOC (Cell 1–2) | Sơ đồ luồng xử lý toàn pipeline (Dữ liệu CIFAR-10 $\to$ Transforms $\to$ Model Backbone $\to$ Training Loop $\to$ Checkpointing & Logs). |
| `docs/figures/transfer_vs_finetune.png` | Sau Step 4 Markdown (Cell 11–12) | So sánh trực quan cơ chế Feature Extraction (Frozen Backbone) vs. Fine-Tuning (All Layers Trainable). |
| `docs/figures/model_parameters_comparison.png` | Sau Step 4 Code (Cell 13–14) | Biểu đồ tham số Trainable vs Frozen trích xuất trực tiếp từ source code `count_parameters()`. |

---

## C. Rà soát Câu chữ, Loại bỏ Dấu vết AI và Thông tin Hành chính

### C1. Loại bỏ thông tin hành chính, tên trường và thông tin sinh viên
- **Tiêu đề trang bìa / Header cells:** Đã xóa bỏ phần thông tin hành chính môn học, tên giảng viên hướng dẫn, tên đơn vị trường học, và khối thông tin nhóm sinh viên (danh sách thành viên, MSSV) khỏi cả hai notebook (`Lab2_Practice2_Main.ipynb`, `Lab2_Results_Analysis.ipynb`) và generator scripts (`generate_main_notebook.py`, `generate_analysis_notebook.py`).
- **Placeholders:** Đã xóa toàn bộ các placeholder sinh viên cần điền, thay thế bằng các hướng dẫn kỹ thuật khách quan và nhãn `*Chờ kết quả thực nghiệm*`.
- **Ghi công tài liệu tham khảo:** Giữ nguyên trích dẫn học thuật của bài giảng trong mục Tài liệu tham khảo (`## 12. Tài liệu Tham khảo`).

### C2. Loại bỏ dấu vết AI và câu chữ quảng bá / sáo rỗng
- **Nhật ký prompting nội bộ:** Đã xóa Section 8 (Phase Dialogue Log) trong `docs/AI_Usage.md` — không đưa checklist các phase triển khai và prompting log nội bộ vào tài liệu cuối.
- **Văn phong báo cáo:**
  - `README.md`: Bỏ các từ ngữ quảng bá ("chuẩn công nghiệp", "toàn diện", "mạnh mẽ"), giữ văn phong kỹ thuật trực tiếp, cô đọng.
  - `docs/progress_status.md`: Viết lại thành Báo cáo Trạng thái Triển khai Hệ thống, loại bỏ các tham chiếu đến checklist hay commit log theo Phase của AI.
  - `docs/results.md`: Loại bỏ lối diễn đạt sáo rỗng ("nguyên tắc Source-of-Truth"), trình bày khách quan phương pháp ghi nhận số liệu.
  - `notebooks/Lab2_Practice2_Main.ipynb`: Bỏ câu nhắc nhở chụp màn hình ở cell 28, thay bằng hướng dẫn kỹ thuật trung lập.

### C3. Vệ sinh đường dẫn máy cá nhân (Paths & Metadata)
- Toàn bộ đường dẫn tuyệt đối dạng `C:\Project\lab 2` trong cell outputs của notebook đã được chuẩn hóa thành `<PROJECT_ROOT>`.
- Toàn bộ FutureWarning chứa đường dẫn máy cá nhân `C:\Users\<user>\...` đã được làm sạch khỏi outputs.
- Notebook metadata chỉ chứa `language_info` chuẩn, không chứa trường nhận diện cá nhân.

---

## D. Kết quả Kiểm thử Toàn vẹn (Verification Results)

| Lệnh kiểm tra | Kết quả | Trạng thái |
|---|---|:---:|
| `python -m pytest -q` | 64 passed (100% pass) | ✅ 100% PASS |
| `py -m pytest tests/test_notebooks.py -v` | 10/10 tests passed | ✅ 100% PASS |
| Kiểm tra cú pháp schema notebook (nbformat v4) | Hợp lệ, không có lỗi cấu trúc | ✅ ĐẠT |
| Kiểm tra thực thi code cells trong notebook | 100% code cells có execution_count hợp lệ | ✅ ĐẠT |
| `git grep -n -i -E "api[_-]?key\|secret\|token\|password"` | 0 kết quả trong toàn bộ source | ✅ SẠCH |

---

## E. Xác nhận Phạm vi Rà soát

- Báo cáo này xác nhận việc rà soát đã được thực hiện trên toàn bộ các file được theo dõi trong working tree và các commit mới.
- Về lịch sử commit trước đây: Commit cũ `4a74105` được tạo trực tiếp trên GitHub web UI chứa email cá nhân cũ. Do tuân thủ nguyên tắc không rewrite lịch sử Git công khai (tránh ảnh hưởng đến các thành viên khác clone repo), commit cũ này được giữ nguyên; tất cả commit từ local đều sử dụng email ẩn của GitHub (`@users.noreply.github.com`).
