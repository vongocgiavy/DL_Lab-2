# Báo cáo Trạng thái Triển khai Hệ thống (System Implementation Status) — Lab 2

Tài liệu này ghi nhận hiện trạng hoàn thiện các thành phần kỹ thuật của repository theo yêu cầu thực hành Practice 2 và các tiêu chuẩn kiểm thử phần mềm sau đợt rà soát ngày 29/10/2026.

**Môi trường thực nghiệm chính thức:**
- Thiết bị tính toán: GPU NVIDIA hỗ trợ CUDA (`device: "cuda"`).
- Thư viện cốt lõi: PyTorch `2.6.0+cu124`, torchvision `0.21.0+cu124`.
- Thời gian huấn luyện 5 epochs: Từ 05m 21s đến 10m 42s mỗi thí nghiệm.

**Môi trường xác minh và kiểm thử cục bộ:**
- Hệ điều hành: Windows 11 64-bit (Intel Core i5-12450HX, 24 GB RAM).
- Trình thông dịch: Python 3.13.5 (tags/v3.13.5:655084c, 64-bit).
- Thư viện cốt lõi: `torch` 2.15.0+cpu, `torchvision` 0.29.1+cpu, `tensorboard` 2.21.0, `pytest` 9.1.1, `jupyter` 1.1.1, `nbconvert` 7.16.6.

---

## 1. Bảng Trạng thái Các Hạng mục Kỹ thuật

| STT | Hạng mục | Trạng thái | Bằng chứng (Lệnh thực thi & Kết quả thật) |
|:---:|:---|:---:|:---|
| 1 | Môi trường & Dependencies | **Hoàn thành** | `python -m pip install -r requirements.txt` (đáp ứng 100% không thiếu package). Hệ thống hỗ trợ đa nền tảng GPU (CUDA) và CPU. |
| 2 | Bộ Kiểm thử Tự động (Unit & Integration Tests) | **Hoàn thành** | `python -m pytest -q` -> **64 passed** (100% pass; kiểm thử mô hình, checkpoint snapshot sâu, metadata recovery, hyperparameter compatibility, data, training loop, config validation, evaluate, reporting, hparam, frozen BN, và notebooks). |
| 3 | Khảo sát Kiến trúc & Đếm Tham số | **Hoàn thành** | `python scripts/inspect_models.py` -> In đủ 6 cấu hình: ResNet18 Transfer (5.130 trainable - 0.05%), ResNet18 Fine-tune (8.398.858 trainable - 75.11%), DenseNet121 Transfer (10.250 trainable - 0.15%), DenseNet121 Fine-tune (2.170.378 trainable - 31.17%), ResNet18 LR1e-4 (5.130 trainable), ResNet18 BS32 (5.130 trainable). |
| 4 | Smoke Test Pipeline Huấn luyện (Cách ly Namespace) | **Hoàn thành** | `python scripts/run_experiments.py --smoke-test --experiment resnet18_transfer` -> Chạy 1 epoch trên subset 120 train / 30 val / 30 test, toàn bộ checkpoint, metrics, event logs ghi riêng vào `checkpoints/smoke/`, `outputs/smoke/`, `runs/smoke/`, hoàn toàn không ảnh hưởng dữ liệu chính thức. |
| 5 | Kiểm tra Bảo mật & Bí mật | **Hoàn thành** | `git grep -n -i -E "api[_-]?key\|secret\|token\|password"` -> 0 kết quả. Toàn bộ repo sạch, không chứa credential. |
| 6 | Thí nghiệm Siêu tham số (Bài tập 2) | **Hoàn thành** | Đã cấu hình `group: hparam` gồm `resnet18_transfer_lr1e4` và `resnet18_transfer_bs32` trong `configs/config.yaml`, hỗ trợ cờ CLI `--group core\|hparam` trong `scripts/run_experiments.py`. |
| 7 | Đo đạc Thực nghiệm BatchNorm Đóng băng | **Hoàn thành** | Tạo `scripts/check_frozen_bn.py` và `tests/test_frozen_bn.py`. Kết quả: Ở `train` mode, 100% buffer (60/60 ResNet18, 363/363 DenseNet121) bị cập nhật; ở `eval` mode, 0% buffer thay đổi. |
| 8 | Notebook Thực hành Chính (`Lab2_Practice2_Main.ipynb`) | **Hoàn thành** | Thực thi thành công không lỗi bằng `python -m nbconvert --execute` (đầy đủ 7 bước và 3 bài tập, có sơ đồ minh họa). |
| 9 | Notebook Phân tích Kết quả (`Lab2_Results_Analysis.ipynb`) | **Hoàn thành** | Thực thi thành công không lỗi bằng `python -m nbconvert --execute` (phân tích đa biến, trích xuất top cặp nhầm lẫn, phân tích khoảng cách train/val). |
| 10 | `README.md` | **Hoàn thành** | Đã viết lại hoàn chỉnh: không chứa đường dẫn tuyệt đối, cập nhật 64/64 tests, mô tả môi trường GPU/CPU chính xác, bảng tham số chuẩn, hướng dẫn thực thi rõ ràng, kết quả thực nghiệm đầy đủ. |
| 11 | `docs/project_description.md` | **Hoàn thành** | Đã cập nhật mô tả môi trường phần cứng, bảng đối chiếu chi tiết 7 bước + 3 bài tập + 2 mục tiêu của Practice 2 và giải trình kỹ thuật việc không dùng VGG16. |
| 12 | `docs/project_design.md` | **Hoàn thành** | Đã bổ sung bảng Interface chi tiết từng hàm của các module, các quyết định thiết kế kỹ thuật và cơ chế phòng vệ checkpoint / namespace isolation. |
| 13 | `docs/experiment_plan.md` | **Hoàn thành** | Phân định rõ biến kiểm soát, biến độc lập, môi trường GPU CUDA thực tế và ghi nhận giới hạn hạt giống đơn lẻ (`seed=42`). |
| 14 | `docs/results.md` | **Hoàn thành** | Đồng bộ bảng 6 thí nghiệm, mô tả môi trường GPU CUDA chính xác, trình bày khách quan khoảng cách train/val và tỷ lệ tham số, loại bỏ các kết luận võ đoán. |
| 15 | `docs/AI_Usage.md` | **Hoàn thành** | Khai báo minh bạch công cụ, phạm vi hỗ trợ và phương pháp kiểm chứng thực tế. |
| 16 | Huấn luyện Đầy đủ 5 Epochs | **Hoàn thành** | Cả 6 thí nghiệm đã hoàn tất huấn luyện 5 epochs trên GPU CUDA. Metrics lưu tại `outputs/metrics/`, ma trận tại `outputs/figures/`. |
| 17 | Gia cố Hệ thống & An toàn Kỹ thuật (System Hardening) | **Hoàn thành** | Tách biệt namespace smoke test (`outputs/smoke/`, `checkpoints/smoke/`, `runs/smoke/`), deep snapshot lưu trạng thái `best_checkpoint_state`, kiểm tra tương thích siêu tham số (config fingerprint), khôi phục metadata tự động khi file meta bị hỏng/mất, loại bỏ bypass SSL, cách ly BatchNorm trong TensorBoard `log_graph()`, phòng vệ kiểm tra siêu tham số và phát hiện giá trị Loss bất thường (`NaN`/`Inf`). |

---

## 2. Tiêu chuẩn Hoàn thành Kỹ thuật (Definition of Done)

1. **Chất lượng mã nguồn (Code Quality):** Thiết kế theo module (`src/lab2/`), phân tách rõ ràng các tầng logic, đầy đủ type annotations và docstrings.
2. **Độ bao phủ kiểm thử (Test Coverage):** 64 tests kiểm thử tự động vượt qua 100% không có cảnh báo lỗi.
3. **Tính tái lập (Reproducibility):** Khóa seed cố định `seed=42`, tách train/val/test minh bạch, không rò rỉ dữ liệu.
4. **Tính tương tác (Interactivity):** 2 notebooks Jupyter hoàn chỉnh, thực thi sạch, không lỗi, trực quan hóa đầy đủ đồ thị và ma trận nhầm lẫn.
5. **Tính trung thực khoa học (Truthful Reporting):** Mọi tham số và kết quả đo đạc đều có nguồn gốc thực tế trích xuất trực tiếp từ summary.csv và metrics JSON, môi trường thực nghiệm được ghi nhận đúng thực tế GPU CUDA.
6. **An toàn dữ liệu thực nghiệm:** Cách ly hoàn toàn chế độ smoke test và kiểm thử nhanh, bảo vệ 100% checkpoint và kết quả chính thức khỏi bị ghi đè.
