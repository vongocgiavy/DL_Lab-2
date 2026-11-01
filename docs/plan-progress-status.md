# KẾ HOẠCH THỰC NGHIỆM & BÁO CÁO TIẾN ĐỘ HỆ THỐNG (PLAN-PROGRESS-STATUS.MD)

**Chủ đề:** Kế hoạch thực nghiệm, phân tích rủi ro, theo dõi tiến độ 12 mốc kỹ thuật, xác thực Definition of Done (DoD) và khai báo sử dụng AI  
**Tiêu chuẩn thực hiện:** Source-First, Design-First, Test-Driven, Reproducible

---

## 1. KẾ HOẠCH THỰC NGHIỆM & KIỂM SOÁT BIẾN (EXPERIMENT PLAN)

### 1.1. Ma trận kiểm soát biến (Controlled Variables Matrix)

| Thí nghiệm ID | Biến độc lập (Thay đổi) | Biến phụ thuộc (Quan sát) | Các biến kiểm soát (Giữ cố định) |
|:---|:---|:---|:---|
| `resnet18_transfer` | Chiến lược: Transfer Learning (Feature Extraction) | Loss, Accuracy, Thời gian, Số tham số | ResNet-18, LR $10^{-3}$, Batch $16$, 5 epochs, Seed $42$, ImageNet Normalize |
| `resnet18_finetune` | Chiến lược: Fine-Tuning (Mở khóa `layer4`) | Loss, Accuracy, Thời gian, Số tham số | ResNet-18, Head LR $10^{-3}$, Backbone LR $10^{-4}$, Batch $16$, 5 epochs, Seed $42$ |
| `densenet121_transfer` | Kiến trúc: DenseNet-121 (Khóa `features`) | Loss, Accuracy, Thời gian, Số tham số | DenseNet-121, LR $10^{-3}$, Batch $16$, 5 epochs, Seed $42$ |
| `densenet121_finetune` | Kiến trúc: DenseNet-121 (Mở `denseblock4`) | Loss, Accuracy, Thời gian, Số tham số | DenseNet-121, Head LR $10^{-3}$, Backbone LR $10^{-4}$, Batch $16$, 5 epochs, Seed $42$ |
| `resnet18_transfer_lr1e4` | Siêu tham số: Giảm Learning Rate xuống $10^{-4}$ | Tốc độ hội tụ, Test Accuracy | ResNet-18, Transfer, Batch $16$, 5 epochs, Seed $42$ |
| `resnet18_transfer_bs32` | Siêu tham số: Tăng Batch Size lên $32$ | Độ ổn định gradient, Thời gian/epoch | ResNet-18, Transfer, LR $10^{-3}$, 5 epochs, Seed $42$ |

---

## 2. PHÂN TÍCH RỦI RO & PHƯƠNG ÁN GIẢM THIỂU (RISK MANAGEMENT)

| Rủi ro kỹ thuật | Mức độ ảnh hưởng | Khả năng xảy ra | Giải pháp ngăn chặn trong hệ thống |
|:---|:---:|:---:|:---|
| **Quá khớp (Overfitting)** | Cao | Trung bình | Áp dụng Data Augmentation (`RandomCrop`, `RandomHorizontalFlip`), Weight Decay $10^{-4}$, giới hạn 5 epochs. |
| **Quên thảm họa (Catastrophic Forgetting)** | Nghiêm trọng | Cao | Triển khai Differential Learning Rate: $\eta_{backbone} = 10^{-4} = \frac{1}{10} \eta_{head}$. |
| **Rò rỉ dữ liệu (Data Leakage)** | Nghiêm trọng | Thấp | Tập Test 10.000 ảnh độc lập 100%, chỉ được nạp một lần duy nhất ở cuối trên Best Checkpoint. |
| **Hỏng file Checkpoint khi crash** | Trung bình | Thấp | Áp dụng cơ chế ghi nguyên tử `atomic_save()` qua file `.tmp` và `os.replace()`. |
| **Ghi đè kết quả khi chạy Smoke Test** | Trung bình | Thấp | Tách namespace độc lập (`--smoke-test`), tách cấu hình và file lưu trữ. |

---

## 3. THEO DÕI TIẾN ĐỘ 12 MỐC KỸ THUẬT (MILESTONE TRACKER)

| Mốc (Milestone) | Nội dung công việc | File kiểm chứng | Trạng thái |
|:---:|:---|:---|:---:|
| **M1** | Thiết lập cấu trúc thư mục chuẩn và môi trường ảo | `src/lab2/`, `configs/`, `tests/` | **HOÀN THÀNH 100%** |
| **M2** | Pipeline nạp dữ liệu CIFAR-10, tiền xử lý và chia train/val/test | [`src/lab2/data.py`](file:///d:/Hoc_Sau/Lab 2/src/lab2/data.py) | **HOÀN THÀNH 100%** |
| **M3** | Module mô hình ResNet-18 & DenseNet-121 với cơ chế freeze/unfreeze | [`src/lab2/models.py`](file:///d:/Hoc_Sau/Lab 2/src/lab2/models.py) | **HOÀN THÀNH 100%** |
| **M4** | Quản lý tham số và kiểm tra cấu trúc qua CLI | [`scripts/inspect_models.py`](file:///d:/Hoc_Sau/Lab 2/scripts/inspect_models.py) | **HOÀN THÀNH 100%** |
| **M5** | Vòng lặp huấn luyện, validation và cơ chế Differential LR | [`src/lab2/train.py`](file:///d:/Hoc_Sau/Lab 2/src/lab2/train.py) | **HOÀN THÀNH 100%** |
| **M6** | Quản lý checkpoint an toàn nguyên tử (Atomic save) | [`src/lab2/checkpoint.py`](file:///d:/Hoc_Sau/Lab 2/src/lab2/checkpoint.py) | **HOÀN THÀNH 100%** |
| **M7** | Tích hợp hệ thống giám sát đa luồng qua TensorBoard | [`src/lab2/tensorboard_logger.py`](file:///d:/Hoc_Sau/Lab 2/src/lab2/tensorboard_logger.py) | **HOÀN THÀNH 100%** |
| **M8** | Module đánh giá độc lập trên tập Test và ma trận nhầm lẫn | [`src/lab2/evaluate.py`](file:///d:/Hoc_Sau/Lab 2/src/lab2/evaluate.py) | **HOÀN THÀNH 100%** |
| **M9** | Bộ kiểm thử tự động 53 unit & integration tests | [`tests/`](file:///d:/Hoc_Sau/Lab 2/tests/) | **HOÀN THÀNH 100%** |
| **M10** | Trực quan hóa dữ liệu thô, PCA và kiến trúc giám sát | [`scripts/visualize_raw_pca.py`](file:///d:/Hoc_Sau/Lab 2/scripts/visualize_raw_pca.py) | **HOÀN THÀNH 100%** |
| **M11** | Chạy đầy đủ 6 thí nghiệm và tổng hợp metrics JSON / CSV | [`outputs/metrics/summary.csv`](file:///d:/Hoc_Sau/Lab 2/outputs/metrics/summary.csv) | **HOÀN THÀNH 100%** |
| **M12** | Bộ 10 slide thuyết trình kết hợp kiến trúc và code thực tế | [`outputs/figures/slides/`](file:///d:/Hoc_Sau/Lab 2/outputs/figures/slides/) | **HOÀN THÀNH 100%** |

---

## 4. XÁC THỰC DEFINITION OF DONE (DoD CHECKLIST)

Theo quy chuẩn kỹ thuật trong [`AGENTS.md`](file:///d:/Hoc_Sau/Lab 2/AGENTS.md), toàn bộ các tiêu chuẩn hoàn thành dự án đã được rà soát và đáp ứng trọn vẹn:

- [x] **Đủ yêu cầu:** Đạt 100% các tiêu chí thực nghiệm Practice 2 (Pretrained models, Transfer Learning, Fine-Tuning, thay classifier head, khảo sát LR và Batch size).
- [x] **Code chạy:** Toàn bộ script và module thực thi không có bất kỳ lỗi cú pháp hay runtime exception nào.
- [x] **Test passed:** Toàn bộ **53/53 bài unit & integration tests** đều vượt qua (`pytest -v` passed).
- [x] **No secrets:** Đã quét rà soát toàn bộ dự án, xác nhận không có API key, token, mật khẩu hay biến `.env` nào bị lộ.
- [x] **Code review:** Mã nguồn được tách module rõ ràng, có đầy đủ Type Hints, Docstrings và tuân thủ chuẩn PEP 8.
- [x] **GitHub:** Trạng thái nhánh `main` đồng bộ hoàn toàn với repository GitHub [`vongocgiavy/DL_Lab-2`](https://github.com/vongocgiavy/DL_Lab-2).
- [x] **Docs updated:** Toàn bộ thư mục tài liệu `/docs` được chuẩn hóa thành 7 tài liệu kỹ thuật chuyên sâu.
- [x] **Committed:** Mọi thay đổi đều được commit theo chuẩn quốc tế Conventional Commits (`feat:`, `fix:`, `docs:`).

---

## 5. BÁO CÁO SỬ DỤNG AI MINH BẠCH (AI USAGE LOG)

Tuân thủ nghiêm ngặt chuẩn mực liêm chính học thuật:
- **Công cụ AI sử dụng:** Antigravity (Google DeepMind Advanced Agentic Coding Assistant).
- **Mục đích chính:** Hỗ trợ thiết kế cấu trúc module, tối ưu hóa pipeline dữ liệu, tự động hóa kịch bản sinh biểu đồ và bộ kiểm thử tự động.
- **Đóng góp của sinh viên/người phát triển:** Trực tiếp định hướng cấu hình thực nghiệm, thẩm định tính chính xác của lý thuyết, rà soát kết quả đo đạc thực tế, phát hiện và đính chính các sai lệch số liệu.
- **Phương pháp kiểm chứng (Validation):** Không dựa vào kết quả AI sinh ra nếu chưa chạy thử nghiệm thực tế; mọi số liệu đều được đối chiếu trực tiếp từ các file kết quả nhị phân và JSON thực tế.

