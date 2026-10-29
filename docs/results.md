# Experimental Results & Analysis — Lab 2

## 1. Giới thiệu và Phương pháp Ghi nhận Kết quả
Báo cáo ghi nhận các kết quả đo đạc thực nghiệm trên môi trường tính toán CPU (Intel Core i5-12450HX, 24 GB RAM, Windows 11).
Các trường dữ liệu chưa hoàn thành đủ 5 epochs được ghi nhận bằng nhãn `*Chờ kết quả thực nghiệm*`.

---

## 2. Kết quả Khảo sát Kiến trúc và Phân tích Tham số (Architecture Inspection)

Được trích xuất trực tiếp thông qua lệnh `py scripts/inspect_models.py`:

| Experiment ID | Nhóm | Kiến trúc | Phương pháp | Tổng tham số (Total) | Tham số Trainable | Tham số Bị đóng băng (Frozen) | Tỷ lệ Trainable (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `resnet18_transfer` | `core` | ResNet18 | Transfer Learning | 11,181,642 | 5,130 | 11,176,512 | 0.05% |
| `resnet18_finetune` | `core` | ResNet18 | Fine-Tuning | 11,181,642 | 8,398,858 | 2,782,784 | 75.11% |
| `densenet121_transfer` | `core` | DenseNet121 | Transfer Learning | 6,964,106 | 10,250 | 6,953,856 | 0.15% |
| `densenet121_finetune` | `core` | DenseNet121 | Fine-Tuning | 6,964,106 | 2,170,378 | 4,793,728 | 31.17% |
| `resnet18_transfer_lr1e4` | `hparam` | ResNet18 | Transfer Learning | 11,181,642 | 5,130 | 11,176,512 | 0.05% |
| `resnet18_transfer_bs32` | `hparam` | ResNet18 | Transfer Learning | 11,181,642 | 5,130 | 11,176,512 | 0.05% |

### Nhận xét phân tích kiến trúc:
1. **ResNet18 Transfer Learning:** Chỉ duy nhất tầng kết nối đầy đủ cuối cùng (`fc.weight` và `fc.bias`) với $512 \times 10 + 10 = 5.130$ tham số được cập nhật. Toàn bộ các khối `conv1`, `bn1`, `layer1`, `layer2`, `layer3`, `layer4` đều đóng băng hoàn toàn.
2. **ResNet18 Fine-Tuning:** Khi mở khóa khối tích chập sâu nhất `layer4` (gồm 2 BasicBlocks) cùng `fc`, số lượng tham số trainable tăng vọt lên 8.398.858 (chiếm 75.11% tổng mô hình), cho phép mô hình thích nghi các bộ lọc biểu diễn đặc trưng cấp cao theo dữ liệu CIFAR-10.
3. **DenseNet121 Transfer Learning:** Tầng phân loại `classifier` mới nhận đầu vào $1.024$ kênh, tạo ra $1.024 \times 10 + 10 = 10.250$ tham số cần huấn luyện (chiếm 0.15%).
4. **DenseNet121 Fine-Tuning:** Khác với ResNet, cấu trúc DenseNet có cơ chế nối ghép đặc trưng (feature concatenation). Việc mở khóa `denseblock4` (16 dense layers), `norm5` và `classifier` chỉ kích hoạt 2.170.378 tham số (chiếm 31.17%), giúp fine-tuning rất nhẹ nhàng và tiết kiệm tài nguyên tính toán hơn so với ResNet18.

---

## 3. Bảng Kết quả Thực nghiệm (Official Experiments Results)

Bảng tổng hợp chính thức trích xuất tự động từ `outputs/metrics/summary.csv`:

| Experiment ID | Epochs | Batch | LR | Trainable params | Train Loss/Acc (epoch cuối) | Best Val Acc (epoch) | Test Loss | Test Acc | Thời gian TB mỗi epoch | Tổng thời gian |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `resnet18_transfer` | 5 | 16 | 0.001 | 5,130 | 0.7718 / 73.85% | 77.94% (ep 2) | 0.6644 | **77.98%** | 01m 06s | 05m 35s |
| `resnet18_finetune` | 5 | 16 | 0.001 | 8,398,858 | 0.2488 / 91.48% | 91.68% (ep 4) | 0.2989 | **90.24%** | 01m 26s | 07m 15s |
| `densenet121_transfer` | 5 | 16 | 0.001 | 10,250 | 0.6860 / 76.89% | 81.84% (ep 5) | 0.5560 | **81.34%** | 01m 48s | 09m 03s |
| `densenet121_finetune` | 5 | 16 | 0.001 | 2,170,378 | 0.2912 / 89.94% | 90.78% (ep 5) | 0.2982 | **90.65%** | 02m 08s | 10m 42s |
| `resnet18_transfer_lr1e4` | 5 | 16 | 0.0001 | 5,130 | 0.7605 / 73.96% | 77.56% (ep 5) | 0.6643 | **78.32%** | 01m 12s | 06m 03s |
| `resnet18_transfer_bs32` | 5 | 32 | 0.001 | 5,130 | 0.7029 / 76.12% | 78.92% (ep 4) | 0.6276 | **79.04%** | 01m 04s | 05m 21s |

---

## 4. Phân tích Chi tiết Kết quả Thực nghiệm

### 4.1. So sánh 4 Thí nghiệm Nòng cốt (Core Experiments)
- **Transfer Learning vs Fine-Tuning trên ResNet18:**
  - *Số liệu thực tế:* Fine-Tuning giúp Test Accuracy tăng vượt bậc từ **77.98% lên 90.24%** (+12.26%), Test Loss giảm từ 0.6644 xuống 0.2989.
  - *Đánh đổi (Trade-off):* Thời gian mỗi epoch tăng từ 01m 06s lên 01m 26s do lan truyền ngược qua 8.4 triệu tham số của `layer4`.
- **ResNet18 vs DenseNet121 trong Transfer Learning:**
  - *Số liệu thực tế:* DenseNet121 đạt **81.34%**, cao hơn **+3.36%** so với ResNet18 (77.98%), Test Loss thấp hơn đáng kể (0.5560 vs 0.6644).
  - *Giải thích:* Cơ chế Dense Connectivity nối ghép toàn bộ bản đồ đặc trưng trước đó tạo ra biểu diễn đặc trưng 1024 chiều giàu thông tin hơn vector 512 chiều của ResNet18.
- **Hiệu quả Tham số trong Fine-Tuning:**
  - Cả ResNet18 và DenseNet121 đều chạm ngưỡng hội tụ ~90.2%–90.6% Test Acc.
  - Tuy nhiên, DenseNet121 chỉ cần mở khóa **2,170,378 tham số** (31.17%), trong khi ResNet18 phải mở khóa **8,398,858 tham số** (75.11%). DenseNet121 tiết kiệm tài nguyên tham số hơn đáng kể.
- **Hiện tượng Overfitting:**
  - Ở cả hai mô hình Fine-Tuning, Train Accuracy đạt xấp xỉ 90–91.5% và Test Accuracy đạt ~90.2%–90.6%, khoảng cách giữa train và validation/test loss nhỏ (~0.05), chứng minh việc dùng Differential LR (`lr_backbone = 1e-4`, `lr_head = 1e-3`) đã kiểm soát overfitting cực kỳ hiệu quả mà không làm suy thoái trọng số tiền huấn luyện ImageNet.

### 4.2. Khảo sát Siêu tham số (Hyperparameter Experiments)
- **Tốc độ học (Learning Rate: `1e-3` vs `1e-4`):**
  - Cấu hình `lr=1e-4` (`resnet18_transfer_lr1e4`) đạt Test Acc **78.32%**, nhỉnh hơn 77.98% của cấu hình chuẩn `lr=1e-3`. Tốc độ học nhỏ hơn giúp quá trình tối ưu classification head mượt mà, tránh dao động lớn quanh cực tiểu ở các epoch cuối.
- **Kích thước Batch (Batch Size: `16` vs `32`):**
  - Cấu hình `batch_size=32` (`resnet18_transfer_bs32`) rút ngắn thời gian mỗi epoch từ 01m 06s xuống **01m 04s** (tăng thông lượng vector hóa trên CPU), đồng thời đạt Test Acc **79.04%** (+1.06% so với batch 16) nhờ gradient mini-batch ít ồn hơn.

---

## 5. Phân tích Độ chính xác từng lớp & Ma trận Nhầm lẫn (Per-Class & Error Analysis)

### 5.1. Bảng Độ chính xác từng lớp (%)
Dữ liệu trích xuất chính thức từ các tệp `outputs/metrics/*_test_metrics.json`:

| Lớp (Class) | `resnet18_transfer` | `resnet18_finetune` | `densenet121_transfer` | `densenet121_finetune` |
|---|:---:|:---:|:---:|:---:|
| 0. Airplane | 82.0% | 95.0% | 87.6% | 94.0% |
| 1. Automobile | 85.6% | 80.4% | 85.7% | 93.0% |
| 2. Bird | 61.8% | 90.4% | 74.2% | 87.0% |
| 3. Cat | 70.1% | 77.8% | 71.4% | 81.6% |
| 4. Deer | 70.8% | 90.4% | 76.1% | 86.6% |
| 5. Dog | 61.9% | 91.6% | 70.9% | 84.3% |
| 6. Frog | 91.9% | 91.4% | 86.3% | 94.4% |
| 7. Horse | 80.9% | 92.7% | 83.4% | 96.2% |
| 8. Ship | 87.3% | 94.3% | 85.9% | 93.9% |
| 9. Truck | 87.5% | 98.4% | 91.9% | 95.5% |

### 5.2. Các cặp lớp dễ nhầm lẫn nhất (Top Confused Pairs)
Phân tích trích xuất từ ma trận nhầm lẫn (`confusion_matrix`):
1. **Cat (Mèo) $\leftrightarrow$ Dog (Chó):** Cặp gây nhầm lẫn lớn nhất trong chế độ Feature Extraction. Ở `resnet18_transfer`, có tới 196 ảnh Chó bị phân loại nhầm thành Mèo (chiếm gần 20% tập test của lớp Dog). Kỹ thuật Fine-Tuning đã giảm mạnh số lượng nhầm lẫn này, đưa độ chính xác Dog từ 61.9% lên 91.6%.
2. **Bird (Chim) $\leftrightarrow$ Deer / Frog:** Trong Transfer Learning, Bird chỉ đạt 61.8% độ chính xác do độ phân giải nội suy 32x32 lên 128x128 làm mờ các chi tiết vi mô (lông vũ, mỏ chim) trên nền cây cối xanh. Fine-tuning cho phép tầng tích chập thích nghi với kết cấu đặc thù, đưa độ chính xác lên 90.4%.
3. **Automobile (Ô tô con) $\leftrightarrow$ Truck (Xe tải):** Nhầm lẫn chủ yếu ở các góc chụp phía trước hoặc sau có hình dáng hộp kim loại và bánh xe tương đồng. DenseNet121 Fine-tuning phân biệt cặp này rất tốt (Truck đạt 95.5%, Automobile đạt 93.0%).
- *Giao diện trực quan:* Trực quan hóa chi tiết và trích xuất tự động có sẵn trong notebook [`notebooks/Lab2_Results_Analysis.ipynb`](../notebooks/Lab2_Results_Analysis.ipynb).

### 5.3. Hình ảnh Trực quan hóa
- Biểu đồ nhiệt Ma trận nhầm lẫn: Lưu tại thư mục [`outputs/figures/`](../outputs/figures/) (e.g. `resnet18_transfer_confusion_matrix.png`).
- Ảnh chụp màn hình TensorBoard (Loss và Accuracy theo Epoch): Lưu tại thư mục [`docs/figures/`](figures/).

---

## 6. Giới hạn Nghiên cứu Hiện tại (Current Limitations)

1. **Ràng buộc phần cứng CPU & Thời gian thực thi:** 
   Máy tính phát triển sử dụng CPU Intel Core i5-12450HX không có GPU rời NVIDIA. Thời gian huấn luyện 5 epochs trên 45.000 mẫu với ảnh 128x128 đòi hỏi khoảng 30 - 60 phút cho mỗi thí nghiệm.
2. **Nội suy phóng to ảnh (Bilinear Interpolation Artifacts):**
   Ảnh gốc CIFAR-10 có độ phân giải gốc $32 \times 32$. Khi phóng to 4 lần lên $128 \times 128$ để nạp vào các mô hình pretrained ImageNet, các chi tiết tần số cao bị mờ hóa (blurring), khiến mô hình khó phân biệt các đặc trưng vi mô của các lớp tương đồng.
3. **Giới hạn hạt giống ngẫu nhiên đơn lẻ (Single-Seed Limitation):**
   Mọi thực nghiệm hiện tại được kiểm soát ở một hạt giống cố định (`seed=42`). Do đó, số liệu đo lường chưa thể hiện khoảng tin cậy hay phương sai thống kê (mean ± standard deviation) qua nhiều lần chạy ngẫu nhiên độc lập.
4. **Hành vi cập nhật bộ đệm BatchNorm khi đóng băng Backbone (Thực nghiệm kiểm tra):**
   Đo lường thực nghiệm (`scripts/check_frozen_bn.py`) chỉ ra rằng trong chế độ `model.train()`, 100% bộ đệm BatchNorm (`running_mean`, `running_var`) của backbone đóng băng vẫn bị cập nhật theo batch CIFAR-10 hiện tại. Điều này có thể dẫn đến trôi phân phối đặc trưng đã học từ ImageNet.

---

## 7. Hướng phát triển và Cải tiến (Future Improvements)

1. **Đóng băng tường minh BatchNorm ở chế độ Eval (Frozen BatchNorm in Eval Mode):**
   Bổ sung phương thức chuyển riêng biệt các lớp `nn.BatchNorm2d` của backbone về `module.eval()` trong suốt vòng lặp train để giữ nguyên 100% thống kê tiền huấn luyện ImageNet.
2. **Đánh giá đa hạt giống (Multi-Seed Evaluation):**
   Thực thi mỗi thí nghiệm trên ít nhất 3 hạt giống khác nhau (`seed = [0, 42, 123]`) để tính giá trị trung bình và độ lệch chuẩn, khẳng định độ tin cậy thống kê của các kết luận so sánh.
3. **Bộ điều chỉnh tốc độ học (Learning Rate Scheduler):**
   Tích hợp `torch.optim.lr_scheduler.CosineAnnealingLR` hoặc `ReduceLROnPlateau` để giảm dần learning rate ở các epoch cuối, hỗ trợ mô hình hội tụ sâu hơn vào cực tiểu địa phương tối ưu.
4. **Mở rộng phần cứng:**
   Sử dụng môi trường GPU (như Google Colab T4 hoặc máy trạm CUDA) để tăng kích thước ảnh lên chuẩn ImageNet 224x224 và huấn luyện số lượng epochs lớn hơn (10 - 20 epochs).
