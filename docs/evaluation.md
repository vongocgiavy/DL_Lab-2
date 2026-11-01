# TÀI LIỆU ĐÁNH GIÁ & KẾT QUẢ THỰC NGHIỆM ĐỐI CHỨNG (EVALUATION.MD)

**Chủ đề:** Đánh giá độc lập trên tập Test (10.000 ảnh), phân tích đối chứng 6 thí nghiệm, ma trận nhầm lẫn và độ chính xác theo lớp  
**Module triển khai:** [`src/lab2/evaluate.py`](file:///d:/Hoc_Sau/Lab 2/src/lab2/evaluate.py), [`scripts/make_results_table.py`](file:///d:/Hoc_Sau/Lab 2/scripts/make_results_table.py)  
**Tệp số liệu nguồn:** [`outputs/metrics/summary.csv`](file:///d:/Hoc_Sau/Lab 2/outputs/metrics/summary.csv), [`outputs/metrics/*_test_metrics.json`](file:///d:/Hoc_Sau/Lab 2/outputs/metrics/)

---

## 1. NGUYÊN TẮC ĐÁNH GIÁ KHÁCH QUAN & PHÒNG CHỐNG DATA LEAKAGE

1. **Tập Test cách ly tuyệt đối:** Tập Test gồm 10.000 ảnh CIFAR-10 không bao giờ được tham gia vào quá trình lan truyền ngược hoặc chọn lọc siêu tham số.
2. **Nguyên tắc Best Checkpoint:** Chỉ có checkpoint đạt `val_acc` cao nhất trên tập Validation (`*_best.pth`) mới được nạp vào để đánh giá trên tập Test đúng **một lần duy nhất**. Tuyệt đối không chọn mô hình dựa trên độ chính xác tập Test.
3. **Môi trường đo lường đồng nhất:** Cố định phần cứng, framework (`torch 2.6.0+cu124` / `torchvision 0.21.0+cu124`) và hạt giống ngẫu nhiên `seed = 42`.

---

## 2. BẢNG TỔNG HỢP KẾT QUẢ CHÍNH THỨC (6 THÍ NGHIỆM)

Dữ liệu đối chiếu 100% từ bảng tổng kết [`outputs/metrics/summary.csv`](file:///d:/Hoc_Sau/Lab 2/outputs/metrics/summary.csv):

| STT | Thí nghiệm ID | Nhóm | Mô hình | Phương pháp | LR (Head / Backbone) | Batch | Tham số Trainable (%) | Train Loss / Acc (ep 5) | Best Val Acc (epoch) | Test Loss | **Test Acc (%)** | Thời gian TB/epoch | Tổng thời gian |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | `resnet18_transfer` *(Baseline)* | Core | ResNet-18 | Transfer | 1e-3 / — | 16 | 5.130 *(0,05%)* | 0.7718 / 73.85% | 77.94% *(ep 2)* | 0.6644 | **77.98%** | 01m 06s | 05m 35s |
| 2 | `resnet18_finetune` | Core | ResNet-18 | Fine-Tuning | 1e-3 / 1e-4 | 16 | 8.398.858 *(75,11%)* | 0.2488 / 91.48% | 91.68% *(ep 4)* | 0.2989 | **90.24%** | 01m 26s | 07m 15s |
| 3 | `densenet121_transfer` | Core | DenseNet-121 | Transfer | 1e-3 / — | 16 | 10.250 *(0,15%)* | 0.6860 / 76.89% | 81.84% *(ep 5)* | 0.5560 | **81.34%** | 01m 48s | 09m 03s |
| 4 | `densenet121_finetune` | Core | DenseNet-121 | Fine-Tuning | 1e-3 / 1e-4 | 16 | 2.170.378 *(31,17%)* | 0.2912 / 89.94% | 90.78% *(ep 5)* | 0.2982 | **90.65%** | 02m 08s | 10m 42s |
| 5 | `resnet18_transfer_lr1e4` | Hparam | ResNet-18 | Transfer | 1e-4 / — | 16 | 5.130 *(0,05%)* | 0.7605 / 73.96% | 77.56% *(ep 5)* | 0.6643 | **78.32%** | 01m 12s | 06m 03s |
| 6 | `resnet18_transfer_bs32` | Hparam | ResNet-18 | Transfer | 1e-3 / — | 32 | 5.130 *(0,05%)* | 0.7029 / 76.12% | 78.92% *(ep 4)* | 0.6276 | **79.04%** | 01m 04s | 05m 21s |

---

## 3. PHÂN TÍCH SO SÁNH ĐỐI CHỨNG ĐA CHIỀU

### 3.1. Đối chứng 1: Transfer Learning vs. Fine-Tuning
- **Trên ResNet-18:** Fine-Tuning tăng Test Accuracy từ **77,98% lên 90,24% (+12,26%)**. Chi phí đánh đổi: số lượng tham số trainable tăng ~1.637 lần (từ 5.130 lên 8.398.858), thời gian huấn luyện tăng $29,8\%$ (từ 5m35s lên 7m15s).
- **Trên DenseNet-121:** Fine-Tuning tăng Test Accuracy từ **81,34% lên 90,65% (+9,31%)**. Chi phí đánh đổi: tham số trainable tăng ~211 lần (từ 10.250 lên 2.170.378), thời gian tăng $18,2\%$ (từ 9m03s lên 10m42s).
- **Kết luận:** Mở khóa các tầng tích chập sâu nhất là bước ngoặt bắt buộc để đưa độ chính xác vượt mốc 90% trên CIFAR-10, giúp các bộ lọc cấp cao tái học các biểu diễn ngữ nghĩa của đối tượng phân giải nhỏ.

### 3.2. Đối chứng 2: ResNet-18 vs. DenseNet-121
- **Hiệu quả trích xuất đặc trưng (Transfer Learning):** DenseNet-121 (**81,34%**) áp đảo ResNet-18 (**77,98%**, chênh lệch **+3,36%**). Cơ chế ghép nối kênh tích lũy bản đồ đặc trưng đa tầng giúp vector 1.024 chiều giàu biểu diễn hơn vector 512 chiều của phép cộng phần dư.
- **Độ chính xác tối đa (Fine-Tuning):** DenseNet-121 đạt **90,65%** (cao nhất dự án), cao hơn ResNet-18 (**90,24%**).
- **Nghịch lý Hiệu quả Tham số vs. Thời gian:**
  - DenseNet-121 chỉ cần cập nhật **2.170.378 tham số** (bằng 1/4 so với 8.398.858 của ResNet-18).
  - Tuy nhiên, thời gian huấn luyện DenseNet-121 lại lâu nhất (**10m42s** so với **7m15s** của ResNet-18) do thao tác ghép kênh tensor liên tục đòi hỏi băng thông bộ nhớ và chi phí ghép nối cao hơn phép cộng trực tiếp.

### 3.3. Đối chứng 3: Khảo sát Siêu tham số (Hyperparameters Sensitivity)
- **Tốc độ học (Learning Rate: 1e-3 vs. 1e-4):**
  - LR $10^{-3}$ (Baseline): Đạt đỉnh validation sớm ở epoch 2 (77,94%), sau đó dao động nhẹ quanh cực trị, Test Acc đạt **77,98%**.
  - LR $10^{-4}$: Hội tụ chậm hơn trên tập train nhưng ổn định và tiến xa hơn trên tập test, đạt đỉnh ở epoch 5 (77,56%), Test Acc đạt **78,32%** (+0,34%).
- **Kích thước Batch (Batch Size: 16 vs. 32):**
  - Tăng Batch Size lên 32 giúp Test Acc đạt **79,04%** (+1,06% so với batch 16), đồng thời rút ngắn thời gian huấn luyện xuống **05m 21s** (nhanh nhất toàn bộ 6 thí nghiệm) nhờ tận dụng khả năng tính toán song song tốt hơn.

---

## 4. MA TRẬN NHẦM LẪN & ĐỘ CHÍNH XÁC THEO LỚP (PER-CLASS ACCURACY)

Trích xuất trực tiếp từ các file kết quả chi tiết [`outputs/metrics/*_test_metrics.json`](file:///d:/Hoc_Sau/Lab 2/outputs/metrics/):

| Lớp (Class) | Số mẫu test | ResNet-18 Transfer Acc | ResNet-18 Fine-tune Acc | DenseNet-121 Transfer Acc | DenseNet-121 Fine-tune Acc | Nhận xét đối chiếu |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `airplane` | 1.000 | 82,0% | **95,0%** | 87,6% | 94,0% | Nhận diện xuất sắc trên cả hai mô hình |
| `automobile` | 1.000 | 85,6% | **80,4%** | 85,7% | **93,0%** | **DenseNet vượt trội (+12,6%)**, ít nhầm sang `truck` |
| `bird` | 1.000 | 61,8% | **90,4%** | 74,2% | 87,0% | ResNet nhỉnh hơn |
| `cat` | 1.000 | 70,1% | **77,8%** *(kém nhất)* | 71,4% | **81,6%** *(kém nhất)* | Lớp khó nhất của CIFAR-10 do biến dạng tư thế lớn |
| `deer` | 1.000 | 70,8% | **90,4%** | 76,1% | 86,6% | ResNet nhỉnh hơn |
| `dog` | 1.000 | 61,9% | **91,6%** | 70,9% | 84,3% | ResNet phân biệt chó tốt hơn |
| `frog` | 1.000 | 91,9% | 91,4% | 86,3% | **94,4%** | DenseNet tốt hơn |
| `horse` | 1.000 | 80,9% | 92,7% | 83,4% | **96,2%** *(tốt nhất)* | DenseNet nhận diện dáng ngựa rất tốt |
| `ship` | 1.000 | 87,3% | 94,3% | 85,9% | 93,9% | Tương đương, rất cao |
| `truck` | 1.000 | 87,5% | **98,4%** *(tốt nhất)* | 91,9% | 95,5% | ResNet bắt đặc trưng xe tải cực mạnh |

### Top các cặp nhầm lẫn phổ biến nhất:
- **ResNet-18 Fine-Tuning:**
  1. `automobile` $\to$ `truck`: **174 mẫu** *(Xe hơi bị nhận dạng nhầm thành xe tải do góc chụp vuông vức)*
  2. `cat` $\to$ `dog`: **138 mẫu** *(Mèo bị nhầm thành chó do đặc trưng mặt thú 4 chân lông xù)*
  3. `dog` $\to$ `cat`: **59 mẫu** *(Chó bị nhầm thành mèo)*
- **DenseNet-121 Fine-Tuning:**
  1. `dog` $\to$ `cat`: **92 mẫu**
  2. `cat` $\to$ `dog`: **90 mẫu**
  3. `deer` $\to$ `horse`: **47 mẫu**
  4. `automobile` $\to$ `truck`: chỉ **46 mẫu** *(DenseNet khắc phục triệt để lỗi nhầm lẫn xe hơi - xe tải so với ResNet)*

---

## 5. HẠN CHẾ THỰC NGHIỆM & KHOA HỌC CẦN LƯU Ý

1. **Ràng buộc Seed ngẫu nhiên:** Toàn bộ 6 thí nghiệm được thực hiện trên **cùng một hạt giống duy nhất (`seed=42`)**. Do đó, sự chênh lệch nhỏ (như 90,65% vs 90,24%) chưa đủ để khẳng định có ý nghĩa thống kê nếu không đo khoảng tin cậy qua 3-5 seeds.
2. **Không suy diễn quan hệ nhân quả tuyệt đối:** Chưa đủ căn cứ để kết luận DenseNet luôn vượt trội ResNet trên mọi bài toán; ưu thế ở đây gắn liền với việc các feature maps đa tầng hỗ trợ tốt cho ảnh phân giải nhỏ được nội suy ($128 \times 128$).
3. **Môi trường thực nghiệm:** Dữ liệu lịch sử [`outputs/metrics/resnet18_transfer_history.json`](file:///d:/Hoc_Sau/Lab 2/outputs/metrics/resnet18_transfer_history.json) ghi nhận runtime sử dụng CUDA backend (`torch 2.6.0+cu124`). Cần nói rõ môi trường thực tế khi thuyết trình để đảm bảo tính trung thực học thuật.

