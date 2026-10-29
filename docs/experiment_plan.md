# Experiment Plan — Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning

## 1. Mục đích thực nghiệm (Experiment Goals)
Kế hoạch thực nghiệm này được thiết kế nhằm so sánh có hệ thống và công bằng (fair comparison) giữa hai họ kiến trúc mạng sâu kinh điển (**ResNet18** và **DenseNet121**) dưới hai phương thức thích nghi mô hình (**Transfer Learning** và **Fine-Tuning**) trên tập dữ liệu phân loại ảnh **CIFAR-10**, đồng thời khảo sát ảnh hưởng của các siêu tham số then chốt (Learning Rate và Batch Size).

---

## 2. Ma trận 6 Thí nghiệm Chi tiết (Experiment Matrix)

Dự án được chia thành 2 nhóm thí nghiệm rõ ràng: Nhóm **Core** (4 thí nghiệm đối đầu kiến trúc và phương pháp) và Nhóm **Hyperparameter** (2 thí nghiệm khảo sát biến số):

| Experiment ID | Nhóm (Group) | Kiến trúc | Phương pháp | Tầng huấn luyện (Trainable) | Tầng đóng băng (Frozen) | Optimizer & Learning Rate | Batch Size | Epochs |
|---|:---:|:---:|:---:|---|---|---|:---:|:---:|
| `resnet18_transfer` | `core` | ResNet18 | Transfer Learning | Duy nhất `model.fc` | `conv1`, `bn1`, `layer1`..`layer4` | Adam (`lr=1e-3`, `wd=1e-4`) | 16 | 5 |
| `resnet18_finetune` | `core` | ResNet18 | Fine-Tuning | `model.layer4` + `model.fc` | `conv1`, `bn1`, `layer1`..`layer3` | Adam (`lr_backbone=1e-4`, `lr_head=1e-3`) | 16 | 5 |
| `densenet121_transfer` | `core` | DenseNet121 | Transfer Learning | Duy nhất `model.classifier` | Toàn bộ `features` (`conv0`..`norm5`) | Adam (`lr=1e-3`, `wd=1e-4`) | 16 | 5 |
| `densenet121_finetune` | `core` | DenseNet121 | Fine-Tuning | `denseblock4` + `norm5` + `classifier` | `conv0` đến `denseblock3` | Adam (`lr_backbone=1e-4`, `lr_head=1e-3`) | 16 | 5 |
| `resnet18_transfer_lr1e4` | `hparam` | ResNet18 | Transfer Learning | Duy nhất `model.fc` | `conv1`, `bn1`, `layer1`..`layer4` | Adam (**`lr=1e-4`**, `wd=1e-4`) | 16 | 5 |
| `resnet18_transfer_bs32` | `hparam` | ResNet18 | Transfer Learning | Duy nhất `model.fc` | `conv1`, `bn1`, `layer1`..`layer4` | Adam (`lr=1e-3`, `wd=1e-4`) | **32** | 5 |

---

## 3. Các biến cần kiểm soát (Controlled Variables)

Để đảm bảo tính khoa học và giá trị so sánh công bằng giữa các cấu hình:
1. **Dữ liệu phân chia (Data Split):** Toàn bộ 6 thí nghiệm dùng chung 100% cùng quy tắc chia dữ liệu:
   - 45.000 mẫu train.
   - 5.000 mẫu validation (seed ngẫu nhiên cố định = 42).
   - 10.000 mẫu test chính thức.
2. **Quy tắc tiền xử lý (Preprocessing):**
   - Kích thước ảnh: $128 \times 128$.
   - Chuẩn hóa: ImageNet mean `[0.485, 0.456, 0.406]` và std `[0.229, 0.224, 0.225]`.
   - Augmentation: Giống hệt nhau trên tập train (`RandomCrop`, `RandomHorizontalFlip(p=0.5)`) và không có augmentation trên validation/test.
3. **Hàm mất mát (Loss Function):** Sử dụng `nn.CrossEntropyLoss()` cho toàn bộ các thí nghiệm.
4. **Quy tắc chọn Checkpoint:** Dựa trên cùng 1 thước đo duy nhất: `val_acc` cao nhất trong quá trình huấn luyện.
5. **Môi trường phần cứng:** Các thí nghiệm 5 epoch chính thức được huấn luyện trên GPU NVIDIA CUDA (`device='cuda'`, PyTorch 2.6.0+cu124) để đảm bảo tính khả thi về thời gian thực nghiệm (5–11 phút/thí nghiệm); hệ thống hỗ trợ thực thi trên CPU thông qua cờ `--device cpu` hoặc khi chạy smoke test / unit test.
6. **Giới hạn hạt giống ngẫu nhiên (Single-Seed Limitation):** Toàn bộ thực nghiệm hiện cố định tại một hạt giống ngẫu nhiên duy nhất (`seed=42`). Mặc dù đảm bảo tính tái lập 100% cho bài thực hành, kết quả chưa phản ánh phương sai thống kê (mean ± std qua nhiều seed khác nhau như 0, 42, 123). Đây là một giới hạn thực nghiệm được ghi nhận tường minh.

---

## 4. Biến độc lập (Independent Variables)

### 4.1. Trong nhóm 4 thí nghiệm Core:
1. **Kiến trúc mạng:**
   - **ResNet18:** Kết nối tắt phần dư (Residual Connections, cộng dồn tensor), 11.18 triệu tham số.
   - **DenseNet121:** Kết nối dồn kênh dày đặc (Dense Connectivity, ghép nối tensor theo chiều channel), 6.96 triệu tham số.
2. **Phương pháp thích nghi:**
   - **Transfer Learning (Feature Extraction):** Giữ nguyên toàn bộ backbone trích xuất đặc trưng của ImageNet, chỉ tối ưu classification head mới cho 10 lớp CIFAR-10.
   - **Fine-Tuning:** Mở khóa các khối tích chập sâu nhất của backbone kết hợp tối ưu đồng thời classification head với learning rate vi phân (differential learning rate).

### 4.2. Trong nhóm 2 thí nghiệm Hyperparameter:
1. **Tốc độ học (Learning Rate):** Khảo sát `resnet18_transfer` ở 2 mức tốc độ học: `lr = 1e-3` (chuẩn) vs `lr = 1e-4` (nhỏ hơn 10 lần).
2. **Kích thước Batch (Batch Size):** Khảo sát `resnet18_transfer` ở 2 mức kích thước mini-batch: `batch_size = 16` (chuẩn) vs `batch_size = 32` (gấp đôi).

---

## 5. Tiêu chí Đánh giá và Biến phụ thuộc (Evaluation Metrics)

1. **Độ chính xác (Accuracy):**
   - Training Accuracy theo từng epoch.
   - Validation Accuracy theo từng epoch.
   - **Test Accuracy cuối cùng** trên tập kiểm thử 10.000 mẫu độc lập (thước đo chính).
2. **Độ mất mát (Loss):**
   - Training Loss & Validation Loss qua từng epoch (phát hiện sớm hiện tượng Underfitting hoặc Overfitting).
   - Test Loss.
3. **Độ chính xác theo từng lớp (Per-class Accuracy):**
   - Phân tích chi tiết độ chính xác phân loại trên từng lớp trong 10 lớp của CIFAR-10 (nhận diện các lớp khó/dễ).
4. **Ma trận nhầm lẫn (Confusion Matrix):**
   - Trực quan hóa heatmap và xác định các cặp lớp có tỷ lệ nhầm lẫn cao nhất (e.g. Cat vs Dog, Automobile vs Truck).
5. **Hiệu năng và Tài nguyên tính toán:**
   - Số lượng tham số tối ưu (`trainable parameters`) và tỷ lệ phần trăm trên tổng số tham số (`% trainable`).
   - Thời gian huấn luyện trung bình mỗi epoch và tổng thời gian huấn luyện đo trên CPU.

---

## 6. Quy trình Thực thi (Execution Procedure)

1. **Bước 1 — Kiểm thử tự động:** Chạy `python -m pytest -q` đảm bảo 62/62 unit & integration tests vượt qua thành công.
2. **Bước 2 — Smoke Test:** Chạy `py scripts/run_experiments.py --smoke-test` để kiểm tra thông suốt luồng data, forward, backward, checkpoint, TensorBoard và evaluation.
3. **Bước 3 — Huấn luyện 4 thí nghiệm Core:**
   ```bash
   py scripts/run_experiments.py --all
   # hoặc:
   py scripts/run_experiments.py --group core
   ```
4. **Bước 4 — Huấn luyện 2 thí nghiệm Hyperparameter:**
   ```bash
   py scripts/run_experiments.py --group hparam
   ```
5. **Bước 5 — Giám sát đường cong qua TensorBoard:**
   ```bash
   tensorboard --logdir=runs
   ```
6. **Bước 6 — Sinh bảng tổng hợp định lượng:**
   ```bash
   py scripts/make_results_table.py
   ```
7. **Bước 7 — Phân tích lỗi nâng cao trên Notebook:**
   Mở [`notebooks/Lab2_Results_Analysis.ipynb`](../notebooks/Lab2_Results_Analysis.ipynb) để trích xuất các cặp nhầm lẫn hàng đầu và trực quan hóa phân tích đa biến.
