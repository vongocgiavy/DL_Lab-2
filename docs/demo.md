# HƯỚNG DẪN THỰC THI & DEMO HỆ THỐNG (DEMO.MD)

**Chủ đề:** Hướng dẫn cài đặt, chạy thực nghiệm CLI, sinh biểu đồ trực quan, khởi chạy TensorBoard và thực hiện suy luận (Inference Demo)  
**Môi trường:** Python 3.10+, PyTorch 2.x, Torchvision, CUDA hoặc CPU  
**Tập lệnh CLI:** Thư mục [`scripts/`](file:///d:/Hoc_Sau/Lab 2/scripts/)

---

## 1. HƯỚNG DẪN CÀI ĐẶT NHANH (QUICKSTART)

### 1.1. Chuẩn bị môi trường Python
```bash
# Tạo và kích hoạt môi trường ảo (Virtual Environment)
python -m venv .venv
# Trên Windows:
.venv\Scripts\activate
# Trên Linux/macOS:
source .venv/bin/activate

# Cài đặt toàn bộ các thư viện phụ thuộc
pip install -r requirements.txt
```

### 1.2. Kiểm tra tính toàn vẹn hệ thống qua Unit Tests
Chạy toàn bộ bộ kiểm thử tự động gồm **53 unit & integration tests**:
```bash
pytest -v
```
*(Yêu cầu: 53 passed in ~40s).*

---

## 2. CHẠY THỰC NGHIỆM QUA DÒNG LỆNH CLI

### 2.1. Chế độ Smoke Test (Kiểm tra khói nhanh trong 30 giây)
Xác minh code chạy thông suốt không lỗi cú pháp hay shape mismatch trước khi tốn tài nguyên chạy full:
```bash
python scripts/run_experiments.py --smoke-test
```
*(Chạy 1 epoch trên tập dữ liệu rút gọn 120 mẫu train / 30 mẫu val / 30 mẫu test).*

### 2.2. Chạy một thí nghiệm đơn lẻ
```bash
# 1. ResNet-18 Transfer Learning (Baseline)
python scripts/run_experiments.py --experiment resnet18_transfer

# 2. ResNet-18 Fine-Tuning
python scripts/run_experiments.py --experiment resnet18_finetune

# 3. DenseNet-121 Transfer Learning
python scripts/run_experiments.py --experiment densenet121_transfer

# 4. DenseNet-121 Fine-Tuning
python scripts/run_experiments.py --experiment densenet121_finetune

# 5. Khảo sát Learning Rate 1e-4
python scripts/run_experiments.py --experiment resnet18_transfer_lr1e4

# 6. Khảo sát Batch Size 32
python scripts/run_experiments.py --experiment resnet18_transfer_bs32
```

### 2.3. Chạy theo nhóm thí nghiệm
```bash
# Chạy 4 thí nghiệm nòng cốt (Core experiments)
python scripts/run_experiments.py --group core

# Chạy 2 thí nghiệm siêu tham số (Hyperparameters experiments)
python scripts/run_experiments.py --group hparam
```

---

## 3. CÔNG CỤ PHÂN TÍCH & TRỰC QUAN HÓA

### 3.1. Đếm và phân tích cấu trúc tham số mô hình
```bash
python scripts/inspect_models.py
```
*(Hiển thị bảng chi tiết số tham số Trainable, Frozen và tỷ lệ % cho cả ResNet-18 và DenseNet-121).*

### 3.2. Sinh các biểu đồ dữ liệu thô & PCA
```bash
# 1. Trực quan hóa lưới ảnh thô 10 lớp và phân bố RGB
python scripts/visualize_raw_data.py
# -> Xuất file: outputs/figures/cifar10_raw_samples_grid.png & cifar10_raw_distribution.png

# 2. Phân tích Thành phần Chính (PCA) trên pixel thô 3072 chiều
python scripts/visualize_raw_pca.py
# -> Xuất file: outputs/figures/cifar10_raw_pca.png

# 3. Sinh sơ đồ kiến trúc giám sát (Monitoring Diagram)
python scripts/generate_monitoring_diagram.py
# -> Xuất file: outputs/figures/monitoring_architecture.png
```

### 3.3. Đánh giá Checkpoint và Tổng hợp bảng kết quả
```bash
# Đánh giá lại toàn bộ các checkpoint tốt nhất trên tập Test 10.000 ảnh
python scripts/evaluate_checkpoints.py

# Đọc các file JSON trong outputs/metrics/ và sinh bảng kết quả summary.csv
python scripts/make_results_table.py
```

---

## 4. GIÁM SÁT THỜI GIAN THỰC VỚI TENSORBOARD

Khởi động giao diện TensorBoard để theo dõi trực quan các đường cong huấn luyện:
```bash
tensorboard --logdir=runs
```
Mở trình duyệt truy cập: **`http://localhost:6006`**

- **Scalars Tab:** So sánh trực quan đường cong Train/Val Loss và Train/Val Accuracy giữa các thí nghiệm.
- **Images Tab:** Xem các ảnh mẫu batch đầu vào sau khi qua tiền xử lý ($128 \times 128$) và data augmentation.
- **Graphs Tab:** Khảo sát luồng tensor tính toán bên trong mạng ResNet-18 và DenseNet-121.
- **Histograms Tab:** Kiểm tra sự thay đổi của phân phối trọng số và gradient theo từng epoch.

---

## 5. DEMO SUY LUẬN VỚI MẪU ẢNH MỚI (INFERENCE CODE SNIPPET)

Đoạn mã Python thực hiện nạp Best Checkpoint và dự đoán nhãn lớp cho một ảnh đầu vào bất kỳ:

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
from lab2.models import build_model
from lab2.data import CIFAR10_CLASSES, IMAGENET_MEAN, IMAGENET_STD

def predict_single_image(image_path: str, checkpoint_path: str = "./checkpoints/resnet18_finetune_best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Xây dựng mô hình và nạp checkpoint tốt nhất
    model = build_model(model_name="resnet18", method="finetune", num_classes=10, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 2. Tiền xử lý ảnh đầu vào
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    raw_img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(raw_img).unsqueeze(0).to(device)  # Shape: [1, 3, 128, 128]

    # 3. Suy luận không tính gradient
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_prob, top_idx = torch.topk(probabilities, k=3)

    print(f"\n--- KẾT QUẢ DỰ ĐOÁN CHO: {image_path} ---")
    for rank in range(3):
        cls_name = CIFAR10_CLASSES[top_idx[rank].item()]
        prob_pct = top_prob[rank].item() * 100
        print(f"Top {rank + 1}: {cls_name:12s} - Xác suất: {prob_pct:.2f}%")

# Sử dụng:
# predict_single_image("data/test_sample.jpg")
```

