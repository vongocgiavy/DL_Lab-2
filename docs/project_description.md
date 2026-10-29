# Project Description — Lab 2: Pretrained Models, Transfer Learning & Fine-Tuning

## 1. Giới thiệu bài toán (Problem Statement)
Trong thị giác máy tính, việc huấn luyện các mạng nơ-ron tích chập (Convolutional Neural Networks - CNNs) từ đầu (from scratch) trên các tập dữ liệu nhỏ thường gặp nhiều khó khăn:
- Nguy cơ quá khớp (overfitting) rất cao do số lượng tham số lớn so với kích thước tập dữ liệu.
- Đòi hỏi tài nguyên tính toán phần cứng lớn và thời gian huấn luyện kéo dài.
- Khó hội tụ và dễ bị hiện tượng triệt tiêu gradient (vanishing gradient).

Để khắc phục các thách thức trên, kỹ thuật **Transfer Learning (Học chuyển giao)** và **Fine-Tuning (Tinh chỉnh tham số)** cho phép tái sử dụng các tri thức (biểu diễn đặc trưng không gian sâu) từ các mô hình đã được huấn luyện trước (Pretrained Models) trên các tập dữ liệu quy mô lớn (như ImageNet - 1.2 triệu ảnh, 1.000 lớp) để áp dụng vào bài toán phân loại ảnh trên tập dữ liệu mục tiêu **CIFAR-10**.

## 2. Mục tiêu dự án (Objectives)
- **Tiếp cận và khai thác Pretrained Models:** Sử dụng chính thống các pretrained weights từ `torchvision.models` (ResNet18 và DenseNet121).
- **So sánh 2 chiến lược thích nghi mô hình:**
  1. *Transfer Learning (Feature Extraction):* Đóng băng hoàn toàn backbone trích xuất đặc trưng, chỉ huấn luyện classifier mới cho 10 lớp CIFAR-10.
  2. *Fine-Tuning:* Mở khóa các tầng tích chập sâu nhất của backbone (layer4 của ResNet18; denseblock4 và norm5 của DenseNet121) cùng với classifier mới để tối ưu hóa đồng thời với learning rate phù hợp.
- **Tuân thủ quy trình khoa học (Scientific & Test-Driven Workflow):**
  - Đảm bảo chia tập dữ liệu tái lập (reproducible split), tuyệt đối không để rò rỉ dữ liệu (data leakage) giữa tập train, validation và test.
  - Sử dụng tiêu chí validation metric để lựa chọn checkpoint tốt nhất.
  - Chỉ đánh giá tập Test sau khi đã khóa checkpoint tốt nhất.
  - Giám sát toàn bộ tiến trình huấn luyện qua TensorBoard theo đúng nội dung Chương 3 (Training Deep Neural Networks).

## 3. Phạm vi dự án (Scope)
- **Tập dữ liệu:** CIFAR-10 (60.000 ảnh màu 32x32, 10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
  - Train: 45.000 ảnh.
  - Validation: 5.000 ảnh (tách từ train chính thức bằng seed cố định `seed=42`).
  - Test: 10.000 ảnh (giữ nguyên tập test chính thức).
- **Kiến trúc mô hình:**
  - `ResNet18` (Residual Networks - 18 layers).
  - `DenseNet121` (Densely Connected Convolutional Networks - 121 layers).
- **Môi trường phần cứng:** Tối ưu hóa CPU-first (Intel Core i5-12450HX, 24 GB RAM, Windows 11).

## 4. Đối chiếu yêu cầu bài thực hành (Practice 2 Requirements Cross-Reference)

Bảng dưới đây đối chiếu chi tiết giữa các mục tiêu, 7 bước thực hiện và 3 bài tập mở rộng trong tài liệu thực hành Practice 2 (Slide 23–24, Chương 3) với mã nguồn và notebook thực tế của dự án:

| Hạng mục | Yêu cầu bài thực hành Practice 2 | Phương pháp triển khai trong dự án | File / Vị trí minh chứng | Trạng thái |
|---|---|---|---|---|
| **Mục tiêu 1** | Hiểu cơ chế Transfer Learning & Fine-Tuning | Đóng băng tham số bằng `requires_grad=False`, thay thế classification head cho 10 lớp; mở khóa tầng sâu cho fine-tuning | [`src/lab2/models.py`](../src/lab2/models.py), [`notebooks/Lab2_Practice2_Main.ipynb`](../notebooks/Lab2_Practice2_Main.ipynb) Bước 3-5 | Đã hoàn thành |
| **Mục tiêu 2** | Đo lường trade-off giữa thời gian, tham số và độ chính xác | Đo số tham số huấn luyện (0.05% vs 75.11% ResNet, 0.15% vs 31.17% DenseNet), đo runtime và đánh giá độ chính xác | [`scripts/inspect_models.py`](../scripts/inspect_models.py), [`scripts/make_results_table.py`](../scripts/make_results_table.py), [`notebooks/Lab2_Results_Analysis.ipynb`](../notebooks/Lab2_Results_Analysis.ipynb) | Đã hoàn thành |
| **Bước 1** | Cài đặt môi trường & thư viện | Khởi tạo venv, cài đặt PyTorch, torchvision, TensorBoard, PyYAML, pytest, jupyter | [`requirements.txt`](../requirements.txt), [`README.md`](../README.md) | Đã hoàn thành |
| **Bước 2** | Nạp & tiền xử lý dữ liệu CIFAR-10 | Resize 128x128, ImageNet normalize, data augmentation tập train (RandomCrop, RandomHorizontalFlip), split 45k train / 5k val | [`src/lab2/data.py`](../src/lab2/data.py), [`tests/test_data.py`](../tests/test_data.py) | Đã hoàn thành |
| **Bước 3** | Khởi tạo mô hình Pretrained | Nạp ResNet18 (`ResNet18_Weights.DEFAULT`) và DenseNet121 (`DenseNet121_Weights.DEFAULT`) từ `torchvision.models` | [`src/lab2/models.py`](../src/lab2/models.py), [`tests/test_models.py`](../tests/test_models.py) | Đã hoàn thành |
| **Bước 4** | Thực hiện Transfer Learning (Feature Extraction) | Đóng băng toàn bộ backbone, chỉ mở khóa `fc` (ResNet18, 5.130 params) hoặc `classifier` (DenseNet121, 10.250 params) | [`src/lab2/models.py`](../src/lab2/models.py), [`src/lab2/train.py`](../src/lab2/train.py) | Đã hoàn thành |
| **Bước 5** | Thực hiện Fine-Tuning | Mở khóa tầng sâu (`layer4` + `fc` cho ResNet18; `denseblock4` + `norm5` + `classifier` cho DenseNet121), dùng differential LR | [`src/lab2/models.py`](../src/lab2/models.py), [`src/lab2/train.py`](../src/lab2/train.py) | Đã hoàn thành |
| **Bước 6** | Đánh giá & so sánh hiệu năng | Đánh giá checkpoint tốt nhất trên tập Test độc lập: Test Loss, Test Accuracy, Per-class Accuracy, Confusion Matrix | [`src/lab2/evaluate.py`](../src/lab2/evaluate.py), [`scripts/evaluate_checkpoints.py`](../scripts/evaluate_checkpoints.py) | Đã hoàn thành |
| **Bước 7** | Trực quan hóa & TensorBoard | Ghi log `SummaryWriter` cho loss/acc train/val mỗi epoch, vẽ confusion matrix heatmap và đồ thị so sánh | [`src/lab2/train.py`](../src/lab2/train.py), [`src/lab2/utils.py`](../src/lab2/utils.py), [`notebooks/Lab2_Practice2_Main.ipynb`](../notebooks/Lab2_Practice2_Main.ipynb) | Đã hoàn thành |
| **Bài tập 1** | Thử nghiệm kiến trúc khác (ResNet18, VGG16, DenseNet121) | Triển khai ResNet18 và DenseNet121 cho cả Transfer Learning và Fine-Tuning (4 cấu hình). Không dùng VGG16 (chi tiết giải trình bên dưới) | [`src/lab2/models.py`](../src/lab2/models.py), [`notebooks/Lab2_Practice2_Main.ipynb`](../notebooks/Lab2_Practice2_Main.ipynb) Bài tập 1 | Đã hoàn thành |
| **Bài tập 2** | Thử nghiệm siêu tham số (Learning Rate, Batch Size) | Thiết kế 2 thí nghiệm kiểm soát: `resnet18_transfer_lr1e4` (LR=1e-4 vs 1e-3) và `resnet18_transfer_bs32` (batch=32 vs 64) | [`configs/config.yaml`](../configs/config.yaml), [`notebooks/Lab2_Practice2_Main.ipynb`](../notebooks/Lab2_Practice2_Main.ipynb) Bài tập 2 | Đã hoàn thành |
| **Bài tập 3** | Phân tích lỗi & độ chính xác từng lớp | Tính per-class accuracy, tìm các cặp lớp nhầm lẫn nhiều nhất từ Confusion Matrix và phân tích nguyên nhân thị giác | [`notebooks/Lab2_Results_Analysis.ipynb`](../notebooks/Lab2_Results_Analysis.ipynb), [`src/lab2/evaluate.py`](../src/lab2/evaluate.py) | Đã hoàn thành |

### Giải trình về lựa chọn kiến trúc trong Bài tập 1 (Tại sao không dùng VGG16)
Trong đề cương thực hành gợi ý các kiến trúc như ResNet18, VGG16, DenseNet121. Dự án lựa chọn **ResNet18** và **DenseNet121**, không triển khai **VGG16** vì các lý do kỹ thuật sau:
1. **Quy mô tham số và tài nguyên CPU:** VGG16 có hơn 138 triệu tham số, trong đó riêng 3 tầng Fully Connected chiếm tới ~120 triệu tham số. Trên môi trường huấn luyện CPU không có GPU rời, việc nạp và lan truyền xuôi/ngược qua 138M tham số tiêu tốn thời gian rất lớn và bộ nhớ RAM cao gấp 12-20 lần so với ResNet18 (11.2M tham số) và DenseNet121 (7.0M tham số).
2. **Tính đại diện về mặt kiến trúc hiện đại:** 
   - ResNet18 đại diện cho trường phái **Residual Connections (kết nối tắt cộng dồn)** giải quyết triệt để vấn đề triệt tiêu gradient.
   - DenseNet121 đại diện cho trường phái **Dense Connectivity (kết nối dồn kênh ghép nối)** tối đa hóa việc tái sử dụng đặc trưng không gian.
   - So sánh giữa ResNet18 và DenseNet121 mang lại giá trị học thuật cao hơn nhiều so với việc đưa thêm một kiến trúc tuần tự cổ điển nặng nề như VGG16 vào một môi trường tính toán hạn chế.

## 5. Đầu ra dự kiến (Expected Deliverables)
1. **Mã nguồn modular hoàn chỉnh:**
   - Cấu trúc thư mục chuẩn mực (`src/lab2/`, `scripts/`, `configs/`, `tests/`).
   - Module nạp cấu hình, chuẩn bị dữ liệu, xây dựng mô hình, vòng lặp train/val, checkpoint, TensorBoard và đánh giá.
2. **Bộ kiểm thử tự động (Unit & Integration Tests):**
   - Đảm bảo 100% test pass với pytest (62/62 tests), kiểm chứng độc lập tính đóng băng/mở khóa tham số, luồng tính toán gradient, pipeline dữ liệu, checkpoint snapshot sâu và notebook syntax.
3. **Checkpoints & Logs:**
   - Các file checkpoint tốt nhất (`*_best.pth`) và mới nhất (`*_latest.pth`).
   - TensorBoard tfevent logs theo dõi đường cong Loss và Accuracy.
4. **Notebook tương tác:**
   - [`notebooks/Lab2_Practice2_Main.ipynb`](../notebooks/Lab2_Practice2_Main.ipynb): Thực hiện toàn bộ 7 bước và 3 bài tập của đề bài.
   - [`notebooks/Lab2_Results_Analysis.ipynb`](../notebooks/Lab2_Results_Analysis.ipynb): Đọc và phân tích sâu kết quả đa biến, ma trận nhầm lẫn và hiện tượng overfitting.
5. **Báo cáo định lượng:**
   - Bảng tổng hợp kết quả định lượng trên tập test (Test Loss, Test Accuracy, Per-class Accuracy, Confusion Matrix).
