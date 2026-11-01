"""
Script to generate the Experiment Monitoring Architecture Diagram for Lab 2.
Generates: outputs/figures/monitoring_architecture.png
Visualizes:
- Training & Validation Engine
- TensorBoardLogger Streams (Scalars, Histograms, Images, Graph)
- Disk Storage Structure (runs/<experiment>/)
- TensorBoard Live Interactive Dashboard
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_monitoring_diagram():
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=240)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.axis("off")

    # Header
    ax.text(0.8, 9.0, "KIẾN TRÚC GIÁM SÁT THỰC NGHIỆM THỜI GIAN THỰC (MONITORING ARCHITECTURE)",
            fontsize=18, fontweight="bold", color="#0D47A1")
    ax.text(0.8, 8.6, "Hệ thống logging đa luồng qua TensorBoard: Giám sát Loss, Accuracy, Phân phối Trọng số, Ảnh Augmentation & Đồ thị Tính toán",
            fontsize=11.5, fontstyle="italic", color="#546E7A")
    ax.axhline(8.4, xmin=0.04, xmax=0.96, color="#CFD8DC", linewidth=1.5)

    # ========================== CỘT 1: NGUỒN PHÁT DỮ LIỆU (TRAINING ENGINE) ==========================
    engine_card = patches.FancyBboxPatch((0.8, 1.2), 3.6, 6.8, boxstyle="round,pad=0.15",
                                         facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=2.0)
    ax.add_patch(engine_card)
    ax.text(2.6, 7.6, "TIẾN TRÌNH HUẤN LUYỆN\n(TRAINING ENGINE)", ha="center", fontsize=12, fontweight="bold", color="#0D47A1")
    ax.text(2.6, 7.15, "src/lab2/train.py", ha="center", fontsize=9.5, fontweight="bold", color="#1565C0")

    sub_blocks = [
        {"title": "1. Khởi tạo & Dữ liệu", "lines": ["• DataLoader (Batch 16/32)", "• Model (ResNet18 / DenseNet)", "• ImageNet Weights V1"], "bg": "#FFFFFF", "border": "#90CAF9"},
        {"title": "2. Forward & Loss", "lines": ["• CrossEntropyLoss", "• Kiểm tra NaN / Inf Loss", "• Log Top-1 Accuracy"], "bg": "#FFFFFF", "border": "#90CAF9"},
        {"title": "3. Backward & Step", "lines": ["• Adam Optimizer", "• Differential LR (1e-4 / 1e-3)", "• Trích xuất Gradients & Weights"], "bg": "#FFFFFF", "border": "#90CAF9"},
        {"title": "4. Validation Loop", "lines": ["• model.eval() + no_grad()", "• val_loss & val_acc", "• Trigger Best Checkpoint"], "bg": "#FFFFFF", "border": "#90CAF9"}
    ]

    for idx, sb in enumerate(sub_blocks):
        by = 5.75 - idx * 1.45
        box = patches.FancyBboxPatch((1.0, by), 3.2, 1.2, boxstyle="round,pad=0.1",
                                     facecolor=sb["bg"], edgecolor=sb["border"], linewidth=1.4)
        ax.add_patch(box)
        ax.text(1.15, by + 0.95, sb["title"], fontsize=9.5, fontweight="bold", color="#0D47A1")
        ly = by + 0.68
        for line in sb["lines"]:
            ax.text(1.15, ly, line, fontsize=8.2, color="#37474F")
            ly -= 0.28

    # Arrow 1: Engine -> TensorBoard Logger
    ax.annotate("", xy=(5.0, 4.6), xytext=(4.5, 4.6),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45", color="#1565C0", lw=2.5))
    ax.text(4.75, 4.85, "Telemetry\nStream", ha="center", fontsize=8.5, fontweight="bold", color="#1565C0")

    # ========================== CỘT 2: MODULE LOGGING (TENSORBOARD LOGGER) ==========================
    logger_card = patches.FancyBboxPatch((5.1, 1.2), 5.4, 6.8, boxstyle="round,pad=0.15",
                                         facecolor="#F3E5F5", edgecolor="#7B1FA2", linewidth=2.0)
    ax.add_patch(logger_card)
    ax.text(7.8, 7.6, "TENSORBOARD LOGGER MODULE", ha="center", fontsize=12, fontweight="bold", color="#4A148C")
    ax.text(7.8, 7.15, "src/lab2/tensorboard_logger.py (SummaryWriter)", ha="center", fontsize=9.5, fontweight="bold", color="#6A1B9A")

    streams = [
        {
            "tag": "1. SCALARS MONITOR (Đường cong độ đo)",
            "desc": "• Loss/train & Loss/val theo từng epoch\n• Accuracy/train & Accuracy/val (%)\n• Thời gian thực thi mỗi epoch (seconds)\n-> Phát hiện Overfitting / Underfitting sớm",
            "bg": "#E8F5E9", "border": "#388E3C", "c_tag": "#1B5E20"
        },
        {
            "tag": "2. HISTOGRAMS MONITOR (Phân phối trọng số)",
            "desc": "• Phân phối Weights & Biases theo từng tầng (conv1, layer1..4, fc)\n• Kiểm tra gradient vanishing / exploding\n• Xác thực requires_grad=False khóa đúng trọng số",
            "bg": "#FFF3E0", "border": "#F57C00", "c_tag": "#E65100"
        },
        {
            "tag": "3. IMAGE BATCH MONITOR (Trực quan hóa ảnh)",
            "desc": "• Ảnh sau tiền xử lý (Resize 128x128)\n• Kiểm chứng RandomCrop (padding reflect) & RandomHorizontalFlip\n• Đảm bảo dữ liệu đầu vào mô hình không bị dị dạng",
            "bg": "#E0F7FA", "border": "#00838F", "c_tag": "#006064"
        },
        {
            "tag": "4. GRAPH MONITOR (Đồ thị kiến trúc tính toán)",
            "desc": "• PyTorch JIT execution graph (torch.jit.trace)\n• Trực quan hóa luồng tensor qua Residual / Dense blocks\n• Kiểm chứng số kênh đầu ra Classifier Head = 10",
            "bg": "#EDE7F6", "border": "#512DA8", "c_tag": "#311B92"
        }
    ]

    for idx, st in enumerate(streams):
        sy = 5.75 - idx * 1.45
        box = patches.FancyBboxPatch((5.3, sy), 5.0, 1.25, boxstyle="round,pad=0.1",
                                     facecolor=st["bg"], edgecolor=st["border"], linewidth=1.5)
        ax.add_patch(box)
        ax.text(5.45, sy + 1.0, st["tag"], fontsize=9.2, fontweight="bold", color=st["c_tag"])
        ax.text(5.45, sy + 0.52, st["desc"], fontsize=8.0, color="#212121", linespacing=1.35)

    # Arrow 2: Logger -> Storage & Dashboard
    ax.annotate("", xy=(11.1, 4.6), xytext=(10.6, 4.6),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.45", color="#7B1FA2", lw=2.5))
    ax.text(10.85, 4.85, "Atomic\nWrite", ha="center", fontsize=8.5, fontweight="bold", color="#7B1FA2")

    # ========================== CỘT 3: LƯU TRỮ VÀ DASHBOARD ==========================
    dash_card = patches.FancyBboxPatch((11.2, 1.2), 4.0, 6.8, boxstyle="round,pad=0.15",
                                       facecolor="#ECEFF1", edgecolor="#455A64", linewidth=2.0)
    ax.add_patch(dash_card)
    ax.text(13.2, 7.6, "LƯU TRỮ & GIAO DIỆN\n(STORAGE & DASHBOARD)", ha="center", fontsize=12, fontweight="bold", color="#263238")
    ax.text(13.2, 7.15, "Local Disk & Web UI (Port 6006)", ha="center", fontsize=9.5, fontweight="bold", color="#37474F")

    # Storage Box
    store_box = patches.FancyBboxPatch((11.4, 4.2), 3.6, 2.7, boxstyle="round,pad=0.1",
                                       facecolor="#FFFFFF", edgecolor="#78909C", linewidth=1.5)
    ax.add_patch(store_box)
    ax.text(11.6, 6.55, "CẤU TRÚC LOG TRÊN ĐĨA:", fontsize=9.5, fontweight="bold", color="#37474F")
    ax.text(11.6, 5.25,
            "runs/\n"
            "├── resnet18_transfer/\n"
            "│   └── events.out.tfevents.*\n"
            "├── resnet18_finetune/\n"
            "├── densenet121_transfer/\n"
            "├── densenet121_finetune/\n"
            "├── resnet18_transfer_lr1e4/\n"
            "└── resnet18_transfer_bs32/",
            fontsize=8.0, fontfamily="monospace", color="#263238", linespacing=1.2)

    # UI Dashboard Box
    ui_box = patches.FancyBboxPatch((11.4, 1.45), 3.6, 2.5, boxstyle="round,pad=0.1",
                                    facecolor="#FFF9C4", edgecolor="#FBC02D", linewidth=1.6)
    ax.add_patch(ui_box)
    ax.text(11.6, 3.65, "TENSORBOARD DASHBOARD", fontsize=9.5, fontweight="bold", color="#F57F17")
    ax.text(11.6, 2.55,
            "• Khởi chạy trực tiếp bằng lệnh:\n"
            "  tensorboard --logdir=runs\n"
            "• Giao diện trực quan trên trình duyệt:\n"
            "  http://localhost:6006\n"
            "• So sánh song song 6 thí nghiệm\n"
            "• Smoothing loss, tải dữ liệu CSV/SVG",
            fontsize=8.2, color="#37474F", linespacing=1.35)

    # Footer note
    ax.text(0.8, 0.65,
            "* Nguyên tắc triển khai: Module TensorBoardLogger tự động đóng luồng an toàn qua khối try...finally logger.close(), "
            "đảm bảo không bị rò rỉ bộ nhớ hay khóa file sự kiện (events file) trên hệ thống.",
            fontsize=9.2, fontstyle="italic", color="#546E7A")

    save_path = os.path.join(FIGURES_DIR, "monitoring_architecture.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Đã lưu sơ đồ giám sát (Monitoring Diagram): {save_path}")


if __name__ == "__main__":
    generate_monitoring_diagram()
