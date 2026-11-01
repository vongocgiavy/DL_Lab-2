"""
Script tạo các hình ảnh slide thuyết trình chuẩn 16:9 chất lượng cao (300 DPI) cho bài Lab 2.
Phong cách thiết kế:
- Nền trắng tinh gọn, hiện đại
- Khối tensor 3D màu xanh dương đậm/nhạt với chú thích kích thước (H x W x C)
- Hộp thao tác viền xanh
- Tầng đóng băng (Frozen) tô xanh nhạt kèm ký hiệu khóa [LOCKED]
- Tầng huấn luyện (Trainable) tô màu cam kèm ký hiệu [TRAINABLE]
- Mũi tên xanh điều hướng luồng dữ liệu
- Số liệu thực nghiệm chính xác 100% từ Lab 2
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "mpl_cache")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIDES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures", "slides")
os.makedirs(SLIDES_DIR, exist_ok=True)


def draw_isometric_cube(ax, x, y, dx, dy, dz, color_top="#64B5F6", color_front="#1976D2", color_side="#0D47A1", alpha=0.95):
    """Vẽ khối tensor 3D isometric chuẩn với 3 mặt (trên, trước, bên)."""
    # Chiếu isometric:
    # Mặt trước (Front face)
    p_front = [
        (x, y),
        (x + dx, y),
        (x + dx, y + dy),
        (x, y + dy)
    ]
    poly_front = Polygon(p_front, closed=True, facecolor=color_front, edgecolor="#0D47A1", linewidth=1.2, alpha=alpha)
    ax.add_patch(poly_front)

    # Mặt trên (Top face)
    p_top = [
        (x, y + dy),
        (x + dx, y + dy),
        (x + dx + dz * 0.5, y + dy + dz * 0.35),
        (x + dz * 0.5, y + dy + dz * 0.35)
    ]
    poly_top = Polygon(p_top, closed=True, facecolor=color_top, edgecolor="#0D47A1", linewidth=1.2, alpha=alpha)
    ax.add_patch(poly_top)

    # Mặt bên phải (Side face)
    p_side = [
        (x + dx, y),
        (x + dx + dz * 0.5, y + dz * 0.35),
        (x + dx + dz * 0.5, y + dy + dz * 0.35),
        (x + dx, y + dy)
    ]
    poly_side = Polygon(p_side, closed=True, facecolor=color_side, edgecolor="#0D47A1", linewidth=1.2, alpha=alpha)
    ax.add_patch(poly_side)


def draw_lock_icon(ax, x, y, scale=0.15, color="#1565C0"):
    """Vẽ biểu tượng ổ khóa vector sắc nét."""
    body_w = 0.4 * scale
    body_h = 0.35 * scale
    body = patches.Rectangle((x - body_w / 2, y - body_h / 2), body_w, body_h,
                             facecolor=color, edgecolor="#0D47A1", linewidth=1, zorder=10)
    ax.add_patch(body)
    shackle = patches.Arc((x, y + body_h / 2), body_w * 0.8, body_h * 1.2,
                          angle=0, theta1=0, theta2=180, edgecolor=color, linewidth=2.5 * scale, zorder=9)
    ax.add_patch(shackle)


def generate_slide_01():
    """Slide 1: Pretrained Model và ImageNet-1K."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.axis("off")

    # Header
    ax.text(0.8, 8.4, "BƯỚC 1: PRETRAINED MODEL & BỘ TRỌNG SỐ IMAGENET-1K",
            fontsize=20, fontweight="bold", color="#0D47A1")
    ax.text(0.8, 8.0, "Kế thừa biểu diễn đặc trưng thị giác từ 1.28 triệu ảnh sang bài toán CIFAR-10",
            fontsize=12, fontstyle="italic", color="#546E7A")
    ax.axhline(7.8, xmin=0.05, xmax=0.95, color="#E0E0E0", linewidth=1.5)

    # ========================== PHẦN 1: IMAGENET PRETRAINING ==========================
    # Box bao quanh Phần 1
    panel1 = patches.FancyBboxPatch((0.8, 4.4), 14.4, 3.1, boxstyle="round,pad=0.2",
                                    facecolor="#F8FAFC", edgecolor="#B0BEC5", linewidth=1.5)
    ax.add_patch(panel1)
    ax.text(1.2, 7.15, "[1] TIỀN HUẤN LUYỆN NGUỒN: IMAGENET-1K (Pre-training Phase)",
            fontsize=12, fontweight="bold", color="#37474F")

    # Tensor Input ImageNet (224x224x3)
    draw_isometric_cube(ax, 1.5, 5.0, 1.1, 1.1, 0.4, color_top="#90CAF9", color_front="#2196F3", color_side="#1565C0")
    ax.text(2.05, 4.6, "Ảnh ImageNet\n224 x 224 x 3", ha="center", fontsize=9, fontweight="bold", color="#0D47A1")

    # Arrow 1
    ax.annotate("", xy=(3.8, 5.7), xytext=(3.1, 5.7),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#1976D2", lw=2.5))

    # Backbone ImageNet
    bb_box1 = patches.FancyBboxPatch((3.9, 4.8), 5.6, 1.8, boxstyle="round,pad=0.15",
                                     facecolor="#E3F2FD", edgecolor="#1976D2", linewidth=2)
    ax.add_patch(bb_box1)
    ax.text(6.7, 6.15, "CONVOLUTIONAL BACKBONE", ha="center", fontsize=11, fontweight="bold", color="#0D47A1")
    ax.text(6.7, 5.7, "ResNet-18 (conv1 -> layer4)  |  DenseNet-121 (features)", ha="center", fontsize=9.5, fontweight="bold", color="#1565C0")
    ax.text(6.7, 5.25, "Học 1.281.167 ảnh qua hàng triệu tham số trích xuất đặc trưng\n(Edges -> Textures -> Parts -> Semantic Objects)",
            ha="center", fontsize=8.5, color="#37474F")

    # Arrow 2
    ax.annotate("", xy=(10.5, 5.7), xytext=(9.8, 5.7),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#1976D2", lw=2.5))

    # Original Head (1000 classes)
    head_box1 = patches.FancyBboxPatch((10.6, 4.8), 4.2, 1.8, boxstyle="round,pad=0.15",
                                       facecolor="#ECEFF1", edgecolor="#78909C", linewidth=1.8)
    ax.add_patch(head_box1)
    ax.text(12.7, 6.1, "Head Gốc ImageNet", ha="center", fontsize=10.5, fontweight="bold", color="#455A64")
    ax.text(12.7, 5.6, "Linear(512/1024 -> 1000)", ha="center", fontsize=9, color="#455A64")
    ax.text(12.7, 5.15, "[LOẠI BỎ KHI SANG LAB 2]\n1.000 xác suất lớp ImageNet", ha="center", fontsize=8.5, color="#D32F2F", fontweight="bold")

    # ========================== PHẦN 2: CHUYỂN TIẾP SANG CIFAR-10 ==========================
    panel2 = patches.FancyBboxPatch((0.8, 0.9), 14.4, 3.2, boxstyle="round,pad=0.2",
                                    facecolor="#FFFFFF", edgecolor="#1976D2", linewidth=2.0)
    ax.add_patch(panel2)
    ax.text(1.2, 3.75, "[2] THÍCH NGHI SANG TÁC VỤ ĐÍCH: CIFAR-10 (Downstream Adaptation)",
            fontsize=12, fontweight="bold", color="#0D47A1")

    # Input Tensor CIFAR-10 (128x128x3 sau khi resize từ 32x32x3)
    draw_isometric_cube(ax, 1.5, 1.5, 0.9, 0.9, 0.35, color_top="#90CAF9", color_front="#1976D2", color_side="#0D47A1")
    ax.text(1.95, 1.15, "CIFAR-10 (Resize)\n128 x 128 x 3", ha="center", fontsize=9, fontweight="bold", color="#0D47A1")

    # Arrow 3
    ax.annotate("", xy=(3.6, 2.2), xytext=(2.9, 2.2),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#1976D2", lw=2.5))

    # Frozen/Tuned Backbone CIFAR-10
    bb_box2 = patches.FancyBboxPatch((3.7, 1.3), 5.7, 2.0, boxstyle="round,pad=0.15",
                                     facecolor="#E1F5FE", edgecolor="#0288D1", linewidth=2)
    ax.add_patch(bb_box2)
    ax.text(6.55, 2.9, "PRETRAINED BACKBONE (GIỮ NGUYÊN)", ha="center", fontsize=10.5, fontweight="bold", color="#01579B")
    draw_lock_icon(ax, 4.2, 2.88, scale=1.3, color="#0277BD")
    ax.text(6.55, 2.45, "Đóng băng (requires_grad = False) trong Transfer Learning\nHoặc mở tầng sâu trong Fine-Tuning",
            ha="center", fontsize=8.8, color="#0277BD")
    ax.text(6.55, 1.6, "ResNet-18 Backbone: 11.176.512 tham số\nDenseNet-121 Backbone: 6.953.856 tham số",
            ha="center", fontsize=8.5, fontweight="bold", color="#37474F")

    # Arrow 4 with Feature Vector dimension
    ax.annotate("", xy=(10.3, 2.2), xytext=(9.6, 2.2),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.4", color="#1976D2", lw=2.5))
    ax.text(9.95, 2.5, "Vector đặc trưng\n512 / 1024", ha="center", fontsize=8, fontweight="bold", color="#0D47A1")

    # Trainable Head CIFAR-10 (10 classes)
    head_box2 = patches.FancyBboxPatch((10.4, 1.3), 4.4, 2.0, boxstyle="round,pad=0.15",
                                       facecolor="#FFF3E0", edgecolor="#F57C00", linewidth=2.2)
    ax.add_patch(head_box2)
    ax.text(12.6, 2.9, "TẦNG PHÂN LOẠI MỚI (HEAD)", ha="center", fontsize=10.5, fontweight="bold", color="#E65100")
    ax.text(12.6, 2.45, "[TRAINABLE] (requires_grad = True)", ha="center", fontsize=9, fontweight="bold", color="#E65100")
    ax.text(12.6, 1.95, "ResNet: Linear(512 -> 10)  =  5.130 params\nDenseNet: Linear(1024 -> 10) = 10.250 params",
            ha="center", fontsize=8.5, color="#BF360C")
    ax.text(12.6, 1.5, "=> 10 Logits xác suất lớp CIFAR-10", ha="center", fontsize=8.5, fontweight="bold", color="#1B5E20")

    # Footer note
    ax.text(0.8, 0.4, "* Điểm mấu chốt: Tận dụng biểu diễn không gian mạnh mẽ từ ImageNet-1K giúp mô hình hội tụ nhanh chỉ sau 5 epochs trên tập CIFAR-10.",
            fontsize=9.5, fontstyle="italic", color="#455A64")

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    plt.tight_layout()
    save_path = os.path.join(SLIDES_DIR, "slide_01_pretrained_imagenet.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"-> Đã tạo Slide 1: {save_path}")


def generate_slide_02():
    """Slide 2: Quy trình thực nghiệm tổng thể (Pipeline 6 bước)."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.axis("off")

    # Header
    ax.text(0.8, 8.4, "BƯỚC 2: QUY TRÌNH THỰC NGHIỆM TỔNG THỂ (PIPELINE 6 BƯỚC)",
            fontsize=20, fontweight="bold", color="#0D47A1")
    ax.text(0.8, 8.0, "Quy trình khép kín, chuẩn mực khoa học, kiểm soát chặt chẽ data leakage và tính tái lập",
            fontsize=12, fontstyle="italic", color="#546E7A")
    ax.axhline(7.8, xmin=0.05, xmax=0.95, color="#E0E0E0", linewidth=1.5)

    stages = [
        {
            "step": "BƯỚC 1",
            "title": "Dữ liệu CIFAR-10",
            "tensor": "32 x 32 x 3",
            "lines": [
                "• 60.000 ảnh màu RGB",
                "• 10 lớp cân bằng",
                "• Split (seed=42):",
                "  - Train: 45.000 ảnh",
                "  - Val: 5.000 ảnh",
                "  - Test: 10.000 ảnh",
                "• Test set độc lập 100%"
            ],
            "bg": "#E3F2FD",
            "border": "#1976D2"
        },
        {
            "step": "BƯỚC 2",
            "title": "Tiền xử lý & Aug",
            "tensor": "128 x 128 x 3",
            "lines": [
                "• Resize: 128x128",
                "• ImageNet Normalize:",
                "  mean=[0.485,0.456,0.406]",
                "  std=[0.229,0.224,0.225]",
                "• Train: RandomCrop",
                "  & RandomHorizontalFlip",
                "• Val/Test: Deterministic"
            ],
            "bg": "#EDE7F6",
            "border": "#512DA8"
        },
        {
            "step": "BƯỚC 3",
            "title": "Mô hình Pretrained",
            "tensor": "11.18M / 6.96M",
            "lines": [
                "• ResNet-18 ImageNet",
                "  Total: 11.181.642",
                "• DenseNet-121 ImageNet",
                "  Total: 6.964.106",
                "• Thay Head phân loại:",
                "  Linear(512/1024, 10)",
                "• Chuẩn bị trọng số V1"
            ],
            "bg": "#E8F5E9",
            "border": "#2E7D32"
        },
        {
            "step": "BƯỚC 4",
            "title": "Chiến lược thích nghi",
            "tensor": "TL vs FT",
            "lines": [
                "• Transfer Learning:",
                "  Khóa toàn bộ backbone",
                "  Chỉ học Head (lr 1e-3)",
                "• Fine-Tuning:",
                "  Mở tầng sâu (layer4 /",
                "  denseblock4 + norm5)",
                "• Differential LR: 1e-4 / 1e-3"
            ],
            "bg": "#FFF3E0",
            "border": "#EF6C00"
        },
        {
            "step": "BƯỚC 5",
            "title": "Vòng lặp huấn luyện",
            "tensor": "5 Epochs",
            "lines": [
                "• Optimizer: Adam",
                "  weight decay = 1e-4",
                "• Loss: CrossEntropyLoss",
                "• Batch size: 16 (và 32)",
                "• model.train() / eval()",
                "• Chọn Best Checkpoint",
                "  theo val_acc cao nhất"
            ],
            "bg": "#FCE4EC",
            "border": "#C2185B"
        },
        {
            "step": "BƯỚC 6",
            "title": "Đánh giá độc lập",
            "tensor": "10.000 Test Set",
            "lines": [
                "• Load Best Checkpoint",
                "• Đánh giá 1 lần duy nhất",
                "  trên 10.000 ảnh test",
                "• Test Accuracy & Loss",
                "• Per-class Accuracy",
                "• Confusion Matrix",
                "• Không data leakage"
            ],
            "bg": "#E0F2F1",
            "border": "#00796B"
        }
    ]

    card_w = 2.22
    card_h = 4.8
    start_x = 0.8
    spacing = 0.23
    y_card = 2.7

    for i, st in enumerate(stages):
        x = start_x + i * (card_w + spacing)
        
        # Thẻ bước
        card = patches.FancyBboxPatch((x, y_card), card_w, card_h, boxstyle="round,pad=0.1",
                                      facecolor=st["bg"], edgecolor=st["border"], linewidth=2.0)
        ax.add_patch(card)

        # Step badge
        badge = patches.FancyBboxPatch((x + 0.35, y_card + card_h - 0.45), card_w - 0.7, 0.36,
                                       boxstyle="round,pad=0.05", facecolor=st["border"], edgecolor="none")
        ax.add_patch(badge)
        ax.text(x + card_w / 2, y_card + card_h - 0.27, st["step"], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#FFFFFF")

        # Title
        ax.text(x + card_w / 2, y_card + card_h - 0.75, st["title"], ha="center", va="center",
                fontsize=11, fontweight="bold", color=st["border"])

        # Tensor Shape Tag
        tag = patches.FancyBboxPatch((x + 0.2, y_card + card_h - 1.25), card_w - 0.4, 0.35,
                                     boxstyle="round,pad=0.04", facecolor="#FFFFFF", edgecolor=st["border"], linewidth=1.2)
        ax.add_patch(tag)
        ax.text(x + card_w / 2, y_card + card_h - 1.07, f"Tensor: {st['tensor']}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#0D47A1")

        # Detail lines
        line_y = y_card + card_h - 1.6
        for line in st["lines"]:
            is_bold = "•" in line and ("Transfer" in line or "Fine-Tuning" in line or "Split" in line or "ResNet" in line)
            ax.text(x + 0.15, line_y, line, fontsize=8.2, color="#212121",
                    fontweight="bold" if is_bold else "normal")
            line_y -= 0.35

        # Arrow between cards
        if i < len(stages) - 1:
            arrow_x = x + card_w + 0.04
            ax.annotate("", xy=(arrow_x + spacing - 0.08, y_card + card_h / 2),
                        xytext=(arrow_x, y_card + card_h / 2),
                        arrowprops=dict(arrowstyle="->,head_width=0.32,head_length=0.4",
                                        color="#1565C0", lw=2.2))

    # ========================== KẾT QUẢ THỰC NGHIỆM LAB 2 ==========================
    res_panel = patches.FancyBboxPatch((0.8, 0.65), 14.4, 1.8, boxstyle="round,pad=0.15",
                                       facecolor="#F8FAFC", edgecolor="#1976D2", linewidth=1.8)
    ax.add_patch(res_panel)
    ax.text(1.1, 2.15, "TỔNG KẾT KẾT QUẢ TEST ACCURACY TRÊN 6 THÍ NGHIỆM ĐÃ THỰC HIỆN:",
            fontsize=11, fontweight="bold", color="#0D47A1")

    metrics_text = (
        "1. ResNet-18 Transfer:   77.98%  (5.130 trainable params | lr=1e-3, bs=16)\n"
        "2. ResNet-18 Fine-tune:  90.24%  (8.398.858 trainable params | lr_bb=1e-4, lr_head=1e-3)\n"
        "3. DenseNet-121 Transfer: 81.34%  (10.250 trainable params | lr=1e-3, bs=16)\n"
        "4. DenseNet-121 Fine-tune: 90.65%  (2.170.378 trainable params | lr_bb=1e-4, lr_head=1e-3)"
    )
    hparam_text = (
        "5. ResNet-18 LR 1e-4:    78.32%  (Thử nghiệm độ nhạy Learning Rate)\n"
        "6. ResNet-18 Batch 32:   79.04%  (Thử nghiệm tăng Batch Size từ 16 lên 32)\n"
        "* Ràng buộc khoa học: Đánh giá duy nhất trên seed=42; Test set 10k ảnh hoàn toàn cách ly;\n"
        "  Chưa đủ dữ liệu đa-seed để kết luận bản chất DenseNet vượt trội ResNet."
    )
    ax.text(1.2, 1.75, metrics_text, fontsize=9.2, color="#0D47A1", fontweight="bold", linespacing=1.35)
    ax.text(8.3, 1.75, hparam_text, fontsize=8.8, color="#37474F", linespacing=1.35)

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    plt.tight_layout()
    save_path = os.path.join(SLIDES_DIR, "slide_02_pipeline_workflow.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"-> Đã tạo Slide 2: {save_path}")


if __name__ == "__main__":
    generate_slide_01()
    generate_slide_02()
