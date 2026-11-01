"""
Script tạo các hình ảnh minh họa kỹ thuật (Diagrams) cho tài liệu và Notebook bài Lab 2.
Tuân thủ nguyên tắc:
- Không tạo dữ liệu/số liệu giả định.
- Số liệu tham số khớp 100% với hàm count_parameters trong src/lab2/models.py.
- Sơ đồ khái niệm ghi rõ tính chất mô tả cấu trúc, không phải kết quả thực nghiệm.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def draw_pipeline_workflow():
    """Vẽ sơ đồ quy trình thực nghiệm End-to-End của Lab 2."""
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=300)
    ax.axis("off")

    stages = [
        {"title": "1. Dữ liệu CIFAR-10", "desc": "60.000 ảnh màu (32x32)\n45k Train | 5k Val | 10k Test\nSeed tái lập = 42", "color": "#E3F2FD", "border": "#1976D2"},
        {"title": "2. Tiền xử lý", "desc": "Resize 128x128\nImageNet Normalization\nRandomCrop & Flip (Train)", "color": "#EDE7F6", "border": "#512DA8"},
        {"title": "3. Pretrained Model", "desc": "ResNet18 (11.2M params)\nDenseNet121 (7.0M params)\nImageNet Weights chuẩn", "color": "#E8F5E9", "border": "#388E3C"},
        {"title": "4. Thích nghi mô hình", "desc": "Transfer Learning (Freeze)\nFine-Tuning (Unfreeze deep)\nDifferential Learning Rates", "color": "#FFF3E0", "border": "#F57C00"},
        {"title": "5. Huấn luyện & Đánh giá", "desc": "Adam Optimizer\nCrossEntropyLoss\nVal Acc Best Checkpoint\nTensorBoard Logging", "color": "#FCE4EC", "border": "#C2185B"},
        {"title": "6. Báo cáo độc lập", "desc": "Test Accuracy & Loss\nPer-class Accuracy\nConfusion Matrix Heatmap", "color": "#E0F2F1", "border": "#00796B"}
    ]

    n = len(stages)
    box_w = 1.8
    box_h = 2.4
    spacing = 0.5
    start_x = 0.5
    y = 1.0

    for i, s in enumerate(stages):
        x = start_x + i * (box_w + spacing)
        rect = patches.FancyBboxPatch((x, y), box_w, box_h, boxstyle="round,pad=0.12",
                                      facecolor=s["color"], edgecolor=s["border"], linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, y + box_h - 0.45, s["title"], ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=s["border"])
        ax.text(x + box_w / 2, y + box_h / 2 - 0.3, s["desc"], ha="center", va="center",
                fontsize=8, color="#212121", linespacing=1.35)

        if i < n - 1:
            arrow_start = x + box_w + 0.08
            arrow_end = arrow_start + spacing - 0.16
            ax.annotate("", xy=(arrow_end, y + box_h / 2), xytext=(arrow_start, y + box_h / 2),
                        arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45",
                                        color="#455A64", lw=2))

    ax.set_xlim(0, start_x + n * box_w + (n - 1) * spacing + 0.5)
    ax.set_ylim(0.5, 4.0)
    plt.title("QUY TRÌNH THỰC NGHIỆM TỔNG THỂ (LAB 2 WORKFLOW PIPELINE)",
              fontsize=12, fontweight="bold", pad=15, color="#263238")
    save_path = os.path.join(FIGURES_DIR, "pipeline_workflow.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu sơ đồ quy trình: {save_path}")


def draw_transfer_vs_finetune():
    """Vẽ sơ đồ so sánh khái niệm Transfer Learning vs Fine-Tuning."""
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=300)
    ax.axis("off")

    # Panel 1: Transfer Learning
    ax.text(0.5, 4.8, "Chiến lược 1: Transfer Learning (Feature Extraction)", fontsize=11, fontweight="bold", color="#1565C0")
    # Backbone
    rect_bb1 = patches.FancyBboxPatch((0.5, 3.2), 6.5, 1.3, boxstyle="round,pad=0.08",
                                      facecolor="#BBDEFB", edgecolor="#1976D2", linewidth=2)
    ax.add_patch(rect_bb1)
    ax.text(3.75, 4.0, "Pretrained Backbone (ResNet18 / DenseNet121)", ha="center", va="center", fontsize=10, fontweight="bold", color="#0D47A1")
    ax.text(3.75, 3.55, "ĐÓNG BĂNG HOÀN TOÀN (requires_grad = False)\nTrọng số trích xuất đặc trưng ImageNet được giữ nguyên 100%", ha="center", va="center", fontsize=8.5, color="#1A237E")

    # Arrow 1
    ax.annotate("", xy=(7.8, 3.85), xytext=(7.1, 3.85),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color="#37474F", lw=2))

    # Head 1
    rect_h1 = patches.FancyBboxPatch((7.9, 3.2), 4.6, 1.3, boxstyle="round,pad=0.08",
                                     facecolor="#C8E6C9", edgecolor="#388E3C", linewidth=2)
    ax.add_patch(rect_h1)
    ax.text(10.2, 4.0, "New Classifier Head (10 lớp)", ha="center", va="center", fontsize=10, fontweight="bold", color="#1B5E20")
    ax.text(10.2, 3.55, "HUẤN LUYỆN (requires_grad = True)\nlr = 1e-3 | ResNet: 5.130 tham số (0.05%)", ha="center", va="center", fontsize=8.5, color="#1B5E20")

    # Panel 2: Fine-Tuning
    ax.text(0.5, 2.4, "Chiến lược 2: Fine-Tuning (Tinh chỉnh tham số tầng sâu)", fontsize=11, fontweight="bold", color="#E65100")
    # Early Backbone
    rect_bb2 = patches.FancyBboxPatch((0.5, 0.8), 3.8, 1.3, boxstyle="round,pad=0.08",
                                      facecolor="#BBDEFB", edgecolor="#1976D2", linewidth=2)
    ax.add_patch(rect_bb2)
    ax.text(2.4, 1.6, "Backbone Tầng Nông\n(conv1..layer3 / denseblock1..3)", ha="center", va="center", fontsize=9, fontweight="bold", color="#0D47A1")
    ax.text(2.4, 1.15, "ĐÓNG BĂNG (requires_grad = False)\nGiữ đặc trưng cạnh/biên chung", ha="center", va="center", fontsize=8, color="#1A237E")

    # Arrow 2a
    ax.annotate("", xy=(4.9, 1.45), xytext=(4.4, 1.45),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color="#37474F", lw=2))

    # Deep Backbone
    rect_deep = patches.FancyBboxPatch((5.0, 0.8), 3.6, 1.3, boxstyle="round,pad=0.08",
                                       facecolor="#FFE082", edgecolor="#FFA000", linewidth=2)
    ax.add_patch(rect_deep)
    ax.text(6.8, 1.6, "Tầng Sâu (layer4 / denseblock4)", ha="center", va="center", fontsize=9, fontweight="bold", color="#E65100")
    ax.text(6.8, 1.15, "MỞ KHÓA (requires_grad = True)\nlr_backbone = 1e-4 (Differential LR)", ha="center", va="center", fontsize=8, color="#E65100")

    # Arrow 2b
    ax.annotate("", xy=(9.2, 1.45), xytext=(8.7, 1.45),
                arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.45", color="#37474F", lw=2))

    # Head 2
    rect_h2 = patches.FancyBboxPatch((9.3, 0.8), 3.2, 1.3, boxstyle="round,pad=0.08",
                                     facecolor="#C8E6C9", edgecolor="#388E3C", linewidth=2)
    ax.add_patch(rect_h2)
    ax.text(10.9, 1.6, "New Classifier Head", ha="center", va="center", fontsize=9, fontweight="bold", color="#1B5E20")
    ax.text(10.9, 1.15, "HUẤN LUYỆN (requires_grad=True)\nlr_head = 1e-3 (10x lr_backbone)", ha="center", va="center", fontsize=8, color="#1B5E20")

    # Caption note
    ax.text(6.5, 0.2, "* Sơ đồ khái niệm kiến trúc: Minh họa cơ chế phân tách và điều chỉnh tham số giữa 2 chiến lược thích nghi.",
            ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#546E7A")

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.3)
    plt.title("SO SÁNH CƠ CHẾ: TRANSFER LEARNING VS FINE-TUNING",
              fontsize=12, fontweight="bold", pad=12, color="#263238")
    save_path = os.path.join(FIGURES_DIR, "transfer_vs_finetune.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu sơ đồ Transfer vs Fine-tune: {save_path}")


def draw_parameter_distribution():
    """Vẽ biểu đồ phân phối tham số Trainable vs Frozen dựa trên số đo thực tế từ inspect_models.py."""
    configs = [
        "ResNet18\nTransfer",
        "ResNet18\nFine-tune",
        "DenseNet121\nTransfer",
        "DenseNet121\nFine-tune"
    ]
    trainable = [5130, 8398858, 10250, 2170378]
    frozen = [11176512, 2782784, 6953856, 4793728]
    totals = [t + f for t, f in zip(trainable, frozen)]
    pcts = [(t / tot) * 100 for t, tot in zip(trainable, totals)]

    x = np.arange(len(configs))
    width = 0.55

    fig, ax = plt.subplots(figsize=(9.5, 5), dpi=300)

    # Stacked bar chart
    p1 = ax.bar(x, [f / 1e6 for f in frozen], width, label="Frozen Parameters (Đóng băng)",
                color="#90CAF9", edgecolor="#1565C0", linewidth=1.2)
    p2 = ax.bar(x, [t / 1e6 for t in trainable], width, bottom=[f / 1e6 for f in frozen],
                label="Trainable Parameters (Huấn luyện)", color="#FF8A65", edgecolor="#D84315", linewidth=1.2)

    ax.set_ylabel("Số lượng tham số (Triệu - Millions)", fontsize=10, fontweight="bold")
    ax.set_title("PHÂN PHỐI THAM SỐ TRAINABLE VS FROZEN (ĐO TỪ SOURCE-FIRST)",
                 fontsize=11, fontweight="bold", pad=15, color="#263238")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Add percentages and counts text
    for i in range(len(configs)):
        total_m = totals[i] / 1e6
        ax.text(x[i], total_m + 0.25, f"{pcts[i]:.2f}% Trainable\n({trainable[i]:,} / {totals[i]:,})",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#263238")

    ax.set_ylim(0, 13.5)
    save_path = os.path.join(FIGURES_DIR, "model_parameters_comparison.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"-> Đã lưu biểu đồ tham số: {save_path}")


if __name__ == "__main__":
    draw_pipeline_workflow()
    draw_transfer_vs_finetune()
    draw_parameter_distribution()
    print("Hoàn thành tạo toàn bộ diagrams cho dự án!")
