"""
Script to compute and visualize PCA (Principal Component Analysis) on Raw CIFAR-10 data.
Generates: docs/figures/cifar10_raw_pca.png
Clean, publication-ready layout with ZERO text overlap.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
import torchvision.datasets as datasets

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def generate_raw_pca_visualization(data_dir: str = "./data", samples_per_class: int = 300):
    print("Loading CIFAR-10 raw dataset...")
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=False)
    classes = train_set.classes
    targets = np.array(train_set.targets)

    # Collect balanced subset across 10 classes
    selected_indices = []
    for c_idx in range(10):
        c_samples = np.where(targets == c_idx)[0][:samples_per_class]
        selected_indices.extend(c_samples)
    selected_indices = np.array(selected_indices)

    # Flatten raw images to [N, 3072] in range [0, 1]
    X_raw = train_set.data[selected_indices].reshape(len(selected_indices), -1).astype(np.float32) / 255.0
    y_raw = targets[selected_indices]
    num_samples = len(X_raw)

    print(f"Executing PCA on raw space: {num_samples} samples x 3072 dimensions...")
    n_components = 60
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_raw)

    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio) * 100.0

    # 10 distinctive colors for 10 classes
    class_colors = [
        "#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA",
        "#00ACC1", "#6D4C41", "#F4511E", "#3949AB", "#546E7A"
    ]

    fig = plt.figure(figsize=(19, 9.5), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 0.9], height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.25)

    # ==================== SUBPLOT 1: 2D PCA Scatter ====================
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor("#FAFAFA")

    # Plot sample scatter points
    for c_idx in range(10):
        mask = (y_raw == c_idx)
        ax1.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=class_colors[c_idx], label=f"{c_idx}: {classes[c_idx]}",
            alpha=0.45, s=24, edgecolors="none"
        )

    # Calculate centroids and label offsets radially to avoid text collision
    # Pre-calculated clean non-overlapping label coordinates
    offsets = {
        0: (X_pca[y_raw == 0, 0].mean() + 1.8, X_pca[y_raw == 0, 1].mean() + 0.8),    # airplane (right)
        1: (X_pca[y_raw == 1, 0].mean() - 1.2, X_pca[y_raw == 1, 1].mean() - 2.2),    # automobile (bottom)
        2: (X_pca[y_raw == 2, 0].mean() + 2.0, X_pca[y_raw == 2, 1].mean() + 2.0),    # bird (top-right)
        3: (X_pca[y_raw == 3, 0].mean() - 0.5, X_pca[y_raw == 3, 1].mean() + 3.2),    # cat (top)
        4: (X_pca[y_raw == 4, 0].mean() - 3.2, X_pca[y_raw == 4, 1].mean() + 1.8),    # deer (top-left)
        5: (X_pca[y_raw == 5, 0].mean() - 2.8, X_pca[y_raw == 5, 1].mean() + 3.8),    # dog (high-top-left)
        6: (X_pca[y_raw == 6, 0].mean() - 3.5, X_pca[y_raw == 6, 1].mean() - 0.5),    # frog (far-left)
        7: (X_pca[y_raw == 7, 0].mean() + 2.2, X_pca[y_raw == 7, 1].mean() - 0.2),    # horse (center-right)
        8: (X_pca[y_raw == 8, 0].mean() + 2.2, X_pca[y_raw == 8, 1].mean() - 1.8),    # ship (mid-bottom-right)
        9: (X_pca[y_raw == 9, 0].mean() + 1.8, X_pca[y_raw == 9, 1].mean() - 2.8),    # truck (bottom-right)
    }

    for c_idx in range(10):
        mask = (y_raw == c_idx)
        cx = np.mean(X_pca[mask, 0])
        cy = np.mean(X_pca[mask, 1])

        # Centroid badge with index number
        ax1.scatter(cx, cy, color=class_colors[c_idx], s=160, edgecolors="#0D47A1", linewidth=2.0, zorder=6)
        ax1.text(cx, cy, str(c_idx), color="#FFFFFF", fontweight="bold", fontsize=8.5, ha="center", va="center", zorder=7)

        # Leader line with annotation box at non-overlapping location
        lx, ly = offsets[c_idx]
        ax1.annotate(
            f"[{c_idx}] {classes[c_idx]}",
            xy=(cx, cy), xytext=(lx, ly),
            arrowprops=dict(arrowstyle="->", color=class_colors[c_idx], lw=1.5, shrinkA=4, shrinkB=6),
            fontsize=9.2, fontweight="bold", color="#212121",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=class_colors[c_idx], linewidth=1.5, alpha=0.95),
            zorder=8
        )

    pc1_pct = var_ratio[0] * 100
    pc2_pct = var_ratio[1] * 100
    ax1.set_title(
        f"(A) Không gian 2D PCA trên Pixel Thô CIFAR-10 (3.072 chiều -> 2 chiều)\n"
        f"Hiện tượng chồng lấn dữ dội: Không thể phân tách tuyến tính 10 lớp",
        fontsize=12.5, fontweight="bold", color="#0D47A1", pad=12
    )
    ax1.set_xlabel(f"Thành phần chính 1 (PC1) - Giải thích {pc1_pct:.1f}% phương sai", fontsize=11, fontweight="bold", color="#37474F")
    ax1.set_ylabel(f"Thành phần chính 2 (PC2) - Giải thích {pc2_pct:.1f}% phương sai", fontsize=11, fontweight="bold", color="#37474F")
    ax1.set_xlim(-15, 18)
    ax1.set_ylim(-13, 14)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(loc="lower left", ncol=2, fontsize=8.5, framealpha=0.92, title="10 Lớp CIFAR-10", title_fontsize=9)

    # ==================== SUBPLOT 2: Cumulative Variance ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FAFAFA")
    comps = np.arange(1, n_components + 1)
    ax2.plot(comps, cum_var, color="#1976D2", lw=2.5, marker="o", markersize=3, label="Phương sai tích lũy (%)")
    ax2.axhline(50, color="#E65100", linestyle="--", lw=1.5, label="Ngưỡng 50% phương sai")
    ax2.axhline(80, color="#388E3C", linestyle="--", lw=1.5, label="Ngưỡng 80% phương sai")

    # Info card for thresholds (avoids colliding with curve)
    idx_50 = np.where(cum_var >= 50.0)[0]
    n_50 = comps[idx_50[0]] if len(idx_50) > 0 else "N/A"
    idx_80 = np.where(cum_var >= 80.0)[0]
    n_80 = comps[idx_80[0]] if len(idx_80) > 0 else ">60"

    info_card = patches.FancyBboxPatch((28, 12), 29, 32, boxstyle="round,pad=0.2",
                                       facecolor="#FFFFFF", edgecolor="#B0BEC5", linewidth=1.2)
    ax2.add_patch(info_card)
    ax2.text(30, 36, "ĐẶC TÍNH PHƯƠNG SAI:", fontsize=9, fontweight="bold", color="#0D47A1")
    ax2.text(30, 28, f"• Cần {n_50} thành phần -> 50%", fontsize=8.8, color="#E65100", fontweight="bold")
    ax2.text(30, 20, f"• Cần {n_80} thành phần -> 80%", fontsize=8.8, color="#388E3C", fontweight="bold")
    ax2.text(30, 14, f"• Top 2 thành phần: {cum_var[1]:.1f}%", fontsize=8.5, color="#546E7A")

    ax2.set_title("(B) Tỷ lệ Phương sai Tích lũy theo Số Thành phần Chính", fontsize=11.5, fontweight="bold", color="#0D47A1", pad=10)
    ax2.set_xlabel("Số lượng thành phần chính (Principal Components)", fontsize=9.5, fontweight="bold", color="#37474F")
    ax2.set_ylabel("Phương sai tích lũy (%)", fontsize=9.5, fontweight="bold", color="#37474F")
    ax2.set_ylim(0, 105)
    ax2.set_xlim(1, n_components)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # ==================== SUBPLOT 3: Top-4 Eigen-Images ====================
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.set_title(
        "(C) 4 Thành phần Chính Đầu tiên (Eigen-Images 32x32)\n"
        "Chỉ nắm bắt độ sáng nền và viền màu thô, không nắm bắt ngữ nghĩa đối tượng",
        fontsize=11.5, fontweight="bold", color="#0D47A1", pad=12
    )

    eigen_vectors = pca.components_[:4]
    sub_axes = [ax3.inset_axes([0.02 + i * 0.25, 0.08, 0.21, 0.75]) for i in range(4)]

    for i in range(4):
        e_img = eigen_vectors[i].reshape(32, 32, 3)
        e_min, e_max = e_img.min(), e_img.max()
        e_norm = (e_img - e_min) / (e_max - e_min + 1e-8)
        sub_axes[i].imshow(e_norm)
        sub_axes[i].set_title(f"PC #{i+1}\n({var_ratio[i]*100:.1f}%)", fontsize=9.5, fontweight="bold", color="#1565C0")
        sub_axes[i].set_xticks([])
        sub_axes[i].set_yticks([])
        for spine in sub_axes[i].spines.values():
            spine.set_edgecolor("#1976D2")
            spine.set_linewidth(1.5)

    fig.suptitle(
        "PHÂN TÍCH THÀNH PHẦN CHÍNH (PCA) TRÊN DỮ LIỆU THÔ CIFAR-10\n"
        "Minh chứng thực nghiệm: Không gian pixel thô không phân tách được ngữ nghĩa -> Cần Pretrained CNN Backbone",
        fontsize=13.5, fontweight="bold", color="#0D47A1", y=0.98
    )

    save_path = os.path.join(FIGURES_DIR, "cifar10_raw_pca.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ PCA dữ liệu thô (đã khử dính chữ): {save_path}")


if __name__ == "__main__":
    generate_raw_pca_visualization()
