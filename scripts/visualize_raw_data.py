"""
Script to visualize raw CIFAR-10 data:
1. cifar10_raw_samples_grid.png: 10 classes x 5 samples grid of raw 32x32 images.
2. cifar10_raw_distribution.png: Class balance and RGB channel distributions.
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
import torchvision.datasets as datasets
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def visualize_raw_samples_grid(data_dir: str = "./data", samples_per_class: int = 6):
    """Generates a clean publication-quality grid of raw CIFAR-10 images (32x32)."""
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=False)
    classes = dataset.classes  # ['airplane', 'automobile', 'bird', ...]
    
    # Collect indices for each class
    targets = np.array(dataset.targets)
    class_indices = {i: np.where(targets == i)[0][:samples_per_class] for i in range(10)}

    fig, axes = plt.subplots(10, samples_per_class, figsize=(samples_per_class * 2.2, 10 * 1.6), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    
    plt.subplots_adjust(top=0.94, bottom=0.03, left=0.12, right=0.97, hspace=0.35, wspace=0.15)
    
    fig.suptitle(
        "TRỰC QUAN HÓA DỮ LIỆU THÔ CIFAR-10 (RAW 32x32 RGB IMAGES)\n"
        "60.000 ảnh màu RGB (32 x 32 x 3) • 10 lớp cân bằng (6.000 ảnh/lớp) • Định dạng uint8 [0 - 255]",
        fontsize=14, fontweight="bold", color="#0D47A1", y=0.98
    )

    for c_idx, class_name in enumerate(classes):
        indices = class_indices[c_idx]
        for s_idx, img_idx in enumerate(indices):
            ax = axes[c_idx, s_idx]
            img, label = dataset[img_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Border styling
            for spine in ax.spines.values():
                spine.set_edgecolor("#90CAF9")
                spine.set_linewidth(1.5)
            
            if c_idx == 0:
                ax.set_title(f"Mẫu #{s_idx + 1}", fontsize=10, fontweight="bold", color="#1565C0", pad=5)
            
            if s_idx == 0:
                ax.set_ylabel(
                    f"{class_name.upper()}\n(Lớp {c_idx})",
                    fontsize=10.5, fontweight="bold", color="#0D47A1",
                    rotation=0, labelpad=55, va="center"
                )

    save_path = os.path.join(FIGURES_DIR, "cifar10_raw_samples_grid.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Đã lưu lưới ảnh thô: {save_path}")


def visualize_raw_distributions(data_dir: str = "./data"):
    """Visualizes class balance and raw pixel RGB intensity distributions."""
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=False)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=False)
    classes = train_set.classes

    train_targets = np.array(train_set.targets)
    test_targets = np.array(test_set.targets)

    # Class balance (Train 45k, Val 5k, Test 10k)
    # Using seed 42 split from lab
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(train_targets))
    val_indices = indices[:5000]
    actual_train_indices = indices[5000:]

    train_counts = [np.sum(train_targets[actual_train_indices] == i) for i in range(10)]
    val_counts = [np.sum(train_targets[val_indices] == i) for i in range(10)]
    test_counts = [np.sum(test_targets == i) for i in range(10)]

    fig = plt.figure(figsize=(16, 7), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], wspace=0.25)

    # Subplot 1: Stacked Bar Chart for Split Distributions
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#FAFAFA")
    x = np.arange(len(classes))
    width = 0.65

    p1 = ax1.bar(x, train_counts, width, label="Train (45.000)", color="#1976D2", edgecolor="#0D47A1", alpha=0.9)
    p2 = ax1.bar(x, val_counts, width, bottom=train_counts, label="Validation (5.000)", color="#FFA726", edgecolor="#E65100", alpha=0.9)
    p3 = ax1.bar(x, test_counts, width, bottom=np.array(train_counts) + np.array(val_counts), label="Test (10.000)", color="#66BB6A", edgecolor="#2E7D32", alpha=0.9)

    ax1.set_title("Phân bố số lượng mẫu theo 10 lớp (Tổng: 60.000 ảnh)", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax1.set_xlabel("Lớp đối tượng CIFAR-10", fontsize=11, fontweight="bold", color="#37474F")
    ax1.set_ylabel("Số lượng mẫu (Ảnh)", fontsize=11, fontweight="bold", color="#37474F")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.capitalize() for c in classes], rotation=30, ha="right", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 7200)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=10)

    # Annotate total per bar
    for i in range(10):
        tot = train_counts[i] + val_counts[i] + test_counts[i]
        ax1.text(i, tot + 120, f"{tot:,}", ha="center", fontsize=9, fontweight="bold", color="#37474F")

    # Subplot 2: RGB Pixel Intensity Distribution
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#FAFAFA")

    # Compute RGB histogram from a sample of 5.000 images
    raw_images = train_set.data[:5000].astype(np.float32) / 255.0  # (5000, 32, 32, 3)
    r_chan = raw_images[:, :, :, 0].flatten()
    g_chan = raw_images[:, :, :, 1].flatten()
    b_chan = raw_images[:, :, :, 2].flatten()

    r_mean, r_std = np.mean(r_chan), np.std(r_chan)
    g_mean, g_std = np.mean(g_chan), np.std(g_chan)
    b_chan_mean, b_chan_std = np.mean(b_chan), np.std(b_chan)

    bins = np.linspace(0, 1.0, 50)
    ax2.hist(r_chan, bins=bins, color="#E53935", alpha=0.45, label=f"Kênh R (μ={r_mean:.3f}, σ={r_std:.3f})", density=True)
    ax2.hist(g_chan, bins=bins, color="#43A047", alpha=0.45, label=f"Kênh G (μ={g_mean:.3f}, σ={g_std:.3f})", density=True)
    ax2.hist(b_chan, bins=bins, color="#1E88E5", alpha=0.45, label=f"Kênh B (μ={b_chan_mean:.3f}, σ={b_chan_std:.3f})", density=True)

    ax2.set_title("Phân bố cường độ điểm ảnh 3 kênh màu (R, G, B)", fontsize=12, fontweight="bold", color="#0D47A1", pad=12)
    ax2.set_xlabel("Giá trị chuẩn hóa [0.0 - 1.0]", fontsize=11, fontweight="bold", color="#37474F")
    ax2.set_ylabel("Mật độ xác suất (Density)", fontsize=11, fontweight="bold", color="#37474F")
    ax2.grid(axis="both", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    fig.suptitle(
        "THỐNG KÊ PHÂN BỐ DỮ LIỆU THÔ CIFAR-10\n"
        "Độ cân bằng lớp hoàn hảo • Đặc tính phân phối màu sắc 3 kênh",
        fontsize=13.5, fontweight="bold", color="#0D47A1", y=0.99
    )

    save_path = os.path.join(FIGURES_DIR, "cifar10_raw_distribution.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[OK] Đã lưu thống kê phân bố: {save_path}")


if __name__ == "__main__":
    visualize_raw_samples_grid()
    visualize_raw_distributions()
