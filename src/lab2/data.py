"""
CIFAR-10 data loading, transformations, and reproducible splitting.
"""

import os
from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

from lab2.logger import get_logger, to_relative_path

CIFAR10_CLASSES: List[str] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

# Standard ImageNet normalization parameters required by torchvision pretrained backbones
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    image_size: int = 128,
    is_train: bool = True
) -> transforms.Compose:
    """
    Returns image transformation pipeline.
    Train: Resize, subtle data augmentation (RandomCrop, RandomHorizontalFlip), ToTensor, Normalize.
    Val/Test: Resize, ToTensor, Normalize (NO random augmentations).
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.125), padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class DatasetWithTransform(Dataset):
    """Wraps a Subset or Dataset to apply a specific transform dynamically."""

    def __init__(self, subset_or_dataset, transform):
        self.dataset = subset_or_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, target = self.dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def load_raw_cifar10(data_dir: str = "./data", download: bool = True) -> Tuple[Dataset, Dataset]:
    """
    Loads raw PIL CIFAR-10 train and test sets.
    Raises clear RuntimeError if dataset is missing and download fails.
    """
    os.makedirs(data_dir, exist_ok=True)

    try:
        raw_train = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=None)
        raw_test = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=None)
    except Exception as e:
        logger = get_logger()
        logger.exception(
            f"Failed to load or download CIFAR-10 dataset in '{to_relative_path(data_dir)}': {type(e).__name__}: {e}"
        )
        raise RuntimeError(
            f"Failed to load or download CIFAR-10 dataset in '{data_dir}'. "
            f"If offline or certificate verification fails, please ensure dataset files are pre-downloaded. Details: {e}"
        )

    return raw_train, raw_test


def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 16,
    val_split: float = 0.1,
    seed: int = 42,
    image_size: int = 128,
    num_workers: int = 0,
    subset_size: Optional[int] = None,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Builds reproducible train, validation, and test DataLoaders for CIFAR-10.

    - Splits the official 50,000 training set into train and val using a fixed generator seed.
    - Keeps the official 10,000 test set completely untouched for unbiased evaluation.
    - If subset_size is provided, extracts a small subset for rapid smoke testing.
    """
    raw_train, raw_test = load_raw_cifar10(data_dir=data_dir, download=download)

    total_train_len = len(raw_train)
    val_len = int(total_train_len * val_split)
    train_len = total_train_len - val_len

    # Reproducible split using fixed seed generator
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        raw_train,
        [train_len, val_len],
        generator=generator
    )

    train_transform = get_transforms(image_size=image_size, is_train=True)
    eval_transform = get_transforms(image_size=image_size, is_train=False)

    # Optional subset extraction for quick verification/smoke testing
    if subset_size is not None and subset_size > 0:
        smoke_train_len = min(subset_size, len(train_subset))
        smoke_val_len = min(max(10, subset_size // 4), len(val_subset))
        smoke_test_len = min(max(10, subset_size // 4), len(raw_test))

        train_subset = Subset(train_subset, list(range(smoke_train_len)))
        val_subset = Subset(val_subset, list(range(smoke_val_len)))
        test_dataset = Subset(raw_test, list(range(smoke_test_len)))
    else:
        test_dataset = raw_test

    train_data = DatasetWithTransform(train_subset, transform=train_transform)
    val_data = DatasetWithTransform(val_subset, transform=eval_transform)
    test_data = DatasetWithTransform(test_dataset, transform=eval_transform)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader, CIFAR10_CLASSES
