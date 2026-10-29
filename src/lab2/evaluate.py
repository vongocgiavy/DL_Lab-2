"""
Evaluation pipeline on the test dataset with metrics and confusion matrix generation.
"""

import json
import os
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lab2.checkpoint import load_checkpoint
from lab2.config import ExperimentConfig
from lab2.logger import get_logger, to_relative_path
from lab2.models import build_model
from lab2.utils import plot_confusion_matrix


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Evaluates model on dataloader without updating weights.
    Computes overall accuracy, loss, per-class accuracy, and confusion matrix.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_classes = len(class_names)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            # Check for NaN / Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger = get_logger()
                msg = f"Evaluation encountered {loss.item()} loss on batch."
                logger.error(msg)
                raise FloatingPointError(msg)

            batch_size = data.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long().item(), p.long().item()] += 1

    overall_loss = running_loss / total if total > 0 else 0.0
    overall_acc = 100.0 * correct / total if total > 0 else 0.0

    # Per-class accuracy
    per_class_acc = {}
    for i, name in enumerate(class_names):
        class_total = confusion_matrix[i, :].sum()
        class_correct = confusion_matrix[i, i]
        acc = (100.0 * class_correct / class_total) if class_total > 0 else 0.0
        per_class_acc[name] = round(float(acc), 2)

    return {
        "test_loss": float(overall_loss),
        "test_accuracy": float(overall_acc),
        "total_test_samples": int(total),
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion_matrix.tolist()
    }


def evaluate_checkpoint_file(
    checkpoint_path: str,
    test_loader: DataLoader,
    class_names: List[str],
    outputs_dir: str = "./outputs"
) -> Dict[str, Any]:
    """
    Loads model state from checkpoint and runs test evaluation.
    Saves JSON metrics to outputs/metrics/ and Confusion Matrix to outputs/figures/.
    """
    logger = get_logger()
    rel_ckpt = to_relative_path(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: '{rel_ckpt}'")
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Inspect checkpoint metadata to know model architecture and method
    device = torch.device("cpu")
    try:
        raw_ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        logger.exception(f"Failed to load checkpoint file at '{rel_ckpt}': {type(e).__name__}: {e}")
        raise RuntimeError(f"Failed to load checkpoint file at '{checkpoint_path}': {e}")

    model_name = raw_ckpt.get("model_name", "resnet18")
    method = raw_ckpt.get("method", "transfer")
    exp_name = raw_ckpt.get("experiment_name", os.path.splitext(os.path.basename(checkpoint_path))[0])

    model = build_model(
        model_name=model_name,
        method=method,
        num_classes=len(class_names),
        pretrained=False  # Weights will be loaded from checkpoint
    )
    load_checkpoint(checkpoint_path, model=model, device=device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names
    )

    metrics["experiment_name"] = exp_name
    metrics["checkpoint_evaluated"] = to_relative_path(checkpoint_path)

    # Save metrics JSON
    metrics_dir = os.path.join(outputs_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    json_path = os.path.join(metrics_dir, f"{exp_name}_test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save Confusion Matrix figure
    figures_dir = os.path.join(outputs_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    cm_path = os.path.join(figures_dir, f"{exp_name}_confusion_matrix.png")
    plot_confusion_matrix(
        cm=np.array(metrics["confusion_matrix"]),
        class_names=class_names,
        save_path=cm_path,
        title=f"Confusion Matrix: {exp_name}"
    )

    logger.info(
        f"Evaluation completed for '{exp_name}': "
        f"test_loss={metrics['test_loss']:.4f}, test_acc={metrics['test_accuracy']:.2f}%"
    )

    print(f"\nEvaluation Results for [{exp_name}]:")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
    print(f"Metrics saved to: {json_path}")
    print(f"Figure saved to: {cm_path}")

    return metrics
