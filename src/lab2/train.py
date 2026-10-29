"""
Training and validation loop orchestration following Chapter 3 slides.
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from lab2.checkpoint import save_checkpoint, save_run_weights_and_manage_best
from lab2.config import ExperimentConfig
from lab2.logger import get_logger, to_relative_path
from lab2.tensorboard_logger import TensorBoardLogger
from lab2.utils import format_time


def create_optimizer(
    model: nn.Module,
    cfg: ExperimentConfig
) -> optim.Optimizer:
    """
    Creates optimizer configured for the experiment.
    Supports differential learning rates for backbone vs classifier head in fine-tuning.
    """
    if cfg.method == "finetune" and cfg.backbone_lr is not None:
        # Separate parameters into backbone unfrozen group and classifier head group
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc.") or name.startswith("classifier."):
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {"params": backbone_params, "lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.learning_rate}
        ]
    else:
        # Transfer learning or uniform learning rate
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.learning_rate}
        ]

    if cfg.optimizer.lower() == "adam":
        optimizer = optim.Adam(param_groups, weight_decay=cfg.weight_decay)
    elif cfg.optimizer.lower() == "sgd":
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'. Supported: 'adam', 'sgd'.")

    return optimizer


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    exp_name: str = "",
    epoch: int = 1
) -> Tuple[float, float]:
    """
    Executes one training epoch following Slides 8 & 9.
    Returns: (epoch_loss, epoch_accuracy_percent)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)

        # 1. Zero gradients
        optimizer.zero_grad()

        # 2. Forward pass
        outputs = model(data)

        # 3. Compute loss
        loss = criterion(outputs, labels)

        # Check for NaN / Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger = get_logger()
            msg = (
                f"Experiment '{exp_name}' encountered {loss.item()} loss "
                f"at epoch {epoch}, batch {batch_idx}."
            )
            logger.error(msg)
            raise FloatingPointError(msg)

        # 4. Backward pass
        loss.backward()

        # 5. Optimization step
        optimizer.step()

        # Accumulate metrics
        batch_size = data.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = 100.0 * correct / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    exp_name: str = "",
    epoch: int = 1
) -> Tuple[float, float]:
    """
    Executes validation loop following Slides 10 & 11.
    Returns: (val_loss, val_accuracy_percent)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            # Check for NaN / Inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                logger = get_logger()
                msg = (
                    f"Experiment '{exp_name}' encountered {loss.item()} validation loss "
                    f"at epoch {epoch}, batch {batch_idx}."
                )
                logger.error(msg)
                raise FloatingPointError(msg)

            batch_size = data.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = 100.0 * correct / total if total > 0 else 0.0
    return val_loss, val_acc


def run_training(
    cfg: ExperimentConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: Optional[TensorBoardLogger] = None
) -> Dict[str, Any]:
    """
    Runs full training and validation lifecycle across configured epochs.
    Tracks best checkpoint based on validation metric (val_acc or val_loss).
    """
    app_logger = get_logger()
    app_logger.info(
        f"Starting run: experiment='{cfg.name}', model='{cfg.model_name}', method='{cfg.method}', "
        f"seed={cfg.seed}, device='{cfg.device}', torch={torch.__version__}, torchvision={torchvision.__version__}"
    )

    device = torch.device(cfg.device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, cfg)

    # Log model architecture graph and sample batch to TensorBoard if logger available
    if logger is not None:
        logger.log_graph(model, input_size=(1, 3, cfg.image_size, cfg.image_size))
        try:
            sample_data, _ = next(iter(train_loader))
            logger.log_sample_images("CIFAR10/Train_Batch_Samples", sample_data, step=0)
        except Exception:
            pass

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_times": []
    }

    best_metric = -float("inf") if cfg.checkpoint_metric == "val_acc" else float("inf")
    best_epoch = 0
    best_val_acc = 0.0
    best_model_state_dict: Dict[str, Any] = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    best_checkpoint_state: Dict[str, Any] = {}

    latest_ckpt_path = os.path.join(cfg.checkpoints_dir, f"{cfg.name}_latest.pth")
    best_ckpt_path = os.path.join(cfg.checkpoints_dir, f"{cfg.name}_best.pth")

    print(f"\n==================================================")
    print(f"Starting Experiment: {cfg.name}")
    print(f"Model: {cfg.model_name} | Method: {cfg.method}")
    print(f"Epochs: {cfg.epochs} | Batch Size: {cfg.batch_size} | Device: {cfg.device}")
    print(f"==================================================")

    total_start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, exp_name=cfg.name, epoch=epoch
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, exp_name=cfg.name, epoch=epoch
        )

        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_duration)

        # TensorBoard Logging
        if logger is not None:
            logger.log_scalars(epoch, {
                "Loss/train": train_loss,
                "Loss/val": val_loss,
                "Accuracy/train": train_acc,
                "Accuracy/val": val_acc,
            })
            logger.log_histograms(model, step=epoch)

        # Checkpoint evaluation
        is_best = False
        if cfg.checkpoint_metric == "val_acc":
            if val_acc > best_metric:
                best_metric = val_acc
                best_epoch = epoch
                is_best = True
        else:
            if val_loss < best_metric:
                best_metric = val_loss
                best_epoch = epoch
                is_best = True

        checkpoint_state = {
            "epoch": epoch,
            "experiment_name": cfg.name,
            "model_name": cfg.model_name,
            "method": cfg.method,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "config": vars(cfg)
        }

        if is_best:
            best_model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_checkpoint_state = {
                "epoch": epoch,
                "experiment_name": cfg.name,
                "model_name": cfg.model_name,
                "method": cfg.method,
                "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "config": vars(cfg)
            }
            best_val_acc = val_acc

        save_checkpoint(
            state=checkpoint_state,
            filepath=latest_ckpt_path,
            is_best=is_best,
            best_filepath=best_ckpt_path
        )

        star = " * [Best]" if is_best else ""
        app_logger.info(
            f"Epoch [{epoch:02d}/{cfg.epochs:02d}] for '{cfg.name}': "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, "
            f"time={format_time(epoch_duration)}{star}"
        )
        print(
            f"Epoch [{epoch:02d}/{cfg.epochs:02d}] "
            f"Time: {format_time(epoch_duration)} | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%{star}"
        )

    total_time = time.time() - total_start_time

    # Ensure fallback if no epoch was marked best
    if not best_checkpoint_state:
        best_model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_checkpoint_state = {
            "epoch": cfg.epochs,
            "experiment_name": cfg.name,
            "model_name": cfg.model_name,
            "method": cfg.method,
            "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
            "train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
            "val_loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
            "val_acc": history["val_acc"][-1] if history["val_acc"] else 0.0,
            "best_metric": best_metric,
            "best_epoch": cfg.epochs,
            "config": vars(cfg)
        }
        best_val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
        best_epoch = cfg.epochs

    # Manage run weights and best checkpoint across runs
    effective_val_acc = best_val_acc if cfg.checkpoint_metric != "val_acc" else best_metric
    ckpt_summary = save_run_weights_and_manage_best(
        checkpoint_dir=cfg.checkpoints_dir,
        exp_name=cfg.name,
        run_best_state_dict=best_model_state_dict,
        run_best_checkpoint_state=best_checkpoint_state,
        run_best_val_acc=effective_val_acc,
        run_best_epoch=best_epoch,
        seed=cfg.seed,
        val_split=cfg.val_split,
        keep_latest=False,
        checkpoint_metric=cfg.checkpoint_metric,
        config_dict=vars(cfg)
    )

    app_logger.info(
        f"Finished run: experiment='{cfg.name}', total_time={format_time(total_time)}, "
        f"best_epoch={best_epoch}, best_{cfg.checkpoint_metric}={best_metric:.2f}"
    )
    print(f"\nTraining completed in {format_time(total_time)}.")
    print(f"Best checkpoint at epoch {best_epoch} with {cfg.checkpoint_metric}: {best_metric:.2f}")
    print(f"Saved: {best_ckpt_path}")

    return {
        "experiment_name": cfg.name,
        "total_time_seconds": total_time,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "best_checkpoint_path": best_ckpt_path,
        "weights_path": ckpt_summary["weights_path"],
        "checkpoint_summary": ckpt_summary,
        "history": history
    }
