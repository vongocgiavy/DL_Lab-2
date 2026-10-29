"""
TensorBoard logging integration aligned with Chapter 3 slides.
"""

import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils


class TensorBoardLogger:
    """Manages SummaryWriter and structured logging for Lab 2 experiments."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_scalars(self, step: int, metrics: Dict[str, float]) -> None:
        """Logs a dictionary of scalar metrics (e.g. Loss/train, Accuracy/val) at a given epoch/step."""
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, global_step=step)
        self.writer.flush()

    def log_graph(self, model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 128, 128)) -> None:
        """Visualizes model architecture graph with a dummy input (Slide 14 & 15)."""
        was_training = model.training
        try:
            model.eval()
            device = next(model.parameters()).device
            dummy_input = torch.randn(*input_size, device=device)
            with torch.no_grad():
                self.writer.add_graph(model, dummy_input)
            self.writer.flush()
        except Exception as e:
            # Graph visualization may fail on certain dynamic layers; warn instead of crashing
            print(f"[TensorBoardLogger] Warning: Could not log model graph: {e}")
        finally:
            model.train(was_training)

    def log_sample_images(self, tag: str, images: torch.Tensor, step: int = 0, max_images: int = 16) -> None:
        """Visualizes sample images from dataset (Slide 14 & 15)."""
        try:
            img_subset = images[:max_images].cpu()
            grid = vutils.make_grid(img_subset, normalize=True, scale_each=True)
            self.writer.add_image(tag, grid, global_step=step)
            self.writer.flush()
        except Exception as e:
            print(f"[TensorBoardLogger] Warning: Could not log sample images: {e}")

    def log_histograms(self, model: nn.Module, step: int) -> None:
        """Visualizes parameter weight distributions (Slide 14 & 15)."""
        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(name, param.detach().cpu(), global_step=step)
            self.writer.flush()
        except Exception as e:
            print(f"[TensorBoardLogger] Warning: Could not log parameter histograms: {e}")

    def close(self) -> None:
        """Closes the SummaryWriter session."""
        if self.writer is not None:
            self.writer.close()
