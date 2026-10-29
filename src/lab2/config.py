"""
Configuration loader and validator for Lab 2 experiments.
"""

from dataclasses import dataclass, field
import math
import os
from typing import Dict, List, Optional
import yaml


@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    method: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    description: str = ""
    optimizer: str = "adam"
    backbone_lr: Optional[float] = None
    num_classes: int = 10
    image_size: int = 128
    val_split: float = 0.1
    seed: int = 42
    device: str = "cpu"
    num_workers: int = 0
    data_dir: str = "./data"
    checkpoints_dir: str = "./checkpoints"
    runs_dir: str = "./runs"
    outputs_dir: str = "./outputs"
    checkpoint_metric: str = "val_acc"
    group: str = "core"

    def validate(self) -> None:
        """Validates configuration parameters, raising ValueError with clear message if invalid."""
        if self.group not in ["core", "hparam"]:
            raise ValueError(f"group must be 'core' or 'hparam', got '{self.group}'")
        if self.model_name not in ["resnet18", "densenet121"]:
            raise ValueError(
                f"Invalid model_name '{self.model_name}'. Supported models: ['resnet18', 'densenet121']"
            )
        if self.method not in ["transfer", "finetune"]:
            raise ValueError(
                f"Invalid method '{self.method}'. Supported methods: ['transfer', 'finetune']"
            )
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.optimizer.lower() not in ["adam", "sgd"]:
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got '{self.optimizer}'")
        if self.backbone_lr is not None and (self.backbone_lr <= 0 or not math.isfinite(self.backbone_lr)):
            raise ValueError(f"backbone_lr must be > 0, got {self.backbone_lr}")
        if not (0.0 < self.val_split < 0.5):
            raise ValueError(f"val_split must be between 0.0 and 0.5, got {self.val_split}")
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {self.num_classes}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")
        if self.checkpoint_metric not in ["val_acc", "val_loss"]:
            raise ValueError(
                f"checkpoint_metric must be 'val_acc' or 'val_loss', got '{self.checkpoint_metric}'"
            )


def load_yaml(config_path: str) -> dict:
    """Loads raw YAML configuration dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}")
    return data


def get_all_experiment_names(
    config_path: str = "./configs/config.yaml",
    group: Optional[str] = None
) -> List[str]:
    """
    Returns list of experiment IDs defined in config.yaml.
    Optionally filters by group (e.g. 'core' or 'hparam').
    """
    data = load_yaml(config_path)
    experiments = data.get("experiments", {})
    if group is None:
        return list(experiments.keys())
    return [
        eid for eid, exp_cfg in experiments.items()
        if (exp_cfg.get("group", "core") if isinstance(exp_cfg, dict) else "core") == group
    ]


def load_config(
    experiment_id: str,
    config_path: str = "./configs/config.yaml",
    overrides: Optional[dict] = None
) -> ExperimentConfig:
    """
    Loads and validates configuration for a specific experiment ID, merging global settings.
    """
    data = load_yaml(config_path)
    global_cfg = data.get("global", {})
    experiments = data.get("experiments", {})

    if experiment_id not in experiments:
        available = list(experiments.keys())
        raise KeyError(
            f"Experiment '{experiment_id}' not found in {config_path}. Available: {available}"
        )

    exp_cfg = experiments[experiment_id]

    # Combine global + experiment settings
    merged = {
        "name": experiment_id,
        "model_name": exp_cfg.get("model_name"),
        "method": exp_cfg.get("method"),
        "epochs": exp_cfg.get("epochs", 5),
        "batch_size": exp_cfg.get("batch_size", 16),
        "learning_rate": exp_cfg.get("learning_rate", 0.001),
        "weight_decay": exp_cfg.get("weight_decay", 0.0001),
        "description": exp_cfg.get("description", ""),
        "optimizer": exp_cfg.get("optimizer", "adam"),
        "backbone_lr": exp_cfg.get("backbone_lr", None),
        "num_classes": global_cfg.get("num_classes", 10),
        "image_size": global_cfg.get("image_size", 128),
        "val_split": global_cfg.get("val_split", 0.1),
        "seed": global_cfg.get("seed", 42),
        "device": global_cfg.get("device", "cpu"),
        "num_workers": global_cfg.get("num_workers", 0),
        "data_dir": global_cfg.get("data_dir", "./data"),
        "checkpoints_dir": global_cfg.get("checkpoints_dir", "./checkpoints"),
        "runs_dir": global_cfg.get("runs_dir", "./runs"),
        "outputs_dir": global_cfg.get("outputs_dir", "./outputs"),
        "checkpoint_metric": global_cfg.get("checkpoint_metric", "val_acc"),
        "group": exp_cfg.get("group", "core"),
    }

    if overrides:
        merged.update(overrides)

    config = ExperimentConfig(**merged)
    config.validate()
    return config
