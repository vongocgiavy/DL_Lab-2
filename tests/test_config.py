"""
Unit tests for configuration loading and validation.
"""

import pytest
import yaml
from lab2.config import load_config, load_yaml, get_all_experiment_names, ExperimentConfig


def test_load_config_merging(tmp_path):
    """Verify load_config merges global and experiment settings correctly."""
    sample_yaml = {
        "global": {
            "num_classes": 10,
            "image_size": 128,
            "val_split": 0.15,
            "seed": 42,
            "device": "cpu",
            "checkpoint_metric": "val_acc",
            "data_dir": "./custom_data",
        },
        "experiments": {
            "exp_transfer": {
                "model_name": "resnet18",
                "method": "transfer",
                "epochs": 7,
                "batch_size": 32,
                "learning_rate": 0.002,
                "weight_decay": 0.0005,
                "description": "Test transfer",
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_yaml, f)

    cfg = load_config("exp_transfer", config_path=str(config_file))

    assert cfg.name == "exp_transfer"
    assert cfg.model_name == "resnet18"
    assert cfg.method == "transfer"
    assert cfg.epochs == 7
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 0.002
    assert cfg.weight_decay == 0.0005
    # Merged from global
    assert cfg.val_split == 0.15
    assert cfg.num_classes == 10
    assert cfg.image_size == 128
    assert cfg.data_dir == "./custom_data"
    assert cfg.checkpoint_metric == "val_acc"


def test_load_config_overrides(tmp_path):
    """Verify overrides dict overrides both global and experiment settings."""
    sample_yaml = {
        "global": {
            "val_split": 0.1,
            "device": "cpu",
        },
        "experiments": {
            "exp_test": {
                "model_name": "resnet18",
                "method": "transfer",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_yaml, f)

    cfg = load_config(
        "exp_test",
        config_path=str(config_file),
        overrides={"epochs": 1, "batch_size": 64, "learning_rate": 0.0005}
    )

    assert cfg.epochs == 1
    assert cfg.batch_size == 64
    assert cfg.learning_rate == 0.0005


def test_load_config_nonexistent_experiment(tmp_path):
    """Verify loading non-existent experiment raises KeyError."""
    sample_yaml = {
        "global": {},
        "experiments": {
            "existing_exp": {
                "model_name": "resnet18",
                "method": "transfer",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_yaml, f)

    with pytest.raises(KeyError, match="not found"):
        load_config("nonexistent_exp", config_path=str(config_file))


@pytest.mark.parametrize("invalid_field,invalid_val,expected_match", [
    ("val_split", 0.7, "val_split must be between 0.0 and 0.5"),
    ("val_split", 0.0, "val_split must be between 0.0 and 0.5"),
    ("checkpoint_metric", "invalid_metric", "checkpoint_metric must be 'val_acc' or 'val_loss'"),
    ("epochs", -1, "epochs must be > 0"),
    ("batch_size", 0, "batch_size must be > 0"),
    ("learning_rate", 0.0, "learning_rate must be > 0"),
    ("weight_decay", -0.01, "weight_decay must be >= 0"),
    ("optimizer", "rmsprop", "optimizer must be 'adam' or 'sgd'"),
    ("model_name", "alexnet", "Invalid model_name"),
    ("method", "pretrain", "Invalid method"),
    ("group", "invalid_group", "group must be 'core' or 'hparam'"),
])
def test_load_config_invalid_values(tmp_path, invalid_field, invalid_val, expected_match):
    """Verify invalid configuration values raise ValueError."""
    sample_yaml = {
        "global": {
            "val_split": 0.1,
            "checkpoint_metric": "val_acc",
        },
        "experiments": {
            "exp_test": {
                "model_name": "resnet18",
                "method": "transfer",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            }
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(sample_yaml, f)

    with pytest.raises(ValueError, match=expected_match):
        load_config("exp_test", config_path=str(config_file), overrides={invalid_field: invalid_val})


def test_experiment_group_default_and_filtering():
    """Verify default group is 'core' and get_all_experiment_names correctly filters groups."""
    cfg = load_config("resnet18_transfer")
    assert cfg.group == "core"

    core_names = get_all_experiment_names(group="core")
    hparam_names = get_all_experiment_names(group="hparam")
    all_names = get_all_experiment_names()

    # --all (core) must contain only the 4 core experiments and NOT hparam
    assert len(core_names) == 4
    for exp in ["resnet18_transfer", "resnet18_finetune", "densenet121_transfer", "densenet121_finetune"]:
        assert exp in core_names
    assert "resnet18_transfer_lr1e4" not in core_names
    assert "resnet18_transfer_bs32" not in core_names

    # hparam group must contain exactly the 2 new hparam experiments
    assert len(hparam_names) == 2
    assert "resnet18_transfer_lr1e4" in hparam_names
    assert "resnet18_transfer_bs32" in hparam_names

    # Total experiments must equal core + hparam
    assert len(all_names) == 6
    assert set(core_names).isdisjoint(set(hparam_names))
    assert set(core_names).union(set(hparam_names)) == set(all_names)


def test_hparam_configs_differ_by_single_field():
    """Verify hparam configs only differ from resnet18_transfer by exactly one hyperparameter."""
    yaml_data = load_yaml("./configs/config.yaml")
    exps = yaml_data["experiments"]

    base = exps["resnet18_transfer"]
    lr_exp = exps["resnet18_transfer_lr1e4"]
    bs_exp = exps["resnet18_transfer_bs32"]

    # Verify lr1e4 differs only in learning_rate
    assert lr_exp["learning_rate"] == 0.0001
    assert base["learning_rate"] == 0.001
    assert lr_exp["group"] == "hparam"
    for k in ["model_name", "method", "epochs", "batch_size", "optimizer", "weight_decay"]:
        assert lr_exp[k] == base[k], f"Field '{k}' differed unexpectedly"

    # Verify bs32 differs only in batch_size
    assert bs_exp["batch_size"] == 32
    assert base["batch_size"] == 16
    assert bs_exp["group"] == "hparam"
    for k in ["model_name", "method", "epochs", "learning_rate", "optimizer", "weight_decay"]:
        assert bs_exp[k] == base[k], f"Field '{k}' differed unexpectedly"

