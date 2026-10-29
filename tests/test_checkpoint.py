"""
Unit tests for checkpoint saving, loading, atomic weight management, and logging.
All tests run offline on CPU using tmp_path without loading pre-trained weights.
"""

import json
import os
import unittest.mock as mock
import pytest
import torch
import torch.nn as nn

from lab2.checkpoint import (
    atomic_save,
    load_checkpoint,
    prune_old_weights,
    save_checkpoint,
    save_run_weights_and_manage_best,
    verify_checkpoint_integrity
)
from lab2.logger import setup_logging
from lab2.models import build_model


class TinyTestModel(nn.Module):
    """Small offline model for fast unit testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# -----------------------------------------------------------------------------
# Test 7: Original tests pass untouched
# -----------------------------------------------------------------------------

def test_checkpoint_save_and_load(tmp_path):
    """Verify that saving and loading checkpoint faithfully recovers model weights and metadata."""
    ckpt_path = os.path.join(tmp_path, "test_ckpt.pth")
    best_path = os.path.join(tmp_path, "test_best.pth")

    model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    # Modify fc weight with known values
    with torch.no_grad():
        model.fc.weight.fill_(3.1415)

    state = {
        "epoch": 3,
        "model_name": "resnet18",
        "method": "transfer",
        "model_state_dict": model.state_dict(),
        "val_acc": 85.5,
        "val_loss": 0.42
    }

    save_checkpoint(state, filepath=ckpt_path, is_best=True, best_filepath=best_path)

    assert os.path.exists(ckpt_path)
    assert os.path.exists(best_path)

    new_model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    loaded_state = load_checkpoint(ckpt_path, model=new_model)

    assert loaded_state["epoch"] == 3
    assert loaded_state["val_acc"] == 85.5
    assert torch.equal(new_model.fc.weight, model.fc.weight)


def test_load_nonexistent_checkpoint():
    """Verify FileNotFoundError is raised for missing checkpoint."""
    model = build_model("resnet18", method="transfer", num_classes=10, pretrained=False)
    with pytest.raises(FileNotFoundError):
        load_checkpoint("non_existent_file.pth", model=model)


# -----------------------------------------------------------------------------
# Test 1: First run creates best, meta, and weights
# -----------------------------------------------------------------------------

def test_first_run_creates_best_meta_weights(tmp_path):
    """(1) Lần chạy đầu tạo best.pth, best_meta.json và weights.pth."""
    model = TinyTestModel()
    with torch.no_grad():
        model.fc.weight.fill_(1.5)
    state_dict = model.state_dict()
    ckpt_state = {
        "epoch": 2,
        "model_state_dict": state_dict,
        "val_acc": 74.0,
        "experiment_name": "exp_first"
    }

    res = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_first",
        run_best_state_dict=state_dict,
        run_best_checkpoint_state=ckpt_state,
        run_best_val_acc=74.0,
        run_best_epoch=2,
        seed=42,
        val_split=0.1
    )

    assert res["best_updated"] is True
    weights_path = tmp_path / "exp_first_weights.pth"
    best_path = tmp_path / "exp_first_best.pth"
    meta_path = tmp_path / "exp_first_best_meta.json"

    assert weights_path.exists()
    assert best_path.exists()
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["experiment_name"] == "exp_first"
    assert meta["best_val_acc"] == 74.0
    assert meta["epoch"] == 2
    assert meta["seed"] == 42
    assert meta["val_split"] == 0.1
    assert "saved_at" in meta


# -----------------------------------------------------------------------------
# Test 2: Second run with lower val_acc -> weights replaced, best unchanged
# -----------------------------------------------------------------------------

def test_second_run_lower_val_acc_weights_replaced_best_unchanged(tmp_path):
    """(2) Lần chạy thứ hai có val_acc thấp hơn: best không đổi, weights được thay."""
    model1 = TinyTestModel()
    with torch.no_grad():
        model1.fc.weight.fill_(1.0)
    state1 = model1.state_dict()
    ckpt1 = {"epoch": 3, "model_state_dict": state1, "val_acc": 80.0}

    # Run 1: 80.0%
    save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_test",
        run_best_state_dict=state1,
        run_best_checkpoint_state=ckpt1,
        run_best_val_acc=80.0,
        run_best_epoch=3,
        seed=42,
        val_split=0.1
    )

    # Run 2: 65.0% (lower!)
    model2 = TinyTestModel()
    with torch.no_grad():
        model2.fc.weight.fill_(2.0)
    state2 = model2.state_dict()
    ckpt2 = {"epoch": 2, "model_state_dict": state2, "val_acc": 65.0}

    res2 = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_test",
        run_best_state_dict=state2,
        run_best_checkpoint_state=ckpt2,
        run_best_val_acc=65.0,
        run_best_epoch=2,
        seed=42,
        val_split=0.1
    )

    assert res2["best_updated"] is False

    # Best must still retain Run 1 weights (1.0)
    best_data = torch.load(tmp_path / "exp_test_best.pth", map_location="cpu")
    assert torch.allclose(best_data["model_state_dict"]["fc.weight"], torch.tensor(1.0))

    # Weights must be replaced with Run 2 weights (2.0)
    weights_data = torch.load(tmp_path / "exp_test_weights.pth", map_location="cpu")
    assert torch.allclose(weights_data["fc.weight"], torch.tensor(2.0))

    # Meta must still retain Run 1 best val_acc (80.0)
    with open(tmp_path / "exp_test_best_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["best_val_acc"] == 80.0


# -----------------------------------------------------------------------------
# Test 3: Third run with higher val_acc -> best and meta updated
# -----------------------------------------------------------------------------

def test_third_run_higher_val_acc_updates_best_and_meta(tmp_path):
    """(3) Lần chạy có val_acc cao hơn: best và meta được cập nhật."""
    model1 = TinyTestModel()
    with torch.no_grad():
        model1.fc.weight.fill_(1.0)
    state1 = model1.state_dict()
    ckpt1 = {"epoch": 3, "model_state_dict": state1, "val_acc": 80.0}

    # Initial Run: 80.0%
    save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_test",
        run_best_state_dict=state1,
        run_best_checkpoint_state=ckpt1,
        run_best_val_acc=80.0,
        run_best_epoch=3,
        seed=42,
        val_split=0.1
    )

    # Next Run: 92.5% (higher!)
    model3 = TinyTestModel()
    with torch.no_grad():
        model3.fc.weight.fill_(3.0)
    state3 = model3.state_dict()
    ckpt3 = {"epoch": 5, "model_state_dict": state3, "val_acc": 92.5}

    res3 = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_test",
        run_best_state_dict=state3,
        run_best_checkpoint_state=ckpt3,
        run_best_val_acc=92.5,
        run_best_epoch=5,
        seed=42,
        val_split=0.1
    )

    assert res3["best_updated"] is True

    # Best must now have Run 3 weights (3.0)
    best_data = torch.load(tmp_path / "exp_test_best.pth", map_location="cpu")
    assert torch.allclose(best_data["model_state_dict"]["fc.weight"], torch.tensor(3.0))

    # Meta must be updated to 92.5
    with open(tmp_path / "exp_test_best_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["best_val_acc"] == 92.5
    assert meta["epoch"] == 5


# -----------------------------------------------------------------------------
# Test 4: Prune old weights & safety
# -----------------------------------------------------------------------------

def test_prune_old_weights_and_safety(tmp_path):
    """
    (4) File lạ <exp>_old.pth bị xóa và có dòng log;
    file của thí nghiệm khác và file ngoài thư mục không bị đụng.
    """
    # Create test files
    (tmp_path / "myexp_best.pth").write_text("best")
    (tmp_path / "myexp_weights.pth").write_text("weights")
    (tmp_path / "myexp_old.pth").write_text("old")
    (tmp_path / "myexp_stale.tmp").write_text("tmp")
    (tmp_path / "other_exp_old.pth").write_text("other")
    (tmp_path / "other_exp_best.pth").write_text("other_best")

    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "myexp_external.pth").write_text("external")

    # Test dry_run=True: should identify files without deleting
    dry_pruned = prune_old_weights(
        str(tmp_path),
        "myexp",
        keep={"myexp_best.pth", "myexp_weights.pth"},
        dry_run=True
    )
    assert "myexp_old.pth" in dry_pruned
    assert "myexp_stale.tmp" in dry_pruned
    assert (tmp_path / "myexp_old.pth").exists()  # Not deleted

    # Test dry_run=False: should delete matching files
    pruned = prune_old_weights(
        str(tmp_path),
        "myexp",
        keep={"myexp_best.pth", "myexp_weights.pth"},
        dry_run=False
    )
    assert "myexp_old.pth" in pruned
    assert "myexp_stale.tmp" in pruned

    assert not (tmp_path / "myexp_old.pth").exists()
    assert not (tmp_path / "myexp_stale.tmp").exists()
    assert (tmp_path / "myexp_best.pth").exists()
    assert (tmp_path / "myexp_weights.pth").exists()

    # Safety checks: other experiment files must not be touched
    assert (tmp_path / "other_exp_old.pth").exists()
    assert (tmp_path / "other_exp_best.pth").exists()
    # Safety checks: outside directory files must not be touched
    assert (outside_dir / "myexp_external.pth").exists()


# -----------------------------------------------------------------------------
# Test 5: Atomic save failure preserves old best
# -----------------------------------------------------------------------------

def test_atomic_save_failure_preserves_old_best(tmp_path):
    """(5) Lỗi ghi giữa chừng không làm mất best cũ."""
    best_file = tmp_path / "save_fail_test.pth"
    original_data = {"epoch": 1, "status": "original_intact"}
    torch.save(original_data, best_file)

    # Simulate an unexpected failure during save
    with mock.patch("torch.save", side_effect=IOError("Simulated write failure")):
        with pytest.raises(IOError):
            atomic_save({"epoch": 2, "status": "new_corrupt"}, str(best_file), is_torch=True)

    # Verify old file is completely intact
    loaded = torch.load(best_file, map_location="cpu")
    assert loaded["status"] == "original_intact"
    assert loaded["epoch"] == 1


# -----------------------------------------------------------------------------
# Test 6: Incomparable seed or val_split warns and preserves best
# -----------------------------------------------------------------------------

def test_incomparable_seed_or_val_split_warns_and_preserves_best(tmp_path):
    """(6) Seed/val_split khác nhau: best không đổi và ghi log WARNING."""
    model = TinyTestModel()
    with torch.no_grad():
        model.fc.weight.fill_(1.0)
    state1 = model.state_dict()
    ckpt1 = {"model_state_dict": state1, "val_acc": 70.0}

    # Run 1: seed=42, val_split=0.1
    save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_split",
        run_best_state_dict=state1,
        run_best_checkpoint_state=ckpt1,
        run_best_val_acc=70.0,
        run_best_epoch=1,
        seed=42,
        val_split=0.1
    )

    # Run 2: seed=99 (different seed!), val_acc=95.0 (much higher)
    with torch.no_grad():
        model.fc.weight.fill_(2.0)
    state2 = model.state_dict()
    ckpt2 = {"model_state_dict": state2, "val_acc": 95.0}

    res2 = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_split",
        run_best_state_dict=state2,
        run_best_checkpoint_state=ckpt2,
        run_best_val_acc=95.0,
        run_best_epoch=1,
        seed=99,
        val_split=0.1
    )

    # Best must NOT be updated because seeds are incomparable
    assert res2["best_updated"] is False

    best_loaded = torch.load(tmp_path / "exp_split_best.pth", map_location="cpu")
    assert torch.allclose(best_loaded["model_state_dict"]["fc.weight"], torch.tensor(1.0))

    # Run weights MUST be updated to state2 (2.0)
    weights_loaded = torch.load(tmp_path / "exp_split_weights.pth", map_location="cpu")
    assert torch.allclose(weights_loaded["fc.weight"], torch.tensor(2.0))


# -----------------------------------------------------------------------------
# Test 8: Setup logging is idempotent and logs errors with traceback
# -----------------------------------------------------------------------------

def test_setup_logging_idempotent_and_error_traceback(tmp_path):
    """
    (8) setup_logging gọi hai lần không nhân đôi handler;
    lỗi được ghi vào errors.log kèm traceback.
    """
    log_dir = str(tmp_path / "logs")

    # Calling twice should not duplicate handlers
    logger1 = setup_logging(log_dir=log_dir)
    assert len(logger1.handlers) == 3

    logger2 = setup_logging(log_dir=log_dir)
    assert len(logger2.handlers) == 3

    # Log an error with exception traceback
    try:
        _ = 1 / 0
    except ZeroDivisionError:
        logger2.exception("Test exception logging with traceback")

    errors_log = os.path.join(log_dir, "errors.log")
    assert os.path.exists(errors_log)

    with open(errors_log, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Test exception logging with traceback" in content
    assert "ZeroDivisionError" in content
    assert "Traceback" in content


# -----------------------------------------------------------------------------
# Test 9: Checkpoint management with val_loss metric (lower is better)
# -----------------------------------------------------------------------------

def test_manage_best_with_val_loss_metric(tmp_path):
    """(9) save_run_weights_and_manage_best với val_loss: giá trị nhỏ hơn là tốt hơn."""
    model = TinyTestModel()
    with torch.no_grad():
        model.fc.weight.fill_(1.0)
    state1 = model.state_dict()
    ckpt1 = {"epoch": 1, "model_state_dict": state1, "val_loss": 0.8}

    # Run 1: val_loss = 0.8
    res1 = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_loss",
        run_best_state_dict=state1,
        run_best_checkpoint_state=ckpt1,
        run_best_val_acc=0.8,
        run_best_epoch=1,
        seed=42,
        val_split=0.1,
        checkpoint_metric="val_loss"
    )
    assert res1["best_updated"] is True

    # Run 2: val_loss = 0.4 (lower -> better -> should update best)
    with torch.no_grad():
        model.fc.weight.fill_(2.0)
    state2 = model.state_dict()
    ckpt2 = {"epoch": 2, "model_state_dict": state2, "val_loss": 0.4}

    res2 = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_loss",
        run_best_state_dict=state2,
        run_best_checkpoint_state=ckpt2,
        run_best_val_acc=0.4,
        run_best_epoch=2,
        seed=42,
        val_split=0.1,
        checkpoint_metric="val_loss"
    )
    assert res2["best_updated"] is True

    # Run 3: val_loss = 0.9 (higher -> worse -> should NOT update best)
    with torch.no_grad():
        model.fc.weight.fill_(3.0)
    state3 = model.state_dict()
    ckpt3 = {"epoch": 3, "model_state_dict": state3, "val_loss": 0.9}

    res3 = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_loss",
        run_best_state_dict=state3,
        run_best_checkpoint_state=ckpt3,
        run_best_val_acc=0.9,
        run_best_epoch=3,
        seed=42,
        val_split=0.1,
        checkpoint_metric="val_loss"
    )
    assert res3["best_updated"] is False

    # Best file must still hold state2 (fill 2.0)
    best_loaded = torch.load(tmp_path / "exp_loss_best.pth", map_location="cpu")
    assert torch.allclose(best_loaded["model_state_dict"]["fc.weight"], torch.tensor(2.0))


# -----------------------------------------------------------------------------
# Test 10: Best checkpoint state dictionary deep snapshot isolation
# -----------------------------------------------------------------------------

def test_best_checkpoint_state_is_deep_independent_snapshot(tmp_path):
    """(10) Trọng số trong best_checkpoint_state được sao chép độc lập, không bị thay đổi khi model tiếp tục cập nhật."""
    import copy
    model = TinyTestModel()
    with torch.no_grad():
        model.fc.weight.fill_(5.0)

    # Snapshot created as in train.py
    best_checkpoint_state = {
        "epoch": 1,
        "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "val_acc": 90.0
    }

    # Subsequent training modifies model in-place
    with torch.no_grad():
        model.fc.weight.fill_(99.0)

    # Stored snapshot MUST remain unchanged at 5.0
    assert torch.allclose(best_checkpoint_state["model_state_dict"]["fc.weight"], torch.tensor(5.0))
    assert not torch.allclose(best_checkpoint_state["model_state_dict"]["fc.weight"], model.fc.weight)



# -----------------------------------------------------------------------------
# Test 11: Incompatible hyperparameters retain existing best checkpoint
# -----------------------------------------------------------------------------

def test_incompatible_hyperparameters_retains_existing_best(tmp_path):
    """(11) Khác biệt hyperparameter (learning_rate, batch_size) từ chối cập nhật best checkpoint."""
    model = TinyTestModel()
    with torch.no_grad():
        model.fc.weight.fill_(1.0)
    state1 = model.state_dict()
    ckpt1 = {
        "epoch": 2,
        "model_state_dict": state1,
        "val_acc": 75.0,
        "config": {"learning_rate": 0.001, "batch_size": 16, "model_name": "resnet18"}
    }

    # Run 1: lr=0.001
    save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_hparam",
        run_best_state_dict=state1,
        run_best_checkpoint_state=ckpt1,
        run_best_val_acc=75.0,
        run_best_epoch=2,
        seed=42,
        val_split=0.1,
        config_dict=ckpt1["config"]
    )

    # Run 2: lr=0.0001 (incompatible LR!), higher acc=85.0
    with torch.no_grad():
        model.fc.weight.fill_(2.0)
    state2 = model.state_dict()
    ckpt2 = {
        "epoch": 3,
        "model_state_dict": state2,
        "val_acc": 85.0,
        "config": {"learning_rate": 0.0001, "batch_size": 16, "model_name": "resnet18"}
    }

    res2 = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_hparam",
        run_best_state_dict=state2,
        run_best_checkpoint_state=ckpt2,
        run_best_val_acc=85.0,
        run_best_epoch=3,
        seed=42,
        val_split=0.1,
        config_dict=ckpt2["config"]
    )

    # Best must not be updated due to hyperparameter mismatch
    assert res2["best_updated"] is False
    best_data = torch.load(tmp_path / "exp_hparam_best.pth", map_location="cpu")
    assert torch.allclose(best_data["model_state_dict"]["fc.weight"], torch.tensor(1.0))


# -----------------------------------------------------------------------------
# Test 12: Recovery of metadata from existing best.pth when meta.json is missing
# -----------------------------------------------------------------------------

def test_missing_meta_recovers_from_existing_best_checkpoint(tmp_path):
    """(12) Khi best_meta.json bị mất, hệ thống khôi phục từ best.pth và bảo toàn nếu run mới kém hơn."""
    model = TinyTestModel()
    with torch.no_grad():
        model.fc.weight.fill_(7.0)
    state1 = model.state_dict()
    ckpt1 = {
        "epoch": 4,
        "model_state_dict": state1,
        "best_metric": 88.0,
        "val_acc": 88.0,
        "config": {"seed": 42, "val_split": 0.1, "checkpoint_metric": "val_acc"}
    }

    # Save initial run
    save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_recover",
        run_best_state_dict=state1,
        run_best_checkpoint_state=ckpt1,
        run_best_val_acc=88.0,
        run_best_epoch=4,
        seed=42,
        val_split=0.1
    )

    # Delete meta.json to simulate corruption/loss
    meta_path = tmp_path / "exp_recover_best_meta.json"
    assert meta_path.exists()
    meta_path.unlink()

    # New run with lower val_acc (60.0%)
    with torch.no_grad():
        model.fc.weight.fill_(8.0)
    state2 = model.state_dict()
    ckpt2 = {
        "epoch": 2,
        "model_state_dict": state2,
        "val_acc": 60.0,
        "config": {"seed": 42, "val_split": 0.1, "checkpoint_metric": "val_acc"}
    }

    res2 = save_run_weights_and_manage_best(
        checkpoint_dir=str(tmp_path),
        exp_name="exp_recover",
        run_best_state_dict=state2,
        run_best_checkpoint_state=ckpt2,
        run_best_val_acc=60.0,
        run_best_epoch=2,
        seed=42,
        val_split=0.1
    )

    # Should NOT overwrite recovered 88.0% best
    assert res2["best_updated"] is False
    best_data = torch.load(tmp_path / "exp_recover_best.pth", map_location="cpu")
    assert torch.allclose(best_data["model_state_dict"]["fc.weight"], torch.tensor(7.0))
