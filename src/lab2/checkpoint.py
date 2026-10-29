"""
Checkpoint saving, loading, and safe weight management utilities with integrity verification.
"""

from datetime import datetime, timezone
import json
import os
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Set
import torch
import torch.nn as nn
import torch.optim as optim

from lab2.logger import get_logger, to_relative_path


def atomic_save(obj: Any, target_path: str, is_torch: bool = True) -> None:
    """
    Saves obj atomically by first writing to a temporary file in the same directory,
    then renaming with os.replace. If an error occurs, the temporary file is removed
    and the target_path remains unchanged.
    """
    target_path = os.path.abspath(target_path)
    dir_name = os.path.dirname(target_path)
    os.makedirs(dir_name, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix="tmp_", suffix=".tmp")
    os.close(fd)

    try:
        if is_torch:
            torch.save(obj, tmp_path)
        else:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def verify_checkpoint_integrity(
    filepath: str,
    expected_keys: Optional[Iterable[str]] = None
) -> bool:
    """
    Verifies that saved checkpoint file can be loaded with torch.load(..., map_location='cpu')
    and its state_dict keys match expected_keys (if provided).
    """
    logger = get_logger()
    try:
        loaded = torch.load(filepath, map_location="cpu")
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            logger.error(
                f"Integrity check failed: loaded object from '{to_relative_path(filepath)}' is not a dict"
            )
            return False

        if expected_keys is not None:
            actual_keys = set(state_dict.keys())
            exp_keys = set(expected_keys)
            if actual_keys != exp_keys:
                logger.error(
                    f"Integrity check failed: keys mismatch in '{to_relative_path(filepath)}'"
                )
                return False
        return True
    except Exception as e:
        logger.exception(f"Integrity verification failed for '{to_relative_path(filepath)}': {e}")
        return False


def prune_old_weights(
    checkpoint_dir: str,
    exp_name: str,
    keep: Iterable[str],
    dry_run: bool = False
) -> List[str]:
    """
    Safely prunes old weight/checkpoint files for the specified experiment.
    Safety guarantees:
    - Never deletes best checkpoint (<exp>_best.pth).
    - Never touches files belonging to other experiments.
    - Never deletes files outside checkpoint_dir.
    - Skips symlinks.
    - Only matches files starting with '<exp>_' and ending with .pth or .tmp.
    - Supports dry_run mode.
    """
    logger = get_logger()
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        return []

    keep_filenames: Set[str] = {os.path.basename(k) for k in keep}
    # Best checkpoint must never be deleted under any circumstances
    keep_filenames.add(f"{exp_name}_best.pth")

    prefix = f"{exp_name}_"
    pruned: List[str] = []

    for filename in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, filename)

        # Skip symlinks and directories
        if os.path.islink(file_path) or not os.path.isfile(file_path):
            continue

        # Must start with exact experiment prefix
        if not filename.startswith(prefix):
            continue

        # Must end with .pth or .tmp
        if not (filename.endswith(".pth") or filename.endswith(".tmp")):
            continue

        # Check keep list
        if filename in keep_filenames:
            continue

        pruned.append(filename)
        rel_path = to_relative_path(file_path)

        if dry_run:
            logger.info(f"[Dry Run] Would prune old checkpoint: '{rel_path}'")
        else:
            try:
                size = os.path.getsize(file_path)
                os.remove(file_path)
                logger.info(f"Pruned old checkpoint: '{rel_path}' ({size} bytes)")
            except Exception as e:
                logger.exception(f"Failed to prune old checkpoint '{rel_path}': {e}")

    return pruned


def save_run_weights_and_manage_best(
    checkpoint_dir: str,
    exp_name: str,
    run_best_state_dict: Dict[str, Any],
    run_best_checkpoint_state: Dict[str, Any],
    run_best_val_acc: float,
    run_best_epoch: int,
    seed: int,
    val_split: float,
    keep_latest: bool = False,
    checkpoint_metric: str = "val_acc"
) -> Dict[str, Any]:
    """
    Manages weights and best checkpoint for an experiment after each run:
    1. Always saves checkpoints/<exp>_weights.pth with state_dict of best epoch in this run.
    2. Keeps best across all runs in checkpoints/<exp>_best.pth + <exp>_best_meta.json.
    3. Replaces best if run metric improves over previous best in meta (higher val_acc or lower val_loss).
    4. Warns and preserves best if seed or val_split differs.
    5. Replaces best with warning if best.pth exists but meta is missing.
    6. Prunes old weights and .tmp files of this experiment.
    """
    logger = get_logger()
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    weights_path = os.path.join(checkpoint_dir, f"{exp_name}_weights.pth")
    best_path = os.path.join(checkpoint_dir, f"{exp_name}_best.pth")
    meta_path = os.path.join(checkpoint_dir, f"{exp_name}_best_meta.json")

    # 1. Always save run weights (only state_dict) atomically
    atomic_save(run_best_state_dict, weights_path, is_torch=True)
    if not verify_checkpoint_integrity(weights_path, expected_keys=run_best_state_dict.keys()):
        raise RuntimeError(f"Integrity check failed after saving '{to_relative_path(weights_path)}'")
    weights_size = os.path.getsize(weights_path)
    logger.info(f"Saved run weights: '{to_relative_path(weights_path)}' ({weights_size} bytes)")

    # 2. Check metadata to decide if best checkpoint should be updated
    should_update_best = False
    meta_exists = os.path.exists(meta_path)
    best_exists = os.path.exists(best_path)

    if not meta_exists:
        if best_exists:
            logger.warning(
                f"Existing best checkpoint '{to_relative_path(best_path)}' has no metadata file '{to_relative_path(meta_path)}'. "
                f"Overwriting with new run best checkpoint."
            )
        else:
            logger.info(
                f"No previous best checkpoint found for '{exp_name}'. Initializing best checkpoint."
            )
        should_update_best = True
    else:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
        except Exception as e:
            logger.warning(
                f"Could not read metadata at '{to_relative_path(meta_path)}': {e}. "
                f"Overwriting with new run best."
            )
            existing_meta = {}

        prev_seed = existing_meta.get("seed")
        prev_val_split = existing_meta.get("val_split")
        
        # Support both val_acc (higher is better) and val_loss (lower is better)
        if checkpoint_metric == "val_loss":
            default_prev = float("inf")
            prev_best_metric = existing_meta.get("best_metric", existing_meta.get("best_val_loss", default_prev))
            is_better = run_best_val_acc < prev_best_metric
        else:
            default_prev = -float("inf")
            prev_best_metric = existing_meta.get("best_metric", existing_meta.get("best_val_acc", default_prev))
            is_better = run_best_val_acc > prev_best_metric

        # Check comparability
        if (prev_seed is not None and prev_seed != seed) or (prev_val_split is not None and prev_val_split != val_split):
            logger.warning(
                f"Configuration mismatch for '{exp_name}': previous run had seed={prev_seed}, val_split={prev_val_split}; "
                f"current run has seed={seed}, val_split={val_split}. "
                f"Results are incomparable; retaining existing best checkpoint and updating only '{exp_name}_weights.pth'."
            )
            should_update_best = False
        else:
            if is_better:
                logger.info(
                    f"New {checkpoint_metric} ({run_best_val_acc:.4f}) improved over previous best ({prev_best_metric:.4f}). "
                    f"Updating best checkpoint for '{exp_name}'."
                )
                should_update_best = True
            else:
                logger.info(
                    f"New {checkpoint_metric} ({run_best_val_acc:.4f}) did not improve over previous best ({prev_best_metric:.4f}). "
                    f"Retaining existing best checkpoint for '{exp_name}'."
                )
                should_update_best = False

    # 3. Update best checkpoint and metadata if applicable
    if should_update_best:
        atomic_save(run_best_checkpoint_state, best_path, is_torch=True)
        if not verify_checkpoint_integrity(best_path, expected_keys=run_best_state_dict.keys()):
            raise RuntimeError(f"Integrity check failed after saving '{to_relative_path(best_path)}'")
        best_size = os.path.getsize(best_path)
        logger.info(f"Saved best checkpoint: '{to_relative_path(best_path)}' ({best_size} bytes)")

        new_meta = {
            "experiment_name": exp_name,
            "best_val_acc": float(run_best_val_acc),
            "best_metric": float(run_best_val_acc),
            "checkpoint_metric": checkpoint_metric,
            "epoch": int(run_best_epoch),
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "seed": int(seed),
            "val_split": float(val_split)
        }
        atomic_save(new_meta, meta_path, is_torch=False)
        logger.info(f"Saved best metadata: '{to_relative_path(meta_path)}'")

    # 4. Prune old weights
    keep = {f"{exp_name}_best.pth", f"{exp_name}_weights.pth"}
    if keep_latest:
        keep.add(f"{exp_name}_latest.pth")

    pruned = prune_old_weights(checkpoint_dir, exp_name, keep=keep, dry_run=False)

    return {
        "weights_path": weights_path,
        "best_path": best_path,
        "meta_path": meta_path,
        "best_updated": should_update_best,
        "pruned_files": pruned
    }


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None
) -> None:
    """
    Saves training state dictionary to disk atomically.
    If is_best is True, also creates an atomic copy at best_filepath.
    """
    logger = get_logger()
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    atomic_save(state, filepath, is_torch=True)
    size = os.path.getsize(filepath)
    logger.info(f"Saved checkpoint: '{to_relative_path(filepath)}' ({size} bytes)")

    if is_best and best_filepath:
        os.makedirs(os.path.dirname(os.path.abspath(best_filepath)), exist_ok=True)
        atomic_save(state, best_filepath, is_torch=True)
        best_size = os.path.getsize(best_filepath)
        logger.info(f"Saved best checkpoint: '{to_relative_path(best_filepath)}' ({best_size} bytes)")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Loads model state_dict and optional optimizer state_dict from a checkpoint file.
    Returns the metadata dictionary containing epoch, metrics, and config.
    """
    logger = get_logger()
    rel_path = to_relative_path(filepath)

    if not os.path.exists(filepath):
        logger.error(f"Checkpoint file not found: '{rel_path}'")
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

    try:
        checkpoint = torch.load(filepath, map_location=device)
    except Exception as e:
        logger.exception(f"Corrupted or invalid checkpoint file at '{rel_path}': {type(e).__name__}: {e}")
        raise RuntimeError(f"Corrupted or invalid checkpoint file at '{filepath}': {e}")

    if not isinstance(checkpoint, dict):
        msg = f"Expected checkpoint at '{rel_path}' to be a dictionary, got {type(checkpoint).__name__}"
        logger.error(msg)
        raise ValueError(msg)

    if "model_state_dict" not in checkpoint:
        msg = f"Checkpoint at '{rel_path}' is missing 'model_state_dict'"
        logger.error(msg)
        raise KeyError(msg)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
