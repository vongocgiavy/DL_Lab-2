"""
CLI script to evaluate saved checkpoints on CIFAR-10 test set.
"""

import argparse
import glob
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lab2.config import load_config, load_yaml
from lab2.data import get_dataloaders
from lab2.evaluate import evaluate_checkpoint_file
from lab2.logger import setup_logging, get_logger, to_relative_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on CIFAR-10 test set.")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file (.pth).")
    parser.add_argument("--all", action="store_true", help="Evaluate all *_best.pth checkpoints in checkpoints/ folder.")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config.yaml.")
    parser.add_argument("--smoke-test", action="store_true", help="Use small subset of test data for fast test evaluation.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation dataloader (default: 16).")

    args = parser.parse_args()

    logger = setup_logging()

    yaml_data = load_yaml(args.config)
    global_cfg = yaml_data.get("global", {})
    checkpoints_dir = global_cfg.get("checkpoints_dir", "./checkpoints")
    outputs_dir = global_cfg.get("outputs_dir", "./outputs")

    if not args.checkpoint and not args.all:
        print("Please provide --checkpoint <path> or --all.")
        sys.exit(1)

    if args.all:
        checkpoint_paths = sorted(glob.glob(os.path.join(checkpoints_dir, "*_best.pth")))
        if not checkpoint_paths:
            print(f"No '*_best.pth' checkpoints found in '{checkpoints_dir}'.")
            sys.exit(0)
    else:
        checkpoint_paths = [args.checkpoint]

    # Load test dataloader once
    subset_size = 50 if args.smoke_test else None
    _, _, test_loader, class_names = get_dataloaders(
        data_dir=global_cfg.get("data_dir", "./data"),
        batch_size=args.batch_size,
        val_split=global_cfg.get("val_split", 0.1),
        seed=global_cfg.get("seed", 42),
        image_size=global_cfg.get("image_size", 128),
        num_workers=0,
        subset_size=subset_size
    )

    print("=" * 65)
    print(f"EVALUATING {len(checkpoint_paths)} CHECKPOINT(S) ON CIFAR-10 TEST SET")
    print("=" * 65)

    for ckpt in checkpoint_paths:
        rel_ckpt = to_relative_path(ckpt)
        if not os.path.exists(ckpt):
            logger.error(f"Checkpoint not found: {rel_ckpt}")
            print(f"[Error] Checkpoint not found: {ckpt}")
            continue
        try:
            logger.info(f"Evaluating checkpoint: {rel_ckpt}")
            test_metrics = evaluate_checkpoint_file(
                checkpoint_path=ckpt,
                test_loader=test_loader,
                class_names=class_names,
                outputs_dir=outputs_dir
            )
            logger.info(
                f"Evaluation completed for '{rel_ckpt}': "
                f"test_loss={test_metrics['test_loss']:.4f}, test_acc={test_metrics['test_accuracy']:.2f}%"
            )
        except Exception as e:
            logger.exception(f"Failed to evaluate checkpoint '{rel_ckpt}': {type(e).__name__}: {e}")
            print(f"[Error] Failed to evaluate checkpoint '{rel_ckpt}': {e}")


if __name__ == "__main__":
    main()
