"""
CLI script to run Lab 2 experiments (single experiment or all 4 sequentially).
Supports --smoke-test flag for rapid CPU verification and --device for GPU/CPU selection.
"""

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from lab2.config import get_all_experiment_names, load_config
from lab2.data import get_dataloaders
from lab2.evaluate import evaluate_checkpoint_file
from lab2.logger import setup_logging, get_logger, to_relative_path
from lab2.models import build_model, count_parameters
from lab2.tensorboard_logger import TensorBoardLogger
from lab2.train import run_training
from lab2.utils import set_seed, save_history_json


def run_single_experiment(
    experiment_id: str,
    config_path: str = "./configs/config.yaml",
    smoke_test: bool = False,
    override_epochs: int = None,
    override_batch_size: int = None,
    override_device: str = None
):
    overrides = {}
    if smoke_test:
        overrides["epochs"] = 1
        overrides["batch_size"] = 16
    if override_epochs is not None:
        overrides["epochs"] = override_epochs
    if override_batch_size is not None:
        overrides["batch_size"] = override_batch_size
    if override_device is not None:
        overrides["device"] = override_device

    cfg = load_config(experiment_id=experiment_id, config_path=config_path, overrides=overrides)

    # Smoke test namespace isolation: prevent polluting official checkpoints, metrics, and runs
    if smoke_test:
        cfg.checkpoints_dir = os.path.join(cfg.checkpoints_dir, "smoke")
        cfg.runs_dir = os.path.join(cfg.runs_dir, "smoke")
        cfg.outputs_dir = os.path.join(cfg.outputs_dir, "smoke")
        os.makedirs(cfg.checkpoints_dir, exist_ok=True)
        os.makedirs(cfg.runs_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.outputs_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(cfg.outputs_dir, "figures"), exist_ok=True)

    set_seed(cfg.seed)

    print(f"\n=======================================================")
    print(f"EXPERIMENT RUNNER: {cfg.name}")
    print(f"Description: {cfg.description}")
    print(f"Smoke Test Mode: {smoke_test}")
    print(f"Compute Device: {cfg.device}")
    print(f"=======================================================")

    # Prepare Data
    subset_size = 120 if smoke_test else None
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
        image_size=cfg.image_size,
        num_workers=cfg.num_workers,
        subset_size=subset_size
    )

    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # Build Model
    model = build_model(
        model_name=cfg.model_name,
        method=cfg.method,
        num_classes=cfg.num_classes,
        pretrained=True
    )

    param_counts = count_parameters(model)
    print(
        f"Model: {cfg.model_name} | Method: {cfg.method} | "
        f"Trainable Params: {param_counts['trainable']:,} / {param_counts['total']:,} "
        f"({param_counts['trainable'] / param_counts['total'] * 100:.2f}%)"
    )

    # Initialize TensorBoard Logger
    log_dir = os.path.join(cfg.runs_dir, cfg.name)
    logger = TensorBoardLogger(log_dir=log_dir)

    try:
        # Run Training Loop
        train_results = run_training(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger
        )

        # Save training history JSON
        history_path = os.path.join(cfg.outputs_dir, "metrics", f"{cfg.name}_history.json")
        save_history_json(results=train_results, cfg=cfg, path=history_path)

        # Evaluate Best Checkpoint on Test Set (Practice 2 Step 7)
        best_checkpoint_path = train_results["best_checkpoint_path"]
        print(f"\nEvaluating Best Checkpoint on Unbiased Test Set: {best_checkpoint_path}")
        test_metrics = evaluate_checkpoint_file(
            checkpoint_path=best_checkpoint_path,
            test_loader=test_loader,
            class_names=class_names,
            outputs_dir=cfg.outputs_dir
        )

        return {
            "experiment": cfg.name,
            "checkpoint_metric": cfg.checkpoint_metric,
            "train_results": train_results,
            "test_metrics": test_metrics
        }
    finally:
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="Run Lab 2 CIFAR-10 Transfer Learning / Fine-Tuning Experiments.")
    parser.add_argument("--experiment", type=str, help="Specific experiment ID to run (e.g. resnet18_transfer).")
    parser.add_argument("--all", action="store_true", help="Run all 4 core experiments sequentially.")
    parser.add_argument("--group", type=str, choices=["core", "hparam"], default=None, help="Run all experiments in a specific group (core or hparam).")
    parser.add_argument("--config", type=str, default="./configs/config.yaml", help="Path to config.yaml file.")
    parser.add_argument("--smoke-test", action="store_true", help="Run rapid smoke test on tiny subset (1 epoch) for pipeline verification.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Override compute device ('cpu' or 'cuda').")

    args = parser.parse_args()

    # Initialize logger
    app_logger = setup_logging()

    all_available = get_all_experiment_names(args.config)

    if not args.experiment and not args.all and not args.group:
        print("Please specify --experiment <ID>, --group <core|hparam>, or --all.")
        print(f"Available experiments in {args.config}: {all_available}")
        sys.exit(1)

    if args.experiment:
        if args.experiment not in all_available:
            rel_config = to_relative_path(args.config)
            app_logger.error(
                f"Experiment '{args.experiment}' not found in configuration '{rel_config}'. Available: {all_available}"
            )
            print(f"[Error] Experiment '{args.experiment}' not found in {args.config}. Available: {all_available}")
            sys.exit(1)
        experiments_to_run = [args.experiment]
    elif args.group:
        experiments_to_run = get_all_experiment_names(args.config, group=args.group)
    elif args.all:
        experiments_to_run = get_all_experiment_names(args.config, group="core")

    all_results = []
    failed_experiments = []
    for exp_id in experiments_to_run:
        try:
            res = run_single_experiment(
                experiment_id=exp_id,
                config_path=args.config,
                smoke_test=args.smoke_test,
                override_epochs=args.epochs,
                override_batch_size=args.batch_size,
                override_device=args.device
            )
            all_results.append(res)
        except Exception as e:
            app_logger.exception(
                f"Experiment '{exp_id}' encountered exception: {type(e).__name__}: {e}"
            )
            failed_experiments.append((exp_id, f"{type(e).__name__}: {e}"))

    if all_results and not failed_experiments:
        print("\n" + "=" * 65)
        print("ALL REQUESTED EXPERIMENTS COMPLETED SUCCESSFULLY")
        print("=" * 65)
        for r in all_results:
            exp = r["experiment"]
            metric_name = r.get("checkpoint_metric", "val_acc")
            best_val = r["train_results"]["best_metric"]
            metric_label = "Best Val Acc" if metric_name == "val_acc" else f"Best {metric_name.replace('_', ' ').title()}"
            val_format = f"{best_val:.2f}%" if metric_name == "val_acc" else f"{best_val:.4f}"
            test_acc = r["test_metrics"]["test_accuracy"]
            test_loss = r["test_metrics"]["test_loss"]
            print(f"[{exp}] {metric_label}: {val_format} | Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
        print("=" * 65)
    elif all_results and failed_experiments:
        print("\n" + "=" * 65)
        print("PARTIAL RUN COMPLETED (SOME EXPERIMENTS SUCCEEDED, SOME FAILED)")
        print("=" * 65)
        for r in all_results:
            exp = r["experiment"]
            metric_name = r.get("checkpoint_metric", "val_acc")
            best_val = r["train_results"]["best_metric"]
            metric_label = "Best Val Acc" if metric_name == "val_acc" else f"Best {metric_name.replace('_', ' ').title()}"
            val_format = f"{best_val:.2f}%" if metric_name == "val_acc" else f"{best_val:.4f}"
            test_acc = r["test_metrics"]["test_accuracy"]
            test_loss = r["test_metrics"]["test_loss"]
            print(f"[{exp}] {metric_label}: {val_format} | Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}")
        print("=" * 65)

    if failed_experiments:
        print("\n" + "=" * 65)
        print("EXPERIMENT RUN SUMMARY: FAILED EXPERIMENTS")
        print("=" * 65)
        for exp_id, err_msg in failed_experiments:
            print(f"FAILED: [{exp_id}] -> {err_msg}")
        print("=" * 65)
        sys.exit(1)


if __name__ == "__main__":
    main()
