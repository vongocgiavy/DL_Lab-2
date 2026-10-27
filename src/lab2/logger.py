"""
Structured logging module for Lab 2 with console, training, and error logs.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from typing import Optional

LOGGER_NAME = "lab2"


def get_project_root() -> str:
    """Returns absolute path to project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def to_relative_path(path: str, base_dir: Optional[str] = None) -> str:
    """
    Converts absolute path to relative path with forward slashes for clean logging.
    Avoids leaking absolute system paths, machine names, or user directories.
    """
    if not path:
        return ""
    if base_dir is None:
        base_dir = get_project_root()
    try:
        rel = os.path.relpath(path, base_dir)
        return rel.replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns the 'lab2' logger.
    - Console handler: INFO and above.
    - logs/training.log: INFO and above (RotatingFileHandler, 5MB, 2 backups).
    - logs/errors.log: ERROR and above (RotatingFileHandler, 5MB, 2 backups).

    Safe to call multiple times without duplicating handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    # Close and remove existing handlers to prevent duplicate handlers when called multiple times
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)

    os.makedirs(log_dir, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # 1. Console Handler (INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Training log file (INFO+)
    training_log_path = os.path.join(log_dir, "training.log")
    train_handler = RotatingFileHandler(
        training_log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8"
    )
    train_handler.setLevel(logging.INFO)
    train_handler.setFormatter(formatter)
    logger.addHandler(train_handler)

    # 3. Error log file (ERROR+)
    errors_log_path = os.path.join(log_dir, "errors.log")
    err_handler = RotatingFileHandler(
        errors_log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding="utf-8"
    )
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)

    return logger


def get_logger() -> logging.Logger:
    """Retrieves existing 'lab2' logger or sets up default if none exists."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        return setup_logging()
    return logger
