"""Utility functions for HSPRN training and evaluation."""

import os
import random
import logging
import yaml
import torch
import numpy as np


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

class Config:
    """Simple nested namespace from a dict."""

    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)


def load_config(path, default_path=None):
    """Load YAML config, optionally merging with a default config.

    Args:
        path: Path to dataset-specific YAML config.
        default_path: Path to default YAML config (merged first).

    Returns:
        Config object.
    """
    def _merge(base, override):
        """Recursively merge override into base."""
        merged = base.copy()
        for k, v in override.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = _merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    cfg_dict = {}

    if default_path and os.path.exists(default_path):
        with open(default_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}

    with open(path, "r") as f:
        override = yaml.safe_load(f) or {}

    cfg_dict = _merge(cfg_dict, override)
    return Config(cfg_dict)


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------------

def save_checkpoint(state, filepath):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """Load training checkpoint.

    Returns:
        dict with 'epoch' and any other saved state.
    """
    ckpt = torch.load(filepath, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def setup_logger(name, log_dir=None, level=logging.INFO):
    """Setup a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ------------------------------------------------------------------
# AverageMeter
# ------------------------------------------------------------------

class AverageMeter:
    """Computes and stores running average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
