"""Utility functions for SPRN training and evaluation."""

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
            # YAML reads an unquoted `1:` (training.phases stage keys) as int;
            # setattr needs a str. Readers of these keys (_stage_number) accept both.
            k = str(k)
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            else:
                setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)


def _deep_merge(base, override):
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def compose_config(dataset, method="lpr", config_root="config"):
    """Build a config from three layers: method < family < dataset.

    method  (config/methods/<m>.yaml)   what the model is (LPR is the default)
    family  (config/families/<f>.yaml)  the problem protocol: loss, metric, schedule
    dataset (config/datasets/<d>.yaml)  paths only, plus which family it belongs to

    ``{slots}`` in the method file is replaced by the dataset's slot prefix, and
    checkpoints/logs go to ./checkpoints/<method>/<dataset> (never a legacy dir).
    """
    def read(rel):
        path = os.path.join(config_root, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found (dataset={dataset!r}, method={method!r})")
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    ds = read(os.path.join("datasets", f"{dataset}.yaml"))
    family, slots = ds.pop("family"), ds.pop("slots")
    ds.pop("tag", None)
    meth = read(os.path.join("methods", f"{method}.yaml"))
    meth = yaml.safe_load(yaml.safe_dump(meth).replace("{slots}", slots))
    fam = read(os.path.join("families", f"{family}.yaml"))
    cfg = _deep_merge(_deep_merge(meth, fam), ds)
    cfg.setdefault("training", {})
    cfg["training"]["save_dir"] = f"./checkpoints/{method}/{dataset}"
    cfg["training"]["log_dir"] = f"./logs/{method}/{dataset}"
    return Config(cfg), cfg


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
