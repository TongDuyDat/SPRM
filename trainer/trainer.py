"""
HSPRN Trainer — training loop with ZSL/GZSL evaluation.

This trainer works with the pre-extracted feature pipeline (xlsa17).
It handles:
  - Training on seen-class features
  - Periodic ZSL and GZSL evaluation
  - Checkpointing (best H metric)
  - Learning rate scheduling
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tools.utils import AverageMeter, save_checkpoint, setup_logger
from tools.evaluator import ZSLEvaluator


class Trainer:
    """Training loop for HSPRN on ZSL benchmarks."""

    def __init__(self, cfg, model, criterion, train_dataset, test_seen_dataset,
                 test_unseen_dataset, attribute, seen_classes, unseen_classes,
                 device="cuda"):
        """
        Args:
            cfg: Config object.
            model: HSPRN model instance.
            criterion: HSPRNLoss instance.
            train_dataset: ZSLDataset (trainval split).
            test_seen_dataset: ZSLDataset (test_seen split).
            test_unseen_dataset: ZSLDataset (test_unseen split).
            attribute: (C, attr_dim) class attribute matrix (torch.Tensor).
            seen_classes: np.array of seen class indices.
            unseen_classes: np.array of unseen class indices.
            device: 'cuda' or 'cpu'.
        """
        self.cfg = cfg
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.device = device

        self.attribute = attribute.to(device)
        self.seen_classes = seen_classes
        self.unseen_classes = unseen_classes

        # Data loaders
        bs = getattr(cfg.data, "batch_size", 64)
        nw = getattr(cfg.data, "num_workers", 4)

        self.train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=True, drop_last=True,
        )
        self.test_seen_loader = DataLoader(
            test_seen_dataset, batch_size=bs * 2, shuffle=False,
            num_workers=nw, pin_memory=True,
        )
        self.test_unseen_loader = DataLoader(
            test_unseen_dataset, batch_size=bs * 2, shuffle=False,
            num_workers=nw, pin_memory=True,
        )

        # Optimizer
        lr = getattr(cfg.training, "lr", 1e-4)
        wd = getattr(cfg.training, "weight_decay", 1e-4)
        opt_name = getattr(cfg.training, "optimizer", "adam")

        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Scheduler
        epochs = getattr(cfg.training, "epochs", 50)
        sched_name = getattr(cfg.training, "scheduler", "cosine")
        warmup = getattr(cfg.training, "warmup_epochs", 5)

        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs - warmup, eta_min=1e-6
            )
        elif sched_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, epochs // 3), gamma=0.1
            )
        else:
            self.scheduler = None

        # Evaluator
        self.evaluator = ZSLEvaluator(seen_classes, unseen_classes)

        # Training state
        self.epochs = epochs
        self.eval_freq = getattr(cfg.training, "eval_freq", 1)
        self.save_dir = getattr(cfg.training, "save_dir", "./checkpoints")
        self.best_H = 0.0
        self.best_epoch = 0

        # Logger
        log_dir = getattr(cfg.training, "log_dir", "./logs")
        dataset_name = getattr(cfg.data, "dataset", "ZSL")
        self.logger = setup_logger(f"HSPRN-{dataset_name}", log_dir)

    def _prepare_semantic_input(self, class_indices=None):
        """Prepare semantic input for the model.

        Returns the class-level semantic tensor used by PriSematicConstructor.
        This can be the XLSA17 attribute matrix or cached CLIP/BERT embeddings
        produced offline by tools/extract_semantic_embeddings.py.

        Override this method only if the semantic tensor is stored differently.
        """
        if class_indices is not None:
            return self.attribute[class_indices]
        return self.attribute

    def train_one_epoch(self, epoch):
        """Train for one epoch on seen classes."""
        self.model.train()
        loss_meter = AverageMeter()
        loss_cls_meter = AverageMeter()

        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Get seen-class semantic embeddings as input
            seen_attr = self._prepare_semantic_input(self.seen_classes)

            # Forward: model expects (text_feature, vision_feature)
            # For feature-based training, vision_feature is the pre-extracted feature
            out = self.model(seen_attr, features)
            if hasattr(out, 'scores'):
                s_c = out.scores
                # Optional full loss
                if getattr(self.cfg.model.loss, 'use_full', False):
                    total_loss, loss_dict = self.criterion(
                        s_c, mapped_labels,
                        v_tilde=out.v_embed,
                        phi_t=out.phi_t,
                        p_sem=out.p_sem,
                        p_ref=out.p_ref,
                        p_sem_unnorm=out.p_sem_unnorm,
                    )
                    loss = total_loss
                else:
                    tau = getattr(self.cfg.model.loss, "temperature", 0.07)
                    loss = nn.functional.cross_entropy(s_c / tau, mapped_labels)
            else:
                s_c = out
                tau = getattr(self.cfg.model.loss, "temperature", 0.07)
                loss = nn.functional.cross_entropy(s_c / tau, mapped_labels)

            # Remap labels to seen-class contiguous indices for cross-entropy
            label_map = {int(c): i for i, c in enumerate(self.seen_classes)}
            mapped_labels = torch.tensor(
                [label_map[int(l)] for l in labels.cpu()],
                dtype=torch.long, device=self.device
            )

            # Compute loss
            # For simplicity, use cross-entropy on compatibility scores
            tau = getattr(self.cfg.model.loss, "temperature", 0.07)
            loss = nn.functional.cross_entropy(s_c / tau, mapped_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), features.size(0))
            loss_cls_meter.update(loss.item(), features.size(0))

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": loss_meter.avg,
            "loss_cls": loss_cls_meter.avg,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def _compute_scores(self, loader, class_indices):
        """Compute compatibility scores for all samples in a loader.

        Args:
            loader: DataLoader.
            class_indices: np.array of class indices to score against.

        Returns:
            all_scores: (N, C_total) tensor. Only columns in class_indices
                        are filled; others are -inf.
            all_labels: (N,) tensor of original labels.
        """
        self.model.eval()
        all_scores_list = []
        all_labels_list = []

        attr = self._prepare_semantic_input(class_indices)
        num_total_classes = self.attribute.shape[0]

        for features, labels in loader:
            features = features.to(self.device)

            # Model returns (B, len(class_indices)) scores
            scores_subset = self.model(attr, features)

            # Map back to full score matrix (B, C_total) with -inf for unscored
            scores_full = torch.full(
                (features.size(0), num_total_classes),
                -float("inf"), device=self.device
            )
            for i, c in enumerate(class_indices):
                scores_full[:, c] = scores_subset[:, i]

            all_scores_list.append(scores_full.cpu())
            all_labels_list.append(labels)

        return torch.cat(all_scores_list, dim=0), torch.cat(all_labels_list, dim=0)

    def evaluate(self):
        """Run ZSL and GZSL evaluation.

        Returns:
            dict with keys: 'zsl_T1', 'gzsl_u', 'gzsl_s', 'gzsl_H'
        """
        all_classes = self.evaluator.all_classes

        # GZSL: score all test samples against ALL classes
        scores_seen, labels_seen = self._compute_scores(
            self.test_seen_loader, all_classes
        )
        scores_unseen, labels_unseen = self._compute_scores(
            self.test_unseen_loader, all_classes
        )

        # ZSL: conventional — only unseen classes
        zsl_T1 = self.evaluator.eval_zsl(scores_unseen, labels_unseen)

        # GZSL
        u, s, H = self.evaluator.eval_gzsl(
            scores_seen, labels_seen, scores_unseen, labels_unseen
        )

        return {
            "zsl_T1": zsl_T1 * 100,
            "gzsl_u": u * 100,
            "gzsl_s": s * 100,
            "gzsl_H": H * 100,
        }

    def train(self):
        """Full training loop."""
        dataset_name = getattr(self.cfg.data, "dataset", "ZSL")
        self.logger.info(f"=== Training HSPRN on {dataset_name} ===")
        self.logger.info(f"Seen classes: {len(self.seen_classes)}, "
                         f"Unseen classes: {len(self.unseen_classes)}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}, "
                         f"Test seen: {len(self.test_seen_loader.dataset)}, "
                         f"Test unseen: {len(self.test_unseen_loader.dataset)}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_stats = self.train_one_epoch(epoch)
            elapsed = time.time() - t0

            log_msg = (
                f"Epoch [{epoch}/{self.epochs}] "
                f"Loss: {train_stats['loss']:.4f} "
                f"LR: {train_stats['lr']:.6f} "
                f"Time: {elapsed:.1f}s"
            )

            if epoch % self.eval_freq == 0 or epoch == self.epochs:
                eval_stats = self.evaluate()
                log_msg += (
                    f" | ZSL T1: {eval_stats['zsl_T1']:.1f}% "
                    f"| GZSL u: {eval_stats['gzsl_u']:.1f}% "
                    f"s: {eval_stats['gzsl_s']:.1f}% "
                    f"H: {eval_stats['gzsl_H']:.1f}%"
                )

                # Save best model by GZSL H
                if eval_stats["gzsl_H"] > self.best_H:
                    self.best_H = eval_stats["gzsl_H"]
                    self.best_epoch = epoch
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_H": self.best_H,
                            "eval_stats": eval_stats,
                        },
                        os.path.join(self.save_dir, dataset_name, "best.pth"),
                    )
                    log_msg += " *BEST*"

            self.logger.info(log_msg)

        self.logger.info(
            f"=== Done. Best H: {self.best_H:.1f}% at epoch {self.best_epoch} ==="
        )

        return self.best_H, self.best_epoch
