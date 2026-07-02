"""
HSPRN Trainer — training loop with ZSL/GZSL evaluation.

This trainer supports the xlsa17 feature pipeline and the raw-image pipeline.
It handles:
  - Training on seen-class images or features
  - Semantic input from raw class-description records or cached tensors
  - Periodic ZSL and GZSL evaluation
  - Checkpointing (best H metric)
  - Learning rate scheduling
"""

import os
import time
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
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
        self.num_total_classes = int(attribute.shape[0])

        self.semantic_source = getattr(
            getattr(cfg, "data", object()),
            "semantic_source",
            "attribute",
        )
        self.class_semantics = None
        if self.semantic_source == "class_desc":
            source_dataset = self._unwrap_dataset(train_dataset)
            if not hasattr(source_dataset, "get_class_semantics"):
                raise ValueError(
                    "semantic_source='class_desc' requires a dataset with "
                    "get_class_semantics()."
                )
            self.class_semantics = source_dataset.get_class_semantics()
            if len(self.class_semantics) != self.num_total_classes:
                raise ValueError(
                    "Class semantic record count mismatch: got "
                    f"{len(self.class_semantics)}, expected {self.num_total_classes}."
                )

        # Data loaders
        bs = getattr(cfg.data, "batch_size", 64)
        nw = getattr(cfg.data, "num_workers", 4)

        self.train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=device == "cuda",
            drop_last=getattr(cfg.data, "drop_last", False),
        )
        self.test_seen_loader = DataLoader(
            test_seen_dataset, batch_size=bs * 2, shuffle=False,
            num_workers=nw, pin_memory=device == "cuda",
        )
        self.test_unseen_loader = DataLoader(
            test_unseen_dataset, batch_size=bs * 2, shuffle=False,
            num_workers=nw, pin_memory=device == "cuda",
        )

        # Optimizer
        lr = getattr(cfg.training, "lr", 1e-4)
        wd = getattr(cfg.training, "weight_decay", 1e-4)
        opt_name = getattr(cfg.training, "optimizer", "adam")
        trainable_parameters = [param for param in model.parameters() if param.requires_grad]
        if not trainable_parameters:
            raise ValueError("Model has no trainable parameters after applying freeze settings.")

        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(trainable_parameters, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=wd)

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

    @staticmethod
    def _unwrap_dataset(dataset):
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return dataset

    def _prepare_semantic_input(self, class_indices=None):
        """Prepare semantic input for the model.

        In class_desc mode this returns raw name_c/attr_c/hier_c/desc_c records
        from the dataset loader. Otherwise it returns class-level tensors:
        XLSA17 attributes or explicit cached semantic embeddings.
        """
        if self.class_semantics is not None:
            indices = (
                range(len(self.class_semantics))
                if class_indices is None else class_indices
            )
            return {
                str(int(class_index)): self.class_semantics[int(class_index)]
                for class_index in indices
            }

        if class_indices is not None:
            return {
                str(int(class_index)): self.attribute[int(class_index)]
                for class_index in class_indices
            }
        return {
            str(class_index): self.attribute[class_index]
            for class_index in range(self.attribute.shape[0])
        }

    def train_one_epoch(self, epoch):
        """Train for one epoch on seen classes."""
        self.model.train()
        loss_meter = AverageMeter()
        component_meters = {}

        label_map = {int(c): i for i, c in enumerate(self.seen_classes)}

        progress = tqdm(
            self.train_loader,
            desc=f"Train {epoch}/{self.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            features, labels = batch[:2]
            features = features.to(self.device)
            labels = labels.to(self.device)
            mapped_labels = torch.tensor(
                [label_map[int(l)] for l in labels.cpu()],
                dtype=torch.long, device=self.device
            )

            # Get seen-class semantic embeddings as input
            seen_attr = self._prepare_semantic_input(self.seen_classes)

            # Forward: model expects (text_feature, vision_feature)
            out = self.model(seen_attr, features)
            if hasattr(out, 'scores'):
                s_c = out.scores
                loss, loss_dict = self.criterion(
                    s_c, mapped_labels,
                    v_tilde=out.v_embed,
                    p_sem=out.p_sem,
                    p_ref=out.p_ref,
                )
            else:
                raise ValueError(
                    "Full HSPRN objective requires model.loss.return_full_output=true "
                    "so the trainer can access v_tilde, p_sem, and p_ref."
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), features.size(0))
            for name, value in loss_dict.items():
                component_meters.setdefault(name, AverageMeter()).update(value, features.size(0))
            progress.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                cls=f"{component_meters['loss_cls'].avg:.4f}",
                align=f"{component_meters['loss_align'].avg:.4f}",
                pres=f"{component_meters['loss_pres'].avg:.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                refresh=False,
            )

        if self.scheduler is not None:
            self.scheduler.step()

        stats = {
            "loss": loss_meter.avg,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        stats.update({name: meter.avg for name, meter in component_meters.items()})
        return stats

    @torch.no_grad()
    def _compute_scores(self, loader, class_indices, description):
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
        num_total_classes = self.num_total_classes

        for batch in tqdm(
            loader,
            desc=description,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        ):
            features, labels = batch[:2]
            features = features.to(self.device)

            # Model returns (B, len(class_indices)) scores
            out = self.model(attr, features)
            scores_subset = out.scores if hasattr(out, "scores") else out

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
            self.test_seen_loader, all_classes, "Eval seen"
        )
        scores_unseen, labels_unseen = self._compute_scores(
            self.test_unseen_loader, all_classes, "Eval unseen"
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

    def evaluate_calibrated(self):
        """Run ZSL and GZSL evaluation with calibrated stacking.

        Returns:
            dict with keys: 'zsl_T1', 'gzsl_u', 'gzsl_s', 'gzsl_H',
                            'cal_u', 'cal_s', 'cal_H', 'cal_gamma'
        """
        all_classes = self.evaluator.all_classes

        scores_seen, labels_seen = self._compute_scores(
            self.test_seen_loader, all_classes, "Eval seen"
        )
        scores_unseen, labels_unseen = self._compute_scores(
            self.test_unseen_loader, all_classes, "Eval unseen"
        )

        # ZSL
        zsl_T1 = self.evaluator.eval_zsl(scores_unseen, labels_unseen)

        # GZSL (standard)
        u, s, H = self.evaluator.eval_gzsl(
            scores_seen, labels_seen, scores_unseen, labels_unseen
        )

        # GZSL (calibrated stacking)
        cal_u, cal_s, cal_H, cal_gamma = self.evaluator.eval_gzsl_calibrated(
            scores_seen, labels_seen, scores_unseen, labels_unseen
        )

        return {
            "zsl_T1": zsl_T1 * 100,
            "gzsl_u": u * 100,
            "gzsl_s": s * 100,
            "gzsl_H": H * 100,
            "cal_u": cal_u * 100,
            "cal_s": cal_s * 100,
            "cal_H": cal_H * 100,
            "cal_gamma": cal_gamma,
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
                eval_stats = self.evaluate_calibrated()
                log_msg += (
                    f" | ZSL T1: {eval_stats['zsl_T1']:.1f}% "
                    f"| GZSL u: {eval_stats['gzsl_u']:.1f}% "
                    f"s: {eval_stats['gzsl_s']:.1f}% "
                    f"H: {eval_stats['gzsl_H']:.1f}%"
                    f" | Cal u: {eval_stats['cal_u']:.1f}% "
                    f"s: {eval_stats['cal_s']:.1f}% "
                    f"H: {eval_stats['cal_H']:.1f}% "
                    f"γ: {eval_stats['cal_gamma']:.2f}"
                )

                # Save best model by calibrated GZSL H
                if eval_stats["cal_H"] > self.best_H:
                    self.best_H = eval_stats["cal_H"]
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
            f"=== Done. Best Cal H: {self.best_H:.1f}% at epoch {self.best_epoch} ==="
        )

        return self.best_H, self.best_epoch

