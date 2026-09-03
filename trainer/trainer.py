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
import numpy as np
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

        # v10 L_attn: GT attribute matrix [C_total, n_attr], loaded ONCE from
        # the prepared metadata. This is NOT self.attribute — with
        # semantic_source='embedding' that tensor holds the precomputed CLIP
        # phi [C_total, 512]. Rows are indexed by GLOBAL class ids (the raw
        # dataset labels). Optional override via data.gt_attribute_path.
        data_cfg = getattr(cfg, "data", object())
        gt_attr_path = getattr(data_cfg, "gt_attribute_path", None)
        if gt_attr_path is None:
            dataset_name = getattr(data_cfg, "dataset", None)
            prepared_root = getattr(data_cfg, "prepared_root", "./prepared_datasets")
            if dataset_name is not None:
                candidate = os.path.join(
                    prepared_root, str(dataset_name), "metadata", "attributes.npy")
                if os.path.exists(candidate):
                    gt_attr_path = candidate
        self.gt_attribute = None
        if gt_attr_path is not None:
            gt = torch.from_numpy(np.load(gt_attr_path)).float().to(device)
            if gt.shape[0] != self.num_total_classes:
                raise ValueError(
                    f"GT attribute matrix has {gt.shape[0]} rows, expected "
                    f"{self.num_total_classes} classes (from {gt_attr_path})."
                )
            self.gt_attribute = gt

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
            # Plan 3: hard-fail incomplete after-LLM fields when unify is on.
            experiment = getattr(cfg, "experiment", None)
            if experiment is not None and bool(getattr(experiment, "unify_after_llm", False)):
                from tools.semantic_policy import assert_full_after_llm_semantics

                assert_full_after_llm_semantics(
                    self.class_semantics,
                    num_total_classes=self.num_total_classes,
                    context="trainer class_semantics",
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
        # Probe mode (training.probe=true): no training updates at all — the
        # model is evaluated exactly once as initialized (R0 geometry probe).
        self.probe_mode = bool(getattr(cfg.training, "probe", False))
        lr = getattr(cfg.training, "lr", 1e-4)
        wd = getattr(cfg.training, "weight_decay", 1e-4)
        opt_name = getattr(cfg.training, "optimizer", "adam")
        trainable_parameters = [param for param in model.parameters() if param.requires_grad]
        if not trainable_parameters and not self.probe_mode:
            raise ValueError("Model has no trainable parameters after applying freeze settings.")

        if self.probe_mode:
            self.optimizer = None
            self.scheduler = None
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(trainable_parameters, lr=lr, weight_decay=wd, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=wd)

        # Scheduler
        epochs = getattr(cfg.training, "epochs", 50)
        sched_name = getattr(cfg.training, "scheduler", "cosine")
        warmup = getattr(cfg.training, "warmup_epochs", 5)

        if self.probe_mode:
            pass
        elif sched_name == "cosine":
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
        self.best_H = -1.0
        self.best_epoch = 0
        # v9: T1 (ZSL unseen) là chỉ tiêu quyết định, nên song song với best.pth
        # (chọn theo Cal H, giữ nguyên giao thức so sánh với v8) ta lưu thêm
        # best_t1.pth chọn theo T1.
        self.best_T1 = -1.0
        self.best_T1_epoch = 0
        self.last_eval_stats = None
        # v9.6: lưu trọng số epoch cuối (final.pth) — checkpoint chính thức về
        # giao thức (best/best_t1 chọn theo chỉ số trên tập test).
        self.save_final = True

        # Logger
        log_dir = getattr(cfg.training, "log_dir", "./logs")
        dataset_name = getattr(cfg.data, "dataset", "ZSL")
        self.logger = setup_logger(f"HSPRN-{dataset_name}", log_dir)

        # CE over all classes (Path-B): include unseen text prototypes in s_c
        # so the classifier gets an explicit negative push on unseen logits.
        self.ce_all_classes = bool(getattr(cfg.training, "ce_all_classes", False))
        if self.ce_all_classes:
            self._train_classes = list(range(self.num_total_classes))
            self._label_map = None  # identity: dataset labels are global ids
            self.logger.info(
                f"CE over all {self.num_total_classes} classes "
                "(unseen prototypes as in-batch negatives)."
            )
        else:
            self._train_classes = self.seen_classes
            self._label_map = {int(c): i for i, c in enumerate(self.seen_classes)}

        # Frozen text encoder: encode class semantics once → reuse every batch.
        # Cache stores phi_t (pre-Ws); Ws stays trainable and still runs per step.
        self._phi_cache = None
        self._maybe_build_frozen_phi_cache()

        # Ws identity anchor: penalty ws_anchor * ||Ws - I||_F^2 (counters the
        # fitted-Ws seen-shortcut found in the E3 ablation experiments).
        semantic_cfg = getattr(getattr(cfg, "model", object()), "semantic", None)
        self.ws_anchor = float(getattr(semantic_cfg, "ws_anchor", 0.0)) if semantic_cfg is not None else 0.0
        self._ws_eye = None
        if self.ws_anchor > 0:
            ws = getattr(getattr(model, "psc", None), "Ws", None)
            if ws is None:
                raise ValueError("ws_anchor requires a trainable model.psc.Ws module.")
            dim = ws.weight.shape[-1]
            self._ws_eye = torch.eye(dim, device=self.device)
            self.logger.info(
                f"Ws identity anchor ON (weight={self.ws_anchor}g, "
                f"init=identity {dim}x{dim})."
            )

        # Wv identity anchor: penalty wv_anchor * ||Wv - Wv_init||_F^2 over the
        # aggregator projector (aligned-geometry countermeasure; init stored by
        # MultiFeatureAggregator when wv_init='identity').
        mfa = getattr(model, "mfa", None)
        self.wv_anchor = float(getattr(mfa, "wv_anchor", 0.0)) if mfa is not None else 0.0
        self._wv_anchor_state = getattr(mfa, "_wv_anchor_state", None) if self.wv_anchor > 0 else None
        if self.wv_anchor > 0:
            if self._wv_anchor_state is None:
                raise ValueError(
                    "wv_anchor>0 requires model.aggregator.wv_init='identity'."
                )
            self.logger.info(f"Wv identity anchor ON (weight={self.wv_anchor}).")

        # Semantic-structure-preserving constraint (SSPPC): keeps W_s nearly
        # isometric on the full class manifold (seen + unseen) by anchoring the
        # Gram matrix of normalized(Ws phi) to that of the frozen phi cache.
        loss_cfg = getattr(getattr(cfg, "model", object()), "loss", None)
        self.struct_weight = float(getattr(loss_cfg, "struct_weight", 0.0)) if loss_cfg is not None else 0.0
        self._gram_target = None
        if self.struct_weight > 0:
            if self._phi_cache is None:
                raise ValueError(
                    "struct_weight requires the frozen phi cache "
                    "(semantic backbone must be frozen in class_desc mode)."
                )
            with torch.no_grad():
                phi_n = torch.nn.functional.normalize(
                    self._phi_cache.to(self.device), dim=-1
                )
                self._gram_target = phi_n @ phi_n.T
            self.logger.info(
                f"Sematic-structure anchor ON (weight={self.struct_weight}, "
                f"Gram target over {self._gram_target.shape[0]} classes)."
            )

        # Align over all classes (E9): CE keeps seen-only labels, but L_align
        # becomes a 200-class softmax so unseen prototypes act as negatives and
        # the visual side (mfa) learns the full class constellation.
        self.align_all_classes = bool(getattr(loss_cfg, "align_all_classes", False)) if loss_cfg is not None else False
        if self.align_all_classes:
            if self._phi_cache is None:
                raise ValueError("align_all_classes requires the frozen phi cache.")
            # Replace the seen-only align term (lambda2) with a 200-class one.
            self._align_weight = self.criterion.lambda2
            self.criterion.lambda2 = 0.0
            self.logger.info(
                "Align ALL classes ON (unseen prototypes as align negatives)."
            )

        # Pair-based prototype separation (RCNn, Đề xuất 1): hinge loss that
        # pushes ALL class prototypes (seen + unseen) apart in the shared space,
        # L_sep = mean_{c != c'} max(0, cos(p_c, p_c') - sep_margin).
        # Gradient flows only through Ws (phi cache is frozen), giving unseen
        # prototypes an explicit repulsive force against seen ones.
        self.sep_weight = float(getattr(loss_cfg, "sep_weight", 0.0)) if loss_cfg is not None else 0.0
        self.sep_margin = float(getattr(loss_cfg, "sep_margin", 0.0)) if loss_cfg is not None else 0.0
        if self.sep_weight > 0:
            if self._phi_cache is None:
                raise ValueError(
                    "sep_weight requires the frozen phi cache "
                    "(semantic backbone must be frozen in class_desc mode)."
                )
            self.logger.info(
                f"Prototype separation loss ON (weight={self.sep_weight}, "
                f"margin={self.sep_margin}, "
                f"over {self._phi_cache.shape[0]} classes)."
            )

        # Transfer-alignment (L_trans): distills the frozen phi-space class
        # constellation (seen + unseen) into the visual space using the REAL
        # pooled mean of each seen class. lambda_trans>0 activates it; the
        # teacher and pooled means are cached once in __init__.
        self._lambda_trans = float(getattr(loss_cfg, "lambda_trans", 0.0)) if loss_cfg is not None else 0.0
        # v11: các ràng buộc hình học mới (Báo cáo u-collapse, Fix 1-4).
        self._lambda_zs = float(getattr(loss_cfg, "lambda_zs", 0.0)) if loss_cfg is not None else 0.0
        self._lambda_rel = float(getattr(loss_cfg, "lambda_rel", 0.0)) if loss_cfg is not None else 0.0
        self._lambda_tan = float(getattr(loss_cfg, "lambda_tan", 0.0)) if loss_cfg is not None else 0.0
        # v12: Tri-Loss mode check
        self._is_tri_loss = getattr(self.criterion, "loss_mode", "legacy") == "tri_loss"
        # v13: Phased Training configuration (docs/KE_HOACH_PHASED_TRAINING_HSPRN_V13.md)
        self.phased_training = bool(getattr(cfg.training, "phased_training", False)) if hasattr(cfg, "training") else False
        self.phase1_epochs = int(getattr(cfg.training, "phase1_epochs", 15))
        self.phase2_epochs = int(getattr(cfg.training, "phase2_epochs", 40))
        self._current_phase = None
        # v12 (L_gen): synthetic-unseen gradient source. An offline bank of
        # per-unseen-class backbone statistics (tools/build_synthetic_bank.py:
        # protocol-safe trainval stats transferred from seen classes via the
        # frozen phi geometry) is sampled each step and pushed through the
        # LIVE inference path against the full-class bank — the first
        # discriminative gradient for the unseen columns that flows through
        # the exact scoring path used at eval.
        self._lambda_gen = float(getattr(loss_cfg, "lambda_gen", 0.0)) if loss_cfg is not None else 0.0
        self._gen_samples = int(getattr(loss_cfg, "gen_samples_per_step", 64)) if loss_cfg is not None else 64
        self._syn = None
        if self._lambda_gen > 0:
            if self._phi_cache is None:
                raise ValueError("lambda_gen>0 requires the frozen phi cache.")
            from types import SimpleNamespace
            bank = torch.load(
                str(getattr(loss_cfg, "gen_bank_path",
                            "./pre-trained/cub_v12_synbank_b16.pt")),
                map_location="cpu",
            )
            bank_unseen = [int(c) for c in bank["unseen_classes"]]
            if sorted(bank_unseen) != sorted(int(c) for c in self.unseen_classes):
                raise ValueError(
                    "gen bank unseen classes do not match this dataset's "
                    "unseen classes — rebuild the bank for this config."
                )
            n_unseen = len(bank_unseen)
            expected = {
                "cls_mean": (n_unseen, 512), "cls_std": (n_unseen, 512),
                "gap_mean": (n_unseen, 3, 768), "gap_std": (n_unseen, 3, 768),
                "tok_mean_h": (n_unseen, 197, 768),
            }
            for key, shape in expected.items():
                got = tuple(bank[key].shape)
                if got != shape:
                    raise ValueError(
                        f"gen bank '{key}' has shape {got}, expected {shape}."
                    )
                if not torch.isfinite(bank[key]).all():
                    raise ValueError(f"gen bank '{key}' has non-finite values.")
            self._syn = SimpleNamespace(
                cls_mean=bank["cls_mean"].to(self.device),
                cls_std=bank["cls_std"].to(self.device),
                gap_mean=bank["gap_mean"].to(self.device),
                gap_std=bank["gap_std"].to(self.device),
                tok_mean_h=bank["tok_mean_h"].to(self.device),
                unseen_global_ids=torch.as_tensor(
                    bank_unseen, dtype=torch.long, device=self.device),
            )
            est = bank.get("estimator", {})
            holdout = est.get("holdout_metrics", {})
            self.logger.info(
                "Synthetic-unseen bank ON (lambda_gen="
                f"{self._lambda_gen:.2f}, B_syn={self._gen_samples}, "
                f"{n_unseen} unseen classes, estimator={est.get('kind')} "
                f"{est.get('params')}, holdout cos_mean="
                f"{holdout.get('cos_mean', float('nan')):.4f}, nn_top1="
                f"{holdout.get('nn_top1', float('nan')):.4f})"
            )
        self._level_means = None
        self._teacher = None
        if self._lambda_zs > 0 or self._lambda_rel > 0 or self._lambda_tan > 0:
            if self._phi_cache is None:
                raise ValueError(
                    "lambda_zs/lambda_rel/lambda_tan require the frozen phi cache."
                )
        if self._lambda_trans > 0:
            if self._phi_cache is None:
                raise ValueError(
                    "lambda_trans>0 requires the frozen phi cache "
                    "(semantic backbone must be frozen in class_desc mode)."
                )
            self._build_trans_caches()
            n_levels = (
                len(self._level_means["tokens"])
                if isinstance(self._level_means, dict)
                else len(self._level_means)
                if isinstance(self._level_means, list)
                else 1
            )
            self.logger.info(
                f"Transfer-alignment ON (lambda_trans={self._lambda_trans}, "
                f"tau_teacher={self.criterion.tau_teacher}, "
                f"levels={n_levels}, "
                f"teacher={tuple(self._teacher.shape)})."
            )

        # v11 Fix 2a: IPR cfg_mode='cfg' needs the trainval null condition
        # before the first optimizer step (r(∅) subtraction is part of the
        # forward path, so it must exist from batch one).
        ipr = getattr(self.model, "ipr", None)
        if ipr is not None and getattr(ipr, "cfg_mode", "legacy") == "cfg":
            self._build_null_condition()

    @staticmethod
    def _unwrap_dataset(dataset):
        while isinstance(dataset, Subset):
            dataset = dataset.dataset
        return dataset

    def _build_trans_caches(self):
        """Cache per-class pooled level means and the frozen teacher.

        1) Pooled means: for every seen class, accumulate GAP'd multi-level
           features over the trainval split only (protocol-safe; no test
           images), frozen backbone, one full pass with the eval transform.
           When features are pre-extracted 2D tensors, a single mean per
           class is stored (lists collapsed).
        2) Teacher: _teacher[s] = softmax(cos(phi_s, phi_all)/tau_teacher)
           restricted to the semantic similarity constellation of all classes
           (seen + unseen), frozen and detached.
        """
        import torch.nn.functional as F

        device = self.device
        self.model.eval()
        # Protocol-safe: class means are pooled from trainval only (never
        # from test splits).
        loaders = [self.train_loader]
        seen_set = set(int(c) for c in self.seen_classes)
        row_of = {int(c): i for i, c in enumerate(self.seen_classes)}

        # Deterministic means: temporarily use the eval (no-augment) transform.
        ds = self._unwrap_dataset(self.train_loader.dataset)
        orig_transform = getattr(ds, "transform", None)
        if hasattr(ds, "_default_transform"):
            ds.transform = ds._default_transform("test")

        sums, counts, cls_sums = None, None, None
        for loader in loaders:
            for batch in loader:
                features, labels = batch[:2]
                features = features.to(device)
                with torch.no_grad():
                    pooled_levels = self.model.vis_encoder(features)
                cls_batch = None
                if isinstance(pooled_levels, dict):
                    # v10 anchored path: GAP the token levels for the
                    # multi-level means and keep the bit-exact cls column so
                    # mfa receives the same {"tokens", "cls"} input as at
                    # train time.
                    cls_batch = pooled_levels["cls"].float()
                    pooled_levels = [
                        self.model.mfa.gap(t).float()
                        for t in pooled_levels["tokens"]
                    ]
                elif isinstance(pooled_levels, (list, tuple)) and not torch.is_tensor(pooled_levels):
                    pooled_levels = [
                        self.model.mfa.gap(level).float() for level in pooled_levels
                    ]
                else:
                    pooled_levels = self.model.mfa.gap(pooled_levels).float()

                if sums is None:
                    num_seen = len(self.seen_classes)
                    if isinstance(pooled_levels, list):
                        sums = [
                            torch.zeros(num_seen, level.shape[-1], device=device)
                            for level in pooled_levels
                        ]
                    else:
                        sums = torch.zeros(
                            num_seen, pooled_levels.shape[-1], device=device
                        )
                    if cls_batch is not None:
                        cls_sums = torch.zeros(
                            num_seen, cls_batch.shape[-1], device=device
                        )
                    counts = torch.zeros(num_seen, device=device)

                for label in labels.unique().tolist():
                    if int(label) not in seen_set:
                        continue
                    row = row_of[int(label)]
                    mask = labels == label
                    n = float(mask.sum())
                    if isinstance(sums, list):
                        for i, level in enumerate(pooled_levels):
                            sums[i][row] += level[mask].sum(dim=0)
                    else:
                        sums[row] += pooled_levels[mask].sum(dim=0)
                    if cls_sums is not None:
                        cls_sums[row] += cls_batch[mask].sum(dim=0)
                    counts[row] += n

        if sums is None:
            raise RuntimeError("Transfer-alignment cache found no trainval batches.")
        if orig_transform is not None:
            ds.transform = orig_transform
        empty = int((counts == 0).sum())
        if empty:
            raise RuntimeError(
                f"Transfer-alignment cache: {empty} seen classes have no images."
            )

        if cls_sums is not None:
            self._level_means = {
                "tokens": [
                    (level_sum / counts.unsqueeze(1)).detach()
                    for level_sum in sums
                ],
                "cls": (cls_sums / counts.unsqueeze(1)).detach(),
            }
        elif isinstance(sums, list):
            self._level_means = [
                (level_sum / counts.unsqueeze(1)).detach() for level_sum in sums
            ]
        else:
            self._level_means = (sums / counts.unsqueeze(1)).detach()

        # Frozen teacher: softmax over the phi-space cosine constellation of
        # ALL classes (seen + unseen), rows restricted to seen classes.
        phi = F.normalize(self._phi_cache.to(device), dim=-1)
        cos_all = phi @ phi.T
        self._teacher = torch.softmax(
            cos_all / self.criterion.tau_teacher, dim=-1
        ).detach()
        self._teacher = self._teacher[
            torch.as_tensor(self.seen_classes, dtype=torch.long, device=device)
        ]

        # Sanity: dry-run through the aggregator to confirm the visual side
        # accepts the cached means and outputs one row per seen class.
        with torch.no_grad():
            m_probe = self.model.mfa(self._level_means)
        if m_probe.shape[0] != self._teacher.shape[0] or m_probe.shape[-1] != self.model.mfa.out_dim:
            raise ValueError(
                "Transfer-alignment shape mismatch: "
                f"m_s={tuple(m_probe.shape)}, teacher={tuple(self._teacher.shape)}"
            )
        self.model.train()

    @torch.no_grad()
    def _build_null_condition(self):
        """v11 Fix 2a (IPR-v2 CFG): cache the trainval null condition.

        One pass over trainval with the frozen backbone (eval transform,
        protocol-safe — never test splits) accumulates:
          - mean patch-token map of the deepest level  -> r(∅) condition
          - mean v_tilde (post-MFA)                    -> gate condition
        The IPR re-evaluates r(∅) with its current weights on every forward
        (detached), so this cache only fixes the CONDITION, not the
        reaction. Called once before the first train step when the IPR is
        a CrossAttentionPrototypeRefiner in cfg_mode='cfg'.
        """
        ipr = getattr(self.model, "ipr", None)
        if ipr is None or not hasattr(ipr, "set_null_condition"):
            return
        device = self.device
        self.model.eval()
        ds = self._unwrap_dataset(self.train_loader.dataset)
        orig_transform = getattr(ds, "transform", None)
        if hasattr(ds, "_default_transform"):
            ds.transform = ds._default_transform("test")

        token_sum = None
        v_sum = None
        n = 0
        for batch in self.train_loader:
            features, _ = batch[:2]
            features = features.to(device)
            visual_levels = self.model.vis_encoder(features)
            if not isinstance(visual_levels, dict):
                raise ValueError(
                    "IPR cfg_mode='cfg' requires the anchored vision path "
                    "(vision.cls_anchor=true)."
                )
            tokens_h = visual_levels["tokens"][-1]
            v_tilde = self.model.mfa(visual_levels)
            if token_sum is None:
                token_sum = torch.zeros_like(tokens_h[0])
                v_sum = torch.zeros_like(v_tilde[0])
            token_sum += tokens_h.sum(dim=0)
            v_sum += v_tilde.sum(dim=0)
            n += features.shape[0]
        if n == 0:
            raise RuntimeError("Null-condition cache found no trainval batches.")
        if orig_transform is not None:
            ds.transform = orig_transform
        ipr.set_null_condition(
            (token_sum / n).unsqueeze(0),
            (v_sum / n).unsqueeze(0),
        )
        self.model.train()
        self.logger.info(
            f"IPR CFG null condition cached over {n} trainval images."
        )

    def _semantic_encoder_is_frozen(self) -> bool:
        semantic_cfg = getattr(getattr(self.cfg, "model", object()), "semantic", None)
        if semantic_cfg is not None and hasattr(semantic_cfg, "freeze"):
            return bool(semantic_cfg.freeze)
        psc = getattr(self.model, "psc", None)
        return bool(getattr(psc, "freeze_encoder", False))

    def _maybe_build_frozen_phi_cache(self):
        """One-shot CLIP/text encode for all classes when the text tower is frozen."""
        # Precomputed-embedding mode (data.semantic_source='embedding'): the
        # class phi_t (pre-Ws) for ALL classes was loaded from
        # data.semantic_embedding_path into self.attribute, and the model's psc
        # uses a parameter-free identity backbone (no text tower). Use those
        # embeddings directly as the frozen phi cache so the full-bank losses
        # (L_trans, pres_target all/unseen, L_sem, struct/sep) keep working.
        if self.semantic_source == "embedding":
            psc = getattr(self.model, "psc", None)
            if psc is None or getattr(psc, "backbone_type", None) != "embedding":
                raise ValueError(
                    "semantic_source='embedding' requires model.psc to use the "
                    "parameter-free embedding backbone (set model.semantic.input_dim)."
                )
            if int(self.attribute.shape[1]) != int(getattr(psc, "encoder_dim", self.attribute.shape[1])):
                raise ValueError(
                    f"Precomputed semantic dim {self.attribute.shape[1]} != "
                    f"psc.encoder_dim {getattr(psc, 'encoder_dim', None)}."
                )
            self._phi_cache = self.attribute.detach().to(self.device)
            self.logger.info(
                f"Precomputed semantic embeddings -> frozen phi cache "
                f"shape={tuple(self._phi_cache.shape)} (text tower NOT loaded)."
            )
            return

        if self.class_semantics is None:
            return
        if not self._semantic_encoder_is_frozen():
            self.logger.info(
                "Semantic encoder is trainable — skip phi cache "
                "(encode class_desc every batch)."
            )
            return
        psc = getattr(self.model, "psc", None)
        if psc is None or not hasattr(psc, "encode_class_phi"):
            self.logger.warning(
                "Model has no psc.encode_class_phi; cannot cache frozen semantics."
            )
            return

        n_cls = len(self.class_semantics)
        self.logger.info(
            f"Caching frozen semantic phi_t for {n_cls} classes "
            f"(itemwise={bool(getattr(psc, 'itemwise_pool', False))}) …"
        )
        t0 = time.time()
        # encode_class_phi is @torch.no_grad; keep model in train() for heads after.
        was_training = self.model.training
        self.model.eval()
        phi = psc.encode_class_phi(self.class_semantics)
        if not torch.is_tensor(phi) or phi.dim() != 2:
            raise RuntimeError(
                f"encode_class_phi must return (C, D) tensor, got {type(phi)} "
                f"shape={getattr(phi, 'shape', None)}"
            )
        if int(phi.shape[0]) != n_cls:
            raise RuntimeError(
                f"phi cache class count mismatch: got {phi.shape[0]}, expected {n_cls}"
            )
        self._phi_cache = phi.detach().to(self.device, non_blocking=True)
        if was_training:
            self.model.train()
        elapsed = time.time() - t0
        self.logger.info(
            f"Cached phi_t shape={tuple(self._phi_cache.shape)} "
            f"device={self._phi_cache.device} in {elapsed:.1f}s "
            f"(reused every train/eval batch)."
        )

    def _prepare_semantic_input(self, class_indices=None):
        """Prepare semantic input for the model.

        When a frozen phi cache exists, returns per-class tensors (skips CLIP).
        In class_desc mode without cache, returns raw name_c/attr_c/hier_c/desc_c
        records. Otherwise returns XLSA17 attributes or cached embeddings.
        """
        if self._phi_cache is not None:
            indices = (
                range(self._phi_cache.shape[0])
                if class_indices is None
                else class_indices
            )
            # Dict of tensors → PriSematicConstructor uses phi as-is, then Ws.
            return {
                str(int(class_index)): self._phi_cache[int(class_index)]
                for class_index in indices
            }

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

    def _apply_phase_schedule(self, epoch):
        """v13 Phased Training: Điều phối mở/khóa tham số và cấu hình loss theo Phase.

        - Phase 1 (Epoch 1 -> phase1_epochs, vd 15):
          Học trích xuất thuộc tính khách quan. Mở IPR (W_qa, W_k, W_vca).
          Khóa cứng W_multi ≡ 0, W_s ≡ I, λ_r ≡ 0. Chỉ chạy L_attr (Cosine 312 thuộc tính).
        - Phase 2 (Epoch phase1_epochs+1 -> phase2_epochs, vd 16 -> 40):
          Thích nghi thị giác đa tầng MFA. Mở W_multi.
          Khóa cứng IPR và W_s. Chạy L_task (KD 200) + 2.0 L_fm + 0.1 L_cls.
        - Phase 3 (Epoch phase2_epochs+1 -> epochs, vd 41 -> 50):
          Hiệu chỉnh tương quan ngữ nghĩa W_s. Mở W_s (BA low-rank).
          Khóa cứng W_multi và IPR. Chạy L_rel (RKD 200 lớp) + 0.05 L_cls.
        """
        if not self.phased_training:
            return

        target_phase = 1 if epoch <= self.phase1_epochs else (2 if epoch <= self.phase2_epochs else 3)
        if target_phase == self._current_phase:
            return

        self._current_phase = target_phase
        self.logger.info(f"=== [PHASED TRAINING] CHUYỂN SANG PHASE {target_phase} TẠI EPOCH {epoch} ===")

        # Lấy tham chiếu các module
        mfa = getattr(self.model, "mfa", None)
        ipr = getattr(self.model, "ipr", None)
        psc = getattr(self.model, "psc", None)

        if target_phase == 1:
            self.logger.info(">>> Phase 1: Học Attribute Grounding (Mở IPR, Khóa W_multi, W_local và W_s)")
            # Mở IPR
            if ipr is not None:
                ipr.requires_grad_(True)
                if hasattr(ipr, "lambda_r"):
                    ipr.lambda_r.requires_grad_(False)
                    with torch.no_grad():
                        ipr.lambda_r.fill_(0.0)  # Prototype hoàn toàn là p_sem
            # Khóa W_multi, W_local và W_s
            if mfa is not None and hasattr(mfa, "W_multi"):
                mfa.W_multi.requires_grad_(False)
                with torch.no_grad():
                    mfa.W_multi.weight.zero_()
                    if mfa.W_multi.bias is not None:
                        mfa.W_multi.bias.zero_()
            if hasattr(self.model, "W_local"):
                self.model.W_local.requires_grad_(False)
                with torch.no_grad():
                    self.model.W_local.weight.zero_()
            if psc is not None and hasattr(psc, "Ws"):
                psc.Ws.requires_grad_(False)

            # Cấu hình Loss Phase 1: Chỉ chạy L_attr
            self.criterion.lambda_task = 0.0
            self.criterion.lambda_visual = 0.0
            self.criterion.lambda_attr = 1.0
            self.criterion.lambda_cls = 0.0
            self.criterion.lambda_rel = 0.0
            self.criterion.lambda_sem = 0.0

        elif target_phase == 2:
            self.logger.info(">>> Phase 2: Thích nghi Thị giác Đa tầng Local-Global (Mở W_multi & W_local, Khóa IPR và W_s)")
            # Khóa IPR
            if ipr is not None:
                ipr.requires_grad_(False)
            # Mở W_multi và W_local
            if mfa is not None and hasattr(mfa, "W_multi"):
                mfa.W_multi.requires_grad_(True)
            if hasattr(self.model, "W_local"):
                self.model.W_local.requires_grad_(True)
            # Khóa W_s
            if psc is not None and hasattr(psc, "Ws"):
                psc.Ws.requires_grad_(False)

            # Cấu hình Loss Phase 2: L_task + 2.0 L_visual + 0.1 L_cls
            self.criterion.lambda_task = 1.0
            self.criterion.lambda_visual = 2.0
            self.criterion.lambda_attr = 0.0
            self.criterion.lambda_cls = 0.1
            self.criterion.lambda_rel = 0.0
            self.criterion.lambda_sem = 0.0

        elif target_phase == 3:
            self.logger.info(">>> Phase 3: Hiệu chỉnh Ngữ nghĩa W_s (Mở W_s, Khóa Visual Stream và IPR)")
            # Khóa IPR, W_multi và W_local
            if ipr is not None:
                ipr.requires_grad_(False)
            if mfa is not None and hasattr(mfa, "W_multi"):
                mfa.W_multi.requires_grad_(False)
            if hasattr(self.model, "W_local"):
                self.model.W_local.requires_grad_(False)
            # Mở W_s (BA)
            if psc is not None and hasattr(psc, "Ws"):
                psc.Ws.requires_grad_(True)

            # Cấu hình Loss Phase 3: L_task + 0.2 L_rel + 0.05 L_cls
            self.criterion.lambda_task = 1.0
            self.criterion.lambda_visual = 0.0
            self.criterion.lambda_attr = 0.0
            self.criterion.lambda_cls = 0.05
            self.criterion.lambda_rel = 0.2
            self.criterion.lambda_sem = 0.0

        # Reset optimizer param groups cho đúng các biến requires_grad=True
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        training_cfg = getattr(self.cfg, "training", object())
        lr = float(getattr(training_cfg, "lr", 1e-4))
        wd = float(getattr(training_cfg, "weight_decay", 1e-4))
        opt_type = str(getattr(training_cfg, "optimizer", "adamw")).lower()

        if opt_type == "adam":
            self.optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)
        elif opt_type == "sgd":
            self.optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=wd)
        else:
            self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

        self.logger.info(f"Optimizer khởi tạo lại cho Phase {target_phase}: {len(trainable_params)} trainable tensors, lr={lr:.2e}.")

    def train_one_epoch(self, epoch):
        """Train for one epoch on seen classes."""
        self._apply_phase_schedule(epoch)
        self.model.train()
        loss_meter = AverageMeter()
        component_meters = {}

        label_map = self._label_map

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
            if label_map is None:
                mapped_labels = labels
            else:
                mapped_labels = torch.tensor(
                    [label_map[int(l)] for l in labels.cpu()],
                    dtype=torch.long, device=self.device
                )

            # Semantic embeddings of the classes used by this step's CE.
            seen_attr = self._prepare_semantic_input(self._train_classes)

            # Forward: model expects (text_feature, vision_feature)
            out = self.model(seen_attr, features)
            if hasattr(out, 'scores'):
                s_c = out.scores
                m_s = None
                p_all = None
                # Full-class prototype bank (seen + unseen) from the frozen phi
                # cache — used by L_trans, and by pres_target in ('all',
                # 'unseen') so unseen refined prototypes are regularized too.
                need_full_bank = self._is_tri_loss or self._lambda_gen > 0 or (
                    self._lambda_trans > 0 or self._lambda_zs > 0 or
                    self._lambda_rel > 0 or self._lambda_tan > 0 or
                    self.criterion.pres_target in ("all", "unseen")
                    and self._phi_cache is not None
                    and len(self._train_classes) != self._phi_cache.shape[0]
                )
                if need_full_bank:
                    phi_all = self._phi_cache.to(self.device)
                    p_all = torch.nn.functional.normalize(
                        self.model.psc.project_phi(
                            phi_all, torch.arange(phi_all.shape[0], device=self.device)
                        ),
                        dim=-1,
                    )
                if self._lambda_trans > 0:
                    m_s = self.model.mfa(self._level_means)
                p_ref_all = None
                pres_mask = None
                if ((self._is_tri_loss or self._lambda_zs > 0 or self.criterion.pres_target in ("all", "unseen"))
                        and p_all is not None
                        and len(self._train_classes) != p_all.shape[0]):
                    # v10 cross-attn IPR needs the patch-token map and returns
                    # (p_hat, attr_pred); legacy IPR takes 2 args, returns p_hat.
                    tokens_h = getattr(out, "tokens", None)
                    if tokens_h is not None:
                        ipr_out = self.model.ipr(p_all, out.v_embed, tokens_h)[0]
                    else:
                        ipr_out = self.model.ipr(p_all, out.v_embed)
                    p_ref_all = self.model.rp_bank(ipr_out)
                    if self.criterion.pres_target == "unseen":
                        pres_mask = torch.ones(
                            p_all.shape[0], dtype=torch.bool, device=self.device
                        )
                        pres_mask[torch.as_tensor(
                            self.seen_classes, dtype=torch.long,
                            device=self.device,
                        )] = False
                # L_sem (Eq. 33) needs phi of the CE-bank classes, in the same
                # row order as out.p_sem_unnorm (i.e. _train_classes order).
                phi_bank = None
                if self.criterion.lambda_sem > 0 and self._phi_cache is not None:
                    bank_idx = torch.as_tensor(
                        [int(c) for c in self._train_classes],
                        dtype=torch.long, device=self.device,
                    )
                    phi_bank = self._phi_cache.to(self.device)[bank_idx]
                # v10 L_attn / v12 tri_loss: GT attribute rows for this batch. labels are
                # GLOBAL class ids and self.gt_attribute is the [C_all, n_attr]
                # GT matrix loaded from metadata/attributes.npy — index with
                # raw labels, not mapped_labels.
                gt_attr = None
                attr_pred = getattr(out, "attr_pred", None)
                if attr_pred is not None and (self.criterion.lambda_attn > 0 or getattr(self.criterion, "lambda_attr", 0) > 0 or self._is_tri_loss):
                    if self.gt_attribute is None:
                        raise ValueError(
                            "Attribute loss requires the GT attribute matrix "
                            "(prepared metadata/attributes.npy or "
                            "data.gt_attribute_path)."
                        )
                    gt_attr = self.gt_attribute[labels]
                # v11 L_zs / v12 tri_loss: scores of the ACTUAL refined path over ALL
                # classes — exactly the tensor eval computes, now carrying
                # gradient into W_multi (via v_tilde), Ws (via p_all) and
                # the IPR (via p_ref_all) simultaneously.
                scores_all = None
                if self._lambda_zs > 0 or self._is_tri_loss:
                    if p_ref_all is None:
                        raise ValueError(
                            "L_zs / Tri-Loss requires pres_target in ('all', "
                            "'unseen') or tri_loss mode so the full-class refined bank exists."
                        )
                    scores_all = self.model.msh(out.v_embed, p_ref_all)
                loss, loss_dict = self.criterion(
                    s_c, mapped_labels,
                    v_tilde=out.v_embed,
                    p_sem=out.p_sem,
                    p_ref=out.p_ref,
                    m_s=m_s,
                    p_all=p_all if self._lambda_trans > 0 else None,
                    teacher=self._teacher,
                    phi_t=phi_bank,
                    p_sem_unnorm=out.p_sem_unnorm if self.criterion.lambda_sem > 0 else None,
                    p_ref_all=p_ref_all,
                    p_sem_all=p_all if p_ref_all is not None else None,
                    pres_mask=pres_mask,
                    attr_pred=attr_pred,
                    gt_attr=gt_attr,
                    cls_anchor=getattr(out, "cls_anchor", None),
                    scores_all=scores_all,
                    phi_all=phi_all if (scores_all is not None or self._is_tri_loss or self.criterion.lambda_rel > 0) else None,
                    p_sem_all_rel=p_all if (self._lambda_rel > 0 or self.criterion.lambda_rel > 0 or self._lambda_tan > 0 or self.criterion.lambda_tan > 0 or self._is_tri_loss) else None,
                )
                if self._ws_eye is not None:
                    psc = self.model.psc
                    if hasattr(psc, "ws_delta_norm_sq"):
                        ws_pen = psc.ws_delta_norm_sq()
                    else:
                        ws = psc.Ws
                        ws_pen = ((ws.weight - self._ws_eye) ** 2).sum()
                    loss = loss + self.ws_anchor * ws_pen
                    loss_dict["loss_wsanchor"] = float(
                        (self.ws_anchor * ws_pen).detach().cpu()
                    )
                if self._wv_anchor_state is not None:
                    mfa = self.model.mfa
                    wv_parts = [proj.weight for proj in (mfa.level_proj or [])]
                    wv_parts.append(mfa.Wv.weight)
                    if mfa.single_dim == mfa.out_dim:
                        wv_parts.append(mfa.Wv_single.weight)
                    wv_pen = sum(
                        ((w - w0.to(w.device)) ** 2).sum()
                        for w, w0 in zip(wv_parts, self._wv_anchor_state)
                    )
                    loss = loss + self.wv_anchor * wv_pen
                    loss_dict["loss_wvanchor"] = float(
                        (self.wv_anchor * wv_pen).detach().cpu()
                    )
                # v12 L_gen: sample a synthetic-unseen sub-batch from the
                # offline bank and push it through the LIVE inference path
                # (mfa -> ipr -> rp_bank -> msh) against the full 200-class
                # bank. CE on GLOBAL unseen ids gives the unseen columns a
                # discriminative gradient through exactly the path eval
                # uses; p_all below is the same tensor the real-image IPR
                # call used, so Ws receives one joint gradient.
                if self._lambda_gen > 0:
                    idx = torch.randint(
                        len(self._syn.unseen_global_ids),
                        (self._gen_samples,), device=self.device)
                    y_syn = self._syn.unseen_global_ids[idx]  # GLOBAL ids
                    eps_c = torch.randn(self._gen_samples, 512, device=self.device)
                    eps_g = torch.randn(self._gen_samples, 2, 768, device=self.device)
                    cls_syn = self._syn.cls_mean[idx] + self._syn.cls_std[idx] * eps_c
                    gap_lm = (self._syn.gap_mean[idx][:, :2]
                              + self._syn.gap_std[idx][:, :2] * eps_g)
                    tok_syn = self._syn.tok_mean_h[idx]  # no noise (Run 1)
                    visual_levels_syn = {
                        "tokens": [gap_lm[:, 0].unsqueeze(1).expand(-1, 197, -1),
                                   gap_lm[:, 1].unsqueeze(1).expand(-1, 197, -1),
                                   tok_syn],
                        "cls": cls_syn,
                    }
                    v_tilde_syn = self.model.mfa(visual_levels_syn)
                    p_hat_syn, attr_pred_syn = self.model.ipr(
                        p_all, v_tilde_syn, tok_syn)
                    p_ref_syn = self.model.rp_bank(p_hat_syn)
                    scores_syn = self.model.msh(v_tilde_syn, p_ref_syn)
                    l_gen = torch.nn.functional.cross_entropy(
                        scores_syn / self.criterion.tau, y_syn)
                    loss = loss + self._lambda_gen * l_gen
                    loss_dict["loss_gen"] = float(
                        (self._lambda_gen * l_gen).detach().cpu())
                    # Same L_attn formula as the real batch: MSE against the
                    # L2-normalized GT attribute row of the GLOBAL id.
                    if attr_pred_syn is not None and self.criterion.lambda_attn > 0:
                        l_gen_attn = torch.nn.functional.mse_loss(
                            attr_pred_syn,
                            torch.nn.functional.normalize(
                                self.gt_attribute[y_syn], dim=-1))
                        loss = loss + self.criterion.lambda_attn * l_gen_attn
                        loss_dict["loss_genattn"] = float(
                            (self.criterion.lambda_attn * l_gen_attn).detach().cpu())
                p_all = None
                if self._gram_target is not None or self.align_all_classes or self.sep_weight > 0:
                    phi_all = self._phi_cache.to(self.device)
                    p_all = torch.nn.functional.normalize(
                        self.model.psc.project_phi(
                            phi_all, torch.arange(phi_all.shape[0], device=self.device)
                        ), dim=-1
                    )
                if self._gram_target is not None:
                    gram_pen = ((p_all @ p_all.T - self._gram_target) ** 2).mean()
                    loss = loss + self.struct_weight * gram_pen
                    loss_dict["loss_struct"] = float(
                        (self.struct_weight * gram_pen).detach().cpu()
                    )
                if self.sep_weight > 0:
                    cos_all = p_all @ p_all.T
                    off_mask = ~torch.eye(
                        cos_all.shape[0], dtype=torch.bool, device=cos_all.device
                    )
                    sep = torch.clamp(
                        cos_all[off_mask] - self.sep_margin, min=0.0
                    ).mean()
                    loss = loss + self.sep_weight * sep
                    loss_dict["loss_sep"] = float(
                        (self.sep_weight * sep).detach().cpu()
                    )
                if self.align_all_classes:
                    logits = torch.nn.functional.cosine_similarity(
                        out.v_embed.unsqueeze(1), p_all.unsqueeze(0), dim=-1
                    ) / self.criterion.tau
                    align_all = torch.nn.functional.cross_entropy(logits, labels)
                    loss = loss + self._align_weight * align_all
                    loss_dict["loss_alignall"] = float(
                        (self._align_weight * align_all).detach().cpu()
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
            progress_dict = {"loss": f"{loss_meter.avg:.4f}"}
            if "loss_cls" in component_meters:
                progress_dict["cls"] = f"{component_meters['loss_cls'].avg:.4f}"
            if "loss_task" in component_meters:
                progress_dict["task"] = f"{component_meters['loss_task'].avg:.4f}"
            if "loss_visual" in component_meters:
                progress_dict["vis"] = f"{component_meters['loss_visual'].avg:.4f}"
            elif "loss_fm" in component_meters:
                progress_dict["fm"] = f"{component_meters['loss_fm'].avg:.4f}"
            if "loss_attr" in component_meters:
                progress_dict["attr"] = f"{component_meters['loss_attr'].avg:.4f}"
            elif "loss_attn" in component_meters:
                progress_dict["attn"] = f"{component_meters['loss_attn'].avg:.4f}"
            progress_dict["lr"] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
            progress.set_postfix(**progress_dict, refresh=False)

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

        # Hubness diagnostic: top-1 win concentration on seen vs unseen classes.
        hub = self.evaluator.hubness_stats(scores_seen, scores_unseen)

        return {
            "zsl_T1": zsl_T1 * 100,
            "gzsl_u": u * 100,
            "gzsl_s": s * 100,
            "gzsl_H": H * 100,
            "cal_u": cal_u * 100,
            "cal_s": cal_s * 100,
            "cal_H": cal_H * 100,
            "cal_gamma": cal_gamma,
            "hub_seen": hub["hub_seen_ratio"],
            "hub_unseen": hub["hub_unseen_ratio"],
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

        # Probe mode: evaluate the initialized model once, no parameter updates
        # (R0 geometry probe — measures the pre-aligned frozen geometry).
        if self.probe_mode:
            self.logger.info("Probe mode: single evaluation, no training updates.")
            eval_stats = self.evaluate_calibrated()
            self.last_eval_stats = dict(eval_stats)
            log_msg = (
                f"[PROBE] ZSL T1: {eval_stats['zsl_T1']:.1f}% "
                f"| GZSL u: {eval_stats['gzsl_u']:.1f}% "
                f"s: {eval_stats['gzsl_s']:.1f}% "
                f"H: {eval_stats['gzsl_H']:.1f}%"
                f" | Cal u: {eval_stats['cal_u']:.1f}% "
                f"s: {eval_stats['cal_s']:.1f}% "
                f"H: {eval_stats['cal_H']:.1f}% "
                f"γ: {eval_stats['cal_gamma']:.2f}"
                f" | hub seen/unseen: {eval_stats['hub_seen']:.2f}/"
                f"{eval_stats['hub_unseen']:.2f}"
            )
            self.logger.info(log_msg)
            self.best_H = eval_stats["cal_H"]
            self.best_epoch = 0
            self.logger.info(
                f"=== Done (probe). Cal H: {self.best_H:.1f}% ==="
            )
            return self.best_H, self.best_epoch

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            ckpt_dir = os.path.join(self.save_dir, getattr(
                self.cfg.data, "dataset", "ZSL"))
            train_stats = self.train_one_epoch(epoch)
            elapsed = time.time() - t0

            log_msg = (
                f"Epoch [{epoch}/{self.epochs}] "
                f"Loss: {train_stats['loss']:.4f} "
                f"LR: {train_stats['lr']:.6f} "
                f"Time: {elapsed:.1f}s"
            )
            # Component diagnostics: semantic/anchor penalties and low-rank
            # drift magnitude (only present when the respective term is on).
            extras = []
            for name in ("loss_task", "loss_visual", "loss_attr",
                         "loss_sem", "loss_wsanchor", "loss_wvanchor",
                         "loss_attn", "loss_trans", "loss_fm", "loss_zs",
                         "loss_rel", "loss_tan", "loss_gen", "loss_genattn"):
                if name in train_stats:
                    extras.append(f"{name[5:]}={train_stats[name]:.4f}")
            if "cos_v_cls" in train_stats:
                extras.append(f"cosv={train_stats['cos_v_cls']:.4f}")
            if extras:
                log_msg += " | " + " ".join(extras)
            psc = getattr(self.model, "psc", None)
            if psc is not None and getattr(psc, "ws_rank", 0) > 0 \
                    and hasattr(psc, "ws_delta_norm_sq"):
                with torch.no_grad():
                    ba_norm = float(psc.ws_delta_norm_sq().sqrt())
                log_msg += f" ||BA||={ba_norm:.4f}"
            mfa = getattr(self.model, "mfa", None)
            if mfa is not None and hasattr(mfa, "W_multi"):
                with torch.no_grad():
                    wm_norm = float(mfa.W_multi.weight.norm())
                log_msg += f" ||W_multi||={wm_norm:.2f}"
            if hasattr(self.model, "W_local"):
                with torch.no_grad():
                    wloc_norm = float(self.model.W_local.weight.norm())
                log_msg += f" ||W_loc||={wloc_norm:.2f}"
            ipr = getattr(self.model, "ipr", None)
            if ipr is not None and hasattr(ipr, "lambda_r"):
                with torch.no_grad():
                    lr_val = float(ipr.lambda_r.detach())
                log_msg += f" λ_r={lr_val:.4f}"

            if epoch % self.eval_freq == 0 or epoch == self.epochs:
                eval_stats = self.evaluate_calibrated()
                self.last_eval_stats = dict(eval_stats)
                log_msg += (
                    f" | ZSL T1: {eval_stats['zsl_T1']:.1f}% "
                    f"| GZSL u: {eval_stats['gzsl_u']:.1f}% "
                    f"s: {eval_stats['gzsl_s']:.1f}% "
                    f"H: {eval_stats['gzsl_H']:.1f}%"
                    f" | Cal u: {eval_stats['cal_u']:.1f}% "
                    f"s: {eval_stats['cal_s']:.1f}% "
                    f"H: {eval_stats['cal_H']:.1f}% "
                    f"γ: {eval_stats['cal_gamma']:.2f}"
                    f" | hub s/u: {eval_stats['hub_seen']:.2f}/"
                    f"{eval_stats['hub_unseen']:.2f}"
                )

                # Save best model by calibrated GZSL H (giao thức chọn checkpoint
                # không đổi so với v8 -> kết quả so sánh trực tiếp được), và bằng
                # ZSL T1 (chỉ tiêu quyết định của v9).
                state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "eval_stats": eval_stats,
                }
                if eval_stats["cal_H"] > self.best_H:
                    self.best_H = eval_stats["cal_H"]
                    self.best_epoch = epoch
                    save_checkpoint({**state, "best_H": self.best_H},
                                    os.path.join(ckpt_dir, "best.pth"))
                    log_msg += " *BEST*"
                if eval_stats["zsl_T1"] > self.best_T1:
                    self.best_T1 = eval_stats["zsl_T1"]
                    self.best_T1_epoch = epoch
                    save_checkpoint({**state, "best_H": self.best_H},
                                    os.path.join(ckpt_dir, "best_t1.pth"))
                    log_msg += " *BEST_T1*"

            self.logger.info(log_msg)

        # v9.6: final.pth = trọng số epoch cuối của lịch hoàn chỉnh. Đây là
        # checkpoint "chính thức" về giao thức (best/best_t1 chọn theo chỉ số
        # trên tập test nên chỉ dùng làm tham chiếu upper-bound).
        if getattr(self, "save_final", True):
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "eval_stats": dict(self.last_eval_stats or {}),
            }
            save_checkpoint(state, os.path.join(ckpt_dir, "final.pth"))

        self.logger.info(
            f"=== Done. Best Cal H: {self.best_H:.1f}% at epoch {self.best_epoch} "
            f"| Best ZSL T1: {self.best_T1:.1f}% at epoch {self.best_T1_epoch} ==="
        )

        return self.best_H, self.best_epoch
