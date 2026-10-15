"""Multi-label trainer — MS-COCO, NUS-WIDE (doc A.4 family).

This family differs from the other two in four ways, all of them here:

1. **Criterion** — Asymmetric Loss (Ridnik et al., ICCV 2021) instead of
   Tri-Loss. The exact call that produced the reported numbers is preserved:
   ``AsymmetricLoss(scores / tau, targets)`` with ``tau=0.07``. ASL applies its
   own sigmoid, so dividing by a small tau saturates it and degenerates the
   negative term — that is a real design flaw, but every §3-§5 number was
   measured with it, so it stays until a deliberate re-run. See
   ``losses/asymmetric.py``.

2. **Labels** — multi-hot [N, C] over the FULL seen+unseen eval space, not a
   class index. ``_label_map`` must therefore be identity.

3. **Metric** — top-k P/R/F1 + mAP over the full column space
   (``MultilabelTopK``), which is what makes GZSL real: over an all-unseen
   column space "generalized" means nothing and GZSL would equal ZSL.

    trainer: {type: "MultiLabelTrainer"}
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from losses.asymmetric import build_asymmetric_loss
from tools.registry import TRAINERS
from tools.utils import AverageMeter

from .base import TrainerBase


@TRAINERS.register("MultiLabelTrainer")
class MultiLabelTrainer(TrainerBase):
    """v14 engine with ASL, multi-hot labels and the top-k/mAP protocol."""

    #: Multi-label runs write ``last.pth``, not ``final.pth``.
    FINAL_CKPT = "last.pth"

    #: ASL decomposition, printed on the epoch line.
    LOG_COMPONENTS = ("loss_task", "loss_visual", "loss_rel", "loss_ws", "loss_attr")

    #: Evaluation protocol of this family. Fixed by the trainer, not by config.
    METRIC_TYPE = "MultilabelTopK"

    _ML_LOG_KEYS = (
        ("zsl_top3", "ZSL F1@3"), ("zsl_top5", "F1@5"),
        ("gzsl_top3", "GZSL F1@3"), ("gzsl_top5", "F1@5"),
        ("mAP_unseen", "mAP"),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loss_cfg = self.cfg.model.loss

        # Multi-hot targets carry no class index, so the seen-class remap that
        # the index path applies must be disabled.
        self._label_map = None

        # ASL, verbatim from the scripts. Installed in place of the Tri-Loss criterion
        # the caller passed.
        self.criterion = build_asymmetric_loss(self.cfg).to(self.device)
        self._asl_tau = float(getattr(loss_cfg, "temperature", 0.07))
        self._lambda_task = float(getattr(loss_cfg, "lambda_task", 1.0))
        self._lambda_visual = float(getattr(loss_cfg, "lambda_visual", 2.0))
        self._lambda_ws = float(getattr(loss_cfg, "lambda_ws", 0.01))

        # Training scores against the SEEN columns only, matching the old
        # loops: COCO used ``phi_65[seen_pos]`` (48 of 65) and NUS used all 925
        # seen tags. Held here as an explicit tensor instead of re-slicing a
        # dict per batch. ``torch.arange`` because the builder's seen
        # columns are the first n_seen of the eval space (0..n_seen-1).
        self._train_seen_idx = list(map(int, self.seen_classes))

        # Phase 3's relational term compares the trained prototype geometry with the
        # frozen phi geometry. Both banks are built lazily on the first epoch.
        self._p_sem_seen = None
        self._phi_seen_bank = None

        # L_attr target per image = elementwise max of the slot-target rows of its
        # positive labels. Rows follow the train target columns (= seen positions, in
        # order). ``self.gt_attribute`` is the slot target built by TrainerBase.
        self._seen_slot_target = self.gt_attribute[
            torch.as_tensor(self._train_seen_idx, device=self.gt_attribute.device)]

    def _seen_semantic_banks(self):
        """(p_sem_seen, phi_seen) — normalized [n_seen, D], built once.

        ``p_sem_seen`` comes from the model's own semantic path (``psc``) so it matches
        exactly what ``forward`` scores against; ``phi_seen`` is the frozen pre-Ws
        geometry it is pulled toward.
        """
        attr = self._prepare_semantic_input(self._train_seen_idx)
        psc = self.model.psc
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            out = psc(attr)
        if was_training:
            self.model.train()
        p_sem = out["p_sem_norm"].detach()
        phi = F.normalize(out["phi_t"].detach(), dim=-1)
        return p_sem, phi

    @staticmethod
    def multihot_slot_target(targets, table):
        """Per-image slot target: max over the rows of its positive labels.

        targets [B, C_seen] multi-hot, table [C_seen, K] -> [B, K]. Images with
        no positive get zeros (the caller masks them out).
        """
        masked = table.unsqueeze(0) * targets.unsqueeze(-1)            # [B, C, K]
        neg = torch.finfo(masked.dtype).min
        masked = masked.masked_fill(targets.unsqueeze(-1) <= 0, neg)
        out = masked.max(dim=1).values
        return torch.where(targets.sum(1, keepdim=True) > 0, out, torch.zeros_like(out))

    def _stage_weights(self):
        """Loss weights of the stage this epoch is in.

        Read from the phase table (config) for everything the loop applies that
        ASL itself does not carry — ASL is a single positive/negative objective,
        so ``lambda_task``/``lambda_visual`` are the loop's job, and the
        remaining names (``lambda_attr`` …) simply do not apply to this family.
        """
        weights = self.phase_loss_weights or {}
        lam_attr = float(weights.get("lambda_attr", 0.0))
        return (
            float(weights.get("lambda_task", self._lambda_task)),
            float(weights.get("lambda_visual", self._lambda_visual)),
            float(weights.get("lambda_rel", 0.0)),
            lam_attr,
        )

    def train_one_epoch(self, epoch):
        """ASL optimisation over multi-hot labels, under the phase schedule.

        The base loop is replaced rather than branched: Tri-Loss is a
        soft-distillation objective with no form that accepts multi-hot
        targets, so nothing is shared beyond the optimizer/scheduler
        bookkeeping. The phase schedule is NOT shared implicitly, though — the
        call below is what keeps the stages running for this family; dropping
        it silently trains the whole schedule in stage-1 geometry.
        """
        self._apply_phase_schedule(epoch)
        lam_task, lam_visual, lam_rel, lam_attr = self._stage_weights()
        if lam_rel > 0 and self._p_sem_seen is None:
            self._p_sem_seen, self._phi_seen_bank = self._seen_semantic_banks()
        lam_ws = self._lambda_ws

        self.model.train()
        loss_meter = AverageMeter()
        component_meters = {k: AverageMeter() for k in
                            ("loss_task", "loss_visual", "loss_rel", "loss_ws", "loss_attr")}

        optimizer = self.optimizer
        scaler = getattr(self, "scaler", None)
        use_amp = scaler is not None and scaler.is_enabled()

        progress = tqdm(self.train_loader,
                        desc=f"Train {epoch}/{self.epochs}",
                        unit="batch", dynamic_ncols=True)
        for batch in progress:
            features, targets = batch[:2]
            features = features.to(self.device)
            # Targets are the loader's own label space, which is exactly the
            # seen columns — the builder's seen_classes are 0..n_seen-1.
            targets = targets.to(self.device).float()

            # Train-time attributes: seen columns only (the multi-hot target
            # has one column per seen label, so scores and targets agree).
            attr = self._prepare_semantic_input(self._train_seen_idx)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = self.model(attr, features)
                scores = out.scores if hasattr(out, "scores") else out
                l_task = self.criterion(scores / self._asl_tau, targets)
                if l_task.dim() > 0:
                    l_task = l_task.mean()

                # Stage 2: retention of the frozen zero-shot anchor. Computed
                # against ``v_embed`` (the normalized v_enhanced the model
                # scores with) and ``cls_anchor`` — unlike the older scripts,
                # which looked for a field that never existed and so trained
                # with this term pinned at 0.
                l_vis = torch.zeros((), device=self.device)
                if lam_visual > 0 and out.cls_anchor is not None:
                    l_vis = (1.0 - F.cosine_similarity(
                        out.v_embed, out.cls_anchor, dim=-1)).mean()

                # Stage 3: relational KD pulling the seen prototype geometry
                # toward the frozen phi geometry (the tri-loss L_rel analogue).
                l_rel = torch.zeros((), device=self.device)
                if lam_rel > 0 and self._p_sem_seen is not None:
                    with torch.no_grad():
                        d0 = torch.cdist(self._phi_seen_bank, self._phi_seen_bank)
                        p0 = F.softmax(-d0, dim=-1)
                    d1 = torch.cdist(self._p_sem_seen, self._p_sem_seen)
                    l_rel = F.kl_div(F.log_softmax(-d1, dim=-1), p0,
                                     reduction="batchmean")

                # LPR: slot grounding without human attributes.
                l_attr = torch.zeros((), device=self.device)
                attr_pred = getattr(out, "attr_pred", None)
                if lam_attr > 0 and attr_pred is not None and self._seen_slot_target is not None:
                    tgt = self.multihot_slot_target(targets, self._seen_slot_target)
                    has_pos = targets.sum(1) > 0
                    if bool(has_pos.any()):
                        l_attr = (1.0 - F.cosine_similarity(
                            attr_pred[has_pos].float(), F.normalize(tgt[has_pos], dim=-1),
                            dim=-1)).mean()

                l_ws = torch.zeros((), device=self.device)
                psc = getattr(self.model, "psc", None)
                if psc is not None and hasattr(psc, "ws_delta_norm_sq"):
                    l_ws = psc.ws_delta_norm_sq()

                loss = (lam_task * l_task
                        + lam_visual * l_vis
                        + lam_rel * l_rel
                        + lam_attr * l_attr
                        + lam_ws * l_ws)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_meter.update(loss.item(), features.size(0))
            component_meters["loss_task"].update(float(l_task.detach()),
                                                 features.size(0))
            component_meters["loss_visual"].update(float(l_vis.detach()),
                                                   features.size(0))
            component_meters["loss_rel"].update(float(l_rel.detach()),
                                                features.size(0))
            component_meters["loss_ws"].update(float(l_ws.detach()),
                                               features.size(0))
            component_meters["loss_attr"].update(float(l_attr.detach()),
                                                 features.size(0))
            progress.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if getattr(self, "scheduler", None) is not None:
            self.scheduler.step()

        return {
            "loss": loss_meter.avg,
            "lr": optimizer.param_groups[0]["lr"],
            **{k: m.avg for k, m in component_meters.items()},
        }

    def evaluate(self):
        """Top-k P/R/F1 + mAP over all columns (doc A.4).

        Scored over the FULL eval space in one pass. The base class scores seen
        and unseen separately because point-based ZSL needs the two splits apart;
        multi-label does not — one image set is ranked against all 1006 columns,
        and the ZSL/GZSL distinction is which COLUMNS the metric looks at, not
        which images. That is precisely the distinction the old NUS code got
        wrong (it sliced rows twice and reported ZSL as GZSL).

        Ground truth comes from the dataset, which carries the aligned full-space
        table (``eval_targets``; set by the builder — the loader itself only has
        the seen/81 labels).
        """
        from tools.metrics import METRICS

        metric = METRICS.build(self.METRIC_TYPE,
                               unseen_pos=list(map(int, self.unseen_classes)))

        all_cols = list(range(self.num_total_classes))
        attr = self._prepare_semantic_input(all_cols)

        self.model.eval()
        chunks, label_chunks = [], []
        with torch.no_grad():
            for batch in tqdm(self.test_unseen_loader, desc="Eval all",
                              unit="batch", dynamic_ncols=True, leave=False):
                features = batch[0].to(self.device)
                out = self.model(attr, features)
                s = out.scores if hasattr(out, "scores") else out
                chunks.append(s.float().cpu().numpy())
                label_chunks.append(batch[1].cpu().numpy())
        scores = np.concatenate(chunks, axis=0)

        # Prefer the labels the loader just handed us when they span the whole
        # eval space (COCO: 65 == 65) — that avoids a second image-decode pass.
        # NUS is the case that must fall back: its loader carries 81 columns
        # while the eval space is 1006.
        loaded = np.concatenate(label_chunks, axis=0)
        targets = (loaded if loaded.shape[1] == self.num_total_classes
                   else self._eval_targets())
        if targets.shape[0] != scores.shape[0]:
            raise ValueError(
                f"MultiLabelTrainer: {scores.shape[0]} scored images but "
                f"{targets.shape[0]} ground-truth rows — the eval subset and the "
                "ground-truth table disagree."
            )
        return metric.compute(scores, targets)

    def _eval_targets(self):
        """Full-space ground truth in ``test_unseen_loader`` row order."""
        ds = self.test_unseen_loader.dataset
        indices = None
        if isinstance(ds, Subset):          # smoke runs subset the eval split
            indices = list(ds.indices)
            ds = ds.dataset
        gt = getattr(ds, "eval_targets", None)
        if gt is None:
            raise ValueError(
                "MultiLabelTrainer needs the dataset's full eval-space ground "
                "truth (ds.eval_targets). The dataset builder must attach it — "
                "the loader's own labels are the seen/81 subset only."
            )
        gt = np.asarray(gt, dtype=np.float32)
        return gt if indices is None else gt[indices]

    def _eval_stats(self):
        return self.evaluate()

    def _log_eval(self, stats):
        parts = []
        for key, label in self._ML_LOG_KEYS:
            cell = stats.get(key)
            if isinstance(cell, dict):
                cell = cell.get("f1")
            if cell is not None:
                parts.append(f"{label}: {cell:.2f}%")
        return " | " + " ".join(parts) if parts else ""

    def _selection(self, stats):
        """No best*.pth for multi-label: only ``last.pth`` survives."""
        return []

    def _summary(self):
        """Last-epoch top-k line, since there is no best-checkpoint metric."""
        last = self.last_eval_stats or {}
        parts = []
        for key, label in self._ML_LOG_KEYS:
            cell = last.get(key)
            if isinstance(cell, dict):
                cell = cell.get("f1")
            if cell is not None:
                parts.append(f"{label}: {cell:.2f}%")
        tail = " | ".join(parts) if parts else "no evaluation"
        return f"=== Done. Epoch {self.epochs} (last.pth) | {tail} ==="
