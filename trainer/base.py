"""SPRN (LPR) trainer: phased 3-stage training with ZSL/GZSL evaluation.

``TrainerBase`` implements the classic Tri-Loss loop; ``CZSLTrainer`` and
``MultiLabelTrainer`` override only where the benchmark differs. ``trainer.type``
in the config picks one.
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tools.utils import AverageMeter, save_checkpoint, setup_logger
from tools.evaluator import ZSLEvaluator


def _as_dict(node):
    """Plain dict from a config node, which ``compose_config`` wraps as ``Config``.

    The phase table is read by key (stage number, weight name), so it needs
    real item access; ``Config`` stores everything as attributes.
    """
    if isinstance(node, dict):
        return dict(node)
    return dict(vars(node)) if hasattr(node, "__dict__") else {}


def _stage_number(key):
    """Stage key from YAML, which yields ints for ``1:`` and strs for ``"1":``.

    Both spellings are accepted; anything else is a config error worth naming,
    since a bad key would otherwise drop a whole stage silently.
    """
    try:
        return int(key)
    except (TypeError, ValueError):
        raise ValueError(
            f"training.phases has a non-numeric stage key {key!r}. Use stage "
            "numbers 1..N (quote them in YAML: \"1\": ...).") from None



class TrainerBase:
    """Training loop for SPRN on ZSL benchmarks.

    Subclasses override:

        train_one_epoch  criterion + label layout
        _eval_stats      metric protocol for the family
        _log_eval        which numbers reach the epoch line
        _selection       which numbers choose the best checkpoint
        FINAL_CKPT       final checkpoint filename
    """

    #: classic/CZSL write ``final.pth`` (weights of the last epoch). Multi-label writes ``last.pth``.
    FINAL_CKPT = "final.pth"

    #: Component losses printed on the epoch line.
    LOG_COMPONENTS = ("loss_task", "loss_visual", "loss_attr", "loss_rel")

    def __init__(self, cfg, model, criterion, train_dataset, test_seen_dataset,
                 test_unseen_dataset, attribute, seen_classes, unseen_classes,
                 device="cuda"):
        """
        Args:
            cfg: Config object.
            model: SPRN model instance.
            criterion: SPRNLoss instance.
            train_dataset / test_seen_dataset / test_unseen_dataset: torch Datasets.
            attribute: (C, D) precomputed class text embeddings phi (torch.Tensor).
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

        # Supervision target for L_attr: slot-affinity table A [C, K] built from text
        # (tools/build_latent_slots.py). Rows are indexed by GLOBAL class ids; rows of
        # unseen classes are zero and never indexed (labels are seen).
        rc = cfg.model.refiner
        tgt = torch.load(rc.slot_target_path, map_location="cpu")
        A = tgt["A"].float().to(device)
        if A.shape[0] != self.num_total_classes:
            raise ValueError(
                f"slot target {rc.slot_target_path} has {A.shape[0]} rows, expected "
                f"{self.num_total_classes} classes.")
        n_slots = torch.load(rc.slot_path, map_location="cpu")["embeddings"].shape[0]
        if A.shape[1] != n_slots:
            raise ValueError(
                f"slot target has K={A.shape[1]} but slot bank has K={n_slots}.")
        self.gt_attribute = A

        # Data loaders
        bs = getattr(cfg.data, "batch_size", 64)
        nw = getattr(cfg.data, "num_workers", 4)

        self.train_loader = DataLoader(
            train_dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=device == "cuda",
        )
        self.test_seen_loader = DataLoader(
            test_seen_dataset, batch_size=bs * 2, shuffle=False,
            num_workers=nw, pin_memory=device == "cuda",
        )
        self.test_unseen_loader = DataLoader(
            test_unseen_dataset, batch_size=bs * 2, shuffle=False,
            num_workers=nw, pin_memory=device == "cuda",
        )

        # Mixed precision. Only the multi-label loop consults ``self.scaler``; the
        # base loop runs full precision either way.
        self.use_amp = bool(getattr(cfg.training, "amp", False)) and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise ValueError("Model has no trainable parameters after applying freeze settings.")
        self.optimizer = torch.optim.AdamW(
            trainable_parameters, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

        # Scheduler. Intentional quirk: it is built on the FIRST optimizer; every phase
        # transition builds a new AdamW (constant lr) and the scheduler keeps stepping
        # the old one.
        epochs = cfg.training.epochs
        warmup = cfg.training.warmup_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs - warmup, eta_min=1e-6)

        self.evaluator = ZSLEvaluator(seen_classes, unseen_classes)

        # Training state
        self.epochs = epochs
        self.eval_freq = getattr(cfg.training, "eval_freq", 1)
        self.save_dir = getattr(cfg.training, "save_dir", "./checkpoints")
        self.best_H = -1.0
        self.best_epoch = 0
        self.best_T1 = -1.0
        self.best_T1_epoch = 0
        self.last_eval_stats = None
        self.save_final = True

        log_dir = getattr(cfg.training, "log_dir", "./logs")
        self.logger = setup_logger(f"SPRN-{cfg.data.dataset}", log_dir)

        self._train_classes = self.seen_classes
        self._label_map = {int(c): i for i, c in enumerate(self.seen_classes)}

        self._phi_cache = None
        self._build_phi_cache()

        self.ws_anchor = float(getattr(cfg.model.semantic, "ws_anchor", 0.0))
        if self.ws_anchor > 0:
            self.logger.info(f"Ws identity anchor ON (weight={self.ws_anchor}).")

        self._phases = self._build_phase_table(cfg)
        self.phase_loss_weights = {}
        self._current_phase = None

    # ------------------------------------------------------------------
    # Semantic input
    # ------------------------------------------------------------------

    def _build_phi_cache(self):
        """Frozen class text embeddings phi [C, D] (loaded from data.semantic_embedding_path)."""
        psc = self.model.psc
        if getattr(psc, "backbone_type", None) != "embedding":
            raise ValueError(
                "trainer needs model.psc to use the parameter-free embedding backbone "
                "(set model.semantic.input_dim).")
        if int(self.attribute.shape[1]) != int(psc.encoder_dim):
            raise ValueError(
                f"Precomputed semantic dim {self.attribute.shape[1]} != "
                f"psc.encoder_dim {psc.encoder_dim}.")
        self._phi_cache = self.attribute.detach()
        self.logger.info(
            f"Precomputed semantic embeddings -> frozen phi cache "
            f"shape={tuple(self._phi_cache.shape)} (text tower NOT loaded)."
        )

    def _prepare_semantic_input(self, class_indices=None):
        """Per-class phi tensors as the dict ``PriSematicConstructor`` expects."""
        idx = range(self._phi_cache.shape[0]) if class_indices is None else class_indices
        return {str(int(i)): self._phi_cache[int(i)] for i in idx}

    # ------------------------------------------------------------------
    # Phased training
    # ------------------------------------------------------------------

    def _build_phase_table(self, cfg):
        """Stage table: boundaries + what each stage opens/freezes/scores.

        Read from ``cfg.training.phases``. ``epochs`` is the stage's LAST epoch; the
        last stage is always stretched to ``training.epochs``.
        """
        raw = getattr(cfg.training, "phases", None)
        if not raw:
            raise ValueError("training.phases is required (3-stage table).")
        table = {}
        for key, spec in _as_dict(raw).items():
            spec = _as_dict(spec or {})
            spec["loss"] = _as_dict(spec.get("loss") or {})
            spec.setdefault("train", [])
            spec.setdefault("freeze", [])
            spec.setdefault("zero", [])
            table[_stage_number(key)] = spec
        table.setdefault(max(table), {})["epochs"] = None
        self._fill_stage_boundaries(table, tail=int(cfg.training.epochs))
        return table

    @staticmethod
    def _fill_stage_boundaries(table, tail):
        """Resolve each stage's first/last epoch.

        Only the LAST stage may leave ``epochs`` null; it then absorbs whatever
        remains, so a run never ends mid-stage. Any earlier open stage is a
        config error — it would make the following stages unreachable.
        """
        ordered = sorted(table)
        last = ordered[-1]
        for phase in ordered[:-1]:
            if table[phase].get("epochs") is None:
                raise ValueError(
                    f"training.phases: stage {phase} has no 'epochs' but is not "
                    f"the last stage ({last}). Only the final stage may be open.")
        if table[last].get("epochs") is None:
            table[last]["epochs"] = tail

        prev = 0
        for phase in ordered:
            spec = table[phase]
            spec["epochs"] = int(spec["epochs"])
            spec["from"] = prev + 1
            if spec["epochs"] <= prev:
                raise ValueError(
                    f"training.phases: stage {phase} ends at epoch "
                    f"{spec['epochs']}, not after stage {phase - 1} "
                    f"(ends {prev}). Boundaries must be strictly increasing.")
            prev = spec["epochs"]
        if prev != tail:
            raise ValueError(
                f"training.phases must cover exactly {tail} epochs; the last "
                f"stage ends at {prev}. Adjust the stage boundaries.")

    def _resolve_phase_target(self, name):
        """Resolve a phase-table entry ('mfa.W_multi', 'psc.Ws') to a module.

        Raises when the model does not have it: a stage that silently no-ops
        would train the wrong geometry without saying so.
        """
        obj = self.model
        for part in name.split("."):
            if not hasattr(obj, part):
                raise AttributeError(
                    f"training.phases references {name!r}, but "
                    f"{type(obj).__name__} has no {part!r}. Fix the config or "
                    "drop it from the stage.")
            obj = getattr(obj, part)
        if not isinstance(obj, torch.nn.Module):
            raise TypeError(f"training.phases target {name!r} is not an nn.Module "
                            f"(got {type(obj).__name__}).")
        return obj

    def _phase_loss_weights(self, phase):
        """Loss weights this stage wants. Weights the criterion does not carry are kept
        in ``self.phase_loss_weights`` for a subclass loop (ASL) to apply itself."""
        return _as_dict(self._phases[phase].get("loss", {}))

    def _apply_phase_schedule(self, epoch):
        """Open/freeze parameters and set loss weights for the stage ``epoch`` falls in.

        Every training loop MUST call this at the start of each epoch; a loop that
        skips it trains the whole schedule with stage-1 geometry, silently.
        """
        target_phase = self._phase_at(epoch)
        if target_phase == self._current_phase:
            return

        self._apply_phase_transition(target_phase)

    def _phase_at(self, epoch):
        """Which configured stage ``epoch`` falls into."""
        for phase, spec in sorted(self._phases.items()):
            if epoch <= int(spec["epochs"]):
                return phase
        return max(self._phases)

    def _apply_phase_transition(self, target_phase):
        """Apply one stage's config: freeze/unfreeze, loss weights, optimizer."""
        spec = self._phases[target_phase]
        self._current_phase = target_phase
        self.logger.info(
            f"=== [PHASED TRAINING] CHUYỂN SANG STAGE {target_phase} "
            f"(epoch {spec['from']}-{spec['epochs']}) ===")

        self.logger.info(f">>> Stage {target_phase}: {spec.get('note', '')}".rstrip())

        # ``zero`` is a SUBSET of ``freeze``, not a per-stage flag: stage 1
        # zeroes the visual stream so the geometry starts at the anchor
        # (W_multi == 0 => v_global == normalize(cls)), but it must NOT zero
        # psc.Ws — that one is only locked, and wiping it would destroy the
        # low-rank init the run depends on.
        for name in spec.get("freeze", []):
            self._resolve_phase_target(name).requires_grad_(False)

        for name in spec.get("zero", []):
            # nn.Module has no zero_(); zero every tensor it holds so the
            # branch starts exactly at the anchor.
            with torch.no_grad():
                for param in self._resolve_phase_target(name).parameters(recurse=True):
                    param.zero_()

        for name in spec.get("train", []):
            self._resolve_phase_target(name).requires_grad_(True)


        # LPR class branch (ipr.proto): trains with the semantic adapter, in the stage
        # that opens psc.Ws (stage 3) or lists "ipr.proto". It stays inactive before
        # that stage (earlier stages score v P^T) and active afterwards (the buffer is
        # saved with the checkpoint). Must run AFTER the train loop: stage 1's
        # ``train: [ipr]`` re-enables proto params, and this disables them again.
        proto = self.model.ipr.proto
        train = spec.get("train", [])
        opens = "ipr.proto" in train or "psc.Ws" in train
        proto.requires_grad_(opens)
        if opens:
            proto.active.fill_(True)

        # Loss weights: stage table (config) is authoritative.
        for name, value in self._phase_loss_weights(target_phase).items():
            if hasattr(self.criterion, name):
                setattr(self.criterion, name, value)
                continue
            # Criterion has no such attribute (ASL only has gamma/clip): the subclass
            # loop reads it from phase_loss_weights.
            self.phase_loss_weights[name] = value

        # Each stage builds a new optimizer over the requires_grad=True params.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        lr = float(self.cfg.training.lr)
        wd = float(self.cfg.training.weight_decay)
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)

        self.logger.info(f"Optimizer khởi tạo lại cho Phase {target_phase}: {len(trainable_params)} trainable tensors, lr={lr:.2e}.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch):
        """Train for one epoch on seen classes (Tri-Loss)."""
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
            mapped_labels = torch.tensor(
                [label_map[int(l)] for l in labels.cpu()],
                dtype=torch.long, device=self.device
            )

            # Semantic embeddings of the seen classes (the CE bank).
            seen_attr = self._prepare_semantic_input(self._train_classes)
            out = self.model(seen_attr, features)

            # Full-class bank (seen + unseen) from the frozen phi cache.
            phi_all = self._phi_cache
            p_all = F.normalize(self.model.psc.Ws(phi_all), dim=-1)
            scores_all = self.model.ipr.score(out.v_embed, p_all, out.slot_readout)
            gt_attr = self.gt_attribute[labels]               # labels are GLOBAL ids
            loss, loss_dict = self.criterion(
                out.scores, mapped_labels,
                v_tilde=out.v_embed,
                cls_anchor=out.cls_anchor,
                scores_all=scores_all,
                phi_all=phi_all,
                attr_pred=out.attr_pred,
                gt_attr=gt_attr,
                p_sem_all_rel=p_all,
            )
            if self.ws_anchor > 0:
                ws_pen = self.model.psc.ws_delta_norm_sq()
                loss = loss + self.ws_anchor * ws_pen
                loss_dict["loss_wsanchor"] = float((self.ws_anchor * ws_pen).detach().cpu())

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
            if "loss_attr" in component_meters:
                progress_dict["attr"] = f"{component_meters['loss_attr'].avg:.4f}"
            progress_dict["lr"] = f"{self.optimizer.param_groups[0]['lr']:.2e}"
            progress.set_postfix(**progress_dict, refresh=False)

        self.scheduler.step()

        stats = {
            "loss": loss_meter.avg,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        stats.update({name: meter.avg for name, meter in component_meters.items()})
        return stats

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

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

    def _eval_stats(self):
        """Per-epoch evaluation for this family.

        Base (v14 classic/CZSL): the point-based metric via ``ZSLEvaluator``.
        Subclasses return their own protocol's dict. Every returned dict must
        carry whatever ``_log_eval`` and ``_selection`` read.
        """
        return self.evaluate_calibrated()

    def _log_eval(self, stats):
        """One-line summary of ``stats`` for the epoch log."""
        s = (
            f" | ZSL T1: {stats['zsl_T1']:.1f}% "
            f"| GZSL u: {stats['gzsl_u']:.1f}% "
            f"s: {stats['gzsl_s']:.1f}% "
            f"H: {stats['gzsl_H']:.1f}%"
            f" | Cal u: {stats['cal_u']:.1f}% "
            f"s: {stats['cal_s']:.1f}% "
            f"H: {stats['cal_H']:.1f}% "
            f"γ: {stats['cal_gamma']:.2f}"
            f" | hub s/u: {stats['hub_seen']:.2f}/"
            f"{stats['hub_unseen']:.2f}"
        )
        return s

    def _selection(self, stats):
        """Checkpoint selection, keyed by the metric this family reports.

        v14 protocol (unchanged since v8/v9 so runs stay comparable):
          'best.pth'    by calibrated GZSL H
          'best_t1.pth' by ZSL T1
        Returns [(value, monitor_key, filename), ...]; a run with no such
        metric (multi-label) returns [] and keeps only ``last.pth``.
        """
        return [(stats["cal_H"], "best_H", "best.pth"),
                (stats["zsl_T1"], "best_T1", "best_t1.pth")]

    def _summary(self):
        """Closing line of the run, in this family's own terms.

        The counters the base loop keeps (``best_H``/``best_T1``) only mean
        something for a family that has a selection metric. Reporting them for
        one that returns ``[]`` from ``_selection`` would print ``-1.0%``, so
        multi-label overrides this rather than printing a number it never
        computed.
        """
        return (f"=== Done. Best Cal H: {self.best_H:.1f}% at epoch "
                f"{self.best_epoch} | Best ZSL T1: {self.best_T1:.1f}% at "
                f"epoch {self.best_T1_epoch} ===")

    def train(self):
        """Full training loop."""
        dataset_name = self.cfg.data.dataset
        self.logger.info(f"=== Training SPRN on {dataset_name} ===")
        self.logger.info(f"Seen classes: {len(self.seen_classes)}, "
                         f"Unseen classes: {len(self.unseen_classes)}")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}, "
                         f"Test seen: {len(self.test_seen_loader.dataset)}, "
                         f"Test unseen: {len(self.test_unseen_loader.dataset)}")

        ckpt_dir = os.path.join(self.save_dir, dataset_name)
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
            extras = []
            for name in self.LOG_COMPONENTS:
                if name in train_stats:
                    extras.append(f"{name[5:]}={train_stats[name]:.4f}")
            if "cos_v_cls" in train_stats:
                extras.append(f"cosv={train_stats['cos_v_cls']:.4f}")
            if extras:
                log_msg += " | " + " ".join(extras)
            with torch.no_grad():
                log_msg += f" ||BA||={float(self.model.psc.ws_delta_norm_sq().sqrt()):.4f}"
                log_msg += f" ||W_multi||={float(self.model.mfa.W_multi.weight.norm()):.2f}"
                log_msg += f" ||W_loc||={float(self.model.W_local.weight.norm()):.2f}"

            if epoch % self.eval_freq == 0 or epoch == self.epochs:
                eval_stats = self._eval_stats()
                self.last_eval_stats = dict(eval_stats)
                log_msg += self._log_eval(eval_stats)

                # Save best model by the family's own metric (see _selection).
                state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "eval_stats": eval_stats,
                }
                for value, monitor, filename in self._selection(eval_stats):
                    best = getattr(self, monitor)
                    if value > best:
                        setattr(self, monitor, value)
                        if monitor == "best_H":
                            self.best_epoch = epoch
                        else:
                            self.best_T1_epoch = epoch
                        save_checkpoint({**state, "best_H": self.best_H},
                                        os.path.join(ckpt_dir, filename))
                        log_msg += f" *{filename[:-4].upper()}*"

            self.logger.info(log_msg)

        # final.pth = weights of the last epoch of the full schedule. best/best_t1 are
        # picked on test metrics, so they are upper-bound references only.
        if self.save_final:
            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "eval_stats": dict(self.last_eval_stats or {}),
            }
            save_checkpoint(state, os.path.join(ckpt_dir, self.FINAL_CKPT))

        self.logger.info(self._summary())

        return self.best_H, self.best_epoch
