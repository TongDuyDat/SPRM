import torch
import torch.nn as nn
import torch.nn.functional as F

class HSPRNLoss(nn.Module):
    """
    4.8 Training Objective
    Computes Eq. (32), (34)-(36) from
    docs/Hierarchical_Semantic_Prototype_Refinement_Model__HSPRM_.md.
    """
    def __init__(self, cfg):
        super(HSPRNLoss, self).__init__()
        self.cfg = cfg

        loss_cfg = cfg.model.loss
        self.loss_mode = str(getattr(loss_cfg, "loss_mode", "legacy")).lower()
        if self.loss_mode not in ("legacy", "tri_loss"):
            raise ValueError("model.loss.loss_mode must be 'legacy' or 'tri_loss'.")

        self.tau = float(getattr(loss_cfg, "temperature", 0.07))
        # v12 Tri-Loss configuration
        self.lambda_task = float(getattr(loss_cfg, "lambda_task", 1.0))
        self.lambda_visual = float(getattr(loss_cfg, "lambda_visual", getattr(loss_cfg, "lambda_fm", 2.0)))
        self.lambda_attr = float(getattr(loss_cfg, "lambda_attr", getattr(loss_cfg, "lambda_attn", 0.5)))
        self.attr_loss_type = str(getattr(loss_cfg, "attr_loss_type", "cosine")).lower()
        self.lambda_cls = float(getattr(loss_cfg, "lambda_cls", 0.0))

        if self.loss_mode == "legacy":
            required = ("lambda1", "lambda2", "lambda3", "temperature")
            missing = [name for name in required if not hasattr(loss_cfg, name)]
            if missing:
                raise ValueError(
                    "HSPRN legacy loss hyperparameters must be set in config: "
                    + ", ".join(missing)
                )
            self.lambda1 = loss_cfg.lambda1
            self.lambda2 = loss_cfg.lambda2
            self.lambda3 = loss_cfg.lambda3
        else:
            self.lambda1 = getattr(loss_cfg, "lambda1", 0.0)
            self.lambda2 = getattr(loss_cfg, "lambda2", 0.0)
            self.lambda3 = getattr(loss_cfg, "lambda3", 0.0)
        # "cosine" keeps the paper Eq. (34) form (GT class only, no negatives);
        # "nce" uses softmax over all seen classes so the term has real
        # discriminative gradient instead of being dwarfed by L_cls.
        self.align_mode = str(getattr(loss_cfg, "align_mode", "cosine")).lower()
        if self.align_mode not in ("cosine", "nce"):
            raise ValueError("model.loss.align_mode must be 'cosine' or 'nce'.")
        # "gt" keeps the paper Eq. (35) form (GT row only); "all" regularizes
        # every class prototype so the refiner cannot drift on unlabeled rows;
        # "unseen" (APR) regularizes unseen rows only, freeing seen prototypes
        # to fit the seen classifier.
        self.pres_target = str(getattr(loss_cfg, "pres_target", "gt")).lower()
        if self.pres_target not in ("gt", "all", "unseen"):
            raise ValueError("model.loss.pres_target must be 'gt', 'all' or 'unseen'.")
        # "mse" matches Eq. (35); "cosine" is scale-invariant for L2-normalized
        # vectors and gives a slightly stronger signal for small residuals.
        self.pres_mode = str(getattr(loss_cfg, "pres_mode", "mse")).lower()
        if self.pres_mode not in ("mse", "cosine"):
            raise ValueError("model.loss.pres_mode must be 'mse' or 'cosine'.")

        # Transfer-alignment (L_trans): distills the frozen phi-space class
        # constellation (seen + unseen) into the visual space of the real
        # seen class means. lambda_trans=0 disables it.
        self.lambda_trans = float(getattr(loss_cfg, "lambda_trans", 0.0))
        self.tau_visual = float(getattr(loss_cfg, "tau_visual", self.tau))
        self.tau_teacher = float(getattr(loss_cfg, "tau_teacher", 0.1))

        # Semantic consistency (Eq. 33): keeps the projected prototype aligned
        # with the frozen text encoding phi(t_c). Only meaningful once Ws is
        # not the identity (ws_rank>0 or a fitted Ws); lambda_sem=0 disables.
        self.lambda_sem = float(getattr(loss_cfg, "lambda_sem", 0.0))

        # v10 attention supervision (L_attn, PSVMA Eq. 3 style): MSE between
        # the predicted attribute presence (GMP over patch attention) and the
        # L2-normalized GT attribute row. lambda_attn=0 disables.
        self.lambda_attn = float(getattr(loss_cfg, "lambda_attn", 0.0))
        # v11 Fix 1 (L_fm): frozen-anchor cosine retention for v_tilde —
        # WiSE-FT in feature space. Every excessive rotation of v_tilde away
        # from normalize(cls) pays a budget; scale-invariant (cosine), unlike
        # a raw weight-norm penalty. lambda_fm=0 disables.
        self.lambda_fm = float(getattr(loss_cfg, "lambda_fm", 0.0))
        # v11 Fix 3 (L_zs): distill the FROZEN zero-shot posterior
        # q = softmax(tau*cos(cls_anchor, phi_y)) over ALL classes into the
        # actual refined scoring path p = softmax(tau*scores_all). First
        # gradient source containing unseen classes that flows through the
        # exact eval path (W_multi -> v_tilde, Ws -> p_all, IPR -> p_ref_all).
        # Replaces L_trans (which taught m_s — detached from the scoring
        # path — and was empirically 10u worse). lambda_zs=0 disables.
        self.lambda_zs = float(getattr(loss_cfg, "lambda_zs", 0.0))
        # v11 Fix 4 (L_rel): relational KD (RKD-D, Park et al. CVPR 2019)
        # over the FULL prototype set — preserves the pairwise distance
        # distribution of the frozen phi constellation inside Ws(φ). Norm-
        # penalty (ws_anchor) cannot protect relations; this does.
        self.lambda_rel = float(getattr(loss_cfg, "lambda_rel", 0.0))
        # v11 Fix 2b (L_tan): tangent-budget penalty for the IPR residual —
        # mean tan(angle) of the prototype rotation p_sem -> p̂, the
        # scale-invariant geometric quantity of "bending" (companion of L_fm
        # on the text side). lambda_tan=0 disables.
        self.lambda_tan = float(getattr(loss_cfg, "lambda_tan", 0.0))

    def forward(
        self,
        s_c,
        y,
        v_tilde=None,
        p_sem=None,
        p_ref=None,
        m_s=None,
        p_all=None,
        teacher=None,
        phi_t=None,
        p_sem_unnorm=None,
        p_ref_all=None,
        p_sem_all=None,
        pres_mask=None,
        attr_pred=None,
        gt_attr=None,
        cls_anchor=None,
        scores_all=None,
        phi_all=None,
        p_sem_all_rel=None,
    ):
        """
        Args:
            s_c: Compatibility scores (B, C)
            y: Ground-truth class labels (B,)
            v_tilde: Aggregated visual features (B, D)
            p_sem: Normalized semantic prototypes (C, D), i.e. tilde p_sem
            p_ref: Refined semantic prototypes (B, C, D)
            m_s: Normalized projections of the real seen-class pooled means,
                 shape (S, D) — used only by L_trans.
            p_all: Normalized prototypes for ALL classes (seen + unseen),
                   shape (C_all, D) — used only by L_trans.
            teacher: Frozen soft teacher distribution over all classes,
                     shape (S, C_all) — used only by L_trans.
            phi_t: Frozen text encodings (C, D) — used only by L_sem (Eq. 33).
            p_sem_unnorm: Unnormalized semantic prototypes (C, D) — used only
                          by L_sem.
            p_ref_all/p_sem_all: refined/semantic prototypes over ALL classes
                     (B, C_all, D) / (C_all, D) — used by pres_target in
                     ("all", "unseen") when the training bank is seen-only;
                     fall back to p_ref/p_sem when not provided.
            pres_mask: Boolean tensor (C_all,) selecting which rows of the
                     full bank L_pres applies to — used by pres_target=
                     "unseen" (True at unseen rows).
            attr_pred: Predicted attribute presence (B, n_attr) from the
                     cross-attention IPR attribute branch.
            gt_attr: GT attribute rows for the batch labels (B, n_attr);
                     target is L2-normalized here (refiner returns raw preds).
        Returns:
            total_loss, loss_dict
        """
        B = s_c.shape[0]
        required = {
            "v_tilde": v_tilde,
            "p_sem": p_sem,
            "p_ref": p_ref,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Full HSPRN loss requires intermediate tensors; missing: "
                + ", ".join(missing)
            )

        # ==================================================================
        # v12 Tri-Loss Mode (Streamlined 3-Component Loss Formulation)
        # ==================================================================
        if self.loss_mode == "tri_loss":
            # 1) Task Loss (Full-Bank 200-class Zero-Shot Soft Distillation)
            if cls_anchor is None or scores_all is None or phi_all is None:
                raise ValueError("tri_loss mode requires cls_anchor, scores_all and phi_all.")
            if scores_all.shape[-1] != phi_all.shape[0]:
                raise ValueError(
                    f"Tri-Loss L_task shape mismatch: scores_all has {scores_all.shape[-1]} "
                    f"columns, phi_all has {phi_all.shape[0]} classes."
                )
            with torch.no_grad():
                phi_n = F.normalize(phi_all, dim=-1)
                teacher_logits = (cls_anchor @ phi_n.T) / self.tau
                q = F.softmax(teacher_logits, dim=-1)
            log_p = F.log_softmax(scores_all / self.tau, dim=-1)
            # Standard KD gradient scale (no tau^2 reduction since L_task is primary)
            l_task = F.kl_div(log_p, q, reduction="batchmean")

            # Optional auxiliary seen-only hard CE (Trial 2 hybrid)
            l_cls = None
            if self.lambda_cls > 0:
                l_cls = F.cross_entropy(s_c / self.tau, y)

            # 2) Visual Feature Anchor Loss (WiSE-FT Cosine Retention)
            if cls_anchor is None or v_tilde is None:
                raise ValueError("tri_loss visual anchor requires cls_anchor and v_tilde.")
            l_visual = (1.0 - F.cosine_similarity(v_tilde, cls_anchor, dim=-1)).mean()

            # 3) Attribute Orientation Loss (IPR Fine-Grained Grounding)
            l_attr = None
            if self.lambda_attr > 0:
                if attr_pred is None or gt_attr is None:
                    raise ValueError("tri_loss attribute grounding requires attr_pred and gt_attr.")
                if self.attr_loss_type == "cosine":
                    l_attr = (1.0 - F.cosine_similarity(attr_pred, F.normalize(gt_attr, dim=-1), dim=-1)).mean()
                else:
                    l_attr = F.mse_loss(attr_pred, F.normalize(gt_attr, dim=-1))

            # 4) Relational KD for Ws (Phase 3)
            l_rel = None
            if self.lambda_rel > 0:
                if phi_all is None or p_sem_all_rel is None:
                    raise ValueError("tri_loss lambda_rel>0 requires phi_all and p_sem_all_rel.")
                with torch.no_grad():
                    phi_n = F.normalize(phi_all, dim=-1)
                    d0 = torch.cdist(phi_n, phi_n)
                    p0 = F.softmax(-d0, dim=-1)
                d1 = torch.cdist(p_sem_all_rel, p_sem_all_rel)
                p1 = F.log_softmax(-d1, dim=-1)
                l_rel = F.kl_div(p1, p0, reduction="batchmean")

            total_loss = self.lambda_task * l_task + self.lambda_visual * l_visual
            if l_attr is not None:
                total_loss = total_loss + self.lambda_attr * l_attr
            if l_cls is not None:
                total_loss = total_loss + self.lambda_cls * l_cls
            if l_rel is not None:
                total_loss = total_loss + self.lambda_rel * l_rel

            loss_dict = {
                "loss_task": float((self.lambda_task * l_task).detach().cpu()),
                "loss_visual": float((self.lambda_visual * l_visual).detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "cos_v_cls": float(F.cosine_similarity(v_tilde, cls_anchor, dim=-1).mean().detach().cpu()),
            }
            if l_attr is not None:
                loss_dict["loss_attr"] = float((self.lambda_attr * l_attr).detach().cpu())
            if l_cls is not None:
                loss_dict["loss_cls"] = float((self.lambda_cls * l_cls).detach().cpu())
            if l_rel is not None:
                loss_dict["loss_rel"] = float((self.lambda_rel * l_rel).detach().cpu())
            return total_loss, loss_dict

        # 1) Classification (Eq. 32)
        l_cls = F.cross_entropy(s_c / self.tau, y)

        # 2) Visual-Semantic alignment (Eq. 34)
        if v_tilde.shape[-1] != p_sem.shape[-1]:
            raise ValueError(
                "Eq. (34) requires v_tilde and tilde p_sem_y to share the same dimension, "
                f"got {v_tilde.shape[-1]} and {p_sem.shape[-1]}."
            )
        p_sem_y = p_sem[y]
        if self.align_mode == "nce":
            # Softmax over all seen prototypes: pull v_tilde to its class and
            # push it away from the other seen classes (same tau as L_cls).
            logits = F.cosine_similarity(
                v_tilde.unsqueeze(1), p_sem.unsqueeze(0), dim=-1
            ) / self.tau
            l_align = F.cross_entropy(logits, y)
        else:
            l_align = (1.0 - F.cosine_similarity(v_tilde, p_sem_y, dim=-1)).mean()

        # 3) Prototype preservation (Eq. 35)
        if p_ref.shape[-1] != p_sem.shape[-1]:
            raise ValueError(
                "Eq. (35) requires tilde p_ref_y and tilde p_sem_y to share the same "
                f"dimension, got {p_ref.shape[-1]} and {p_sem.shape[-1]}."
            )
        if self.pres_target in ("all", "unseen"):
            # Prefer the full-class bank (seen + unseen) when the trainer
            # supplies it; otherwise fall back to the training bank.
            p_ref_target = p_ref_all if p_ref_all is not None else p_ref
            p_sem_target = p_sem_all if p_sem_all is not None else p_sem
            if self.pres_target == "unseen":
                if pres_mask is None:
                    raise ValueError(
                        "pres_target='unseen' requires a pres_mask over the "
                        "full-class bank."
                    )
                p_ref_target = p_ref_target[:, pres_mask]
                p_sem_target = p_sem_target[pres_mask]
                if p_sem_target.shape[0] == 0:
                    raise ValueError("pres_mask selects zero rows.")
            p_sem_all_rows = p_sem_target.unsqueeze(0)  # (1, C_sel, D)
            if self.pres_mode == "cosine":
                l_pres = (
                    1.0 - F.cosine_similarity(p_ref_target, p_sem_all_rows, dim=-1)
                ).mean()
            else:
                l_pres = ((p_ref_target - p_sem_all_rows) ** 2).mean()
        else:
            batch_indices = torch.arange(B, device=y.device)
            p_ref_y = p_ref[batch_indices, y]
            if self.pres_mode == "cosine":
                l_pres = (
                    1.0 - F.cosine_similarity(p_ref_y, p_sem_y, dim=-1)
                ).mean()
            else:
                l_pres = ((p_ref_y - p_sem_y) ** 2).mean(dim=-1).mean()

        # 4) Transfer-alignment over the full class constellation (L_trans).
        # Student: soft 200-class distribution of each real seen-class mean
        # (m_s) matched against all prototypes incl. unseen (p_all).
        # Teacher: frozen phi-space soft nearest-neighbor distribution.
        l_trans = None
        if self.lambda_trans > 0:
            if m_s is None or p_all is None or teacher is None:
                raise ValueError(
                    "lambda_trans>0 requires m_s, p_all, and teacher tensors."
                )
            student_logits = (m_s @ p_all.T) / self.tau_visual
            log_student = F.log_softmax(student_logits, dim=-1)
            l_trans = F.kl_div(
                log_student, teacher, reduction="batchmean"
            )

        # 5) Semantic consistency (Eq. 33): 1 - cos(p_sem_y, phi(t_y)).
        l_sem = None
        if self.lambda_sem > 0:
            if phi_t is None or p_sem_unnorm is None:
                raise ValueError(
                    "lambda_sem>0 requires phi_t and p_sem_unnorm tensors."
                )
            phi_y = F.normalize(phi_t[y], dim=-1)
            l_sem = (
                1.0 - F.cosine_similarity(p_sem_unnorm[y], phi_y, dim=-1)
            ).mean()

        # 6) Attention supervision (L_attn): attribute presence from patch
        # attention vs the L2-normalized GT attribute row of the label.
        l_attn = None
        if self.lambda_attn > 0:
            if attr_pred is None or gt_attr is None:
                raise ValueError(
                    "lambda_attn>0 requires attr_pred and gt_attr tensors."
                )
            if attr_pred.shape != gt_attr.shape:
                raise ValueError(
                    f"attr_pred shape {tuple(attr_pred.shape)} does not match "
                    f"gt_attr shape {tuple(gt_attr.shape)}."
                )
            l_attn = F.mse_loss(attr_pred, F.normalize(gt_attr, dim=-1))

        # 7) Frozen-anchor cosine retention (L_fm, v11 Fix 1):
        #    mean_b (1 - cos(v_tilde_b, cls_anchor_b)).
        l_fm = None
        if self.lambda_fm > 0:
            if cls_anchor is None:
                raise ValueError(
                    "lambda_fm>0 requires the cls_anchor tensor "
                    "(anchored MFA vision path)."
                )
            if cls_anchor.shape != v_tilde.shape:
                raise ValueError(
                    f"cls_anchor shape {tuple(cls_anchor.shape)} != v_tilde "
                    f"shape {tuple(v_tilde.shape)}."
                )
            l_fm = (
                1.0 - F.cosine_similarity(v_tilde, cls_anchor, dim=-1)
            ).mean()

        # 8) Zero-shot posterior distillation (L_zs, v11 Fix 3): KL between
        #    the frozen zero-shot posterior (cls_anchor vs raw phi over ALL
        #    classes) and the student posterior over the ACTUAL refined
        #    scoring path (scores_all from the full-class refined bank).
        #    tau² scaling follows the standard KD gradient form
        #    (Hinton et al.) so the term is invariant to the CE scale.
        l_zs = None
        if self.lambda_zs > 0:
            if cls_anchor is None or scores_all is None or phi_all is None:
                raise ValueError(
                    "lambda_zs>0 requires cls_anchor, scores_all and phi_all."
                )
            if scores_all.shape[-1] != phi_all.shape[0]:
                raise ValueError(
                    "L_zs class-count mismatch: scores_all has "
                    f"{scores_all.shape[-1]} columns, phi_all has "
                    f"{phi_all.shape[0]} classes."
                )
            with torch.no_grad():
                phi_n = F.normalize(phi_all, dim=-1)
                teacher_logits = (cls_anchor @ phi_n.T) / self.tau
                q = F.softmax(teacher_logits, dim=-1)
            log_p = F.log_softmax(scores_all / self.tau, dim=-1)
            # F.kl_div(input=log_q(student), target=p(teacher)).
            l_zs = (self.tau ** 2) * F.kl_div(log_p, q, reduction="batchmean")

        # 9) Relational KD for the prototype bank (L_rel, v11 Fix 4):
        #    row-softmax over pairwise Euclidean distances of the LEARNED
        #    full-class prototypes must match that of the FROZEN phi rows.
        #    Row-softmax normalization is scale-tolerant: a global dilation
        #    of Ws (useful for seen discrimination) is free; only
        #    idiosyncratic per-pair deformations pay.
        l_rel = None
        if self.lambda_rel > 0:
            if phi_all is None or p_sem_all_rel is None:
                raise ValueError(
                    "lambda_rel>0 requires phi_all and the full-class "
                    "semantic prototypes p_sem_all_rel."
                )
            if p_sem_all_rel.shape[0] != phi_all.shape[0]:
                raise ValueError(
                    "L_rel class-count mismatch: p_sem_all_rel has "
                    f"{p_sem_all_rel.shape[0]} rows, phi_all has "
                    f"{phi_all.shape[0]}."
                )
            with torch.no_grad():
                phi_n = F.normalize(phi_all, dim=-1)
                d0 = torch.cdist(phi_n, phi_n)                 # frozen
                p0 = F.softmax(-d0, dim=-1)
            d1 = torch.cdist(p_sem_all_rel, p_sem_all_rel)     # learned
            p1 = F.log_softmax(-d1, dim=-1)
            l_rel = F.kl_div(p1, p0, reduction="batchmean")

        # 10) Tangent budget for the IPR residual (L_tan, v11 Fix 2b):
        #     mean_c tan(angle) of the rotation p_sem -> p̂ on the unit
        #     sphere, computed on the full-class bank so unseen rows are
        #     budgeted too. Scale-invariant: shrinking the residual does
        #     not shrink the ANGLE unless the direction changes.
        l_tan = None
        if self.lambda_tan > 0:
            if p_ref_all is None or p_sem_all_rel is None:
                raise ValueError(
                    "lambda_tan>0 requires the full-class refined bank "
                    "p_ref_all and semantic bank p_sem_all_rel."
                )
            # Mean over the batch first: p_ref_all is per-image [B, C, d];
            # every sample's rotation pays the budget, not just element 0.
            p_hat = (p_ref_all.mean(dim=0) if p_ref_all.dim() == 3
                     else p_ref_all)
            # p_hat (mean of unit vectors) may shrink slightly — renorm so
            # the budget measures the ANGLE only.
            p_hat = F.normalize(p_hat, dim=-1)
            p_sem_n = F.normalize(p_sem_all_rel, dim=-1)
            dot = (p_hat * p_sem_n).sum(dim=-1, keepdim=True)
            tangent = p_hat - dot * p_sem_n
            l_tan = (tangent.norm(dim=-1) / dot.clamp(min=1e-6)).mean()

        total_loss = (
            self.lambda1 * l_cls +
            self.lambda2 * l_align +
            self.lambda3 * l_pres
        )
        if l_trans is not None:
            total_loss = total_loss + self.lambda_trans * l_trans
        if l_sem is not None:
            total_loss = total_loss + self.lambda_sem * l_sem
        if l_attn is not None:
            total_loss = total_loss + self.lambda_attn * l_attn
        if l_fm is not None:
            total_loss = total_loss + self.lambda_fm * l_fm
        if l_zs is not None:
            total_loss = total_loss + self.lambda_zs * l_zs
        if l_rel is not None:
            total_loss = total_loss + self.lambda_rel * l_rel
        if l_tan is not None:
            total_loss = total_loss + self.lambda_tan * l_tan

        loss_dict = {
            'loss_cls': float(l_cls.detach().cpu()),
            'loss_align': float(l_align.detach().cpu()),
            'loss_pres': float(l_pres.detach().cpu()),
            'total_loss': float(total_loss.detach().cpu()),
        }
        if l_trans is not None:
            loss_dict['loss_trans'] = float(
                (self.lambda_trans * l_trans).detach().cpu()
            )
        if l_sem is not None:
            loss_dict['loss_sem'] = float(
                (self.lambda_sem * l_sem).detach().cpu()
            )
        if l_attn is not None:
            loss_dict['loss_attn'] = float(
                (self.lambda_attn * l_attn).detach().cpu()
            )
        if l_fm is not None:
            loss_dict['loss_fm'] = float(
                (self.lambda_fm * l_fm).detach().cpu()
            )
        if l_zs is not None:
            loss_dict['loss_zs'] = float(
                (self.lambda_zs * l_zs).detach().cpu()
            )
        if l_rel is not None:
            loss_dict['loss_rel'] = float(
                (self.lambda_rel * l_rel).detach().cpu()
            )
        if l_tan is not None:
            loss_dict['loss_tan'] = float(
                (self.lambda_tan * l_tan).detach().cpu()
            )
        # Monitoring (Fix 5): mean cosine of the visual rotation budget.
        if cls_anchor is not None and v_tilde is not None \
                and cls_anchor.shape == v_tilde.shape:
            loss_dict['cos_v_cls'] = float(
                F.cosine_similarity(v_tilde, cls_anchor, dim=-1)
                .mean().detach().cpu()
            )
        return total_loss, loss_dict
