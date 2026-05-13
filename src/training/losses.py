"""Loss functions: class-weighted CE, focal loss, multi-task masked sum."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification.

    L = -α_c * (1 - p_c)^γ * log(p_c)

    Supports per-class α weights and ignores samples with label == ignore_index.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != self.ignore_index
        if mask.sum() == 0:
            return logits.sum() * 0.0
        logits, target = logits[mask], target[mask]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        tgt_log_p = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        tgt_p = probs.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - tgt_p).clamp(min=1e-8) ** self.gamma
        loss = -focal_term * tgt_log_p

        if self.alpha is not None:
            a = self.alpha.to(logits.device)[target]
            loss = a * loss
        return loss.mean()


class MaskedCrossEntropy(nn.Module):
    """CE that ignores -1 labels (so we can combine single-task datasets)."""

    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int = -1):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != self.ignore_index
        if mask.sum() == 0:
            return logits.sum() * 0.0
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(logits[mask], target[mask], weight=w)


@dataclass
class MultiTaskLossOutput:
    total: torch.Tensor
    stress_ce: torch.Tensor
    fatigue_ce: torch.Tensor
    stress_focal: torch.Tensor


class MultiTaskLoss(nn.Module):
    """Weighted sum of:
        loss = w_s * CE(stress) + w_f * CE(fatigue) + w_focal * focal(stress)
    """

    def __init__(
        self,
        stress_class_weight: torch.Tensor | None = None,
        fatigue_class_weight: torch.Tensor | None = None,
        weight_stress: float = 1.0,
        weight_fatigue: float = 1.0,
        weight_focal: float = 0.5,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.ce_stress = MaskedCrossEntropy(weight=stress_class_weight)
        self.ce_fatigue = MaskedCrossEntropy(weight=fatigue_class_weight)
        self.focal_stress = FocalLoss(gamma=focal_gamma, alpha=stress_class_weight)
        self.weight_stress = weight_stress
        self.weight_fatigue = weight_fatigue
        self.weight_focal = weight_focal

    def forward(
        self,
        stress_logits: torch.Tensor,
        fatigue_logits: torch.Tensor,
        stress_target: torch.Tensor,
        fatigue_target: torch.Tensor,
    ) -> MultiTaskLossOutput:
        ce_s = self.ce_stress(stress_logits, stress_target)
        ce_f = self.ce_fatigue(fatigue_logits, fatigue_target)
        fl_s = self.focal_stress(stress_logits, stress_target)
        total = (
            self.weight_stress * ce_s
            + self.weight_fatigue * ce_f
            + self.weight_focal * fl_s
        )
        return MultiTaskLossOutput(
            total=total, stress_ce=ce_s, fatigue_ce=ce_f, stress_focal=fl_s,
        )
