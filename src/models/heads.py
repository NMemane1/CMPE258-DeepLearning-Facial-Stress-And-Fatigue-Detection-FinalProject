"""Multi-task heads: shared MLP trunk + task-specific classifiers."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class HeadsOutput:
    stress_logits: torch.Tensor
    fatigue_logits: torch.Tensor
    shared_features: torch.Tensor


class SharedMLP(nn.Module):
    """Shared trunk over the backbone CLS features."""

    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationHead(nn.Module):
    """A small classifier head (single linear + BN)."""

    def __init__(self, in_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.dropout(x)
        return self.fc(x)


class MultiTaskHeads(nn.Module):
    """Assembles shared trunk + per-task heads."""

    def __init__(
        self,
        embedding_dim: int,
        shared_hidden: int,
        num_stress_classes: int,
        num_fatigue_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.trunk = SharedMLP(embedding_dim, shared_hidden, dropout)
        self.stress_head = ClassificationHead(shared_hidden, num_stress_classes, dropout)
        self.fatigue_head = ClassificationHead(shared_hidden, num_fatigue_classes, dropout)

    def forward(self, cls_features: torch.Tensor) -> HeadsOutput:
        shared = self.trunk(cls_features)
        return HeadsOutput(
            stress_logits=self.stress_head(shared),
            fatigue_logits=self.fatigue_head(shared),
            shared_features=shared,
        )
