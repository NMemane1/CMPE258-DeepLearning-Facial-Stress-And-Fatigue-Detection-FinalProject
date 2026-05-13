"""Training callbacks: early stopping, LR monitoring."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track whether validation metric has stopped improving."""
    patience: int = 3
    mode: str = "max"   # "max" for accuracy-like metrics, "min" for loss
    min_delta: float = 0.0

    def __post_init__(self):
        self.best = float("-inf") if self.mode == "max" else float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = (value > self.best + self.min_delta) if self.mode == "max" \
                   else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


class CheckpointTracker:
    """Track top-K checkpoints by a metric."""
    def __init__(self, k: int = 3, mode: str = "max"):
        self.k = k
        self.mode = mode
        self.entries: list[tuple[float, str]] = []

    def update(self, value: float, path: str) -> tuple[bool, str | None]:
        """Add a new checkpoint; return (kept, evicted_path_or_None)."""
        self.entries.append((value, path))
        self.entries.sort(key=lambda t: t[0], reverse=(self.mode == "max"))
        evicted = None
        if len(self.entries) > self.k:
            _, evicted = self.entries.pop()
        return any(p == path for _, p in self.entries), evicted
