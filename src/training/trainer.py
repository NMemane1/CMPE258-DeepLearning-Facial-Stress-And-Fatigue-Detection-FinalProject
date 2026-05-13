"""Training loop.

Designed to be self-contained and runnable from CLI or notebook.
Logs to W&B and TensorBoard simultaneously.
"""
from __future__ import annotations

import os
import time
import json
import math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.models.stress_fatigue_model import StressFatigueModel
from src.training.losses import MultiTaskLoss
from src.evaluation.metrics import compute_metrics_dict


def _maybe_wandb_init(cfg):
    try:
        import wandb
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.experiment.name,
            notes=cfg.experiment.notes,
            config=dict(cfg) if hasattr(cfg, "__iter__") else cfg,
        )
        return wandb
    except Exception as e:
        print(f"[warn] wandb init failed: {e}. Continuing without W&B.")
        return None


def _build_scheduler(optimizer, num_training_steps: int, warmup_steps: int, min_lr_factor: float):
    """Linear warmup + cosine decay."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_factor, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _get_device(precision: str) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_metric: float = -1.0
    epochs_no_improve: int = 0


class Trainer:
    """Main trainer object."""

    def __init__(self, cfg, model: StressFatigueModel, loss_fn: MultiTaskLoss,
                 train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        self.cfg = cfg
        self.device = _get_device(cfg.training.precision)
        print(f"[trainer] device = {self.device}")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.output_dir = Path(cfg.experiment.output_dir) / cfg.experiment.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)

        self.tb = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        self.wandb = _maybe_wandb_init(cfg)

        param_groups = model.get_optimizer_param_groups(
            lr_backbone=cfg.training.lr_backbone,
            lr_head=cfg.training.lr_head,
            weight_decay=cfg.training.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(param_groups, betas=tuple(cfg.training.betas))

        total_steps = len(train_loader) * cfg.training.num_epochs
        self.scheduler = _build_scheduler(
            self.optimizer,
            num_training_steps=total_steps,
            warmup_steps=cfg.training.warmup_steps,
            min_lr_factor=cfg.training.min_lr / max(cfg.training.lr_head, 1e-12),
        )

        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda" and cfg.training.precision == "fp16"))
        self.state = TrainState()

        param_info = model.parameter_summary()
        print(f"[trainer] params: total={param_info['total']:,} trainable={param_info['trainable']:,} ({param_info['trainable_pct']:.1f}%)")

    # ------------------------------------------------------------
    # Train / val loops
    # ------------------------------------------------------------

    def fit(self):
        for epoch in range(self.cfg.training.num_epochs):
            self.state.epoch = epoch
            t0 = time.time()
            train_loss = self._train_one_epoch()
            val_metrics = self._evaluate(self.val_loader, split_name="val")
            dt = time.time() - t0

            primary = val_metrics["balanced_acc_mean"]
            improved = primary > self.state.best_metric
            if improved:
                self.state.best_metric = primary
                self.state.epochs_no_improve = 0
                self._save_checkpoint("best")
            else:
                self.state.epochs_no_improve += 1

            print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
                  f"val_balanced_acc={primary:.4f} (best={self.state.best_metric:.4f}) "
                  f"time={dt:.1f}s {'★' if improved else ''}")

            if self.state.epochs_no_improve >= self.cfg.training.early_stopping_patience:
                print(f"[trainer] early stopping after {epoch+1} epochs")
                break

        # final test eval with best checkpoint
        self._load_checkpoint("best")
        test_metrics = self._evaluate(self.test_loader, split_name="test")
        print(f"[trainer] final test metrics: {json.dumps(test_metrics, indent=2)}")

        with open(self.output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        self.tb.close()
        if self.wandb:
            self.wandb.finish()

        return test_metrics

    def _train_one_epoch(self) -> float:
        self.model.train()
        running = 0.0
        n_batches = 0
        for batch in self.train_loader:
            images = batch["image"].to(self.device, non_blocking=True)
            stress = batch["stress_label"].to(self.device, non_blocking=True)
            fatigue = batch["fatigue_label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            use_amp = self.device.type == "cuda" and self.cfg.training.precision in ("fp16",)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = self.model(images)
                loss_out = self.loss_fn(
                    out.stress_logits, out.fatigue_logits,
                    stress, fatigue,
                )
                loss = loss_out.total / max(self.cfg.training.accumulation_steps, 1)

            if use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.gradient_clip)
                self.optimizer.step()

            self.scheduler.step()
            running += loss_out.total.item()
            n_batches += 1
            self.state.step += 1

            if self.state.step % self.cfg.training.log_every_n_steps == 0:
                lr = self.scheduler.get_last_lr()[0]
                self.tb.add_scalar("train/loss", loss_out.total.item(), self.state.step)
                self.tb.add_scalar("train/loss_stress_ce", loss_out.stress_ce.item(), self.state.step)
                self.tb.add_scalar("train/loss_fatigue_ce", loss_out.fatigue_ce.item(), self.state.step)
                self.tb.add_scalar("train/loss_stress_focal", loss_out.stress_focal.item(), self.state.step)
                self.tb.add_scalar("train/lr", lr, self.state.step)
                if self.wandb:
                    self.wandb.log({
                        "train/loss": loss_out.total.item(),
                        "train/loss_stress_ce": loss_out.stress_ce.item(),
                        "train/loss_fatigue_ce": loss_out.fatigue_ce.item(),
                        "train/loss_stress_focal": loss_out.stress_focal.item(),
                        "train/lr": lr,
                        "step": self.state.step,
                    })

        return running / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, split_name: str) -> dict:
        self.model.eval()
        all_stress_preds, all_stress_tgt, all_stress_probs = [], [], []
        all_fatigue_preds, all_fatigue_tgt, all_fatigue_probs = [], [], []

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            out = self.model(images)
            s_probs = torch.softmax(out.stress_logits, dim=-1).cpu().numpy()
            f_probs = torch.softmax(out.fatigue_logits, dim=-1).cpu().numpy()
            s_tgt = batch["stress_label"].numpy()
            f_tgt = batch["fatigue_label"].numpy()
            all_stress_probs.append(s_probs)
            all_stress_tgt.append(s_tgt)
            all_fatigue_probs.append(f_probs)
            all_fatigue_tgt.append(f_tgt)

        s_probs = np.concatenate(all_stress_probs)
        s_tgt = np.concatenate(all_stress_tgt)
        f_probs = np.concatenate(all_fatigue_probs)
        f_tgt = np.concatenate(all_fatigue_tgt)

        metrics = compute_metrics_dict(s_probs, s_tgt, f_probs, f_tgt)

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.tb.add_scalar(f"{split_name}/{k}", v, self.state.step)
        if self.wandb:
            self.wandb.log({f"{split_name}/{k}": v for k, v in metrics.items()
                            if isinstance(v, (int, float))})
        return metrics

    # ------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------

    def _save_checkpoint(self, name: str):
        path = self.output_dir / "checkpoints" / name
        self.model.save_pretrained(path)

    def _load_checkpoint(self, name: str):
        path = self.output_dir / "checkpoints" / name
        if (path / "pytorch_model.bin").exists():
            state = torch.load(path / "pytorch_model.bin", map_location=self.device)
            self.model.load_state_dict(state)
