"""Main training entry point.

Usage:
    python -m src.training.train --config src/config/config.yaml
    python -m src.training.train --config src/config/config.yaml \
        training.lr_head=1e-4 model.unfreeze_from_layer=8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf

from src.data.dataset import (
    DrowsinessDataset, FER2013Dataset, CombinedFacialDataset,
    make_subject_grouped_split, compute_class_weights,
)
from src.data.transforms import build_train_transform, build_eval_transform
from src.models.stress_fatigue_model import StressFatigueModel
from src.training.losses import MultiTaskLoss
from src.training.trainer import Trainer

from torch.utils.data import DataLoader


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="src/config/config.yaml")
    p.add_argument("overrides", nargs="*",
                   help="dotted overrides, e.g. training.lr_head=1e-4")
    return p.parse_args()


def load_config(path: str, overrides: list[str]):
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    print("=" * 60)
    print("CONFIG")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.experiment.seed)

    # ---- build datasets ----
    train_tf = build_train_transform(cfg)
    eval_tf  = build_eval_transform(cfg)

    print("\n[train] loading drowsiness dataset…")
    drowsy_full = DrowsinessDataset(cfg.data.drowsiness_root, transform=None)
    print(f"  → {len(drowsy_full)} samples")

    print("[train] loading FER-2013 dataset…")
    fer_full = FER2013Dataset(cfg.data.fer_root, transform=None)
    print(f"  → {len(fer_full)} samples")

    drow_split = make_subject_grouped_split(
        drowsy_full,
        cfg.data.train_split, cfg.data.val_split, cfg.data.test_split,
        seed=cfg.experiment.seed,
    )
    fer_split = make_subject_grouped_split(
        fer_full,
        cfg.data.train_split, cfg.data.val_split, cfg.data.test_split,
        seed=cfg.experiment.seed,
    )

    train_ds = CombinedFacialDataset([
        DrowsinessDataset(cfg.data.drowsiness_root, indices=drow_split.train, transform=train_tf),
        FER2013Dataset(cfg.data.fer_root, indices=fer_split.train, transform=train_tf),
    ])
    val_ds = CombinedFacialDataset([
        DrowsinessDataset(cfg.data.drowsiness_root, indices=drow_split.val, transform=eval_tf),
        FER2013Dataset(cfg.data.fer_root, indices=fer_split.val, transform=eval_tf),
    ])
    test_ds = CombinedFacialDataset([
        DrowsinessDataset(cfg.data.drowsiness_root, indices=drow_split.test, transform=eval_tf),
        FER2013Dataset(cfg.data.fer_root, indices=fer_split.test, transform=eval_tf),
    ])

    print(f"\n[train] splits: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size, shuffle=False, drop_last=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory,
    )

    # ---- class weights ----
    stress_w = compute_class_weights(fer_full, "stress", cfg.model.num_stress_classes)
    fatigue_w = compute_class_weights(drowsy_full, "fatigue", cfg.model.num_fatigue_classes)
    print(f"[train] stress class weights: {stress_w.tolist()}")
    print(f"[train] fatigue class weights: {fatigue_w.tolist()}")

    # ---- model + loss ----
    model = StressFatigueModel(
        backbone_name=cfg.model.backbone,
        embedding_dim=cfg.model.embedding_dim,
        shared_hidden=cfg.model.shared_hidden,
        num_stress_classes=cfg.model.num_stress_classes,
        num_fatigue_classes=cfg.model.num_fatigue_classes,
        dropout=cfg.model.dropout,
        unfreeze_from_layer=cfg.model.unfreeze_from_layer,
    )

    loss_fn = MultiTaskLoss(
        stress_class_weight=stress_w,
        fatigue_class_weight=fatigue_w,
        weight_stress=cfg.training.loss_weights.stress,
        weight_fatigue=cfg.training.loss_weights.fatigue,
        weight_focal=cfg.training.loss_weights.focal if cfg.model.use_focal_loss else 0.0,
        focal_gamma=cfg.model.focal_gamma,
    )

    # ---- train ----
    trainer = Trainer(cfg, model, loss_fn, train_loader, val_loader, test_loader)
    test_metrics = trainer.fit()

    print("\n[done] training complete.")
    return test_metrics


if __name__ == "__main__":
    sys.exit(main() is not None)
