"""Dataset classes for facial stress & fatigue detection.

This module defines two PyTorch Datasets that produce a unified output format:
    {
        "image": torch.Tensor [3, H, W],
        "stress_label": torch.LongTensor (0=low, 1=moderate, 2=high), or -1 if N/A
        "fatigue_label": torch.LongTensor (0=alert, 1=drowsy), or -1 if N/A
        "subject_id": str (for subject-grouped splits)
    }

For multi-task training, labels can be missing (-1) for samples that come
from a dataset only labeling one task. The loss function ignores -1 labels
per-task. This lets us combine two single-task datasets without re-labeling.
"""
from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit


# ============================================================
# Constants — label maps
# ============================================================

# FER-2013 emotion → stress level mapping. See methodology.md §2.
FER_EMOTION_TO_STRESS = {
    "angry":    2,   # high stress
    "disgust":  2,   # high stress
    "fear":     2,   # high stress
    "sad":      1,   # moderate stress
    "happy":    0,   # low stress
    "surprise": 0,   # low stress
    "neutral":  0,   # low stress
}

FER_INT_TO_EMOTION = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise", 6: "neutral",
}

STRESS_CLASS_NAMES = ["low", "moderate", "high"]
FATIGUE_CLASS_NAMES = ["alert", "drowsy"]


# ============================================================
# Helper: deterministic subject ID hashing for FER
# ============================================================

def _hash_subject(path: str, buckets: int = 10000) -> str:
    """FER-2013 has no subject metadata. We pseudo-group by filename hash
    so train/test splits don't accidentally share the same image twice."""
    h = hashlib.md5(path.encode()).hexdigest()
    return f"fer_{int(h, 16) % buckets}"


# ============================================================
# Drowsiness dataset (Kaggle: dheerajperumandla/drowsiness-dataset)
# ============================================================

class DrowsinessDataset(Dataset):
    """Loads the Kaggle Drowsiness dataset.

    Expected directory layout after download:
        root/
            yawn/*.jpg          → drowsy
            no_yawn/*.jpg       → alert
            Closed/*.jpg        → drowsy
            Open/*.jpg          → alert
    """

    DIRS_DROWSY = {"yawn", "Closed"}
    DIRS_ALERT  = {"no_yawn", "Open"}

    def __init__(
        self,
        root: str,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.records = self._scan_root()
        if indices is not None:
            self.records = [self.records[i] for i in indices]

    def _scan_root(self):
        records = []
        for sub in self.DIRS_DROWSY | self.DIRS_ALERT:
            d = self.root / sub
            if not d.exists():
                continue
            label = 1 if sub in self.DIRS_DROWSY else 0
            for img_path in sorted(d.glob("*.*")):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                # use filename prefix as pseudo-subject ID
                subj = img_path.stem.split("_")[0]
                records.append({
                    "path": str(img_path),
                    "fatigue_label": label,
                    "stress_label": -1,         # not labeled
                    "subject_id": f"drowsy_{subj}",
                    "source": "drowsiness",
                })
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "stress_label": torch.tensor(rec["stress_label"], dtype=torch.long),
            "fatigue_label": torch.tensor(rec["fatigue_label"], dtype=torch.long),
            "subject_id": rec["subject_id"],
            "source": rec["source"],
        }


# ============================================================
# FER-2013 dataset
# ============================================================

class FER2013Dataset(Dataset):
    """FER-2013, with emotion → stress label remap.

    Supports two on-disk layouts:
        (a) ``fer2013.csv`` with columns [emotion, pixels, Usage]
        (b) folder-per-class layout: root/train/<emotion>/*.png
    """

    def __init__(
        self,
        root: str,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.records = self._scan_root()
        if indices is not None:
            self.records = [self.records[i] for i in indices]

    def _scan_root(self):
        records = []
        csv_path = self.root / "fer2013.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for i, row in df.iterrows():
                emotion_int = int(row["emotion"])
                emotion = FER_INT_TO_EMOTION[emotion_int]
                stress = FER_EMOTION_TO_STRESS[emotion]
                records.append({
                    "csv_index": i,
                    "pixels": row["pixels"],
                    "stress_label": stress,
                    "fatigue_label": -1,
                    "subject_id": _hash_subject(f"fer_{i}"),
                    "source": "fer2013",
                })
        else:
            # folder-per-class layout
            for split_dir in ("train", "test"):
                base = self.root / split_dir
                if not base.exists():
                    base = self.root  # flat layout fallback
                for emotion_dir in base.iterdir():
                    if not emotion_dir.is_dir():
                        continue
                    emotion = emotion_dir.name.lower()
                    if emotion not in FER_EMOTION_TO_STRESS:
                        continue
                    stress = FER_EMOTION_TO_STRESS[emotion]
                    for img_path in sorted(emotion_dir.glob("*.*")):
                        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                            continue
                        records.append({
                            "path": str(img_path),
                            "stress_label": stress,
                            "fatigue_label": -1,
                            "subject_id": _hash_subject(str(img_path)),
                            "source": "fer2013",
                        })
        return records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if "pixels" in rec:
            # CSV layout: pixels are space-separated string of 48*48 ints
            arr = np.fromstring(rec["pixels"], sep=" ", dtype=np.uint8).reshape(48, 48)
            img = Image.fromarray(arr).convert("RGB")
        else:
            img = Image.open(rec["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "stress_label": torch.tensor(rec["stress_label"], dtype=torch.long),
            "fatigue_label": torch.tensor(rec["fatigue_label"], dtype=torch.long),
            "subject_id": rec["subject_id"],
            "source": rec["source"],
        }


# ============================================================
# Split helpers
# ============================================================

@dataclass
class SplitIndices:
    train: list[int]
    val: list[int]
    test: list[int]


def make_subject_grouped_split(
    dataset: Dataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Stratified split that keeps all images from a subject in one split.

    Prevents subject leakage when the dataset has per-subject grouping.
    For datasets without real subject IDs (FER-2013), the pseudo-subject
    hash still works to spread images across splits roughly randomly.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    subject_ids = np.array([dataset.records[i]["subject_id"] for i in range(len(dataset))])
    indices = np.arange(len(dataset))

    # Step 1: split off test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(indices, groups=subject_ids))

    # Step 2: split remaining into train/val
    relative_val = val_frac / (train_frac + val_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
    sub_train_idx, sub_val_idx = next(
        gss2.split(train_val_idx, groups=subject_ids[train_val_idx])
    )

    train_idx = train_val_idx[sub_train_idx].tolist()
    val_idx = train_val_idx[sub_val_idx].tolist()

    return SplitIndices(train=train_idx, val=val_idx, test=test_idx.tolist())


# ============================================================
# Combined multi-task dataset
# ============================================================

class CombinedFacialDataset(Dataset):
    """Wraps multiple datasets behind one __getitem__.

    Each sample carries optional labels for each task; missing labels are -1.
    """

    def __init__(self, datasets: Sequence[Dataset]):
        self.datasets = datasets
        self.cum_lens = np.cumsum([0] + [len(d) for d in datasets])

    def __len__(self):
        return int(self.cum_lens[-1])

    def __getitem__(self, idx):
        ds_idx = int(np.searchsorted(self.cum_lens, idx, side="right") - 1)
        local_idx = idx - int(self.cum_lens[ds_idx])
        return self.datasets[ds_idx][local_idx]


def build_dataloaders(cfg, train_tf, eval_tf) -> dict[str, DataLoader]:
    """Build train/val/test dataloaders from config."""
    drowsiness = DrowsinessDataset(cfg.data.drowsiness_root, transform=None)
    fer = FER2013Dataset(cfg.data.fer_root, transform=None)

    drow_split = make_subject_grouped_split(
        drowsiness,
        cfg.data.train_split, cfg.data.val_split, cfg.data.test_split,
        seed=cfg.experiment.seed,
    )
    fer_split = make_subject_grouped_split(
        fer,
        cfg.data.train_split, cfg.data.val_split, cfg.data.test_split,
        seed=cfg.experiment.seed,
    )

    def make(ds_class, root, idx, tf):
        return ds_class(root, indices=idx, transform=tf)

    train_combined = CombinedFacialDataset([
        make(DrowsinessDataset, cfg.data.drowsiness_root, drow_split.train, train_tf),
        make(FER2013Dataset, cfg.data.fer_root, fer_split.train, train_tf),
    ])
    val_combined = CombinedFacialDataset([
        make(DrowsinessDataset, cfg.data.drowsiness_root, drow_split.val, eval_tf),
        make(FER2013Dataset, cfg.data.fer_root, fer_split.val, eval_tf),
    ])
    test_combined = CombinedFacialDataset([
        make(DrowsinessDataset, cfg.data.drowsiness_root, drow_split.test, eval_tf),
        make(FER2013Dataset, cfg.data.fer_root, fer_split.test, eval_tf),
    ])

    kw = dict(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )
    return {
        "train": DataLoader(train_combined, shuffle=True,  drop_last=True,  **kw),
        "val":   DataLoader(val_combined,   shuffle=False, drop_last=False, **kw),
        "test":  DataLoader(test_combined,  shuffle=False, drop_last=False, **kw),
    }


def compute_class_weights(dataset: Dataset, task: str, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for class-balanced loss."""
    label_key = f"{task}_label"
    counts = np.zeros(num_classes, dtype=np.int64)
    for rec in (dataset.records if hasattr(dataset, "records") else []):
        v = rec[label_key]
        if 0 <= v < num_classes:
            counts[v] += 1
    if counts.sum() == 0:
        return torch.ones(num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)
