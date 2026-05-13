"""Metric computation and plotting utilities."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, brier_score_loss,
)


VALID_LABEL_MIN = 0


def _filter(probs: np.ndarray, target: np.ndarray):
    mask = target >= VALID_LABEL_MIN
    return probs[mask], target[mask]


def _ece(probs: np.ndarray, target: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    if len(target) == 0:
        return 0.0
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    accuracies = (preds == target).astype(np.float32)

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += abs(bin_conf - bin_acc) * (in_bin.sum() / len(target))
    return float(ece)


def metrics_for_task(probs: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """Compute metrics for one task."""
    probs, target = _filter(probs, target)
    if len(target) == 0:
        return {
            "balanced_acc": 0.0, "macro_f1": 0.0, "auc": 0.0, "ece": 0.0, "n_samples": 0,
        }
    preds = probs.argmax(axis=1)

    out = {
        "balanced_acc": float(balanced_accuracy_score(target, preds)),
        "macro_f1":     float(f1_score(target, preds, average="macro", zero_division=0)),
        "ece":          _ece(probs, target),
        "n_samples":    int(len(target)),
    }

    try:
        if num_classes == 2:
            out["auc"] = float(roc_auc_score(target, probs[:, 1]))
        else:
            out["auc"] = float(roc_auc_score(target, probs, multi_class="ovr", average="macro"))
    except ValueError:
        out["auc"] = 0.0
    return out


def compute_metrics_dict(
    stress_probs: np.ndarray,
    stress_target: np.ndarray,
    fatigue_probs: np.ndarray,
    fatigue_target: np.ndarray,
) -> dict:
    s = metrics_for_task(stress_probs, stress_target, num_classes=stress_probs.shape[1])
    f = metrics_for_task(fatigue_probs, fatigue_target, num_classes=fatigue_probs.shape[1])
    return {
        # per-task
        "stress/balanced_acc": s["balanced_acc"],
        "stress/macro_f1":     s["macro_f1"],
        "stress/auc":          s["auc"],
        "stress/ece":          s["ece"],
        "stress/n":            s["n_samples"],
        "fatigue/balanced_acc": f["balanced_acc"],
        "fatigue/macro_f1":     f["macro_f1"],
        "fatigue/auc":          f["auc"],
        "fatigue/ece":          f["ece"],
        "fatigue/n":            f["n_samples"],
        # aggregates
        "balanced_acc_mean": (s["balanced_acc"] + f["balanced_acc"]) / 2,
        "macro_f1_mean":     (s["macro_f1"] + f["macro_f1"]) / 2,
    }


def plot_confusion_matrix(target, preds, class_names, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(target, preds, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(probs: np.ndarray, target: np.ndarray, out_path: str, n_bins: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    accuracies = (preds == target).astype(np.float32)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, bin_acc, bin_conf, weights = [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        bin_acc.append(accuracies[in_bin].mean())
        bin_conf.append(confidences[in_bin].mean())
        weights.append(in_bin.sum() / len(target))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
    if centers:
        ax.bar(centers, bin_acc, width=1/n_bins, alpha=0.6, edgecolor="black", label="Accuracy")
        ax.plot(centers, bin_conf, "o-", color="red", label="Confidence")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
