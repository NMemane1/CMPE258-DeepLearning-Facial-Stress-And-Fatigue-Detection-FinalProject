"""Run the ablation matrix.

Each ablation modifies one component of the full system. Results are
aggregated into a single JSON for the writeup.

Usage:
    python -m src.evaluation.ablation --base-config src/config/config.yaml --out ablation_results.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from omegaconf import OmegaConf

# Each ablation is a list of dotted-key overrides.
ABLATIONS = {
    "A1_resnet_stress_only": [
        "model.backbone=resnet50",
        "model.use_resnet_baseline=true",
        "training.loss_weights.fatigue=0.0",
        "training.loss_weights.focal=0.0",
        "augmentation.enabled=false",
        "experiment.name=ablation_A1_resnet_stress_only",
    ],
    "A2_dinov2_stress_only": [
        "training.loss_weights.fatigue=0.0",
        "training.loss_weights.focal=0.0",
        "augmentation.enabled=false",
        "experiment.name=ablation_A2_dinov2_stress_only",
    ],
    "A3_dinov2_multitask_noaug": [
        "training.loss_weights.focal=0.0",
        "augmentation.enabled=false",
        "experiment.name=ablation_A3_dinov2_multitask_noaug",
    ],
    "A4_dinov2_multitask_aug": [
        "training.loss_weights.focal=0.0",
        "experiment.name=ablation_A4_dinov2_multitask_aug",
    ],
    "A5_balanced_loss": [
        "training.loss_weights.focal=0.0",
        "experiment.name=ablation_A5_balanced_loss",
    ],
    "A6_full_ours": [
        "experiment.name=ablation_A6_full_ours",
    ],
}


def run_one(base_config: str, overrides: list[str]) -> dict:
    # Late import so this script can be loaded without GPU deps.
    from src.training.train import load_config, main as train_main
    import sys
    sys.argv = ["train.py", "--config", base_config, *overrides]
    metrics = train_main()
    return metrics or {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="src/config/config.yaml")
    p.add_argument("--out", default="ablation_results.json")
    p.add_argument("--only", nargs="*", default=None, help="run only these ablation keys")
    args = p.parse_args()

    results = {}
    keys = args.only or list(ABLATIONS.keys())
    for key in keys:
        if key not in ABLATIONS:
            print(f"[warn] unknown ablation: {key}; skipping")
            continue
        print(f"\n{'='*70}\nABLATION: {key}\n{'='*70}")
        try:
            metrics = run_one(args.base_config, ABLATIONS[key])
            results[key] = metrics
        except Exception as e:
            print(f"[error] ablation {key} failed: {e}")
            results[key] = {"error": str(e)}

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
