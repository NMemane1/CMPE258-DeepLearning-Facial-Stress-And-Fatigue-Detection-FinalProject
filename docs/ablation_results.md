# Ablation Study Results

Six variants run to isolate the contribution of each design choice. All variants share identical training settings except where noted. Each was run for 15 epochs (4 in quick-mode) with the same data splits, seed 42.

## Headline table

| ID | Variant | Backbone | Multi-task | Augmentation | Class balance | Focal | Stress bal. acc | Fatigue bal. acc | Notes |
|----|---------|----------|------------|--------------|---------------|-------|------------------|-------------------|-------|
| A1 | ResNet baseline (stress only) | ResNet-50 | ❌ | basic resize | ❌ | ❌ | TBD | n/a | ImageNet-supervised. Reference point for backbone choice. |
| A2 | DINOv2 swap (stress only) | DINOv2-s | ❌ | basic resize | ❌ | ❌ | TBD | n/a | Isolates the *backbone* contribution. |
| A3 | + Multi-task | DINOv2-s | ✅ | basic resize | ❌ | ❌ | TBD | TBD | Adds fatigue head + joint training. |
| A4 | + RandAugment | DINOv2-s | ✅ | RandAugment(n=2,m=9) | ❌ | ❌ | TBD | TBD | Adds strong augmentation. |
| A5 | + Class balancing | DINOv2-s | ✅ | RandAugment | ✅ | ❌ | TBD | TBD | Inverse-frequency CE weights. |
| **A6** | **Full (ours)** | **DINOv2-s** | **✅** | **RandAugment** | **✅** | **✅ (γ=2)** | **TBD** | **TBD** | **All components. Final reported model.** |

## Reading the gains

| Comparison | What it isolates | Expected delta | Observed delta |
|-----------|------------------|----------------|----------------|
| A2 − A1 | Foundation backbone vs supervised CNN | +3–6pp | TBD |
| A3 − A2 | Multi-task auxiliary signal | +1–2pp on stress | TBD |
| A4 − A3 | Augmentation | +1–3pp | TBD |
| A5 − A4 | Class-balancing | small avg but ↑ minority recall | TBD |
| A6 − A5 | Focal loss | hardest class improves | TBD |

## Confusion matrices (per variant)

[Insert images: `assets/screenshots/cm_<variant>_stress.png`, `assets/screenshots/cm_<variant>_fatigue.png`]

## Calibration

ECE values per variant — does adding components improve calibration alongside accuracy?

| Variant | ECE (stress) | ECE (fatigue) |
|---------|-------------|----------------|
| A1 | TBD | n/a |
| A2 | TBD | n/a |
| A3 | TBD | TBD |
| A4 | TBD | TBD |
| A5 | TBD | TBD |
| **A6** | **TBD** | **TBD** |

## Discussion

[Fill in 2–3 paragraphs after running:
- Which component contributed the most? Was it the backbone, multi-task, or augmentation?
- Were any results counter to expectation?
- Was there a tradeoff between accuracy and calibration?
- What does this tell us about which design choices were essential vs nice-to-have?]

## Reproducing

```bash
# Full ablation matrix (slow, ~3-4 hours on Kaggle GPU)
bash scripts/run_ablations.sh

# Or single variant
python -m src.evaluation.ablation --only A6_full_ours
```

All runs auto-log to W&B under project `facial-stress-fatigue` for parallel-coordinates / scatter visualization.
