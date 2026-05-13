# Facial Stress & Fatigue Detection
## A Multi-Task Vision-Foundation Approach with LLM-Powered Wellness Recommendations

**[Your Name]** — CMPE 258, San Jose State University
**[Partner Name]** — CMPE 258, San Jose State University

---

## Abstract

We present an end-to-end deep learning system for facial stress and fatigue detection that combines a self-supervised vision foundation model (DINOv2) with multi-task classification heads, and adds a prompt-engineered LLM layer that converts model predictions into personalized natural-language wellness recommendations. The full system is deployed as a production-grade MLOps pipeline with experiment tracking (Weights & Biases), model versioning (HuggingFace Hub), automated CI/CD (GitHub Actions), drift monitoring (Evidently AI), and one-click auto-deployment to HuggingFace Spaces. On a combined Kaggle Drowsiness + FER-2013 (emotion → stress remap) dataset, our model achieves **[X.X]%** balanced accuracy on the stress task (3-class) and **[Y.Y]%** on the fatigue task (binary). Ablation studies isolate the contribution of the foundation backbone (+[Z]pp over ResNet-50), multi-task learning (+[A]pp), and our augmentation + class-balancing strategy.

**Keywords:** deep learning, MLOps, foundation models, multi-task learning, facial affect recognition, prompt engineering, DINOv2.

---

## 1. Introduction

[Fill in: motivation, the gap in literature, your contribution.]

Key contributions:
- (i) A unified pipeline combining drowsiness and stress detection in one multi-task model
- (ii) Application of DINOv2 to facial affect — a setting it wasn't pretrained for
- (iii) An LLM explanation layer demonstrating prompt engineering on top of a vision model
- (iv) A complete MLOps deployment with CI/CD, auto-retraining, and drift monitoring

---

## 2. Related Work

**Facial affect recognition.** [Brief survey of FER literature — Ekman emotions, deep CNNs, FER-2013 benchmark.]

**Drowsiness detection.** [PERCLOS, recent CNN/ViT approaches, applications.]

**Foundation models for vision.** [DINO, MAE, DINOv2 — discuss why self-supervised pretraining is appealing for affective tasks where labeled data is scarce.]

**Multi-task learning.** [Caruana 1997, more recent work, especially shared-trunk architectures.]

**MLOps for student projects.** [Brief note on Google's MLOps maturity model and how ours maps to it.]

---

## 3. Approach

### 3.1 Problem Formulation

Given a facial image $x \in \mathbb{R}^{H \times W \times 3}$, predict:
- $y_s \in \{\text{low}, \text{moderate}, \text{high}\}$ — stress level (3-class)
- $y_f \in \{\text{alert}, \text{drowsy}\}$ — fatigue state (binary)

And emit a natural-language recommendation $r$ conditioned on $(y_s, y_f)$ and their model confidences.

### 3.2 Architecture

[Insert architecture figure from `assets/architecture.png`]

The full model is:
$$f(x) = (h_s(g(\phi(x))), h_f(g(\phi(x))))$$
where $\phi$ is a frozen-then-partially-fine-tuned DINOv2-small backbone (21M params), $g$ is a shared MLP trunk (384 → 256), and $h_s, h_f$ are linear classifier heads. Output probabilities feed into a templated few-shot prompt evaluated by Claude (Anthropic) to produce the wellness recommendation.

### 3.3 Design Decisions

[Summarize from `docs/methodology.md` — backbone, freeze schedule, activation, normalization, loss, augmentation, optimizer, schedule.]

---

## 4. Datasets

| Dataset | Source | Used for | # samples | Split |
|---------|--------|----------|-----------|-------|
| Drowsiness Detection | Kaggle (dheerajperumandla) | Fatigue labels | ~2,900 | Subject-grouped 70/15/15 |
| FER-2013 | Kaggle (msambare) | Stress labels via emotion remap | ~35,000 | Pseudo-subject-grouped 70/15/15 |

### 4.1 Emotion → Stress remapping

[Reproduce the FER table from methodology.md.]

### 4.2 Preprocessing

[Face detection w/ MediaPipe, resize to 224, normalization, grayscale → 3-channel for FER.]

---

## 5. Experiments

### 5.1 Setup

- **Hardware:** Kaggle TPU v3-8 (free tier)
- **Training time:** ~45 minutes for the full model
- **Frameworks:** PyTorch 2.1, HuggingFace Transformers 4.40, W&B 0.16
- **Reproducibility:** fixed seed (42), config-driven (`src/config/config.yaml`)

### 5.2 Main results

| Task | Metric | Value |
|------|--------|-------|
| Stress | Balanced accuracy | TBD |
| Stress | Macro F1 | TBD |
| Stress | AUC-ROC | TBD |
| Stress | ECE (calibration error) | TBD |
| Fatigue | Balanced accuracy | TBD |
| Fatigue | Macro F1 | TBD |
| Fatigue | AUC-ROC | TBD |
| Fatigue | ECE | TBD |

[Insert TensorBoard + W&B screenshots.]
[Insert confusion-matrix figures for stress and fatigue.]
[Insert calibration / reliability diagram.]

### 5.3 Ablation studies

| Variant | Backbone | Multi-task | Aug | Loss | Stress balanced acc | Fatigue balanced acc |
|---------|----------|------------|-----|------|---------------------|----------------------|
| A1 ResNet stress-only | ResNet-50 | ✗ | basic | CE | TBD | — |
| A2 DINOv2 stress-only | DINOv2-s | ✗ | basic | CE | TBD | — |
| A3 DINOv2 multi-task | DINOv2-s | ✓ | basic | CE | TBD | TBD |
| A4 + RandAugment | DINOv2-s | ✓ | RandAugment | CE | TBD | TBD |
| A5 + class balancing | DINOv2-s | ✓ | RandAugment | weighted CE | TBD | TBD |
| A6 + focal (full) | DINOv2-s | ✓ | RandAugment | weighted CE + focal | **TBD** | **TBD** |

[Discuss findings: which component contributes most, surprising patterns, etc.]

### 5.4 Hyperparameter sweep

[Insert W&B sweep parallel-coordinates plot screenshot.]
[Best config summary; sensitivity discussion.]

### 5.5 LLM explanation evaluation

[Discuss qualitatively: did the LLM produce sensible recommendations? Edge cases?]

---

## 6. MLOps Pipeline

[Reproduce the maturity-level matrix from README. Show CI badge, the auto-deploy flow, drift-detection plan.]

[Screenshot of the GitHub Actions runs, the HF Space deployment, the TensorBoard or W&B dashboards.]

---

## 7. Limitations and Future Work

- FER-2013 emotion-to-stress mapping is a heuristic, not validated against ground-truth PSS questionnaires.
- Single-frame inference; temporal cues (eye closures over time) would likely improve fatigue detection.
- Subject diversity in training data is unmeasured; potential fairness concerns.
- LLM explanations rely on closed-source API; on-device alternatives (e.g. Phi-3) would improve privacy.

---

## 8. Conclusion

[Recap headline results + contributions + impact.]

---

## References

[Bibliography in any consistent format — IEEE or ACM is fine.]

1. Oquab, M., et al. *DINOv2: Learning Robust Visual Features without Supervision.* 2023.
2. Caron, M., et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021.
3. Goodfellow, I., et al. *Challenges in Representation Learning: A report on three machine learning contests.* (FER-2013) 2013.
4. Lin, T.-Y., et al. *Focal Loss for Dense Object Detection.* ICCV 2017.
5. Caruana, R. *Multitask Learning.* Machine Learning 1997.
6. Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization.* ICLR 2019.
7. Cubuk, E. D., et al. *RandAugment: Practical automated data augmentation.* NeurIPS 2020.
