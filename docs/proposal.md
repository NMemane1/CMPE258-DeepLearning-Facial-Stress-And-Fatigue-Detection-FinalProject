# Project Proposal — Facial Stress & Fatigue Detection

**Author:** [Your Name]
**Course:** CMPE 258 — Deep Learning
**Date:** [Submission Date]
**Repo:** https://github.com/NMemane1/CMPE258-DeepLearning-Facial-Stress-And-Fatigue-Detection-FinalProject

---

## 1. Problem Statement

We aim to build an end-to-end deep learning system that, given a single facial image, predicts (a) the subject's stress level on a 3-class scale (low / moderate / high) and (b) their fatigue state on a 2-class scale (alert / drowsy), and then generates a personalized natural-language wellness recommendation. The system must be deployable as a real-time web application and built according to MLOps best practices, with full reproducibility, automated retraining, and monitoring.

## 2. Why This Problem Matters

- Workplace burnout and chronic fatigue cost the U.S. economy ~$300B annually (CDC).
- Existing detection methods (PSS questionnaires, KSS scales) are self-report and prone to bias.
- Vision-based detection offers passive, real-time signal without sensor burden.
- The intersection with foundation models lets us produce more accurate, more explainable outputs than classical CNN affect-recognition.

## 3. Approach Overview

We will fine-tune **DINOv2** — a self-supervised vision foundation model from Meta — as a shared backbone, with two task-specific classification heads (stress, fatigue) trained jointly via multi-task learning. We will then add a prompt-engineered LLM (Claude or GPT) inference layer that translates predictions into natural-language wellness suggestions, satisfying the rubric's "prompting and prompt engineering" requirement.

The rationale for this approach:
1. **Foundation-model backbones outperform from-scratch CNNs** on small-data facial-affect tasks (Caron et al., 2021; Oquab et al., 2023).
2. **Multi-task learning provides inductive bias** — stress and fatigue share facial cues (eye openness, brow tension), so a shared trunk regularizes both tasks.
3. **LLM explainability layer** turns raw probabilities into actionable language users can use, and demonstrates prompt-engineering competence.

We considered three alternatives before settling on this approach:
- **A pure CNN baseline (ResNet-50)** — included as an ablation, not the final model.
- **CLIP zero-shot classification** — rejected: zero-shot accuracy on affective states is poor without fine-tuning.
- **Multi-modal (vision + landmark coordinates)** — rejected for scope: adds a second pipeline without strong evidence of large gains.

## 4. Data

| Source | Type | Use | Size |
|--------|------|-----|------|
| Kaggle Drowsiness Dataset | Image (face crops) | Fatigue labels (alert/drowsy) | ~2,900 images |
| FER-2013 | Image (48×48 grayscale) | Stress labels via emotion remapping | ~35,000 images |

**Stress label remapping** (from FER-2013 emotion classes):
- `angry`, `fear`, `disgust` → high stress
- `sad` → moderate stress
- `happy`, `surprise`, `neutral` → low stress

**Preprocessing:**
- Face detection + alignment with MediaPipe (drops images where no face detected)
- Resize to 224×224
- Normalize with DINOv2 statistics
- Convert grayscale FER-2013 to 3-channel by replication

**Splits:** stratified 70/15/15 train/val/test. For Kaggle Drowsiness we split by subject ID to prevent leakage; FER-2013 splits are random within class.

**Augmentation:** RandAugment (n=2, m=9), horizontal flip (probability 0.5), color jitter for training only.

## 5. Methods

### 5.1 Model architecture
- Backbone: `dinov2-small` (21M params, 384-d embedding)
- Freeze strategy: layers 0–8 frozen, layers 9–11 fine-tuned
- Shared MLP: 384 → 256, GELU, Dropout(0.3)
- Stress head: 256 → 3
- Fatigue head: 256 → 2

### 5.2 Training
- Loss: `α * CE_stress + β * CE_fatigue + γ * focal_loss` with class weights from inverse frequency
- Optimizer: AdamW, lr=3e-4 for heads, lr=3e-5 for unfrozen backbone layers (discriminative learning rates)
- Schedule: linear warmup (500 steps) + cosine decay
- Batch size: 64 on TPU v3-8, 16 on Colab GPU
- Epochs: 15 with early stopping (patience 3 on val balanced accuracy)
- Mixed precision: bfloat16 on TPU, fp16 on GPU

### 5.3 Evaluation
- **Per-task metrics:** balanced accuracy, macro F1, AUC-ROC, per-class precision/recall, ECE for calibration
- **Confusion matrices** for each task
- **Calibration plots** (reliability diagrams)
- **Embedding visualization** (t-SNE / UMAP of penultimate layer)
- **Grad-CAM** attention visualizations for qualitative analysis

## 6. Experiments Plan

**Ablation matrix** (six variants A1–A6, as detailed in README) isolates contribution of:
- Backbone choice (ResNet-50 vs DINOv2)
- Single-task vs multi-task
- Augmentation strategy
- Loss function design

**Hyperparameter sweep** via W&B over LR, weight decay, dropout, unfreeze depth, augmentation strength. ~20 trials with Bayesian optimization.

## 7. End Deliverables (per rubric)

By the deadline, this repository will contain:

1. **Full source code** — data pipeline, model, training, inference, web app
2. **Trained model artifact** — hosted on HuggingFace Hub, versioned
3. **Deployed Gradio web app** — live on HuggingFace Spaces, with webcam + upload support
4. **Training notebook** — end-to-end Kaggle TPU notebook for reproducibility
5. **Evaluation artifacts** — TensorBoard logs, W&B dashboard, all plots
6. **Ablation study** — six variants run + writeup in `docs/ablation_results.md`
7. **Hyperparameter sweep** — config + W&B sweep dashboard
8. **MLOps pipeline** — GitHub Actions CI/CD, Docker, auto-deploy, drift monitoring
9. **DeepWiki/Repomix documentation** — auto-generated codebase docs
10. **Final report** — 6–8 page paper covering all rubric sections
11. **Slide deck** — for live presentation
12. **Video recordings** — long presentation + demo of website
13. **Screenshots folder** — every artifact captured

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| TPU training fails on Kaggle | Fallback to Colab GPU notebook (also provided) |
| FER-2013 → stress mapping is noisy | Acknowledged in report; include ablation on label scheme |
| Cold-start latency on HF Spaces | Pre-load model in startup, profile and document |
| LLM API rate limits | Cache common predictions; provide offline fallback canned responses |
| Class imbalance | Class-weighted loss + balanced sampling |

## 9. Timeline

| Day | Tasks |
|-----|-------|
| 1 | Repo scaffold, proposal, data pipeline, model code, app skeleton, MLOps configs |
| 2 | Train baseline + DINOv2 model, run ablation A1–A3, log to W&B |
| 3 | Ablations A4–A6, hyperparameter sweep, write up methodology |
| 4 | Deploy HF Space, capture screenshots, draft report |
| 5 | Polish report, record video, finalize README |

## 10. Success Criteria

A submission counts as successful if:
- ✅ Model trains end-to-end on Kaggle TPU
- ✅ Live deployed app accepts an image and returns predictions + LLM explanation
- ✅ Balanced accuracy > 60% on stress, > 80% on fatigue (rough benchmarks; targets refined post-baseline)
- ✅ All six ablations completed
- ✅ Sweep produces at least 15 logged runs
- ✅ CI/CD passes on main; auto-deploy works
- ✅ Report and video link are in README
