# Project Proposal — Facial Stress & Fatigue Detection

**Course:** CMPE 258 — Deep Learning · San José State University · Spring 2026
**Authors:** Nikita Memane, Sankalp Wahane
**Repo:** https://github.com/NMemane1/CMPE258-DeepLearning-Facial-Stress-And-Fatigue-Detection-FinalProject

---

## 1. Problem Statement

We propose to build an end-to-end deep learning system that, given a
single RGB face image, predicts the subject's **stress level on a
three-class scale** (low / moderate / high) and **fatigue state on a
binary scale** (alert / fatigued), and then generates a short
supportive natural-language wellness suggestion conditioned on those
predictions. The system will be deployed as a live, public web app
backed by an automated MLOps pipeline so that every result is
reproducible by a third party.

## 2. Why This Problem Matters

- Stress and fatigue have well-documented effects on health, safety,
  and productivity, yet they remain hard to monitor passively in
  everyday life.
- Self-report instruments (PSS, KSS) are noisy and intrusive;
  physiological sensors (HRV, EDA, EEG) are reliable but require
  wearable hardware that most users will not adopt for an ambient
  wellness signal.
- A camera-based estimator that works on a single image — no
  wearables, no calibration — is the missing third modality. It
  could power study-break reminders, driver-monitoring assists,
  workplace dashboards, or be embedded as a soft signal in any app
  that already has webcam access.
- Recent self-supervised vision foundation models (DINOv2) have
  dramatically improved small-data transfer learning, making this
  achievable on a course budget without millions of labelled images.

## 3. Approach Overview

We will fine-tune `facebook/dinov2-small` as a shared backbone with
two task-specific classification heads (stress: 3 classes, fatigue: 2
classes) trained jointly via masked multi-task learning. Lower
transformer layers will be frozen (generic feature reuse); upper
layers will be fine-tuned for task specialization. A prompt-engineered
LLM layer (Claude with a deterministic fallback) will turn the raw
probabilities into a short wellness sentence — satisfying the
"prompting and prompt engineering" rubric item and adding a usable
user-facing output.

**Why this approach over alternatives we considered:**

- **Pure CNN baseline (ResNet-50)** — Rejected as the final model
  because published evaluations show DINOv2 transfers better than
  ImageNet-supervised CNNs on small downstream classification tasks.
  Will be included as the conceptual baseline in the ablation
  discussion.
- **CLIP zero-shot classification** — Rejected: zero-shot affect
  recognition is too noisy without fine-tuning, and the language
  alignment of CLIP is overhead we don't use.
- **Two independent single-task models** — Rejected: the fatigue
  dataset is small (~400 samples). A multi-task setup lets the
  larger stress task regularize the fatigue task via the shared
  representation.

## 4. Data Plan

| Source | Type | Use | Estimated size |
|---|---|---|---|
| FER-2013 (`msambare/fer2013`) | 48×48 grayscale face images, 7 emotion labels | Stress (via emotion → stress proxy mapping) | ~35,000 images |
| Yawn-Eye Dataset (`serenaraju/yawn-eye-dataset-new`) | RGB face crops, 4 raw classes | Fatigue (collapsed to alert vs fatigued) | ~400–600 images |

**Stress proxy mapping** (will be documented honestly as a proxy, not
ground truth):

- `happy`, `surprise`, `neutral` → **low**
- `sad` → **moderate**
- `angry`, `fear`, `disgust` → **high**

**Preprocessing:**

- Optional face detection + crop with MediaPipe (for in-the-wild
  inputs at inference time)
- Resize to 224×224 with 1.15× pre-resize + center crop
- Normalize with DINOv2 statistics
- Grayscale → 3-channel by tiling (FER-2013)

**Splits:** 70/15/15 train/val/test, subject-grouped where subject
metadata is available (yawn-eye), random stratified within class
where it is not (FER-2013, which has no subject metadata in the
public release). The lack of subject-grouped FER-2013 splits is a
limitation we will document.

**Augmentation:** RandAugment(N=2, M=9) + horizontal flip(p=0.5) +
mild color jitter on training images only.

## 5. Methods Plan

### 5.1 Architecture

- Backbone: `dinov2-small`, 12-layer ViT, 384-d output
- Freeze strategy: patch embeddings + layers 0–8 frozen; layers 9–11
  + final LayerNorm fine-tuned
- Shared MLP trunk: 384 → 256 → 256 with LayerNorm, GELU, Dropout(0.3)
- Stress head: Linear(256 → 3) with BatchNorm + Dropout
- Fatigue head: Linear(256 → 2) with BatchNorm + Dropout
- Estimated parameter count: ~22M total, ~5.5M trainable (~25%)

### 5.2 Training

- Loss: masked multi-task CE with inverse-frequency class weights +
  focal loss term (γ=2.0) on the stress task
- Optimizer: AdamW with discriminative learning rates (backbone
  3e-5, heads 3e-4), weight decay 0.01
- Schedule: linear warmup 500 steps → cosine decay to 1e-6
- Batch size: 32 (Kaggle T4 fp16) or 64 (TPU bfloat16) depending on
  available compute
- Epochs: 15 max with early stopping (patience 3 on val mean
  balanced accuracy)
- Mixed precision: fp16 (GPU) / bfloat16 (TPU)
- Seed: 42, fully reproducible
- Tracking: TensorBoard event logs (W&B optional)

### 5.3 Evaluation

- **Per-task metrics:** balanced accuracy, macro-F1, ROC-AUC,
  Expected Calibration Error (ECE) at 15 bins
- **Confusion matrices** for both tasks
- **Reliability diagrams** for calibration analysis
- **Failure mode analysis** — which classes get confused with which,
  and is the failure benign (adjacent classes) or operationally bad
  (low vs high)?
- **Qualitative live-demo check** with a real face on the deployed app

## 6. Experiments Plan

We will *not* run a full ablation matrix (each variant would cost
~25 min and the GPU budget is limited). Instead we will deliver a
**design-choice analysis** in `docs/ablations.md`:

- Each component (multi-task vs independent, DINOv2 vs ResNet, freeze
  depth, focal loss, augmentation, subject-grouped splits) gets a
  separate section
- Each section is labelled `OBSERVED` (computed from the trained
  checkpoint + saved predictions) or `ARGUED` (first-principles +
  cited literature)
- This is more useful to a reader than a table of weakly-different
  ablation numbers would be

## 7. End Deliverables

By the deadline this repository will contain:

1. **Source code** — `src/` (data, models, training, evaluation,
   inference), `app/` (Gradio demo)
2. **Trained model artifact** — versioned on the Hugging Face Hub at
   `NMemane1/facial-stress-fatigue-dinov2`
3. **Deployed Gradio web app** — live on Hugging Face Spaces at
   `NMemane1/facial-stress-fatigue`, supporting upload + webcam
4. **Training notebook** — `notebooks/02_training_kaggle.ipynb`,
   end-to-end on Kaggle GPU
5. **Evaluation artifacts** — `outputs/` with TensorBoard logs,
   `test_metrics.json`, four PNG plots (two confusion matrices, two
   calibration diagrams)
6. **Design-choice analysis** — `docs/ablations.md`
7. **MLOps pipeline** — `.github/workflows/ci.yml`,
   `.github/workflows/deploy_hf.yml` (auto-deploy on push to main),
   `.github/workflows/retrain.yml` (scaffold)
8. **DeepWiki / repo walkthrough** — `docs/DEEPWIKI.md`
9. **Final report** — `docs/report.md`, 6–8 pages following the
   rubric structure
10. **Slide deck** — `docs/slides.md`, 14 slides with speaker notes
11. **This proposal** — `docs/proposal.md`
12. **Updated README** — at repo root with links to all deliverables
13. **Video recordings** — long presentation + short demo (links
    added to README after recording)

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Kaggle GPU quota runs out mid-training | Notebook saves checkpoints every epoch; can resume |
| FER-2013 → stress mapping is too noisy to learn from | Acknowledge transparently; report stress metrics as "estimated stress from facial affect," not "stress" |
| Fatigue dataset is too small for meaningful test metrics | Acknowledge n=33 explicitly in every doc; treat fatigue numbers as directional, not headline |
| Cold-start latency on HF Spaces basic tier | Pre-load model in startup; document expected latency |
| LLM API rate limits or auth failures | Deterministic canned-response fallback so the demo never breaks |
| HF Space build/runtime errors | CI runs basic checks on every push; auto-deploy via GitHub Actions provides fast iteration |
| Class imbalance on FER-2013 | Class-weighted loss + focal loss; balanced-accuracy as primary metric |

## 9. Timeline

| Day | Tasks |
|---|---|
| 1 | Repo scaffolding, proposal, data pipeline, model code, app skeleton |
| 2 | Train baseline DINOv2 multi-task model on Kaggle GPU, capture TensorBoard |
| 3 | Evaluation: confusion matrices, calibration, design-choice analysis writeup |
| 4 | Deploy HF Space, debug deployment, capture screenshots, draft report |
| 5 | Polish report and slides, record video, finalize README, submit |

## 10. Success Criteria

A submission counts as successful if:

- Model trains end-to-end on Kaggle GPU and achieves
  **balanced accuracy ≥ 0.65 on stress (n=5k+)** — this would be a
  defensible result for a 5.5M-trainable-parameter multi-task model
  on a proxy-labelled dataset
- Live deployed app accepts an image and returns predictions + a
  wellness suggestion with **< 1 second** total latency on CPU basic
- CI/CD passes on `main` and auto-deploys to the HF Space on every
  push
- Report, slides, proposal, ablations, and DeepWiki are all in
  `docs/` and linked from the README
- All metrics in the report are **real numbers from
  `outputs/test_metrics.json`** — no invented numbers anywhere
- **Honest caveats** about the fatigue n=33 test split and the
  stress proxy labels appear in the report, the slides, and the
  README
