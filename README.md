# Facial Stress & Fatigue Detection — Multi-Task Deep Learning with DINOv2

**CMPE 258 Deep Learning — Final Project — San José State University**

A multi-task deep learning system that estimates **stress** (3 classes) and
**fatigue** (2 classes) from a single face image using a self-supervised
DINOv2 vision-transformer backbone, then uses a language-model layer to turn
the raw predictions into a short, supportive wellness recommendation. The
project is built as an end-to-end ML pipeline: data preparation → GPU training
with experiment tracking → model hosting → automatically-deployed web demo.

---

## Team

| Member | Contributions |
|---|---|
| Nikita Memane | Model architecture, multi-task training pipeline, data preparation, evaluation & calibration analysis, MLOps / CI-CD setup, Gradio app, deployment, documentation |
| Sankalp Wahane | Presentation, methodology review, report review |

> Replace with your team's final attribution before submission if it changes.

---

## Links

| Deliverable | Link |
|---|---|
| GitHub repository | https://github.com/NMemane1/CMPE258-DeepLearning-Facial-Stress-And-Fatigue-Detection-FinalProject |
| Live demo (HF Space) | https://huggingface.co/spaces/NMemane1/facial-stress-fatigue |
| Trained model (HF Hub) | https://huggingface.co/NMemane1/facial-stress-fatigue-dinov2 |
| Training notebook (Kaggle) | <!-- TODO: paste public Kaggle notebook URL here --> |
| Long presentation video | <!-- TODO: paste YouTube long-form video URL here --> |
| Demo video | <!-- TODO: paste YouTube demo video URL here --> |
| Slide deck | <!-- TODO: paste SlideShare / PDF URL here --> |

---

## Documentation

All written deliverables live under [`docs/`](docs/):

| Document | What it is |
|---|---|
| [`docs/report.md`](docs/report.md) | Full 6–8 page formal report (abstract → related work → data → methods → experiments → limitations → conclusion → references) |
| [`docs/ablations.md`](docs/ablations.md) | Design-choice analysis — every component justified, each labelled `OBSERVED` or `ARGUED` |
| [`docs/slides.md`](docs/slides.md) | 14-slide presentation deck with speaker notes (render with Marp / reveal.js, or paste into Slides/Keynote) |
| [`docs/proposal.md`](docs/proposal.md) | Project proposal (problem, approach, datasets, methods, deliverables, risks, timeline, success criteria) |
| [`docs/methodology.md`](docs/methodology.md) | "Why we chose what we chose" — long-form rationale for every hyperparameter (loss, activation, norm, augmentation, optimizer, schedule, LLM prompt design, evaluation choices) |
| [`docs/architecture.md`](docs/architecture.md) | System architecture diagrams (model, training data flow, CI/CD) and parameter budget |
| [`docs/DEEPWIKI.md`](docs/DEEPWIKI.md) | File-by-file walkthrough of the entire codebase — `src/`, `app/`, `notebooks/`, `.github/workflows/` |

---

## Problem & Motivation

Stress and fatigue affect health, safety, and productivity, yet they are hard
to monitor passively. Drivers, students, and shift workers rarely notice their
own state until it is severe. A camera-based estimator that runs on a single
image — no wearables, no calibration — could power ambient wellness tools,
driver-monitoring systems, or study-break reminders.

This project asks: **can one self-supervised vision backbone, fine-tuned with
two lightweight heads, jointly estimate stress and fatigue well enough to be
useful?** We also add a small language-model layer so the system communicates
its output as a helpful suggestion rather than a bare number.

---

## Inputs, Outputs, and Key Metrics

- **Input:** a single RGB face image (any resolution; resized to 224×224).
- **Outputs:**
  - Stress — one of `low`, `moderate`, `high` (+ class probabilities)
  - Fatigue — one of `alert`, `fatigued` (+ class probabilities)
  - A short natural-language wellness suggestion derived from the two predictions.
- **Key metrics:** balanced accuracy (handles class imbalance), macro-F1,
  ROC-AUC, and Expected Calibration Error (ECE) — because for a wellness tool,
  *calibrated confidence* matters as much as raw accuracy.

---

## Data

Two public datasets are combined into one multi-task training set. Each sample
carries a label for only one task; a masked loss lets them train jointly.

| Dataset | Used for | Size | Notes |
|---|---|---|---|
| FER-2013 (`msambare/fer2013`) | Stress (3 classes) | 35,887 images | Emotion labels mapped to a stress proxy: neutral/happy → low, surprise/sad → moderate, angry/fear/disgust → high |
| Yawn-Eye Dataset (`serenaraju/yawn-eye-dataset-new`) | Fatigue (2 classes) | 433 images used | Eye-closed / yawn cues mapped to `alert` vs `fatigued` |

**Splits.** 70 / 15 / 15 train / val / test, **subject-grouped** so the same
person cannot appear in both train and test (prevents identity leakage).
Final split sizes: **train = 25,300, val = 5,653, test = 5,367**.

**Preprocessing & augmentation.** Images resized to 224×224 and normalized
with the DINOv2 processor statistics. Training augmentation: RandAugment
(N=2, M=9), horizontal flip (p=0.5), and mild color jitter. Class weights are
computed from the training distribution to counter imbalance.

> **Honest limitation.** The fatigue test split is small (n = 33). The fatigue
> metrics below are therefore encouraging but not statistically robust; the
> stress metrics (n = 5,334) are far more reliable. This is discussed further
> in the report.

---

## Model Architecture

```
          Input image (224×224×3)
                   │
        ┌──────────▼───────────┐
        │  DINOv2-small (ViT)  │   facebook/dinov2-small
        │  layers 0-8 frozen   │   self-supervised foundation model
        │  layers 9+ fine-tuned│
        └──────────┬───────────┘
                   │ CLS token (384-d)
        ┌──────────▼───────────┐
        │   Shared trunk       │   Linear(384→256) → GELU → Dropout(0.3)
        └─────┬──────────┬─────┘
              │          │
      ┌───────▼──┐   ┌───▼────────┐
      │ Stress   │   │ Fatigue    │
      │ head     │   │ head       │
      │ Linear→3 │   │ Linear→2   │
      └──────────┘   └────────────┘
```

- **Backbone:** `facebook/dinov2-small`, a self-supervised ViT. Lower 9
  transformer layers frozen, upper layers fine-tuned.
- **Shared trunk:** a single `Linear → GELU → Dropout` block so both tasks
  share representation capacity.
- **Two heads:** independent linear classifiers for stress and fatigue.
- **Parameter count:** 22.2M total, **5.49M trainable (24.7%)** — most of the
  backbone stays frozen, which makes training fast and regularizes the model.

### Why these design choices

| Choice | Reason |
|---|---|
| DINOv2 backbone | Self-supervised ViT features transfer well to faces without a massive labelled dataset. |
| Freeze lower layers | Generic low-level features need no retraining; freezing them cuts compute and overfitting. |
| Multi-task (shared trunk, 2 heads) | One backbone serving both tasks is more data-efficient and forces a more general representation. |
| Masked multi-task loss | Each sample only updates the head it has a label for, so two single-task datasets combine cleanly. |
| Class-balanced **focal loss** | Stress classes are imbalanced; focal loss down-weights easy examples and class weights correct frequency bias. |
| GELU activation | Smooth activation that pairs well with transformer-style trunks. |
| Dropout 0.3 | Regularization on the small trainable head given a modest dataset. |
| Cosine LR schedule + warmup | Stable fine-tuning of a pretrained ViT; warmup avoids early divergence. |
| AdamW, separate LRs | Backbone LR 3e-5 (gentle), head LR 3e-4 (faster) — standard for fine-tuning. |
| Mixed precision (fp16) | ~2× faster training on the T4 GPU with no accuracy loss. |

---

## Training

| Setting | Value |
|---|---|
| Hardware | Kaggle Tesla T4 GPU (×2 available; single-GPU run) |
| Precision | Mixed precision (fp16) |
| Batch size | 32 |
| Optimizer | AdamW (β = 0.9, 0.999), weight decay 0.01 |
| LR | backbone 3e-5, head 3e-4, cosine schedule, 500 warmup steps |
| Epochs | 15 max, **early stopping after 11** (patience 3 on mean balanced acc) |
| Loss | Masked multi-task CE + focal (γ = 2.0), class-balanced |
| Tracking | TensorBoard (event file in `outputs/`) |
| Total runtime | ~25 minutes |

### Training curve (validation balanced accuracy)

| Epoch | Train loss | Val balanced acc | Best |
|---|---|---|---|
| 0 | 1.1742 | 0.5824 | ★ |
| 1 | 0.9643 | 0.5974 | ★ |
| 2 | 0.8976 | 0.6145 | ★ |
| 3 | 0.8431 | 0.6233 | ★ |
| 4 | 0.7943 | 0.6288 | ★ |
| 5 | 0.7517 | 0.6104 | |
| 6 | 0.7081 | 0.6168 | |
| 7 | 0.6746 | **0.6311** | ★ (best) |
| 8 | 0.6153 | 0.6176 | |
| 9 | 0.5654 | 0.6107 | |
| 10 | 0.5373 | 0.5923 | |

Training loss falls steadily while validation balanced accuracy peaks at
epoch 7 (0.6311) and then plateaus — early stopping correctly halts training
at epoch 10 and the epoch-7 checkpoint is kept as the best model.

---

## Results

Final metrics on the held-out **test** set, using the best (epoch-7) checkpoint:

| Metric | Stress (3-class, n=5,334) | Fatigue (2-class, n=33) |
|---|---|---|
| Balanced accuracy | **0.7145** | 0.9615 |
| Macro F1 | **0.7003** | 0.9678 |
| ROC-AUC | **0.8921** | 0.9885 |
| Expected Calibration Error (ECE) | **0.0537** | 0.0303 |

**Combined:** mean balanced accuracy **0.8380**, mean macro-F1 **0.8340**.

### Reading the results

- **Stress (the reliable number):** 0.71 balanced accuracy and 0.89 AUC on a
  3-class problem with 5,334 test samples is a solid result for a model with
  only 5.5M trainable parameters fine-tuned for 25 minutes. The confusion
  matrix shows a clear diagonal — most errors are between *adjacent* classes
  (e.g. `moderate` vs `high`), which is the expected, benign failure mode.
- **Calibration:** an ECE of 0.054 means the model's confidence is well-aligned
  with its accuracy — important for a wellness tool that should not be
  over-confident. The reliability diagram tracks the diagonal closely.
- **Fatigue:** the numbers are very high, but **n = 33** — this split is too
  small to draw strong conclusions. We report it transparently rather than
  over-claiming.

### Evaluation artifacts

All generated by the training run and stored under `outputs/plots/`:

| Artifact | File |
|---|---|
| Stress confusion matrix | `outputs/plots/cm_stress.png` |
| Fatigue confusion matrix | `outputs/plots/cm_fatigue.png` |
| Stress calibration / reliability diagram | `outputs/plots/calibration_stress.png` |
| Fatigue calibration / reliability diagram | `outputs/plots/calibration_fatigue.png` |
| TensorBoard training curves | `outputs/dinov2_multitask_full/tb/` |
| Raw test metrics (JSON) | `outputs/dinov2_multitask_full/test_metrics.json` |

To view the TensorBoard curves locally:
```bash
pip install tensorboard
tensorboard --logdir outputs/dinov2_multitask_full/tb/
```

---

## MLOps Pipeline

This project is built as a reproducible, automated pipeline rather than a
single notebook.

```
  Data (Kaggle datasets)
        │
        ▼
  Training notebook on Kaggle GPU  ──▶  TensorBoard tracking
        │
        ▼
  Best checkpoint  ──▶  pushed to HF Hub model repo
        │
        ▼
  GitHub repo (source of truth)
        │  push to main
        ▼
  GitHub Actions workflow (.github/workflows/deploy_hf.yml)
        │  automatically syncs app/ + src/
        ▼
  HF Space rebuilds  ──▶  live Gradio demo
```

- **Automated training** — a single Kaggle notebook prepares data, trains,
  evaluates, generates plots, and uploads the model. Fully reproducible
  (fixed seed 42).
- **Experiment tracking** — TensorBoard event logs captured every run.
- **Model management** — best checkpoint versioned on the HF Hub.
- **Automated deployment (CI/CD)** — every push to `main` triggers a GitHub
  Actions workflow that redeploys the app to the HF Space. No manual deploy
  step.
- **Reproducibility** — pinned config (`src/config/config.yaml`), seeded
  splits, and a documented environment.

This corresponds to roughly **MLOps maturity level 2–3**: automated training
with centralized tracking, plus automated deployment from version control.

---

## Repository Structure

```
.
├── README.md                  ← this file
├── app/
│   ├── app.py                 ← Gradio demo (3 tabs: Predict / Methodology / About)
│   └── requirements.txt       ← Space dependencies
├── src/
│   ├── config/                ← config.yaml, sweep.yaml
│   ├── data/                  ← dataset classes, transforms, subject-grouped splits
│   ├── models/                ← DINOv2 backbone, heads, multi-task model
│   ├── training/              ← trainer, losses (masked CE + focal), callbacks
│   ├── evaluation/            ← metrics (balanced acc, ECE), calibration plots, ablation
│   └── inference/             ← face detector, LLM explainer, predict
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_kaggle.ipynb       ← main training entry point
│   └── 03_ablation_studies.ipynb
├── outputs/                   ← trained checkpoint, plots, TensorBoard logs, metrics
├── docs/                      ← proposal, methodology, architecture, report
├── tests/                     ← unit tests for data / models / inference
└── .github/workflows/         ← CI + automated HF Space deployment
```

---

## Reproducing This Project

### Run the demo locally
```bash
git clone https://github.com/NMemane1/CMPE258-DeepLearning-Facial-Stress-And-Fatigue-Detection-FinalProject.git
cd CMPE258-DeepLearning-Facial-Stress-And-Fatigue-Detection-FinalProject
pip install -r app/requirements.txt
python app/app.py
```
The app downloads the trained weights from the HF Hub automatically.

### Retrain the model
1. Open `notebooks/02_training_kaggle.ipynb` on Kaggle.
2. Attach the two datasets (`msambare/fer2013`,
   `serenaraju/yawn-eye-dataset-new`) and set the accelerator to **GPU T4**.
3. Add a `WANDB_API_KEY` / `HF_TOKEN` secret if you want logging / model
   upload, then **Save & Run All**.
4. The best checkpoint, plots, and metrics are written to `/kaggle/working/outputs/`.

### Optional: LLM wellness layer
Set an `ANTHROPIC_API_KEY` environment variable (or HF Space secret) to enable
Claude-generated wellness notes. Without it, the app uses a deterministic
fallback so it still works end-to-end.

---

## Limitations & Future Work

- **Fatigue test set is small (n=33)** — the fatigue metrics need a larger,
  purpose-built evaluation set before they can be trusted.
- **Stress labels are a proxy** — derived from FER-2013 emotion labels, not
  ground-truth stress measurements. A dataset with physiological stress labels
  would make the task more faithful.
- **Single-image, single-frame** — no temporal information. A short video clip
  would give much stronger fatigue cues (blink rate, micro-expressions).
- **Demographic coverage** — FER-2013 has known demographic skew; a fairness
  audit across age / skin tone / gender is important future work.
- **Future:** temporal modeling, a larger DINOv2 backbone, proper physiological
  labels, and on-device (mobile) inference.

---

## Acknowledgements

- `facebook/dinov2-small` — Meta AI self-supervised vision foundation model.
- FER-2013 dataset (Manas Sambare mirror) and the Yawn-Eye drowsiness dataset.
- Built for CMPE 258 Deep Learning, San José State University, Spring 2026.

> **Disclaimer.** This is an academic project. It is not a medical device and
> must not be used for diagnosis or any safety-critical decision.
