# DeepWiki — Codebase Walkthrough

A file-by-file guide to every module in `src/` and the supporting
`app/`, `notebooks/`, and `.github/` directories. Each entry gives
the file's purpose plus its key classes/functions in one to three
sentences. Treat this as a map to the code; for deep details, jump to
the actual file.

---

## Top-level layout

```
.
├── README.md             Project overview, headline metrics, MLOps map
├── app/                  Gradio web demo (deployed to HF Space)
├── src/                  Core ML pipeline (data, models, training, eval, inference)
├── notebooks/            Jupyter notebooks (data exploration, training, ablations)
├── outputs/              (gitignored) trained checkpoint, plots, TensorBoard, metrics
├── docs/                 Report, ablations, slides, proposal, this file
├── tests/                Unit tests for data / models / inference
├── scripts/              Shell scripts (run_ablations.sh, etc.)
├── assets/               Static images (architecture diagrams, screenshots)
├── pyproject.toml        Package metadata
├── requirements.txt      Repo-level Python deps
├── Dockerfile            Container build
├── docker-compose.yml    Local dev compose
└── .github/workflows/    CI/CD (ci.yml, deploy_hf.yml, retrain.yml)
```

---

## `app/` — Gradio web demo

### `app/app.py`

The Gradio app deployed to Hugging Face Spaces. Three tabs:
**Predict** (image upload / webcam → stress + fatigue + wellness),
**Methodology** (rubric-required design rationale surfaced in-app),
and **About**. Imports the real `StressFatigueModel` from
`src/models/` so checkpoint keys match training. Includes a
defensive `_startup()` that never lets a load failure prevent
Gradio from registering API routes, a `_to_pil()` helper that
normalizes Gradio 5.x image inputs (PIL / ndarray / dict / path) and
applies EXIF orientation, a `_placeholder_labels()` helper so
`gr.Label` never receives empty dicts, and a top-level try/except in
`predict()` so an exception cannot kill the worker (which would
otherwise manifest as "No API found" in the UI). Also includes a
monkey-patch on `gradio_client.utils` to tolerate bool JSON-Schema
values — a known bug in gradio==5.9.1 that crashes `launch()` on the
HF base image.

### `app/requirements.txt`

The Space's Python deps. Only `transformers==4.46.3` and
`huggingface_hub==0.26.2` are pinned (these govern the runtime
behaviour that matters for `AutoImageProcessor` + `hf_hub_download`);
everything else is unpinned so HF's base image can resolve compatible
versions. `gradio==5.9.1` is enforced by HF regardless of what's in
this file.

---

## `src/config/` — Configuration

### `src/config/config.yaml`

The single source of truth for training: paths, model architecture
(backbone name, freeze depth, dropout), training hyperparameters
(batch size, LRs, optimizer, schedule, loss weights, focal γ),
augmentation settings, and logging targets. Read by the trainer via
`omegaconf`. Dotted-key overrides (`training.lr_head=1e-4`) make
ablation variants one-liners.

### `src/config/sweep.yaml`

Weights & Biases sweep configuration (scaffold; not used in the
final run). Defines the search space for a Bayesian hyperparameter
sweep over learning rates, weight decay, dropout, and unfreeze depth.

---

## `src/data/` — Data pipeline

### `src/data/dataset.py`

Defines three PyTorch `Dataset` classes that emit a unified output
format: `{"image": tensor, "stress_label": long, "fatigue_label":
long, "subject_id": str}`. Missing labels are represented as `-1` so
the masked loss can ignore them.

Key classes/functions:

- `DrowsinessDataset` / `YawnEyeDataset` — fatigue-only dataset
  reader, maps eye-closed/yawn folder structure to alert vs fatigued.
- `FER2013Dataset` — stress-only dataset reader, applies the
  emotion→stress proxy mapping defined by `FER_EMOTION_TO_STRESS`.
- `CombinedFacialDataset` — concatenates the two single-task
  datasets into a multi-task training set with `-1` masking on the
  task-not-labelled side.
- `make_subject_grouped_split` — wraps scikit-learn's
  `GroupShuffleSplit` to produce subject-grouped 70/15/15 splits
  where subject metadata exists.
- `compute_class_weights` — inverse-frequency class weights for
  the training distribution, used by `MaskedCrossEntropy` and
  `FocalLoss`.

### `src/data/transforms.py`

Builds `torchvision.transforms.Compose` pipelines for training, eval,
and standalone inference. Training pipeline includes
`RandAugment(N=2, M=9)`, horizontal flip, and mild color jitter; eval
and inference are augmentation-free. All pipelines end with
DINOv2-statistic normalization (ImageNet mean/std).

### `src/data/download.py`

CLI script that downloads the two Kaggle datasets via the
`kaggle` API and unzips into a target folder. No-ops on Kaggle
notebooks where the datasets are already mounted at `/kaggle/input`.

---

## `src/models/` — Model definitions

### `src/models/stress_fatigue_model.py`

The end-to-end `StressFatigueModel(nn.Module)`. Composes a
`DINOv2Backbone` (or optional ResNet-50 baseline) with a
`MultiTaskHeads` block. Forward returns a `ModelOutput` dataclass
with `stress_logits`, `fatigue_logits`, `shared_features`, and
`backbone_features`. Provides `get_optimizer_param_groups()` for
discriminative learning rates (lower LR for backbone, higher LR for
heads) and `save_pretrained` / `from_pretrained` for HF-style save
and load.

### `src/models/backbone.py`

`DINOv2Backbone(nn.Module)` wraps `transformers.Dinov2Model` with
configurable partial-freeze (`_apply_freeze_schedule()`),
CLS-token extraction in `forward()`, and an optional
attention-map output for Grad-CAM-style visualisations
(`get_attention_for_cls()`). The `embed_dim` is read from the
`Dinov2Config` so the model adapts automatically if the backbone
is swapped.

### `src/models/heads.py`

Defines the multi-task head block:

- `SharedMLP` — `Linear → LayerNorm → GELU → Dropout` repeated
  twice (with an intermediate `Linear → LayerNorm` block, so the
  state dict has keys `net.0` through `net.7`).
- `ClassificationHead` — `BatchNorm1d → Dropout → Linear` for each
  task head, with small-std initialization on the final linear
  layer.
- `MultiTaskHeads` — composes `SharedMLP` with two
  `ClassificationHead`s (one for stress with 3 outputs, one for
  fatigue with 2). Returns a `HeadsOutput` dataclass.

---

## `src/training/` — Training loop and losses

### `src/training/losses.py`

Three loss classes:

- `FocalLoss(γ, α, ignore_index)` — multi-class focal loss with
  optional per-class α weights and `-1` ignoring. Implements the
  standard formula `-α (1-p)^γ log p`.
- `MaskedCrossEntropy(weight, ignore_index)` — class-weighted CE
  that ignores `-1` labels.
- `MultiTaskLoss` — combines the per-task losses with configurable
  coefficients (`α=β=1.0, γ=0.5` for stress focal).

### `src/training/trainer.py`

The `Trainer` class. Wraps the train loop with TensorBoard logging,
W&B logging (optional, soft-fails if not configured), AMP
mixed-precision support, grad clipping, cosine-warmup LR schedule
(`_build_scheduler`), per-epoch validation, and best-checkpoint
saving on the `val/balanced_acc_mean` metric. Designed to be runnable
from both the CLI entry point and the training notebook.

### `src/training/callbacks.py`

- `EarlyStopping` — patience-based early stop on a validation
  metric in `max` or `min` mode.
- `CheckpointTracker` — top-K checkpoint manager that evicts the
  worst when a better one arrives.

### `src/training/train.py`

CLI entry point. Parses args (config path + dotted overrides via
`omegaconf.OmegaConf.from_dotlist`), sets seed, builds datasets,
dataloaders, model, loss, optimizer, scheduler, trainer, and runs
the training loop. Run as `python -m src.training.train --config
src/config/config.yaml`.

---

## `src/evaluation/` — Metrics and ablations

### `src/evaluation/metrics.py`

Pure-numpy / sklearn metric computation for the test set:

- `metrics_for_task(probs, target, num_classes)` — returns a dict
  with balanced accuracy, macro-F1, ROC-AUC, ECE, and `n_samples`.
- `_ece(probs, target, n_bins=15)` — 15-bin expected calibration
  error.
- Plotting helpers for confusion matrices and reliability diagrams
  (writes the four PNGs in `outputs/plots/`).

### `src/evaluation/ablation.py`

CLI driver that runs ablation variants A1–A6 by overriding the
training config (each variant is a list of dotted-key overrides).
Aggregates results into a single JSON for the writeup. Scaffolded
but not used for the final report — we deliver a design-choice
analysis in `docs/ablations.md` grounded in the single full run +
literature instead of running every variant (compute budget
constraint).

---

## `src/inference/` — Web-app inference pipeline

### `src/inference/face_detector.py`

`FaceDetector` — MediaPipe-based face detector + bounding-box crop
with configurable margin. On import failure (MediaPipe missing) it
falls back to a no-op so the app still runs. Picks the largest face
when multiple are detected.

### `src/inference/llm_explainer.py`

Generates the wellness recommendation:

- `LLMExplainer` — talks to the Anthropic API (Claude) with a
  carefully engineered system prompt, three few-shot examples for
  tone consistency, explicit constraints (length, no medical
  claims), and `temperature=0.7`.
- `ExplainerInput` — dataclass with `stress_label`,
  `fatigue_label`, and per-task confidences.
- Falls back to a templated canned response if `ANTHROPIC_API_KEY`
  is missing or the API errors.

### `src/inference/predict.py`

End-to-end inference orchestrator: face detection → preprocessing
→ model forward → LLM explanation. Returns a `PredictionResult`
dataclass with class names, probabilities, confidences, the
wellness message, latency in ms, and `face_detected`. Used by both
the Gradio app and the (optional) standalone CLI inference.

---

## `src/utils/`

### `src/utils/logging.py`

`setup_logger(name, level, log_file)` — small wrapper around
Python's stdlib `logging` that builds a logger with console + optional
file output and consistent formatting. Idempotent (clears handlers
before adding).

---

## `notebooks/`

### `notebooks/01_data_exploration.ipynb`

Data exploration: class distributions, sample visualizations,
emotion-to-stress mapping sanity checks.

### `notebooks/02_training_kaggle.ipynb`

**Main training entry point.** Pure-Python notebook that runs on a
free Kaggle Tesla T4. Mounts the two datasets, calls into
`src.training.train`, captures TensorBoard logs, saves the best
checkpoint, generates the four PNG plots, and uploads the model to
the HF Hub. Designed to "Save & Run All" without manual intervention
given a `HF_TOKEN` Kaggle secret.

### `notebooks/03_ablation_studies.ipynb`

Ablation-runner notebook (scaffold). Iterates over the variants in
`src/evaluation/ablation.py`. Not used for the final report;
`docs/ablations.md` is the delivered ablation discussion.

---

## `.github/workflows/`

### `.github/workflows/ci.yml`

CI on every push: lint with `ruff`, unit tests with `pytest`,
project-structure checks.

### `.github/workflows/deploy_hf.yml`

The deploy pipeline. On every push to `main`, copies `app/app.py`,
`app/requirements.txt`, and the entire `src/` tree into `/tmp/space/`,
adds a generated README with the HF Space front-matter, initializes
a fresh git repo, force-pushes to the HF Space remote, which then
auto-rebuilds and redeploys the Gradio app. Authenticated via
`HF_TOKEN`, `HF_USERNAME`, `HF_SPACE_NAME` repo secrets.

### `.github/workflows/retrain.yml`

Scheduled retraining scaffold (cron). Not actively used; documents
the intended retrain pathway for future work.

### `.github/workflows/space_ops.yml`

Operations workflow with a `workflow_dispatch` input (`status` /
`restart` / `factory_rebuild` / `logs`). Lets a developer drive the
HF Space programmatically from the CLI via `gh workflow run
space_ops.yml -f action=<...>`. Used during debugging to remotely
restart the Space and poll runtime status. Authenticates with the
same `HF_TOKEN` secret as `deploy_hf.yml`.

---

## `tests/`

Unit tests for data loaders, transforms, model forward pass, and
inference orchestrator. Run with `pytest tests/`. Not exhaustive but
covers the parts of the system most prone to silent breakage.

---

## `outputs/` (gitignored)

Generated artifacts from a training run:

- `outputs/dinov2_multitask_full/checkpoints/` — saved
  `pytorch_model.bin` and `config.json`
- `outputs/dinov2_multitask_full/tb/` — TensorBoard event files
- `outputs/dinov2_multitask_full/test_metrics.json` — verbatim test
  metrics (the source of every number in the report)
- `outputs/plots/cm_stress.png` — stress confusion matrix
- `outputs/plots/cm_fatigue.png` — fatigue confusion matrix
- `outputs/plots/calibration_stress.png` — stress reliability diagram
- `outputs/plots/calibration_fatigue.png` — fatigue reliability diagram
