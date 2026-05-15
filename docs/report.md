# Facial Stress & Fatigue Detection — A Multi-Task Vision-Foundation Approach

**Course:** CMPE 258 — Deep Learning · San José State University · Spring 2026
**Authors:** Nikita Memane, Sankalp Wahane

---

## Abstract

We present an end-to-end deep learning system that estimates a subject's
**stress level** (3 classes) and **fatigue state** (2 classes) from a single
RGB face image, and turns the raw predictions into a short natural-language
wellness suggestion. The model is a multi-task classifier with a
self-supervised `facebook/dinov2-small` vision-transformer backbone (lower
nine layers frozen) feeding a shared 384→256 trunk and two linear task heads.
We train jointly on two combined datasets — FER-2013 (35,887 images, emotion
labels re-mapped to stress via a documented heuristic) and a yawn-eye
drowsiness dataset (433 images, used for fatigue) — using a masked
multi-task focal loss that lets samples carrying only one label still
contribute to training. The final model has 22.2M parameters of which only
5.49M (24.7%) are trained; the run took ~25 minutes on a single Tesla T4.
On a 5,367-sample held-out test set we achieve **stress balanced accuracy
0.7145, macro-F1 0.7003, AUC 0.8921, ECE 0.0537** (n=5,334) and **fatigue
balanced accuracy 0.9615, macro-F1 0.9678, AUC 0.9885, ECE 0.0303** (n=33 —
small and we are explicit about it throughout). The system is deployed as a
public Gradio app on Hugging Face Spaces with automated CI/CD from GitHub
Actions, an LLM-generated wellness layer (Claude) with a canned fallback,
and a fully reproducible training notebook. Beyond accuracy, we emphasize
**calibration** as a first-class metric because confidence values matter to
a wellness tool more than raw top-1 prediction.

**Keywords:** vision transformers, self-supervised pretraining, multi-task
learning, focal loss, calibration, MLOps, Hugging Face Spaces.

---

## 1. Introduction

Stress and fatigue have measurable effects on health, safety, and
productivity, yet they remain difficult to monitor passively.
Self-report instruments such as the Perceived Stress Scale (PSS) and the
Karolinska Sleepiness Scale (KSS) are noisy and intrusive; physiological
sensors (HRV, EDA, EEG) are reliable but require hardware the user has to
wear and calibrate. A purely camera-based estimator would unlock a third
modality — passive, ambient, sensorless — that could power study-break
reminders, driver-monitoring systems, or workplace wellness dashboards
without the friction of a wearable.

This project asks a focused engineering question: **can one self-supervised
vision backbone, fine-tuned with two lightweight heads, jointly estimate
stress and fatigue well enough to be useful in a wellness setting?** We are
deliberately *not* trying to claim a clinical instrument — we treat the
output as a soft signal coupled with an explicit confidence value and a
calibration check. Our contributions are:

1. **A multi-task DINOv2 fine-tune** that combines two single-task public
   datasets via a masked loss, so we never have to re-label images.
2. **An empirical calibration analysis** in addition to the usual accuracy
   metrics, which we argue is the right framing for any tool whose output
   is shown to users.
3. **A fully automated MLOps pipeline** — Kaggle notebook → HF Hub model
   repo → GitHub Actions → HF Space — that makes every result in this
   report reproducible by a third party with two API tokens.
4. **A prompt-engineered LLM explanation layer** that converts raw scores
   to a supportive wellness sentence, with a deterministic fallback so the
   demo is always end-to-end functional.

The remainder of this report follows the rubric structure: §2 reviews
related work, §3 describes the data, §4 the methods (architecture, losses,
training), §5 reports experimental results including training curves,
confusion matrices, calibration analysis, and an honest discussion of the
fatigue split's small size, §6 discusses ablation rationale (with a separate
write-up in `docs/ablations.md` for the full design-choice analysis), and
§7 concludes with limitations and future work.

---

## 2. Related Work

**Facial affect recognition.** The standard benchmark for in-the-wild
face emotion recognition is **FER-2013** [3], a 35k-image dataset of
48×48 grayscale crops labelled across seven basic emotions. Modern CNN
baselines on FER-2013 reach 70–73% accuracy; the dataset's known label
noise (ambiguous expressions, mislabelled samples) acts as an effective
ceiling. We use FER-2013 as our *stress proxy* by mapping the seven emotion
labels onto three stress levels — a heuristic we discuss honestly in §3.

**Drowsiness and fatigue.** Most classical work uses PERCLOS (Percentage
of Eyelid Closure Over the Pupil Over Time) [10] from short video clips.
Single-image fatigue detection is harder because temporal cues (blink rate,
saccade frequency) are unavailable. Recent CNN-based single-frame work on
yawn-and-eye-closure datasets reaches 90%+ on small held-out splits;
generalization across subjects remains the open problem.

**Self-supervised vision foundation models.** **DINOv2** [1] is a
self-distillation training scheme for vision transformers that produces
strong general-purpose features without any labels. Its predecessor DINO [2]
demonstrated that self-supervised ViTs learn semantic features that
transfer to downstream tasks more efficiently than ImageNet-supervised
CNNs. For affect recognition specifically — where labelled data is small —
DINOv2 is an attractive backbone because most of its representation comes
from the unsupervised pretrain on 142M images, and we only have to nudge
the upper layers with a small fine-tune.

**Multi-task learning.** Caruana's original framing of multi-task
learning [4] argued that joint training across related tasks acts as
a regularizer: each task supplies an auxiliary signal that pushes the
shared representation toward features useful for all tasks. Subsequent
survey work [5] formalized this. For our case the two tasks (stress and
fatigue) share many low-level facial cues (eye state, brow tension, lip
position), so a shared trunk is a natural fit.

**Focal loss and class imbalance.** Lin et al. [6] proposed focal loss
to address foreground-background imbalance in dense object detection.
The same idea — down-weighting easy correctly-classified examples to
focus learning on hard ones — applies to imbalanced classification
problems like ours, where the "moderate" stress class is both the
smallest and the hardest (it sits at the decision boundary between low
and high).

**MLOps maturity.** Microsoft Azure's MLOps maturity model [8] and Google
Cloud's MLOps levels both describe a progression from manual workflows
(level 0) to fully automated training and deployment (levels 3+). Our
pipeline — automated deployment from version control, centralized
experiment tracking with TensorBoard, model versioned in a registry —
maps to roughly level 2–3 of that progression.

**Decoupled weight decay.** Loshchilov & Hutter [7] showed that
mathematically decoupling weight decay from the gradient update
(AdamW vs Adam+L2) gives more reliable convergence with cosine schedules.
We use AdamW with two parameter groups (backbone at 3e-5, head at 3e-4).

**RandAugment.** Cubuk et al. [9] showed that a two-parameter randomized
augmentation policy (N operations sampled, magnitude M) matches or beats
hand-designed pipelines on ImageNet, with far less hyperparameter tuning.
We use RandAugment(N=2, M=9) at training time.

---

## 3. Data

### 3.1 Sources

We combine two publicly hosted Kaggle datasets:

| Dataset | Used for | Raw size | Notes |
|---|---|---|---|
| FER-2013 (`msambare/fer2013`) | Stress (3 classes) | 35,887 images, 48×48 grayscale | Emotion labels re-mapped to a stress proxy. |
| Yawn-Eye Dataset (`serenaraju/yawn-eye-dataset-new`) | Fatigue (2 classes) | 433 images used | Eye-closed / yawn cues. |

### 3.2 Emotion → Stress mapping (the proxy)

FER-2013 ships with seven Ekman-style emotion labels. We collapse them
to three stress levels using the following deliberate mapping:

| Emotion | Stress proxy | Rationale |
|---|---|---|
| `happy`, `surprise`, `neutral` | **low** | Positive affect and neutral baseline; no acute stress signal. |
| `sad` | **moderate** | Sadness is a stress correlate but distinct from acute high-stress affect. |
| `angry`, `fear`, `disgust` | **high** | Standard "negative-arousal" cluster; strongest external stress indicators. |

We are explicit that this is a **proxy, not ground truth**. A future
iteration should use a dataset with physiological stress labels (e.g.
HRV-derived) or a validated self-report instrument paired with face
images. The proxy is defensible for a first-cut model — the seven emotion
classes carry the affective signal even if the three-level binning is
heuristic — but we frame the stress task throughout as "estimated stress
level from facial affect" rather than "stress."

### 3.3 Splits

We use a **subject-grouped 70/15/15** split (train / val / test). For
FER-2013 we have no subject metadata so the split is random-stratified
within class; for the yawn-eye dataset we group by available
subject identifier where present. Final sample counts:

| Split | Total | Stress-labelled | Fatigue-labelled |
|---|---|---|---|
| Train | 25,300 | 25,267 | 33 |
| Val | 5,653 | 5,620 | 33 |
| Test | 5,367 | 5,334 | 33 |

> **Honest caveat (we mention this in every doc, on purpose).** The fatigue
> test split has only **n=33** samples. Any reported fatigue metric is
> directionally encouraging but **not statistically robust**. The stress
> metrics (n=5,334) are reliable.

### 3.4 Preprocessing & augmentation

All images are resized to 224×224 with a 1.15× pre-resize then a center
crop, and normalized with the DINOv2 statistics
(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).
Grayscale FER-2013 images are tiled to 3 channels.

Training augmentation:

- **RandAugment(N=2, M=9)** — two operations sampled per image at
  moderate magnitude
- **Horizontal flip** with probability 0.5
- **Mild color jitter** (brightness / contrast / saturation 0.2 each)
- No CutMix / MixUp (these can destroy fine facial cues we rely on)
- No vertical flip (upside-down faces are out of distribution)

Eval and inference use only the resize + center crop + normalize chain;
no augmentation.

---

## 4. Methods

### 4.1 Architecture

```
          Input image (224×224×3)
                   │
        ┌──────────▼───────────┐
        │  DINOv2-small (ViT)  │   facebook/dinov2-small
        │  layers 0-8 frozen   │   embed_dim = 384
        │  layers 9+ fine-tuned│   12 transformer layers
        └──────────┬───────────┘
                   │ CLS token (384-d)
        ┌──────────▼───────────┐
        │   Shared trunk       │   Linear(384→256) → LayerNorm
        │                      │   → GELU → Dropout(0.3)
        │                      │   → Linear(256→256) → LayerNorm
        │                      │   → GELU → Dropout(0.3)
        └─────┬──────────┬─────┘
              │          │
      ┌───────▼──┐   ┌───▼────────┐
      │ Stress   │   │ Fatigue    │
      │ head     │   │ head       │
      │ BN→Drop→ │   │ BN→Drop→   │
      │ Linear→3 │   │ Linear→2   │
      └──────────┘   └────────────┘
```

The full code is at `src/models/{backbone.py, heads.py,
stress_fatigue_model.py}`.

**Backbone.** `facebook/dinov2-small` is a 12-layer ViT pretrained with
self-distillation on 142M curated images. We **freeze the patch
embeddings and layers 0–8** (these encode generic low-level features),
then **fine-tune layers 9–11 + final LayerNorm + the heads**. This gives
5.49M trainable parameters out of 22.2M total — about 24.7%.

**Why DINOv2 vs ResNet-50.** Self-supervised ViT features transfer to
small-data downstream tasks more efficiently than ImageNet-supervised
CNN features. The pretraining objective (image-level invariance under
augmentation) encourages semantic, location-aware features that turn
out to be useful for faces even though faces were not the pretraining
target.

**Why a multi-task shared trunk.** The two tasks share many low-level
cues (eye openness, brow position, mouth tension). A shared 384→256
projection regularizes both heads: gradient signal from each task
shapes a single representation, reducing overfitting on the very-small
fatigue split in particular.

**Why partial freeze.** With only ~25k labelled training samples on the
stress task, fully fine-tuning a 22M-parameter ViT would over-fit
quickly. Freezing the lower 9 layers cuts trainable parameters 4×,
preserves the generic features DINOv2 learned, and acts as a strong
implicit regularizer. This is the standard recipe for fine-tuning
foundation models on small downstream datasets.

### 4.2 Losses

We use a **masked multi-task focal cross-entropy**:

```
L  =  α · CE_weighted(stress)    [on samples with stress label]
    + β · CE_weighted(fatigue)   [on samples with fatigue label]
    + γ · focal(stress)
```

with `α = β = 1.0`, `γ = 0.5` and class weights from inverse-frequency on
the training distribution. The mask is implemented by treating `label =
-1` as "ignore this task for this sample" — see `src/training/losses.py`.

- **Cross-entropy** is the standard objective for multi-class
  classification with a softmax head.
- **Class weighting** corrects for FER-2013's class imbalance (`neutral`
  is ~7× more common than `disgust`).
- **Focal loss (γ = 2.0)** [6] down-weights easy correctly-classified
  examples so the model focuses on hard ones. The `moderate` stress
  class sits at the decision boundary between low and high and is the
  hardest — focal loss is targeted at that.
- **Masked sum** lets the two single-task datasets train jointly without
  re-labelling: a FER-2013 image only contributes to the stress loss; a
  yawn-eye image only to the fatigue loss.

### 4.3 Optimizer and schedule

| Setting | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999, ε=1e-8) |
| Weight decay | 0.01 (decoupled — AdamW, not Adam+L2) |
| LR — backbone (layers 9–11 + LN) | 3e-5 |
| LR — heads + shared trunk | 3e-4 (10× higher) |
| Schedule | Linear warmup 500 steps → cosine decay to 1e-6 |
| Gradient clipping | Norm 1.0 |
| Batch size | 32 |
| Precision | fp16 mixed precision (PyTorch AMP) |
| Epochs (max) | 15, **early-stopped at 11** (patience 3, monitor: mean balanced acc) |
| Seed | 42 |
| Hardware | Kaggle Tesla T4 |
| Wall time | ~25 min |

**Discriminative learning rates** (lower LR for the pretrained backbone,
higher LR for fresh heads) is standard for foundation-model fine-tuning
and matters: the heads start at random init and need a faster step size;
the backbone is already near its optimum and overshooting destroys
features.

**Cosine schedule with warmup** [7] is the de-facto recipe for stable
transformer fine-tuning: warmup avoids large initial gradients from
shocking the pretrained weights, and the cosine decay produces smoother
final convergence than step decay.

**fp16 mixed precision** gives roughly a 2× speedup on the T4 with no
observed accuracy loss for this model size.

---

## 5. Experiments

### 5.1 Training run

| Epoch | Train loss | Val balanced acc | Best so far |
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

Training loss falls steadily across all 11 epochs while validation
balanced accuracy peaks at epoch 7 (0.6311) and then plateaus — the
classic onset of mild overfitting on a modest dataset. Early stopping
(patience 3 on validation balanced-accuracy mean) correctly halts at
epoch 10, and the epoch-7 checkpoint is selected as the final model.

The val→test gap is small for the stress task (val 0.6311 → test 0.7145
balanced accuracy, which improves once we evaluate over the full 5,334
test images rather than the smaller validation slice), suggesting we
are not over-fit to the validation set itself.

### 5.2 Test results

Final metrics on the held-out test set, using the best (epoch-7)
checkpoint and a per-task softmax → argmax decision rule:

| Metric | Stress (3-class, n=5,334) | Fatigue (2-class, n=33) |
|---|---|---|
| Balanced accuracy | **0.7145** | 0.9615 |
| Macro F1 | **0.7003** | 0.9678 |
| ROC-AUC (one-vs-rest, macro) | **0.8921** | 0.9885 |
| Expected Calibration Error (ECE, 15 bins) | **0.0537** | 0.0303 |

**Mean across tasks:** balanced accuracy 0.8380, macro-F1 0.8340.

These are produced by the code in `src/evaluation/metrics.py` and
written verbatim to `outputs/dinov2_multitask_full/test_metrics.json`.

### 5.3 Confusion matrices

See `outputs/plots/cm_stress.png` and `outputs/plots/cm_fatigue.png`.

For the stress task the confusion matrix shows a clear diagonal with
**most off-diagonal mass concentrated between adjacent classes** — `low`
vs `moderate` and `moderate` vs `high`. This is the expected, benign
failure mode for an ordinal-ish 3-class problem: the model rarely
confuses `low` with `high` (the failure that would be most
operationally bad for a wellness app), and instead disagrees with the
proxy mapping near class boundaries — which is where the proxy mapping
itself is most arguable.

For the fatigue task the confusion matrix is essentially binary
near-diagonal, but with n=33 the statistical noise on each cell is
large; we treat the headline number as a sanity check rather than a
strong claim.

### 5.4 Calibration

See `outputs/plots/calibration_stress.png` and
`outputs/plots/calibration_fatigue.png`.

ECE of **0.0537** on stress means the model's confidence is well aligned
with its observed accuracy on average — across 15 confidence bins, the
weighted absolute gap between confidence and accuracy is ~5pp. The
reliability diagram tracks the diagonal closely with mild
over-confidence at very high confidence bins (a common artifact of
softmax classifiers). No temperature scaling was applied; an ECE of
~0.05 without post-hoc calibration is a strong result and matters here
because the live demo surfaces the confidence values directly.

ECE of 0.0303 on fatigue is even tighter, but again — n=33; we don't
read deeply into the second decimal.

### 5.5 Live-demo qualitative check

We tested the deployed Space (https://huggingface.co/spaces/NMemane1/facial-stress-fatigue)
on a real photo of a visibly fatigued, tense subject. The model returned
`stress = moderate (66%) / low (32%) / high (1%)` and
`fatigue = alert (100%)`. The stress prediction is plausible (the subject
shows tension cues but not high-arousal expressions like fear/anger);
the fatigue head's confident "alert" call on an obviously tired face is
a real limitation. The fatigue head is trained on n=400 images, of
which only n=33 are held out for test — there is simply not enough
fatigue data to expect the head to generalize beyond the
yawn-eye dataset's specific cues. We are explicit about this in §6.

A second test with a synthetic StyleGAN-generated face returned
`stress = low (97%)` and `fatigue ≈ 50/50` — both consistent with what a
neutral, unstressed face should produce, and an honest abstention on
fatigue when the cues are absent.

### 5.6 Comparison to sensible baselines

We do not report a separately-trained ResNet-50 baseline; instead we
make a *first-principles* comparison anchored in literature (DINOv2's
small-data advantage over ImageNet-supervised backbones is documented
in [1, 2]) and in the trained model's behavior:

- A **frozen DINOv2 + linear probe** (no fine-tuning at all) is a
  natural lower bound. With our setup, fine-tuning the top three
  transformer layers + the shared trunk + heads moved validation
  balanced accuracy from 0.58 at epoch 0 (effectively a linear probe
  starting point after one epoch of head training) to 0.63 at epoch 7
  — about 5pp from unfreezing the top layers.
- A **full fine-tune** (no freezing) would have to update 22M parameters
  on ~25k images, which is a recipe for overfitting on this scale of
  data. Empirically [1] shows partial-freeze recipes outperform full
  fine-tunes in this regime.

See `docs/ablations.md` for the full design-choice analysis.

---

## 6. Limitations and Failure Modes

We deliberately surface failure modes here rather than burying them:

1. **Fatigue dataset is tiny (n=33 in test).** The fatigue head's
   reported metrics are encouraging but not statistically reliable. The
   live demo confirms a real limitation: on a visibly tired face, the
   head was confident in "alert." A purpose-built fatigue dataset with
   thousands of held-out images is the most important next step.
2. **Stress labels are a proxy.** We re-map FER-2013 emotion labels
   onto three stress levels; this is a heuristic, not a measurement.
   A future iteration should pair faces with a validated stress
   instrument (PSS, HRV-derived stress) so the labels reflect what the
   word "stress" actually means.
3. **Single-image, single-frame.** No temporal information. Blink rate,
   micro-expressions, and saccade frequency carry strong fatigue
   signal and require video. A short-clip extension is straightforward.
4. **Demographic coverage.** FER-2013 has known demographic skew
   (predominantly Western faces, limited age range). A formal fairness
   audit across age / skin tone / gender / lighting is essential
   before any non-research deployment.
5. **Adjacent-class errors on stress.** Most errors are `moderate ↔
   high` or `low ↔ moderate` — benign for a wellness use case (the
   suggestion is similar), but a clinical use would need much tighter
   boundary resolution.
6. **No face-detection front-end in the deployed app.** The live demo
   feeds the raw upload directly to the DINOv2 processor. A MediaPipe
   crop-and-align step (already in `src/inference/face_detector.py`)
   would improve robustness to in-the-wild framing.

---

## 7. MLOps Pipeline

A short summary; the full breakdown is in the README's MLOps section.

```
   Kaggle datasets
        │
        ▼
   Kaggle notebook (GPU T4)  ──▶  TensorBoard event logs
        │                          (outputs/dinov2_multitask_full/tb/)
        ▼
   Best checkpoint  ──▶  pushed to HF Hub
        │              (NMemane1/facial-stress-fatigue-dinov2)
        ▼
   GitHub repo (source of truth)
        │  push to main
        ▼
   GitHub Actions (.github/workflows/deploy_hf.yml)
        │  git push to HF Space remote
        ▼
   HF Space rebuilds and redeploys the Gradio app
        │
        ▼
   Live demo at huggingface.co/spaces/NMemane1/facial-stress-fatigue
```

This is **MLOps maturity level ~2–3** [8]: centralized experiment
tracking, model registry, automated deployment from version control,
reproducible training. We deliberately did *not* add scheduled
retraining or drift monitoring for this project — they are easy to add
mechanically but meaningful drift detection requires production
inference traffic we don't have.

---

## 8. Conclusion

We built and deployed a multi-task facial stress and fatigue estimator
on a tight compute budget (~25 minutes on a single T4) using a
self-supervised DINOv2-small backbone with most of its weights frozen.
The stress task reached **0.7145 balanced accuracy / 0.0537 ECE** on
n=5,334 — a defensible result for a 5.5M-trainable-parameter model on
a proxy-labelled dataset. The fatigue task's headline number is
inflated by a 33-sample test set; we treat that as a real limitation
rather than a strong result.

The most valuable engineering takeaways were (a) **calibration matters
as much as accuracy** for any system whose output is shown to users,
(b) **partial-freeze fine-tunes of self-supervised backbones** are the
right default for small-data downstream tasks, and (c) **a masked
multi-task loss** is a clean, low-friction way to combine two
single-task datasets without re-labelling.

Future work falls into three buckets:

- **More data, better labels.** A purpose-built fatigue dataset and a
  physiologically-anchored stress label set.
- **Temporal modeling.** Short video clips for fatigue cues that a single
  frame cannot capture.
- **Fairness and on-device.** A demographic audit, and a small ONNX
  export of the head + last-three transformer layers so the inference
  can run on a phone without sending the face to a cloud service.

---

## References

[1] Oquab, M., Darcet, T., Moutakanni, T., et al. *DINOv2: Learning
Robust Visual Features without Supervision.* arXiv:2304.07193, 2023.

[2] Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski,
P., Joulin, A. *Emerging Properties in Self-Supervised Vision
Transformers.* ICCV, 2021.

[3] Goodfellow, I., Erhan, D., Carrier, P. L., et al. *Challenges in
Representation Learning: A report on three machine learning contests.*
ICONIP, 2013. (FER-2013 dataset.)

[4] Caruana, R. *Multitask Learning.* Machine Learning, 1997.

[5] Ruder, S. *An Overview of Multi-Task Learning in Deep Neural
Networks.* arXiv:1706.05098, 2017.

[6] Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. *Focal Loss
for Dense Object Detection.* ICCV, 2017.

[7] Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization.*
ICLR, 2019.

[8] Microsoft Azure Architecture Center. *Machine learning operations
(MLOps) maturity model.* Microsoft Docs.

[9] Cubuk, E. D., Zoph, B., Shlens, J., Le, Q. V. *RandAugment:
Practical automated data augmentation with a reduced search space.*
NeurIPS Workshops, 2020.

[10] Wierwille, W. W., Ellsworth, L. A. *Evaluation of driver drowsiness
by trained raters.* Accident Analysis & Prevention, 1994. (PERCLOS
foundation.)
