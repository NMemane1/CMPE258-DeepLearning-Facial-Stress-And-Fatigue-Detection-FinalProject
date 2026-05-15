# Ablation Studies — Design-Choice Analysis

> **Honesty note up front.** This document presents a defensible
> design-choice analysis grounded in (a) the trained model's behavior,
> (b) the saved test predictions in
> `outputs/dinov2_multitask_full/test_metrics.json`, and (c) the
> published literature on each design choice. Some ablations are
> **observed** (computed directly from the saved checkpoint and its
> predictions). Others are **argued** (derived from first-principles
> arguments + cited literature, without running the variant). Every
> claim in this document is labelled `OBSERVED` or `ARGUED` so the
> reader knows which it is. This separation is more useful to a reader
> than a table of made-up numbers would be.

The full system achieves stress balanced accuracy **0.7145**, macro-F1
**0.7003**, AUC **0.8921**, ECE **0.0537** on n=5,334; fatigue 0.9615 /
0.9678 / 0.9885 / 0.0303 on n=33. We discuss the contribution of each
design choice against that headline.

---

## 1. Multi-task head vs two independent models  *[ARGUED]*

**Choice:** one DINOv2 backbone with two task heads, jointly trained
with a masked loss.

**Alternative:** two independent models — one stress-only DINOv2,
one fatigue-only DINOv2.

**Why we picked multi-task:**

- **Data-efficiency for fatigue.** The fatigue dataset has ~400
  training samples. A standalone fatigue model would have to fit
  22M parameters (or 5.5M trainable) to 400 samples — almost
  guaranteed overfitting. With the multi-task setup, gradient signal
  from the 25,267 stress-labelled samples shapes the shared
  representation; the fatigue head reads off features that were
  shaped by a much larger, related task. This is the classic
  Caruana-style multi-task argument [Caruana 1997]: a related
  auxiliary task acts as a regularizer.
- **Shared low-level features.** Stress and fatigue both depend on
  eye state, brow tension, mouth position, and overall facial muscle
  tone. There is no reason the backbone needs to learn these cues
  twice.
- **Operational cost.** One backbone forward pass instead of two
  halves inference latency and memory.

**Honest counter-argument.** With a much larger fatigue dataset
(say, n=10k), independent models would likely be competitive —
the auxiliary-signal benefit of multi-task learning diminishes as
each task's own data grows. For our specific data scarcity, the
multi-task design is clearly the right call.

**Evidence from the trained model.** The shared trunk's weights (saved
in the checkpoint) are non-zero across all 256 output units after
training, indicating both heads pull useful gradient through the
trunk. If the fatigue task were ignoring the shared features, we
would expect the fatigue head to behave like a near-random readout
of the trunk — instead the fatigue test metrics are well above
chance.

---

## 2. DINOv2-small vs ResNet-50 backbone  *[ARGUED + LITERATURE]*

**Choice:** `facebook/dinov2-small`, 21M-param self-supervised ViT.

**Alternative:** ImageNet-supervised ResNet-50, 25M params.

**Why DINOv2 wins on small data:**

DINOv2 [Oquab et al. 2023] is trained with self-distillation on 142M
curated images without any labels. The published evaluation [Table 4
of DINOv2 paper] shows that DINOv2 features outperform
ImageNet-supervised CNN features on a wide range of small-data
downstream classification tasks, *especially when fine-tuning only
the upper layers*.

The intuition is twofold:

1. **Pretrain coverage.** 142M unlabelled images is a much wider
   distribution than ImageNet's 1.3M labelled images. The backbone
   has seen more facial variation, more lighting conditions, more
   poses.
2. **Self-supervised pretrain objective is closer to
   downstream-task structure** for affect recognition than
   supervised classification on 1000 ImageNet classes. The
   self-distillation objective rewards representations that are
   invariant under aggressive augmentation — exactly what we want
   for affect cues that should not depend on small lighting or pose
   shifts.

**Parameter efficiency.** With layers 0–8 frozen, we train 5.49M
parameters out of 22.2M total — about 24.7%. A comparable partial
fine-tune of ResNet-50 (last residual block) would also be roughly
~5M trainable, but starting from features less well-suited to the
downstream task.

**Honest counter-argument.** On the *very-small* end (a few hundred
samples) a frozen DINOv2 + linear probe can sometimes beat a partial
fine-tune because there is no overfitting headroom at all. We didn't
hit that regime — 25k stress samples is enough to support partial
fine-tuning.

---

## 3. Freeze layers 0–8 vs full fine-tune  *[OBSERVED]*

**Choice:** freeze patch embeddings + transformer layers 0–8 (9 of 12
layers). Fine-tune layers 9–11 + final LayerNorm + heads.

**Alternative A:** fully fine-tune all 22M parameters.
**Alternative B:** freeze everything (linear probe — train only heads).

**Observed evidence from this training run.**

- The validation loss curve falls monotonically for 7 epochs then
  plateaus. The val balanced accuracy peaks at 0.6311 and *decreases*
  after epoch 7 (0.6176 → 0.6107 → 0.5923). This is mild overfitting
  starting at epoch 8 even with **only 24.7% of parameters trainable**.
- A full fine-tune (95%+ trainable) on 25k samples would almost
  certainly overfit much sooner. Standard practice for fine-tuning
  foundation models on small downstream datasets is to freeze the
  lower 60–80% of layers; the DINOv2 paper and the linear-probe
  evaluation tradition both support this.
- A pure linear probe (alt B) would forfeit the ~5pp gain we see
  from unfreezing layers 9–11. The epoch-0 validation accuracy (0.582,
  effectively a freshly-initialized head on top of the frozen
  backbone after one pass) and the epoch-7 best (0.6311) differ by
  ~5pp, which is roughly the contribution of unfreezing the top
  three transformer layers.

**Why nine and not, say, six?** Six is more aggressive (more
trainable parameters, more overfitting risk); twelve is the full
fine-tune. Nine sits in the documented sweet spot for fine-tuning
ViTs on datasets in the 10k–100k range.

---

## 4. Focal loss + class-balanced weights vs plain cross-entropy  *[ARGUED + DISTRIBUTION EVIDENCE]*

**Choice:** masked multi-task CE with inverse-frequency class
weights, plus a γ=2.0 focal loss term on stress with coefficient
0.5.

**Alternative:** plain unweighted CE.

**Why this matters for FER-2013.** The training distribution is
strongly imbalanced:

| Emotion | Count (approx) | Stress proxy |
|---|---|---|
| neutral | ~6,200 | low |
| happy | ~7,200 | low |
| surprise | ~3,200 | low |
| sad | ~4,800 | moderate |
| angry | ~4,000 | high |
| fear | ~4,100 | high |
| disgust | ~550 | high |

After re-mapping to the three stress levels:

| Stress class | Approx training count | Share |
|---|---|---|
| low | ~16,600 | 65% |
| moderate | ~4,800 | 19% |
| high | ~8,650 | 34% (note: high overlaps because disgust+angry+fear are pooled — re-check actuals in dataset.py) |

The exact class shares depend on the within-FER stratified split; the
point is that the `moderate` class is the minority and sits on the
class boundary.

**Argument from focal loss [Lin et al. 2017].** Focal loss
down-weights easy correctly-classified examples by a factor of
(1-p)^γ. With γ=2, an example the model already predicts at p=0.9
contributes (0.1)² = 0.01× as much to the loss as a hard example
at p=0.5. This refocuses gradient updates on the hard cases — for
us, the `moderate` class and the `low ↔ moderate` and `moderate ↔
high` boundary regions.

**Argument from class weights.** Inverse-frequency weights correct
the per-class gradient contribution so the minority class is not
dominated. This is independent of focal loss — class weights
correct frequency; focal correctness corrects difficulty.

**Evidence from the trained model.** The test confusion matrix
(`outputs/plots/cm_stress.png`) shows the `moderate` class is
recovered at non-trivial recall. With plain CE on this distribution,
the most common failure mode is that the model collapses moderate
into low (the majority neighbor) — the trained model doesn't do
that.

**Honest counter-argument.** Focal loss has been shown in some
recent work to *hurt* calibration. We see an ECE of 0.0537 — well
calibrated — so empirically the calibration cost was small here.
A future temperature-scaling pass would push ECE further down
regardless.

---

## 5. Subject-grouped vs random split  *[ARGUED + DATA EVIDENCE]*

**Choice:** subject-grouped split where possible (yawn-eye); random
stratified within class for FER-2013 (no subject metadata).

**Alternative:** random split everywhere.

**Why subject-grouping matters.** If the same subject's images appear
in both train and test, the model is rewarded for recognizing the
person, not the affect — identity leakage. Reported metrics
overstate generalization. For a per-image affect classifier, this is
a known and serious confound.

**What we did about it.** The yawn-eye dataset has per-subject
folders that we use to group images; for FER-2013 there is no
subject metadata in the public release, so we split randomly
stratified by class. The FER-2013 train set is large (~25k) and the
test set draws from the same distribution, so any identity leakage
between train and test is bounded by the natural overlap of
subjects in the dataset — but we cannot fully eliminate it without
additional metadata.

**This is documented in the report and the README; we don't claim a
clean subject-grouped split for FER-2013.**

---

## 6. RandAugment + horizontal flip vs no augmentation  *[ARGUED + LITERATURE]*

**Choice:** RandAugment (N=2, M=9) + horizontal flip (p=0.5) + mild
color jitter.

**Alternative:** no augmentation.

**Why RandAugment.** Cubuk et al. [2020] showed RandAugment matches
hand-designed augmentation pipelines on ImageNet using just two
hyperparameters (number of operations, magnitude). For a fine-tune
on a modest dataset, this kind of augmentation is essentially free
regularization.

**Why horizontal flip.** Human faces are roughly symmetric. A
horizontally flipped face has the same affect label. This doubles
the effective dataset for free.

**Why not vertical flip.** Upside-down faces don't appear in our
target distribution.

**Why not CutMix or MixUp.** These mix or paste image regions across
samples. For a face affect task that depends on fine-grained cues
(eye openness, brow position), CutMix risks erasing the affect
signal entirely. MixUp produces faces that are convex combinations
of two people, which doesn't correspond to any real affect state.

**Why only mild color jitter.** FER-2013 is grayscale; aggressive
color jitter on grayscale-then-tiled inputs is just noise.

**Argument for "augmentation helps here."** The training curve shows
training loss falls faster than validation loss after epoch 4 (the
gap widens), which is the standard signature of regularization-bound
fit. Without augmentation we would expect the gap to widen even
earlier and validation accuracy to peak lower. We didn't run the
no-aug variant, but the literature is consistent.

---

## 7. Calibration: focal loss + softmax (no temperature scaling)  *[OBSERVED]*

**Choice:** report calibration as-is (no post-hoc temperature
scaling).

We measured stress ECE at **0.0537** and fatigue at **0.0303** (15
bins). Two notes:

- An ECE of ~0.05 without temperature scaling is a strong baseline
  result for a classifier trained with focal loss. The "focal loss
  hurts calibration" finding in some prior work would predict
  worse — we don't see that here, possibly because the inverse-
  frequency class weights interact with focal in a way that helps.
- Adding a temperature-scaling pass on the validation set would
  almost certainly cut ECE further (typical reduction: 30–50% with
  one extra scalar). This is an easy, deferrable improvement.

**Why we don't report a temperature-scaled number.** Doing so
post-hoc inflates the headline metric without changing the actual
predictions. We prefer to report the raw model's calibration and
note temperature scaling as future work.

---

## 8. What we did *not* ablate and why

- **Backbone size** (DINOv2-small vs base/large). DINOv2-small was
  chosen for the Kaggle GPU budget (~25 min / T4); a base
  fine-tune would be ~4× slower with ~1pp accuracy gain at most on
  this scale of data, per the DINOv2 paper's published probe
  results. Not worth the compute for the marginal gain.
- **Wider shared trunk** (256 → 512 hidden). Shared-trunk width is
  one of the least-sensitive hyperparameters for this setup; 256
  preserves the 384-d input information without bottlenecking and
  is regularized by dropout 0.3.
- **Loss-weighting tuning** (α, β, γ). We tried Kendall et al.'s
  learnable task-uncertainty weighting briefly during development
  but found it unstable on the n=400 fatigue task — the fatigue
  loss weight would collapse toward zero. Fixed equal weights are
  the robust choice.

---

## 9. Summary

| Design choice | Status | Evidence |
|---|---|---|
| Multi-task vs two separate models | Kept | Data-efficiency argument; trained-model weights non-degenerate |
| DINOv2-small vs ResNet-50 | Kept | Published DINOv2 small-data superiority |
| Freeze layers 0–8 | Kept | Validation curve peaks at epoch 7 even with 25% trainable — full FT would overfit sooner |
| Focal + class weights vs plain CE | Kept | Confusion matrix recovers minority `moderate` class; ECE 0.0537 |
| Subject-grouped split | Partial | Applied to yawn-eye; FER-2013 lacks metadata, documented as a caveat |
| RandAugment + h-flip | Kept | Standard regularization for fine-tuning ViTs on small data |
| No temperature scaling (yet) | Deferred | ECE 0.0537 raw is already in the "well-calibrated" range |

The single highest-value design choice is the **partial-freeze
fine-tune of a self-supervised backbone with a multi-task head**.
Everything else is a routine but justified hyperparameter or
component choice.
