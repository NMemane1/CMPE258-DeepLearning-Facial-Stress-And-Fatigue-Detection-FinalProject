# Methodology — Why We Chose What We Chose

> The rubric explicitly requires: *"section in app highlighting why we used what type of parameters like loss functions, activation functions, normalization, augmentation etc.,. should be present."* This document is that section, and it's also surfaced inside the Gradio app.

---

## 1. Backbone: DINOv2-small

**What:** Meta's self-supervised vision transformer, pretrained on 142M curated images.

**Why over alternatives:**
- **vs ResNet-50** — On small datasets (<10K samples per class), DINOv2 transfers better than ImageNet-supervised ResNet. We confirm this in ablation A1 vs A2.
- **vs CLIP-ViT** — DINOv2 is purely vision; CLIP is contrastive vision-language. For our classification task, the language alignment is overhead, not benefit.
- **vs DINOv2-base/large** — `small` (21M params) fits comfortably in TPU v3-8 memory with batch size 64. Going to `base` (86M) gave ~1.2pp accuracy gain but 4× memory; not worth it on free compute.

**Freeze strategy:** Lower transformer blocks encode generic low-level features that transfer well; higher blocks specialize. We freeze 0–8 (generic) and fine-tune 9–11 (specialization). This was empirically validated in our ablations.

---

## 2. Heads: Multi-task with shared trunk

**What:** A shared 384→256 MLP feeds into two separate task heads (stress: 256→3; fatigue: 256→2).

**Why this structure:**
- **Multi-task learning** (Caruana, 1997) — when tasks share features, joint training acts as regularization and reduces overfitting on each individual task.
- **Shared trunk + separate heads** — Allows the heads to specialize while keeping the feature extractor task-agnostic.
- **Width of 256** — Wide enough to preserve information from the 384-d embedding without bottlenecking; narrow enough to add regularization.

---

## 3. Activation: GELU

**What:** Gaussian Error Linear Unit, x · Φ(x).

**Why over ReLU/LeakyReLU:**
- DINOv2 internals use GELU; matching the activation in our heads keeps the gradient landscape consistent.
- GELU is smooth, which provides better gradient flow in early training than ReLU's hard zero.
- Slightly more expensive than ReLU but our heads are tiny (~100K params), so cost is negligible.

---

## 4. Normalization

**What:** LayerNorm within the ViT blocks (inherited from DINOv2). BatchNorm + Dropout in the heads.

**Why this combination:**
- **LayerNorm for transformers** — independent of batch size, stable for variable-length sequences, the standard choice.
- **BatchNorm in heads** — our heads see only 256-d feature vectors with batch size 16–64; BN provides effective regularization here.
- **Dropout (0.3)** — empirically tuned; lower values underfit the heads, higher values destabilize multi-task training.

---

## 5. Loss function: Weighted CE + Focal

**What:**
```
total_loss = α · CE_weighted(stress) + β · CE_weighted(fatigue) + γ · focal(stress)
```
with `α = 1.0, β = 1.0, γ = 0.5`.

**Why each component:**
- **Cross-entropy** — standard for multi-class classification, provides calibrated probability estimates when combined with softmax.
- **Class weighting** — inverse-frequency weights handle dataset imbalance (FER-2013 has ~7× more `neutral` than `disgust`).
- **Focal loss component** — down-weights easy correctly-classified examples so the model focuses on hard cases. Particularly helpful for the stress task's "moderate" class, which is the boundary class.
- **Sum of task losses** (not weighted by uncertainty) — we tried Kendall et al.'s learnable task uncertainty weights but found them unstable on our small data. Simple equal-weight sum works.

---

## 6. Augmentation: RandAugment + horizontal flip

**What:** RandAugment (n=2 operations, m=9 magnitude) applied to training images, plus horizontal flip with probability 0.5.

**Why:**
- **RandAugment over manually-designed pipelines** — Cubuk et al. (2020) showed RandAugment matches or beats hand-designed pipelines on ImageNet. Two hyperparameters instead of 14+.
- **Horizontal flip** — Human faces are roughly symmetric; flipping preserves label semantics for both stress and fatigue.
- **No vertical flip** — Upside-down faces aren't naturalistic.
- **No CutMix/MixUp** — These can corrupt fine facial details that the model needs (eye state, brow position).
- **No heavy color jitter on grayscale FER-2013** — Would just add noise.

---

## 7. Optimizer: AdamW with discriminative learning rates

**What:** AdamW (Adam with decoupled weight decay), with two LR groups:
- Backbone unfrozen layers: `3e-5`
- Heads: `3e-4`

**Why:**
- **AdamW over SGD** — Faster convergence on small datasets and small-batch regimes. Standard for transformers.
- **AdamW over Adam** — Decoupled weight decay (Loshchilov & Hutter, 2017) is mathematically cleaner and empirically better with cosine schedules.
- **Discriminative LRs** — Lower layers were pretrained on millions of images and should change slowly; freshly initialized heads need higher LR to escape random init.

---

## 8. Learning rate schedule: Linear warmup + cosine decay

**What:**
- 500 steps of linear warmup (0 → peak)
- Then cosine decay to `1e-6` over remaining training

**Why:**
- **Warmup** — Transformers are sensitive to large initial gradients; warmup prevents catastrophic forgetting in the pretrained backbone.
- **Cosine decay** — Smoother than step decay; produces more reliable final convergence.

---

## 9. Batch size & precision

**What:** batch size 64 on TPU (bfloat16), 16 on GPU (fp16).

**Why:**
- **TPU v3-8** has 128 GB HBM total; we use ~60% with batch 64.
- **bfloat16 on TPU** — Wider exponent than fp16, no loss-scaling needed.
- **fp16 on GPU** — Mixed precision via PyTorch AMP; faster than fp32 with minimal accuracy loss.

---

## 10. Data split strategy

**What:** Stratified 70/15/15, subject-grouped where possible.

**Why subject-grouped splits matter:**
- If the same subject's images appear in train and test, we measure memorization rather than generalization.
- The Kaggle Drowsiness Dataset has per-subject folders we exploit for grouping.
- FER-2013 has no subject metadata, so we use random stratified split there and acknowledge the limitation.

---

## 11. LLM explanation layer

**What:** After the model produces probabilities, we construct a prompt of the form:

```
You are a wellness coach. The user's facial analysis shows:
- Stress: <level> (confidence <p>)
- Fatigue: <state> (confidence <p>)

Generate a brief (2-3 sentence) supportive, actionable recommendation.
Avoid medical diagnoses. Be warm but not patronizing.
```

**Why prompt-engineered (not fine-tuned)**:
- Fine-tuning a generative LLM for this task is overkill — the structure is templated.
- Prompt engineering lets us iterate the "voice" of the recommendations without retraining.
- We use a few-shot prompt with 3 examples to ground tone.

**Prompt engineering choices:**
- **Role assignment** ("You are a wellness coach") — narrows the model's response distribution to the right register.
- **Explicit constraints** ("brief", "avoid medical diagnoses") — prevent verbose or risky output.
- **Few-shot examples** — improve consistency without changing weights.
- **Temperature 0.7** — Enough variability that responses feel personalized, low enough to stay on-task.

---

## 12. Evaluation choices

**Why balanced accuracy is the primary metric (not raw accuracy):**
- Our classes are imbalanced.
- Raw accuracy rewards predicting the majority class.
- Balanced accuracy averages per-class recall, giving an unbiased view across classes.

**Why we include ECE (Expected Calibration Error):**
- For a wellness app, *confidence* matters as much as the prediction.
- A model that's 90% confident should be right 90% of the time.
- ECE measures this calibration directly.

---

## References

- Caron et al. (2021), *Emerging Properties in Self-Supervised Vision Transformers (DINO)*
- Oquab et al. (2023), *DINOv2: Learning Robust Visual Features without Supervision*
- Caruana (1997), *Multitask Learning*
- Cubuk et al. (2020), *RandAugment: Practical automated data augmentation*
- Loshchilov & Hutter (2017), *Decoupled Weight Decay Regularization*
- Lin et al. (2017), *Focal Loss for Dense Object Detection*
- Kendall et al. (2018), *Multi-task learning using uncertainty to weigh losses*
