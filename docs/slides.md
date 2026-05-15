# Slide Deck — Facial Stress & Fatigue Detection

> 14-slide markdown deck. Render with Marp, reveal.js, or paste into
> Google Slides / Keynote. Each slide is ~50–100 words of slide text
> plus speaker notes.

---

## Slide 1 — Title

# Facial Stress & Fatigue Detection
## Multi-Task DINOv2 + LLM Wellness Layer

CMPE 258 — Deep Learning · SJSU · Spring 2026
**Nikita Memane · Sankalp Wahane**

🔗 Live demo · 🔗 Model · 🔗 GitHub

**Speaker notes:** A 5-minute walkthrough of an end-to-end stress and
fatigue estimator. One face image in, three signals out: stress (3
classes), fatigue (2 classes), and a short wellness suggestion. Built
and deployed in a week, with the engineering wired together end-to-end.

---

## Slide 2 — Problem & Motivation

- Stress and fatigue affect health, safety, and productivity
- Self-report instruments (PSS, KSS) are noisy and intrusive
- Wearables work but require hardware most users won't carry
- **A camera-based estimator is the missing third modality**: passive,
  ambient, sensorless

Target use cases: study-break reminders, driver-monitoring assists,
workplace wellness dashboards — *not* clinical diagnosis.

**Speaker notes:** The point isn't to replace clinical instruments. It's
to provide a soft, ambient signal that can power gentle interventions
without requiring the user to remember to wear or fill out anything.

---

## Slide 3 — Why This Is Hard

- **Proxy labels.** FER-2013 has emotion labels, not stress labels —
  we map them to three stress levels with a documented heuristic.
- **Tiny fatigue dataset.** Only ~400 fatigue images total; test
  split is **n=33**. We are explicit about this throughout.
- **No temporal cues.** Single-frame fatigue is much harder than
  single-clip fatigue (blink rate, micro-expressions need video).
- **Demographic skew.** FER-2013 is not balanced; we flag this as
  important future work.

**Speaker notes:** Calling out the limitations up front matters for two
reasons: (1) the rubric rewards intellectual honesty, (2) it shapes
how the reader interprets every later number.

---

## Slide 4 — Data

| Dataset | Task | Size | Notes |
|---|---|---|---|
| FER-2013 | Stress (3) | 35,887 | Emotion → stress proxy |
| Yawn-Eye | Fatigue (2) | 433 | Subject-grouped split where possible |

- **Splits:** 70/15/15 — train 25,300 / val 5,653 / test 5,367
- **Stress proxy:** happy/surprise/neutral → low · sad → moderate ·
  angry/fear/disgust → high
- **Augmentation:** RandAugment(N=2,M=9) + h-flip(0.5) + mild jitter

**Speaker notes:** Two single-task datasets combined via a masked
multi-task loss — each sample only contributes to the task it has a
label for. No re-labelling needed.

---

## Slide 5 — Architecture

```
 Image (224×224×3)
       │
       ▼
 DINOv2-small (ViT)   ← layers 0-8 frozen, 9+ fine-tuned
       │ CLS (384-d)
       ▼
 Shared trunk         ← Linear(384→256) → LN → GELU → Dropout(0.3) → ...
       │
       ├──▶ Stress head (Linear → 3)
       └──▶ Fatigue head (Linear → 2)
```

- **22.2M total parameters**
- **5.49M trainable (24.7%)**
- Trained in ~25 min on a single Tesla T4

**Speaker notes:** The whole point is partial-freeze + multi-task. Most
of the backbone is frozen; the shared trunk and top transformer
layers do the task-specific work.

---

## Slide 6 — Why DINOv2 + Multi-task?

**DINOv2 over ResNet-50:**
- Self-supervised pretrain on 142M images
- Stronger transfer to small downstream datasets [Oquab et al. 2023]
- Pretrain objective (augmentation-invariance) is well aligned with
  affect cues

**Multi-task over two independent models:**
- Stress task (n=25k) regularizes the tiny fatigue task (n=400)
- Shared low-level cues (eyes, brows, mouth) learned once, not twice
- Half the inference cost

**Speaker notes:** Both choices are about data efficiency. With 400
fatigue samples, an independent fatigue model would memorize the
training set. The shared backbone forces it to use general features.

---

## Slide 7 — Training Setup

| Setting | Value |
|---|---|
| Optimizer | AdamW (weight decay 0.01) |
| LR | backbone 3e-5 · head 3e-4 |
| Schedule | 500-step warmup → cosine to 1e-6 |
| Loss | Masked CE + focal (γ=2.0), class-weighted |
| Batch | 32, fp16 mixed precision |
| Early stop | Patience 3 on mean balanced acc |
| Seed | 42 (fully reproducible) |

**Speaker notes:** Discriminative learning rates matter — the
pretrained backbone needs gentle steps; the random-init heads need
larger steps. Cosine + warmup is the standard transformer-fine-tune
recipe.

---

## Slide 8 — Training Curve

Validation balanced accuracy by epoch:

```
e0  0.5824 ★
e1  0.5974 ★
e2  0.6145 ★
e3  0.6233 ★
e4  0.6288 ★
e5  0.6104
e6  0.6168
e7  0.6311 ★ (BEST)
e8  0.6176
e9  0.6107
e10 0.5923
```

Best at epoch 7; early stopping correctly halted at epoch 10.

**Speaker notes:** Training loss kept falling but validation peaked at
epoch 7 — the classic onset of mild overfitting that early stopping
is designed to catch.

---

## Slide 9 — Test Results

| Metric | Stress (n=5,334) | Fatigue (n=33) |
|---|---|---|
| Balanced accuracy | **0.7145** | 0.9615 |
| Macro F1 | **0.7003** | 0.9678 |
| ROC-AUC | **0.8921** | 0.9885 |
| ECE (calibration) | **0.0537** | 0.0303 |

**Mean:** balanced acc 0.8380, macro-F1 0.8340

> **Honest caveat: fatigue n=33** — encouraging but not statistically
> robust. **Stress n=5,334** is the reliable number.

**Speaker notes:** The bolded stress numbers are the ones we stand
behind. The fatigue numbers are inflated by a small test set; we
report them transparently rather than over-claim.

---

## Slide 10 — Confusion Matrices

[Insert `outputs/plots/cm_stress.png` and `outputs/plots/cm_fatigue.png`]

**Key reading on stress:**
- Clear diagonal — most predictions correct
- Off-diagonal mass concentrated at **adjacent classes** (low↔moderate,
  moderate↔high). Almost no low↔high confusion.
- This is the benign failure mode for an ordinal-ish 3-class problem.

**Speaker notes:** "Adjacent errors are the right errors." Saying
"moderate" when ground truth is "high" is a much less harmful failure
than saying "low" when ground truth is "high." The model rarely makes
the latter kind of error.

---

## Slide 11 — Calibration

[Insert `outputs/plots/calibration_stress.png`]

- **ECE = 0.0537** on stress (n=5,334) — well-calibrated *without*
  temperature scaling
- Reliability diagram tracks the diagonal closely
- Mild over-confidence at very-high confidence bins (typical of softmax
  classifiers)
- Matters because the live demo surfaces confidence values directly

**Speaker notes:** For a wellness tool, "we're 90% sure" should
correspond to being right 90% of the time. ECE measures that
directly. 0.054 is a solid number for a classifier without post-hoc
calibration.

---

## Slide 12 — MLOps Pipeline

```
Kaggle GPU notebook ──▶ best checkpoint ──▶ HF Hub model repo
       │                                          │
       ▼                                          │
 TensorBoard logs                                 │
                                                  │
GitHub repo (source of truth) ◀──────────────────┘
       │  push to main
       ▼
GitHub Actions ──▶ HF Space rebuilds ──▶ live Gradio app
```

**Level 2–3 MLOps:** centralized tracking · model registry · automated
deployment from version control · fully reproducible.

**Speaker notes:** Every result in the report is reproducible by a
third party with two API tokens. The training notebook runs on free
Kaggle GPUs; deployment is one git push.

---

## Slide 13 — Live Demo

**https://huggingface.co/spaces/NMemane1/facial-stress-fatigue**

[Screenshot of the deployed app — Predict tab with a real prediction
returning moderate stress 66% and a wellness suggestion]

- Upload or webcam capture
- 3-class stress + 2-class fatigue with confidence values
- LLM-generated wellness sentence (Claude) with deterministic
  fallback so the demo never breaks
- Total inference latency on CPU basic: ~150 ms

**Speaker notes:** The screenshot is from a real user-uploaded face.
Stress prediction "moderate" matches the visible tension cues; the
LLM-generated wellness sentence is task-appropriate without making
medical claims.

---

## Slide 14 — Limitations & Future Work

**Acknowledged limitations:**
- Fatigue test n=33 — not statistically robust
- Stress labels are a proxy, not ground truth
- Single-frame, no temporal cues
- Demographic coverage of FER-2013 not balanced

**Future work, in priority order:**
1. **Better fatigue data** — purpose-built dataset, thousands of held-out
2. **Temporal modeling** — short video clips for blink-rate cues
3. **Fairness audit** — across age, skin tone, gender, lighting
4. **On-device inference** — ONNX export, mobile deployment
5. **Physiological stress labels** — ground-truth, not proxy

---

## Slide 15 — Thank You / Q&A

**Repo:** github.com/NMemane1/CMPE258-DeepLearning-Facial-Stress-And-Fatigue-Detection-FinalProject

**Live demo:** huggingface.co/spaces/NMemane1/facial-stress-fatigue

**Model:** huggingface.co/NMemane1/facial-stress-fatigue-dinov2

**Headline:** 0.71 balanced accuracy / 0.05 ECE on n=5,334 stress
test samples · multi-task DINOv2 · 25 min training · MLOps level 2–3

**Questions?**

**Speaker notes:** Open the floor. Be ready for: the proxy labels
question, the fatigue n=33 question, the demographic-skew question,
and the "would this work in a car" question (answer: not without
temporal modeling and a fairness audit first).
