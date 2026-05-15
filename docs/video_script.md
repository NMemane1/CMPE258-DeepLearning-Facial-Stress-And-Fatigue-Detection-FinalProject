# Video Presentation Script

**Presenters:** Nikita Memane · Sankalp Wahane
**Target length:** ~10 minutes (long presentation) + a separate ~2-minute demo cut
**Format:** Screen recording with both presenters narrating in turn

> **How to use this script.** Lines in **bold caps** are speaker labels.
> Lines in `[brackets]` are **stage directions** — what to share on
> screen, what to click, when to switch windows. Italicized lines are
> the words to say. Time stamps are rough targets, not hard rules.
> Practice once aloud; tweak any sentence so it sounds like you.

---

## Before you start recording — setup checklist

Open these tabs/windows in this order, hidden behind your desktop:

1. **OBS or QuickTime** for screen recording (full-screen capture at
   1080p, 30 fps). Test audio levels first.
2. **Tab A** — the GitHub repo:
   `https://github.com/NMemane1/CMPE258-DeepLearning-Facial-Stress-And-Fatigue-Detection-FinalProject`
3. **Tab B** — the rendered `README.md` on GitHub (scrolled to top).
4. **Tab C** — `docs/report.md` rendered on GitHub.
5. **Tab D** — `docs/ablations.md` rendered on GitHub.
6. **Tab E** — the live Hugging Face Space:
   `https://huggingface.co/spaces/NMemane1/facial-stress-fatigue`
7. **Tab F** — the HF model repo:
   `https://huggingface.co/NMemane1/facial-stress-fatigue-dinov2`
8. **Finder window** open at `~/Downloads/outputs/plots/` so you can
   quickly drag `cm_stress.png` and `calibration_stress.png` into a
   preview window when needed.
9. **A real test face image** ready on your Desktop (the same
   stressed-woman `.webp` you used during development — that
   prediction worked: moderate stress 66%, low 32%, high 1%).
10. **Camera-on overlay** (small circle in the corner) if your tool
    supports it, so the viewer sees you while you narrate.

Close Slack, Mail, and any notifications. Put your phone on silent.

---

## 0:00 – 0:30 · Title & introductions

**[Share Tab B — the README rendered on GitHub. Scroll so the title
and the Team table are both visible.]**

**NIKITA:** *Hi, I'm Nikita Memane.*

**SANKALP:** *And I'm Sankalp Wahane. This is our CMPE 258 final
project: Facial Stress and Fatigue Detection — a multi-task deep
learning system that takes one face image and returns a stress
level, a fatigue state, and a short wellness suggestion.*

**NIKITA:** *Everything is live: the trained model is on the Hugging
Face Hub, the demo is deployed on Hugging Face Spaces, and the whole
repo is on GitHub. We'll walk through the problem, the model, the
training, the results, and then end with a live demo.*

---

## 0:30 – 1:30 · Problem & motivation

**[Stay on Tab B — README. Scroll slowly down to the "Problem &
Motivation" section.]**

**SANKALP:** *Stress and fatigue affect health, safety, and
productivity, but they're hard to measure passively. Self-report
scales like PSS and KSS are noisy and intrusive. Wearables work but
require hardware most people won't carry around for an ambient
wellness signal.*

**SANKALP:** *A camera-based estimator is the missing third
modality — it's passive, it's sensorless, and it can be embedded in
anything that already has webcam access. Think of study-break
reminders, driver-monitoring assists, or workplace wellness
dashboards. We are explicitly **not** trying to build a medical
device.*

**NIKITA:** *Our research question was focused: can one
self-supervised vision backbone, fine-tuned with two lightweight
heads, jointly estimate stress and fatigue well enough to be useful?
And just as importantly — can we be honest about its calibration,
not just its accuracy.*

---

## 1:30 – 2:15 · Why this is hard (intellectual honesty up front)

**[Scroll to the "Data" section's honest-limitation callout, then
to "Limitations" near the bottom of the README. Highlight or zoom in
on the n=33 caveat box.]**

**SANKALP:** *We want to flag the limitations up front rather than
hiding them in the back of the report. Three things to know.*

**SANKALP:** *First — our stress labels are a **proxy**. We re-map
FER-2013's seven emotion classes onto three stress levels with a
documented heuristic. It's defensible but it's not ground-truth
stress measurement.*

**SANKALP:** *Second — the fatigue test set is tiny. Only thirty-three
samples. The fatigue metrics we'll show are directionally
encouraging but not statistically robust. Stress, with five-thousand
three-hundred-thirty-four test samples, is the reliable number.*

**SANKALP:** *Third — it's single-frame. No temporal cues. Real-world
fatigue detection wants video.*

**NIKITA:** *We mention these caveats in the abstract, the data
section, the experiments, and on this slide on purpose. Honest
limitations make a more useful model.*

---

## 2:15 – 3:30 · Data & approach

**[Switch to Tab C — docs/report.md on GitHub. Scroll to Section 3
"Data".]**

**NIKITA:** *We combine two public Kaggle datasets. FER-2013 gives us
thirty-five-thousand-plus face images with seven emotion labels —
we re-map them to three stress levels: happy, surprise, and neutral
become low; sad becomes moderate; angry, fear, and disgust become
high. The yawn-eye drowsiness dataset gives us a small set of
images labelled alert versus fatigued.*

**NIKITA:** *Each image carries a label for only one task. We use a
**masked multi-task loss** so a FER image only contributes to the
stress head, and a yawn-eye image only contributes to the fatigue
head. This lets us combine two single-task datasets without any
re-labelling.*

**NIKITA:** *We use a subject-grouped seventy-fifteen-fifteen split
where subject metadata is available — that prevents the same
person's photos from appearing in both train and test, which would
let the model recognize the identity instead of the affect.*

**[Scroll to the data table; pause on it for two seconds.]**

**NIKITA:** *Final split sizes: twenty-five-thousand-three-hundred
train, five-thousand-six-hundred-fifty-three val, and
five-thousand-three-hundred-sixty-seven test.*

---

## 3:30 – 4:30 · Architecture

**[Scroll down in docs/report.md to Section 4 "Methods" — the ASCII
architecture diagram. Pause on the diagram for the explanation.]**

**NIKITA:** *Here's the architecture. The input is a two-hundred-
twenty-four by two-hundred-twenty-four RGB image. The backbone is
DINOv2-small — Meta's self-supervised vision transformer, pretrained
on a hundred-and-forty-two million unlabelled images. We freeze
patch embeddings and the lower nine transformer layers, and we
fine-tune only the top three.*

**NIKITA:** *The CLS token comes out as a three-hundred-eighty-four-
dimensional embedding. We feed that into a small shared trunk — two
Linear-LayerNorm-GELU-Dropout blocks — and then two task heads:
three-class stress and binary fatigue.*

**[Scroll to the parameter-count table or the "Why these design
choices" table.]**

**NIKITA:** *Total: twenty-two-point-two million parameters,
five-point-four-nine million trainable — about twenty-five percent.
With only twenty-five-thousand training samples on stress, a full
fine-tune would overfit hard. Freezing the lower layers preserves
DINOv2's generic features and acts as a strong implicit regularizer.*

**NIKITA:** *The multi-task structure also matters: the larger
stress task regularizes the tiny fatigue task through the shared
trunk. Both heads benefit from features shaped by both signals.*

---

## 4:30 – 5:30 · Training setup and curve

**[Switch to Tab D — docs/ablations.md or back to docs/report.md.
Scroll to the training-curve table in the experiments section.]**

**NIKITA:** *Training was twenty-five minutes on one Kaggle Tesla T4.
AdamW optimizer with discriminative learning rates — three-e-minus-
five for the backbone, three-e-minus-four for the heads. Cosine
schedule with five-hundred steps of warmup. Mixed precision fp16.
Class-balanced cross-entropy plus a focal-loss term with gamma
two-point-zero, to push the model toward the hard moderate-stress
boundary cases.*

**[Highlight the epoch-by-epoch table.]**

**NIKITA:** *The training curve. Training loss falls monotonically
across eleven epochs. Validation balanced accuracy climbs to
zero-point-six-three-one-one at epoch seven and then plateaus.
Early stopping, patience three, correctly halts at epoch ten and
keeps the epoch-seven checkpoint as the best model. This is the
textbook signature of mild overfitting onset — exactly what early
stopping is designed to catch.*

---

## 5:30 – 6:30 · Test results & calibration

**[Scroll to the test-results table (Section 5.2 of report.md). Then
open Finder, drag `outputs/plots/cm_stress.png` into Preview, and
share that window. Then `calibration_stress.png`.]**

**NIKITA:** *Final test metrics. Stress task, on five-thousand-
three-hundred-thirty-four test samples: balanced accuracy
zero-point-seven-one-four-five, macro-F1 zero-point-seven-zero,
ROC-AUC zero-point-eight-nine-two-one, and Expected Calibration
Error zero-point-zero-five-three-seven.*

**[Show cm_stress.png in Preview.]**

**NIKITA:** *The confusion matrix on stress. Clear diagonal — most
predictions correct. The interesting thing is **where the errors
are**: almost all the off-diagonal mass is between **adjacent**
classes. Low confused with moderate, moderate confused with high.
The model almost never says "low" when ground truth is "high",
which is the failure mode that would matter most for a wellness
tool.*

**[Switch to calibration_stress.png in Preview.]**

**NIKITA:** *Calibration. Expected Calibration Error of zero-point-
zero-five-three-seven means the model's confidence matches its
observed accuracy to within about five percentage points across the
confidence range. The reliability diagram tracks the diagonal
closely. This matters because the live demo surfaces confidence
values directly to the user — and we did this **without** any
post-hoc temperature scaling, which is a strong baseline.*

**NIKITA:** *On fatigue, the numbers look very high — balanced
accuracy zero-point-nine-six — but again, n equals thirty-three.
We report it but we don't lean on it.*

---

## 6:30 – 8:00 · Live demo

**[Switch to Tab E — the live Hugging Face Space at
huggingface.co/spaces/NMemane1/facial-stress-fatigue. The page
should already show the Predict tab. If the Space is asleep, click
on it to wake it up before you start recording this section.]**

**SANKALP:** *Now Nikita is going to walk us through the live
deployed app. This is running on a free Hugging Face Space tier —
two virtual CPUs, sixteen gigs of RAM, no GPU. The model lives on
the Hugging Face Hub at NMemane1 slash facial-stress-fatigue-dinov2,
and the Space pulls it on startup.*

**[Click the Methodology tab briefly.]**

**NIKITA:** *Inside the app there are three tabs. The Methodology
tab — this is the rubric-required "why we chose what we chose"
section, embedded right in the demo so a user can read the design
rationale alongside the predictions. The About tab links to the
model repo and the GitHub repo.*

**[Click back to the Predict tab.]**

**NIKITA:** *And here's the Predict tab. We can upload a face image
or use the webcam.*

**[Drag the stressed-woman .webp from Desktop into the Face image
upload area. Wait for it to load. Click the "Analyze" button.]**

**NIKITA:** *Let me upload a real test image — this is a real
photograph of a visibly tense, tired subject.*

**[Wait ~1 second for the prediction. Point at the stress output.]**

**NIKITA:** *Here's the result. Stress: moderate at sixty-six
percent, low at thirty-two percent, high at one percent. That's a
plausible reading — the subject is showing tension cues but not the
high-arousal expressions like fear or anger that map to the "high"
class. Fatigue: alert at a hundred percent — which is honestly the
fatigue head's limitation showing. With only four-hundred fatigue
training images, the head doesn't generalize well outside the
yawn-eye dataset's specific cues. We discuss this in the report's
limitations section.*

**[Point at the wellness suggestion box.]**

**NIKITA:** *And the wellness suggestion — this is generated by
Claude through a prompt-engineered template that the rubric asks
for. There's a deterministic fallback so the app works even if
the Anthropic API is unavailable.*

**SANKALP:** *Behind the scenes, every push to the main GitHub
branch triggers a GitHub Actions workflow that copies the app code
and the source tree to the Hugging Face Space, which auto-rebuilds.
The whole deploy from "git push" to "live updated demo" takes about
two minutes with no manual steps.*

---

## 8:00 – 8:45 · Honest limitations

**[Switch back to Tab C — docs/report.md, scroll to Section 6
"Limitations and Failure Modes".]**

**SANKALP:** *Let's be honest about what doesn't work.*

**SANKALP:** *One — the fatigue dataset is tiny. Thirty-three test
samples is not a reliable evaluation. Building a proper fatigue
dataset with thousands of held-out examples is the single most
important next step.*

**SANKALP:** *Two — stress labels are a proxy from facial emotion,
not a physiological measurement. A future version should use
heart-rate-variability data or a validated stress questionnaire
paired with face images.*

**SANKALP:** *Three — no temporal modeling. Blink rate, eye-closure
duration, micro-expression dynamics — those carry strong fatigue
signal and require short video clips.*

**SANKALP:** *Four — FER-2013 has known demographic skew. Any
non-research deployment needs a formal fairness audit across age,
skin tone, gender, and lighting conditions first.*

---

## 8:45 – 9:30 · Future work & MLOps takeaway

**[Switch to Tab B — README, scroll to "MLOps Pipeline".]**

**NIKITA:** *Quickly on engineering takeaways. The whole pipeline is
reproducible: Kaggle notebook trains, pushes checkpoint to the
Hugging Face Hub, GitHub Actions deploys the Space on every push to
main. This is roughly MLOps maturity level two-to-three —
centralized experiment tracking with TensorBoard, model versioned
in a registry, automated deployment from version control.*

**SANKALP:** *Future work, in priority order. One: a proper fatigue
dataset. Two: temporal modeling on short video clips. Three: a
fairness audit. Four: on-device inference with ONNX export so the
face never leaves the user's phone. Five: physiologically-grounded
stress labels instead of the emotion proxy.*

---

## 9:30 – 10:00 · Closing

**[Switch back to Tab B — top of the README, so the Links table is
visible.]**

**NIKITA:** *The repo, the deployed demo, and the model are all
linked at the top of the README. The full report, the design-choice
ablation analysis, the slide deck, the project proposal, and a
file-by-file code walkthrough are all under docs slash.*

**SANKALP:** *Headline result we stand behind: zero-point-seven-one
balanced accuracy and zero-point-zero-five calibration error on
five-thousand-three-hundred-thirty-four stress test samples, with
a twenty-five-minute training run and only five-and-a-half million
trainable parameters. Plus a live demo a user can try right now.*

**NIKITA:** *Thank you for watching. Happy to answer questions.*

**[Stop recording.]**

---

# Short demo-only cut (~2 minutes)

For the separate "Demo video" link in the README. This is just the
live-Space walkthrough with minimal framing.

## 0:00 – 0:20 · Intro

**[Share Tab E — the live HF Space, on the Predict tab.]**

**NIKITA:** *Quick demo of our CMPE 258 final project — Facial
Stress and Fatigue Detection. This is the live deployed app on
Hugging Face Spaces. Upload a face image, get back a three-class
stress reading, a two-class fatigue reading, and an LLM-generated
wellness suggestion.*

## 0:20 – 0:50 · The three tabs

**[Click Methodology tab.]**

**NIKITA:** *The Methodology tab explains every design decision —
which loss, which activation, which augmentation, and why.*

**[Click About tab.]**

**NIKITA:** *The About tab links to the model and the GitHub repo.*

**[Click back to Predict.]**

## 0:50 – 1:40 · The prediction

**[Upload the stressed-woman .webp. Click Analyze.]**

**NIKITA:** *Uploading a real test face. The model returns:
moderate stress at sixty-six percent, alert on fatigue, and a
wellness suggestion. The whole inference takes about a hundred and
fifty milliseconds on CPU.*

**[Pause on the result so the viewer can read the wellness sentence.]**

**NIKITA:** *Behind this single button click: a DINOv2-small vision
transformer with the lower nine layers frozen, a shared MLP trunk,
two classification heads, and a Claude API call for the wellness
sentence — with a deterministic fallback if the API is unavailable.*

## 1:40 – 2:00 · Wrap

**[Switch to Tab B — the README.]**

**NIKITA:** *Full repo at the link in the description. The trained
model is public on the Hugging Face Hub. Thanks for watching.*

**[Stop recording.]**

---

# Tips while recording

- **Speak slower than you think you need to.** Reading a script on
  camera always sounds rushed in playback. Aim for about a hundred
  and forty words per minute.
- **Pause for two seconds** after each major number or claim. It
  gives the viewer time to read the screen.
- **Don't apologize on camera.** If you mis-speak, just keep going
  and re-do that take in editing. You can splice.
- **Mouse cursor** — move it deliberately to point at things on
  screen as you talk about them. Don't just leave it parked in a
  corner.
- **Don't read every word from this script.** Skim it once aloud,
  then record looking at the screen content with the script in your
  peripheral vision. The phrasing should sound like you, not like
  me.
- **Time check** — if the long version runs over twelve minutes in
  the first take, cut the MLOps Takeaway block (8:45-9:30) — it's
  the most prunable section.

Good luck.
