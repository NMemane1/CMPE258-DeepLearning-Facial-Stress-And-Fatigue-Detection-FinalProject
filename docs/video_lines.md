# Video Lines — What You Actually Say

> Pure spoken lines for the long presentation video and the short
> demo video. No stage directions, no timestamps — just the words.
>
> **N =** Nikita Memane · **S =** Sankalp Wahane
>
> Read this as a teleprompter. Pause at every paragraph break. The
> full script with screen-by-screen directions is in
> [`video_script.md`](video_script.md).

---

# LONG PRESENTATION VIDEO (~10 minutes)

---

### Intro

**N:** Hi, I'm Nikita Memane.

**S:** And I'm Sankalp Wahane. This is our CMPE 258 final project — Facial Stress and Fatigue Detection. A multi-task deep learning system that takes one face image and returns a stress level, a fatigue state, and a short wellness suggestion.

**N:** Everything is live. The trained model is on the Hugging Face Hub. The demo is deployed on Hugging Face Spaces. The whole repo is on GitHub. We'll walk through the problem, the model, the training, the results — and then end with a live demo.

---

### Problem and motivation

**S:** Stress and fatigue affect health, safety, and productivity, but they're hard to measure passively. Self-report scales like PSS and KSS are noisy and intrusive. Wearables work, but require hardware most people won't carry around for an ambient wellness signal.

**S:** A camera-based estimator is the missing third modality. It's passive. It's sensorless. It can be embedded in anything that already has webcam access. Think study-break reminders, driver-monitoring assists, or workplace wellness dashboards. We are explicitly *not* trying to build a medical device.

**N:** Our research question was focused — can one self-supervised vision backbone, fine-tuned with two lightweight heads, jointly estimate stress and fatigue well enough to be useful? And just as importantly — can we be honest about its calibration, not just its accuracy.

---

### Honest limitations up front

**S:** We want to flag the limitations up front rather than burying them at the end. Three things to know.

**S:** First — our stress labels are a proxy. We re-map FER-2013's seven emotion classes onto three stress levels using a documented heuristic. It's defensible, but it's not ground-truth stress measurement.

**S:** Second — the fatigue test set is tiny. Only thirty-three samples. The fatigue numbers we'll show are directionally encouraging, but not statistically robust. Stress, with five thousand three hundred thirty-four test samples, is the reliable number.

**S:** Third — it's single-frame. No temporal information. Real-world fatigue detection wants short video clips.

**N:** We mention these caveats in the abstract, the data section, and the experiments — on purpose. Honest limitations make a more useful model.

---

### Data

**N:** We combine two public Kaggle datasets. FER-2013 gives us over thirty-five thousand face images with seven emotion labels. We re-map those — happy, surprise, and neutral become low stress; sad becomes moderate; angry, fear, and disgust become high. The yawn-eye drowsiness dataset gives us a small set labelled alert versus fatigued.

**N:** Each image carries a label for only one task. We use a *masked multi-task loss* — so a FER image only updates the stress head, and a yawn-eye image only updates the fatigue head. This lets us combine two single-task datasets without any re-labelling.

**N:** We use a subject-grouped seventy / fifteen / fifteen split where subject metadata is available. That prevents the same person's photos from appearing in both train and test — which would let the model recognize the identity instead of the affect.

**N:** Final split sizes — twenty-five thousand three hundred train, five thousand six hundred fifty-three validation, five thousand three hundred sixty-seven test.

---

### Architecture

**N:** Here's the architecture. The input is a two-twenty-four by two-twenty-four RGB image. The backbone is DINOv2-small — Meta's self-supervised vision transformer, pretrained on a hundred forty-two million unlabelled images. We freeze the patch embeddings and the lower nine transformer layers, and fine-tune only the top three.

**N:** The CLS token comes out as a three-hundred-eighty-four dimensional embedding. We feed that into a small shared trunk — two Linear–LayerNorm–GELU–Dropout blocks — and then two task heads. Three-class stress, two-class fatigue.

**N:** Twenty-two-point-two million parameters total. Five-point-four-nine million trainable — about twenty-five percent. With only twenty-five thousand stress training samples, a full fine-tune would overfit hard. Freezing the lower layers preserves DINOv2's generic features and acts as a strong implicit regularizer.

**N:** The multi-task structure also matters. The larger stress task regularizes the tiny fatigue task through the shared trunk. Both heads benefit from features shaped by both signals.

---

### Training setup and curve

**N:** Training was twenty-five minutes on one Kaggle Tesla T4. AdamW optimizer, with discriminative learning rates — three-e-minus-five for the backbone, three-e-minus-four for the heads. Cosine schedule with five hundred steps of warmup. Mixed-precision fp16. Class-balanced cross-entropy plus a focal-loss term with gamma equals two, to push the model toward the hard moderate-stress boundary cases.

**N:** The training curve. Training loss falls monotonically across eleven epochs. Validation balanced accuracy climbs to zero-point-six-three-one-one at epoch seven, and then plateaus. Early stopping — patience three — correctly halts at epoch ten and keeps the epoch-seven checkpoint as the best model. This is the textbook signature of mild overfitting onset, which is exactly what early stopping is designed to catch.

---

### Test results

**N:** Final test metrics. Stress, on five thousand three hundred thirty-four test samples — balanced accuracy zero-point-seven-one-four-five. Macro-F1 zero-point-seven. ROC-AUC zero-point-eight-nine-two-one. Expected Calibration Error zero-point-zero-five-three-seven.

**N:** The confusion matrix on stress. Clear diagonal — most predictions are correct. The interesting thing is *where* the errors are. Almost all the off-diagonal mass is between adjacent classes. Low confused with moderate. Moderate confused with high. The model almost never says "low" when ground truth is "high." That would be the failure mode that matters most for a wellness tool — and it doesn't happen.

**N:** Calibration. Expected Calibration Error of zero-point-zero-five means the model's confidence matches its observed accuracy to within about five percentage points across the confidence range. The reliability diagram tracks the diagonal closely. This matters because the live demo shows the confidence values directly to the user — and we got this *without* any post-hoc temperature scaling. Strong baseline.

**N:** On fatigue, the numbers look very high — balanced accuracy zero-point-nine-six. But again — n equals thirty-three. We report it, but we don't lean on it.

---

### Live demo

**S:** Now Nikita's going to walk us through the live deployed app. This is running on a free Hugging Face Space tier — two virtual CPUs, sixteen gigs of RAM, no GPU. The model lives on the Hugging Face Hub at NMemane1 slash facial-stress-fatigue-dinov2. The Space pulls it on startup.

**N:** Inside the app there are three tabs. The Methodology tab is the rubric-required "why we chose what we chose" section, embedded right in the demo so a user can read the design rationale alongside the predictions. The About tab links to the model repo and the GitHub repo.

**N:** And here's the Predict tab. We can upload a face image, or use the webcam.

**N:** Let me upload a real test image. This is a real photograph of a visibly tense, tired subject.

**N:** Here's the result. Stress — moderate at sixty-six percent. Low at thirty-two. High at one. That's a plausible reading. The subject is showing tension cues but not the high-arousal expressions like fear or anger that map to the "high" class. Fatigue — alert at a hundred percent. Which is honestly the fatigue head's limitation showing. With only four hundred fatigue training images, the head doesn't generalize well outside the yawn-eye dataset's specific cues. We discuss this in the report.

**N:** And the wellness suggestion — this is generated by Claude through a prompt-engineered template that the rubric asks for. There's a deterministic fallback so the app works even if the Anthropic API is unavailable.

**S:** Behind the scenes — every push to the main GitHub branch triggers a GitHub Actions workflow that copies the app code and the source tree to the Hugging Face Space, which auto-rebuilds. The whole deploy from "git push" to "live updated demo" takes about two minutes, with no manual steps.

---

### Limitations

**S:** Let's be honest about what doesn't work.

**S:** One — the fatigue dataset is tiny. Thirty-three test samples is not a reliable evaluation. Building a proper fatigue dataset with thousands of held-out examples is the single most important next step.

**S:** Two — stress labels are a proxy from facial emotion, not a physiological measurement. A future version should use heart-rate-variability data, or a validated stress questionnaire paired with face images.

**S:** Three — no temporal modeling. Blink rate, eye-closure duration, micro-expression dynamics — those carry strong fatigue signal and they require short video clips.

**S:** Four — FER-2013 has known demographic skew. Any non-research deployment needs a formal fairness audit across age, skin tone, gender, and lighting first.

---

### MLOps takeaway and future work

**N:** Quickly on engineering. The whole pipeline is reproducible. Kaggle notebook trains the model, pushes the checkpoint to the Hugging Face Hub. GitHub Actions deploys the Space on every push to main. This is roughly MLOps maturity level two-to-three — centralized experiment tracking with TensorBoard, model versioned in a registry, automated deployment from version control.

**S:** Future work in priority order. One — a proper fatigue dataset. Two — temporal modeling on short video clips. Three — a fairness audit. Four — on-device inference with ONNX export, so the face never leaves the user's phone. Five — physiologically-grounded stress labels instead of the emotion proxy.

---

### Closing

**N:** The repo, the deployed demo, and the model are all linked at the top of the README. The full report, the design-choice ablation analysis, the slide deck, the project proposal, and a file-by-file code walkthrough are all under docs slash.

**S:** Headline result we stand behind — zero-point-seven-one balanced accuracy and zero-point-zero-five calibration error on five thousand three hundred thirty-four stress test samples. Twenty-five-minute training run. Five-and-a-half million trainable parameters. Plus a live demo a user can try right now.

**N:** Thank you for watching. Happy to answer questions.

---

---

# SHORT DEMO VIDEO (~2 minutes)

---

### Open

**N:** Quick demo of our CMPE 258 final project — Facial Stress and Fatigue Detection. This is the live deployed app on Hugging Face Spaces. Upload a face image, get back a three-class stress reading, a two-class fatigue reading, and an LLM-generated wellness suggestion.

---

### Three tabs

**N:** The Methodology tab explains every design decision — which loss, which activation, which augmentation, and why.

**N:** The About tab links to the model and the GitHub repo.

---

### The prediction

**N:** Uploading a real test face. The model returns — moderate stress at sixty-six percent, alert on fatigue, and a wellness suggestion. The whole inference takes about a hundred and fifty milliseconds on CPU.

**N:** Behind this single button click — a DINOv2-small vision transformer with the lower nine layers frozen, a shared MLP trunk, two classification heads, and a Claude API call for the wellness sentence, with a deterministic fallback if the API is unavailable.

---

### Wrap

**N:** Full repo at the link in the description. The trained model is public on the Hugging Face Hub. Thanks for watching.
