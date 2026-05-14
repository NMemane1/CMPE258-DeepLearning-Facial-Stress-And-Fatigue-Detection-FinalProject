# =============================================================================
# Facial Stress & Fatigue Detection - Gradio Space
# CMPE 258 Deep Learning Final Project - San Jose State University
#
# Multi-task DINOv2-small model with two classification heads:
#   - Stress  : 3 classes  (low / moderate / high)
#   - Fatigue : 2 classes  (alert / fatigued)
#
# The trained checkpoint is pulled from the HF Hub model repo at startup.
# An optional LLM layer (Claude) turns the raw predictions into a short
# wellness recommendation; if no API key is configured it falls back to a
# deterministic canned recommendation so the Space still works end-to-end.
# =============================================================================

import os
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoImageProcessor

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HF_MODEL_REPO = "NMemane1/facial-stress-fatigue-dinov2"
BACKBONE_NAME = "facebook/dinov2-small"
STRESS_CLASSES = ["low", "moderate", "high"]
FATIGUE_CLASSES = ["alert", "fatigued"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------------------------------------------------------
# Model definition - mirrors src/models/stress_fatigue_model.py from training
# -----------------------------------------------------------------------------
class StressFatigueModel(nn.Module):
    """DINOv2 backbone + shared trunk + two task-specific classification heads."""

    def __init__(
        self,
        backbone_name=BACKBONE_NAME,
        embedding_dim=384,
        shared_hidden=256,
        num_stress_classes=3,
        num_fatigue_classes=2,
        dropout=0.3,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)

        self.shared = nn.Sequential(
            nn.Linear(embedding_dim, shared_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.stress_head = nn.Linear(shared_hidden, num_stress_classes)
        self.fatigue_head = nn.Linear(shared_hidden, num_fatigue_classes)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        # DINOv2 CLS token = pooled image representation
        cls = outputs.last_hidden_state[:, 0, :]
        h = self.shared(cls)
        return self.stress_head(h), self.fatigue_head(h)


# -----------------------------------------------------------------------------
# Load model + processor once at startup
# -----------------------------------------------------------------------------
print("Loading image processor...")
processor = AutoImageProcessor.from_pretrained(BACKBONE_NAME)

print("Building model...")
model = StressFatigueModel()

print(f"Downloading trained weights from {HF_MODEL_REPO} ...")
try:
    weights_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="pytorch_model.bin")
    state = torch.load(weights_path, map_location="cpu")
    # tolerate either a raw state_dict or a {"state_dict": ...} wrapper
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights. missing={len(missing)} unexpected={len(unexpected)}")
    WEIGHTS_OK = True
except Exception as e:  # noqa: BLE001
    print(f"WARNING: could not load trained weights ({e}). Using untrained backbone.")
    WEIGHTS_OK = False

model.to(DEVICE).eval()


# -----------------------------------------------------------------------------
# LLM wellness layer (optional - falls back to canned text)
# -----------------------------------------------------------------------------
def wellness_recommendation(stress_label, fatigue_label):
    """Return a short, supportive wellness note for the predicted state."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            prompt = (
                f"A facial-analysis model estimates this person's stress level as "
                f"'{stress_label}' and fatigue level as '{fatigue_label}'. "
                f"Write 2-3 short, supportive, practical wellness suggestions. "
                f"Be warm and non-alarming. Do not give medical advice."
            )
            msg = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:  # noqa: BLE001
            print(f"LLM call failed, using fallback: {e}")

    # Deterministic fallback so the Space works without an API key
    canned = {
        ("low", "alert"): "You look calm and alert. Keep up whatever you're doing - "
                          "regular short breaks and steady hydration help maintain this.",
        ("moderate", "alert"): "Mild stress detected. A two-minute breathing break or a "
                               "short walk can help reset before it builds up.",
        ("high", "alert"): "Elevated stress signals. Consider stepping away for a few "
                           "minutes, stretching, and prioritising one task at a time.",
        ("low", "fatigued"): "You seem relaxed but tired. Hydration, natural light, and "
                            "a short rest can help restore energy.",
        ("moderate", "fatigued"): "Both mild stress and fatigue are showing. A proper "
                                  "break - ideally away from screens - is a good idea.",
        ("high", "fatigued"): "High stress combined with fatigue. This is a good moment "
                             "to pause, rest, and return to demanding work later.",
    }
    return canned.get(
        (stress_label, fatigue_label),
        "Take a moment for yourself - short breaks, hydration, and rest go a long way.",
    )


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict(image):
    if image is None:
        return None, None, "Please upload or capture a face image first."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    stress_logits, fatigue_logits = model(inputs["pixel_values"])

    stress_probs = torch.softmax(stress_logits, dim=-1)[0].cpu().numpy()
    fatigue_probs = torch.softmax(fatigue_logits, dim=-1)[0].cpu().numpy()

    stress_label = STRESS_CLASSES[int(np.argmax(stress_probs))]
    fatigue_label = FATIGUE_CLASSES[int(np.argmax(fatigue_probs))]

    stress_out = {c: float(p) for c, p in zip(STRESS_CLASSES, stress_probs)}
    fatigue_out = {c: float(p) for c, p in zip(FATIGUE_CLASSES, fatigue_probs)}

    note = wellness_recommendation(stress_label, fatigue_label)
    if not WEIGHTS_OK:
        note = "[Demo mode: trained weights unavailable] " + note

    return stress_out, fatigue_out, note


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
METHODOLOGY_MD = """
## How this model works

**Architecture.** A `facebook/dinov2-small` vision transformer backbone (a
self-supervised foundation model) produces a 384-dimensional embedding of the
face image. A shared trunk (`Linear -> GELU -> Dropout`) feeds two
task-specific linear heads - one for **stress** (3 classes) and one for
**fatigue** (2 classes). This is a *multi-task* design: one backbone, two
predictions, trained jointly.

**Why DINOv2.** Self-supervised ViT features transfer well to face analysis
without needing a huge labelled dataset. The lower 9 transformer layers are
frozen; only the top layers + heads are fine-tuned, which keeps training fast
and reduces overfitting on a modest dataset.

**Why these design choices.**
- *Class-balanced focal loss* - the stress classes are imbalanced, so focal
  loss down-weights easy examples and class weights correct for frequency.
- *RandAugment + horizontal flip* - augmentation improves robustness to
  lighting and pose.
- *Cosine LR schedule with warmup* - stable fine-tuning of a pretrained ViT.
- *Masked multi-task loss* - each sample only contributes to the task it has a
  label for, so the two datasets can be combined cleanly.

**Training.** 11 epochs on a Tesla T4 GPU, mixed-precision (fp16), early
stopping on mean balanced accuracy. Tracked with TensorBoard.
"""

ABOUT_MD = """
## About

**Facial Stress & Fatigue Detection** - CMPE 258 Deep Learning Final Project,
San Jose State University.

This Space demonstrates inference for a multi-task deep learning model that
estimates stress and fatigue from a single face image, then uses a language
model layer to turn the raw scores into a short, supportive wellness note.

- **Model repo:** [`NMemane1/facial-stress-fatigue-dinov2`](https://huggingface.co/NMemane1/facial-stress-fatigue-dinov2)
- **Code:** see the linked GitHub repository in the project README
- **Datasets:** FER-2013 (emotion -> stress mapping) and a yawn/eye drowsiness
  dataset (-> fatigue)

*This is an academic project. It is not a medical device and must not be used
for diagnosis.*
"""

with gr.Blocks(title="Facial Stress & Fatigue Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Facial Stress & Fatigue Detection\n"
        "Upload or capture a face image. The model predicts **stress** (low / "
        "moderate / high) and **fatigue** (alert / fatigued), then suggests a "
        "short wellness note."
    )

    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="Face image", sources=["upload", "webcam"])
                run_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                stress_out = gr.Label(label="Stress", num_top_classes=3)
                fatigue_out = gr.Label(label="Fatigue", num_top_classes=2)
                note_out = gr.Textbox(label="Wellness suggestion", lines=4)
        run_btn.click(predict, inputs=image_in, outputs=[stress_out, fatigue_out, note_out])

    with gr.Tab("Methodology"):
        gr.Markdown(METHODOLOGY_MD)

    with gr.Tab("About"):
        gr.Markdown(ABOUT_MD)


if __name__ == "__main__":
    demo.launch()
