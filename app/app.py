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
import logging
import traceback

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoImageProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("app")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HF_MODEL_REPO = "NMemane1/facial-stress-fatigue-dinov2"
BACKBONE_NAME = "facebook/dinov2-small"
STRESS_CLASSES = ["low", "moderate", "high"]
FATIGUE_CLASSES = ["alert", "fatigued"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Cap input resolution before the processor — phone photos are often 4000×3000
# and can OOM the worker on the CPU-tier Space.
MAX_IMAGE_EDGE = 1024


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
        cls = outputs.last_hidden_state[:, 0, :]
        h = self.shared(cls)
        return self.stress_head(h), self.fatigue_head(h)


# -----------------------------------------------------------------------------
# Startup — every step wrapped so a failure here can't kill Gradio's API routes.
# -----------------------------------------------------------------------------
def _startup():
    processor = None
    model = None
    weights_ok = False
    init_error = ""

    try:
        log.info("Loading image processor for %s ...", BACKBONE_NAME)
        processor = AutoImageProcessor.from_pretrained(BACKBONE_NAME)
    except Exception as e:
        log.exception("processor load failed")
        init_error = f"processor: {e!r}"
        return processor, model, weights_ok, init_error

    try:
        log.info("Building model...")
        model = StressFatigueModel()
    except Exception as e:
        log.exception("model build failed")
        init_error = f"backbone: {e!r}"
        return processor, model, weights_ok, init_error

    try:
        log.info("Downloading trained weights from %s ...", HF_MODEL_REPO)
        weights_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="pytorch_model.bin")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Strip common DDP / torch.compile prefixes — TPU/multi-GPU runs save
        # with these and they make every key "unexpected" otherwise.
        cleaned = {}
        for k, v in state.items():
            nk = k
            for prefix in ("module.", "_orig_mod."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            cleaned[nk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        log.info(
            "Loaded weights. missing=%d unexpected=%d",
            len(missing), len(unexpected),
        )
        if missing:
            log.info("first missing keys: %s", missing[:5])
        if unexpected:
            log.info("first unexpected keys: %s", unexpected[:5])
        weights_ok = True
    except Exception as e:
        log.exception("weights load failed")
        # Not fatal — we keep the untrained backbone so the UI still responds.

    try:
        model.to(DEVICE).eval()
    except Exception as e:
        log.exception("model.to(device) failed")
        init_error = f"to_device: {e!r}"
        model = None

    return processor, model, weights_ok, init_error


PROCESSOR, MODEL, WEIGHTS_OK, INIT_ERROR = _startup()


# -----------------------------------------------------------------------------
# LLM wellness layer (optional - falls back to canned text)
# -----------------------------------------------------------------------------
_CANNED = {
    ("low", "alert"): "You look calm and alert. Keep up whatever you're doing — "
                      "regular short breaks and steady hydration help maintain this.",
    ("moderate", "alert"): "Mild stress detected. A two-minute breathing break or a "
                           "short walk can help reset before it builds up.",
    ("high", "alert"): "Elevated stress signals. Consider stepping away for a few "
                       "minutes, stretching, and prioritising one task at a time.",
    ("low", "fatigued"): "You seem relaxed but tired. Hydration, natural light, and "
                         "a short rest can help restore energy.",
    ("moderate", "fatigued"): "Both mild stress and fatigue are showing. A proper "
                              "break — ideally away from screens — is a good idea.",
    ("high", "fatigued"): "High stress combined with fatigue. This is a good moment "
                          "to pause, rest, and return to demanding work later.",
}


def wellness_recommendation(stress_label, fatigue_label):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            import anthropic  # imported lazily so a bad install can't kill startup
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
        except Exception as e:
            log.warning("LLM call failed, using canned fallback: %s", e)

    return _CANNED.get(
        (stress_label, fatigue_label),
        "Take a moment for yourself — short breaks, hydration, and rest go a long way.",
    )


# -----------------------------------------------------------------------------
# Image normalization — Gradio 5.x can hand us a few different shapes
# depending on browser, source, and component config. Normalize them all here.
# -----------------------------------------------------------------------------
def _to_pil(image):
    if image is None:
        return None
    # editor/sketchpad components return a dict; handle gracefully even though
    # we only configure upload/webcam.
    if isinstance(image, dict):
        for key in ("composite", "image", "background"):
            if image.get(key) is not None:
                image = image[key]
                break
        else:
            return None
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        # Last-ditch attempt
        image = Image.fromarray(np.asarray(image))
    # Phone photos store rotation in EXIF — apply it before resizing.
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_EDGE:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS)
    return image


# -----------------------------------------------------------------------------
# Inference — every path returns three values; we never let an exception
# bubble out of this function, because that's what kills the Gradio worker
# and produces the dreaded "No API found" error in the UI.
# -----------------------------------------------------------------------------
def _placeholder_labels(reason="unavailable"):
    """Non-empty label dicts so gr.Label renders cleanly instead of 'Error'."""
    stress = {c: 0.0 for c in STRESS_CLASSES}
    fatigue = {c: 0.0 for c in FATIGUE_CLASSES}
    stress[reason] = 1.0 if reason in stress else 0.0
    fatigue[reason] = 1.0 if reason in fatigue else 0.0
    # If reason isn't a known class, mark every class equally so the chart
    # still draws something readable.
    if sum(stress.values()) == 0:
        stress = {c: 1.0 / len(STRESS_CLASSES) for c in STRESS_CLASSES}
    if sum(fatigue.values()) == 0:
        fatigue = {c: 1.0 / len(FATIGUE_CLASSES) for c in FATIGUE_CLASSES}
    return stress, fatigue


@torch.no_grad()
def predict(image):
    try:
        if MODEL is None or PROCESSOR is None:
            err = INIT_ERROR or "model not initialized"
            s, f = _placeholder_labels()
            return s, f, f"Service unavailable at startup: {err}"

        pil = _to_pil(image)
        if pil is None:
            s, f = _placeholder_labels()
            return s, f, "Please upload or capture a face image first."

        inputs = PROCESSOR(images=pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)

        stress_logits, fatigue_logits = MODEL(pixel_values)

        stress_probs = torch.softmax(stress_logits, dim=-1)[0].detach().cpu().numpy()
        fatigue_probs = torch.softmax(fatigue_logits, dim=-1)[0].detach().cpu().numpy()

        stress_label = STRESS_CLASSES[int(np.argmax(stress_probs))]
        fatigue_label = FATIGUE_CLASSES[int(np.argmax(fatigue_probs))]

        stress_out = {c: float(p) for c, p in zip(STRESS_CLASSES, stress_probs)}
        fatigue_out = {c: float(p) for c, p in zip(FATIGUE_CLASSES, fatigue_probs)}

        note = wellness_recommendation(stress_label, fatigue_label)
        if not WEIGHTS_OK:
            note = "[Demo mode: trained weights unavailable] " + note
        return stress_out, fatigue_out, note
    except Exception as e:
        log.error("predict() failed:\n%s", traceback.format_exc())
        s, f = _placeholder_labels()
        return s, f, f"Inference error: {type(e).__name__}: {e}"


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
METHODOLOGY_MD = """
## How this model works

**Architecture.** A `facebook/dinov2-small` vision transformer backbone (a
self-supervised foundation model) produces a 384-dimensional embedding of the
face image. A shared trunk (`Linear → GELU → Dropout`) feeds two
task-specific linear heads — one for **stress** (3 classes) and one for
**fatigue** (2 classes). This is a *multi-task* design: one backbone, two
predictions, trained jointly.

**Why DINOv2.** Self-supervised ViT features transfer well to face analysis
without needing a huge labelled dataset. The lower 9 transformer layers are
frozen; only the top layers + heads are fine-tuned, which keeps training fast
and reduces overfitting on a modest dataset.

**Why these design choices.**
- *Class-balanced focal loss* — the stress classes are imbalanced, so focal
  loss down-weights easy examples and class weights correct for frequency.
- *RandAugment + horizontal flip* — augmentation improves robustness to
  lighting and pose.
- *Cosine LR schedule with warmup* — stable fine-tuning of a pretrained ViT.
- *Masked multi-task loss* — each sample only contributes to the task it has a
  label for, so the two datasets can be combined cleanly.

**Training.** 11 epochs on a Tesla T4 GPU, mixed-precision (fp16), early
stopping on mean balanced accuracy. Tracked with TensorBoard.
"""

ABOUT_MD = """
## About

**Facial Stress & Fatigue Detection** — CMPE 258 Deep Learning Final Project,
San Jose State University.

This Space demonstrates inference for a multi-task deep learning model that
estimates stress and fatigue from a single face image, then uses a language
model layer to turn the raw scores into a short, supportive wellness note.

- **Model repo:** [`NMemane1/facial-stress-fatigue-dinov2`](https://huggingface.co/NMemane1/facial-stress-fatigue-dinov2)
- **Code:** see the linked GitHub repository in the project README
- **Datasets:** FER-2013 (emotion → stress mapping) and a yawn/eye drowsiness
  dataset (→ fatigue)

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
    demo.queue(default_concurrency_limit=1).launch()
