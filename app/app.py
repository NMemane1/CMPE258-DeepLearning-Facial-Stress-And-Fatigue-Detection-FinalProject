# =============================================================================
# Facial Stress & Fatigue Detection - Gradio Space
# CMPE 258 Deep Learning Final Project - San Jose State University
#
# Imports the actual training-time StressFatigueModel from src/models/ so
# checkpoint keys match. Wraps everything defensively so a startup failure
# can't kill the Gradio API routes (which manifests as "No API found").
# =============================================================================

import os
import sys
import json
import logging
import traceback

import numpy as np
import torch
from PIL import Image, ImageOps
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

# -----------------------------------------------------------------------------
# Monkey-patch gradio_client.utils to tolerate bool JSON-Schema values.
#
# gradio==5.9.1 (which HF Spaces hard-installs and we can't upgrade) crashes
# in launch() while building the API info: schemas with `additionalProperties:
# False` (a bool) hit `"const" in schema` and raise TypeError. This kills the
# whole app before Gradio ever binds a port, which the frontend then reports
# as "No API found". The patch makes both helpers return "Any" on a bool.
# -----------------------------------------------------------------------------
try:
    from gradio_client import utils as _gc_utils

    _orig_jsptp = _gc_utils._json_schema_to_python_type
    _orig_gt = _gc_utils.get_type

    def _safe_jsptp(schema, defs=None):
        if isinstance(schema, bool):
            return "Any"
        return _orig_jsptp(schema, defs)

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _orig_gt(schema)

    _gc_utils._json_schema_to_python_type = _safe_jsptp
    _gc_utils.get_type = _safe_get_type
except Exception:  # noqa: BLE001
    pass

# The Space layout (created by .github/workflows/deploy_hf.yml) places src/ next
# to app.py at /home/user/app. Make sure that's importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.models.stress_fatigue_model import StressFatigueModel  # noqa: E402

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
MAX_IMAGE_EDGE = 1024


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
        return processor, model, weights_ok, f"processor: {e!r}"

    try:
        log.info("Downloading config + weights from %s ...", HF_MODEL_REPO)
        try:
            cfg_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="config.json")
            with open(cfg_path) as fh:
                cfg = json.load(fh)
        except Exception as e:
            log.warning("config.json download failed (%s); using defaults", e)
            cfg = {}
        # use_resnet_baseline must be off in deployment
        cfg["use_resnet_baseline"] = False

        log.info("Building model with config: %s", cfg)
        model = StressFatigueModel(**cfg)
    except Exception as e:
        log.exception("model build failed")
        return processor, model, weights_ok, f"build: {e!r}"

    try:
        weights_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="pytorch_model.bin")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
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
        # We accept a small number of missing keys (e.g. backbone layers that
        # were frozen and not saved), but if essentially nothing loaded we
        # treat that as a failure for the UI's purposes.
        loaded_keys = len(cleaned) - len(unexpected)
        weights_ok = loaded_keys > 50  # rough sanity threshold
        if not weights_ok:
            log.warning("Only %d keys loaded — treating as untrained.", loaded_keys)
    except Exception as e:
        log.exception("weights load failed")

    try:
        model.to(DEVICE).eval()
    except Exception as e:
        log.exception("model.to(device) failed")
        return processor, None, weights_ok, f"to_device: {e!r}"

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
        except Exception as e:
            log.warning("LLM call failed, using canned fallback: %s", e)

    return _CANNED.get(
        (stress_label, fatigue_label),
        "Take a moment for yourself — short breaks, hydration, and rest go a long way.",
    )


# -----------------------------------------------------------------------------
# Image normalization
# -----------------------------------------------------------------------------
def _to_pil(image):
    if image is None:
        return None
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
        image = Image.fromarray(np.asarray(image))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    if max(image.size) > MAX_IMAGE_EDGE:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS)
    return image


def _placeholder_labels():
    return (
        {c: 1.0 / len(STRESS_CLASSES) for c in STRESS_CLASSES},
        {c: 1.0 / len(FATIGUE_CLASSES) for c in FATIGUE_CLASSES},
    )


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
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

        out = MODEL(pixel_values)
        # The training model returns a ModelOutput dataclass.
        stress_logits = out.stress_logits
        fatigue_logits = out.fatigue_logits

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
face image. A shared trunk (`Linear → LayerNorm → GELU → Dropout → Linear → ...`)
feeds two task-specific classifier heads — one for **stress** (3 classes) and
one for **fatigue** (2 classes).

**Why DINOv2.** Self-supervised ViT features transfer well to face analysis
without needing a huge labelled dataset. Lower transformer layers are frozen;
only the top layers + heads are fine-tuned.

**Training.** Multi-task focal loss, RandAugment, cosine LR schedule with
warmup. Tracked with TensorBoard.
"""

ABOUT_MD = """
## About

**Facial Stress & Fatigue Detection** — CMPE 258 Deep Learning Final Project,
San Jose State University.

- **Model repo:** [`NMemane1/facial-stress-fatigue-dinov2`](https://huggingface.co/NMemane1/facial-stress-fatigue-dinov2)
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
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )
