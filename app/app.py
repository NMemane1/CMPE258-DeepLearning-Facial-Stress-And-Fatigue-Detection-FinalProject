"""Gradio web application for facial stress & fatigue detection.

Run locally:
    python -m app.app

Deployment to HuggingFace Spaces is handled by `.github/workflows/deploy_hf.yml`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow running as `python -m app.app` from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from PIL import Image

from src.inference.predict import InferencePipeline


# ============================================================
# Build pipeline once at startup
# ============================================================

MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL_REPO = os.environ.get("MODEL_REPO", "NMemane1/facial-stress-fatigue-dinov2")
ENABLE_LLM = os.environ.get("ENABLE_LLM", "true").lower() == "true"

print("=" * 60)
print(" Facial Stress & Fatigue Detection — Web App")
print("=" * 60)
print(f"  MODEL_PATH: {MODEL_PATH or '(unset, will try HF Hub)'}")
print(f"  MODEL_REPO: {MODEL_REPO}")
print(f"  ENABLE_LLM: {ENABLE_LLM}")
print("=" * 60)

PIPELINE = InferencePipeline(
    model_path=MODEL_PATH or None,
    model_repo=MODEL_REPO if not MODEL_PATH else None,
    enable_llm=ENABLE_LLM,
)


# ============================================================
# Prediction handler
# ============================================================

def predict_image(image):
    if image is None:
        return None, None, "Please upload or capture an image.", ""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    result = PIPELINE.predict(image)

    stress_chart = {
        f"Low":      result.stress_probs[0],
        f"Moderate": result.stress_probs[1],
        f"High":     result.stress_probs[2],
    }
    fatigue_chart = {
        "Alert":  result.fatigue_probs[0],
        "Drowsy": result.fatigue_probs[1],
    }

    summary = (
        f"**Stress:** {result.stress_class_name.upper()} "
        f"({result.stress_confidence:.0%} confidence)  \n"
        f"**Fatigue:** {result.fatigue_class_name.upper()} "
        f"({result.fatigue_confidence:.0%} confidence)  \n\n"
        f"_Latency: {result.latency_ms:.0f}ms · Face detected: {result.face_detected}_"
    )
    return stress_chart, fatigue_chart, summary, result.wellness_message


# ============================================================
# Methodology section content (rubric requirement)
# ============================================================

METHODOLOGY_MD = """
### Why We Chose What We Chose

This panel summarizes our design decisions. See `docs/methodology.md` in the repo for full justification.

| Choice | What | Why |
|--------|------|-----|
| **Backbone** | DINOv2-small (Meta, 2023) | Self-supervised pretraining transfers better than ImageNet supervision on small-data affective tasks. |
| **Multi-task heads** | Shared 384→256 MLP + per-task classifier | Stress & fatigue share facial cues — joint training acts as regularization. |
| **Activation** | GELU | Matches the activation in the DINOv2 transformer; smoother gradients than ReLU. |
| **Normalization** | LayerNorm (ViT) + BatchNorm (heads) | LayerNorm is standard for transformers; BN regularizes the small head MLPs. |
| **Loss** | Class-weighted CE + focal loss (γ=2) | Class imbalance + focal down-weights easy examples. |
| **Optimizer** | AdamW, discriminative LR | Backbone needs slow updates (3e-5); freshly initialized heads need fast (3e-4). |
| **Schedule** | Linear warmup + cosine decay | Prevents catastrophic forgetting early; smooth convergence. |
| **Augmentation** | RandAugment(n=2, m=9) + horizontal flip | Strong, automated augmentation pipeline; faces are roughly symmetric. |
| **LLM explanation** | Few-shot prompt, T=0.7 | Templated structure with grounded tone via examples — no LLM fine-tuning needed. |

### Pipeline
```
Image → MediaPipe face crop → DINOv2 backbone → shared MLP →
  ├─ Stress classifier (3-class)
  └─ Fatigue classifier (2-class)
      ↓
  LLM-engineered prompt → wellness recommendation
```
"""


# ============================================================
# Gradio interface
# ============================================================

EXAMPLES = []   # populate after deployment with sample images in assets/

with gr.Blocks(
    title="Facial Stress & Fatigue Detection",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:
    gr.Markdown(
        """
        # 🧠 Facial Stress & Fatigue Detection
        **CMPE 258 Final Project** — Multi-task vision foundation model with LLM-powered wellness recommendations.

        Upload a face image or use your webcam. The model predicts your stress level (low / moderate / high)
        and fatigue state (alert / drowsy), and a prompt-engineered LLM generates a personalized recommendation.

        > _Disclaimer: This is a course project, not a medical device. Predictions are signals, not diagnoses._
        """
    )

    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(type="pil", label="Input image (upload or webcam)", sources=["upload", "webcam"])
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                out_stress = gr.Label(label="Stress probabilities", num_top_classes=3)
                out_fatigue = gr.Label(label="Fatigue probabilities", num_top_classes=2)
        out_summary = gr.Markdown(label="Summary")
        out_message = gr.Textbox(label="Wellness recommendation (LLM)", lines=4, interactive=False)
        btn.click(predict_image, inputs=inp_image,
                  outputs=[out_stress, out_fatigue, out_summary, out_message])

    with gr.Tab("Methodology"):
        gr.Markdown(METHODOLOGY_MD)

    with gr.Tab("About"):
        gr.Markdown(
            """
            ### About this project
            - **Course:** CMPE 258, San Jose State University
            - **Backbone:** [DINOv2-small](https://huggingface.co/facebook/dinov2-small) (Meta)
            - **Datasets:** [Kaggle Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset),
              [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) (emotion → stress remap)
            - **LLM:** Claude (Anthropic) via prompt engineering — no fine-tuning
            - **Code:** [github.com/NMemane1/...](https://github.com/NMemane1/CMPE258-DeepLearning-Facial-Stress-And-Fatigue-Detection-FinalProject)
            - **Training infrastructure:** Kaggle TPU v3-8 (free tier) + Weights & Biases tracking
            """
        )


if __name__ == "__main__":
    demo.queue(max_size=20).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )
