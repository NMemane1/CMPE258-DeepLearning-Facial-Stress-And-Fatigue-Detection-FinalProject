"""Inference pipeline.

Orchestrates: face detection → preprocessing → model forward → LLM explanation.
Used by both the Gradio app and standalone CLI inference.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.models.stress_fatigue_model import StressFatigueModel
from src.data.transforms import build_inference_transform
from src.data.dataset import STRESS_CLASS_NAMES, FATIGUE_CLASS_NAMES
from src.inference.face_detector import FaceDetector
from src.inference.llm_explainer import LLMExplainer, ExplainerInput


@dataclass
class PredictionResult:
    stress_class_name: str
    stress_probs: list[float]       # length 3
    stress_confidence: float
    fatigue_class_name: str
    fatigue_probs: list[float]       # length 2
    fatigue_confidence: float
    wellness_message: str
    latency_ms: float
    face_detected: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _pick_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class InferencePipeline:
    """Single-image prediction pipeline."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_repo: Optional[str] = None,
        device_preference: str = "auto",
        image_size: int = 224,
        enable_llm: bool = True,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-5",
    ):
        self.device = _pick_device(device_preference)
        self.transform = build_inference_transform(image_size)
        self.face_detector = FaceDetector()

        if model_path is None and model_repo is not None:
            model_path = self._download_from_hf(model_repo)

        if model_path and Path(model_path).exists():
            self.model = StressFatigueModel.from_pretrained(model_path, device=str(self.device))
        else:
            print("[inference] no trained checkpoint found — initializing from pretrained backbone for demo mode")
            self.model = StressFatigueModel().to(self.device).eval()

        self.llm = LLMExplainer(provider=llm_provider, model=llm_model) if enable_llm else None

    def _download_from_hf(self, repo: str) -> Optional[str]:
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=repo)
        except Exception as e:
            print(f"[inference] HF download failed: {e}")
            return None

    @torch.no_grad()
    def predict(self, image: Image.Image) -> PredictionResult:
        t0 = time.time()

        face_image = self.face_detector.detect_and_crop(image)
        face_detected = face_image is not image  # heuristic

        tensor = self.transform(face_image).unsqueeze(0).to(self.device)
        out = self.model(tensor)

        s_probs = torch.softmax(out.stress_logits, dim=-1).cpu().numpy()[0]
        f_probs = torch.softmax(out.fatigue_logits, dim=-1).cpu().numpy()[0]
        s_class = int(s_probs.argmax())
        f_class = int(f_probs.argmax())

        msg = ""
        if self.llm is not None:
            msg = self.llm.explain(ExplainerInput(
                stress_class=s_class, stress_prob=float(s_probs[s_class]),
                fatigue_class=f_class, fatigue_prob=float(f_probs[f_class]),
            ))

        return PredictionResult(
            stress_class_name=STRESS_CLASS_NAMES[s_class],
            stress_probs=s_probs.tolist(),
            stress_confidence=float(s_probs[s_class]),
            fatigue_class_name=FATIGUE_CLASS_NAMES[f_class],
            fatigue_probs=f_probs.tolist(),
            fatigue_confidence=float(f_probs[f_class]),
            wellness_message=msg,
            latency_ms=1000 * (time.time() - t0),
            face_detected=face_detected,
        )
