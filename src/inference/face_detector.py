"""Face detection + alignment using MediaPipe.

Wraps MediaPipe's lightweight face detector for the inference pipeline.
On import failure, falls back to a no-op (returns the input image unchanged)
so the app can still run for testing.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from PIL import Image


class FaceDetector:
    """Detect the largest face in an image and crop with margin."""

    def __init__(self, margin: float = 0.20, min_detection_confidence: float = 0.5):
        self.margin = margin
        self._detector = None
        try:
            import mediapipe as mp
            self._mp = mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=min_detection_confidence,
            )
        except Exception as e:
            print(f"[face_detector] mediapipe unavailable ({e}); face crop disabled")

    def detect_and_crop(self, image: Image.Image) -> Image.Image:
        if self._detector is None:
            return image
        rgb = np.asarray(image.convert("RGB"))
        h, w = rgb.shape[:2]
        results = self._detector.process(rgb)
        if not results.detections:
            return image

        # pick largest face
        best, best_area = None, 0
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            area = bb.width * bb.height
            if area > best_area:
                best, best_area = bb, area

        x = max(0, int((best.xmin - self.margin * best.width) * w))
        y = max(0, int((best.ymin - self.margin * best.height) * h))
        x2 = min(w, int((best.xmin + best.width * (1 + self.margin)) * w))
        y2 = min(h, int((best.ymin + best.height * (1 + self.margin)) * h))
        if x2 <= x or y2 <= y:
            return image
        return Image.fromarray(rgb[y:y2, x:x2])
