"""Tests for inference pipeline (LLM explainer + metrics)."""
import pytest
import numpy as np


class TestLLMExplainer:
    def test_canned_response_high_stress_drowsy(self):
        from src.inference.llm_explainer import canned_response, ExplainerInput
        inp = ExplainerInput(stress_class=2, stress_prob=0.8, fatigue_class=1, fatigue_prob=0.7)
        msg = canned_response(inp)
        assert len(msg) > 20
        assert "stress" in msg.lower() or "tired" in msg.lower() or "break" in msg.lower()

    def test_canned_response_low_stress_alert(self):
        from src.inference.llm_explainer import canned_response, ExplainerInput
        inp = ExplainerInput(stress_class=0, stress_prob=0.9, fatigue_class=0, fatigue_prob=0.9)
        msg = canned_response(inp)
        assert len(msg) > 10

    def test_format_input(self):
        from src.inference.llm_explainer import format_input, ExplainerInput
        inp = ExplainerInput(stress_class=2, stress_prob=0.82, fatigue_class=1, fatigue_prob=0.71)
        s = format_input(inp)
        assert "high" in s.lower()
        assert "drowsy" in s.lower()
        assert "0.82" in s
        assert "0.71" in s

    def test_explainer_handles_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        from src.inference.llm_explainer import LLMExplainer, ExplainerInput
        ex = LLMExplainer(provider="anthropic", fallback_to_canned=True)
        msg = ex.explain(ExplainerInput(0, 0.9, 0, 0.9))
        assert len(msg) > 10


class TestMetrics:
    def test_metrics_dict_shape(self):
        from src.evaluation.metrics import compute_metrics_dict
        rng = np.random.default_rng(42)
        s_probs = rng.dirichlet(np.ones(3), size=100)
        s_tgt = rng.integers(0, 3, size=100)
        f_probs = rng.dirichlet(np.ones(2), size=100)
        f_tgt = rng.integers(0, 2, size=100)
        out = compute_metrics_dict(s_probs, s_tgt, f_probs, f_tgt)
        for k in ("stress/balanced_acc", "fatigue/balanced_acc",
                  "balanced_acc_mean", "macro_f1_mean"):
            assert k in out
            assert 0.0 <= out[k] <= 1.0

    def test_metrics_handles_missing_labels(self):
        from src.evaluation.metrics import compute_metrics_dict
        rng = np.random.default_rng(0)
        s_probs = rng.dirichlet(np.ones(3), size=50)
        s_tgt = np.full(50, -1, dtype=int)   # all missing
        f_probs = rng.dirichlet(np.ones(2), size=50)
        f_tgt = rng.integers(0, 2, size=50)
        out = compute_metrics_dict(s_probs, s_tgt, f_probs, f_tgt)
        assert out["stress/n"] == 0
        assert out["fatigue/n"] == 50
