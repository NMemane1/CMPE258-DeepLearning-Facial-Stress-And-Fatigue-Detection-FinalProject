"""LLM-powered wellness recommendation generator.

Takes a prediction (stress level + fatigue state + confidences) and produces
a short, supportive natural-language recommendation via a foundation model.

The prompt is intentionally engineered with:
- A role assignment ("wellness coach")
- Explicit constraints (length, no medical claims, supportive tone)
- Few-shot examples for tone consistency
- Structured output guidance

If the API call fails or is not configured, falls back to a templated
canned response so the demo never breaks.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional


STRESS_NAMES = ["low", "moderate", "high"]
FATIGUE_NAMES = ["alert", "drowsy"]


SYSTEM_PROMPT = """You are a warm, supportive wellness coach. You receive a brief readout of someone's current state from a vision-based detection model and you provide a single short, actionable recommendation.

Rules:
- 2-3 sentences only. No bullet lists.
- Be warm but not patronizing.
- Suggest one concrete action (e.g. a short walk, hydration, a breathing exercise, stretching, taking a real break, talking to someone).
- Never make medical diagnoses or claims.
- If both stress and fatigue are low/alert, validate the positive state and keep it brief.
- Acknowledge the model has limited information — these are signals, not certainties.
"""


FEW_SHOT_EXAMPLES = [
    {
        "input": "Stress: high (0.82), Fatigue: drowsy (0.71)",
        "output": "It looks like things might be running heavy on you right now — both elevated stress and tiredness showing up at once is your body asking for a real pause. Try stepping away from the screen for ten minutes with a glass of water and some slow, deep breaths. Even a short walk outside can reset both signals.",
    },
    {
        "input": "Stress: moderate (0.55), Fatigue: alert (0.78)",
        "output": "You're alert but carrying some tension. A two-minute box-breathing exercise (4 in, 4 hold, 4 out, 4 hold) tends to bring that down nicely. Keep going — you've got the focus right now to use it well.",
    },
    {
        "input": "Stress: low (0.91), Fatigue: alert (0.88)",
        "output": "You're in a great spot — calm and alert. This is a good window for focused work or anything that needs your best attention. Trust the signal and use it.",
    },
]


@dataclass
class ExplainerInput:
    stress_class: int        # 0,1,2
    stress_prob: float       # confidence in chosen class
    fatigue_class: int       # 0,1
    fatigue_prob: float


def format_input(inp: ExplainerInput) -> str:
    return (
        f"Stress: {STRESS_NAMES[inp.stress_class]} ({inp.stress_prob:.2f}), "
        f"Fatigue: {FATIGUE_NAMES[inp.fatigue_class]} ({inp.fatigue_prob:.2f})"
    )


def canned_response(inp: ExplainerInput) -> str:
    """Deterministic fallback when no LLM is configured."""
    s = STRESS_NAMES[inp.stress_class]
    f = FATIGUE_NAMES[inp.fatigue_class]
    if s == "high" and f == "drowsy":
        return ("It looks like both stress and tiredness are elevated. "
                "Consider stepping away from screens for 10 minutes, hydrating, "
                "and taking some slow deep breaths.")
    if s == "high":
        return ("Stress signals are running high. Try a brief breathing exercise "
                "or a short walk to reset before continuing.")
    if f == "drowsy":
        return ("Tiredness is showing up clearly. A glass of water, "
                "a short break, or fresh air might help more than caffeine right now.")
    if s == "moderate":
        return ("Some tension is showing. A two-minute box-breathing exercise "
                "(4 seconds in, 4 hold, 4 out, 4 hold) often helps.")
    return ("You're in a calm, alert state right now — a good moment for focused work.")


class LLMExplainer:
    """Wraps Anthropic / OpenAI API call behind a single interface."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-4-5",
        temperature: float = 0.7,
        max_tokens: int = 200,
        fallback_to_canned: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback = fallback_to_canned
        self._client = None
        self._available = self._init_client()

    def _init_client(self) -> bool:
        try:
            if self.provider == "anthropic":
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    return False
                import anthropic
                self._client = anthropic.Anthropic()
                return True
            elif self.provider == "openai":
                if not os.environ.get("OPENAI_API_KEY"):
                    return False
                from openai import OpenAI
                self._client = OpenAI()
                return True
        except Exception as e:
            print(f"[llm_explainer] client init failed: {e}")
        return False

    def _build_messages(self, inp: ExplainerInput):
        messages = []
        for ex in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})
        messages.append({"role": "user", "content": format_input(inp)})
        return messages

    def explain(self, inp: ExplainerInput) -> str:
        if not self._available:
            return canned_response(inp) if self.fallback else "[LLM not configured]"

        try:
            if self.provider == "anthropic":
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=SYSTEM_PROMPT,
                    messages=self._build_messages(inp),
                )
                return resp.content[0].text.strip()

            elif self.provider == "openai":
                msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + self._build_messages(inp)
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return resp.choices[0].message.content.strip()

        except Exception as e:
            print(f"[llm_explainer] API call failed: {e}")
            return canned_response(inp) if self.fallback else f"[LLM error: {e}]"

        return canned_response(inp)
