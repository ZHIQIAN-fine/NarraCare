"""User Understanding Agent — parses natural-language or structured user input
into a normalised ~15-dimensional feature vector consumed by the recommender.

Supports two modes:
  • **simulation** (no LLM): ``parse_structured(state_dict) -> FeatureVector``
  • **interactive** (LLM):   ``parse(user_input, user_profile) -> dict``
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature vector specification ──────────────────────────────────────────
# dim  name                       source             normalisation
# 0    pain_level                 state              /10
# 1    fatigue_level              state              /10
# 2    rom                        state              /180
# 3    mood                       state              /10
# 4    age                        profile            /100
# 5    days_since_surgery         profile            /365
# 6    stage                      profile            /4
# 7    completion_rate_7d         memory             already 0-1
# 8    consecutive_high_intensity memory             /7
# 9    has_specific_complaints    state              0|1
# 10   feedback_positive          state              0|1
# 11   bmi                        profile            clamp(15,40)/25 → 0-1
# 12   baseline_pain              profile            /10
# 13   baseline_fatigue           profile            /10
# 14   motivation_score           profile            mapping → 0-1
# ──────────────────────────────────────────────────────────────────────────

FEATURE_DIM = 15

PERSONALITY_MOTIVATION: Dict[str, float] = {
    "motivated": 0.9,
    "cautious": 0.5,
    "inconsistent": 0.3,
}


@dataclass
class FeatureVector:
    """Thin wrapper around a numpy array keeping the semantic names around."""

    values: np.ndarray  # shape (FEATURE_DIM,)
    names: List[str] = field(default_factory=lambda: [
        "pain_level", "fatigue_level", "rom", "mood",
        "age", "days_since_surgery", "stage",
        "completion_rate_7d", "consecutive_high_intensity",
        "has_specific_complaints", "feedback_positive",
        "bmi", "baseline_pain", "baseline_fatigue", "motivation_score",
    ])

    @property
    def dim(self) -> int:
        return len(self.values)

    def to_numpy(self) -> np.ndarray:
        return self.values.copy()

    def __repr__(self) -> str:
        pairs = ", ".join(f"{n}={v:.3f}" for n, v in zip(self.names, self.values))
        return f"FeatureVector({pairs})"


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class UserUnderstandingAgent:
    """Extracts a structured state dict (and feature vector) from user input.

    Parameters
    ----------
    config : dict
        The full application config (parsed ``default.yaml``).
    mode : str
        ``"simulation"`` or ``"interactive"``.
    """

    def __init__(self, config: dict, mode: str = "simulation") -> None:
        self.config = config
        self.mode = mode
        self.llm_config = config.get("llm", {})

    # ── LLM-based parsing (interactive mode) ─────────────────────────────

    def parse(self, user_input: str, user_profile: dict) -> dict:
        """Parse free-text user input into a structured state dict.

        In *simulation* mode a simple keyword heuristic is used.
        In *interactive* mode the OpenAI API is called.

        Returns
        -------
        dict
            Keys: pain_level, fatigue_level, rom, mood,
                  specific_complaints, feedback_on_last_task
        """
        if self.mode == "interactive":
            return self._parse_with_llm(user_input, user_profile)
        return self._parse_heuristic(user_input, user_profile)

    # ── Simulation mode: structured → FeatureVector ──────────────────────

    def parse_structured(
        self,
        state_dict: dict,
        user_profile: dict,
        completion_rate_7d: float = 0.0,
        consecutive_high_intensity: int = 0,
    ) -> FeatureVector:
        """Convert an already-structured state dict to a normalised feature vector.

        This is the primary path used in simulation experiments where no NL
        parsing is needed.

        Handles both naming conventions for profile fields (e.g.
        ``baseline_pain`` / ``pain_baseline``).
        """
        vec = np.zeros(FEATURE_DIM, dtype=np.float64)

        # ── Helper to read profile with fallback aliases ──
        def _p(primary: str, *aliases: str, default: float = 0) -> float:
            val = user_profile.get(primary)
            if val is not None:
                return float(val)
            for a in aliases:
                val = user_profile.get(a)
                if val is not None:
                    return float(val)
            return float(default)

        # State features
        vec[0] = _clamp(state_dict.get("pain_level", 0) / 10.0)
        vec[1] = _clamp(state_dict.get("fatigue_level", 0) / 10.0)
        vec[2] = _clamp(state_dict.get("rom", 0) / 180.0)
        vec[3] = _clamp(state_dict.get("mood", 5) / 10.0)

        # Profile features — support both naming conventions
        vec[4] = _clamp(_p("age", default=40) / 100.0)
        vec[5] = _clamp(_p("days_since_surgery", default=0) / 365.0)

        # Stage: may be int (1-4) or string (early_rehab, mid_rehab, ...)
        stage_raw = user_profile.get("stage", user_profile.get("current_stage", 1))
        stage_map = {"early_rehab": 1, "mid_rehab": 2, "late_rehab": 3, "return_to_sport": 4, "return_to_activity": 4, "maintenance": 4}
        if isinstance(stage_raw, str):
            stage_val = stage_map.get(stage_raw, 1)
        else:
            stage_val = int(stage_raw)
        vec[6] = _clamp(stage_val / 4.0)

        # Memory features (supplied by the pipeline)
        vec[7] = _clamp(completion_rate_7d)
        vec[8] = _clamp(consecutive_high_intensity / 7.0)

        # Derived binary features
        complaints = state_dict.get("specific_complaints", [])
        vec[9] = 1.0 if complaints else 0.0

        feedback = state_dict.get("feedback_on_last_task", {})
        if isinstance(feedback, dict):
            vec[10] = 1.0 if feedback.get("positive", False) else 0.0
        else:
            vec[10] = 0.0

        # Additional profile features — support both naming conventions
        bmi_raw = _p("bmi", default=25.0)
        vec[11] = _clamp((bmi_raw - 15.0) / 25.0)  # map [15,40] → [0,1]
        vec[12] = _clamp(_p("baseline_pain", "pain_baseline", default=3) / 10.0)
        vec[13] = _clamp(_p("baseline_fatigue", "fatigue_baseline", default=4) / 10.0)

        personality = user_profile.get("personality", "cautious")
        vec[14] = PERSONALITY_MOTIVATION.get(personality, 0.5)

        return FeatureVector(values=vec)

    # ── Internals ────────────────────────────────────────────────────────

    def _parse_heuristic(self, user_input: str, user_profile: dict) -> dict:
        """Keyword-based heuristic for lightweight (no-LLM) parsing."""
        text = user_input.lower()

        pain = self._extract_numeric(text, r"pain\s*(?:level)?\s*[:=]?\s*(\d+)", default=user_profile.get("baseline_pain", 3))
        fatigue = self._extract_numeric(text, r"fatigue\s*(?:level)?\s*[:=]?\s*(\d+)", default=user_profile.get("baseline_fatigue", 4))
        rom = self._extract_numeric(text, r"rom\s*[:=]?\s*(\d+)", default=user_profile.get("baseline_rom", 90))
        mood = self._extract_numeric(text, r"mood\s*[:=]?\s*(\d+)", default=5)

        # Detect complaints
        complaint_keywords = ["sore", "ache", "stiff", "tight", "swollen", "sharp", "pain"]
        complaints = [kw for kw in complaint_keywords if kw in text]

        # Simple feedback detection
        positive_keywords = ["good", "great", "better", "improved", "easy"]
        negative_keywords = ["hard", "difficult", "worse", "too much", "bad"]
        feedback: Dict[str, Any] = {}
        if any(kw in text for kw in positive_keywords):
            feedback = {"positive": True, "comment": "Positive feedback detected"}
        elif any(kw in text for kw in negative_keywords):
            feedback = {"positive": False, "comment": "Negative feedback detected"}

        return {
            "pain_level": int(pain),
            "fatigue_level": int(fatigue),
            "rom": int(rom),
            "mood": int(mood),
            "specific_complaints": complaints,
            "feedback_on_last_task": feedback,
        }

    def _parse_with_llm(self, user_input: str, user_profile: dict) -> dict:
        """Parse user input using OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai package not installed; falling back to heuristic parsing.")
            return self._parse_heuristic(user_input, user_profile)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set; falling back to heuristic parsing.")
            return self._parse_heuristic(user_input, user_profile)

        client = OpenAI(api_key=api_key)

        system_prompt = (
            "You are a rehabilitation assessment assistant. "
            "Extract the following fields from the user's message as JSON:\n"
            "- pain_level (int, 0-10)\n"
            "- fatigue_level (int, 0-10)\n"
            "- rom (int, degrees, 0-180)\n"
            "- mood (int, 0-10)\n"
            "- specific_complaints (list of strings)\n"
            "- feedback_on_last_task (object with 'positive': bool and 'comment': str)\n\n"
            f"User profile context: age={user_profile.get('age')}, "
            f"surgery={user_profile.get('surgery_type')}, "
            f"days_since_surgery={user_profile.get('days_since_surgery')}, "
            f"baseline_pain={user_profile.get('baseline_pain')}\n\n"
            "If a value is not mentioned, use reasonable defaults based on the profile. "
            "Respond ONLY with valid JSON."
        )

        try:
            response = client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                temperature=self.llm_config.get("temperature", 0.3),
                max_tokens=self.llm_config.get("max_tokens", 512),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
            )
            import json
            content = response.choices[0].message.content.strip()
            # Strip markdown code fence if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            return json.loads(content)
        except Exception as e:
            logger.error("LLM parsing failed: %s — falling back to heuristic", e)
            return self._parse_heuristic(user_input, user_profile)

    @staticmethod
    def _extract_numeric(text: str, pattern: str, default: float) -> float:
        """Extract the first numeric match for *pattern* from *text*."""
        m = re.search(pattern, text)
        if m:
            return float(m.group(1))
        return default
