"""LLM-as-Judge evaluation for NarraCare.

Scores explanation quality and narrative consistency using an LLM
(or random scores in **mock mode** when no API key is available).
"""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation prompts
# ═══════════════════════════════════════════════════════════════════════════

EXPLANATION_QUALITY_PROMPT = """\
You are an expert rehabilitation clinician evaluating the quality of an
exercise recommendation explanation.

## Task recommended
Name: {task_name}
Category: {task_category}
Intensity: {task_intensity}
Description: {task_description}

## User state at recommendation time
Pain level: {pain}/10
Fatigue level: {fatigue}/10
ROM: {rom}°
Mood: {mood}/10
Stage: {stage}

## Explanation provided
{explanation}

## Scoring rubric (1–5)
1 — Irrelevant or factually wrong; no clinical reasoning.
2 — Vaguely relevant but missing key justification; may be misleading.
3 — Adequate; mentions the right exercise category and user state but
    lacks depth or personalisation.
4 — Good; clearly links the exercise choice to the user's current
    clinical state with appropriate reasoning.
5 — Excellent; precise, personalised, clinically sound reasoning that
    a patient would find reassuring and informative.

Respond with ONLY a single integer between 1 and 5.
"""

NARRATIVE_CONSISTENCY_PROMPT = """\
You are evaluating a sequence of daily rehabilitation narratives for
internal consistency, tone coherence, and appropriate progression.

## Narrative sequence (chronological)
{narrative_sequence}

## Scoring rubric (1–5)
1 — Contradictory information across days; tone is jarring or
    inappropriate for a rehabilitation context.
2 — Some inconsistencies; generic language that ignores day-to-day
    changes in user state.
3 — Mostly consistent; acknowledges state changes but occasionally
    repetitive or mismatched.
4 — Good consistency; natural progression of tone and content that
    reflects evolving user state.
5 — Excellent; reads as a coherent multi-day care narrative with smooth
    transitions, appropriate encouragement, and accurate clinical
    references.

Respond with ONLY a single integer between 1 and 5.
"""


# ═══════════════════════════════════════════════════════════════════════════
# LLMJudge
# ═══════════════════════════════════════════════════════════════════════════


class LLMJudge:
    """LLM-as-judge evaluator for recommendation quality.

    Parameters
    ----------
    config : dict | None
        Full application config (parsed ``default.yaml``).
    mock : bool | None
        If *True*, always use random scores (no API calls).
        If *None* (default), auto-detect: use mock mode when
        ``OPENAI_API_KEY`` is not set.
    seed : int
        Random seed for mock mode reproducibility.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        mock: Optional[bool] = None,
        seed: int = 42,
    ) -> None:
        self.config = config or {}
        self.llm_cfg = self.config.get("llm", {})
        self.seed = seed
        self.rng = random.Random(seed)

        # Determine mode
        if mock is not None:
            self.mock = mock
        else:
            self.mock = not bool(os.environ.get("OPENAI_API_KEY"))

        if self.mock:
            logger.info("LLMJudge running in MOCK mode (random scores).")
        else:
            logger.info("LLMJudge running in LIVE mode (OpenAI API).")

    # ── Single evaluations ───────────────────────────────────────────

    def evaluate_explanation_quality(
        self,
        explanation: str,
        task: dict,
        user_state: dict,
    ) -> float:
        """Score a single explanation on a 1–5 scale.

        Parameters
        ----------
        explanation : str
            The textual explanation shown to the user.
        task : dict
            Task dict (must have ``name``, ``category``, ``intensity``,
            ``description``).
        user_state : dict
            Daily state with ``pain_level``, ``fatigue_level``, ``rom``,
            ``mood``.

        Returns
        -------
        float   Score in [1, 5].
        """
        if self.mock:
            return float(self.rng.randint(2, 5))

        prompt = EXPLANATION_QUALITY_PROMPT.format(
            task_name=task.get("name", "N/A"),
            task_category=task.get("category", "N/A"),
            task_intensity=task.get("intensity", "N/A"),
            task_description=task.get("description", "N/A"),
            pain=user_state.get("pain_level", "?"),
            fatigue=user_state.get("fatigue_level", "?"),
            rom=user_state.get("rom", "?"),
            mood=user_state.get("mood", "?"),
            stage=user_state.get("stage", "?"),
            explanation=explanation,
        )
        return self._call_llm(prompt)

    def evaluate_narrative_consistency(
        self,
        narrative_sequence: List[str],
    ) -> float:
        """Score narrative consistency across multiple days (1–5).

        Parameters
        ----------
        narrative_sequence : list[str]
            Chronologically ordered daily narratives.

        Returns
        -------
        float   Score in [1, 5].
        """
        if self.mock:
            return float(self.rng.randint(2, 5))

        formatted = "\n\n---\n\n".join(
            f"**Day {i + 1}**\n{text}" for i, text in enumerate(narrative_sequence)
        )
        prompt = NARRATIVE_CONSISTENCY_PROMPT.format(
            narrative_sequence=formatted,
        )
        return self._call_llm(prompt)

    # ── Batch evaluation ─────────────────────────────────────────────

    def run_evaluation(
        self,
        interaction_logs: List[dict],
        n_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """Sample interactions, evaluate, and return average scores.

        Parameters
        ----------
        interaction_logs : list[dict]
            Logs from ``NarraCarePipeline.run_simulation``.
        n_samples : int | None
            Number of interactions to sample for explanation quality.
            Defaults to the config value or 50.

        Returns
        -------
        dict
            Keys: ``explanation_quality_avg``, ``narrative_consistency_avg``,
            ``explanation_quality_std``, ``narrative_consistency_std``,
            ``n_explanation_samples``, ``n_narrative_samples``.
        """
        eval_cfg = self.config.get("evaluation", {}).get("llm_judge", {})
        if n_samples is None:
            n_samples = eval_cfg.get("n_samples", 50)

        # --- Load task lookup for explanation eval ---
        from evaluation.metrics import _load_task_lookup
        task_lookup = _load_task_lookup()

        # ── Explanation quality ──────────────────────────────────────
        candidates = [
            ix for ix in interaction_logs
            if ix.get("action") == "accept"
            and ix.get("narrative")
            and ix.get("chosen_task_id")
        ]
        sample_size = min(n_samples, len(candidates))
        sampled = self.rng.sample(candidates, sample_size) if candidates else []

        eq_scores: List[float] = []
        for ix in sampled:
            task = task_lookup.get(ix["chosen_task_id"], {})
            explanation = ix.get("narrative", "")
            score = self.evaluate_explanation_quality(
                explanation, task, ix.get("state", {}),
            )
            eq_scores.append(score)

        # ── Narrative consistency ────────────────────────────────────
        # Group narratives by user, take consecutive-day windows
        from collections import defaultdict
        user_narratives: Dict[str, List[tuple]] = defaultdict(list)
        for ix in interaction_logs:
            nar = ix.get("narrative", "")
            if nar:
                user_narratives[ix.get("user_id", "")].append(
                    (ix.get("day", 0), nar),
                )

        nc_scores: List[float] = []
        for uid, day_nars in user_narratives.items():
            day_nars.sort(key=lambda x: x[0])
            # Take windows of 5 consecutive narratives
            texts = [n for _, n in day_nars]
            window = 5
            for start in range(0, len(texts) - window + 1, window):
                seq = texts[start : start + window]
                nc_scores.append(self.evaluate_narrative_consistency(seq))
            if len(nc_scores) >= n_samples:
                break

        return {
            "explanation_quality_avg": float(np.mean(eq_scores)) if eq_scores else 0.0,
            "explanation_quality_std": float(np.std(eq_scores)) if eq_scores else 0.0,
            "narrative_consistency_avg": float(np.mean(nc_scores)) if nc_scores else 0.0,
            "narrative_consistency_std": float(np.std(nc_scores)) if nc_scores else 0.0,
            "n_explanation_samples": len(eq_scores),
            "n_narrative_samples": len(nc_scores),
        }

    # ── LLM call helper ──────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> float:
        """Send prompt to OpenAI and extract integer score."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed — returning mock score.")
            return float(self.rng.randint(2, 5))

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return float(self.rng.randint(2, 5))

        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model=self.llm_cfg.get("model", "gpt-4o-mini"),
                temperature=0.0,
                max_tokens=8,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an evaluation assistant. "
                            "Respond with a single integer (1-5) only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            score = int("".join(c for c in text if c.isdigit())[:1] or "3")
            return float(max(1, min(5, score)))
        except Exception as exc:
            logger.error("LLM judge call failed: %s — returning mock score.", exc)
            return float(self.rng.randint(2, 5))


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Quick self-test with mock mode
    judge = LLMJudge(mock=True)
    score = judge.evaluate_explanation_quality(
        explanation="This low-intensity stretch is recommended because your pain is elevated today.",
        task={"name": "Heel Slides", "category": "flexibility", "intensity": "low", "description": "..."},
        user_state={"pain_level": 6, "fatigue_level": 4, "rom": 90, "mood": 5, "stage": 1},
    )
    print(f"Mock explanation quality score: {score}")

    nc = judge.evaluate_narrative_consistency([
        "Day 1: Welcome! Let's start easy.",
        "Day 2: Great work yesterday. Upping intensity slightly.",
        "Day 3: I see your pain is a bit higher — let's take it gentle.",
    ])
    print(f"Mock narrative consistency score: {nc}")
