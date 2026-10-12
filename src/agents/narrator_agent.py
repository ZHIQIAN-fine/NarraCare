"""Narrator Agent — wraps recommended tasks in human-friendly narrative
text, using either template-based rendering or LLM generation.
"""

from __future__ import annotations

import logging
import os
import re
from string import Template
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Narrative templates — rehabilitation_camp style
# ═══════════════════════════════════════════════════════════════════════════

_REHAB_CAMP_TEMPLATES: List[str] = [
    # Template 1 — motivational / coach-like
    (
        "🏕️ **Day ${day} — Rehab Camp Daily Briefing**\n\n"
        "Good ${time_of_day}, ${user_name}! "
        "Your body is telling us it's at pain level ${pain}/10 and fatigue ${fatigue}/10 today. "
        "That means we'll aim for **${intensity_plan}** today.\n\n"
        "Here's your mission lineup:\n"
        "${task_list}\n\n"
        "Remember: every rep is a step closer to ${goal}. Let's make today count! 💪"
    ),
    # Template 2 — empathetic / warm
    (
        "🌿 **Day ${day} at Your Rehab Camp**\n\n"
        "Hey ${user_name}, I see you're feeling ${mood_text} today "
        "(pain ${pain}/10, fatigue ${fatigue}/10). "
        "That's completely okay — recovery isn't a straight line.\n\n"
        "I've picked some exercises that match how you're feeling:\n"
        "${task_list}\n\n"
        "Take it one task at a time. You're doing great — ${encouragement}"
    ),
    # Template 3 — progress-focused
    (
        "📊 **Day ${day} Progress Report & Plan**\n\n"
        "${user_name}, here's where you stand:\n"
        "• ROM: ${rom}° • Pain: ${pain}/10 • Stage: ${stage}\n"
        "• Completion rate (7-day): ${completion_pct}%\n\n"
        "Based on your progress, today's recommended exercises are:\n"
        "${task_list}\n\n"
        "${progression_note}"
        "Keep it up — consistency is your superpower! 🌟"
    ),
]

_MOOD_MAP: Dict[str, str] = {
    "great": "fantastic",
    "good": "pretty good",
    "okay": "okay",
    "low": "a bit low",
    "bad": "not so great",
}

_ENCOURAGEMENT_POOL: List[str] = [
    "your future self will thank you!",
    "small steps lead to big gains.",
    "rest is part of training too.",
    "you've already come so far!",
    "just keep showing up — that's what matters.",
]

_TIME_OF_DAY = {range(5, 12): "morning", range(12, 17): "afternoon", range(17, 22): "evening"}


def _get_time_of_day(hour: int = 9) -> str:
    for rng, label in _TIME_OF_DAY.items():
        if hour in rng:
            return label
    return "day"


class NarratorAgent:
    """Generates narrative-wrapped recommendations.

    Parameters
    ----------
    config : dict
        Full application config (parsed ``default.yaml``).
    mode : str
        ``"simulation"`` (template-based) or ``"interactive"`` (LLM).
    """

    def __init__(self, config: dict, mode: str = "simulation") -> None:
        self.config = config
        self.mode = mode
        self.narrator_cfg = config.get("narrator", {})
        self.style: str = self.narrator_cfg.get("style", "rehabilitation_camp")
        self.language: str = self.narrator_cfg.get("language", "en")
        self.llm_config = config.get("llm", {})

    # ── Main wrapping method ─────────────────────────────────────────────

    def wrap(
        self,
        tasks: List[dict],
        user_state: dict,
        user_profile: dict,
        day: int,
        completion_rate_7d: float = 0.0,
    ) -> str:
        """Wrap recommended tasks into a narrative string.

        Parameters
        ----------
        tasks : list[dict]
            Recommended task dicts (must have ``task_id``, ``name``, ``intensity``).
        user_state : dict
            Current state (``pain_level``, ``fatigue_level``, ``mood``, ``rom``).
        user_profile : dict
            Static profile (``name``, ``stage``, ``goals``).
        day : int
            Simulation day number.
        completion_rate_7d : float
            Recent 7-day completion rate (0-1).

        Returns
        -------
        str
            Rendered narrative text.
        """
        if self.mode == "interactive":
            return self._wrap_with_llm(tasks, user_state, user_profile, day, completion_rate_7d)
        return self._wrap_template(tasks, user_state, user_profile, day, completion_rate_7d)

    # ── Explanation generation ───────────────────────────────────────────

    def generate_explanation(self, task: dict, user_state: dict) -> str:
        """Generate a brief explanation of *why* this task is recommended.

        Parameters
        ----------
        task : dict
            Task dict with ``task_id``, ``name``, ``category``, ``intensity``.
        user_state : dict
            Current daily state.

        Returns
        -------
        str
        """
        if self.mode == "interactive":
            return self._explain_with_llm(task, user_state)
        return self._explain_template(task, user_state)

    # ── Template-based rendering ─────────────────────────────────────────

    def _wrap_template(
        self,
        tasks: List[dict],
        user_state: dict,
        user_profile: dict,
        day: int,
        completion_rate_7d: float,
    ) -> str:
        # Pick template by cycling through based on day
        template_str = _REHAB_CAMP_TEMPLATES[day % len(_REHAB_CAMP_TEMPLATES)]

        # Build the task list
        task_lines: List[str] = []
        for i, t in enumerate(tasks, 1):
            explanation = self._explain_template(t, user_state)
            line = (
                f"  {i}. **{t.get('name', t.get('task_id', 'Task'))}** "
                f"({t.get('intensity', '?')} intensity, {t.get('category', '?')})\n"
                f"     _{explanation}_"
            )
            task_lines.append(line)
        task_list = "\n".join(task_lines) if task_lines else "  _(No tasks scheduled today — rest day!)_"

        # Mood text
        mood_val = user_state.get("mood", 5)
        if mood_val >= 8:
            mood_text = "great"
        elif mood_val >= 6:
            mood_text = "good"
        elif mood_val >= 4:
            mood_text = "okay"
        elif mood_val >= 2:
            mood_text = "low"
        else:
            mood_text = "bad"
        mood_text = _MOOD_MAP.get(mood_text, mood_text)

        pain = user_state.get("pain_level", 0)
        fatigue = user_state.get("fatigue_level", 0)

        # Intensity plan
        if pain >= 7 or fatigue >= 8:
            intensity_plan = "gentle recovery work"
        elif pain >= 4 or fatigue >= 5:
            intensity_plan = "moderate, well-paced exercises"
        else:
            intensity_plan = "a solid, active session"

        # Encouragement
        import hashlib
        idx = int(hashlib.md5(f"{day}".encode()).hexdigest(), 16) % len(_ENCOURAGEMENT_POOL)
        encouragement = _ENCOURAGEMENT_POOL[idx]

        # Goals
        goals = user_profile.get("goals", [])
        goal = goals[0] if goals else "your recovery goals"

        # Progression note
        stage = user_profile.get("stage", 1)
        if completion_rate_7d >= 0.8:
            progression_note = f"🎯 Great adherence! You may be ready to discuss moving to stage {stage + 1} soon.\n\n"
        else:
            progression_note = ""

        variables = {
            "day": str(day),
            "time_of_day": _get_time_of_day(),
            "user_name": user_profile.get("name", "there"),
            "pain": str(pain),
            "fatigue": str(fatigue),
            "intensity_plan": intensity_plan,
            "task_list": task_list,
            "goal": goal,
            "mood_text": mood_text,
            "encouragement": encouragement,
            "rom": str(user_state.get("rom", "?")),
            "stage": str(stage),
            "completion_pct": f"{completion_rate_7d * 100:.0f}",
            "progression_note": progression_note,
        }

        return Template(template_str).safe_substitute(variables)

    def _explain_template(self, task: dict, user_state: dict) -> str:
        """Generate a short template-based explanation."""
        category = task.get("category", "general")
        intensity = task.get("intensity", "low")
        pain = user_state.get("pain_level", 0)
        fatigue = user_state.get("fatigue_level", 0)

        reasons: List[str] = []

        if category in ("stretching", "breathing", "relaxation"):
            reasons.append("helps with recovery and relaxation")
        elif category == "strengthening":
            reasons.append("builds muscle support around the joint")
        elif category in ("balance", "balance_advanced"):
            reasons.append("improves stability and proprioception")
        elif category == "cardio":
            reasons.append("boosts endurance and circulation")
        else:
            reasons.append("supports your rehabilitation progress")

        if intensity == "low" and (pain >= 5 or fatigue >= 6):
            reasons.append("chosen because your pain/fatigue is elevated today")
        elif intensity == "high" and pain <= 3 and fatigue <= 3:
            reasons.append("you're feeling strong, so we can push a bit more")
        elif intensity == "medium":
            reasons.append("a balanced challenge for your current state")

        return "; ".join(reasons).capitalize() + "."

    # ── LLM-based rendering ──────────────────────────────────────────────

    def _wrap_with_llm(
        self,
        tasks: List[dict],
        user_state: dict,
        user_profile: dict,
        day: int,
        completion_rate_7d: float,
    ) -> str:
        """Use OpenAI to generate a richer narrative."""
        try:
            from openai import OpenAI
        except ImportError:
            logger.warning("openai not installed — falling back to templates.")
            return self._wrap_template(tasks, user_state, user_profile, day, completion_rate_7d)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — falling back to templates.")
            return self._wrap_template(tasks, user_state, user_profile, day, completion_rate_7d)

        client = OpenAI(api_key=api_key)

        task_desc = "\n".join(
            f"- {t.get('name', t['task_id'])} (intensity: {t.get('intensity')}, category: {t.get('category')})"
            for t in tasks
        )

        prompt = (
            f"You are a warm, encouraging rehabilitation coach. "
            f"Write a Day {day} narrative briefing in the 'rehabilitation camp' style.\n\n"
            f"User: {user_profile.get('name', 'User')}, age {user_profile.get('age')}, "
            f"stage {user_profile.get('stage')}, goals: {user_profile.get('goals')}\n"
            f"Today's state: pain={user_state.get('pain_level')}/10, "
            f"fatigue={user_state.get('fatigue_level')}/10, "
            f"ROM={user_state.get('rom')}°, mood={user_state.get('mood')}/10\n"
            f"7-day completion rate: {completion_rate_7d:.0%}\n\n"
            f"Recommended exercises:\n{task_desc}\n\n"
            f"Include: greeting, acknowledgement of how they feel, the exercise list with "
            f"brief reasons, and an encouraging closing. Keep it under 300 words."
        )

        try:
            response = client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                temperature=self.llm_config.get("temperature", 0.7),
                max_tokens=self.llm_config.get("max_tokens", 1024),
                messages=[
                    {"role": "system", "content": "You are NarraCare, a supportive rehabilitation narrative coach."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM narrative generation failed: %s", e)
            return self._wrap_template(tasks, user_state, user_profile, day, completion_rate_7d)

    def _explain_with_llm(self, task: dict, user_state: dict) -> str:
        """Generate a one-sentence LLM explanation for a single task."""
        try:
            from openai import OpenAI
        except ImportError:
            return self._explain_template(task, user_state)

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return self._explain_template(task, user_state)

        client = OpenAI(api_key=api_key)

        prompt = (
            f"In one concise sentence, explain why this rehabilitation exercise is recommended:\n"
            f"Exercise: {task.get('name', task.get('task_id'))} "
            f"(category: {task.get('category')}, intensity: {task.get('intensity')})\n"
            f"Patient state: pain={user_state.get('pain_level')}/10, "
            f"fatigue={user_state.get('fatigue_level')}/10, "
            f"ROM={user_state.get('rom')}°"
        )

        try:
            response = client.chat.completions.create(
                model=self.llm_config.get("model", "gpt-4o-mini"),
                temperature=0.5,
                max_tokens=100,
                messages=[
                    {"role": "system", "content": "You are a rehabilitation expert. Be concise."},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM explanation failed: %s", e)
            return self._explain_template(task, user_state)
