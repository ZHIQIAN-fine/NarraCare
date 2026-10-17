"""
NarraCare Narrator Agent — v1.3.2 Final
========================================

Produces natural-language prompts that frame each pipeline step
(opening, recommendation, rejection, safety trigger, turning point,
weekly reflection, final farewell, daily closing) in one of three
tones (doctor / coach / companion), driven by a 3-Act narrative arc.

Tone selection priority (strictly enforced):
    1. safety_trigger    -> doctor_companion_hybrid
    2. red_light         -> companion (forced)
    3. base_tone         -> transition-zone mix or Act default
    4. yellow_light      -> f"{base_tone}+companion" (50/50 at render time)

Key safety/robustness features:
    * _REQUIRED_KEYS declared as flat sets per scenario_key
    * Missing-context fallback with per-scenario safe templates
    * fallback_counter + total_render_count exported for pipeline log
    * _current_override_reasons kept on the instance so pipeline can
      serialise them into prompt_level_log["all_override_reasons"]

Author: NarraCare team
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =========================================================================
# Static declarations — kept at module scope so pipeline / tests can import.
# =========================================================================

#: Scenario-key -> set of required context keys for the template.
_REQUIRED_KEYS: Dict[str, set] = {
    # Standard scenarios (single-tone templates).
    "daily_opening": {
        "name", "day", "condition", "pain", "category", "safety_note",
        "prev_task", "act1_milestone",
    },
    "recommendation_framing": {
        "task_name", "difficulty", "reason_from_bandit", "name",
        "body_part", "rehab_stage",
    },
    "user_rejection_response": {"pain", "alt_task", "name"},
    "turning_point": {"name", "day", "milestone", "next_phase_summary"},
    "weekly_reflection": {"week", "adherence", "avg_pain", "rom_delta", "trend", "name"},
    "final_farewell": {"name", "duration", "final_metrics", "maintenance_plan"},

    # Safety hybrid (Doctor + Companion compound rendering).
    "safety_agent_trigger_hybrid": {
        "blocked_task", "rule_name", "rule_clinical_reason",
        "safe_task", "same_body_part", "risk_factor",
        "name", "empathy_line",
    },

    # Safety single-tone variants (used when not hybrid).
    "safety_agent_trigger": {
        "blocked_task", "rule_name", "rule_clinical_reason",
        "safe_task", "same_body_part", "risk_factor", "name",
    },

    # Daily Closing variants — v1.3.1 splits neutral into 3 per tone.
    "daily_closing_no_interaction": {"name"},
    "daily_closing_safety_followup": {"name"},
    "daily_closing_skip_recovery": {"name"},
    "daily_closing_concern": {"name", "pain_level", "fatigue_level"},
    "daily_closing_celebration": {"name", "streak_count"},
    "daily_closing_neutral_doctor": {"name", "adherence", "pain_delta", "next_focus"},
    "daily_closing_neutral_coach": {"name"},
    "daily_closing_neutral_companion": {"name", "personal_moment_from_today"},
}

#: Safe fallback templates used when _REQUIRED_KEYS is not fully satisfied.
_FALLBACK_TEMPLATES: Dict[str, str] = {
    "safety_agent_trigger_hybrid": (
        "A safer alternative was selected today. "
        "{name}, we'll push harder when the body is ready."
    ),
    "safety_agent_trigger": (
        "The safety protocol selected a safer alternative today, {name}."
    ),
    "daily_opening": "Day {day} — welcome back, {name}.",
    "recommendation_framing": "Today's plan is ready, {name}.",
    "user_rejection_response": "No problem, {name}. We'll adapt.",
    "turning_point": "A milestone today, {name}. More to come.",
    "weekly_reflection": "Week recap recorded, {name}.",
    "final_farewell": "End of program, {name}. Well done.",
    "daily_closing_no_interaction": "Didn't hear from you today, {name}. Tomorrow's a clean slate.",
    "daily_closing_safety_followup": "The safety rule had the last word today, {name}. We'll push again when ready.",
    "daily_closing_skip_recovery": "Rest day today, {name}. Bodies ask for them.",
    "daily_closing_concern": "Today took something out of you, {name}. Rest tonight.",
    "daily_closing_celebration": "Another one in the books, {name}. See you tomorrow.",
    "daily_closing_neutral_doctor": "Session logged, {name}. Rest well.",
    "daily_closing_neutral_coach": "Done for today, {name}. See you tomorrow.",
    "daily_closing_neutral_companion": "That's another day behind us, {name}.",
    "_default": "Session note recorded, {name}. — NarraCare",
}

#: Whitelist of transition-zone JSON keys.
_TRANSITION_KEYS: Tuple[str, ...] = ("act1_to_act2", "act2_to_act3")


class Narrator:
    """Produces user-facing prompts according to Act + tone + override state.

    The class is deliberately stateless with respect to user-level history
    (that lives in the pipeline). Instance state only tracks aggregate
    counters needed for the final experiment log:

        * total_render_count: total number of _render_prompt calls
        * fallback_counter: {scenario_key: fallback_hits}
        * _current_override_reasons: reasons that triggered the most
          recent _select_tone decision (consumed by pipeline per call)
    """

    # Expose module-level constants on the class for callers that prefer
    # Narrator._REQUIRED_KEYS over the module name.
    _REQUIRED_KEYS = _REQUIRED_KEYS
    _FALLBACK_TEMPLATES = _FALLBACK_TEMPLATES
    _TRANSITION_KEYS = _TRANSITION_KEYS

    def __init__(
        self,
        arcs_path: str | Path,
        templates_path: str | Path,
        seed: int = 42,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("NarraCare.Narrator")
        self.rng = np.random.default_rng(seed)

        with open(arcs_path, "r", encoding="utf-8") as f:
            self.arcs_data = json.load(f)
        with open(templates_path, "r", encoding="utf-8") as f:
            self.templates = json.load(f)

        self.arcs = self.arcs_data["arcs"]
        self.personality_modifiers = self.arcs_data["personality_modifiers"]
        self.transition_rules = self.templates["transition_rules"]
        self.emotional_override = self.transition_rules["emotional_override"]

        # Aggregate counters (consumed by pipeline at experiment end).
        self.total_render_count: int = 0
        self.fallback_counter: Dict[str, int] = {}

        # Per-decision scratch state (consumed per prompt by pipeline).
        self._current_override_reasons: List[str] = []

    # ---------------------------------------------------------------------
    # Arc resolution
    # ---------------------------------------------------------------------
    def _get_arc_context(
        self, user_profile: Dict[str, Any], day: int
    ) -> Tuple[str, Dict[str, Any]]:
        """Resolve (act_key, arc_meta) by day. Falls back to act3 if OOR."""
        condition = user_profile.get("condition", "ACL")
        condition = {"ACL_reconstruction": "ACL", "rotator_cuff_repair": "RotatorCuff",
                     "chronic_lower_back_pain": "BackPain"}.get(condition, condition)
        arc = self.arcs.get(condition, self.arcs["ACL"])
        for act_key in ("act1", "act2", "act3"):
            start, end = arc[act_key]["day_range"]
            if start <= day <= end:
                return act_key, arc[act_key]
        return "act3", arc["act3"]

    # ---------------------------------------------------------------------
    # Base-tone resolution (transition-zone aware, override-free)
    # ---------------------------------------------------------------------
    def _get_base_tone(self, user_profile: Dict[str, Any], day: int) -> str:
        """Return base tone for the day, respecting transition-zone mixing.

        Transition zones are whitelisted via `_TRANSITION_KEYS` so a stray
        future key (e.g. "priority_order_comment") cannot crash the loop.
        """
        for trans_key in self._TRANSITION_KEYS:
            trans = self.transition_rules.get(trans_key)
            if not trans:
                continue
            lo, hi = trans["days"]
            if lo <= day <= hi:
                tones = list(trans["mix"].keys())
                weights = list(trans["mix"].values())
                return str(self.rng.choice(tones, p=np.array(weights) / sum(weights)))
        _, arc_meta = self._get_arc_context(user_profile, day)
        return arc_meta["default_tone"]

    # ---------------------------------------------------------------------
    # Emotional override state
    # ---------------------------------------------------------------------
    def _check_emotional_state(
        self,
        user_state: Dict[str, Any],
        user_history: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        """Classify current emotional-override level: 'red' | 'yellow' | 'normal'.

        Returns (level, reasons). `reasons` is a list of signal tags such as
        "pain_8" / "fatigue_7" / "skips_3" so pipeline can log them for later
        attribution analysis.
        """
        pain = user_state.get("pain_level", 0)
        fatigue = user_state.get("fatigue_level", 0)
        skips = user_history.get("consecutive_skips", 0)

        # Red first (strictly stronger thresholds).
        red = self.emotional_override["red"]["trigger_conditions"]
        red_reasons: List[str] = []
        if pain >= red["pain_gte"]:
            red_reasons.append(f"pain_{pain}")
        if fatigue >= red["fatigue_gte"]:
            red_reasons.append(f"fatigue_{fatigue}")
        if skips >= red["consecutive_skips_gte"]:
            red_reasons.append(f"skips_{skips}")
        if red_reasons:
            return "red", red_reasons

        # Yellow (weaker thresholds, but requires at least one signal).
        yellow = self.emotional_override["yellow"]["trigger_conditions"]
        yellow_reasons: List[str] = []
        if pain >= yellow["pain_gte"]:
            yellow_reasons.append(f"pain_{pain}")
        if fatigue >= yellow["fatigue_gte"]:
            yellow_reasons.append(f"fatigue_{fatigue}")
        if skips >= yellow["consecutive_skips_gte"]:
            yellow_reasons.append(f"skips_{skips}")
        if yellow_reasons:
            return "yellow", yellow_reasons

        return "normal", []

    # ---------------------------------------------------------------------
    # Tone selection — the one-stop decision function
    # ---------------------------------------------------------------------
    def _select_tone(
        self,
        user_profile: Dict[str, Any],
        day: int,
        user_state: Dict[str, Any],
        user_history: Dict[str, Any],
        safety_blocked: bool = False,
    ) -> str:
        """Return the tone label for the current prompt.

        Priority order (hard-coded, mirrors transition_rules.priority_order):
            1. safety_blocked      -> 'doctor_companion_hybrid'
            2. red light           -> 'companion'
            3. base tone (incl. transition-zone mixing)
            4. yellow light        -> '{base_tone}+companion' mixed marker
        """
        # Reset per-call scratch state.
        self._current_override_reasons = []

        # Priority 1: Safety trigger.
        if safety_blocked:
            self._current_override_reasons = ["safety_trigger"]
            return "doctor_companion_hybrid"

        # Priority 2: Red light override.
        level, reasons = self._check_emotional_state(user_state, user_history)
        if level == "red":
            self._current_override_reasons = list(reasons)
            return "companion"

        # Priority 3: Base tone (handles transition zones).
        base_tone = self._get_base_tone(user_profile, day)

        # Priority 4: Yellow light overlay on base tone.
        if level == "yellow":
            self._current_override_reasons = list(reasons)
            if base_tone == "companion":
                return "companion"
            return f"{base_tone}+companion"

        return base_tone

    # ---------------------------------------------------------------------
    # Daily Closing variant selector
    # ---------------------------------------------------------------------
    def _select_daily_closing_variant(
        self,
        day_log: Dict[str, Any],
        user_profile: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Pick the (variant_key, tone) pair for a Daily Closing.

        Branch order mirrors the Section 2.1 spec in v1.3.1 — the first
        matching condition wins (short-circuit evaluation).
        """
        if day_log.get("no_interaction", False):
            return "no_interaction", "companion"
        if day_log.get("safety_blocked_today", False):
            return "safety_followup", "companion"
        if day_log.get("user_skipped", False):
            return "skip_recovery", "companion"
        if (
            day_log.get("pain_delta", 0) > 0
            or day_log.get("fatigue_level", 0) >= 7
        ):
            return "concern", "companion"
        if (
            day_log.get("user_accepted", False)
            and day_log.get("pain_delta", 0) <= 0
        ):
            return "celebration", "coach"
        _, arc_meta = self._get_arc_context(user_profile, day_log["day"])
        return "neutral", arc_meta["default_tone"]

    # ---------------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------------
    def _resolve_actual_tone(self, tone_label: str) -> str:
        """Collapse mixed markers (e.g. 'coach+companion') to the tone that
        will actually be used for THIS single prompt by 50/50 sampling."""
        if "+companion" in tone_label:
            base = tone_label.split("+", 1)[0]
            return str(self.rng.choice([base, "companion"]))
        return tone_label

    def _inject_empathy_line(
        self,
        context: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Auto-fill `empathy_line` for Safety hybrid using personality."""
        if context.get("empathy_line") or user_profile is None:
            return context
        empathy_table = (
            self.templates["scenarios"]["safety_agent_trigger"]
            ["empathy_line_by_personality"]
        )
        personality = user_profile.get("personality", "motivated")
        return {
            **context,
            "empathy_line": empathy_table.get(personality, empathy_table["motivated"]),
        }

    def _apply_streak_suffix(
        self,
        rendered: str,
        tone: str,
        variant: str,
        user_profile: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        """Append streak suffix for inconsistent users on relevant closings."""
        if user_profile.get("personality") != "inconsistent":
            return rendered
        if variant not in ("celebration", "neutral"):
            return rendered
        suffix_table = self.templates["scenarios"]["daily_closing"]["inconsistent_streak_suffix"]
        suffix = suffix_table.get(tone)
        if not suffix:
            return rendered
        try:
            safe_ctx = {
                "streak": context.get("streak", context.get("streak_count", 0)),
                "target": context.get("target", 7),
                "name": context.get("name", "friend"),
                "streak_count": context.get("streak_count", 0),
            }
            return rendered + " " + suffix.format(**safe_ctx)
        except KeyError:
            return rendered

    def _render_prompt(
        self,
        scenario_key: str,
        tone: str,
        context: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render the final string.

        * Increments total_render_count once per call.
        * Validates _REQUIRED_KEYS; on miss, records a fallback and uses
          the per-scenario safe template.
        * Handles the Safety hybrid compound rendering.
        """
        self.total_render_count += 1

        # Auto-inject empathy_line for Safety hybrid before validation.
        if scenario_key == "safety_agent_trigger_hybrid":
            context = self._inject_empathy_line(context, user_profile)

        required = self._REQUIRED_KEYS.get(scenario_key, set())
        missing = required - set(context.keys())

        if missing:
            self.logger.warning(
                "[Narrator] Fallback triggered for %s, missing=%s",
                scenario_key, sorted(missing),
            )
            self.fallback_counter[scenario_key] = (
                self.fallback_counter.get(scenario_key, 0) + 1
            )
            template = self._FALLBACK_TEMPLATES.get(
                scenario_key, self._FALLBACK_TEMPLATES["_default"]
            )
            safe_ctx = {
                k: context.get(k, "[—]")
                for k in (required | {"name", "day"})
            }
            try:
                return template.format(**safe_ctx)
            except KeyError:
                return self._FALLBACK_TEMPLATES["_default"].format(
                    name=context.get("name", "friend")
                )

        # Normal path.
        scenarios = self.templates["scenarios"]

        if scenario_key == "safety_agent_trigger_hybrid":
            hybrid = scenarios["safety_agent_trigger"]["hybrid_doctor_companion"]
            return (
                hybrid["doctor_part"].format(**context)
                + "\n\n"
                + hybrid["companion_part"].format(**context)
            )

        if scenario_key.startswith("daily_closing_"):
            variant = scenario_key.replace("daily_closing_", "", 1)
            # Handle the three neutral sub-variants.
            if variant.startswith("neutral_"):
                tone_name = variant.split("_", 1)[1]
                template = scenarios["daily_closing"]["neutral"][tone_name]
            else:
                sub = scenarios["daily_closing"].get(variant, {})
                template = sub.get(tone) or next(iter(sub.values()), None)
                if template is None:
                    self.logger.warning(
                        "[Narrator] daily_closing variant missing: %s / %s",
                        variant, tone,
                    )
                    return self._FALLBACK_TEMPLATES.get(
                        scenario_key, self._FALLBACK_TEMPLATES["_default"]
                    ).format(name=context.get("name", "friend"))
            return template.format(**context)

        # Default: {scenario}.{tone} lookup.
        template = scenarios[scenario_key][tone]
        return template.format(**context)

    # ---------------------------------------------------------------------
    # Public entry points (used by pipeline)
    # ---------------------------------------------------------------------
    def speak(
        self,
        scenario: str,
        context: Dict[str, Any],
        user_profile: Dict[str, Any],
        day: int,
        user_state: Dict[str, Any],
        user_history: Dict[str, Any],
        safety_blocked: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        """High-level API. Returns (prompt_text, metadata).

        `metadata` carries everything the pipeline needs to populate
        prompt_level_log for Section 4 analysis.
        """
        tone_label = self._select_tone(
            user_profile=user_profile,
            day=day,
            user_state=user_state,
            user_history=user_history,
            safety_blocked=safety_blocked,
        )
        base_tone = self._get_base_tone(user_profile, day)
        actual_tone = self._resolve_actual_tone(tone_label)

        # Resolve scenario_key for scenarios that have per-tone sub-keys.
        if safety_blocked:
            if scenario != "safety_agent_trigger":
                self.logger.warning(
                    "[Narrator] safety_blocked=True received with "
                    "scenario='%s' (expected 'safety_agent_trigger'); "
                    "forcing hybrid rendering.", scenario,
                )
            scenario_key = "safety_agent_trigger_hybrid"
        else:
            scenario_key = scenario

        prompt = self._render_prompt(
            scenario_key=scenario_key,
            tone=actual_tone,
            context=context,
            user_profile=user_profile,
        )

        tone_label_for_log = self._compute_tone_label(tone_label, safety_blocked)
        metadata = {
            "scenario": scenario,
            "scenario_key_used": scenario_key,
            "tone_label": tone_label_for_log,
            "tone_actually_used": actual_tone,
            "tone_originally_planned": base_tone,
            "tone_override_cause": (
                self._current_override_reasons[0]
                if self._current_override_reasons else None
            ),
            "all_override_reasons": list(self._current_override_reasons),
        }
        return prompt, metadata

    def say_daily_closing(
        self,
        day_log: Dict[str, Any],
        user_profile: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Daily Closing entry point. Picks variant + tone + renders.

        NOTE: Daily Closing intentionally **bypasses** the yellow/red
        emotional override. The variant selector already encodes the
        emotional state via the ``concern`` / ``skip_recovery`` /
        ``safety_followup`` branches, so re-applying the override would
        double-count the state (empirically produced always-companion
        closings). ``tone_originally_planned`` is still logged for
        auditability.
        """
        variant, tone = self._select_daily_closing_variant(day_log, user_profile)
        if variant == "neutral":
            scenario_key = f"daily_closing_neutral_{tone}"
        else:
            scenario_key = f"daily_closing_{variant}"

        base_tone = self._get_base_tone(user_profile, day_log["day"])
        prompt = self._render_prompt(
            scenario_key=scenario_key,
            tone=tone,
            context=context,
            user_profile=user_profile,
        )
        prompt = self._apply_streak_suffix(
            rendered=prompt,
            tone=tone,
            variant=variant,
            user_profile=user_profile,
            context=context,
        )
        metadata = {
            "scenario": "daily_closing",
            "scenario_key_used": scenario_key,
            "daily_closing_variant": variant,
            "tone_label": tone,
            "tone_actually_used": tone,
            "tone_originally_planned": base_tone,
            "tone_override_cause": None,
            "all_override_reasons": [],
        }
        return prompt, metadata

    # ---------------------------------------------------------------------
    # Figure 4.3 — tone label classification for stacked bar
    # ---------------------------------------------------------------------
    @staticmethod
    def _compute_tone_label(tone_decision: str, safety_blocked: bool) -> str:
        """Map internal tone decision to the Figure 4.3 slice label.

        The 5 Figure 4.3 slices are mutually exclusive and jointly exhaustive:
            * doctor / coach / companion   (pure, non-override prompts)
            * yellow_mixed                 (yellow-light overlay)
            * safety_hybrid                (safety trigger prompt)
        """
        if safety_blocked:
            return "safety_hybrid"
        if "+companion" in tone_decision:
            return "yellow_mixed"
        return tone_decision  # doctor / coach / companion (incl. red forced)

    # ---------------------------------------------------------------------
    # Aggregate counters (consumed by pipeline._finalize_experiment)
    # ---------------------------------------------------------------------
    def export_aggregate_counters(self) -> Dict[str, Any]:
        """Snapshot for experiment-level log."""
        total = self.total_render_count or 1
        return {
            "narrator_total_prompts": self.total_render_count,
            "narrator_fallback_counts": dict(self.fallback_counter),
            "narrator_fallback_rate": sum(self.fallback_counter.values()) / total,
        }
