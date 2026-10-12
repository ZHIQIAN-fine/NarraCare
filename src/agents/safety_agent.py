"""Safety Agent — enforces safety constraints, detects crises, and gates
stage progression in the rehabilitation pipeline.

Rules are loaded from ``data/safety_rules.json`` and augmented by thresholds
from the application config (``default.yaml``).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SafetyAgent:
    """Applies pre-filter, post-check, crisis detection and progression gating.

    Parameters
    ----------
    config : dict
        Full application config (parsed ``default.yaml``).
    rules_path : str | Path | None
        Path to ``safety_rules.json``.  When *None* the default
        ``data/safety_rules.json`` relative to the project root is used.
    """

    def __init__(
        self,
        config: dict,
        rules_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.safety_cfg = config.get("safety", {})
        self.enabled = self.safety_cfg.get("enable", True)

        # Resolve rules path
        if rules_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            rules_path = project_root / "data" / "safety_rules.json"
        else:
            rules_path = Path(rules_path)

        self.rules: Dict[str, Any] = {}
        if rules_path.exists():
            with open(rules_path, "r", encoding="utf-8") as fh:
                self.rules = json.load(fh)
            logger.info("Loaded safety rules from %s", rules_path)
        else:
            logger.warning("Safety rules file not found at %s — using defaults.", rules_path)

        # Config-level thresholds
        self.pain_threshold_high: int = self.safety_cfg.get("pain_threshold_high", 7)
        self.fatigue_threshold_high: int = self.safety_cfg.get("fatigue_threshold_high", 8)
        self.max_consecutive_high: int = self.safety_cfg.get("max_consecutive_high_intensity", 3)

        # Crisis keywords (merge config + rules file — supports both formats)
        self.crisis_keywords: List[str] = list(self.safety_cfg.get("crisis_keywords", []))
        # Format A: flat keyword lists
        self.crisis_keywords += self.rules.get("crisis_keywords_en", [])
        self.crisis_keywords += self.rules.get("crisis_keywords_zh", [])
        # Format B: structured crisis_rules list (existing data)
        for cr in self.rules.get("crisis_rules", []):
            self.crisis_keywords += cr.get("keywords", [])
        # De-dup while keeping order
        seen: set = set()
        deduped: List[str] = []
        for kw in self.crisis_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                deduped.append(kw)
        self.crisis_keywords = deduped

        # Crisis rules with messages (for richer response)
        self._crisis_rules: List[dict] = self.rules.get("crisis_rules", [])

    # ── Pre-filtering ────────────────────────────────────────────────────

    def pre_filter(
        self,
        candidates: List[dict],
        user_state: dict,
        user_profile: dict,
    ) -> List[dict]:
        """Remove candidate tasks that violate safety contraindications.

        Checks applied (any one blocks the task):
          1. **Contraindication match** – task's contraindication tags overlap
             with the user's personal contraindication list.
          2. **Stage mismatch** – task's ``applicable_stages`` does not include
             the user's current rehabilitation stage.
          3. **High pain + high difficulty** – pain ≥ threshold and task
             difficulty ≥ 4.
          4. **Severe fatigue** – fatigue ≥ threshold and task is not rest /
             flexibility / breathing.
          5. **Low ROM** – ROM < 60 and task difficulty ≥ 3 or high-impact.

        Parameters
        ----------
        candidates : list[dict]
        user_state : dict
        user_profile : dict

        Returns
        -------
        list[dict]
            Filtered list of safe candidates.
        """
        if not self.enabled:
            return candidates

        safe: List[dict] = []
        pain = user_state.get("pain_level", 0)
        fatigue = user_state.get("fatigue_level", 0)
        rom = user_state.get("rom", 180)

        # User's stage (try several field names)
        user_stage = user_profile.get("current_stage",
                     user_profile.get("stage", ""))

        # User's personal contraindications
        user_contras = set(user_profile.get("contraindications", []))

        for task in candidates:
            category = task.get("category", "").lower()
            difficulty = task.get("difficulty", 1)
            task_contras = set(task.get("contraindications", []))
            task_stages = task.get("applicable_stages", [])
            blocked = False
            reason = ""

            # Rule 1: Contraindication match
            overlap = user_contras & task_contras
            if overlap:
                blocked = True
                reason = f"contraindication overlap: {overlap}"

            # Rule 2: Stage mismatch
            if task_stages and user_stage and user_stage not in task_stages:
                blocked = True
                reason = f"stage mismatch: user={user_stage}, task={task_stages}"

            # Rule 3: High pain → block high difficulty
            if pain >= self.pain_threshold_high and difficulty >= 4:
                blocked = True
                reason = f"pain {pain} >= {self.pain_threshold_high} + difficulty {difficulty}"

            # Rule 4: Severe fatigue → only rest/flexibility
            if fatigue >= self.fatigue_threshold_high and category not in ("rest", "flexibility"):
                blocked = True
                reason = f"fatigue {fatigue} >= {self.fatigue_threshold_high} + category={category}"

            # Rule 5: Limited ROM → block difficult/high-impact tasks
            if rom < 60 and (difficulty >= 3 or category == "cardio"):
                blocked = True
                reason = f"low ROM {rom} + difficulty {difficulty}"

            if blocked:
                logger.debug("Blocked %s — %s", task.get("task_id"), reason)
            else:
                safe.append(task)

        logger.info("Pre-filter: %d/%d candidates passed.", len(safe), len(candidates))
        return safe

    # ── Post-check ───────────────────────────────────────────────────────

    def post_check(
        self,
        ranked_tasks: List[dict],
        user_state: dict,
        consecutive_high_intensity: int = 0,
    ) -> List[dict]:
        """Apply dosage and consecutive-intensity checks on already-ranked tasks.

        Parameters
        ----------
        ranked_tasks : list[dict]
            Ordered list (best first) of recommended tasks.
        user_state : dict
            Current daily state.
        consecutive_high_intensity : int
            Number of consecutive days the user has done high-intensity work.

        Returns
        -------
        list[dict]
            Potentially re-ordered / trimmed list.
        """
        if not self.enabled:
            return ranked_tasks

        dosage = self.rules.get("dosage_limits", {})
        max_daily_high = dosage.get("max_daily_high_intensity", 2)
        max_daily = dosage.get("max_daily_exercises", 5)

        checked: List[dict] = []
        high_count = 0

        for task in ranked_tasks:
            if len(checked) >= max_daily:
                break

            intensity = task.get("intensity", "low").lower()

            # Consecutive high-intensity day cap
            if intensity == "high":
                if consecutive_high_intensity >= self.max_consecutive_high:
                    logger.info(
                        "Downgrading %s — %d consecutive high-intensity days.",
                        task.get("task_id"),
                        consecutive_high_intensity,
                    )
                    continue  # skip this high-intensity task
                if high_count >= max_daily_high:
                    continue
                high_count += 1

            checked.append(task)

        return checked

    # ── Crisis detection ─────────────────────────────────────────────────

    def check_crisis(self, user_input: str) -> Tuple[bool, str]:
        """Keyword-based crisis / red-flag detection.

        Parameters
        ----------
        user_input : str
            Raw user text (any language).

        Returns
        -------
        (is_crisis, message)
            ``is_crisis`` is True when a red-flag keyword is detected.
            ``message`` is a user-facing warning.
        """
        text_lower = user_input.lower()

        # Try structured crisis rules first (richer messages)
        if self._crisis_rules:
            for rule in self._crisis_rules:
                for kw in rule.get("keywords", []):
                    if kw.lower() in text_lower:
                        message = rule.get("message", "")
                        if not message:
                            message = (
                                "⚠️  Safety alert: a concern was detected. "
                                "Please stop exercising and contact your healthcare provider."
                            )
                        return True, message

        # Fallback: flat keyword list
        triggered: List[str] = []
        for kw in self.crisis_keywords:
            if kw.lower() in text_lower:
                triggered.append(kw)

        if triggered:
            message = (
                "⚠️  Safety alert: the following concern(s) were detected — "
                f"{', '.join(triggered)}. "
                "Please stop exercising immediately and contact your healthcare provider."
            )
            return True, message

        return False, ""

    # ── Progression gating ───────────────────────────────────────────────

    def check_progression(
        self,
        user_profile: dict,
        history: dict,
    ) -> Tuple[bool, str]:
        """Determine whether the user qualifies to advance to the next stage.

        Supports two JSON rule formats:
        - **Format A** (simple): ``stage_progression`` dict keyed by ``"1_to_2"`` etc.
        - **Format B** (rich):  ``progression_gates`` list with ``from_stage``,
          ``to_stage``, ``condition``, and ``requirements``.

        Parameters
        ----------
        user_profile : dict
            Must include ``stage``, ``days_since_surgery``.
            May include ``surgery_type`` / ``condition`` for format-B matching.
        history : dict
            Must include ``avg_pain_7d``, ``current_rom``,
            ``completion_rate_7d``.

        Returns
        -------
        (can_advance, message)
        """
        stage = user_profile.get("stage", 1)
        days = user_profile.get("days_since_surgery", 0)
        avg_pain = history.get("avg_pain_7d", 10)
        current_rom = history.get("current_rom", 0)
        completion_rate = history.get("completion_rate_7d", 0.0)
        condition = user_profile.get("surgery_type", user_profile.get("condition", ""))

        # ── Format A: simple dict ke──────────────────────────────────────
        progression_simple = self.rules.get("stage_progression", {})
        key = f"{stage}_to_{stage + 1}"
        rule_a = progression_simple.get(key)

        if rule_a is not None:
            return self._evaluate_simple_progression(
                rule_a, stage, days, avg_pain, current_rom, completion_rate
            )

        # ── Format B: progression_gates list ─────────────────────────────
        stage_name_map = {1: "early_rehab", 2: "mid_rehab", 3: "late_rehab", 4: "return_to_sport"}
        from_stage_name = stage_name_map.get(stage, f"stage_{stage}")

        gates: List[dict] = self.rules.get("progression_gates", [])
        matching_gate: Optional[dict] = None

        for gate in gates:
            if gate.get("from_stage") == from_stage_name:
                # If condition specified, prefer matching condition
                gate_cond = gate.get("condition", "")
                if gate_cond and condition and gate_cond == condition:
                    matching_gate = gate
                    break
                elif not matching_gate:
                    matching_gate = gate  # first match as fallback

        if matching_gate is not None:
            return self._evaluate_gate_progression(
                matching_gate, stage, days, avg_pain, current_rom, completion_rate
            )

        return False, f"No progression rule defined for stage {stage} → {stage + 1}."

    def _evaluate_simple_progression(
        self,
        rule: dict,
        stage: int,
        days: int,
        avg_pain: float,
        current_rom: float,
        completion_rate: float,
    ) -> Tuple[bool, str]:
        """Evaluate a simple-format progression rule."""
        reasons: List[str] = []
        can_advance = True

        if days < rule["min_days"]:
            can_advance = False
            reasons.append(f"days_since_surgery ({days}) < required ({rule['min_days']})")
        if current_rom < rule["min_rom"]:
            can_advance = False
            reasons.append(f"ROM ({current_rom}°) < required ({rule['min_rom']}°)")
        if avg_pain > rule["max_pain"]:
            can_advance = False
            reasons.append(f"avg pain ({avg_pain}) > max allowed ({rule['max_pain']})")
        if completion_rate < rule["min_completion_rate"]:
            can_advance = False
            reasons.append(f"completion rate ({completion_rate:.0%}) < required ({rule['min_completion_rate']:.0%})")

        if can_advance:
            msg = f"✅ User qualifies to advance from stage {stage} to stage {stage + 1}."
        else:
            msg = (
                f"❌ Not ready for stage {stage + 1}. "
                f"Unmet criteria: {'; '.join(reasons)}."
            )
        return can_advance, msg

    def _evaluate_gate_progression(
        self,
        gate: dict,
        stage: int,
        days: int,
        avg_pain: float,
        current_rom: float,
        completion_rate: float,
    ) -> Tuple[bool, str]:
        """Evaluate a progression_gates-format rule."""
        reqs = gate.get("requirements", {})
        reasons: List[str] = []
        can_advance = True

        min_days = reqs.get("min_days", 0)
        if days < min_days:
            can_advance = False
            reasons.append(f"days_since_surgery ({days}) < required ({min_days})")

        min_rom = reqs.get("min_rom", 0)
        if min_rom and current_rom < min_rom:
            can_advance = False
            reasons.append(f"ROM ({current_rom}°) < required ({min_rom}°)")

        max_pain = reqs.get("max_pain_avg_7d", 10)
        if avg_pain > max_pain:
            can_advance = False
            reasons.append(f"avg pain ({avg_pain}) > max allowed ({max_pain})")

        min_cr = reqs.get("min_completion_rate_7d", 0)
        if completion_rate < min_cr:
            can_advance = False
            reasons.append(f"completion rate ({completion_rate:.0%}) < required ({min_cr:.0%})")

        to_stage = gate.get("to_stage", f"stage {stage + 1}")
        if can_advance:
            msg = f"✅ User qualifies to advance from {gate.get('from_stage', stage)} to {to_stage}."
        else:
            msg = (
                f"❌ Not ready for {to_stage}. "
                f"Unmet criteria: {'; '.join(reasons)}."
            )
        return can_advance, msg
