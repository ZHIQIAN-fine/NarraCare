"""User Simulator — generates synthetic user behaviour for offline
evaluation of the NarraCare pipeline.

Each simulated user has a *personality* (motivated / cautious / inconsistent)
that governs acceptance probability, skip rate and difficulty preference.

v1.1 — Step A: difficulty-aware scoring replaces broken intensity lookup.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Category constants (avoid string typos leading to dead code) ──────────
_GENTLE_CATEGORIES = frozenset({"flexibility", "rest"})
_HIGH_EXERTION_CATEGORIES = frozenset({"strength", "cardio"})
_ALL_CATEGORIES = frozenset({"flexibility", "rest", "strength", "balance", "cardio"})


class UserSimulator:
    """Simulates daily states, task choices and post-task feedback for
    synthetic users.

    Parameters
    ----------
    config : dict
        Full application config (parsed ``default.yaml``).
    profiles_path : str | Path | None
        Path to ``user_profiles.json``.  Defaults to
        ``data/user_profiles.json`` relative to the project root.
    """

    def __init__(
        self,
        config: dict,
        profiles_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.sim_cfg = config.get("simulation", {})
        self.seed: int = config.get("system", {}).get("seed", 42)
        self.rng = np.random.default_rng(self.seed)

        # Noise parameters
        self.pain_var: float = self.sim_cfg.get("pain_daily_variance", 1.5)
        self.fatigue_var: float = self.sim_cfg.get("fatigue_daily_variance", 2.0)

        # Personality → behaviour mappings
        self.personality_cfg: Dict[str, dict] = self.sim_cfg.get("personality", {})

        # Load user profiles
        if profiles_path is None:
            project_root = Path(__file__).resolve().parent.parent
            profiles_path = project_root / "data" / "user_profiles.json"
        else:
            profiles_path = Path(profiles_path)

        self.profiles: Dict[str, dict] = {}
        if profiles_path.exists():
            with open(profiles_path, "r", encoding="utf-8") as fh:
                profile_list: List[dict] = json.load(fh)
            for p in profile_list:
                self.profiles[p["user_id"]] = p
            logger.info("Loaded %d user profiles from %s", len(self.profiles), profiles_path)
        else:
            logger.warning("User profiles not found at %s", profiles_path)

        # Track evolving per-user state
        self._user_rom: Dict[str, float] = {}
        self._user_tasks_completed: Dict[str, int] = {}

    # ── Profile helpers ─────────────────────────────────────────────────

    @staticmethod
    def _pget(profile: dict, primary: str, *aliases: str, default: Any = 0) -> Any:
        val = profile.get(primary)
        if val is not None:
            return val
        for a in aliases:
            val = profile.get(a)
            if val is not None:
                return val
        return default

    @staticmethod
    def _stage_to_int(stage_raw: Any) -> int:
        _map = {"early_rehab": 1, "mid_rehab": 2, "late_rehab": 3,
                "return_to_sport": 4, "return_to_activity": 4, "maintenance": 4}
        if isinstance(stage_raw, str):
            return _map.get(stage_raw, 1)
        return int(stage_raw) if stage_raw else 1

    # ── Profile access ───────────────────────────────────────────────────

    def get_profile(self, user_id: str) -> dict:
        return self.profiles.get(user_id, {})

    def get_all_user_ids(self) -> List[str]:
        return list(self.profiles.keys())

    # ── Daily state simulation ───────────────────────────────────────────

    def simulate_daily_state(self, user_id: str, day: int) -> dict:
        profile = self.profiles.get(user_id, {})
        baseline_pain = self._pget(profile, "baseline_pain", "pain_baseline", default=4)
        baseline_fatigue = self._pget(profile, "baseline_fatigue", "fatigue_baseline", default=4)
        baseline_rom = self._pget(profile, "baseline_rom", "rom_current", default=90)

        pain_drift = -0.02 * day
        pain = baseline_pain + pain_drift + self.rng.normal(0, self.pain_var)
        pain = float(np.clip(pain, 0, 10))

        fatigue = baseline_fatigue + self.rng.normal(0, self.fatigue_var)
        fatigue = float(np.clip(fatigue, 0, 10))

        completed = self._user_tasks_completed.get(user_id, 0)
        rom_improvement = min(completed * 0.5, 60)
        current_rom = self._user_rom.get(user_id, baseline_rom)
        target_rom = baseline_rom + rom_improvement
        current_rom = current_rom + 0.3 * (target_rom - current_rom) + self.rng.normal(0, 2)
        current_rom = float(np.clip(current_rom, 0, 180))
        self._user_rom[user_id] = current_rom

        mood_base = 10 - (pain + fatigue) / 2
        mood = mood_base + self.rng.normal(0, 1)
        mood = float(np.clip(mood, 0, 10))

        complaints: List[str] = []
        if pain >= 6:
            complaints.append("soreness")
        if self.rng.random() < 0.1:
            complaints.append("stiffness")

        swelling = bool(self.rng.random() < 0.03)

        return {
            "day": day,
            "pain_level": round(pain, 1),
            "fatigue_level": round(fatigue, 1),
            "rom": round(current_rom, 1),
            "mood": round(mood, 1),
            "specific_complaints": complaints,
            "feedback_on_last_task": {},
            "swelling": swelling,
        }

    # ── Choice simulation ────────────────────────────────────────────────

    def simulate_choice(
        self,
        user_id: str,
        options: List[dict],
        state: dict,
    ) -> Tuple[str, dict]:
        if not options:
            return "", {"action": "skipped", "reason": "no_options"}

        profile = self.profiles.get(user_id, {})
        personality = profile.get("personality", "cautious")
        p_cfg = self.personality_cfg.get(personality, {})

        skip_prob = p_cfg.get("skip_prob", 0.1)
        prefer_higher = p_cfg.get("prefer_higher_difficulty", False)
        accept_challenge = p_cfg.get("accept_challenge_prob", 0.5)

        # ── Personality-based skip ──────────────────────────────────
        if self.rng.random() < skip_prob:
            chosen = options[0]
            return chosen["task_id"], {
                "action": "skipped",
                "chosen_task": chosen,
                "reason": "personality_skip",
            }

        # Rank options by difficulty, category, and body-part preference
        scored: List[Tuple[dict, float]] = []
        pain = state.get("pain_level", 5)
        fatigue = state.get("fatigue_level", 5)
        stress = float(np.clip((pain + fatigue) / 20.0, 0, 1))

        for task in options:
            difficulty = task.get("difficulty", 1)
            category = task.get("category", "")
            score = 0.0

            # ── Difficulty preference (personality-sensitive) ─────────
            if prefer_higher:
                ideal_diff = 2 + (1 - stress) * 2     # motivated: [2, 4]
            else:
                ideal_diff = max(1.0, 3 - stress * 3)  # cautious:  [1, 3]
            score -= abs(difficulty - ideal_diff) * 0.5

            # ── Category preference under high pain / fatigue ────────
            if pain > 5 and category in _GENTLE_CATEGORIES:
                score += 0.3
            if fatigue > 6 and category == "rest":
                score += 0.3

            # ── Challenge gate for hard tasks (difficulty ≥ 4) ───────
            if difficulty >= 4 and self.rng.random() > accept_challenge:
                score -= 1.5

            score += self.rng.normal(0, 0.3)  # noise
            scored.append((task, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        chosen_task = scored[0][0]

        return chosen_task["task_id"], {
            "action": "completed",
            "chosen_task": chosen_task,
            "reason": "preference",
        }

    # ── Feedback simulation ──────────────────────────────────────────────

    def simulate_feedback(
        self,
        user_id: str,
        task: dict,
        state: dict,
    ) -> dict:
        profile = self.profiles.get(user_id, {})
        personality = profile.get("personality", "cautious")

        difficulty = task.get("difficulty", 1)
        category = task.get("category", "")

        pain_before = state.get("pain_level", 5)
        fatigue_before = state.get("fatigue_level", 5)

        # --- Determine if completed (may still abandon mid-task)
        p_cfg = self.personality_cfg.get(personality, {})
        if personality == "inconsistent" and self.rng.random() < 0.15:
            completed = False
            action = "partial"
        else:
            completed = True
            action = "completed"

        # --- Pain change (difficulty-sensitive)
        if completed:
            pain_change = (difficulty - 3) * 0.3 + self.rng.normal(0, 0.5)
            if category in _GENTLE_CATEGORIES:
                pain_change -= 0.5
        else:
            pain_change = 0.2

        pain_change = float(np.clip(pain_change, -3, 3))

        # --- Fatigue change (difficulty-sensitive)
        fatigue_change = difficulty * 0.3 + self.rng.normal(0, 0.3)
        if not completed:
            fatigue_change *= 0.5
        fatigue_change = float(np.clip(fatigue_change, -1, 4))

        # --- ROM change
        rom_change = 0.0
        if completed:
            rom_change = 0.3 + self.rng.normal(0, 0.2)
            self._user_tasks_completed[user_id] = (
                self._user_tasks_completed.get(user_id, 0) + 1
            )
        rom_change = float(np.clip(rom_change, -0.5, 2.0))

        # --- Satisfaction
        if completed:
            stress = float(np.clip((pain_before + fatigue_before) / 20.0, 0, 1))
            ideal_difficulty = 1 + (1 - stress) * 4
            mismatch = abs(difficulty - ideal_difficulty) / 4.0
            satisfaction = max(0.0, 1.0 - mismatch) + self.rng.normal(0, 0.1)
            satisfaction = float(np.clip(satisfaction, 0, 1))
        else:
            satisfaction = 0.2

        return {
            "completed": completed,
            "action": action,
            "pain_change": round(pain_change, 2),
            "fatigue_change": round(fatigue_change, 2),
            "rom_change": round(rom_change, 2),
            "satisfaction": round(satisfaction, 3),
            "post_pain": round(float(np.clip(pain_before + pain_change, 0, 10)), 1),
            "post_fatigue": round(float(np.clip(fatigue_before + fatigue_change, 0, 10)), 1),
        }

    # ── Reward computation ───────────────────────────────────────────────

    def compute_reward(
        self,
        feedback: dict,
        state: dict,
        task: dict,
    ) -> float:
        weights = self.config.get("reward", {})
        w_completed = weights.get("w_completed", 0.40)
        w_safety    = weights.get("w_safety", 0.30)
        w_diff      = weights.get("w_difficulty_match", 0.15)
        w_pain      = weights.get("w_pain_penalty", 0.10)
        w_dropout   = weights.get("w_dropout_penalty", 0.05)

        r_completed = 1.0 if feedback.get("completed", False) else 0.0

        pain_change = feedback.get("pain_change", 0)
        r_safety = 1.0 if pain_change <= 1.0 else max(0.0, 1.0 - (pain_change - 1.0) * 0.3)

        # Difficulty match (using actual difficulty scale 1–5)
        difficulty = task.get("difficulty", 1)
        pain = state.get("pain_level", 5)
        fatigue = state.get("fatigue_level", 5)
        stress = float(np.clip((pain + fatigue) / 20.0, 0, 1))
        ideal_difficulty = 1 + (1 - stress) * 4
        mismatch = abs(difficulty - ideal_difficulty) / 4.0
        r_diff_match = max(0.0, 1.0 - mismatch)

        r_pain = max(0.0, 1.0 - max(0, pain_change) * 0.2)
        r_dropout = 0.0 if feedback.get("action") == "skipped" else 1.0

        reward = (
            w_completed * r_completed
            + w_safety * r_safety
            + w_diff * r_diff_match
            + w_pain * r_pain
            + w_dropout * r_dropout
        )

        return float(np.clip(reward, 0.0, 1.0))
