"""User Simulator — Expected-Utility Model (EU-Sim v2).

Generates synthetic user behaviour for offline evaluation of the NarraCare
pipeline.  Each simulated user has a *personality* (motivated / cautious /
inconsistent) that sets a utility threshold; acceptance is determined by a
sigmoid over the Expected Utility of the best recommended task.

EU components (5-dimensional):
  DifficultyFit     — Gaussian kernel around state-dependent ideal difficulty
  PreferenceMatch   — category + body-part relevance for the user
  Novelty           — diminishes with repeated task exposure
  PerceivedRisk     — sigmoid of pain × fatigue × difficulty × contraindication
  FatigueCost       — interaction of current fatigue and task demand

Reference model:
  U(user, task) = α_fit · DifficultyFit + α_pref · PreferenceMatch
                + α_novel · Novelty − β_risk · PerceivedRisk
                − γ_fatigue · FatigueCost
  P(accept) = σ(k · (U − U_threshold(personality)))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Module-level constants
# ═════════════════════════════════════════════════════════════════════════════

# Category sets (frozen to prevent accidental mutation and avoid typo bugs)
_GENTLE_CATEGORIES = frozenset({"flexibility", "rest"})
_HIGH_EXERTION_CATEGORIES = frozenset({"strength", "cardio"})
_ALL_CATEGORIES = frozenset({"flexibility", "rest", "strength", "balance", "cardio"})

# Condition → primary target body-part (for PreferenceMatch)
_CONDITION_BODY: Dict[str, str] = {
    "ACL_reconstruction": "knee",
    "rotator_cuff_repair": "shoulder",
    "chronic_lower_back_pain": "back",
}

# EU model — personality-specific acceptance thresholds
_EU_THRESHOLDS: Dict[str, float] = {
    "motivated": 0.15,
    "cautious": 0.30,
    "inconsistent": 0.22,
}
_SIGMOID_K = 8.0  # steepness of acceptance sigmoid

# EU model — component weights (sum ≈ 1.0 when all positive/negative parts cancel)
_ALPHA_FIT = 0.35
_ALPHA_PREF = 0.20
_ALPHA_NOVEL = 0.10
_BETA_RISK = 0.20
_GAMMA_FATIGUE = 0.15


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
        # EU model — per-user task exposure for Novelty component
        self._user_task_exposure: Dict[str, Dict[str, int]] = {}

    # ── Profile helpers ─────────────────────────────────────────────────

    @staticmethod
    def _pget(profile: dict, primary: str, *aliases: str, default: Any = 0) -> Any:
        """Read a profile field with fallback aliases for different schemas."""
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
        """Convert stage (int or string) to integer 1-4."""
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
        """Generate a daily state by adding random noise to the user's baseline.

        State evolves slowly: ROM improves if the user has been completing
        tasks; pain and fatigue fluctuate randomly around the baseline.

        Parameters
        ----------
        user_id : str
        day : int
            Simulation day (0-indexed).

        Returns
        -------
        dict
            Keys: pain_level, fatigue_level, rom, mood, day,
                  specific_complaints, feedback_on_last_task, swelling.
        """
        profile = self.profiles.get(user_id, {})
        baseline_pain = self._pget(profile, "baseline_pain", "pain_baseline", default=4)
        baseline_fatigue = self._pget(profile, "baseline_fatigue", "fatigue_baseline", default=4)
        baseline_rom = self._pget(profile, "baseline_rom", "rom_current", default=90)

        # --- Pain: fluctuates around baseline with slight improvement over time
        pain_drift = -0.02 * day  # slow improvement
        pain = baseline_pain + pain_drift + self.rng.normal(0, self.pain_var)
        pain = float(np.clip(pain, 0, 10))

        # --- Fatigue: fluctuates more
        fatigue = baseline_fatigue + self.rng.normal(0, self.fatigue_var)
        fatigue = float(np.clip(fatigue, 0, 10))

        # --- ROM: improves if user has been completing tasks
        completed = self._user_tasks_completed.get(user_id, 0)
        rom_improvement = min(completed * 0.5, 60)  # cap improvement at +60°
        current_rom = self._user_rom.get(user_id, baseline_rom)
        target_rom = baseline_rom + rom_improvement
        # Smooth towards target
        current_rom = current_rom + 0.3 * (target_rom - current_rom) + self.rng.normal(0, 2)
        current_rom = float(np.clip(current_rom, 0, 180))
        self._user_rom[user_id] = current_rom

        # --- Mood: correlated with pain/fatigue
        mood_base = 10 - (pain + fatigue) / 2
        mood = mood_base + self.rng.normal(0, 1)
        mood = float(np.clip(mood, 0, 10))

        # --- Occasional complaints
        complaints: List[str] = []
        if pain >= 6:
            complaints.append("soreness")
        if self.rng.random() < 0.1:
            complaints.append("stiffness")

        # --- Swelling (rare)
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

    # ═════════════════════════════════════════════════════════════════════
    # Expected-Utility Model
    # ═════════════════════════════════════════════════════════════════════

    def _compute_utility(self, task: dict, state: dict, profile: dict) -> float:
        """Five-component Expected Utility for a single (user, task) pair.

        Returns a scalar U ∈ roughly [-0.3, 0.65].
        """
        difficulty = task.get("difficulty", 1)
        category = task.get("category", "")
        body_part = task.get("body_part", "")
        task_id = task.get("task_id", "")
        uid = profile.get("user_id", "")

        pain = state.get("pain_level", 5)
        fatigue = state.get("fatigue_level", 5)
        stress = float(np.clip((pain + fatigue) / 20.0, 0, 1))

        # ── 1. DifficultyFit — Gaussian kernel ──────────────────────
        ideal_diff = 1 + (1 - stress) * 4          # [1, 5]
        sigma_fit = 1.5
        difficulty_fit = float(np.exp(
            -((difficulty - ideal_diff) ** 2) / (2 * sigma_fit ** 2)
        ))

        # ── 2. PreferenceMatch — category + body-part ───────────────
        pref = 0.5                                  # neutral baseline
        if pain > 5 and category in _GENTLE_CATEGORIES:
            pref += 0.20
        elif pain > 5 and category in _HIGH_EXERTION_CATEGORIES:
            pref -= 0.20
        if fatigue > 6 and category == "rest":
            pref += 0.15
        condition = profile.get("condition", "")
        target_bp = _CONDITION_BODY.get(condition, "")
        if body_part == target_bp or body_part == "full_body":
            pref += 0.15
        pref = float(np.clip(pref, 0, 1))

        # ── 3. Novelty — decays with repeated exposure ──────────────
        exposure = self._user_task_exposure.get(uid, {})
        seen_count = exposure.get(task_id, 0)
        novelty = 1.0 - min(1.0, seen_count / 3.0)

        # ── 4. PerceivedRisk — sigmoid of weighted risk factors ─────
        user_contras = set(profile.get("contraindications", []))
        task_contras = set(task.get("contraindications", []))
        contra_flag = 1.0 if (user_contras & task_contras) else 0.0
        risk_input = (0.3 * (pain / 10.0) + 0.2 * (fatigue / 10.0)
                      + 0.3 * (difficulty / 5.0) + 0.2 * contra_flag)
        perceived_risk = float(1.0 / (1.0 + np.exp(-5 * (risk_input - 0.5))))

        # ── 5. FatigueCost — current fatigue × task demand ──────────
        fatigue_cost = float(fatigue * difficulty / 50.0)

        # ── Weighted sum ────────────────────────────────────────────
        U = (_ALPHA_FIT * difficulty_fit
             + _ALPHA_PREF * pref
             + _ALPHA_NOVEL * novelty
             - _BETA_RISK * perceived_risk
             - _GAMMA_FATIGUE * fatigue_cost)

        return U

    # ── Choice simulation ────────────────────────────────────────────────

    def simulate_choice(
        self,
        user_id: str,
        options: List[dict],
        state: dict,
    ) -> Tuple[str, dict]:
        """Simulate the user choosing one task from the recommended options.

        The user evaluates each option's Expected Utility, picks the best,
        then accepts/rejects via a sigmoid gate parametrised by personality.

        Parameters
        ----------
        user_id : str
        options : list[dict]
            Recommended tasks (each has ``task_id``, ``difficulty``, etc.).
        state : dict
            Current daily state.

        Returns
        -------
        (chosen_task_id, choice_metadata)
            ``choice_metadata`` includes ``action`` ("completed" | "skipped"),
            ``utility``, ``p_accept``, and ``chosen_task`` dict.
        """
        if not options:
            return "", {"action": "skipped", "reason": "no_options"}

        profile = self.profiles.get(user_id, {})
        personality = profile.get("personality", "cautious")

        # ── Compute utility for each option ─────────────────────────
        utilities: List[Tuple[dict, float]] = []
        for task in options:
            u = self._compute_utility(task, state, profile)
            u += self.rng.normal(0, 0.05)  # perception noise
            utilities.append((task, u))

        utilities.sort(key=lambda x: x[1], reverse=True)
        best_task, best_u = utilities[0]

        # ── Accept / reject via sigmoid ─────────────────────────────
        u_threshold = _EU_THRESHOLDS.get(personality, 0.25)
        p_accept = float(
            1.0 / (1.0 + np.exp(-_SIGMOID_K * (best_u - u_threshold)))
        )

        # Inconsistent personality: inject extra stochasticity
        if personality == "inconsistent":
            p_accept = 0.6 * p_accept + 0.4 * self.rng.random()

        if self.rng.random() < p_accept:
            # ── Accept: update exposure tracker ─────────────────────
            tid = best_task.get("task_id", "")
            if user_id not in self._user_task_exposure:
                self._user_task_exposure[user_id] = {}
            self._user_task_exposure[user_id][tid] = (
                self._user_task_exposure[user_id].get(tid, 0) + 1
            )
            return best_task["task_id"], {
                "action": "completed",
                "chosen_task": best_task,
                "utility": float(best_u),
                "p_accept": float(p_accept),
                "reason": "utility_accept",
            }
        else:
            return best_task["task_id"], {
                "action": "skipped",
                "chosen_task": best_task,
                "utility": float(best_u),
                "p_accept": float(p_accept),
                "reason": "utility_reject",
            }

    # ── Feedback simulation ──────────────────────────────────────────────

    def simulate_feedback(
        self,
        user_id: str,
        task: dict,
        state: dict,
    ) -> dict:
        """Generate post-task feedback based on user personality, state and
        **task attributes** (difficulty, category, body_part).

        Parameters
        ----------
        user_id : str
        task : dict
            The task that was performed (or skipped).
        state : dict
            State *before* the task.

        Returns
        -------
        dict
            Keys: completed (bool), pain_change (float), fatigue_change (float),
            rom_change (float), satisfaction (float 0-1), action (str).
        """
        profile = self.profiles.get(user_id, {})
        personality = profile.get("personality", "cautious")

        difficulty = task.get("difficulty", 1)
        category = task.get("category", "")
        body_part = task.get("body_part", "")

        pain_before = state.get("pain_level", 5)
        fatigue_before = state.get("fatigue_level", 5)

        # --- Completion (harder tasks have higher partial-abandon rate)
        partial_prob = 0.02 * difficulty          # 2%–10% base
        if personality == "inconsistent":
            partial_prob += 0.10
        elif personality == "cautious" and difficulty >= 4:
            partial_prob += 0.05

        if self.rng.random() < partial_prob:
            completed = False
            action = "partial"
        else:
            completed = True
            action = "completed"

        # --- Pain change (difficulty + category sensitive)
        if completed:
            pain_change = (difficulty - 3) * 0.3 + self.rng.normal(0, 0.4)
            if category in _GENTLE_CATEGORIES:
                pain_change -= 0.5
            elif category in _HIGH_EXERTION_CATEGORIES:
                pain_change += 0.15
        else:
            pain_change = 0.1  # slight increase from frustration

        pain_change = float(np.clip(pain_change, -3, 3))

        # --- Fatigue change (difficulty + category sensitive)
        fatigue_change = difficulty * 0.25 + self.rng.normal(0, 0.3)
        if category == "rest":
            fatigue_change -= 0.3
        elif category == "cardio":
            fatigue_change += 0.3
        if not completed:
            fatigue_change *= 0.5
        fatigue_change = float(np.clip(fatigue_change, -1, 4))

        # --- ROM change (small improvement if completed)
        rom_change = 0.0
        if completed:
            rom_change = 0.3 + self.rng.normal(0, 0.2)
            self._user_tasks_completed[user_id] = (
                self._user_tasks_completed.get(user_id, 0) + 1
            )
        rom_change = float(np.clip(rom_change, -0.5, 2.0))

        # --- Satisfaction (difficulty match + body-part relevance)
        if completed:
            stress = float(np.clip(
                (pain_before + fatigue_before) / 20.0, 0, 1
            ))
            ideal_diff = 1 + (1 - stress) * 4      # [1, 5]
            mismatch = abs(difficulty - ideal_diff) / 4.0
            satisfaction = max(0.0, 1.0 - mismatch)
            # Body-part relevance bonus
            condition = profile.get("condition", "")
            target_bp = _CONDITION_BODY.get(condition, "")
            if body_part == target_bp or body_part == "full_body":
                satisfaction += 0.10
            satisfaction += self.rng.normal(0, 0.1)
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
            "post_pain": round(
                float(np.clip(pain_before + pain_change, 0, 10)), 1
            ),
            "post_fatigue": round(
                float(np.clip(fatigue_before + fatigue_change, 0, 10)), 1
            ),
        }

    # ── Reward computation ───────────────────────────────────────────────

    def compute_reward(
        self,
        feedback: dict,
        state: dict,
        task: dict,
    ) -> float:
        """Compute scalar reward from feedback.

        Uses a Gaussian-kernel difficulty match and category-appropriateness
        component so that the reward is **task-sensitive** (not constant
        across all tasks for the same user-day).

        Parameters
        ----------
        feedback : dict
            Output of ``simulate_feedback``.
        state : dict
            State at recommendation time.
        task : dict
            The recommended task.

        Returns
        -------
        float
            Reward in [0, 1].
        """
        weights = self.config.get("reward", {})
        w_completed = weights.get("w_completed", 0.35)
        w_safety    = weights.get("w_safety", 0.25)
        w_diff      = weights.get("w_difficulty_match", 0.20)
        w_pain      = weights.get("w_pain_penalty", 0.10)
        w_dropout   = weights.get("w_dropout_penalty", 0.05)
        w_category  = weights.get("w_category", 0.05)

        # ── r_completed ─────────────────────────────────────────────
        r_completed = 1.0 if feedback.get("completed", False) else 0.0

        # ── r_safety — 1 if pain did not spike badly ────────────────
        pain_change = feedback.get("pain_change", 0)
        r_safety = (
            1.0 if pain_change <= 1.0
            else max(0.0, 1.0 - (pain_change - 1.0) * 0.3)
        )

        # ── r_diff_match — Gaussian kernel on difficulty ────────────
        difficulty = task.get("difficulty", 1)
        category = task.get("category", "")
        pain = state.get("pain_level", 5)
        fatigue = state.get("fatigue_level", 5)
        stress = float(np.clip((pain + fatigue) / 20.0, 0, 1))
        ideal_diff = 1 + (1 - stress) * 4          # [1, 5]
        sigma = 1.5
        r_diff = float(np.exp(
            -((difficulty - ideal_diff) ** 2) / (2 * sigma ** 2)
        ))

        # ── r_category — appropriateness of category for user state ─
        if pain > 5:
            r_cat = 0.9 if category in _GENTLE_CATEGORIES else 0.3
        elif fatigue > 6:
            r_cat = 0.9 if category == "rest" else 0.4
        else:
            r_cat = 0.6  # neutral

        # ── r_pain ──────────────────────────────────────────────────
        r_pain = max(0.0, 1.0 - max(0, pain_change) * 0.2)

        # ── r_dropout ───────────────────────────────────────────────
        r_dropout = 0.0 if feedback.get("action") == "skipped" else 1.0

        reward = (
            w_completed * r_completed
            + w_safety * r_safety
            + w_diff * r_diff
            + w_pain * r_pain
            + w_dropout * r_dropout
            + w_category * r_cat
        )

        return float(np.clip(reward, 0.0, 1.0))
