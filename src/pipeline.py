"""NarraCare Pipeline — main orchestration module for the rehabilitation
recommender system.

Supports two modes:
  • **simulation**: all user behaviour is synthesised by ``UserSimulator``.
  • **interactive**: a human (or LLM proxy) provides natural-language input.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from src.agents.user_understanding import UserUnderstandingAgent, FEATURE_DIM
from src.agents.safety_agent import SafetyAgent
from src.agents.recommender import RecommenderAgent
from src.agents.narrator_agent import NarratorAgent
from src.agents.narrator import Narrator          # P1 — v1.3.2 Narrator
from src.memory import UserMemory
from src.simulator import UserSimulator

logger = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    """Set seed for all random generators used in the project."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# ── Difficulty → intensity mapping ────────────────────────────────────────
_DIFFICULTY_TO_INTENSITY: Dict[int, str] = {
    1: "low",
    2: "low",
    3: "medium",
    4: "high",
    5: "high",
}


# ── SAPS safety-posterior S2 signal threshold ─────────────────────────────
# Aligned with r_safety threshold in simulator.py (compute_reward).
# DO NOT tune post-hoc based on results — honest-outcome guardrail.
SAPS_PAIN_SAFE_THRESHOLD = 1.0


def _resolve_project_root() -> Path:
    """Return the NarraCare project root directory."""
    return Path(__file__).resolve().parent.parent


# ══════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════


class NarraCarePipeline:
    """End-to-end orchestration pipeline for NarraCare.

    Parameters
    ----------
    config_path : str
        Path (relative to project root) to the YAML configuration file.
    config_overrides : dict | None
        Nested dict that is deep-merged on top of the loaded config.
        Useful for experiments (e.g. switching algorithm).
    use_memory : bool
        If *False*, memory-derived features (completion rate, consecutive
        high-intensity days) are zeroed out — used in the *NoMemory*
        ablation.
    """

    def __init__(
        self,
        config_path: str = "configs/default.yaml",
        config_overrides: Optional[dict] = None,
        use_memory: bool = True,
    ) -> None:
        self.project_root = _resolve_project_root()
        self.use_memory = use_memory

        # ── Load & patch config ──────────────────────────────────────
        cfg_file = self.project_root / config_path
        with open(cfg_file, "r", encoding="utf-8") as fh:
            self.config: dict = yaml.safe_load(fh)

        if config_overrides:
            self._deep_update(self.config, config_overrides)

        seed = self.config.get("system", {}).get("seed", 42)
        self.rng = np.random.default_rng(seed)
        # Unify all random seeds for reproducibility
        _set_global_seed(seed)

        # ── Agents ───────────────────────────────────────────────────
        self.user_understanding = UserUnderstandingAgent(self.config, mode="simulation")
        self.safety_agent = SafetyAgent(
            self.config,
            drop_rule=config_overrides.get("safety", {}).get("drop_rule")
            if config_overrides else None,
        )
        # Observe all five runtime predicates even in NoSafety/one-drop runs.
        # This auditor never filters recommendations or changes model state.
        audit_config = copy.deepcopy(self.config)
        audit_config.setdefault("safety", {})["enable"] = True
        self._safety_auditor = SafetyAgent(audit_config)
        self.recommender = RecommenderAgent(self.config)
        # Old narrator — kept for backward compatibility (interactive mode)
        self.narrator_agent = NarratorAgent(self.config, mode="simulation")
        # P2 — New Narrator v1.3.2
        self.narrator = Narrator(
            arcs_path=str(self.project_root / "data" / "narrative_arcs.json"),
            templates_path=str(self.project_root / "data" / "narrator_tone_templates.json"),
            seed=seed,
        )
        # P5 — prompt-level log for experiment analysis
        self._prompt_level_log: List[dict] = []
        # P3 — Mutex flags: prevent double narrator calls within same turn/day
        self._narrator_spoke_this_turn: bool = False
        self._narrator_closed_this_turn: bool = False

        # ── Memory & Simulator ───────────────────────────────────────
        self.memory = UserMemory(self.config)
        self.simulator = UserSimulator(self.config)

        # ── Task library ─────────────────────────────────────────────
        task_lib_path = self.project_root / "data" / "task_library.json"
        with open(task_lib_path, "r", encoding="utf-8") as fh:
            self.task_library: List[dict] = json.load(fh)

        # Augment every task with an 'intensity' label derived from difficulty
        for task in self.task_library:
            if "intensity" not in task:
                task["intensity"] = _DIFFICULTY_TO_INTENSITY.get(
                    task.get("difficulty", 1), "low"
                )

        self.task_lookup: Dict[str, dict] = {
            t["task_id"]: t for t in self.task_library
        }

        # P10 — Sanity check
        self._narrator_sanity_check()

        logger.info(
            "Pipeline ready — %d tasks, algorithm=%s, safety=%s, narrator=%s, narrator_v2=%s",
            len(self.task_library),
            self.config.get("recommender", {}).get("algorithm", "linucb"),
            self.safety_agent.enabled,
            self.config.get("narrator", {}).get("enable", True),
            self.config.get("narrator_v2", {}).get("enable", True),
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _deep_update(base: dict, overrides: dict) -> dict:
        """Recursively merge *overrides* into *base* (in-place)."""
        for key, val in overrides.items():
            if isinstance(val, dict) and isinstance(base.get(key), dict):
                NarraCarePipeline._deep_update(base[key], val)
            else:
                base[key] = val
        return base

    def _normalize_profile(self, profile: dict) -> dict:
        """Return a copy of *profile* with normalised field names so that
        all downstream agents can consume it without alias gymnastics."""
        p = dict(profile)
        # stage
        if "current_stage" in p and "stage" not in p:
            p["stage"] = UserSimulator._stage_to_int(p["current_stage"])
        elif "stage" not in p:
            p["stage"] = 1
        # pain / fatigue baseline
        if "pain_baseline" in p and "baseline_pain" not in p:
            p["baseline_pain"] = p["pain_baseline"]
        if "fatigue_baseline" in p and "baseline_fatigue" not in p:
            p["baseline_fatigue"] = p["fatigue_baseline"]
        # ROM
        if "rom_current" in p and "baseline_rom" not in p:
            p["baseline_rom"] = p["rom_current"] if p["rom_current"] is not None else 90
        return p

    def _get_memory_features(self, user_id: str) -> Tuple[float, int]:
        """Return (completion_rate_7d, consecutive_high_intensity) — or
        zeros when memory is disabled."""
        if not self.use_memory:
            return 0.0, 0
        return (
            self.memory.get_completion_rate(user_id, 7),
            self.memory.get_consecutive_high_intensity(user_id),
        )

    # ── P12 — Boundary helper ────────────────────────────────────────

    def _narrator_user_state(self, state: dict) -> dict:
        """Ensure narrator gets pain_level/fatigue_level regardless of field names."""
        return {
            "pain_level": state.get("pain_level", state.get("pain", 0)),
            "fatigue_level": state.get("fatigue_level", state.get("fatigue", 0)),
        }

    # ── P2 — Emotional override (difficulty cap) ─────────────────────

    def _apply_emotional_override(
        self,
        state: dict,
        profile: dict,
        candidates: List[dict],
        day: int,
    ) -> Tuple[List[dict], str]:
        """Red/yellow light emotional check — cap difficulty if needed.

        Called AFTER safety pre-filter, BEFORE recommender ranking.
        """
        pain = state.get("pain_level", 0)
        fatigue = state.get("fatigue_level", 0)
        original_candidates = list(candidates)

        # Red light: pain > 7 or fatigue > 8 → cap difficulty at 1
        if pain > 7 or fatigue > 8:
            capped = [t for t in original_candidates if t.get("difficulty", 1) <= 1]
            if not capped:
                # Fallback: allow difficulty ≤ 2
                capped = [t for t in original_candidates if t.get("difficulty", 1) <= 2]
            return capped if capped else original_candidates, "red_light"

        # Yellow light: pain > 5 or fatigue > 6 → cap difficulty at 2
        if pain > 5 or fatigue > 6:
            capped = [t for t in original_candidates if t.get("difficulty", 1) <= 2]
            if not capped:
                capped = [t for t in original_candidates if t.get("difficulty", 1) <= 3]
            return capped if capped else original_candidates, "yellow_light"

        return candidates, "green"

    # ── P10 — Sanity check ───────────────────────────────────────────

    def _narrator_sanity_check(self) -> bool:
        """Verify narrator v2 is properly initialized.

        Checks:
          1. Narrator instance exists
          2. Arc data is loaded
          3. Template data is loaded
        """
        try:
            assert self.narrator is not None, "Narrator not initialized"
            assert hasattr(self.narrator, "arcs") and self.narrator.arcs, (
                "Arc data not loaded"
            )
            assert hasattr(self.narrator, "templates") and self.narrator.templates, (
                "Template data not loaded"
            )
            logger.info("Narrator v2 sanity check passed.")
            return True
        except AssertionError as e:
            logger.warning("Narrator v2 sanity check failed: %s", e)
            return False

    # ── Single turn ──────────────────────────────────────────────────

    def run_single_turn(
        self,
        user_id: str,
        user_input_or_state: Any,
        mode: str = "simulation",
        day: int = 0,
        turn: int = 0,
    ) -> dict:
        """Execute one turn of the recommendation pipeline.

        Parameters
        ----------
        user_id : str
        user_input_or_state : dict | str
            In *simulation* mode a state dict; in *interactive* mode a
            natural-language string.
        mode : str
        day, turn : int
            Simulation day / turn indices.

        Returns
        -------
        dict
            Keys: ``recommended``, ``narrative``, ``feature_vector``,
            ``candidates``, ``state``, ``profile``, plus narrator v2 fields.
        """
        # P3 — Reset per-turn mutex flag
        self._narrator_spoke_this_turn = False

        profile = self.memory.get_profile(user_id)
        if not profile:
            raw = self.simulator.get_profile(user_id)
            profile = self._normalize_profile(raw) if raw else {}
            if profile:
                self.memory.set_profile(user_id, profile)

        # 1. Parse state
        if mode == "simulation":
            state = user_input_or_state if isinstance(user_input_or_state, dict) else {}
        else:
            state = self.user_understanding.parse(str(user_input_or_state), profile)
            state.setdefault("day", day)

        # 2. Memory update
        state.setdefault("day", day)
        self.memory.update_state(user_id, state)
        cr7, consec_high = self._get_memory_features(user_id)

        # Build feature vector
        fv = self.user_understanding.parse_structured(
            state, profile,
            completion_rate_7d=cr7,
            consecutive_high_intensity=consec_high,
        )

        # 3. Safety pre-filter
        all_candidates = list(self.task_library)
        if self.safety_agent.enabled:
            safe_candidates = self.safety_agent.pre_filter(all_candidates, state, profile)
        else:
            safe_candidates = list(all_candidates)
        # 3b. Emotional override — cap difficulty if pain/fatigue is high (P2)
        safe_candidates, emotional_level = self._apply_emotional_override(
            state, profile, safe_candidates, day,
        )
        if not safe_candidates:
            # A catalogue gap is an implementation/data error, not permission
            # to bypass constraints or silently omit a daily assignment.
            raise RuntimeError("Daily task coverage gap: no feasible catalogue task; review constraints and catalogue coverage.")

        # 4. Recommender — rank on safe candidates
        #    SAPS needs to know which arms passed pre-filter so its internal
        #    safe-arm gate works; other models ignore set_safe_ids.
        _rec_model = getattr(self.recommender, "model", None)
        if _rec_model is not None and hasattr(_rec_model, "set_safe_ids"):
            _rec_model.set_safe_ids({t["task_id"] for t in safe_candidates})
        recommended = self.recommender.recommend(fv.to_numpy(), safe_candidates)

        # 4b. IER: also rank on ALL candidates (before safety) to measure
        #     how many of bandit's top-K would have been blocked
        ier_k = 5
        if callable(getattr(_rec_model, "select_raw_topk", None)):
            # SAPS.select applies its safe-arm gate even on all_candidates.
            # Use the ungated ranking for both IER and the existing penalty
            # updates below; do not add a second stochastic scoring pass.
            raw_top_k_ids = _rec_model.select_raw_topk(
                fv.to_numpy(), [t["task_id"] for t in all_candidates], ier_k,
            )
        else:
            raw_top_k_all = self.recommender.recommend(
                fv.to_numpy(), all_candidates, top_k_override=ier_k,
            )
            raw_top_k_ids = [t["task_id"] for t in raw_top_k_all]
        safe_candidate_ids = {t["task_id"] for t in safe_candidates}
        infeasible_in_top_k = [tid for tid in raw_top_k_ids if tid not in safe_candidate_ids]

        # 5. Safety post-check
        if self.safety_agent.enabled:
            recommended = self.safety_agent.post_check(
                recommended, state, consecutive_high_intensity=consec_high,
            )
        if not recommended:
            # Recovery fallbacks must pass BOTH safety checks too.
            recovery = [t for t in safe_candidates if t.get("category") == "rest"]
            recommended = self.safety_agent.post_check(
                recovery or safe_candidates, state, consecutive_high_intensity=consec_high,
            )[: self.recommender.top_k]
        recommended = self.safety_agent.pre_filter(recommended, state, profile)
        if not recommended:
            raise RuntimeError("Daily task coverage gap: final safety checks rejected every task.")

        # 6. Narrator (old — backward compatibility)
        narrative = ""
        if self.config.get("narrator", {}).get("enable", True):
            narrative = self.narrator_agent.wrap(
                recommended, state, profile,
                day=day, completion_rate_7d=cr7,
            )

        # 6b. Narrator v2 — speak() for the chosen recommendation (P5)
        narrator_result: Dict[str, Any] = {}
        narrator_scenario = ""
        narrator_tone = ""
        try:
            if (self.config.get("narrator_v2", {}).get("enable", True)
                    and recommended
                    and mode == "simulation"):
                # P4 — assert no double-call within same turn
                assert not self._narrator_spoke_this_turn, (
                    "narrator.speak() already called this turn"
                )
                narrator_user_state = self._narrator_user_state(state)
                user_history = {"consecutive_skips": 0}
                chosen_task = recommended[0]
                narrator_scenario = "recommendation_framing"
                speak_context = {
                    "name": profile.get("name", "friend"),
                    "day": day + 1,
                    "task_name": chosen_task.get(
                        "name", chosen_task.get("task_id", "")
                    ),
                    "difficulty": chosen_task.get("difficulty", 1),
                    "reason_from_bandit": "matched to your current recovery state",
                    "body_part": chosen_task.get("body_part", ""),
                    "rehab_stage": profile.get(
                        "current_stage", profile.get("stage", "")
                    ),
                }
                prompt_text, prompt_meta = self.narrator.speak(
                    scenario=narrator_scenario,
                    context=speak_context,
                    user_profile=profile,
                    day=day + 1,
                    user_state=narrator_user_state,
                    user_history=user_history,
                )
                narrator_result = dict(prompt_meta)
                narrator_result["prompt_text"] = prompt_text
                narrator_tone = prompt_meta.get("tone_actually_used", "")
                self._narrator_spoke_this_turn = True
                # Append to prompt-level log
                prompt_meta_log = dict(prompt_meta)
                prompt_meta_log["day"] = day
                prompt_meta_log["user_id"] = user_id
                self._prompt_level_log.append(prompt_meta_log)
                # Override narrative with v2 text if available
                if prompt_text:
                    narrative = prompt_text
        except Exception as e:
            logger.warning("Narrator v2 speak() failed, falling back: %s", e)
            narrator_result = {}

        return {
            "recommended": recommended,
            "narrative": narrative,
            "feature_vector": fv,
            "candidates": safe_candidates,
            "state": state,
            "profile": profile,
            "raw_top_k_ids": raw_top_k_ids,
            "safe_candidate_ids": list(safe_candidate_ids),
            "infeasible_in_top_k": infeasible_in_top_k,
            "ier": len(infeasible_in_top_k) / len(raw_top_k_ids) if raw_top_k_ids else 0.0,
            # P7 — narrator v2 fields
            "narrator_result": narrator_result,
            "narrator_scenario": narrator_scenario,
            "narrator_tone": narrator_tone,
            "emotional_level": emotional_level,
        }

    # ── Feedback processing ──────────────────────────────────────────

    def process_feedback(
        self,
        user_id: str,
        chosen_task_id: str,
        feedback: dict,
        state: Optional[dict] = None,
        context_vec: Optional[np.ndarray] = None,
        day: int = 0,
    ) -> float:
        """Record feedback, compute reward, update the bandit model.

        Parameters
        ----------
        user_id, chosen_task_id : str
        feedback : dict
            Output of ``UserSimulator.simulate_feedback`` (or equivalent).
        state : dict | None
            State at recommendation time.  If *None* the last stored
            state is used.
        context_vec : np.ndarray | None
            Feature vector at recommendation time.  Re-computed from
            state if not given.
        day : int

        Returns
        -------
        float
            Computed scalar reward.
        """
        if state is None:
            state = self.memory.get_state(user_id)

        if feedback.get("action") == "skipped":
            # Zero is aggregate bookkeeping, not an observed physical reward.
            # Refusal enters memory without pain, ROM or chosen-arm updates.
            self.memory.record_interaction(
                user_id, chosen_task_id, "skipped", 0.0,
                {**state, "task_intensity": "none"}, day=day,
                context_state=dict(state),
            )
            return 0.0

        task = self.task_lookup.get(chosen_task_id, {})
        reward = self.simulator.compute_reward(feedback, state, task)

        # Re-derive context if needed
        if context_vec is None:
            profile = self.memory.get_profile(user_id)
            cr7, consec_high = self._get_memory_features(user_id)
            fv = self.user_understanding.parse_structured(
                state, profile,
                completion_rate_7d=cr7,
                consecutive_high_intensity=consec_high,
            )
            context_vec = fv.to_numpy()

        self.recommender.update(context_vec, chosen_task_id, reward)

        # Build post-state
        post_state = dict(state)
        post_state["pain_level"] = feedback.get("post_pain", state.get("pain_level", 5))
        post_state["fatigue_level"] = feedback.get(
            "post_fatigue", state.get("fatigue_level", 5)
        )
        rom_before = state.get("rom", 90) if state.get("rom") is not None else 90
        post_state["rom"] = rom_before + feedback.get("rom_change", 0)
        post_state["task_intensity"] = task.get("intensity", "low")

        action = feedback.get("action", "completed")
        self.memory.record_interaction(
            user_id=user_id,
            task_id=chosen_task_id,
            action=action,
            reward=reward,
            post_state=post_state,
            day=day,
            context_state=dict(state),
        )
        return reward

    # ── Full simulation ──────────────────────────────────────────────

    def run_simulation(self, n_days: Optional[int] = None) -> List[dict]:
        """Run simulation for all users over *n_days*.

        Returns
        -------
        list[dict]
            Interaction log — each entry has keys ``user_id``, ``day``,
            ``turn``, ``state``, ``candidates``, ``recommended``,
            ``chosen_task_id``, ``action``, ``reward``, ``post_state``.
        """
        sim_cfg = self.config.get("simulation", {})
        if n_days is None:
            n_days = sim_cfg.get("n_days", 30)
        interactions_per_day = sim_cfg.get("interactions_per_day", 1)

        user_ids = self.simulator.get_all_user_ids()
        if not user_ids:
            logger.warning("No user profiles found — aborting simulation.")
            return []

        # Seed memory with normalised profiles
        for uid in user_ids:
            raw = self.simulator.get_profile(uid)
            self.memory.set_profile(uid, self._normalize_profile(raw))

        logs: List[dict] = []

        # ── Safety-informed warm-start for NeuralUCB ─────────────────────
        # Pre-populate per-arm stats so that tasks flagged as unsafe for
        # the majority of user profiles start with a negative prior,
        # drastically reducing cold-start IER.
        model = getattr(self.recommender, "model", None)
        if callable(getattr(model, "warm_start_safety", None)):
            all_ids = {t["task_id"] for t in self.task_library}
            # Intersection of safe tasks across all users' initial states:
            # only tasks safe for EVERY user are considered globally safe.
            intersect_safe: Optional[set] = None
            for uid in user_ids:
                init_state = self.simulator.simulate_daily_state(uid, 0)
                profile = self.memory.get_profile(uid) or {}
                if self.safety_agent.enabled:
                    safe = self.safety_agent.pre_filter(
                        self.task_library, init_state, profile,
                    )
                    user_safe = {t["task_id"] for t in safe}
                    if intersect_safe is None:
                        intersect_safe = user_safe
                    else:
                        intersect_safe &= user_safe
                else:
                    intersect_safe = all_ids
                    break
            if intersect_safe is None:
                intersect_safe = all_ids
            penalty = -self.config.get("reward", {}).get("safety_lambda", 1.0)
            model.warm_start_safety(intersect_safe, all_ids, penalty=penalty)

        for day in range(n_days):
            for uid in user_ids:
                # P3 — Day-level tracking for narrator closing
                _day_accepted = False
                _day_skipped = False
                _day_had_interaction = False
                _day_safety_blocked = False
                _day_last_state: Optional[dict] = None
                _day_prev_pain: Optional[float] = None
                _day_last_post_state: Optional[dict] = None
                # P3 — Reset per-day mutex flag for daily closing
                self._narrator_closed_this_turn = False

                for turn in range(interactions_per_day):
                    state = self.simulator.simulate_daily_state(uid, day)
                    if _day_prev_pain is None:
                        _day_prev_pain = state.get("pain_level", 0)
                    _day_last_state = state

                    result = self.run_single_turn(
                        uid, state, mode="simulation", day=day, turn=turn,
                    )
                    recommended = result["recommended"]
                    profile = result.get("profile", {})
                    if not recommended:
                        continue

                    _day_had_interaction = True

                    # User choice
                    chosen_id, choice_meta = self.simulator.simulate_choice(
                        uid, recommended, state,
                    )

                    if not chosen_id:
                        _day_skipped = True
                        logs.append(self._build_log(
                            uid, day, turn, state, result, "", "skip", 0.0,
                            state, {}, result.get("narrative", ""),
                        ))
                        continue

                    chosen_task = choice_meta.get(
                        "chosen_task", self.task_lookup.get(chosen_id, {}),
                    )
                    action_raw = choice_meta.get("action", "completed")

                    if action_raw == "completed":
                        _day_accepted = True
                    else:
                        _day_skipped = True

                    # Feedback
                    if action_raw == "skipped":
                        feedback = {"completed": False, "action": "skipped",
                                    "executed": False, "source": "simulated_choice"}
                    else:
                        feedback = self.simulator.simulate_feedback(uid, chosen_task, state)
                        feedback.update(executed=True, source="simulated_execution")

                    ctx = result["feature_vector"].to_numpy()
                    reward = self.process_feedback(
                        uid, chosen_id, feedback, state, ctx, day=day,
                    )

                    # ── SAPS Beta-posterior update (S2 signal: post-hoc
                    # pain outcome).  Only the *chosen* arm receives an
                    # observation; pre-filter blocked arms have no post-hoc
                    # signal and their Beta stays at the prior — the hard-
                    # filter layer handles blocked arms separately.
                    # Two-layer safety division is explicit by design.
                    _model = getattr(self.recommender, "model", None)
                    if (feedback.get("executed") and _model is not None
                            and hasattr(_model, "update_safety")):
                        _pain_change = feedback.get("pain_change", 0)
                        _was_safe = _pain_change <= SAPS_PAIN_SAFE_THRESHOLD
                        _model.update_safety(chosen_id, _was_safe)

                    # Lagrangian safety penalty: give negative reward to
                    # infeasible tasks that the bandit wanted to recommend.
                    # P8 — multiply by arc-level safety_lambda_multiplier
                    safety_lambda = self.config.get("reward", {}).get("safety_lambda", 1.0)
                    try:
                        _act_key, _arc_meta = self.narrator._get_arc_context(
                            profile, day + 1,
                        )
                        arc_multiplier = _arc_meta.get("safety_lambda_multiplier", 1.0)
                        effective_lambda = safety_lambda * arc_multiplier
                    except Exception:
                        effective_lambda = safety_lambda
                    for bad_id in result.get("infeasible_in_top_k", []):
                        self.recommender.update(ctx, bad_id, -effective_lambda)

                    # Track safety blocks for day-level log
                    if result.get("infeasible_in_top_k"):
                        _day_safety_blocked = True

                    post_state = dict(state)
                    post_state["pain_level"] = feedback.get(
                        "post_pain", state.get("pain_level")
                    )
                    post_state["fatigue_level"] = feedback.get(
                        "post_fatigue", state.get("fatigue_level")
                    )
                    _day_last_post_state = post_state

                    action_label = "accept" if action_raw == "completed" else "skip"
                    logs.append(self._build_log(
                        uid, day, turn, state, result, chosen_id,
                        action_label, reward, post_state, feedback,
                        result.get("narrative", ""),
                    ))

                # ── End of day for this user: narrator daily closing (P5.2) ──
                try:
                    if (self.config.get("narrator_v2", {}).get("enable", True)
                            and _day_last_state is not None):
                        # P4 — assert no double-call for closing within same day
                        assert not self._narrator_closed_this_turn, (
                            "narrator.say_daily_closing() already called this day"
                        )
                        profile = self.memory.get_profile(uid)
                        effective_post = _day_last_post_state or _day_last_state
                        pain_delta = (
                            (effective_post.get("pain_level", 0) - _day_prev_pain)
                            if _day_prev_pain is not None else 0
                        )

                        # P3 — Mutual-exclusion flags
                        no_interaction = not _day_had_interaction
                        user_accepted = _day_accepted
                        user_skipped = _day_skipped and not _day_accepted

                        # P4 — Mutual-exclusion invariant
                        mutex_sum = (
                            int(no_interaction) + int(user_skipped) + int(user_accepted)
                        )
                        if mutex_sum != 1:
                            # Fallback: force to most reasonable state
                            if _day_had_interaction:
                                if _day_accepted:
                                    no_interaction, user_skipped, user_accepted = (
                                        False, False, True,
                                    )
                                else:
                                    no_interaction, user_skipped, user_accepted = (
                                        False, True, False,
                                    )
                            else:
                                no_interaction, user_skipped, user_accepted = (
                                    True, False, False,
                                )

                        adherence = 1.0 if _day_accepted else 0.0

                        # P7 — day_log for variant selection
                        day_log_closing: Dict[str, Any] = {
                            "day": day + 1,
                            "no_interaction": no_interaction,
                            "user_skipped": user_skipped,
                            "user_accepted": user_accepted,
                            "safety_blocked_today": _day_safety_blocked,
                            "pain_delta": pain_delta,
                            "fatigue_level": effective_post.get("fatigue_level", 0),
                        }

                        # P5.2 — closing_context for template rendering
                        narrator_us = self._narrator_user_state(effective_post)
                        closing_context: Dict[str, Any] = {
                            "name": profile.get("name", "friend"),
                            "pain_level": narrator_us["pain_level"],
                            "fatigue_level": narrator_us["fatigue_level"],
                            "streak_count": profile.get("streak", 0),
                            "streak": profile.get("streak", 0),
                            "target": 7,
                            "adherence": adherence,
                            "pain_delta": pain_delta,
                            "next_focus": "recovery",
                            "personal_moment_from_today": "Another day done",
                        }

                        closing_text, closing_meta = self.narrator.say_daily_closing(
                            day_log=day_log_closing,
                            user_profile=profile,
                            context=closing_context,
                        )
                        self._narrator_closed_this_turn = True
                        closing_meta["day"] = day
                        closing_meta["user_id"] = uid
                        closing_meta["prompt_text"] = closing_text
                        self._prompt_level_log.append(closing_meta)
                except Exception as e:
                    logger.warning(
                        "Narrator daily closing failed for user %s day %d: %s",
                        uid, day, e,
                    )

            if (day + 1) % 10 == 0:
                logger.info(
                    "Day %d/%d — %d interactions so far.", day + 1, n_days, len(logs),
                )

        # P6 — Export narrator aggregate counters
        try:
            narrator_counters = self.narrator.export_aggregate_counters()
            logger.info("Narrator counters: %s", narrator_counters)
        except Exception as e:
            logger.warning("Failed to export narrator counters: %s", e)
            narrator_counters = {}

        logger.info("Simulation finished: %d interactions.", len(logs))
        return logs

    def _build_log(
        self,
        uid: str,
        day: int,
        turn: int,
        state: dict,
        result: dict,
        chosen_id: str,
        action: str,
        reward: float,
        post_state: dict,
        feedback: dict,
        narrative: str,
    ) -> dict:
        # Include user profile info for safety violation checking in metrics
        profile = result.get("profile", {})
        chosen_task = {}
        for t in result.get("recommended", []):
            if t.get("task_id") == chosen_id:
                chosen_task = t
                break
        # SVR counts accepted tasks, not blocked candidates or skipped offers.
        # Reuse the full five-rule pre-filter independently of runtime enable /
        # drop_rule. Emotional caps and post-check dosage limits are not PF_R1-5.
        if action == "accept" and not chosen_task:
            raise ValueError(f"Accepted task {chosen_id!r} is absent from recommendations")
        safety_violation = bool(
            action == "accept"
            and not self._safety_auditor.pre_filter([chosen_task], state, profile)
        )
        # P7 — extract narrator v2 result for log fields
        narrator_result = result.get("narrator_result", {})
        return {
            "user_id": uid,
            "day": day,
            "turn": turn,
            "state": dict(state),
            "candidates": [t["task_id"] for t in result.get("candidates", [])],
            "recommended": [t["task_id"] for t in result.get("recommended", [])],
            "recommended_tasks": [
                {"task_id": t["task_id"], "difficulty": t.get("difficulty", 1),
                 "category": t.get("category", ""), "body_part": t.get("body_part", ""),
                 "contraindications": t.get("contraindications", []),
                 "applicable_stages": t.get("applicable_stages", [])}
                for t in result.get("recommended", [])
            ],
            "chosen_task_id": chosen_id,
            "chosen_task": {
                "task_id": chosen_task.get("task_id", chosen_id),
                "difficulty": chosen_task.get("difficulty", 1),
                "category": chosen_task.get("category", ""),
                "contraindications": chosen_task.get("contraindications", []),
                "applicable_stages": chosen_task.get("applicable_stages", []),
            },
            "action": action,
            "safety_violation": safety_violation,
            "reward": reward,
            "reward_observed": action == "accept" and bool(feedback),
            "post_state": dict(post_state),
            "feedback": dict(feedback) if feedback else {},
            "narrative": narrative,
            "user_condition": profile.get("condition", ""),
            "user_stage": profile.get("current_stage", profile.get("stage", "")),
            "user_contraindications": profile.get("contraindications", []),
            "raw_top_k_ids": result.get("raw_top_k_ids", []),
            "infeasible_in_top_k": result.get("infeasible_in_top_k", []),
            "ier": result.get("ier", 0.0),
            # P7 / P11 — narrator v2 fields (Figure 4.3 convention)
            "prompt_level_log": narrator_result.get("all_override_reasons", []),
            "narrator_tone": narrator_result.get("tone_actually_used", ""),
            "narrator_scenario": result.get("narrator_scenario", ""),
            "emotional_level": result.get("emotional_level", "green"),
        }

    # ── Interactive session ──────────────────────────────────────────

    def interactive_session(self, user_id: str) -> None:
        """Launch an interactive demo loop (reads from stdin).

        Parameters
        ----------
        user_id : str
            Must match one of the profiles in ``user_profiles.json``.
        """
        # Switch agents to interactive mode
        self.user_understanding = UserUnderstandingAgent(self.config, mode="interactive")
        self.narrator_agent = NarratorAgent(self.config, mode="interactive")

        profile_raw = self.simulator.get_profile(user_id)
        if not profile_raw:
            print(f"❌ Unknown user_id: {user_id}")
            return
        profile = self._normalize_profile(profile_raw)
        self.memory.set_profile(user_id, profile)

        print(f"\n{'═' * 60}")
        print("  NarraCare — Interactive Rehabilitation Session")
        print(f"  User : {profile.get('name', user_id)}")
        print(f"  Cond.: {profile.get('condition', 'N/A')}")
        print(f"  Stage: {profile.get('current_stage', profile.get('stage', 'N/A'))}")
        print(f"{'═' * 60}")
        print("Type your daily check-in (or 'quit' to exit).\n")

        day = 0
        while True:
            try:
                user_input = input("[You] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("\n👋 Session ended — take care and keep moving!")
                break

            # Crisis check
            is_crisis, crisis_msg = self.safety_agent.check_crisis(user_input)
            if is_crisis:
                print(f"\n[NarraCare] {crisis_msg}\n")
                continue

            result = self.run_single_turn(
                user_id, user_input, mode="interactive", day=day,
            )

            # Present recommendations
            if result["narrative"]:
                print(f"\n[NarraCare]\n{result['narrative']}\n")
            else:
                print("\n[NarraCare] Recommended exercises:")
                for i, t in enumerate(result["recommended"], 1):
                    print(f"  {i}. {t.get('name', t['task_id'])} "
                          f"({t.get('intensity', '?')} / {t.get('category', '?')})")
                print()

            # Ask user to choose
            print("Enter exercise number to complete, or 'skip':")
            try:
                choice = input("[You] > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if choice.lower() in ("skip", "s", ""):
                print("⏭️  Skipped — rest is part of training too.\n")
                day += 1
                continue

            try:
                idx = int(choice) - 1
                chosen = result["recommended"][idx]
            except (ValueError, IndexError):
                print("Invalid choice — skipping.\n")
                day += 1
                continue

            # Simplified feedback for interactive mode
            feedback = {
                "completed": True,
                "action": "completed",
                "pain_change": round(self.rng.normal(0, 0.5), 2),
                "fatigue_change": round(0.5 + self.rng.normal(0, 0.3), 2),
                "rom_change": round(max(0, 0.3 + self.rng.normal(0, 0.2)), 2),
                "satisfaction": round(float(np.clip(0.8 + self.rng.normal(0, 0.1), 0, 1)), 3),
                "post_pain": result["state"].get("pain_level", 5),
                "post_fatigue": min(
                    10, result["state"].get("fatigue_level", 5) + 0.5,
                ),
            }

            ctx = result["feature_vector"].to_numpy()
            reward = self.process_feedback(
                user_id, chosen["task_id"], feedback,
                result["state"], ctx, day=day,
            )
            print(f"\n✅ Completed {chosen.get('name', chosen['task_id'])}! "
                  f"(reward: {reward:.3f})\n")
            day += 1


# ══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="NarraCare Pipeline")
    parser.add_argument("--user", type=str, default=None,
                        help="Single user ID for mini demo (e.g. U003)")
    parser.add_argument("--day", type=int, default=None,
                        help="Number of days for mini demo")
    parser.add_argument("--policy", type=str, default=None,
                        help="Algorithm: saps, linucb, neural_ucb, etc.")
    parser.add_argument("mode", nargs="?", default=None,
                        help="Legacy mode: 'interactive'")
    args = parser.parse_args()

    overrides = {}
    if args.policy:
        overrides["recommender"] = {"algorithm": args.policy}

    pipeline = NarraCarePipeline("configs/default.yaml",
                                  config_overrides=overrides if overrides else None)

    if args.mode == "interactive":
        uid = args.user or "U001"
        pipeline.interactive_session(uid)
    elif args.user and args.day:
        # Mini demo: single user, N days
        pipeline.simulator.profiles = {
            k: v for k, v in pipeline.simulator.profiles.items()
            if k == args.user
        }
        logs = pipeline.run_simulation(n_days=args.day)
        if logs:
            last = logs[-1]
            task = last.get("chosen_task_id", "?")
            narration = last.get("narrative", last.get("narrator_tone", ""))
            reward = last.get("reward", 0)
            print(f"\n✅ Mini demo: {args.user}, day {args.day}, "
                  f"policy={args.policy or 'linucb'}")
            print(f"   Last recommendation: task={task}")
            print(f"   Narrative tone: {narration}")
            print(f"   Reward: {reward:.4f}")
            print(f"   Total interactions: {len(logs)}")
        else:
            print("No interactions generated.")
    else:
        logs = pipeline.run_simulation()
        print(f"\n✅ Simulation complete — {len(logs)} interactions logged.")
