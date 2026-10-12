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
        config_path: str = "config/default.yaml",
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
        self.safety_agent = SafetyAgent(self.config)
        self.recommender = RecommenderAgent(self.config)
        self.narrator = NarratorAgent(self.config, mode="simulation")

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

        logger.info(
            "Pipeline ready — %d tasks, algorithm=%s, safety=%s, narrator=%s",
            len(self.task_library),
            self.config.get("recommender", {}).get("algorithm", "linucb"),
            self.safety_agent.enabled,
            self.config.get("narrator", {}).get("enable", True),
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
            ``candidates``, ``state``, ``profile``.
        """
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
        if not safe_candidates:
            safe_candidates = [t for t in self.task_library if t.get("category") == "rest"]
        if not safe_candidates:
            safe_candidates = self.task_library[:3]

        # 4. Recommender — rank on safe candidates
        recommended = self.recommender.recommend(fv.to_numpy(), safe_candidates)

        # 4b. IER: also rank on ALL candidates (before safety) to measure
        #     how many of bandit's top-K would have been blocked
        ier_k = 5
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
            recommended = safe_candidates[: self.recommender.top_k]

        # 6. Narrator
        narrative = ""
        if self.config.get("narrator", {}).get("enable", True):
            narrative = self.narrator.wrap(
                recommended, state, profile,
                day=day, completion_rate_7d=cr7,
            )

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

        for day in range(n_days):
            for uid in user_ids:
                for turn in range(interactions_per_day):
                    state = self.simulator.simulate_daily_state(uid, day)
                    result = self.run_single_turn(
                        uid, state, mode="simulation", day=day, turn=turn,
                    )
                    recommended = result["recommended"]
                    if not recommended:
                        continue

                    # User choice
                    chosen_id, choice_meta = self.simulator.simulate_choice(
                        uid, recommended, state,
                    )

                    if not chosen_id:
                        logs.append(self._build_log(
                            uid, day, turn, state, result, "", "skip", 0.0,
                            state, {}, result.get("narrative", ""),
                        ))
                        continue

                    chosen_task = choice_meta.get(
                        "chosen_task", self.task_lookup.get(chosen_id, {}),
                    )
                    action_raw = choice_meta.get("action", "completed")

                    # Feedback
                    feedback = self.simulator.simulate_feedback(uid, chosen_task, state)
                    if action_raw == "skipped":
                        feedback["completed"] = False
                        feedback["action"] = "skipped"

                    ctx = result["feature_vector"].to_numpy()
                    reward = self.process_feedback(
                        uid, chosen_id, feedback, state, ctx, day=day,
                    )

                    # Lagrangian safety penalty: give negative reward to
                    # infeasible tasks that the bandit wanted to recommend.
                    # This teaches the bandit to avoid unsafe tasks on its own.
                    safety_lambda = self.config.get("reward", {}).get("safety_lambda", 1.0)
                    for bad_id in result.get("infeasible_in_top_k", []):
                        self.recommender.update(ctx, bad_id, -safety_lambda)

                    post_state = dict(state)
                    post_state["pain_level"] = feedback.get(
                        "post_pain", state.get("pain_level")
                    )
                    post_state["fatigue_level"] = feedback.get(
                        "post_fatigue", state.get("fatigue_level")
                    )

                    action_label = "accept" if action_raw == "completed" else "skip"
                    logs.append(self._build_log(
                        uid, day, turn, state, result, chosen_id,
                        action_label, reward, post_state, feedback,
                        result.get("narrative", ""),
                    ))

            if (day + 1) % 10 == 0:
                logger.info(
                    "Day %d/%d — %d interactions so far.", day + 1, n_days, len(logs),
                )

        logger.info("Simulation finished: %d interactions.", len(logs))
        return logs

    @staticmethod
    def _build_log(
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
            "reward": reward,
            "post_state": dict(post_state),
            "feedback": dict(feedback) if feedback else {},
            "narrative": narrative,
            "user_condition": profile.get("condition", ""),
            "user_stage": profile.get("current_stage", profile.get("stage", "")),
            "user_contraindications": profile.get("contraindications", []),
            "raw_top_k_ids": result.get("raw_top_k_ids", []),
            "infeasible_in_top_k": result.get("infeasible_in_top_k", []),
            "ier": result.get("ier", 0.0),
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
        self.narrator = NarratorAgent(self.config, mode="interactive")

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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    )

    pipeline = NarraCarePipeline()

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        uid = sys.argv[2] if len(sys.argv) > 2 else "U001"
        pipeline.interactive_session(uid)
    else:
        logs = pipeline.run_simulation()
        print(f"\n✅ Simulation complete — {len(logs)} interactions logged.")
