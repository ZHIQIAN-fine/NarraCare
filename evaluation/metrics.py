"""Evaluation metrics for the NarraCare rehabilitation recommender.

Provides standalone metric functions (precision, recall, nDCG, …) and a
convenience ``compute_all_metrics`` that evaluates everything at once.

All *interaction-level* functions accept a list of interaction-log dicts
produced by ``NarraCarePipeline.run_simulation``.
"""

from __future__ import annotations

import json
import math
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.agents.safety_agent import SafetyAgent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helper: load task library for safety checks
# ═══════════════════════════════════════════════════════════════════════════

_TASK_LOOKUP_CACHE: Optional[Dict[str, dict]] = None


def _load_task_lookup() -> Dict[str, dict]:
    """Load task library as {task_id: task_dict} lookup.  Cached after first call."""
    global _TASK_LOOKUP_CACHE
    if _TASK_LOOKUP_CACHE is not None:
        return _TASK_LOOKUP_CACHE
    project_root = Path(__file__).resolve().parent.parent
    task_path = project_root / "data" / "task_library.json"
    if task_path.exists():
        with open(task_path, "r", encoding="utf-8") as fh:
            tasks = json.load(fh)
        _TASK_LOOKUP_CACHE = {t["task_id"]: t for t in tasks}
    else:
        _TASK_LOOKUP_CACHE = {}
    return _TASK_LOOKUP_CACHE


# ═══════════════════════════════════════════════════════════════════════════
# 1. Standard IR metrics
# ═══════════════════════════════════════════════════════════════════════════


def precision_at_k(
    recommended: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Precision@k — fraction of top-k that are relevant.

    Parameters
    ----------
    recommended : list[str]
        Ranked list of item IDs (best first).
    relevant : set[str]
        Ground-truth relevant item IDs.
    k : int

    Returns
    -------
    float   Value in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    return sum(1 for item in top_k if item in relevant) / len(top_k)


def recall_at_k(
    recommended: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Recall@k — fraction of relevant items captured in top-k."""
    if not relevant or k <= 0:
        return 0.0
    top_k = recommended[:k]
    return sum(1 for item in top_k if item in relevant) / len(relevant)


def ndcg_at_k(
    recommended: List[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at *k*.

    Binary relevance: 1 if item ∈ relevant, else 0.
    """
    if k <= 0 or not relevant:
        return 0.0
    top_k = recommended[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 2)  # positions are 1-indexed

    # Ideal DCG (all relevant items at the top)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. Interaction-level metrics
# ═══════════════════════════════════════════════════════════════════════════


def hit_rate(interactions: List[dict]) -> float:
    """Fraction of turns where user accepted ≥ 1 recommendation."""
    if not interactions:
        return 0.0
    hits = sum(1 for ix in interactions if ix.get("action") == "accept")
    return hits / len(interactions)


def cumulative_reward(interactions: List[dict]) -> float:
    """Sum of reward signals across all interactions."""
    return sum(ix.get("reward", 0.0) for ix in interactions)


def safety_violation_rate(
    interactions: List[dict],
    task_lookup: Optional[Dict[str, dict]] = None,
) -> float:
    """Fraction of accepted interactions that violate at least one safety rule.

    Violations detected (any one triggers a flag):
      V1 – **Contraindication match**: the chosen task carries a contraindication
           tag that also appears in the user's personal contraindication list.
      V2 – **Stage mismatch**: the chosen task's ``applicable_stages`` does not
           include the user's current rehabilitation stage.
      V3 – **High-pain + high-difficulty**: user pain ≥ 7 and chosen task
           difficulty ≥ 4.
      V4 – **Severe-fatigue + non-rest**: user fatigue ≥ 8 and chosen task
           category is not ``rest`` or ``flexibility``.
      V5 – **Limited ROM**: ROM < 60 and difficulty ≥ 3 or category ``cardio``.

    New pipeline logs carry the five-rule ``safety_violation`` observation,
    using the run's configured thresholds. Legacy logs are replayed through
    SafetyAgent with default thresholds; the rule chain is not duplicated.
    """
    if not interactions:
        return 0.0
    if task_lookup is None:
        task_lookup = _load_task_lookup()

    violations = 0
    accepted = 0
    legacy_auditor = None

    for ix in interactions:
        if ix.get("action") != "accept":
            continue
        accepted += 1

        if "safety_violation" in ix:
            violations += bool(ix["safety_violation"])
            continue

        state = ix.get("state", {})
        chosen_id = ix.get("chosen_task_id", "")

        if not chosen_id:
            continue

        # Try enriched log first, fall back to task_lookup
        chosen = ix.get("chosen_task", {})
        if not chosen or not chosen.get("task_id"):
            chosen = task_lookup.get(chosen_id, {})

        if legacy_auditor is None:
            legacy_auditor = SafetyAgent({"safety": {"enable": True}})
        profile = {
            "current_stage": ix.get("user_stage", ""),
            "contraindications": ix.get("user_contraindications", []),
        }
        if not legacy_auditor.pre_filter([chosen], state, profile):
            violations += 1

    return violations / accepted if accepted > 0 else 0.0


def constraint_satisfaction_rate(interactions: List[dict]) -> float:
    """1 − safety_violation_rate (complementary measure)."""
    return 1.0 - safety_violation_rate(interactions)


def diversity(interactions: List[dict], task_lookup: Optional[Dict[str, dict]] = None) -> float:
    """Average intra-list category diversity across interactions.

    Diversity for a single list = |unique categories| / |list length|.
    """
    if not interactions:
        return 0.0
    if task_lookup is None:
        task_lookup = _load_task_lookup()

    diversities: List[float] = []
    for ix in interactions:
        rec_ids = ix.get("recommended", [])
        if not rec_ids:
            continue
        categories = [task_lookup.get(tid, {}).get("category", "unknown") for tid in rec_ids]
        diversities.append(len(set(categories)) / len(categories))

    return float(np.mean(diversities)) if diversities else 0.0


def novelty(
    interactions: List[dict],
    user_history: Optional[Dict[str, Set[str]]] = None,
) -> float:
    """Average fraction of recommended items not previously seen by the user.

    Parameters
    ----------
    interactions : list[dict]
    user_history : dict[str, set[str]] | None
        ``{user_id: set_of_task_ids_seen}``.  If *None* the history is
        built cumulatively from ``interactions`` itself.
    """
    if not interactions:
        return 0.0

    if user_history is None:
        user_history = defaultdict(set)

    running_history: Dict[str, set] = defaultdict(set, {k: set(v) for k, v in user_history.items()})
    novelties: List[float] = []

    for ix in interactions:
        uid = ix.get("user_id", "")
        rec_ids = ix.get("recommended", [])
        if not rec_ids:
            continue
        novel_count = sum(1 for tid in rec_ids if tid not in running_history[uid])
        novelties.append(novel_count / len(rec_ids))
        # Update running history with ALL recommended items (not just chosen)
        # This reflects "exposure" — items shown to the user are no longer novel
        for tid in rec_ids:
            running_history[uid].add(tid)
        chosen = ix.get("chosen_task_id", "")
        if chosen:
            running_history[uid].add(chosen)

    return float(np.mean(novelties)) if novelties else 0.0


def difficulty_adaptation(
    interactions: List[dict],
    task_lookup: Optional[Dict[str, dict]] = None,
) -> float:
    """Pearson correlation between (1 − user stress) and chosen task
    difficulty.  Positive values indicate the system appropriately
    adapts difficulty to user state.

    Stress = (pain + fatigue) / 20.
    """
    if not interactions:
        return 0.0
    if task_lookup is None:
        task_lookup = _load_task_lookup()

    stresses: List[float] = []
    difficulties: List[float] = []

    for ix in interactions:
        chosen = ix.get("chosen_task_id", "")
        if not chosen:
            continue
        state = ix.get("state", {})
        pain = state.get("pain_level", 5)
        fatigue = state.get("fatigue_level", 5)
        stress = (pain + fatigue) / 20.0
        diff = task_lookup.get(chosen, {}).get("difficulty", 3)
        stresses.append(1.0 - stress)  # capacity
        difficulties.append(float(diff))

    if len(stresses) < 3:
        return 0.0

    corr = float(np.corrcoef(stresses, difficulties)[0, 1])
    return corr if not np.isnan(corr) else 0.0


def task_success_rate(interactions: List[dict]) -> float:
    """Fraction of interactions that were accepted AND completed."""
    if not interactions:
        return 0.0
    successes = 0
    for ix in interactions:
        if ix.get("action") == "accept":
            fb = ix.get("feedback", {})
            if fb.get("completed", False) or fb.get("action") == "completed":
                successes += 1
    return successes / len(interactions)


def acceptance_rate(interactions: List[dict]) -> float:
    """Fraction of interactions where user accepted (vs. skipped)."""
    return hit_rate(interactions)  # identical definition


def avg_interaction_turns(interactions: List[dict]) -> float:
    """Average number of interaction turns per user-day pair."""
    if not interactions:
        return 0.0
    pairs: Dict[str, int] = defaultdict(int)
    for ix in interactions:
        key = f"{ix.get('user_id', '')}_{ix.get('day', 0)}"
        pairs[key] += 1
    return float(np.mean(list(pairs.values())))


def adherence_rate(interactions: List[dict], n_days: int = 30) -> float:
    """Per-user average of (days with ≥ 1 interaction / total days)."""
    if not interactions or n_days <= 0:
        return 0.0
    user_active_days: Dict[str, set] = defaultdict(set)
    users: set = set()
    for ix in interactions:
        uid = ix.get("user_id", "")
        users.add(uid)
        if ix.get("action") == "accept":
            user_active_days[uid].add(ix.get("day", -1))

    if not users:
        return 0.0
    rates = [len(user_active_days.get(u, set())) / n_days for u in users]
    return float(np.mean(rates))


# ═══════════════════════════════════════════════════════════════════════════
# 2c. Infeasible Exploration Rate (IER)
# ═══════════════════════════════════════════════════════════════════════════


def infeasible_exploration_rate(interactions: List[dict]) -> float:
    """Average fraction of bandit's raw top-K that would be blocked by Safety.

    IER(t) = |infeasible items in bandit top-K| / |bandit top-K|

    A lower IER indicates greater agreement between this proposal ranking and
    the current feasible set. It alone does not establish learning, clinical
    safety or causal benefit from a safety bonus.
    """
    if not interactions:
        return 0.0
    ier_values: List[float] = []
    for ix in interactions:
        ier_val = ix.get("ier", None)
        if ier_val is not None:
            ier_values.append(ier_val)
    return float(np.mean(ier_values)) if ier_values else 0.0


def infeasible_exploration_rate_per_day(
    interactions: List[dict],
) -> Dict[int, float]:
    """IER averaged per day across all users.  Used for learning-curve plots."""
    from collections import defaultdict
    day_iers: Dict[int, List[float]] = defaultdict(list)
    for ix in interactions:
        day = ix.get("day", 0)
        ier_val = ix.get("ier", None)
        if ier_val is not None:
            day_iers[day].append(ier_val)
    return {d: float(np.mean(vals)) for d, vals in sorted(day_iers.items())}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Aggregate — compute everything
# ═══════════════════════════════════════════════════════════════════════════


def compute_all_metrics(
    interaction_logs: List[dict],
    config: Optional[dict] = None,
) -> Dict[str, float]:
    """Compute all metrics on *interaction_logs* and return a flat dict.

    Parameters
    ----------
    interaction_logs : list[dict]
        Logs produced by ``NarraCarePipeline.run_simulation``.
    config : dict | None
        The application config.  Used for ``top_k_values`` and
        ``n_days`` parameter.

    Returns
    -------
    dict[str, float]
    """
    config = config or {}
    eval_cfg = config.get("evaluation", {})
    top_k_values = eval_cfg.get("top_k_values", [1, 3, 5])
    n_days = config.get("simulation", {}).get("n_days", 30)

    task_lookup = _load_task_lookup()

    results: Dict[str, float] = {}

    # ── Per-interaction IR metrics (averaged) ────────────────────────
    for k in top_k_values:
        prec_vals: List[float] = []
        rec_vals: List[float] = []
        ndcg_vals: List[float] = []

        for ix in interaction_logs:
            recommended = ix.get("recommended", [])
            # Relevant = the chosen task (if accepted)
            relevant: Set[str] = set()
            if ix.get("action") == "accept" and ix.get("chosen_task_id"):
                relevant.add(ix["chosen_task_id"])

            prec_vals.append(precision_at_k(recommended, relevant, k))
            rec_vals.append(recall_at_k(recommended, relevant, k))
            ndcg_vals.append(ndcg_at_k(recommended, relevant, k))

        results[f"precision@{k}"] = float(np.mean(prec_vals)) if prec_vals else 0.0
        results[f"recall@{k}"] = float(np.mean(rec_vals)) if rec_vals else 0.0
        results[f"ndcg@{k}"] = float(np.mean(ndcg_vals)) if ndcg_vals else 0.0

    # ── Scalar metrics ───────────────────────────────────────────────
    results["hit_rate"] = hit_rate(interaction_logs)
    results["cumulative_reward"] = cumulative_reward(interaction_logs)
    results["safety_violation_rate"] = safety_violation_rate(interaction_logs)
    results["constraint_satisfaction"] = constraint_satisfaction_rate(interaction_logs)
    results["diversity"] = diversity(interaction_logs, task_lookup)
    results["novelty"] = novelty(interaction_logs)
    results["difficulty_adaptation"] = difficulty_adaptation(interaction_logs, task_lookup)
    results["task_success_rate"] = task_success_rate(interaction_logs)
    results["acceptance_rate"] = acceptance_rate(interaction_logs)
    results["avg_turns"] = avg_interaction_turns(interaction_logs)
    results["adherence_rate"] = adherence_rate(interaction_logs, n_days)
    results["ier"] = infeasible_exploration_rate(interaction_logs)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

_CACHED_TASK_LOOKUP: Optional[Dict[str, dict]] = None


def _load_task_lookup() -> Dict[str, dict]:
    """Load the task library once and cache it."""
    global _CACHED_TASK_LOOKUP
    if _CACHED_TASK_LOOKUP is not None:
        return _CACHED_TASK_LOOKUP

    project_root = Path(__file__).resolve().parent.parent
    path = project_root / "data" / "task_library.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            tasks = json.load(fh)
        _CACHED_TASK_LOOKUP = {t["task_id"]: t for t in tasks}
    else:
        logger.warning("task_library.json not found at %s", path)
        _CACHED_TASK_LOOKUP = {}
    return _CACHED_TASK_LOOKUP
