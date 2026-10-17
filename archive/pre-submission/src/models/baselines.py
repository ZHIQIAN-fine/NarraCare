"""
Baseline recommenders for the NarraCare rehabilitation system.

All classes expose a unified interface:
    - ``recommend(context, candidate_arm_ids, top_k)``
    - ``update(context, arm_id, reward)``
    - ``reset()``
    - ``get_stats()``

This makes them drop-in replacements for ``LinUCB`` and ``NeuralUCB``
during evaluation and ablation studies.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ======================================================================
# 1. Random Recommender
# ======================================================================


class RandomRecommender:
    """Uniformly random selection from candidate arms.

    Serves as the weakest baseline — no learning, pure exploration.

    Args:
        seed: Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._cumulative_reward: float = 0.0
        self._n_updates: Dict[int, int] = defaultdict(int)

    def recommend(
        self,
        context: np.ndarray,
        candidate_arm_ids: List[int],
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """Randomly select ``top_k`` arms from candidates.

        Args:
            context: Ignored (stateless recommender).
            candidate_arm_ids: Arm IDs available this round.
            top_k: Number of arms to return.

        Returns:
            List of ``(arm_id, score)`` tuples with uniform random scores.
        """
        if not candidate_arm_ids:
            return []

        k = min(top_k, len(candidate_arm_ids))
        scores = self._rng.rand(len(candidate_arm_ids))
        indices = np.argsort(-scores)[:k]
        return [
            (candidate_arm_ids[i], float(scores[i]))
            for i in indices
        ]

    def update(
        self,
        context: np.ndarray,
        arm_id: int,
        reward: float,
    ) -> None:
        """Record reward for statistics (no model update)."""
        self._cumulative_reward += reward
        self._n_updates[arm_id] += 1

    def reset(self) -> None:
        """Reset statistics and RNG."""
        self._rng = np.random.RandomState(self.seed)
        self._cumulative_reward = 0.0
        self._n_updates.clear()

    def get_stats(self) -> Dict[str, object]:
        return {
            "cumulative_reward": self._cumulative_reward,
            "n_updates_per_arm": dict(self._n_updates),
            "total_updates": sum(self._n_updates.values()),
        }

    def __repr__(self) -> str:
        return f"RandomRecommender(seed={self.seed})"


# ======================================================================
# 2. Popularity Recommender
# ======================================================================


class PopularityRecommender:
    """Rank arms by historical completion rate across all users.

    Completion rate for arm *a* is defined as
    ``mean(rewards_a)``; arms with no history receive a score of 0.

    Args:
        seed: Random seed (for tie-breaking only).
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # arm_id -> list of observed rewards
        self._reward_history: Dict[int, List[float]] = defaultdict(list)
        self._cumulative_reward: float = 0.0
        self._n_updates: Dict[int, int] = defaultdict(int)

    def _completion_rate(self, arm_id: int) -> float:
        """Average reward (proxy for completion rate) for an arm."""
        history = self._reward_history.get(arm_id)
        if not history:
            return 0.0
        return float(np.mean(history))

    def recommend(
        self,
        context: np.ndarray,
        candidate_arm_ids: List[int],
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """Rank candidates by historical popularity (mean reward).

        Args:
            context: Ignored (popularity is context-independent).
            candidate_arm_ids: Arm IDs available this round.
            top_k: Number of arms to return.

        Returns:
            Sorted list of ``(arm_id, completion_rate)`` tuples.
        """
        if not candidate_arm_ids:
            return []

        scores = [
            (arm_id, self._completion_rate(arm_id))
            for arm_id in candidate_arm_ids
        ]
        # Shuffle for tie-breaking then sort descending.
        self._rng.shuffle(scores)
        scores.sort(key=lambda x: x[1], reverse=True)

        k = min(top_k, len(scores))
        return scores[:k]

    def update(
        self,
        context: np.ndarray,
        arm_id: int,
        reward: float,
    ) -> None:
        """Record the reward for the chosen arm."""
        self._reward_history[arm_id].append(reward)
        self._cumulative_reward += reward
        self._n_updates[arm_id] += 1

    def reset(self) -> None:
        """Clear all history."""
        self._rng = np.random.RandomState(self.seed)
        self._reward_history.clear()
        self._cumulative_reward = 0.0
        self._n_updates.clear()

    def get_stats(self) -> Dict[str, object]:
        return {
            "cumulative_reward": self._cumulative_reward,
            "n_updates_per_arm": dict(self._n_updates),
            "total_updates": sum(self._n_updates.values()),
            "n_tracked_arms": len(self._reward_history),
        }

    def __repr__(self) -> str:
        return f"PopularityRecommender(seed={self.seed})"


# ======================================================================
# 3. Rule-Based Recommender
# ======================================================================

# Context vector convention (indices into the feature vector):
#   We expect the caller to place these features at known positions.
#   The rule-based recommender reads raw context attributes instead.
#   For flexibility it accepts an optional ``context_meta`` dict.

class RuleBasedRecommender:
    """Fixed clinical rules based on rehabilitation stage and pain level.

    Rules (applied in priority order):
        1. If pain_level >= ``pain_threshold``: only allow exercises with
           difficulty <= 1  (low intensity).
        2. If rehab_stage == ``"early"``: only allow difficulty <= 2.
        3. Otherwise: allow all candidates, prefer lower difficulty first.

    The recommender requires a ``context_meta`` dictionary alongside the
    numeric context vector, containing at least:
        - ``pain_level`` (float, 0–10 scale)
        - ``rehab_stage`` (str: "early" | "mid" | "late")
        - ``candidate_difficulties`` ({arm_id: int}) mapping each
          candidate to its difficulty level (1–5).

    Args:
        pain_threshold: Pain value above which only low-difficulty
            exercises are recommended (default 5, matching the task spec).
        early_stage_max_difficulty: Maximum difficulty during early rehab.
        seed: Random seed for tie-breaking.
    """

    def __init__(
        self,
        pain_threshold: float = 5.0,
        early_stage_max_difficulty: int = 2,
        seed: Optional[int] = 42,
    ) -> None:
        self.pain_threshold = pain_threshold
        self.early_stage_max_difficulty = early_stage_max_difficulty
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        self._cumulative_reward: float = 0.0
        self._n_updates: Dict[int, int] = defaultdict(int)

    def recommend(
        self,
        context: np.ndarray,
        candidate_arm_ids: List[int],
        top_k: int = 3,
        context_meta: Optional[Dict] = None,
    ) -> List[Tuple[int, float]]:
        """Apply clinical rules to filter and rank candidates.

        If ``context_meta`` is not provided, the recommender falls back
        to random selection from ``candidate_arm_ids``.

        Args:
            context: Numeric context vector (may be used for feature
                extraction if ``context_meta`` is absent).
            candidate_arm_ids: Available arm IDs.
            top_k: Number of arms to return.
            context_meta: Optional dict with ``pain_level``,
                ``rehab_stage``, and ``candidate_difficulties``.

        Returns:
            Sorted ``(arm_id, rule_score)`` list.
        """
        if not candidate_arm_ids:
            return []

        # ---- fallback when no metadata is available ----
        if context_meta is None:
            # Attempt to extract pain_level from context vector (index 0)
            # and rehab_stage indicator (index 1: 0=early, 1=mid, 2=late).
            # Difficulty is unknown → treat all candidates equally.
            context = np.asarray(context, dtype=np.float64).ravel()
            pain_level = float(context[0]) if len(context) > 0 else 0.0
            stage_idx = int(round(context[1])) if len(context) > 1 else 1
            stage_map = {0: "early", 1: "mid", 2: "late"}
            rehab_stage = stage_map.get(stage_idx, "mid")
            candidate_difficulties: Dict[int, int] = {
                arm_id: 3 for arm_id in candidate_arm_ids
            }
        else:
            pain_level = float(context_meta.get("pain_level", 0))
            rehab_stage = str(context_meta.get("rehab_stage", "mid"))
            candidate_difficulties = context_meta.get(
                "candidate_difficulties", {}
            )

        # ---- apply rules ----
        max_diff: int = 5  # default: allow all

        if pain_level > self.pain_threshold:
            max_diff = 1  # high pain → low difficulty only
        elif rehab_stage == "early":
            max_diff = self.early_stage_max_difficulty

        filtered: List[Tuple[int, int]] = []
        for arm_id in candidate_arm_ids:
            diff = candidate_difficulties.get(arm_id, 3)
            if diff <= max_diff:
                filtered.append((arm_id, diff))

        # If all filtered out, relax and allow lowest-difficulty candidates.
        if not filtered:
            all_diffs = [
                (arm_id, candidate_difficulties.get(arm_id, 3))
                for arm_id in candidate_arm_ids
            ]
            min_diff = min(d for _, d in all_diffs)
            filtered = [(a, d) for a, d in all_diffs if d == min_diff]

        # Score: prefer lower difficulty (higher score = more preferred).
        scored: List[Tuple[int, float]] = [
            (arm_id, float(max_diff - diff + 1))
            for arm_id, diff in filtered
        ]

        self._rng.shuffle(scored)
        scored.sort(key=lambda x: x[1], reverse=True)

        k = min(top_k, len(scored))
        return scored[:k]

    def update(
        self,
        context: np.ndarray,
        arm_id: int,
        reward: float,
    ) -> None:
        """Record reward (rules are fixed, no learning)."""
        self._cumulative_reward += reward
        self._n_updates[arm_id] += 1

    def reset(self) -> None:
        """Reset statistics."""
        self._rng = np.random.RandomState(self.seed)
        self._cumulative_reward = 0.0
        self._n_updates.clear()

    def get_stats(self) -> Dict[str, object]:
        return {
            "cumulative_reward": self._cumulative_reward,
            "n_updates_per_arm": dict(self._n_updates),
            "total_updates": sum(self._n_updates.values()),
        }

    def __repr__(self) -> str:
        return (
            f"RuleBasedRecommender(pain_threshold={self.pain_threshold}, "
            f"early_max_diff={self.early_stage_max_difficulty}, "
            f"seed={self.seed})"
        )


# ======================================================================
# 4. Epsilon-Greedy Recommender
# ======================================================================


class EpsilonGreedyRecommender:
    """ε-greedy contextual bandit with per-arm mean-reward tracking.

    With probability ε the recommender explores (uniform random among
    candidates); otherwise it exploits by selecting the arms with the
    highest estimated mean reward.

    Args:
        epsilon: Exploration probability in [0, 1].
        seed: Random seed.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        seed: Optional[int] = 42,
    ) -> None:
        self.epsilon = epsilon
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Per-arm statistics
        self._reward_sum: Dict[int, float] = defaultdict(float)
        self._reward_count: Dict[int, int] = defaultdict(int)
        self._cumulative_reward: float = 0.0

    def _mean_reward(self, arm_id: int) -> float:
        """Estimated mean reward for an arm."""
        count = self._reward_count.get(arm_id, 0)
        if count == 0:
            return 0.0
        return self._reward_sum[arm_id] / count

    def recommend(
        self,
        context: np.ndarray,
        candidate_arm_ids: List[int],
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """Select top-k arms using ε-greedy strategy.

        For each of the ``top_k`` slots:
            - With probability ε: pick uniformly at random from remaining
              candidates.
            - Otherwise: pick the candidate with the highest mean reward.

        Args:
            context: Numeric context (currently unused; the estimator is
                context-free, but we keep the interface uniform).
            candidate_arm_ids: Available arm IDs.
            top_k: Number of arms to return.

        Returns:
            List of ``(arm_id, estimated_mean_reward)`` tuples.
        """
        if not candidate_arm_ids:
            return []

        k = min(top_k, len(candidate_arm_ids))
        remaining = list(candidate_arm_ids)
        selected: List[Tuple[int, float]] = []

        for _ in range(k):
            if not remaining:
                break
            if self._rng.rand() < self.epsilon:
                # Explore: uniform random
                idx = self._rng.randint(len(remaining))
            else:
                # Exploit: best mean reward
                means = [self._mean_reward(a) for a in remaining]
                # Add small noise for tie-breaking
                noise = self._rng.rand(len(means)) * 1e-8
                idx = int(np.argmax(np.array(means) + noise))

            arm_id = remaining.pop(idx)
            selected.append((arm_id, self._mean_reward(arm_id)))

        return selected

    def update(
        self,
        context: np.ndarray,
        arm_id: int,
        reward: float,
    ) -> None:
        """Update the per-arm reward estimate.

        Args:
            context: Ignored (context-free estimator).
            arm_id: The arm that was played.
            reward: Observed reward.
        """
        self._reward_sum[arm_id] += reward
        self._reward_count[arm_id] += 1
        self._cumulative_reward += reward

    def reset(self) -> None:
        """Reset all estimates and statistics."""
        self._rng = np.random.RandomState(self.seed)
        self._reward_sum.clear()
        self._reward_count.clear()
        self._cumulative_reward = 0.0

    def get_stats(self) -> Dict[str, object]:
        return {
            "cumulative_reward": self._cumulative_reward,
            "n_updates_per_arm": dict(self._reward_count),
            "total_updates": sum(self._reward_count.values()),
            "mean_rewards": {
                a: self._mean_reward(a) for a in self._reward_count
            },
        }

    def __repr__(self) -> str:
        return (
            f"EpsilonGreedyRecommender(epsilon={self.epsilon}, "
            f"seed={self.seed})"
        )
