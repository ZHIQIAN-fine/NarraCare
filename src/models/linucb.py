"""
LinUCB (Disjoint Model) for Contextual Bandit Recommendation.

Implements the disjoint linear model where each arm maintains independent
parameters. Supports dynamic candidate sets (safety-filtered arms) and
provides upper confidence bound exploration.

Reference:
    Li et al., "A Contextual-Bandit Approach to Personalized News Article
    Recommendation", WWW 2010.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class LinUCB:
    """Disjoint LinUCB contextual bandit algorithm.

    Each arm *a* maintains:
        - A_a: d×d matrix  (A_a = I_d initially)
        - b_a: d-dimensional vector (zeros initially)

    The estimated payoff with exploration bonus is:
        UCB_a(x) = x^T θ_a + α · sqrt(x^T A_a^{-1} x)
    where θ_a = A_a^{-1} b_a.

    Args:
        feature_dim: Dimensionality *d* of the context feature vector.
        alpha: Exploration parameter controlling the width of the
            confidence interval.  Larger values encourage more exploration.
        n_arms: Maximum number of arms the bandit may encounter.
            Arms are identified by integer IDs in ``[0, n_arms)``.
        seed: Random seed for reproducibility (used for tie-breaking).
    """

    def __init__(
        self,
        feature_dim: int = 15,
        alpha: float = 1.0,
        n_arms: int = 100,
        seed: Optional[int] = 42,
    ) -> None:
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.n_arms = n_arms
        self.seed = seed
        self._rng = np.random.RandomState(seed)

        # Per-arm parameters: lazily initialised on first encounter.
        self._A: Dict[int, np.ndarray] = {}
        self._b: Dict[int, np.ndarray] = {}

        # Statistics
        self._cumulative_reward: float = 0.0
        self._n_updates: Dict[int, int] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_arm(self, arm_id: int) -> None:
        """Lazily initialise parameters for an unseen arm."""
        if arm_id not in self._A:
            self._A[arm_id] = np.eye(self.feature_dim)
            self._b[arm_id] = np.zeros(self.feature_dim)

    def _compute_ucb(self, context: np.ndarray, arm_id: int) -> float:
        """Compute the UCB score for a single arm given a context vector.

        Args:
            context: 1-D array of shape ``(feature_dim,)``.
            arm_id: Integer identifier of the arm.

        Returns:
            Scalar UCB value.
        """
        self._ensure_arm(arm_id)
        A_inv = np.linalg.inv(self._A[arm_id])
        theta = A_inv @ self._b[arm_id]
        exploitation = context @ theta
        exploration = self.alpha * np.sqrt(context @ A_inv @ context)
        return float(exploitation + exploration)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recommend(
        self,
        context: np.ndarray,
        candidate_arm_ids: List[int],
        top_k: int = 3,
    ) -> List[Tuple[int, float]]:
        """Select the top-k arms from the candidate set.

        Args:
            context: 1-D numpy array of shape ``(feature_dim,)``.
            candidate_arm_ids: Subset of arm IDs available this round
                (e.g. after safety filtering).
            top_k: Number of arms to return.

        Returns:
            List of ``(arm_id, ucb_score)`` tuples sorted by descending
            UCB score, length ``min(top_k, len(candidate_arm_ids))``.
        """
        context = np.asarray(context, dtype=np.float64).ravel()
        if context.shape[0] != self.feature_dim:
            raise ValueError(
                f"Context dimension {context.shape[0]} != "
                f"expected {self.feature_dim}"
            )

        if not candidate_arm_ids:
            return []

        scores: List[Tuple[int, float]] = []
        for arm_id in candidate_arm_ids:
            ucb = self._compute_ucb(context, arm_id)
            scores.append((arm_id, ucb))

        # Sort descending by UCB; break ties randomly for fairness.
        self._rng.shuffle(scores)  # shuffle first so ties are random
        scores.sort(key=lambda x: x[1], reverse=True)

        k = min(top_k, len(scores))
        return scores[:k]

    def update(
        self,
        context: np.ndarray,
        arm_id: int,
        reward: float,
    ) -> None:
        """Update the model for the chosen arm after observing a reward.

        Args:
            context: 1-D numpy array of shape ``(feature_dim,)``.
            arm_id: The arm that was played.
            reward: Observed reward signal.
        """
        context = np.asarray(context, dtype=np.float64).ravel()
        if context.shape[0] != self.feature_dim:
            raise ValueError(
                f"Context dimension {context.shape[0]} != "
                f"expected {self.feature_dim}"
            )

        self._ensure_arm(arm_id)
        self._A[arm_id] += np.outer(context, context)
        self._b[arm_id] += reward * context

        self._cumulative_reward += reward
        self._n_updates[arm_id] = self._n_updates.get(arm_id, 0) + 1

    def reset(self) -> None:
        """Reset all learned parameters and statistics."""
        self._A.clear()
        self._b.clear()
        self._cumulative_reward = 0.0
        self._n_updates.clear()
        self._rng = np.random.RandomState(self.seed)

    def get_stats(self) -> Dict[str, object]:
        """Return current algorithm statistics.

        Returns:
            Dictionary with:
                - ``cumulative_reward``: Total reward accumulated.
                - ``n_updates_per_arm``: ``{arm_id: count}`` mapping.
                - ``total_updates``: Sum of all per-arm updates.
                - ``n_active_arms``: Number of arms that have been updated
                  at least once.
        """
        return {
            "cumulative_reward": self._cumulative_reward,
            "n_updates_per_arm": dict(self._n_updates),
            "total_updates": sum(self._n_updates.values()),
            "n_active_arms": len(self._n_updates),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LinUCB(feature_dim={self.feature_dim}, alpha={self.alpha}, "
            f"n_arms={self.n_arms}, seed={self.seed})"
        )
