"""
SAPS — Safety-Aware Personalized Selection via Thompson Sampling.

A BanditModel subclass that combines:
  * LinUCB reward backbone (θ^T x + α√(x^T A^{-1} x)) for exploitation
  * Per-arm Beta(α, β) posterior for safety probability
  * Thompson sampling of safety score in the composite selection

Design (R2 T4 spec + user patch):
  * `was_safe` ≡ Safety Agent pre-filter verdict (NOT user accept/reject)
  * Action space: composite-score the FULL catalog; output only safe arms
  * SVR = 0 guaranteed (unsafe arms never in final recommendation)
  * SAPS learns to avoid unsafe arms through (α, β) decay

Composite score:
  S(a) = UCB_reward(a) + τ_safe · s_a,   s_a ~ Beta(α_a, β_a)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class SAPSModel:
    """Safety-Aware Personalized Selection — a BanditModel-compatible class.

    Parameters
    ----------
    feature_dim : int
        Context vector dimension (default 15).
    alpha_ucb : float
        LinUCB exploration bonus weight (same semantics as LinUCB alpha).
    tau_safe : float
        Weight for the Thompson-sampled safety bonus.
    alpha_prior : float
        Beta prior: initial safe-observation count.
    beta_prior : float
        Beta prior: initial unsafe-observation count.
    seed : int
        RNG seed.
    """

    def __init__(
        self,
        feature_dim: int = 15,
        alpha_ucb: float = 1.0,
        tau_safe: float = 2.0,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.d = feature_dim
        self.alpha_ucb = alpha_ucb
        self.tau_safe = tau_safe
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # ── LinUCB reward backbone (per-arm disjoint) ────────────────
        self.A: Dict[str, np.ndarray] = {}     # arm_id → d×d
        self.b: Dict[str, np.ndarray] = {}     # arm_id → d×1

        # ── Safety Beta posteriors ───────────────────────────────────
        self._a0 = alpha_prior
        self._b0 = beta_prior
        self.safe_alpha: Dict[str, float] = {}  # arm_id → α
        self.safe_beta: Dict[str, float] = {}   # arm_id → β

        # ── Per-round safe set (set by pipeline before select()) ─────
        self._current_safe_ids: Set[str] = set()

    # ── Arm init helpers ─────────────────────────────────────────────

    def _init_arm(self, arm_id: str) -> None:
        if arm_id not in self.A:
            self.A[arm_id] = np.identity(self.d)
            self.b[arm_id] = np.zeros((self.d, 1))

    def _get_safety(self, arm_id: str) -> Tuple[float, float]:
        return (
            self.safe_alpha.get(arm_id, self._a0),
            self.safe_beta.get(arm_id, self._b0),
        )

    # ── Core scoring ─────────────────────────────────────────────────

    def _reward_ucb(self, arm_id: str, x_col: np.ndarray) -> float:
        """LinUCB UCB score for one arm."""
        self._init_arm(arm_id)
        A_inv = np.linalg.inv(self.A[arm_id])
        theta = A_inv @ self.b[arm_id]
        mean = float((theta.T @ x_col).item())
        exploration = self.alpha_ucb * float(np.sqrt((x_col.T @ A_inv @ x_col).item()))
        return mean + exploration

    def _thompson_safety(self, arm_id: str) -> float:
        """Thompson-sample p_safe ~ Beta(α, β)."""
        a, b = self._get_safety(arm_id)
        return float(self.rng.beta(a, b))

    # ── BanditModel interface ────────────────────────────────────────

    def set_safe_ids(self, safe_ids: Set[str]) -> None:
        """Called by pipeline each turn BEFORE select().

        Tells SAPS which arms passed the Safety Agent pre-filter.
        """
        self._current_safe_ids = set(safe_ids)

    def select(
        self,
        context: np.ndarray,
        candidate_ids: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Composite-score all candidates; return top-K **safe** arms.

        Returns
        -------
        list[(arm_id, composite_score)]
        """
        x = context.reshape(-1, 1)
        scored: List[Tuple[str, float]] = []

        for arm_id in candidate_ids:
            r_ucb = self._reward_ucb(arm_id, x)
            s = self._thompson_safety(arm_id)
            composite = r_ucb + self.tau_safe * s
            scored.append((arm_id, composite))

        scored.sort(key=lambda t: t[1], reverse=True)

        # SVR guarantee: only return safe arms
        safe_scored = [
            (aid, sc) for aid, sc in scored if aid in self._current_safe_ids
        ]
        return safe_scored[:top_k]

    def select_raw_topk(
        self,
        context: np.ndarray,
        candidate_ids: List[str],
        top_k: int,
    ) -> List[str]:
        """Return raw top-K arm IDs (before safety filter) for IER calc."""
        x = context.reshape(-1, 1)
        scored: List[Tuple[str, float]] = []
        for arm_id in candidate_ids:
            r_ucb = self._reward_ucb(arm_id, x)
            s = self._thompson_safety(arm_id)
            scored.append((arm_id, r_ucb + self.tau_safe * s))
        scored.sort(key=lambda t: t[1], reverse=True)
        return [aid for aid, _ in scored[:top_k]]

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        """Update LinUCB reward model for chosen arm."""
        self._init_arm(task_id)
        x = context.reshape(-1, 1)
        self.A[task_id] = self.A[task_id] + x @ x.T
        self.b[task_id] = self.b[task_id] + reward * x

    # ── Safety posterior update ───────────────────────────────────────

    def update_safety(self, arm_id: str, was_safe: bool) -> None:
        """Update single arm's Beta posterior.

        Parameters
        ----------
        arm_id : str
        was_safe : bool
            True = Safety Agent pre-filter did NOT block this arm.
        """
        a, b = self._get_safety(arm_id)
        if was_safe:
            self.safe_alpha[arm_id] = a + 1
            self.safe_beta[arm_id] = b
        else:
            self.safe_alpha[arm_id] = a
            self.safe_beta[arm_id] = b + 1

    def batch_update_safety(
        self,
        candidate_ids: List[str],
        blocked_ids: Set[str],
    ) -> None:
        """Update safety posteriors for a batch of candidates.

        Called by pipeline after each recommendation round with the
        raw top-K candidates and the set blocked by Safety Agent.
        """
        for arm_id in candidate_ids:
            self.update_safety(arm_id, arm_id not in blocked_ids)

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        n_arms = len(self.safe_alpha)
        if n_arms == 0:
            return {"n_arms": 0}
        p_means = []
        for arm_id in self.safe_alpha:
            a = self.safe_alpha[arm_id]
            b = self.safe_beta.get(arm_id, self._b0)
            p_means.append(a / (a + b))
        return {
            "n_arms": n_arms,
            "mean_p_safe": float(np.mean(p_means)),
            "min_p_safe": float(np.min(p_means)),
            "max_p_safe": float(np.max(p_means)),
        }

    def reset(self) -> None:
        self.A.clear()
        self.b.clear()
        self.safe_alpha.clear()
        self.safe_beta.clear()
        self.rng = np.random.default_rng(self.seed)
        self._current_safe_ids.clear()
