"""Recommender Agent — wraps bandit models (LinUCB, NeuralUCB, baselines)
and exposes a uniform ``recommend`` / ``update`` interface for the pipeline.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Base class
# ═══════════════════════════════════════════════════════════════════════════

class BanditModel(ABC):
    """Abstract base for all recommendation models."""

    @abstractmethod
    def select(
        self,
        context: np.ndarray,
        candidate_ids: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """Return top-k (task_id, score) pairs."""
        ...

    @abstractmethod
    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        """Update model with observed reward."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# LinUCB
# ═══════════════════════════════════════════════════════════════════════════

class LinUCBModel(BanditModel):
    """Disjoint linear UCB (Li et al., 2010)."""

    def __init__(self, feature_dim: int, alpha: float = 1.0, seed: int = 42) -> None:
        self.d = feature_dim
        self.alpha = alpha
        # defensive: seed accepted for API symmetry with NeuralUCBModel; no RNG consumption in current implementation
        self.seed = seed
        # Per-arm parameters (lazily initialised)
        self.A: Dict[str, np.ndarray] = {}  # d×d
        self.b: Dict[str, np.ndarray] = {}  # d×1

    def _init_arm(self, arm_id: str) -> None:
        if arm_id not in self.A:
            self.A[arm_id] = np.identity(self.d)
            self.b[arm_id] = np.zeros((self.d, 1))

    def select(
        self,
        context: np.ndarray,
        candidate_ids: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        x = context.reshape(-1, 1)  # column vector
        scores: List[Tuple[str, float]] = []

        for arm_id in candidate_ids:
            self._init_arm(arm_id)
            A_inv = np.linalg.inv(self.A[arm_id])
            theta = A_inv @ self.b[arm_id]
            mean = (theta.T @ x).item()
            exploration = self.alpha * np.sqrt((x.T @ A_inv @ x).item())
            ucb = mean + exploration
            scores.append((arm_id, ucb))

        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        self._init_arm(task_id)
        x = context.reshape(-1, 1)
        self.A[task_id] += x @ x.T
        self.b[task_id] += reward * x


# ═══════════════════════════════════════════════════════════════════════════
# NeuralUCB (lightweight PyTorch implementation)
# ═══════════════════════════════════════════════════════════════════════════

class NeuralUCBModel(BanditModel):
    """Neural UCB with per-arm statistics and shared context network.

    The neural network learns a context representation, while per-arm
    running statistics (mean reward, pull count) capture arm-specific
    quality — including Lagrangian safety penalties that push unsafe
    arms' means negative.  UCB exploration bonus decays with sqrt(n).

    Falls back to LinUCB if PyTorch is unavailable.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        learning_rate: float = 1e-3,
        alpha: float = 0.1,
        training_epochs: int = 10,
        seed: int = 42,
    ) -> None:
        self.d = feature_dim
        self.alpha = alpha
        self.training_epochs = training_epochs
        self.lr = learning_rate
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        try:
            import torch
            import torch.nn as nn

            torch.manual_seed(seed)
            layers: list = []
            in_dim = feature_dim
            for _ in range(n_layers):
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()
            self.torch = torch
            self._has_torch = True
        except ImportError:
            logger.warning("PyTorch not available — NeuralUCB falls back to LinUCB.")
            self._fallback = LinUCBModel(feature_dim, alpha)
            self._has_torch = False

        # Replay buffer for periodic re-training (context, arm, reward)
        self._buffer: List[Tuple[np.ndarray, str, float]] = []
        # Per-arm running stats: arm_id -> [sum_reward, count]
        self._arm_stats: Dict[str, List[float]] = {}

    def select(
        self,
        context: np.ndarray,
        candidate_ids: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if not self._has_torch:
            return self._fallback.select(context, candidate_ids, top_k)

        torch = self.torch
        self.net.eval()
        x_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            ctx_pred = self.net(x_tensor).item()

        scores: List[Tuple[str, float]] = []
        for arm_id in candidate_ids:
            stats = self._arm_stats.get(arm_id)
            if stats and stats[1] > 0:
                arm_mean = stats[0] / stats[1]
                n_pulls = stats[1]
            else:
                arm_mean = 0.0
                n_pulls = 0
            # UCB exploration bonus decays with pulls
            exploration = self.alpha * self._rng.standard_normal() / np.sqrt(n_pulls + 1)
            score = ctx_pred + arm_mean + exploration
            scores.append((arm_id, score))

        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        if not self._has_torch:
            self._fallback.update(context, task_id, reward)
            return

        self._buffer.append((context, task_id, reward))

        # Update per-arm running stats
        if task_id not in self._arm_stats:
            self._arm_stats[task_id] = [0.0, 0.0]
        self._arm_stats[task_id][0] += reward
        self._arm_stats[task_id][1] += 1

        # Retrain network periodically
        if len(self._buffer) % self.training_epochs == 0:
            self._retrain()

    def _retrain(self) -> None:
        torch = self.torch
        self.net.train()
        for _ in range(self.training_epochs):
            total_loss = 0.0
            for ctx, _, reward in self._buffer[-100:]:  # last 100 samples
                x = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)
                y = torch.tensor([[reward]], dtype=torch.float32)
                pred = self.net(x)
                loss = self.loss_fn(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        logger.debug("NeuralUCB retrained on %d samples, avg loss=%.4f",
                      len(self._buffer[-100:]), total_loss / max(len(self._buffer[-100:]), 1))

    # ── Safety-informed warm-start ───────────────────────────────────────
    def warm_start_safety(
        self,
        safe_ids: set,
        all_ids: set,
        penalty: float = -1.0,
    ) -> None:
        """Pre-populate arm stats so unsafe arms start with a negative prior.

        Called once before the bandit loop begins (e.g., by the pipeline).
        Safe arms keep default 0; unsafe arms receive *penalty* as a single
        pseudo-observation, discouraging the model from ranking them in the
        initial top-k and reducing cold-start IER.
        """
        if not self._has_torch:
            return  # fallback LinUCB does not use _arm_stats
        for arm_id in all_ids:
            if arm_id not in safe_ids and arm_id not in self._arm_stats:
                self._arm_stats[arm_id] = [penalty, 1.0]


# ═══════════════════════════════════════════════════════════════════════════
# Baseline models
# ═══════════════════════════════════════════════════════════════════════════

class RandomModel(BanditModel):
    """Uniform random baseline."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def select(self, context: np.ndarray, candidate_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
        scores = [(cid, self.rng.random()) for cid in candidate_ids]
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        pass  # no learning


class PopularityModel(BanditModel):
    """Recommends by global popularity (total reward count)."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.total_reward: Dict[str, float] = {}

    def select(self, context: np.ndarray, candidate_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
        scores = []
        for cid in candidate_ids:
            avg = self.total_reward.get(cid, 0.0) / max(self.counts.get(cid, 0), 1)
            scores.append((cid, avg))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        self.counts[task_id] = self.counts.get(task_id, 0) + 1
        self.total_reward[task_id] = self.total_reward.get(task_id, 0.0) + reward


class EpsilonGreedyModel(BanditModel):
    """ε-greedy baseline."""

    def __init__(self, epsilon: float = 0.1, seed: int = 42) -> None:
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.counts: Dict[str, int] = {}
        self.total_reward: Dict[str, float] = {}

    def select(self, context: np.ndarray, candidate_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
        if self.rng.random() < self.epsilon:
            # explore
            chosen = self.rng.choice(candidate_ids, size=min(top_k, len(candidate_ids)), replace=False).tolist()
            return [(cid, 0.0) for cid in chosen]
        # exploit
        scores = []
        for cid in candidate_ids:
            avg = self.total_reward.get(cid, 0.0) / max(self.counts.get(cid, 0), 1)
            scores.append((cid, avg))
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        self.counts[task_id] = self.counts.get(task_id, 0) + 1
        self.total_reward[task_id] = self.total_reward.get(task_id, 0.0) + reward


class RuleBasedModel(BanditModel):
    """Simple rule-based heuristic: prefer tasks whose difficulty matches
    the user's current pain/fatigue level."""

    _intensity_score = {"low": 1, "medium": 2, "high": 3}

    def __init__(self) -> None:
        pass

    def select(self, context: np.ndarray, candidate_ids: List[str], top_k: int) -> List[Tuple[str, float]]:
        # context[0] = normalised pain, context[1] = normalised fatigue
        pain = context[0]  # 0-1
        fatigue = context[1]  # 0-1
        stress = (pain + fatigue) / 2.0

        # Lower stress → prefer higher intensity; higher stress → prefer lower
        ideal_intensity = 1 + (1 - stress) * 2  # range [1, 3]

        scores = []
        for cid in candidate_ids:
            # Dummy: score based on distance from ideal (assumes task_id encodes nothing)
            scores.append((cid, -abs(ideal_intensity - 2)))  # placeholder
        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Recommender Agent (public API)
# ═══════════════════════════════════════════════════════════════════════════

_MODEL_REGISTRY: Dict[str, type] = {
    "linucb": LinUCBModel,
    "neural_ucb": NeuralUCBModel,
    "random": RandomModel,
    "popularity": PopularityModel,
    "epsilon_greedy": EpsilonGreedyModel,
    "rule_based": RuleBasedModel,
    # SAPS is handled specially in _build_model (not a BanditModel subclass)
}


class RecommenderAgent:
    """High-level recommender that wraps any bandit model.

    Parameters
    ----------
    config : dict
        Full application config (parsed ``default.yaml``).
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        rec_cfg = config.get("recommender", {})
        self.top_k: int = rec_cfg.get("top_k", 3)
        self.algorithm: str = rec_cfg.get("algorithm", "linucb")
        self.model: BanditModel = self._build_model(rec_cfg)
        logger.info("RecommenderAgent initialised with algorithm=%s", self.algorithm)

    # ── Factory ──────────────────────────────────────────────────────────

    @staticmethod
    def create(config: dict) -> "RecommenderAgent":
        """Factory method to create a RecommenderAgent from config."""
        return RecommenderAgent(config)

    def _build_model(self, rec_cfg: dict) -> BanditModel:
        algo = self.algorithm.lower()
        seed = self.config.get("system", {}).get("seed", 42)

        if algo == "linucb":
            params = rec_cfg.get("linucb", {})
            return LinUCBModel(
                feature_dim=params.get("feature_dim", 15),
                alpha=params.get("alpha", 1.0),
                seed=seed,
            )
        elif algo == "neural_ucb":
            params = rec_cfg.get("neural_ucb", {})
            return NeuralUCBModel(
                feature_dim=params.get("feature_dim", rec_cfg.get("linucb", {}).get("feature_dim", 15)),
                hidden_dim=params.get("hidden_dim", 64),
                n_layers=params.get("n_layers", 2),
                learning_rate=params.get("learning_rate", 1e-3),
                alpha=params.get("alpha", 0.1),
                training_epochs=params.get("training_epochs", 10),
                seed=seed,
            )
        elif algo == "random":
            return RandomModel(seed=seed)
        elif algo == "popularity":
            return PopularityModel()
        elif algo == "epsilon_greedy":
            params = rec_cfg.get("epsilon_greedy", {})
            return EpsilonGreedyModel(epsilon=params.get("epsilon", 0.1), seed=seed)
        elif algo == "rule_based":
            return RuleBasedModel()
        elif algo == "saps":
            from src.policies.saps import SAPSModel
            params = rec_cfg.get("saps", {})
            return SAPSModel(
                feature_dim=params.get("feature_dim", 15),
                alpha_ucb=params.get("alpha_ucb", 1.0),
                tau_safe=params.get("tau_safe", 2.0),
                alpha_prior=params.get("alpha_prior", 1.0),
                beta_prior=params.get("beta_prior", 1.0),
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown recommender algorithm: {algo}")

    # ── Public API ───────────────────────────────────────────────────────

    def recommend(
        self,
        context: np.ndarray,
        candidates: List[dict],
        top_k: Optional[int] = None,
        top_k_override: Optional[int] = None,
    ) -> List[dict]:
        """Recommend top-k tasks from *candidates* given *context* vector.

        Parameters
        ----------
        context : np.ndarray
            Feature vector (shape ``(feature_dim,)``).
        candidates : list[dict]
            Each dict must contain ``task_id`` (and any other metadata).
        top_k : int | None
            Override for the default top_k in config.
        top_k_override : int | None
            Alias for top_k (used by pipeline for IER computation).

        Returns
        -------
        list[dict]
            Ranked list of recommended task dicts, best first.
            Each dict is augmented with a ``"score"`` field.
        """
        k = top_k_override or top_k or self.top_k
        candidate_ids = [c["task_id"] for c in candidates]
        id_to_task = {c["task_id"]: c for c in candidates}

        scored = self.model.select(context, candidate_ids, k)

        results: List[dict] = []
        for task_id, score in scored:
            task = dict(id_to_task[task_id])  # shallow copy
            task["score"] = score
            results.append(task)

        return results

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        """Update the model with an observed interaction.

        Parameters
        ----------
        context : np.ndarray
            Feature vector at recommendation time.
        task_id : str
            The task the user interacted with.
        reward : float
            Observed reward signal.
        """
        self.model.update(context, task_id, reward)
