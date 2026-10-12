"""
Neural UCB for Contextual Bandit Recommendation.

Uses a small MLP to estimate expected rewards and approximates the
Neural Tangent Kernel (NTK) for UCB-style exploration.  The network is
periodically retrained on a replay buffer of past interactions.

Reference:
    Zhou et al., "Neural Contextual Bandits with UCB-based Exploration",
    ICML 2020.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ======================================================================
# MLP backbone
# ======================================================================

class _RewardMLP(nn.Module):
    """Small fully-connected network for reward prediction.

    Architecture: input -> [Linear -> ReLU] × n_layers -> Linear(1).

    Args:
        feature_dim: Input feature dimensionality.
        hidden_dim: Width of every hidden layer.
        n_layers: Number of hidden layers (each followed by ReLU).
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feature_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        # Initialise weights with Xavier uniform for stable gradients.
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, feature_dim)`` or ``(feature_dim,)``.

        Returns:
            Predicted reward, shape ``(batch, 1)`` or ``(1,)``.
        """
        return self.net(x)


# ======================================================================
# Neural UCB
# ======================================================================

class NeuralUCB:
    """Neural Upper Confidence Bound contextual bandit.

    Maintains a shared MLP for reward prediction and uses the gradient of
    the network output w.r.t. its parameters as a feature map (NTK
    approximation) to compute an exploration bonus.

    UCB_a(x) = f_θ(x_a) + α · ‖g_θ(x_a)‖_{Z⁻¹}

    where g_θ is the gradient feature and Z is the accumulated outer-
    product matrix of gradient features (plus regularisation).

    Args:
        feature_dim: Dimensionality of the context-arm feature vector.
        hidden_dim: MLP hidden layer width.
        n_layers: Number of MLP hidden layers.
        alpha: Exploration parameter.
        learning_rate: Adam learning rate for network training.
        training_epochs: Epochs per retraining call.
        buffer_size: Maximum replay buffer capacity.
        retrain_every: Retrain the network every this many updates.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        feature_dim: int = 15,
        hidden_dim: int = 64,
        n_layers: int = 2,
        alpha: float = 0.1,
        learning_rate: float = 0.001,
        training_epochs: int = 10,
        buffer_size: int = 10_000,
        retrain_every: int = 20,
        seed: Optional[int] = 42,
    ) -> None:
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.buffer_size = buffer_size
        self.retrain_every = retrain_every
        self.seed = seed

        self._rng = np.random.RandomState(seed)
        self._set_torch_seed(seed)

        # Network & optimiser
        self._network = _RewardMLP(feature_dim, hidden_dim, n_layers)
        self._optimizer = optim.Adam(
            self._network.parameters(), lr=learning_rate
        )
        self._loss_fn = nn.MSELoss()

        # Total number of trainable parameters (for NTK dimension).
        self._p = sum(p.numel() for p in self._network.parameters())

        # NTK exploration matrix:  Z = λ I + Σ g g^T  (diagonal approx.)
        # We use a diagonal approximation for memory efficiency.
        self._lambda_reg: float = 1.0
        self._Z_diag = torch.ones(self._p) * self._lambda_reg

        # Replay buffer
        self._buffer_contexts: List[np.ndarray] = []
        self._buffer_arm_ids: List[int] = []
        self._buffer_rewards: List[float] = []

        # Statistics
        self._cumulative_reward: float = 0.0
        self._n_updates: Dict[int, int] = defaultdict(int)
        self._total_updates: int = 0

    @staticmethod
    def _set_torch_seed(seed: Optional[int]) -> None:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # ------------------------------------------------------------------
    # Gradient feature extraction
    # ------------------------------------------------------------------

    def _gradient_feature(self, context: torch.Tensor) -> torch.Tensor:
        """Compute the gradient of the network output w.r.t. all parameters.

        This serves as the NTK feature map φ(x) = ∇_θ f_θ(x).

        Args:
            context: 1-D tensor of shape ``(feature_dim,)``.

        Returns:
            1-D tensor of shape ``(p,)`` — the flattened gradient vector.
        """
        self._network.zero_grad()
        x = context.unsqueeze(0).requires_grad_(False)
        # Enable grad for parameters
        pred = self._network(x).squeeze()
        pred.backward()

        grads: list[torch.Tensor] = []
        for param in self._network.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().flatten())
            else:
                grads.append(torch.zeros(param.numel()))
        return torch.cat(grads)

    # ------------------------------------------------------------------
    # UCB computation
    # ------------------------------------------------------------------

    def _compute_ucb(self, context: np.ndarray) -> Tuple[float, torch.Tensor]:
        """Compute the UCB score for a context vector.

        Returns:
            Tuple of (ucb_score, gradient_feature).
        """
        x = torch.as_tensor(context, dtype=torch.float32)
        with torch.no_grad():
            pred = self._network(x.unsqueeze(0)).item()

        g = self._gradient_feature(x)

        # Exploration bonus: α · sqrt(g^T Z^{-1} g)  (diagonal approx.)
        Z_inv_diag = 1.0 / self._Z_diag
        exploration = self.alpha * torch.sqrt(
            torch.sum(g * g * Z_inv_diag)
        ).item()

        return pred + exploration, g

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
            candidate_arm_ids: Arm IDs available this round.
            top_k: Number of arms to recommend.

        Returns:
            List of ``(arm_id, ucb_score)`` tuples, descending by score.
        """
        context = np.asarray(context, dtype=np.float32).ravel()
        if context.shape[0] != self.feature_dim:
            raise ValueError(
                f"Context dim {context.shape[0]} != {self.feature_dim}"
            )

        if not candidate_arm_ids:
            return []

        self._network.eval()
        scores: List[Tuple[int, float]] = []
        for arm_id in candidate_arm_ids:
            ucb, _ = self._compute_ucb(context)
            scores.append((arm_id, ucb))

        # Shuffle first for random tie-breaking, then sort descending.
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
        """Record an observation and update the NTK exploration matrix.

        The network is retrained periodically (every ``retrain_every``
        updates) on the full replay buffer.

        Args:
            context: 1-D numpy array of shape ``(feature_dim,)``.
            arm_id: The arm that was played.
            reward: Observed reward.
        """
        context = np.asarray(context, dtype=np.float32).ravel()
        if context.shape[0] != self.feature_dim:
            raise ValueError(
                f"Context dim {context.shape[0]} != {self.feature_dim}"
            )

        # Store in replay buffer (FIFO eviction).
        if len(self._buffer_contexts) >= self.buffer_size:
            self._buffer_contexts.pop(0)
            self._buffer_arm_ids.pop(0)
            self._buffer_rewards.pop(0)

        self._buffer_contexts.append(context.copy())
        self._buffer_arm_ids.append(arm_id)
        self._buffer_rewards.append(reward)

        # Update NTK exploration matrix (diagonal approximation).
        self._network.train()
        _, g = self._compute_ucb(context)
        self._Z_diag += g * g

        # Statistics
        self._cumulative_reward += reward
        self._n_updates[arm_id] += 1
        self._total_updates += 1

        # Periodic retraining
        if self._total_updates % self.retrain_every == 0:
            self.train(epochs=self.training_epochs)

    def train(self, epochs: int = 10) -> float:
        """Retrain the MLP on the replay buffer.

        Args:
            epochs: Number of full passes over the buffer.

        Returns:
            Final epoch average loss.
        """
        if not self._buffer_contexts:
            return 0.0

        self._network.train()
        X = torch.as_tensor(
            np.array(self._buffer_contexts), dtype=torch.float32
        )
        Y = torch.as_tensor(
            self._buffer_rewards, dtype=torch.float32
        ).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(64, len(dataset)), shuffle=True
        )

        last_loss = 0.0
        for _epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for x_batch, y_batch in loader:
                self._optimizer.zero_grad()
                pred = self._network(x_batch)
                loss = self._loss_fn(pred, y_batch)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            last_loss = epoch_loss / max(n_batches, 1)

        return last_loss

    def reset(self) -> None:
        """Reset the network, exploration matrix, buffer, and statistics."""
        self._set_torch_seed(self.seed)
        self._rng = np.random.RandomState(self.seed)

        self._network = _RewardMLP(
            self.feature_dim, self.hidden_dim, self.n_layers
        )
        self._optimizer = optim.Adam(
            self._network.parameters(), lr=self.learning_rate
        )

        self._Z_diag = torch.ones(self._p) * self._lambda_reg

        self._buffer_contexts.clear()
        self._buffer_arm_ids.clear()
        self._buffer_rewards.clear()

        self._cumulative_reward = 0.0
        self._n_updates.clear()
        self._total_updates = 0

    def get_stats(self) -> Dict[str, object]:
        """Return current algorithm statistics.

        Returns:
            Dictionary with cumulative reward, per-arm update counts,
            replay buffer size, and total parameter count.
        """
        return {
            "cumulative_reward": self._cumulative_reward,
            "n_updates_per_arm": dict(self._n_updates),
            "total_updates": self._total_updates,
            "buffer_size": len(self._buffer_contexts),
            "n_parameters": self._p,
        }

    def __repr__(self) -> str:
        return (
            f"NeuralUCB(feature_dim={self.feature_dim}, "
            f"hidden_dim={self.hidden_dim}, n_layers={self.n_layers}, "
            f"alpha={self.alpha}, lr={self.learning_rate}, "
            f"seed={self.seed})"
        )
