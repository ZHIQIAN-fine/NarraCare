"""Reward Composer — computes composite reward from user feedback and state.

The reward computation is integrated into the UserSimulator (src/simulator.py),
method ``compute_reward()``. This module re-exports it for import convenience.

Reward components (weights from configs/default.yaml → reward):
  w_completed         0.35   Task completion signal
  w_safety            0.25   Safety constraint satisfaction
  w_difficulty_match  0.20   State-appropriate difficulty
  w_pain_penalty      0.10   Pain increase penalty
  w_dropout_penalty   0.05   Session dropout penalty
  w_category          0.05   Category preference match
  safety_lambda       1.00   Lagrangian penalty for infeasible exploration
"""
from src.simulator import UserSimulator  # noqa: F401
