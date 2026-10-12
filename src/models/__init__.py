"""
NarraCare recommendation models.

Exports all bandit algorithms and baseline recommenders so they can be
imported directly::

    from src.models import LinUCB, NeuralUCB, RandomRecommender
"""

from src.models.linucb import LinUCB
from src.models.neural_ucb import NeuralUCB
from src.models.baselines import (
    EpsilonGreedyRecommender,
    PopularityRecommender,
    RandomRecommender,
    RuleBasedRecommender,
)

__all__ = [
    "LinUCB",
    "NeuralUCB",
    "RandomRecommender",
    "PopularityRecommender",
    "RuleBasedRecommender",
    "EpsilonGreedyRecommender",
]
