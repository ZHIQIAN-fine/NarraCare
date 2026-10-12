"""NarraCare Agent modules.

Provides the core agents for the rehabilitation recommender pipeline:
- UserUnderstandingAgent: parses user input into structured feature vectors
- SafetyAgent: enforces safety constraints and crisis detection
- RecommenderAgent: bandit-based task recommendation
- NarratorAgent: wraps recommendations in narrative explanations
"""

from src.agents.user_understanding import UserUnderstandingAgent
from src.agents.safety_agent import SafetyAgent
from src.agents.recommender import RecommenderAgent
from src.agents.narrator_agent import NarratorAgent

__all__ = [
    "UserUnderstandingAgent",
    "SafetyAgent",
    "RecommenderAgent",
    "NarratorAgent",
]
