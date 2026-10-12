"""User Memory — per-user storage of profiles, interaction history, daily
states, completion records and stage transitions.

Uses a simple dict-based in-memory store for simulation.  A SQLite backend
can be plugged in for persistence if needed.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InteractionRecord:
    """Single interaction between a user and a task."""

    day: int
    task_id: str
    action: str          # "completed" | "skipped" | "partial"
    reward: float
    context_state: dict  # state at recommendation time
    post_state: dict     # state after interaction


@dataclass
class _UserData:
    """Internal per-user data store."""

    profile: dict = field(default_factory=dict)
    daily_states: Dict[int, dict] = field(default_factory=dict)      # day → state
    interactions: List[InteractionRecord] = field(default_factory=list)
    stage_transitions: List[dict] = field(default_factory=list)       # [{from, to, day}]


class UserMemory:
    """In-memory per-user storage for the NarraCare pipeline.

    Parameters
    ----------
    config : dict
        Full application config (parsed ``default.yaml``).
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}
        self._store: Dict[str, _UserData] = defaultdict(_UserData)

    # ── Profile management ───────────────────────────────────────────────

    def set_profile(self, user_id: str, profile: dict) -> None:
        """Store or overwrite the user profile."""
        self._store[user_id].profile = dict(profile)

    def get_profile(self, user_id: str) -> dict:
        """Return the stored profile (empty dict if unknown user)."""
        return self._store[user_id].profile

    # ── Daily state ──────────────────────────────────────────────────────

    def update_state(self, user_id: str, state: dict) -> None:
        """Record / update the user's state for today.

        ``state`` must contain at least ``day`` (int) alongside clinical
        fields like ``pain_level``, ``fatigue_level``, ``rom``, ``mood``.
        """
        day = state.get("day", len(self._store[user_id].daily_states))
        self._store[user_id].daily_states[day] = dict(state)
        logger.debug("Updated state for %s on day %d", user_id, day)

    def get_state(self, user_id: str, day: Optional[int] = None) -> dict:
        """Return the state for a specific day (latest if *day* is None)."""
        states = self._store[user_id].daily_states
        if not states:
            return {}
        if day is not None:
            return states.get(day, {})
        latest_day = max(states.keys())
        return states[latest_day]

    # ── Interaction recording ────────────────────────────────────────────

    def record_interaction(
        self,
        user_id: str,
        task_id: str,
        action: str,
        reward: float,
        post_state: dict,
        day: Optional[int] = None,
        context_state: Optional[dict] = None,
    ) -> None:
        """Log an interaction with a task.

        Parameters
        ----------
        user_id : str
        task_id : str
        action : str
            One of ``"completed"``, ``"skipped"``, ``"partial"``.
        reward : float
            Computed reward signal.
        post_state : dict
            User state *after* performing (or skipping) the task.
        day : int | None
            Simulation day.  Defaults to last recorded day.
        context_state : dict | None
            State at recommendation time.
        """
        if day is None:
            states = self._store[user_id].daily_states
            day = max(states.keys()) if states else 0

        record = InteractionRecord(
            day=day,
            task_id=task_id,
            action=action,
            reward=reward,
            context_state=context_state or {},
            post_state=dict(post_state),
        )
        self._store[user_id].interactions.append(record)
        logger.debug("Recorded interaction: user=%s task=%s action=%s reward=%.3f",
                      user_id, task_id, action, reward)

    # ── Aggregated queries ───────────────────────────────────────────────

    def get_completion_rate(self, user_id: str, days: int = 7) -> float:
        """Return the task completion rate over the last *days* days.

        Returns
        -------
        float
            Fraction of interactions where ``action == "completed"``.
            Returns 0.0 if no interactions exist in the window.
        """
        interactions = self._store[user_id].interactions
        if not interactions:
            return 0.0

        max_day = max(r.day for r in interactions)
        window_start = max_day - days + 1

        recent = [r for r in interactions if r.day >= window_start]
        if not recent:
            return 0.0

        completed = sum(1 for r in recent if r.action == "completed")
        return completed / len(recent)

    def get_consecutive_high_intensity(self, user_id: str) -> int:
        """Count the number of consecutive most-recent days where the user
        completed at least one high-intensity task.

        Returns
        -------
        int
        """
        interactions = self._store[user_id].interactions
        if not interactions:
            return 0

        # Group completed interactions by day
        day_has_high: Dict[int, bool] = {}
        for r in interactions:
            if r.action == "completed":
                post = r.post_state or {}
                # Check if the *task* was high intensity (stored in post_state or context)
                intensity = (
                    r.context_state.get("task_intensity", "")
                    or post.get("task_intensity", "")
                )
                if intensity == "high":
                    day_has_high[r.day] = True

        if not day_has_high:
            return 0

        # Walk backwards from the most recent day
        max_day = max(day_has_high.keys())
        count = 0
        d = max_day
        while d in day_has_high:
            count += 1
            d -= 1

        return count

    def get_history(self, user_id: str) -> List[dict]:
        """Return the full interaction history as a list of dicts.

        Returns
        -------
        list[dict]
            Each dict has keys: day, task_id, action, reward,
            context_state, post_state.
        """
        return [
            {
                "day": r.day,
                "task_id": r.task_id,
                "action": r.action,
                "reward": r.reward,
                "context_state": r.context_state,
                "post_state": r.post_state,
            }
            for r in self._store[user_id].interactions
        ]

    def get_avg_pain(self, user_id: str, days: int = 7) -> float:
        """Average pain level over the last *days* days."""
        states = self._store[user_id].daily_states
        if not states:
            return 0.0
        max_day = max(states.keys())
        window = [states[d] for d in range(max(0, max_day - days + 1), max_day + 1) if d in states]
        if not window:
            return 0.0
        return sum(s.get("pain_level", 0) for s in window) / len(window)

    def get_current_rom(self, user_id: str) -> float:
        """Most recent ROM value."""
        state = self.get_state(user_id)
        return state.get("rom", 0)

    # ── Stage transitions ────────────────────────────────────────────────

    def record_stage_transition(
        self, user_id: str, from_stage: int, to_stage: int, day: int
    ) -> None:
        self._store[user_id].stage_transitions.append({
            "from": from_stage,
            "to": to_stage,
            "day": day,
        })
        # Also update profile
        self._store[user_id].profile["stage"] = to_stage
        logger.info("Stage transition: user=%s %d→%d on day %d", user_id, from_stage, to_stage, day)

    def get_stage_transitions(self, user_id: str) -> List[dict]:
        return list(self._store[user_id].stage_transitions)

    # ── Utilities ────────────────────────────────────────────────────────

    def get_all_user_ids(self) -> List[str]:
        return list(self._store.keys())

    def summary(self, user_id: str) -> dict:
        """Quick summary dict for a user (useful for safety/progression checks)."""
        return {
            "avg_pain_7d": self.get_avg_pain(user_id, 7),
            "current_rom": self.get_current_rom(user_id),
            "completion_rate_7d": self.get_completion_rate(user_id, 7),
            "consecutive_high_intensity": self.get_consecutive_high_intensity(user_id),
            "total_interactions": len(self._store[user_id].interactions),
        }
