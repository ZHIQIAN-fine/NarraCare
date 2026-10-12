"""Basic tests for UserSimulator — runs at both Step A and Step B."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.simulator import UserSimulator

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return {
        "system": {"seed": 42},
        "simulation": {
            "pain_daily_variance": 1.5,
            "fatigue_daily_variance": 2.0,
            "personality": {
                "motivated": {"accept_challenge_prob": 0.75, "skip_prob": 0.05, "prefer_higher_difficulty": True},
                "cautious": {"accept_challenge_prob": 0.30, "skip_prob": 0.10, "prefer_higher_difficulty": False},
                "inconsistent": {"accept_challenge_prob": 0.50, "skip_prob": 0.25, "prefer_higher_difficulty": False},
            },
        },
        "reward": {},
    }


@pytest.fixture
def simulator(config):
    return UserSimulator(config)


@pytest.fixture
def sample_tasks():
    return [
        {"task_id": "T001", "name": "Easy Stretch", "difficulty": 1, "category": "flexibility", "body_part": "knee", "contraindications": []},
        {"task_id": "T050", "name": "Moderate Strength", "difficulty": 3, "category": "strength", "body_part": "knee", "contraindications": []},
        {"task_id": "T090", "name": "Hard Cardio", "difficulty": 5, "category": "cardio", "body_part": "full_body", "contraindications": ["acute_inflammation"]},
    ]


@pytest.fixture
def low_pain_state():
    return {"day": 10, "pain_level": 2.0, "fatigue_level": 2.0, "rom": 110, "mood": 8.0}


@pytest.fixture
def high_pain_state():
    return {"day": 10, "pain_level": 8.0, "fatigue_level": 7.0, "rom": 70, "mood": 3.0}


# ── Test: simulate_daily_state ────────────────────────────────────────────

def test_daily_state_returns_required_keys(simulator):
    state = simulator.simulate_daily_state("U001", 0)
    for key in ["day", "pain_level", "fatigue_level", "rom", "mood"]:
        assert key in state, f"Missing key: {key}"
    assert 0 <= state["pain_level"] <= 10
    assert 0 <= state["fatigue_level"] <= 10


def test_daily_state_reproducible(config):
    s1 = UserSimulator(config)
    s2 = UserSimulator(config)
    state1 = s1.simulate_daily_state("U001", 5)
    state2 = s2.simulate_daily_state("U001", 5)
    assert state1 == state2, "Same seed should produce identical states"


# ── Test: simulate_choice ─────────────────────────────────────────────────

def test_choice_returns_valid_task_id(simulator, sample_tasks, low_pain_state):
    tid, meta = simulator.simulate_choice("U001", sample_tasks, low_pain_state)
    assert tid in {t["task_id"] for t in sample_tasks}
    assert "action" in meta
    assert meta["action"] in ("completed", "skipped")


def test_choice_empty_options(simulator, low_pain_state):
    tid, meta = simulator.simulate_choice("U001", [], low_pain_state)
    assert tid == ""
    assert meta["action"] == "skipped"


def test_choice_difficulty_affects_scoring(config):
    """Run many trials: under high pain, easy tasks should be chosen more often."""
    sim = UserSimulator(config)
    high_pain = {"day": 5, "pain_level": 8.0, "fatigue_level": 7.0, "rom": 80, "mood": 3.0}
    tasks = [
        {"task_id": "EASY", "difficulty": 1, "category": "rest", "body_part": "knee", "contraindications": []},
        {"task_id": "HARD", "difficulty": 5, "category": "cardio", "body_part": "full_body", "contraindications": []},
    ]
    # Use cautious user (prefers easier)
    easy_count = 0
    n_trials = 200
    for i in range(n_trials):
        sim.rng = np.random.default_rng(i)
        tid, _ = sim.simulate_choice("U004", tasks, high_pain)
        if tid == "EASY":
            easy_count += 1
    # Easy should be picked significantly more often (> 60%)
    assert easy_count / n_trials > 0.60, f"Easy task chosen only {easy_count}/{n_trials} times under high pain"


# ── Test: simulate_feedback ───────────────────────────────────────────────

def test_feedback_returns_required_keys(simulator, sample_tasks, low_pain_state):
    fb = simulator.simulate_feedback("U001", sample_tasks[0], low_pain_state)
    for key in ["completed", "pain_change", "fatigue_change", "satisfaction"]:
        assert key in fb, f"Missing key: {key}"


def test_feedback_difficulty_affects_pain(config):
    """Higher difficulty should produce more pain change on average."""
    sim = UserSimulator(config)
    state = {"day": 5, "pain_level": 4.0, "fatigue_level": 4.0, "rom": 100, "mood": 6.0}
    easy = {"task_id": "E", "difficulty": 1, "category": "strength", "body_part": "knee"}
    hard = {"task_id": "H", "difficulty": 5, "category": "strength", "body_part": "knee"}

    easy_pain, hard_pain = [], []
    for seed in range(100):
        sim.rng = np.random.default_rng(seed)
        fb_e = sim.simulate_feedback("U001", easy, state)
        sim.rng = np.random.default_rng(seed + 10000)
        fb_h = sim.simulate_feedback("U001", hard, state)
        easy_pain.append(fb_e["pain_change"])
        hard_pain.append(fb_h["pain_change"])

    assert np.mean(hard_pain) > np.mean(easy_pain), \
        f"Hard tasks should cause more pain: easy={np.mean(easy_pain):.2f} hard={np.mean(hard_pain):.2f}"


def test_feedback_gentle_category_reduces_pain(config):
    """Flexibility/rest tasks should reduce pain more than strength/cardio."""
    sim = UserSimulator(config)
    state = {"day": 5, "pain_level": 6.0, "fatigue_level": 4.0, "rom": 100, "mood": 5.0}
    gentle = {"task_id": "G", "difficulty": 2, "category": "flexibility", "body_part": "knee"}
    harsh  = {"task_id": "H", "difficulty": 2, "category": "strength", "body_part": "knee"}

    gentle_pain, harsh_pain = [], []
    for seed in range(100):
        sim.rng = np.random.default_rng(seed)
        fb_g = sim.simulate_feedback("U001", gentle, state)
        sim.rng = np.random.default_rng(seed + 10000)
        fb_h = sim.simulate_feedback("U001", harsh, state)
        gentle_pain.append(fb_g["pain_change"])
        harsh_pain.append(fb_h["pain_change"])

    assert np.mean(gentle_pain) < np.mean(harsh_pain), \
        f"Gentle category should cause less pain: gentle={np.mean(gentle_pain):.2f} harsh={np.mean(harsh_pain):.2f}"


# ── Test: compute_reward ──────────────────────────────────────────────────

def test_reward_in_unit_interval(simulator, sample_tasks, low_pain_state):
    fb = simulator.simulate_feedback("U001", sample_tasks[0], low_pain_state)
    r = simulator.compute_reward(fb, low_pain_state, sample_tasks[0])
    assert 0.0 <= r <= 1.0, f"Reward out of range: {r}"


def test_reward_difficulty_sensitivity(config):
    """Reward should differ for easy vs hard tasks given same user state."""
    sim = UserSimulator(config)
    state = {"pain_level": 2.0, "fatigue_level": 2.0}
    fb_completed = {"completed": True, "action": "completed", "pain_change": 0.1}
    easy = {"task_id": "E", "difficulty": 1, "category": "rest"}
    hard = {"task_id": "H", "difficulty": 5, "category": "cardio"}
    r_easy = sim.compute_reward(fb_completed, state, easy)
    r_hard = sim.compute_reward(fb_completed, state, hard)
    assert r_easy != r_hard, "Reward must differ for different difficulty tasks"
