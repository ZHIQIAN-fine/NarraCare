"""Physical outcomes require execution; sampled scores never relax constraints."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from narracare.checkin import compose_checkin, compose_feedback
from narracare.conversation.tools import SessionTools
from src.pipeline import NarraCarePipeline


def pipeline(tau=2):
    return NarraCarePipeline(
        config_overrides={
            "recommender": {"algorithm": "saps", "saps": {"tau_safe": tau}}
        }
    )


def test_skip_never_calls_body_simulation_or_updates_beta(monkeypatch):
    p = pipeline()
    monkeypatch.setattr(p.simulator, "get_all_user_ids", lambda: ["U001"])
    monkeypatch.setattr(
        p.simulator,
        "simulate_choice",
        lambda u, tasks, s: (
            tasks[0]["task_id"],
            {"chosen_task": tasks[0], "action": "skipped"},
        ),
    )

    def forbidden(*args):
        pytest.fail("A skip must not generate or score a physical outcome")

    monkeypatch.setattr(p.simulator, "simulate_feedback", forbidden)
    monkeypatch.setattr(p.simulator, "compute_reward", forbidden)
    logs = p.run_simulation(n_days=3)
    assert len(logs) == 3
    for row in logs:
        assert row["recommended"]
        assert row["reward"] == 0 and not row["reward_observed"]
        assert row["post_state"] == row["state"]
        assert (
            not {"pain_change", "fatigue_change", "rom_change", "satisfaction"}
            & row["feedback"].keys()
        )
    assert p.simulator._user_tasks_completed == {}
    assert p.recommender.model.safe_alpha == p.recommender.model.safe_beta == {}
    assert p.memory.get_completion_rate("U001") == 0


def test_accepted_execution_updates_beta_and_only_completed_tasks_count(monkeypatch):
    p = pipeline()
    monkeypatch.setattr(p.simulator, "get_all_user_ids", lambda: ["U001"])
    monkeypatch.setattr(
        p.simulator,
        "simulate_choice",
        lambda u, tasks, s: (
            tasks[0]["task_id"],
            {"chosen_task": tasks[0], "action": "completed"},
        ),
    )
    logs = p.run_simulation(n_days=4)
    model = p.recommender.model
    observations = sum(a - model._a0 for a in model.safe_alpha.values()) + sum(
        b - model._b0 for b in model.safe_beta.values()
    )
    assert observations == 4
    assert sum(p.simulator._user_tasks_completed.values()) == sum(
        r["feedback"]["completed"] for r in logs
    )
    assert all(r["reward_observed"] and r["feedback"]["executed"] for r in logs)


@pytest.mark.parametrize("tau", [0, 2, 100])
def test_daily_tasks_remain_feasible_across_profiles_states_and_bonus(tau):
    p = pipeline(tau)
    # Boundary/coverage tests, NOT a tau performance search or clinical labels.
    states = [
        {"pain_level": 2, "fatigue_level": 2, "rom": 110, "mood": 7},
        {"pain_level": 7, "fatigue_level": 8, "rom": 59, "mood": 3},
        {"pain_level": 9, "fatigue_level": 9, "rom": 0, "mood": 2},
        {"pain_level": 3, "fatigue_level": 8, "rom": 60, "mood": 5},
    ]
    for uid in p.simulator.get_all_user_ids():
        for state in states:
            result = p.run_single_turn(uid, dict(state))
            tasks = result["recommended"]
            assert tasks, (uid, state)
            assert p.safety_agent.pre_filter(tasks, state, result["profile"]) == tasks
            assert p.safety_agent.post_check(tasks, state) == tasks


def test_recovery_fallback_is_checked_after_consecutive_intensity_cap(monkeypatch):
    p = pipeline()
    high = {
        "task_id": "high",
        "difficulty": 4,
        "intensity": "high",
        "category": "strength",
    }
    rest = {"task_id": "rest", "difficulty": 1, "intensity": "low", "category": "rest"}
    p.task_library = [high, rest]
    p.task_lookup = {t["task_id"]: t for t in p.task_library}
    monkeypatch.setattr(p.memory, "get_consecutive_high_intensity", lambda u: 3)
    monkeypatch.setattr(p.recommender, "recommend", lambda *a, **kw: [high])
    result = p.run_single_turn(
        "U001", {"pain_level": 2, "fatigue_level": 2, "rom": 110, "mood": 7}
    )
    assert result["recommended"] == [rest]


def test_catalogue_gap_does_not_inject_an_unsafe_rest_task():
    p = pipeline()
    p.task_library = [
        {
            "task_id": "bad-rest",
            "category": "rest",
            "difficulty": 1,
            "applicable_stages": ["nonexistent-stage"],
        }
    ]
    with pytest.raises(RuntimeError, match="Daily task coverage gap"):
        p.run_single_turn("U001", {"pain_level": 2, "fatigue_level": 2, "rom": 110})


def test_guided_checkin_flows_through_existing_evidence_guards():
    text = compose_checkin(
        3, 2, 7, "Unknown / not measured", None, ["balance"], ["flexibility"]
    )
    t = SessionTools()
    t.current_message = text
    t.dispatch(
        "update_checkin",
        {
            "fields": {"pain_level": 3, "fatigue_level": 2, "mood": 7, "rom": None},
            "evidence": text,
        },
    )
    t.dispatch(
        "set_preferences",
        {
            "preferred_categories": ["balance"],
            "avoid_categories": ["flexibility"],
            "evidence": text,
        },
    )
    t.dispatch("retrieve_candidates", {})
    t.dispatch("rank_candidates", {})
    t.dispatch("present_recommendation", {})
    assert t.state["rom"] is None and t.pending["category"] == "balance"


@pytest.mark.parametrize(
    "pain,preferred,avoided",
    [(None, [], []), (-1, [], []), (float("nan"), [], []), (3, ["rest"], ["rest"])],
)
def test_guided_checkin_rejects_missing_or_contradictory_inputs(
    pain, preferred, avoided
):
    with pytest.raises(ValueError):
        compose_checkin(pain, 2, 7, "Unknown / not measured", None, preferred, avoided)


def test_satisfaction_is_retained_without_changing_reward():
    rewards = []
    for satisfaction in [0.1, 0.9]:
        t = SessionTools()
        text = compose_checkin(3, 2, 7, "Measured", 110, ["balance"], [])
        t.current_message = text
        t.dispatch(
            "update_checkin",
            {
                "fields": {"pain_level": 3, "fatigue_level": 2, "mood": 7, "rom": 110},
                "evidence": text,
            },
        )
        t.dispatch(
            "set_preferences",
            {
                "preferred_categories": ["balance"],
                "avoid_categories": [],
                "evidence": text,
            },
        )
        for tool in [
            "retrieve_candidates",
            "rank_candidates",
            "present_recommendation",
        ]:
            t.dispatch(tool, {})
        t.current_message = compose_feedback("Completed", 3, 3, satisfaction)
        outcome = t.dispatch(
            "record_feedback",
            {
                "action": "completed",
                "post_pain": 3,
                "post_fatigue": 3,
                "satisfaction": satisfaction,
                "evidence": t.current_message,
            },
        )
        rewards.append(outcome["reward"])
        assert t.days[-1]["feedback"]["satisfaction"] == satisfaction
        assert t.days[-1]["feedback"]["satisfaction_used_in_reward"] is False
        assert t.state == {}  # The next day never inherits symptom measurements.
    assert rewards[0] == rewards[1]


def test_explicit_pain_inability_can_pause_without_inventing_an_outcome():
    t = SessionTools()
    t.current_message = compose_checkin(
        None, None, None, "Unknown / not measured", None, [], [], True
    )
    t.dispatch("pause_due_to_pain", {"evidence": t.current_message})
    assert t.day == 2 and t.days[0]["action"] == "paused"
    assert t.days[0]["reward"] is None and t.days[0]["feedback"] is None
    assert t.recommender.model.safe_alpha == {}


def test_pain_rating_alone_does_not_infer_inability_to_do_any_task():
    t = SessionTools()
    t.current_message = "Pain 9. Please recommend a task."
    with pytest.raises(ValueError, match="explicitly confirm"):
        t.dispatch("pause_due_to_pain", {"evidence": t.current_message})


def test_explicit_preference_cannot_be_silently_ignored_before_retrieval():
    t = SessionTools()
    t.current_message = compose_checkin(3, 2, 7, "Measured", 110, ["flexibility"], [])
    t.dispatch(
        "update_checkin",
        {
            "fields": {"pain_level": 3, "fatigue_level": 2, "mood": 7, "rom": 110},
            "evidence": t.current_message,
        },
    )
    with pytest.raises(ValueError, match="set_preferences"):
        t.dispatch("retrieve_candidates", {})


def test_blank_text_inputs_never_become_zero_ratings():
    with pytest.raises(ValueError, match="Enter pain"):
        compose_checkin("", "", "", "Unknown / not measured", "", [], [])
    text = compose_checkin("0", "2.5", "7", "Unknown / not measured", "", [], [])
    assert "pain 0" in text and "fatigue 2.5" in text and "ROM unknown" in text


def test_final_default_delivers_one_task_and_starts_in_forest():
    from narracare.conversation.tools import SessionTools

    p = pipeline()
    assert p.recommender.top_k == SessionTools().recommender.top_k == 1
    result = p.run_single_turn(
        "U001", {"pain_level": 3, "fatigue_level": 2, "rom": 110, "mood": 7}, day=0
    )
    assert len(result["recommended"]) == 1
    assert result["narrator_result"]["tone_originally_planned"] == "doctor"


@pytest.mark.parametrize(
    "condition,arc",
    [
        ("ACL_reconstruction", "ACL"),
        ("rotator_cuff_repair", "RotatorCuff"),
        ("chronic_lower_back_pain", "BackPain"),
    ],
)
def test_final_narrator_accepts_actual_profile_condition_names(condition, arc):
    p = pipeline()
    assert p.narrator._get_arc_context(
        {"condition": condition}, 16
    ) == p.narrator._get_arc_context({"condition": arc}, 16)
