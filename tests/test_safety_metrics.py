"""Regression checks for ungated IER and accepted-task five-rule SVR."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import NarraCarePipeline
from evaluation.metrics import safety_violation_rate


@pytest.fixture
def pipe():
    return NarraCarePipeline(config_overrides={"recommender": {"algorithm": "saps"}})


SAFE = {"task_id": "safe", "category": "rest", "difficulty": 1,
        "contraindications": [], "applicable_stages": [], "intensity": "low"}
STATE = {"pain_level": 2, "fatigue_level": 2, "rom": 100, "mood": 7}
PROFILE = {"current_stage": "early", "contraindications": ["injury"]}


def log_task(pipe, task, state=None, action="accept"):
    return pipe._build_log(
        "U001", 0, 0, state or STATE,
        {"recommended": [task], "profile": PROFILE},
        task["task_id"], action, 0.0, state or STATE, {}, "",
    )


@pytest.mark.parametrize("task_change,state_change", [
    ({"contraindications": ["injury"]}, {}),
    ({"applicable_stages": ["late"]}, {}),
    ({"difficulty": 4}, {"pain_level": 7}),
    ({"category": "strength"}, {"fatigue_level": 8}),
    ({"difficulty": 3}, {"rom": 59}),
    ({"category": "cardio"}, {"rom": 59}),
])
def test_log_detects_each_full_safety_rule(pipe, task_change, state_change):
    task = dict(SAFE, **task_change)
    state = dict(STATE, **state_change)
    record = log_task(pipe, task, state)
    assert record["safety_violation"] is True
    assert safety_violation_rate([record]) == 1.0
    # Rejecting an unsafe offer is not an execution violation.
    assert log_task(pipe, task, state, action="skip")["safety_violation"] is False


@pytest.mark.parametrize("overrides", [
    {"safety": {"enable": False}},
    {"safety": {"drop_rule": "PF_R5"}},
])
def test_audit_remains_active_when_runtime_filter_is_disabled(overrides):
    pipeline = NarraCarePipeline(config_overrides=overrides)
    assert log_task(pipeline, dict(SAFE, difficulty=3), dict(STATE, rom=59))["safety_violation"]


@pytest.mark.parametrize("task_change,state_change", [
    ({"difficulty": 4}, {"pain_level": 6.99}),
    ({"category": "strength"}, {"fatigue_level": 7.99}),
    ({"difficulty": 3}, {"rom": 60}),
    ({}, {"fatigue_level": 8}),
])
def test_boundary_and_safe_tasks_do_not_count_as_violations(pipe, task_change, state_change):
    assert not log_task(pipe, dict(SAFE, **task_change), dict(STATE, **state_change))["safety_violation"]


def test_svr_denominator_is_accepted_tasks_and_legacy_replay_includes_rom(pipe):
    unsafe = log_task(pipe, dict(SAFE, difficulty=3), dict(STATE, rom=59))
    safe = log_task(pipe, SAFE)
    skip = log_task(pipe, SAFE, action="skip")
    assert safety_violation_rate([unsafe, safe, skip]) == .5
    legacy = [{k: v for k, v in row.items() if k != "safety_violation"}
              for row in [unsafe, safe, skip]]
    assert safety_violation_rate(legacy) == .5
    assert safety_violation_rate([skip]) == 0.0
    assert safety_violation_rate([]) == 0.0


def test_audit_uses_configured_thresholds():
    pipeline = NarraCarePipeline(config_overrides={"safety": {"pain_threshold_high": 9}})
    record = log_task(pipeline, dict(SAFE, difficulty=4), dict(STATE, pain_level=8))
    assert not record["safety_violation"]
    assert safety_violation_rate([record]) == 0.0


def test_accepted_task_must_be_in_recommendations(pipe):
    with pytest.raises(ValueError, match="absent from recommendations"):
        pipe._build_log("U001", 0, 0, STATE, {"recommended": []},
                        "missing", "accept", 0, STATE, {}, "")


def prepare_unsafe_ranking(pipe, monkeypatch):
    unsafe = dict(SAFE, task_id="unsafe", difficulty=5, category="strength", intensity="high")
    pipe.task_library = [copy.deepcopy(SAFE), unsafe]
    pipe.task_lookup = {t["task_id"]: t for t in pipe.task_library}
    monkeypatch.setattr(pipe.recommender.model, "_reward_ucb", lambda aid, x: 100. if aid == "unsafe" else 0.)
    return dict(STATE, pain_level=8)


def test_pipeline_ier_reads_ungated_saps_topk(pipe, monkeypatch):
    state = prepare_unsafe_ranking(pipe, monkeypatch)
    result = pipe.run_single_turn("U001", state)
    assert [t["task_id"] for t in result["recommended"]] == ["safe"]
    assert result["raw_top_k_ids"] == ["unsafe", "safe"]
    assert result["infeasible_in_top_k"] == ["unsafe"]
    assert result["ier"] == .5
    # Blocking another candidate does not make the accepted safe task unsafe.
    record = pipe._build_log("U001", 0, 0, state, result, "safe", "accept", 0, state, {}, "")
    assert not record["safety_violation"]


def test_ungated_ier_still_drives_existing_negative_reward_update(pipe, monkeypatch):
    state = prepare_unsafe_ranking(pipe, monkeypatch)
    monkeypatch.setattr(pipe.simulator, "get_all_user_ids", lambda: ["U001"])
    monkeypatch.setattr(pipe.simulator, "simulate_daily_state", lambda uid, day: dict(state))
    monkeypatch.setattr(pipe.simulator, "simulate_choice", lambda *args: ("safe", {"chosen_task": SAFE, "action": "completed"}))
    calls = []
    original = pipe.recommender.update
    def observe(context, task_id, reward):
        calls.append((task_id, reward))
        return original(context, task_id, reward)
    monkeypatch.setattr(pipe.recommender, "update", observe)
    logs = pipe.run_simulation(n_days=1)
    assert logs[0]["ier"] == .5
    assert any(task == "unsafe" and reward < 0 for task, reward in calls)


def test_raw_ranking_replaces_gated_call_without_extra_rng_draws():
    # Build through the agent so the production SAPS configuration is used.
    a = NarraCarePipeline(config_overrides={"recommender": {"algorithm": "saps"}}).recommender.model
    b = NarraCarePipeline(config_overrides={"recommender": {"algorithm": "saps"}}).recommender.model
    context = np.zeros(a.d)
    ids = ["safe", "unsafe"]
    for model in [a, b]:
        model.set_safe_ids({"safe"})
        model.select(context, ["safe"], 3)
    a.select_raw_topk(context, ids, 5)
    b.select(context, ids, 5)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


def test_formal_runner_uses_accepted_svr_denominator():
    from evaluation.run_submission import mean_user_svr
    rows = [{"user_id": "U001", "action": action, "safety_violation": violation,
             "reward": 0., "ier": 0., "chosen_task_id": "T001"}
            for action, violation in [("accept", True), ("accept", False), ("skip", False)]]
    assert mean_user_svr(rows) == .5
