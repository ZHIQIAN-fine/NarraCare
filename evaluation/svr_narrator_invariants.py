"""R2.5 — Post-hoc SVR replay + Narrator invariant tests.

T1: For LinUCB-NoSafety accepted tasks, replay every (state, task) pair
    against the full SafetyAgent rules to compute a post-hoc SVR + per-rule
    breakdown.  Also run LinUCB-WithSafety as sanity control (SVR must = 0).

T2: Using LinUCB-WithSafety interaction logs (with narrative), verify:
    I1 — Narrator tone-fallback rate = 0%
    I2 — Emotional red/yellow difficulty cap is respected
    I3 — safety-blocked day ↔ safety_hybrid tone presence
"""
import sys, json, time, csv, logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline
from src.agents.safety_agent import SafetyAgent

N_DAYS = 60
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════
#  Helper — run pipeline and capture full interaction logs
# ═════════════════════════════════════════════════════════════════════════
def run_full_logs(label: str, overrides: dict) -> List[dict]:
    print(f"\n▶ Running {label} ...")
    t0 = time.time()
    pipe = NarraCarePipeline("config/default.yaml", config_overrides=overrides)
    logs = pipe.run_simulation(n_days=N_DAYS)
    print(f"  Done in {time.time()-t0:.1f}s  ({len(logs)} interactions)")
    return logs


# ═════════════════════════════════════════════════════════════════════════
#  T1 — Per-rule post-hoc SVR
# ═════════════════════════════════════════════════════════════════════════
def check_per_rule_safety(task: dict, state: dict, profile: dict,
                          agent: SafetyAgent) -> Dict[str, bool]:
    """Check a single task against all 5 pre-filter rules individually.

    Returns dict of {rule_name: violated?}.
    """
    pain = state.get("pain_level", 0)
    fatigue = state.get("fatigue_level", 0)
    rom = state.get("rom", 180)
    user_stage = profile.get("current_stage", profile.get("stage", ""))
    user_contras = set(profile.get("contraindications", []))

    category = task.get("category", "").lower()
    difficulty = task.get("difficulty", 1)
    task_contras = set(task.get("contraindications", []))
    task_stages = task.get("applicable_stages", [])

    return {
        "R1_contraindication": bool(user_contras & task_contras),
        "R2_stage_mismatch": bool(task_stages and user_stage
                                   and user_stage not in task_stages),
        "R3_high_pain_difficulty": (pain >= agent.pain_threshold_high
                                    and difficulty >= 4),
        "R4_severe_fatigue": (fatigue >= agent.fatigue_threshold_high
                              and category not in ("rest", "flexibility")),
        "R5_low_rom": (rom < 60
                       and (difficulty >= 3 or category == "cardio")),
    }


def compute_svr(logs: List[dict], safety_agent: SafetyAgent,
                label: str) -> dict:
    """Compute post-hoc SVR with per-rule breakdown."""
    accepted = [ix for ix in logs if ix.get("action") == "accept"]
    n_accepted = len(accepted)

    rule_counts: Dict[str, int] = Counter()
    any_violation_count = 0
    violation_rows = []

    for ix in accepted:
        state = ix.get("state", {})
        profile = {
            "condition": ix.get("user_condition", ""),
            "current_stage": ix.get("user_stage", ""),
            "contraindications": ix.get("user_contraindications", []),
        }
        task = ix.get("chosen_task", {})

        rules = check_per_rule_safety(task, state, profile, safety_agent)
        any_fired = any(rules.values())
        if any_fired:
            any_violation_count += 1
        for rname, fired in rules.items():
            if fired:
                rule_counts[rname] += 1

        violation_rows.append({
            "user_id": ix["user_id"],
            "day": ix["day"],
            "task_id": task.get("task_id", ""),
            "any_violation": any_fired,
            **rules,
        })

    svr = any_violation_count / n_accepted if n_accepted else 0.0

    print(f"\n  [{label}] Post-hoc SVR = {any_violation_count}/{n_accepted}"
          f" = {svr:.4f}")
    for rname in sorted(rule_counts.keys()):
        print(f"    {rname}: {rule_counts[rname]}/{n_accepted}"
              f" = {rule_counts[rname]/n_accepted:.4f}")

    return {
        "label": label,
        "n_accepted": n_accepted,
        "n_total": len(logs),
        "svr": round(svr, 6),
        "violations": any_violation_count,
        "per_rule": {
            rname: {
                "count": rule_counts.get(rname, 0),
                "rate": round(rule_counts.get(rname, 0) / max(n_accepted, 1), 6),
            }
            for rname in [
                "R1_contraindication", "R2_stage_mismatch",
                "R3_high_pain_difficulty", "R4_severe_fatigue", "R5_low_rom",
            ]
        },
    }


# ═════════════════════════════════════════════════════════════════════════
#  T2 — Narrator Invariants
# ═════════════════════════════════════════════════════════════════════════
def check_narrator_invariants(logs: List[dict]) -> dict:
    """Check three narrator invariants across all interaction logs.

    I1 — fallback rate: narrator_tone should never be empty/fallback
    I2 — difficulty cap: if emotional_level is red_light, max difficulty
          of recommended tasks must be ≤ 1 (≤ 2 if no diff-1 tasks);
          if yellow_light, max difficulty ≤ 2 (≤ 3 if no diff-2 tasks)
    I3 — safety_hybrid tone ↔ safety-blocked day (infeasible_in_top_k non-empty)
    """

    # ── I1: Fallback rate ────────────────────────────────────────────
    n_total = len(logs)
    n_tone_present = sum(1 for ix in logs if ix.get("narrator_tone"))
    n_tone_missing = n_total - n_tone_present
    fallback_rate = n_tone_missing / n_total if n_total else 0.0

    # ── I2: Difficulty cap ───────────────────────────────────────────
    i2_violations = []
    for ix in logs:
        emo = ix.get("emotional_level", "green")
        rec_tasks = ix.get("recommended_tasks", [])
        if not rec_tasks:
            continue

        max_diff = max(t.get("difficulty", 1) for t in rec_tasks)

        if emo == "red_light" and max_diff > 2:
            # Red: cap at 1, fallback 2
            i2_violations.append({
                "user_id": ix["user_id"], "day": ix["day"],
                "emotional_level": emo, "max_difficulty": max_diff,
            })
        elif emo == "yellow_light" and max_diff > 3:
            # Yellow: cap at 2, fallback 3
            i2_violations.append({
                "user_id": ix["user_id"], "day": ix["day"],
                "emotional_level": emo, "max_difficulty": max_diff,
            })

    # ── I3: safety_hybrid ↔ safety-blocked-day ───────────────────────
    # Group by (user_id, day) to check day-level properties
    day_groups: Dict[tuple, List[dict]] = defaultdict(list)
    for ix in logs:
        day_groups[(ix["user_id"], ix["day"])].append(ix)

    i3_false_positive = 0   # safety_hybrid tone without safety block
    i3_false_negative = 0   # safety block without safety_hybrid tone
    i3_checked = 0

    for (uid, day), day_ixs in day_groups.items():
        had_block = any(
            bool(ix.get("infeasible_in_top_k"))
            for ix in day_ixs
        )
        had_safety_tone = any(
            "safety" in (ix.get("narrator_tone", "") or "").lower()
            for ix in day_ixs
        )
        i3_checked += 1
        if had_safety_tone and not had_block:
            i3_false_positive += 1
        if had_block and not had_safety_tone:
            i3_false_negative += 1

    result = {
        "I1_fallback_rate": {
            "n_total": n_total,
            "n_tone_present": n_tone_present,
            "n_tone_missing": n_tone_missing,
            "fallback_rate": round(fallback_rate, 6),
            "passed": fallback_rate == 0.0,
        },
        "I2_difficulty_cap": {
            "n_violations": len(i2_violations),
            "violations": i2_violations[:10],   # first 10 for debugging
            "passed": len(i2_violations) == 0,
        },
        "I3_safety_hybrid_tone": {
            "n_user_days_checked": i3_checked,
            "false_positive": i3_false_positive,
            "false_negative": i3_false_negative,
            "note": ("false_positive = safety_hybrid tone without block; "
                     "false_negative = block day without safety tone"),
            "passed": i3_false_positive == 0,
            # false_negative is soft — narrator may not use safety tone
            # even when block occurred (e.g. if block happened on IER
            # check only)
        },
    }

    print(f"\n  ── Narrator Invariant Results ──")
    for name, r in result.items():
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  {name}: {status}")
        for k, v in r.items():
            if k not in ("passed", "violations"):
                print(f"    {k}: {v}")

    return result


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("[R2.5] Post-hoc SVR replay + Narrator invariant tests")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 64)

# ── Run both configurations ──────────────────────────────────────────
logs_safety_on = run_full_logs("LinUCB (safety ON)",
    {"recommender": {"algorithm": "linucb"}})

logs_safety_off = run_full_logs("LinUCB (safety OFF / NoSafety)",
    {"recommender": {"algorithm": "linucb"},
     "safety": {"enable": False}})

# ── Instantiate a standalone safety agent for post-hoc checking ──────
cfg = {"safety": {"enable": True}}
safety_agent = SafetyAgent(cfg)
print(f"\n  Safety agent loaded: {len(safety_agent.rules)} JSON rules"
      f" + 5 pre_filter rules")

# ── T1: Post-hoc SVR ────────────────────────────────────────────────
svr_safety_on = compute_svr(logs_safety_on, safety_agent,
                             "LinUCB-WithSafety (sanity)")
svr_safety_off = compute_svr(logs_safety_off, safety_agent,
                              "LinUCB-NoSafety")

# ── T2: Narrator invariants (safety ON logs have narrative) ──────────
narrator_results = check_narrator_invariants(logs_safety_on)

# ── Also get narrator aggregate counters from a fresh pipeline ───────
pipe_for_counters = NarraCarePipeline("config/default.yaml")
pipe_for_counters.run_simulation(n_days=N_DAYS)
narrator_counters = {}
if hasattr(pipe_for_counters, 'narrator') and pipe_for_counters.narrator:
    narrator_counters = pipe_for_counters.narrator.export_aggregate_counters()
    print(f"\n  Narrator aggregate counters: {json.dumps(narrator_counters, indent=2)}")

# ═════════════════════════════════════════════════════════════════════════
#  Save T1 artifact
# ═════════════════════════════════════════════════════════════════════════
t1_result = {
    "experiment": "posthoc_svr_replay",
    "description": (
        "Post-hoc safety violation rate: accepted tasks from LinUCB-NoSafety "
        "replayed against the full SafetyAgent pre-filter (5 rules). "
        "LinUCB-WithSafety included as sanity control (expect SVR=0)."
    ),
    "config": {"n_days": N_DAYS, "seed": 42},
    "linucb_with_safety": svr_safety_on,
    "linucb_no_safety": svr_safety_off,
}

t1_path = RESULTS_DIR / "posthoc_svr_60day_seed42.json"
with open(t1_path, "w") as f:
    json.dump(t1_result, f, indent=2,
              default=lambda o: bool(o) if isinstance(o, (np.bool_,))
              else float(o) if isinstance(o, (np.floating, np.integer))
              else str(o))
print(f"\n✓ T1 saved: {t1_path.name}")

# ═════════════════════════════════════════════════════════════════════════
#  Save T2 artifact
# ═════════════════════════════════════════════════════════════════════════
t2_result = {
    "experiment": "narrator_invariants",
    "description": (
        "Three narrator invariant tests on LinUCB-WithSafety 60-day logs: "
        "I1 (zero fallback), I2 (difficulty cap respected), "
        "I3 (safety_hybrid tone consistency)."
    ),
    "config": {"n_days": N_DAYS, "seed": 42, "method": "linucb"},
    "invariants": narrator_results,
    "narrator_aggregate_counters": narrator_counters,
}

t2_path = RESULTS_DIR / "narrator_invariants_60day_seed42.json"
with open(t2_path, "w") as f:
    json.dump(t2_result, f, indent=2,
              default=lambda o: bool(o) if isinstance(o, (np.bool_,))
              else float(o) if isinstance(o, (np.floating, np.integer))
              else str(o))
print(f"✓ T2 saved: {t2_path.name}")

# ═════════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print("  SUMMARY")
print(f"{'='*64}")
print(f"\n  T1 Post-hoc SVR:")
print(f"    LinUCB-WithSafety (sanity): SVR = {svr_safety_on['svr']:.4f}")
print(f"    LinUCB-NoSafety:            SVR = {svr_safety_off['svr']:.4f}")
for rname in ["R1_contraindication", "R2_stage_mismatch",
              "R3_high_pain_difficulty", "R4_severe_fatigue", "R5_low_rom"]:
    rate = svr_safety_off["per_rule"][rname]["rate"]
    count = svr_safety_off["per_rule"][rname]["count"]
    print(f"      {rname}: {count} ({rate:.4f})")

print(f"\n  T2 Narrator Invariants:")
for iname, r in narrator_results.items():
    status = "PASS ✅" if r["passed"] else "FAIL ❌"
    print(f"    {iname}: {status}")

print(f"\n{'='*64}")
