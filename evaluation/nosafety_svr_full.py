"""R2.6 T1 补数据 — NoSafety post-hoc SVR full breakdown with overlap matrix.

Reruns LinUCB-NoSafety (safety.enable=False, seed=42, 60-day, 20-user),
then replays every accepted task against the full 5-rule pre-filter to produce
per-rule hits, total violations (any-rule), and all pairwise overlap counts.

Output: evaluation/results/nosafety_svr_posthoc_full.json
"""
import sys, json, time
from pathlib import Path
from collections import Counter
from itertools import combinations
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import NarraCarePipeline
from src.agents.safety_agent import SafetyAgent

N_DAYS = 60
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RULE_IDS = ["PF_R1", "PF_R2", "PF_R3", "PF_R4", "PF_R5"]

RULE_DEFINITIONS = {
    "PF_R1": "user_contras & task_contras (contraindication tag overlap)",
    "PF_R2": "user_stage not in task_stages (stage mismatch)",
    "PF_R3": "pain >= 7 and difficulty >= 4",
    "PF_R4": "fatigue >= 8 and category not in {rest, flexibility}",
    "PF_R5": "rom < 60 and (difficulty >= 3 or category == 'cardio')",
}


def check_per_rule(task: dict, state: dict, profile: dict,
                   agent: SafetyAgent) -> Dict[str, bool]:
    """Check a single task against all 5 pre-filter rules individually."""
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
        "PF_R1": bool(user_contras & task_contras),
        "PF_R2": bool(task_stages and user_stage and user_stage not in task_stages),
        "PF_R3": (pain >= agent.pain_threshold_high and difficulty >= 4),
        "PF_R4": (fatigue >= agent.fatigue_threshold_high
                  and category not in ("rest", "flexibility")),
        "PF_R5": (rom < 60 and (difficulty >= 3 or category == "cardio")),
    }


# ── Run NoSafety ──────────────────────────────────────────────────────────
print("=" * 64)
print("[R2.6 T1 补数据] NoSafety SVR post-hoc full breakdown")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 64)

t0 = time.time()
pipe = NarraCarePipeline("configs/default.yaml", config_overrides={
    "recommender": {"algorithm": "linucb"},
    "safety": {"enable": False},
})
logs = pipe.run_simulation(n_days=N_DAYS)
print(f"  Run completed in {time.time()-t0:.1f}s  ({len(logs)} interactions)")

# ── Post-hoc replay ──────────────────────────────────────────────────────
safety_agent = SafetyAgent({"safety": {"enable": True}})
accepted = [ix for ix in logs if ix.get("action") == "accept"]
n_accepted = len(accepted)
print(f"  Accepted tasks: {n_accepted}")

per_rule_hits: Dict[str, int] = Counter()
any_rule_count = 0
overlap_counter: Dict[str, int] = Counter()

for ix in accepted:
    state = ix.get("state", {})
    profile = {
        "condition": ix.get("user_condition", ""),
        "current_stage": ix.get("user_stage", ""),
        "contraindications": ix.get("user_contraindications", []),
    }
    task = ix.get("chosen_task", {})

    fired = check_per_rule(task, state, profile, safety_agent)
    fired_rules = [r for r in RULE_IDS if fired[r]]

    if fired_rules:
        any_rule_count += 1

    for r in fired_rules:
        per_rule_hits[r] += 1

    # Pairwise overlaps
    for r_a, r_b in combinations(fired_rules, 2):
        key = f"{r_a}&{r_b}"
        overlap_counter[key] += 1

svr = any_rule_count / n_accepted if n_accepted else 0.0

# ── Build overlap matrix (all pairs, even zeros) ─────────────────────────
overlap_matrix = {}
for r_a, r_b in combinations(RULE_IDS, 2):
    key = f"{r_a}&{r_b}"
    overlap_matrix[key] = overlap_counter.get(key, 0)

# ── Result ────────────────────────────────────────────────────────────────
result = {
    "source_run": "linucb_nosafety_60day.json",
    "commit": "7c17d4c",
    "total_accepted": n_accepted,
    "total_violating": any_rule_count,
    "svr": round(svr, 6),
    "per_rule_hits": {r: per_rule_hits.get(r, 0) for r in RULE_IDS},
    "any_rule_hits": any_rule_count,
    "overlap_matrix": overlap_matrix,
    "rule_definitions": RULE_DEFINITIONS,
}

out_path = RESULTS_DIR / "nosafety_svr_posthoc_full.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2,
              default=lambda o: bool(o) if isinstance(o, (np.bool_,))
              else float(o) if isinstance(o, (np.floating, np.integer))
              else str(o))

# ── Print summary ─────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  SVR = {any_rule_count}/{n_accepted} = {svr:.6f}")
print(f"\n  Per-rule hits:")
for r in RULE_IDS:
    c = per_rule_hits.get(r, 0)
    print(f"    {r}: {c:>4d}  ({c/n_accepted:.4f})  — {RULE_DEFINITIONS[r]}")
print(f"  Sum of per-rule hits: {sum(per_rule_hits.values())}")
print(f"  Any-rule (unique violating tasks): {any_rule_count}")
print(f"\n  Overlap matrix:")
for key, count in sorted(overlap_matrix.items()):
    if count > 0:
        print(f"    {key}: {count}")
print(f"\n✓ Saved: {out_path.name}")
print(f"{'='*64}")
