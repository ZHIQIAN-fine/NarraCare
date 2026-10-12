"""Step 3 Diagnosis: Recommendation overlap between methods.

For same user + same day, compare top-5 recommendation IDs across
LinUCB, Random, Popularity → compute Jaccard similarity.
"""
import sys, json, logging
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline

methods = {
    "LinUCB": {"recommender": {"algorithm": "linucb"}},
    "Random": {"recommender": {"algorithm": "random"}},
    "Popularity": {"recommender": {"algorithm": "popularity"}},
    "RuleBased": {"recommender": {"algorithm": "rule_based"}},
    "EpsilonGreedy": {"recommender": {"algorithm": "epsilon_greedy"}},
}

N_DAYS = 30

# Collect per-(user, day) -> set of recommended task_ids
method_recs = {}  # method_name -> {(uid, day): set(task_ids)}

for name, overrides in methods.items():
    print(f"Running {name}...", flush=True)
    pipe = NarraCarePipeline("config/default.yaml", config_overrides=overrides)
    logs = pipe.run_simulation(n_days=N_DAYS)

    recs = {}
    for ix in logs:
        key = (ix["user_id"], ix["day"])
        recs[key] = set(ix.get("recommended", []))
    method_recs[name] = recs
    print(f"  {name}: {len(logs)} interactions")

# Also count unique safe candidates per user-day for LinUCB run
# Re-run LinUCB to get candidate set sizes
print("\nCounting safe candidate set sizes...")
pipe = NarraCarePipeline("config/default.yaml",
                         config_overrides={"recommender": {"algorithm": "linucb"}})
user_ids = pipe.simulator.get_all_user_ids()
for uid in user_ids:
    raw = pipe.simulator.get_profile(uid)
    pipe.memory.set_profile(uid, pipe._normalize_profile(raw))

cand_sizes = []
for day in [0, 15, 29]:
    for uid in user_ids[:5]:
        state = pipe.simulator.simulate_daily_state(uid, day)
        all_cands = list(pipe.task_library)
        profile = pipe.memory.get_profile(uid)
        if pipe.safety_agent.enabled:
            safe = pipe.safety_agent.pre_filter(all_cands, state, profile)
        else:
            safe = all_cands
        cand_sizes.append(len(safe))

print(f"\nSafe candidate set sizes (sample): min={min(cand_sizes)}, max={max(cand_sizes)}, "
      f"mean={np.mean(cand_sizes):.1f}")

# Compute Jaccard between LinUCB and each other method
print("\n=== Jaccard Similarity: LinUCB vs Others (top-3 recs) ===")
linucb_recs = method_recs["LinUCB"]
all_keys = sorted(linucb_recs.keys())

for other_name in ["Random", "Popularity", "RuleBased", "EpsilonGreedy"]:
    other_recs = method_recs[other_name]
    jaccards = []
    for key in all_keys:
        s1 = linucb_recs.get(key, set())
        s2 = other_recs.get(key, set())
        if s1 or s2:
            j = len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0
            jaccards.append(j)
    if jaccards:
        print(f"  LinUCB vs {other_name:15s}: "
              f"mean_jaccard = {np.mean(jaccards):.4f}, "
              f"median = {np.median(jaccards):.4f}, "
              f"frac_identical = {sum(1 for j in jaccards if j > 0.99) / len(jaccards):.2%}, "
              f"frac_overlap>0.5 = {sum(1 for j in jaccards if j > 0.5) / len(jaccards):.2%}")

# Exact overlap analysis: same 3 tasks?
print("\n=== Exact Match Rate (all 3 recs identical) ===")
for other_name in ["Random", "Popularity", "RuleBased", "EpsilonGreedy"]:
    other_recs = method_recs[other_name]
    exact = 0
    total = 0
    for key in all_keys:
        s1 = linucb_recs.get(key, set())
        s2 = other_recs.get(key, set())
        if s1 and s2:
            total += 1
            if s1 == s2:
                exact += 1
    rate = exact / total if total else 0
    print(f"  LinUCB vs {other_name:15s}: {exact}/{total} = {rate:.2%}")

print("\nDone.")
