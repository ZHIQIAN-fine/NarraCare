"""Step 4 Diagnosis: Reward distribution per method.

Runs all methods, collects per-interaction rewards, prints stats,
and saves histogram plot.
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
all_rewards = {}  # method -> list of rewards
all_actions = {}  # method -> {accept: n, skip: n}

for name, overrides in methods.items():
    print(f"Running {name}...", flush=True)
    pipe = NarraCarePipeline("config/default.yaml", config_overrides=overrides)
    logs = pipe.run_simulation(n_days=N_DAYS)

    rewards = [ix.get("reward", 0.0) for ix in logs]
    all_rewards[name] = rewards

    actions = defaultdict(int)
    for ix in logs:
        actions[ix.get("action", "unknown")] += 1
    all_actions[name] = dict(actions)

print("\n=== Reward Statistics per Method ===")
print(f"{'Method':<18s} {'mean':>7s} {'std':>7s} {'min':>7s} {'max':>7s} {'median':>7s} {'n':>5s}")
for name in methods:
    r = np.array(all_rewards[name])
    print(f"{name:<18s} {np.mean(r):7.4f} {np.std(r):7.4f} {np.min(r):7.4f} {np.max(r):7.4f} "
          f"{np.median(r):7.4f} {len(r):5d}")

print("\n=== Action Breakdown ===")
for name in methods:
    print(f"  {name}: {all_actions[name]}")

# Check reward range
print("\n=== Reward Value Distribution (binned) ===")
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
for name in ["LinUCB", "Random"]:
    r = np.array(all_rewards[name])
    counts, _ = np.histogram(r, bins=bins)
    total = len(r)
    print(f"\n  {name}:")
    for i in range(len(bins)-1):
        bar = "#" * int(40 * counts[i] / total)
        print(f"    [{bins[i]:.1f}-{bins[i+1]:.1f}): {counts[i]:4d} ({counts[i]/total:5.1%}) {bar}")

# Decompose reward for a sample interaction
print("\n=== Reward Decomposition (sample from LinUCB, day 15) ===")
pipe = NarraCarePipeline("config/default.yaml",
                         config_overrides={"recommender": {"algorithm": "linucb"}})
logs = pipe.run_simulation(n_days=30)
day15 = [ix for ix in logs if ix["day"] == 15]
if day15:
    sample = day15[0]
    fb = sample.get("feedback", {})
    state = sample.get("state", {})
    print(f"  User: {sample['user_id']}, Action: {sample['action']}")
    print(f"  Reward: {sample.get('reward', 0):.4f}")
    print(f"  Feedback: completed={fb.get('completed')}, pain_change={fb.get('pain_change')}, "
          f"fatigue_change={fb.get('fatigue_change')}, satisfaction={fb.get('satisfaction')}")
    print(f"  State: pain={state.get('pain_level')}, fatigue={state.get('fatigue_level')}")

    # Manual decomposition
    task_id = sample["chosen_task_id"]
    task = pipe.task_lookup.get(task_id, {})
    intensity = task.get("intensity", "low")
    intensity_val = {"low": 1, "medium": 2, "high": 3}.get(intensity, 1)
    w_completed, w_safety, w_diff, w_pain, w_drop = 0.40, 0.30, 0.15, 0.10, 0.05

    r_completed = 1.0 if fb.get("completed", False) else 0.0
    pain_change = fb.get("pain_change", 0)
    r_safety = 1.0 if pain_change <= 1.0 else max(0.0, 1.0 - (pain_change - 1.0) * 0.3)
    pain = state.get("pain_level", 5)
    fatigue = state.get("fatigue_level", 5)
    stress = (pain + fatigue) / 20.0
    ideal_intensity = 1 + (1 - stress) * 2
    mismatch = abs(intensity_val - ideal_intensity) / 2.0
    r_diff = max(0.0, 1.0 - mismatch)
    r_pain = max(0.0, 1.0 - max(0, pain_change) * 0.2)
    r_dropout = 0.0 if fb.get("action") == "skipped" else 1.0
    total = w_completed*r_completed + w_safety*r_safety + w_diff*r_diff + w_pain*r_pain + w_drop*r_dropout

    print(f"\n  Task: {task_id} '{task.get('name', '?')}' intensity={intensity} (val={intensity_val})")
    print(f"  r_completed={r_completed:.2f} (w={w_completed})")
    print(f"  r_safety={r_safety:.4f} (w={w_safety}), pain_change={pain_change}")
    print(f"  r_diff_match={r_diff:.4f} (w={w_diff}), ideal_intensity={ideal_intensity:.2f}, "
          f"mismatch={mismatch:.4f}")
    print(f"  r_pain={r_pain:.4f} (w={w_pain})")
    print(f"  r_dropout={r_dropout:.2f} (w={w_drop})")
    print(f"  Total = {total:.4f}")
    print(f"  (Note: intensity_val={intensity_val} is computed from "
          f"task.get('intensity','low') which is pipeline-augmented field)")

# Save histogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, name in enumerate(methods):
    r = np.array(all_rewards[name])
    axes[i].hist(r, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
    axes[i].set_title(f"{name}\nmean={np.mean(r):.3f} std={np.std(r):.3f}")
    axes[i].set_xlabel("Reward")
    axes[i].set_ylabel("Count")
    axes[i].axvline(np.mean(r), color='red', linestyle='--', alpha=0.8)
# hide last subplot if odd
if len(methods) < 6:
    axes[-1].set_visible(False)
plt.tight_layout()
out = PROJECT_ROOT / "evaluation" / "results" / "diag_reward_histograms.png"
plt.savefig(out, dpi=120)
print(f"\nSaved histogram: {out}")
print("\nDone.")
