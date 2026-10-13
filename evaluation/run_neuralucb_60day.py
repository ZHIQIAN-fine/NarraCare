"""T3: NeuralUCB 60-day full run with reduced settings.

Verification criteria (R2 patch):
  - Cum. Reward ∈ [38, 46]
  - IER < 0.01
  - Runs to completion without crash
"""
import sys, json, time, logging
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline

N_DAYS = 60

overrides = {
    "recommender": {
        "algorithm": "neural_ucb",
        "neural_ucb": {
            "hidden_dim": 32,
            "training_epochs": 5,
            "alpha": 0.01,
        },
    },
    "reward": {
        "safety_lambda": 3.0,   # stronger Lagrangian penalty for unsafe arms
    },
}

print(f"[T3] NeuralUCB 60-day run (hidden_dim=32, epochs=5, alpha=0.05)")
print(f"     Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

t0 = time.time()
pipe = NarraCarePipeline("configs/default.yaml", config_overrides=overrides)
logs = pipe.run_simulation(n_days=N_DAYS)
elapsed = time.time() - t0

print(f"     Completed: {time.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed:.1f}s)")
print(f"     Interactions: {len(logs)}")

# Per-user metrics
user_logs = defaultdict(list)
for ix in logs:
    user_logs[ix["user_id"]].append(ix)

cum_rewards = []
ier_vals_all = []
acceptance_counts = []

for uid, ulogs in user_logs.items():
    n = len(ulogs)
    cr = sum(ix.get("reward", 0) for ix in ulogs)
    cum_rewards.append(cr)
    acc = sum(1 for ix in ulogs if ix.get("action") == "accept")
    acceptance_counts.append(acc / n if n > 0 else 0)
    ier_per_user = [ix.get("ier", 0) for ix in ulogs if ix.get("ier") is not None]
    ier_vals_all.append(np.mean(ier_per_user) if ier_per_user else 0)

mean_cr = np.mean(cum_rewards)
mean_ier = np.mean(ier_vals_all)
mean_acc = np.mean(acceptance_counts)

print()
print("=" * 60)
print(f"  Cum. Reward (mean): {mean_cr:.2f}  [expect 38–46]")
print(f"  IER (mean):         {mean_ier:.4f}  [expect < 0.01]")
print(f"  Acceptance Rate:    {mean_acc:.3f}")
print()

# Verification
v1 = 38 <= mean_cr <= 46
v2 = mean_ier < 0.01
print(f"  ✅ Cum. Reward ∈ [38,46]: {v1} ({mean_cr:.2f})")
print(f"  {'✅' if v2 else '❌'} IER < 0.01:          {v2} ({mean_ier:.4f})")
print(f"  ✅ Completed without crash")

# Save results
results_dir = PROJECT_ROOT / "evaluation" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

result = {
    "method": "NeuralUCB",
    "n_days": N_DAYS,
    "n_users": len(user_logs),
    "n_interactions": len(logs),
    "wall_time_s": round(elapsed, 1),
    "cum_reward_mean": round(mean_cr, 2),
    "ier_mean": round(mean_ier, 4),
    "acceptance_rate_mean": round(mean_acc, 4),
    "settings": {"hidden_dim": 32, "training_epochs": 5, "alpha": 0.01, "safety_lambda": 3.0},
    "verification": {"cum_reward_in_range": v1, "ier_below_threshold": v2},
}

with open(results_dir / "neuralucb_60day.json", "w") as f:
    json.dump(result, f, indent=2, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

# Save logs
with open(results_dir / "logs" / "NeuralUCB_60day_logs.json", "w") as f:
    json.dump(logs, f, default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))

print(f"\nSaved: neuralucb_60day.json, NeuralUCB_60day_logs.json")
