"""R2.4: Re-run Popularity / RuleBased / EpsilonGreedy + LinUCB-NoSafety
under Step-B EU simulator (60-day × 20-user, seed=42).

Produces per-method JSON + CSV for §5.2 Table 5.1 and §5.4.2.
"""
import sys, json, time, csv, logging
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline

N_DAYS = 60
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_and_save(label: str, algo: str, overrides: dict, filename_stem: str):
    """Run one method, compute per-user metrics, save JSON + CSV."""
    print(f"\n▶ [{label}] running {algo} 60-day ...")
    t0 = time.time()
    pipe = NarraCarePipeline("configs/default.yaml", config_overrides=overrides)
    logs = pipe.run_simulation(n_days=N_DAYS)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(logs)} interactions)")

    user_logs = defaultdict(list)
    for ix in logs:
        user_logs[ix["user_id"]].append(ix)

    per_user = []
    for uid in sorted(user_logs.keys()):
        ul = user_logs[uid]
        n = len(ul)
        cr = sum(x.get("reward", 0) for x in ul)
        acc = sum(1 for x in ul if x.get("action") == "accept") / n if n else 0
        ier_vals = [x.get("ier", 0) for x in ul if x.get("ier") is not None]
        ier = float(np.mean(ier_vals)) if ier_vals else 0.0
        svr_cnt = sum(1 for x in ul if x.get("safety_violation", False))
        svr = svr_cnt / n if n else 0.0
        ts = sum(
            1 for x in ul
            if x.get("action") == "accept"
            and x.get("feedback", {}).get("completed", False)
        ) / n if n else 0.0
        # diversity: unique task_ids recommended / total interactions
        rec_ids = [x.get("chosen_task_id") for x in ul if x.get("chosen_task_id")]
        diversity = len(set(rec_ids)) / len(rec_ids) if rec_ids else 0.0
        per_user.append({
            "user_id": uid,
            "cum_reward": round(cr, 4),
            "ier": round(ier, 4),
            "acceptance_rate": round(acc, 4),
            "svr": round(svr, 4),
            "task_success_rate": round(ts, 4),
            "diversity": round(diversity, 4),
        })

    agg = {
        "cum_reward_mean": round(float(np.mean([r["cum_reward"] for r in per_user])), 4),
        "cum_reward_std":  round(float(np.std([r["cum_reward"] for r in per_user])), 4),
        "ier_mean":        round(float(np.mean([r["ier"] for r in per_user])), 4),
        "acceptance_rate": round(float(np.mean([r["acceptance_rate"] for r in per_user])), 4),
        "svr":             round(float(np.mean([r["svr"] for r in per_user])), 4),
        "diversity_mean":  round(float(np.mean([r["diversity"] for r in per_user])), 4),
    }

    result = {
        "method": algo,
        "n_days": N_DAYS,
        "n_users": len(per_user),
        "n_interactions": len(logs),
        "seed": 42,
        "wall_time_s": round(elapsed, 1),
        "aggregate_metrics": agg,
        "per_user_metrics": per_user,
    }

    # JSON
    jp = RESULTS_DIR / f"{filename_stem}.json"
    with open(jp, "w") as f:
        json.dump(result, f, indent=2,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_,))
                  else float(o) if isinstance(o, (np.floating, np.integer))
                  else str(o))

    # CSV
    cp = RESULTS_DIR / f"{filename_stem}.csv"
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_user[0].keys())
        w.writeheader()
        w.writerows(per_user)

    print(f"  ✓ {jp.name}  +  {cp.name}")
    print(f"  Reward={agg['cum_reward_mean']:.2f}  IER={agg['ier_mean']:.4f}"
          f"  Acc={agg['acceptance_rate']:.4f}  Div={agg['diversity_mean']:.4f}")
    return agg


# ═════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("[R2.4] Baseline re-run under Step-B EU simulator")
print(f"  Seed: 42 | Days: {N_DAYS} | Users: 20")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 64)

results = {}

# 1. Popularity
results["popularity"] = run_and_save(
    "1/4", "popularity",
    {"recommender": {"algorithm": "popularity"}},
    "popularity_60day_seed42",
)

# 2. RuleBased
results["rule_based"] = run_and_save(
    "2/4", "rule_based",
    {"recommender": {"algorithm": "rule_based"}},
    "rule_based_60day_seed42",
)

# 3. EpsilonGreedy (ε=0.10)
results["epsilon_greedy"] = run_and_save(
    "3/4", "epsilon_greedy",
    {"recommender": {"algorithm": "epsilon_greedy",
                      "epsilon_greedy": {"epsilon": 0.10}}},
    "epsilon_greedy_60day_seed42",
)

# 4. LinUCB NoSafety ablation
results["linucb_nosafety"] = run_and_save(
    "4/4", "linucb",
    {"recommender": {"algorithm": "linucb"},
     "safety": {"enable": False}},
    "linucb_nosafety_60day_seed42",
)

# ═════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  Summary Table")
print(f"  {'Method':<20} {'Reward':>8} {'IER':>8} {'Acc':>8} {'Div':>8}")
print("-" * 64)
for name, agg in results.items():
    print(f"  {name:<20} {agg['cum_reward_mean']:>8.2f} {agg['ier_mean']:>8.4f}"
          f" {agg['acceptance_rate']:>8.4f} {agg['diversity_mean']:>8.4f}")
print("=" * 64)
print(f"  Total wall time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
