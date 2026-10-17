"""Unified 60-day evaluation entry point.

Usage:
  python evaluation/run_60day.py --method saps
  python evaluation/run_60day.py --method linucb
  python evaluation/run_60day.py --method all
"""
import sys, json, time, argparse, logging
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline
from evaluation.metrics import safety_violation_rate

N_DAYS = 60
SEED = 42
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["saps", "linucb", "neural_ucb", "random", "popularity",
           "rule_based", "epsilon_greedy"]


def run_method(algo: str, extra_overrides: dict = None) -> dict:
    overrides = {"recommender": {"algorithm": algo}}
    if extra_overrides:
        overrides.update(extra_overrides)
    pipe = NarraCarePipeline("configs/default.yaml", config_overrides=overrides)
    logs = pipe.run_simulation(n_days=N_DAYS)

    user_logs = defaultdict(list)
    for ix in logs:
        user_logs[ix["user_id"]].append(ix)

    cum_rewards, ier_vals, acc_vals, svr_vals, div_vals = [], [], [], [], []
    for uid, ulogs in user_logs.items():
        n = len(ulogs)
        cr = sum(ix.get("reward", 0) for ix in ulogs)
        cum_rewards.append(cr)
        acc = sum(1 for ix in ulogs if ix.get("action") == "accept") / n if n else 0
        acc_vals.append(acc)
        ier_per = [ix.get("ier", 0) for ix in ulogs if ix.get("ier") is not None]
        ier_vals.append(np.mean(ier_per) if ier_per else 0)
        svr_vals.append(safety_violation_rate(ulogs))
        tasks = set(ix.get("chosen_task_id", "") for ix in ulogs)
        div_vals.append(len(tasks) / n if n else 0)

    return {
        "method": algo,
        "n_days": N_DAYS,
        "n_users": len(user_logs),
        "n_interactions": len(logs),
        "seed": SEED,
        "aggregate_metrics": {
            "cum_reward_mean": float(np.mean(cum_rewards)),
            "cum_reward_std": float(np.std(cum_rewards)),
            "ier_mean": float(np.mean(ier_vals)),
            "acceptance_rate": float(np.mean(acc_vals)),
            "svr": float(np.mean(svr_vals)),
            "diversity_mean": float(np.mean(div_vals)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="NarraCare 60-day evaluation")
    parser.add_argument("--method", type=str, default="all",
                        help="Algorithm to run (or 'all')")
    args = parser.parse_args()

    methods = METHODS if args.method == "all" else [args.method]

    for algo in methods:
        print(f"\n{'='*50}")
        print(f"  Running {algo}...")
        t0 = time.time()
        result = run_method(algo)
        elapsed = time.time() - t0
        m = result["aggregate_metrics"]
        print(f"  Done in {elapsed:.1f}s")
        print(f"  Reward={m['cum_reward_mean']:.2f} IER={m['ier_mean']:.4f} "
              f"Acc={m['acceptance_rate']:.4f}")

        out = RESULTS_DIR / f"{algo}_60day.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2,
                      default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        print(f"  Saved: {out.name}")


if __name__ == "__main__":
    main()
