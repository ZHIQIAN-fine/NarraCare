"""Significance testing for NarraCare experiments.

Runs paired t-tests (LinUCB vs each baseline) with Bonferroni correction.
Pairing unit: per-user aggregate metrics over the full simulation.

Usage::

    cd NarraCare
    python -m evaluation.significance_test          # full 60-day run
    python -m evaluation.significance_test --days 30  # quick 30-day run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import NarraCarePipeline
from evaluation.metrics import (
    infeasible_exploration_rate,
    safety_violation_rate,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Per-user metric computation ────────────────────────────────────────

def _per_user_metrics(logs: List[dict], n_days: int) -> Dict[str, Dict[str, float]]:
    """Compute per-user aggregate metrics from interaction logs.

    Returns {user_id: {metric_name: value}}.
    """
    user_logs: Dict[str, List[dict]] = defaultdict(list)
    for ix in logs:
        user_logs[ix.get("user_id", "")].append(ix)

    result: Dict[str, Dict[str, float]] = {}
    for uid, ulogs in user_logs.items():
        n_total = len(ulogs)
        accepted = [ix for ix in ulogs if ix.get("action") == "accept"]
        n_accepted = len(accepted)

        # Acceptance rate
        acc_rate = n_accepted / n_total if n_total > 0 else 0.0

        # Task success rate (accepted AND completed)
        n_success = sum(1 for ix in accepted
                        if ix.get("feedback", {}).get("completed", False))
        task_success = n_success / n_total if n_total > 0 else 0.0

        # Adherence rate (days with at least one interaction / total days)
        active_days = set()
        for ix in ulogs:
            if ix.get("action") == "accept":
                active_days.add(ix.get("day", -1))
        adherence = len(active_days) / n_days if n_days > 0 else 0.0

        # Cumulative reward
        cum_reward = sum(ix.get("reward", 0.0) for ix in ulogs)

        # IER
        ier_vals = [ix.get("ier", 0.0) for ix in ulogs if ix.get("ier") is not None]
        ier = np.mean(ier_vals) if ier_vals else 0.0

        result[uid] = {
            "acceptance_rate": acc_rate,
            "task_success": task_success,
            "adherence": adherence,
            "cumulative_reward": cum_reward,
            "ier": ier,
        }
    return result


# ─── Significance test ──────────────────────────────────────────────────

def _significance_star(p: float, n_comparisons: int) -> str:
    """Return significance marker with Bonferroni correction."""
    adjusted_p = p * n_comparisons  # Bonferroni
    if adjusted_p < 0.001:
        return "***"
    elif adjusted_p < 0.01:
        return "**"
    elif adjusted_p < 0.05:
        return "*"
    else:
        return ""


def run_significance_tests(
    method_user_metrics: Dict[str, Dict[str, Dict[str, float]]],
    primary: str = "LinUCB",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run paired t-tests: primary method vs every other method.

    Parameters
    ----------
    method_user_metrics : {method_name: {user_id: {metric: value}}}
    primary : str
        The method to compare against all baselines.
    metrics : list[str] | None
        Metrics to test. Defaults to the 5 core metrics.

    Returns
    -------
    pd.DataFrame
        Rows = metrics, columns = baselines, cells = "delta (stars)"
    """
    if metrics is None:
        metrics = ["acceptance_rate", "task_success", "adherence",
                    "cumulative_reward", "ier"]

    baselines = [m for m in method_user_metrics if m != primary]
    n_comparisons = len(baselines) * len(metrics)  # total tests for Bonferroni

    primary_data = method_user_metrics[primary]
    user_ids = sorted(primary_data.keys())

    rows = []
    for metric in metrics:
        row = {"Metric": metric}
        for bl in baselines:
            bl_data = method_user_metrics[bl]
            # Paired arrays (same user order)
            a_vals = np.array([primary_data[uid][metric] for uid in user_ids])
            b_vals = np.array([bl_data[uid][metric] for uid in user_ids])

            delta = np.mean(a_vals) - np.mean(b_vals)

            # Paired t-test
            if np.std(a_vals - b_vals) < 1e-10:
                # No variance → can't do t-test
                p_val = 1.0
            else:
                t_stat, p_val = stats.ttest_rel(a_vals, b_vals)

            star = _significance_star(p_val, n_comparisons)

            # Format: delta with sign and star
            if metric == "ier":
                # For IER, negative delta = LinUCB is better (lower)
                cell = f"{delta:+.4f}{star}"
            elif metric == "cumulative_reward":
                cell = f"{delta:+.2f}{star}"
            else:
                cell = f"{delta*100:+.1f}%{star}"

            row[bl] = cell
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Metric")
    return df


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NarraCare significance tests")
    parser.add_argument("--days", type=int, default=60, help="Simulation days")
    args = parser.parse_args()

    n_days = args.days
    print(f"Running significance tests with {n_days} days, 20 users")
    print("=" * 70)

    method_configs = {
        "Random": {"recommender": {"algorithm": "random"}},
        "Popularity": {"recommender": {"algorithm": "popularity"}},
        "RuleBased": {"recommender": {"algorithm": "rule_based"}},
        "EpsilonGreedy": {"recommender": {"algorithm": "epsilon_greedy"}},
        "LinUCB": {"recommender": {"algorithm": "linucb"}},
    }

    # NeuralUCB with reduced settings for speed
    method_configs["NeuralUCB"] = {
        "recommender": {
            "algorithm": "neural_ucb",
            "neural_ucb": {"training_epochs": 3, "hidden_dim": 32},
        }
    }

    method_user_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for name, overrides in method_configs.items():
        t0 = time.time()
        print(f"  Running {name}...", end=" ", flush=True)
        pipe = NarraCarePipeline("config/default.yaml", config_overrides=overrides)
        logs = pipe.run_simulation(n_days=n_days)
        per_user = _per_user_metrics(logs, n_days)
        method_user_metrics[name] = per_user
        elapsed = time.time() - t0
        n_users = len(per_user)
        print(f"{n_users} users, {len(logs)} interactions ({elapsed:.1f}s)")

    # Run significance tests
    print()
    print("=" * 70)
    print("LinUCB vs Baselines — Paired t-test with Bonferroni Correction")
    print(f"Pairing unit: per-user ({len(list(method_user_metrics.values())[0])} users)")
    print(f"Bonferroni correction: {5 * 5} = 25 comparisons")
    print("=" * 70)

    df = run_significance_tests(method_user_metrics, primary="LinUCB")
    print(df.to_string())

    # Save results
    results_dir = _PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(results_dir / "significance_test.csv")
    print(f"\nSaved: {results_dir / 'significance_test.csv'}")

    # Also save raw per-user metrics for reproducibility
    raw = {method: {uid: metrics for uid, metrics in um.items()}
           for method, um in method_user_metrics.items()}
    with open(results_dir / "per_user_metrics.json", "w") as f:
        json.dump(raw, f, indent=2, default=str)
    print(f"Saved: {results_dir / 'per_user_metrics.json'}")

    return df


if __name__ == "__main__":
    main()
