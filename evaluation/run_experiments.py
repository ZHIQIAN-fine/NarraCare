"""Experiment runner for NarraCare evaluation.

Runs all six recommendation methods, performs ablation experiments,
generates comparison tables (CSV) and plots (PNG), and saves everything
under ``evaluation/results/``.

Usage::

    cd NarraCare
    python -m evaluation.run_experiments          # full run
    python -m evaluation.run_experiments --quick   # reduced days/users
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Headless backend must be set before matplotlib import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so that imports resolve correctly
# when invoked as `python -m evaluation.run_experiments`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import NarraCarePipeline
from evaluation.metrics import compute_all_metrics
from evaluation.llm_judge import LLMJudge


# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════

METHODS: List[str] = [
    "linucb",
    "neural_ucb",
    "random",
    "popularity",
    "rule_based",
    "epsilon_greedy",
]

METHOD_DISPLAY: Dict[str, str] = {
    "linucb": "LinUCB",
    "neural_ucb": "NeuralUCB",
    "random": "Random",
    "popularity": "Popularity",
    "rule_based": "RuleBased",
    "epsilon_greedy": "EpsilonGreedy",
}

ABLATIONS: Dict[str, dict] = {
    "Full": {},
    "NoSafety": {"safety": {"enable": False}},
    "NoNarrative": {"narrator": {"enable": False}},
    # NoMemory is handled via the `use_memory` flag, not config override
}

RESULTS_DIR = _PROJECT_ROOT / "evaluation" / "results"


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_single_experiment(
    algorithm: str,
    n_days: int,
    config_overrides: Optional[dict] = None,
    use_memory: bool = True,
) -> Dict[str, Any]:
    """Build a pipeline for *algorithm*, run simulation, compute metrics."""
    overrides = config_overrides or {}
    overrides.setdefault("recommender", {})["algorithm"] = algorithm
    overrides.setdefault("simulation", {})["n_days"] = n_days

    t0 = time.time()
    pipeline = NarraCarePipeline(
        config_overrides=overrides,
        use_memory=use_memory,
    )
    logs = pipeline.run_simulation(n_days)
    elapsed = time.time() - t0

    metrics = compute_all_metrics(logs, pipeline.config)
    metrics["wall_time_s"] = round(elapsed, 2)
    metrics["n_interactions"] = len(logs)

    return {
        "metrics": metrics,
        "logs": logs,
        "config": pipeline.config,
    }


def _cumulative_reward_curve(logs: List[dict]) -> List[float]:
    """Return a list of cumulative reward values, one per interaction."""
    cum = 0.0
    curve: List[float] = []
    for ix in logs:
        cum += ix.get("reward", 0.0)
        curve.append(cum)
    return curve


# ══════════════════════════════════════════════════════════════════════════
# Main experiment functions
# ══════════════════════════════════════════════════════════════════════════


def run_method_comparison(n_days: int = 30) -> pd.DataFrame:
    """Run all 6 methods and return a comparison DataFrame."""
    rows: List[dict] = []
    all_logs: Dict[str, List[dict]] = {}
    all_curves: Dict[str, List[float]] = {}

    for algo in METHODS:
        label = METHOD_DISPLAY[algo]
        logger.info("▶ Running method: %s", label)
        result = _run_single_experiment(algo, n_days)
        row = {"method": label, **result["metrics"]}
        rows.append(row)
        all_logs[label] = result["logs"]
        all_curves[label] = _cumulative_reward_curve(result["logs"])
        logger.info(
            "  ✓ %s done — reward=%.2f  hit_rate=%.3f  time=%.1fs",
            label,
            result["metrics"]["cumulative_reward"],
            result["metrics"]["hit_rate"],
            result["metrics"]["wall_time_s"],
        )

    df = pd.DataFrame(rows).set_index("method")

    # Save
    outdir = _ensure_dir(RESULTS_DIR)
    df.to_csv(outdir / "method_comparison.csv")
    logger.info("Saved method_comparison.csv")

    # Save raw logs (JSON-lines, one file per method)
    logs_dir = _ensure_dir(outdir / "logs")
    for label, logs in all_logs.items():
        with open(logs_dir / f"{label}_logs.json", "w", encoding="utf-8") as fh:
            json.dump(logs, fh, indent=None, default=str)

    # ── Plots ────────────────────────────────────────────────────────
    _plot_bar_chart(df, outdir)
    _plot_cumulative_reward(all_curves, outdir)

    return df


def run_ablation_experiments(
    base_algorithm: str = "linucb",
    n_days: int = 30,
) -> pd.DataFrame:
    """Run ablation experiments: Full vs NoSafety vs NoNarrative vs NoMemory."""
    rows: List[dict] = []
    curves: Dict[str, List[float]] = {}

    for name, overrides in ABLATIONS.items():
        logger.info("▶ Ablation: %s", name)
        result = _run_single_experiment(
            base_algorithm, n_days, config_overrides=copy.deepcopy(overrides),
        )
        row = {"ablation": name, **result["metrics"]}
        rows.append(row)
        curves[name] = _cumulative_reward_curve(result["logs"])

    # NoMemory ablation (special flag)
    logger.info("▶ Ablation: NoMemory")
    result = _run_single_experiment(base_algorithm, n_days, use_memory=False)
    row = {"ablation": "NoMemory", **result["metrics"]}
    rows.append(row)
    curves["NoMemory"] = _cumulative_reward_curve(result["logs"])

    df = pd.DataFrame(rows).set_index("ablation")

    outdir = _ensure_dir(RESULTS_DIR)
    df.to_csv(outdir / "ablation_comparison.csv")
    logger.info("Saved ablation_comparison.csv")

    _plot_ablation_bars(df, outdir)
    _plot_cumulative_reward(curves, outdir, filename="ablation_cumulative_reward.png")

    return df


def run_llm_judge_evaluation(
    interaction_logs: Optional[List[dict]] = None,
    config: Optional[dict] = None,
    n_samples: int = 50,
) -> Dict[str, float]:
    """Run LLM-as-judge evaluation on logs (mock mode if no API key)."""
    judge = LLMJudge(config=config)

    if interaction_logs is None:
        # Run a quick simulation to get logs
        pipeline = NarraCarePipeline()
        interaction_logs = pipeline.run_simulation()
        config = pipeline.config

    results = judge.run_evaluation(interaction_logs, n_samples=n_samples)

    outdir = _ensure_dir(RESULTS_DIR)
    with open(outdir / "llm_judge_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Saved llm_judge_results.json")

    return results


# ══════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════


def _plot_bar_chart(df: pd.DataFrame, outdir: Path) -> None:
    """Bar charts for key metrics across methods."""
    key_metrics = [
        "hit_rate", "cumulative_reward", "safety_violation_rate",
        "diversity", "novelty", "task_success_rate",
    ]
    available = [m for m in key_metrics if m in df.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 8))
    axes = axes.flatten() if n > 1 else [axes]

    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))

    for idx, metric in enumerate(available):
        ax = axes[idx]
        bars = ax.bar(df.index, df[metric], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Method Comparison — Key Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "method_comparison_bars.png", dpi=150)
    plt.close(fig)
    logger.info("Saved method_comparison_bars.png")


def _plot_cumulative_reward(
    curves: Dict[str, List[float]],
    outdir: Path,
    filename: str = "cumulative_reward_over_time.png",
) -> None:
    """Line chart of cumulative reward over interactions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, curve in curves.items():
        ax.plot(curve, label=label, linewidth=1.5)
    ax.set_xlabel("Interaction #")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward Over Time")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", filename)


def _plot_ablation_bars(df: pd.DataFrame, outdir: Path) -> None:
    """Bar chart comparing ablation variants on key metrics."""
    key_metrics = [
        "hit_rate", "cumulative_reward", "safety_violation_rate",
        "diversity", "task_success_rate", "adherence_rate",
    ]
    available = [m for m in key_metrics if m in df.columns]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 8))
    axes = axes.flatten() if n > 1 else [axes]

    colors = plt.cm.Paired(np.linspace(0, 1, len(df)))

    for idx, metric in enumerate(available):
        ax = axes[idx]
        ax.bar(df.index, df[metric], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=25, labelsize=9)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(available), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Ablation Study — LinUCB", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "ablation_comparison_bars.png", dpi=150)
    plt.close(fig)
    logger.info("Saved ablation_comparison_bars.png")


# ══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NarraCare Experiment Runner",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick run with fewer days (5 instead of 30)",
    )
    parser.add_argument(
        "--n-days", type=int, default=None,
        help="Override number of simulation days",
    )
    parser.add_argument(
        "--skip-ablation", action="store_true",
        help="Skip ablation experiments",
    )
    parser.add_argument(
        "--skip-llm-judge", action="store_true",
        help="Skip LLM judge evaluation",
    )
    args = parser.parse_args()

    n_days = args.n_days
    if n_days is None:
        n_days = 5 if args.quick else 30

    _ensure_dir(RESULTS_DIR)

    print(f"\n{'═' * 60}")
    print("  NarraCare — Experiment Runner")
    print(f"  Days: {n_days}  |  Methods: {len(METHODS)}")
    print(f"  Results → {RESULTS_DIR}")
    print(f"{'═' * 60}\n")

    # ── 1. Method comparison ─────────────────────────────────────────
    print("━━ Phase 1: Method Comparison ━━")
    df_methods = run_method_comparison(n_days)
    print("\n📊 Method Comparison Results:\n")
    print(df_methods.to_string())
    print()

    # ── 2. Ablation experiments ──────────────────────────────────────
    if not args.skip_ablation:
        print("━━ Phase 2: Ablation Experiments ━━")
        df_ablation = run_ablation_experiments("linucb", n_days)
        print("\n📊 Ablation Results:\n")
        print(df_ablation.to_string())
        print()

    # ── 3. LLM Judge ─────────────────────────────────────────────────
    if not args.skip_llm_judge:
        print("━━ Phase 3: LLM Judge Evaluation ━━")
        judge_results = run_llm_judge_evaluation(n_samples=min(50, n_days * 5))
        print("\n📊 LLM Judge Results:")
        for k, v in judge_results.items():
            print(f"  {k}: {v}")
        print()

    print(f"{'═' * 60}")
    print(f"  ✅ All experiments complete.  Results in: {RESULTS_DIR}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-32s  %(levelname)-7s  %(message)s",
    )
    main()
