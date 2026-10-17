"""Local, versioned feedback/bonus review. Never overwrite frozen artifacts."""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from evaluation.metrics import safety_violation_rate
from src.pipeline import NarraCarePipeline

logging.basicConfig(level=logging.ERROR)
POLICIES = [
    "saps",
    "linucb",
    "neural_ucb",
    "random",
    "popularity",
    "rule_based",
    "epsilon_greedy",
]


def run(policy, seed, tau=None):
    override = {"system": {"seed": seed}, "recommender": {"algorithm": policy}}
    if tau is not None:
        override["recommender"]["saps"] = {"tau_safe": tau}
    if policy == "neural_ucb":
        override["recommender"]["neural_ucb"] = {
            "hidden_dim": 32,
            "training_epochs": 5,
            "alpha": 0.01,
        }
        override["reward"] = {"safety_lambda": 3.0}
    pipe = NarraCarePipeline(config_overrides=override)
    with tempfile.TemporaryDirectory(prefix="narracare-review-") as tmp:
        old = os.getcwd()
        try:
            os.chdir(tmp)
            logs = pipe.run_simulation(n_days=60)
        finally:
            os.chdir(old)
    by = defaultdict(list)
    for row in logs:
        by[row["user_id"]].append(row)
    metrics = {
        "cumulative_reward": float(
            np.mean([sum(r["reward"] for r in rows) for rows in by.values()])
        ),
        "IER": float(
            np.mean([np.mean([r["ier"] for r in rows]) for rows in by.values()])
        ),
        "SVR": float(np.mean([safety_violation_rate(rows) for rows in by.values()])),
        "acceptance": sum(r["action"] == "accept" for r in logs) / len(logs),
        "interactions": len(logs),
        "task_assigned": sum(bool(r["recommended"]) for r in logs),
        "executed": sum(r["action"] == "accept" for r in logs),
        "skipped": sum(r["action"] == "skip" for r in logs),
        "skips_with_pain_outcome": sum(
            r["action"] == "skip" and "pain_change" in r["feedback"] for r in logs
        ),
        "completed_counter": sum(pipe.simulator._user_tasks_completed.values()),
        "feedback_completed": sum(bool(r["feedback"].get("completed")) for r in logs),
    }
    model = pipe.recommender.model
    if hasattr(model, "safe_alpha"):
        metrics["beta_positive"] = sum(v - model._a0 for v in model.safe_alpha.values())
        metrics["beta_negative"] = sum(v - model._b0 for v in model.safe_beta.values())
    return {
        "policy": policy,
        "seed": seed,
        "tau_safe": tau,
        "metrics": metrics,
        "per_user": {
            u: {
                "reward": sum(r["reward"] for r in rows),
                "IER": float(np.mean([r["ier"] for r in rows])),
            }
            for u, rows in by.items()
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--policies", default=",".join(POLICIES))
    p.add_argument("--seeds", default="42")
    p.add_argument("--bonus", action="store_true")
    a = p.parse_args()
    if a.bonus and a.policies != "saps":
        p.error("--bonus requires --policies saps")
    if not set(a.policies.split(",")) <= set(POLICIES):
        p.error("Unknown policy; legal values: " + ",".join(POLICIES))
    dest = Path(a.output).resolve()
    if (ROOT / "evaluation/results") in dest.parents:
        raise ValueError("Frozen output directory is forbidden")
    rows = []
    for seed in map(int, a.seeds.split(",")):
        for policy in a.policies.split(","):
            for tau in [0.0, 2.0] if a.bonus else [None]:
                row = run(policy, seed, tau)
                rows.append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "per_user"}),
                    flush=True,
                )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "base_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "note": "Exploratory online simulation; no clinical inference. Independent seed runs, not validation/tuning; same seed does not align realized noise across policies. Current working-tree source may differ from base_commit.",
                "runs": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
