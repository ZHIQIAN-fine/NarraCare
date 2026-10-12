"""Pre-fix NeuralUCB 60-day baseline for §5.4 ablation reproducibility.

Reproduces the *broken* NeuralUCB architecture (commit 957e70f) where
select() produces a single context prediction for ALL arms and
differentiates only by random noise — effectively a random policy.

All hyperparameters match post-fix exactly (hidden_dim=32, epochs=5,
alpha=0.01, safety_lambda=3.0) so the ONLY difference is the per-arm
learning fix + safety warm-start.

Paired t-test (n=20 users, seed=42) compares with post-fix results.
"""
import sys, json, time, csv, logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

# ═════════════════════════════════════════════════════════════════════════
# Monkey-patch NeuralUCBModel BEFORE importing the pipeline so that the
# model constructed by _build_model() uses the old broken architecture.
# ═════════════════════════════════════════════════════════════════════════
import src.agents.recommender as _rec_mod

_OriginalNeuralUCBModel = _rec_mod.NeuralUCBModel


class _NeuralUCBModelPreFix(_OriginalNeuralUCBModel):
    """Exact reproduction of the pre-fix (commit 957e70f) NeuralUCBModel.

    Bug: select() computes ONE scalar prediction from the context network,
    then adds i.i.d. random noise per arm.  All arms see the same base
    score → ranking is purely noise-driven → the model is a random policy.

    No _arm_stats, no warm_start_safety.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Remove per-arm state that post-fix added
        if hasattr(self, "_arm_stats"):
            del self._arm_stats

    # Explicitly shadow post-fix method so pipeline warm-start won't fire
    warm_start_safety = None

    def select(
        self,
        context: np.ndarray,
        candidate_ids: List[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        if not self._has_torch:
            return self._fallback.select(context, candidate_ids, top_k)

        torch = self.torch
        self.net.eval()
        x_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        scores: List[Tuple[str, float]] = []
        with torch.no_grad():
            pred = self.net(x_tensor).item()

        # BUG: same pred for every arm; ranking = pure noise
        rng = np.random.default_rng()
        for arm_id in candidate_ids:
            exploration_bonus = self.alpha * rng.standard_normal()
            scores.append((arm_id, pred + exploration_bonus))

        scores.sort(key=lambda t: t[1], reverse=True)
        return scores[:top_k]

    def update(self, context: np.ndarray, task_id: str, reward: float) -> None:
        if not self._has_torch:
            self._fallback.update(context, task_id, reward)
            return

        self._buffer.append((context, task_id, reward))

        # Retrain periodically (no per-arm stats update)
        if len(self._buffer) % self.training_epochs == 0:
            self._retrain()


# Patch the module-level class AND the model registry
_rec_mod.NeuralUCBModel = _NeuralUCBModelPreFix
_rec_mod._MODEL_REGISTRY["neural_ucb"] = _NeuralUCBModelPreFix

# NOW import the pipeline (it will use the patched class)
from src.pipeline import NarraCarePipeline

# ═════════════════════════════════════════════════════════════════════════
# Configuration — identical to post-fix run
# ═════════════════════════════════════════════════════════════════════════
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
        "safety_lambda": 3.0,
    },
}

# ═════════════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("[§5.4 ablation] NeuralUCB PRE-FIX 60-day run")
print(f"  Config: hidden_dim=32, epochs=5, alpha=0.01, safety_lambda=3.0")
print(f"  Seed:   42 (from config/default.yaml)")
print(f"  Start:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 64)

t0 = time.time()
pipe = NarraCarePipeline("config/default.yaml", config_overrides=overrides)

# Confirm the model is actually the pre-fix variant
model = pipe.recommender.model
assert isinstance(model, _NeuralUCBModelPreFix), (
    f"Expected _NeuralUCBModelPreFix, got {type(model).__name__}"
)
assert not hasattr(model, "_arm_stats"), "Pre-fix model must NOT have _arm_stats"
assert not callable(getattr(model, "warm_start_safety", None)), "Pre-fix model must NOT have callable warm_start_safety"
print("  ✓ Model confirmed as pre-fix variant (no _arm_stats, no warm_start_safety)")

logs = pipe.run_simulation(n_days=N_DAYS)
elapsed = time.time() - t0
print(f"\n  Completed: {time.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed:.1f}s)")
print(f"  Interactions: {len(logs)}")

# ═════════════════════════════════════════════════════════════════════════
# Per-user metrics
# ═════════════════════════════════════════════════════════════════════════
user_logs: Dict[str, list] = defaultdict(list)
for ix in logs:
    user_logs[ix["user_id"]].append(ix)

per_user_rows = []
for uid in sorted(user_logs.keys()):
    ulogs = user_logs[uid]
    n = len(ulogs)
    cr = sum(ix.get("reward", 0) for ix in ulogs)
    acc = sum(1 for ix in ulogs if ix.get("action") == "accept") / n if n else 0
    ier_per = [ix.get("ier", 0) for ix in ulogs if ix.get("ier") is not None]
    ier_mean = float(np.mean(ier_per)) if ier_per else 0.0
    svr_count = sum(1 for ix in ulogs if ix.get("safety_violation", False))
    svr = svr_count / n if n else 0.0
    task_success = sum(
        1 for ix in ulogs
        if ix.get("action") == "accept" and ix.get("feedback", {}).get("completed", False)
    ) / n if n else 0.0
    per_user_rows.append({
        "user_id": uid,
        "cum_reward": round(cr, 4),
        "ier": round(ier_mean, 4),
        "acceptance_rate": round(acc, 4),
        "svr": round(svr, 4),
        "task_success_rate": round(task_success, 4),
    })

prefix_rewards = [r["cum_reward"] for r in per_user_rows]
prefix_ier = [r["ier"] for r in per_user_rows]

mean_cr = np.mean(prefix_rewards)
mean_ier = np.mean(prefix_ier)
mean_acc = np.mean([r["acceptance_rate"] for r in per_user_rows])
mean_svr = np.mean([r["svr"] for r in per_user_rows])

print(f"\n  Cum. Reward (mean): {mean_cr:.2f}")
print(f"  IER (mean):         {mean_ier:.4f}")
print(f"  Acceptance Rate:    {mean_acc:.4f}")
print(f"  SVR:                {mean_svr:.4f}")

# ═════════════════════════════════════════════════════════════════════════
# Load post-fix per-user metrics (same seed=42)
# ═════════════════════════════════════════════════════════════════════════
print("\n▶ Loading post-fix logs for paired t-test...")
postfix_logs_path = PROJECT_ROOT / "evaluation" / "results" / "logs" / "NeuralUCB_60day_logs.json"
with open(postfix_logs_path) as f:
    postfix_logs = json.load(f)

postfix_user_logs: Dict[str, list] = defaultdict(list)
for ix in postfix_logs:
    postfix_user_logs[ix["user_id"]].append(ix)

postfix_rewards = []
postfix_ier = []
for uid in sorted(postfix_user_logs.keys()):
    ulogs = postfix_user_logs[uid]
    n = len(ulogs)
    cr = sum(ix.get("reward", 0) for ix in ulogs)
    postfix_rewards.append(cr)
    ier_per = [ix.get("ier", 0) for ix in ulogs if ix.get("ier") is not None]
    postfix_ier.append(float(np.mean(ier_per)) if ier_per else 0.0)

assert len(prefix_rewards) == len(postfix_rewards) == 20, (
    f"User count mismatch: prefix={len(prefix_rewards)}, postfix={len(postfix_rewards)}"
)
print(f"  Post-fix Cum. Reward (mean): {np.mean(postfix_rewards):.2f}")
print(f"  Post-fix IER (mean):         {np.mean(postfix_ier):.4f}")

# ═════════════════════════════════════════════════════════════════════════
# Paired t-test (post-fix vs pre-fix)
# ═════════════════════════════════════════════════════════════════════════
from scipy import stats as sp_stats

reward_delta = np.mean(postfix_rewards) - np.mean(prefix_rewards)
t_rew, p_rew = sp_stats.ttest_rel(postfix_rewards, prefix_rewards)

ier_delta = np.mean(postfix_ier) - np.mean(prefix_ier)
t_ier, p_ier = sp_stats.ttest_rel(postfix_ier, prefix_ier)

print(f"\n{'='*64}")
print("  Paired t-test (post-fix vs pre-fix, n=20, seed=42):")
print(f"    Reward: Δ={reward_delta:+.2f}, t={t_rew:.3f}, p={p_rew:.4f}"
      f" {'***' if p_rew<0.001 else '**' if p_rew<0.01 else '*' if p_rew<0.05 else 'ns'}")
print(f"    IER:    Δ={ier_delta:+.4f}, t={t_ier:.3f}, p={p_ier:.4f}"
      f" {'***' if p_ier<0.001 else '**' if p_ier<0.01 else '*' if p_ier<0.05 else 'ns'}")
print(f"{'='*64}")

# ═════════════════════════════════════════════════════════════════════════
# Save artifacts
# ═════════════════════════════════════════════════════════════════════════
results_dir = PROJECT_ROOT / "evaluation" / "results"

# 1) Full JSON
result_json = {
    "experiment": "neuralucb_prefix_ablation",
    "description": (
        "Pre-fix NeuralUCB (commit 957e70f architecture): select() produces "
        "ONE scalar prediction for all arms, ranks by noise only. "
        "Effectively a random policy."
    ),
    "config": {
        "n_days": N_DAYS,
        "n_users": len(per_user_rows),
        "seed": 42,
        "hidden_dim": 32,
        "training_epochs": 5,
        "alpha": 0.01,
        "safety_lambda": 3.0,
    },
    "aggregate_metrics": {
        "cum_reward_mean": round(float(mean_cr), 4),
        "cum_reward_std": round(float(np.std(prefix_rewards)), 4),
        "ier_mean": round(float(mean_ier), 4),
        "acceptance_rate_mean": round(float(mean_acc), 4),
        "svr": round(float(mean_svr), 4),
    },
    "per_user_metrics": per_user_rows,
    "ablation_stats": {
        "comparison": "post-fix (260508a) vs pre-fix (957e70f architecture)",
        "n_pairs": 20,
        "seed": 42,
        "reward": {
            "postfix_mean": round(float(np.mean(postfix_rewards)), 4),
            "prefix_mean": round(float(mean_cr), 4),
            "delta": round(float(reward_delta), 4),
            "t_statistic": round(float(t_rew), 4),
            "p_value": round(float(p_rew), 6),
        },
        "ier": {
            "postfix_mean": round(float(np.mean(postfix_ier)), 4),
            "prefix_mean": round(float(mean_ier), 4),
            "delta": round(float(ier_delta), 4),
            "t_statistic": round(float(t_ier), 4),
            "p_value": round(float(p_ier), 6),
        },
    },
    "wall_time_s": round(elapsed, 1),
    "n_interactions": len(logs),
}

json_path = results_dir / "neural_ucb_prefix_60day_seed42.json"
with open(json_path, "w") as f:
    json.dump(
        result_json, f, indent=2,
        default=lambda o: bool(o) if isinstance(o, (np.bool_,))
        else float(o) if isinstance(o, (np.floating, np.integer))
        else str(o),
    )
print(f"\n✓ Saved: {json_path.relative_to(PROJECT_ROOT)}")

# 2) Per-user CSV
csv_path = results_dir / "neural_ucb_prefix_60day_seed42_metrics.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=per_user_rows[0].keys())
    writer.writeheader()
    writer.writerows(per_user_rows)
print(f"✓ Saved: {csv_path.relative_to(PROJECT_ROOT)}")

# Print commit-message ready lines
print(f"""
─── commit message snippet ───
Paired t-test (post-fix vs pre-fix, n=20, seed=42):
  Reward: Δ={reward_delta:+.2f}, p={p_rew:.4f}
  IER:    Δ={ier_delta:+.4f}, p={p_ier:.4f}
""")
