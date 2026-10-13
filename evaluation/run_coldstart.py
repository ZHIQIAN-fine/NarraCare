"""R2.7 §T1 — Cold-Start Transfer Experiment (§7.4).

Train SAPS and LinUCB on known-condition cohort, freeze posteriors, and
evaluate transfer to previously-unseen Back-Pain users.

Cohorts (adjusted for actual user_profiles.json):
  Train:   U001–U010 + U016–U020  (10 ACL + 5 Rotator-Cuff)
  Rollout: U011–U015               (5 Chronic Lower Back Pain)

Protocol:
  Phase 1 — Train: 60 days on train cohort, extract posteriors
  Phase 2a — Frozen rollout: load posteriors, disable learning updates, 60 days
  Phase 2b — Retrained rollout: fresh pipeline, normal learning, 60 days
  Phase 3 — Compute metrics, paired t-tests, produce output JSON

Outputs:
  evaluation/results/coldstart_60day_seed42.json   (main result artifact)
  evaluation/results/coldstart_posteriors_saps.json
  evaluation/results/coldstart_posteriors_linucb.json
"""
import sys, json, time, logging, copy
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline

# ── Constants ─────────────────────────────────────────────────────────────
N_DAYS = 60
SEED = 42
TRAIN_IDS = sorted([f"U{i:03d}" for i in list(range(1, 11)) + list(range(16, 21))])
ROLLOUT_IDS = sorted([f"U{i:03d}" for i in range(11, 16)])
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _json_default(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ── Metrics extraction ────────────────────────────────────────────────────
def extract_metrics(logs: list) -> dict:
    """Per-user + aggregate metrics from interaction logs."""
    user_logs: Dict[str, list] = defaultdict(list)
    for ix in logs:
        user_logs[ix["user_id"]].append(ix)

    per_user = []
    for uid in sorted(user_logs):
        ulogs = user_logs[uid]
        n = len(ulogs)
        cr = sum(ix.get("reward", 0) for ix in ulogs)
        acc = sum(1 for ix in ulogs if ix.get("action") == "accept") / n if n else 0
        ier_vals = [ix.get("ier", 0) for ix in ulogs if ix.get("ier") is not None]
        ier = float(np.mean(ier_vals)) if ier_vals else 0
        per_user.append({
            "user_id": uid,
            "cum_reward": float(cr),
            "ier": ier,
            "acceptance_rate": acc,
            "n_interactions": n,
        })

    rewards = [u["cum_reward"] for u in per_user]
    iers = [u["ier"] for u in per_user]
    return {
        "n_users": len(per_user),
        "n_interactions": len(logs),
        "cum_reward_mean": float(np.mean(rewards)),
        "cum_reward_std": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0,
        "ier_mean": float(np.mean(iers)),
        "acceptance_rate_mean": float(np.mean([u["acceptance_rate"] for u in per_user])),
        "per_user": per_user,
    }


def paired_ttest(a: list, b: list, label: str) -> dict:
    """Paired t-test: a - b."""
    a_arr, b_arr = np.array(a), np.array(b)
    diff = a_arr - b_arr
    t_stat, p_val = sp_stats.ttest_rel(a_arr, b_arr)
    return {
        "label": label,
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "n": len(a),
    }


# ── Posterior save / load ─────────────────────────────────────────────────
def save_posteriors_linucb(pipe: NarraCarePipeline) -> dict:
    model = pipe.recommender.model
    return {
        "algorithm": "linucb",
        "A": {k: v.tolist() for k, v in model.A.items()},
        "b": {k: v.tolist() for k, v in model.b.items()},
    }


def save_posteriors_saps(pipe: NarraCarePipeline) -> dict:
    model = pipe.recommender.model
    return {
        "algorithm": "saps",
        "A": {k: v.tolist() for k, v in model.A.items()},
        "b": {k: v.tolist() for k, v in model.b.items()},
        "safe_alpha": dict(model.safe_alpha),
        "safe_beta": dict(model.safe_beta),
    }


def load_posteriors_linucb(pipe: NarraCarePipeline, posteriors: dict):
    model = pipe.recommender.model
    model.A = {k: np.array(v) for k, v in posteriors["A"].items()}
    model.b = {k: np.array(v) for k, v in posteriors["b"].items()}


def load_posteriors_saps(pipe: NarraCarePipeline, posteriors: dict):
    model = pipe.recommender.model
    model.A = {k: np.array(v) for k, v in posteriors["A"].items()}
    model.b = {k: np.array(v) for k, v in posteriors["b"].items()}
    model.safe_alpha = dict(posteriors["safe_alpha"])
    model.safe_beta = dict(posteriors["safe_beta"])


def freeze_updates(pipe: NarraCarePipeline):
    """Monkey-patch recommender.update() to no-op (freeze posteriors)."""
    pipe.recommender.update = lambda ctx, tid, r: None


def filter_users(pipe: NarraCarePipeline, user_ids: list):
    """Restrict simulator to only the specified user IDs."""
    pipe.simulator.profiles = {
        uid: p for uid, p in pipe.simulator.profiles.items()
        if uid in set(user_ids)
    }


# ── Phase runners ─────────────────────────────────────────────────────────
def run_phase(algo: str, user_ids: list, n_days: int = N_DAYS,
              posteriors: dict = None, freeze: bool = False,
              extra_overrides: dict = None) -> tuple:
    """Run one pipeline phase, return (pipe, logs)."""
    overrides = {"recommender": {"algorithm": algo}}
    if extra_overrides:
        for k, v in extra_overrides.items():
            if k in overrides:
                overrides[k].update(v)
            else:
                overrides[k] = v

    pipe = NarraCarePipeline("configs/default.yaml", config_overrides=overrides)
    filter_users(pipe, user_ids)

    # Inject posteriors if provided
    if posteriors:
        if algo == "saps":
            load_posteriors_saps(pipe, posteriors)
        else:
            load_posteriors_linucb(pipe, posteriors)

    # Freeze learning if requested
    if freeze:
        freeze_updates(pipe)

    logs = pipe.run_simulation(n_days=n_days)
    return pipe, logs


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 64)
    print("R2.7 §T1 — Cold-Start Transfer Experiment")
    print(f"  Train cohort:   {TRAIN_IDS}")
    print(f"  Rollout cohort: {ROLLOUT_IDS}")
    print(f"  Days: {N_DAYS}  Seed: {SEED}")
    print("=" * 64)

    results: Dict[str, Any] = {
        "experiment": "coldstart_transfer",
        "train_ids": TRAIN_IDS,
        "rollout_ids": ROLLOUT_IDS,
        "n_days": N_DAYS,
        "seed": SEED,
        "note": "Train={U001-U010,U016-U020} 10ACL+5RC; Rollout={U011-U015} 5 Back-Pain",
        "arms": {},
    }

    for algo in ["saps", "linucb"]:
        print(f"\n{'─'*64}")
        print(f"  Algorithm: {algo.upper()}")
        print(f"{'─'*64}")

        # ── Phase 1: Train ────────────────────────────────────────────────
        print(f"\n  Phase 1: Training on {len(TRAIN_IDS)} users...")
        t0 = time.time()
        train_pipe, train_logs = run_phase(algo, TRAIN_IDS)
        train_time = time.time() - t0
        train_metrics = extract_metrics(train_logs)
        print(f"    Train done: {train_time:.1f}s, {train_metrics['n_interactions']} ix")
        print(f"    Train reward: {train_metrics['cum_reward_mean']:.2f}, "
              f"IER: {train_metrics['ier_mean']:.4f}")

        # Save posteriors
        if algo == "saps":
            posteriors = save_posteriors_saps(train_pipe)
        else:
            posteriors = save_posteriors_linucb(train_pipe)

        post_path = RESULTS_DIR / f"coldstart_posteriors_{algo}.json"
        with open(post_path, "w") as f:
            json.dump(posteriors, f, default=_json_default)
        print(f"    Posteriors saved: {post_path.name} "
              f"({len(posteriors.get('A', {}))} arms)")

        # ── Phase 2a: Frozen rollout ──────────────────────────────────────
        print(f"\n  Phase 2a: Frozen rollout on {len(ROLLOUT_IDS)} Back-Pain users...")
        t0 = time.time()
        _, frozen_logs = run_phase(algo, ROLLOUT_IDS, posteriors=posteriors,
                                   freeze=True)
        frozen_time = time.time() - t0
        frozen_metrics = extract_metrics(frozen_logs)
        print(f"    Frozen done: {frozen_time:.1f}s")
        print(f"    Frozen reward: {frozen_metrics['cum_reward_mean']:.2f}, "
              f"IER: {frozen_metrics['ier_mean']:.4f}, "
              f"Acc: {frozen_metrics['acceptance_rate_mean']:.4f}")

        # ── Phase 2b: Retrained rollout ───────────────────────────────────
        print(f"\n  Phase 2b: Retrained (fresh) on {len(ROLLOUT_IDS)} Back-Pain users...")
        t0 = time.time()
        _, retrained_logs = run_phase(algo, ROLLOUT_IDS)
        retrained_time = time.time() - t0
        retrained_metrics = extract_metrics(retrained_logs)
        print(f"    Retrained done: {retrained_time:.1f}s")
        print(f"    Retrained reward: {retrained_metrics['cum_reward_mean']:.2f}, "
              f"IER: {retrained_metrics['ier_mean']:.4f}, "
              f"Acc: {retrained_metrics['acceptance_rate_mean']:.4f}")

        # ── Phase 3: Compare ──────────────────────────────────────────────
        print(f"\n  Phase 3: Paired t-tests (frozen vs retrained)...")
        frozen_rewards = [u["cum_reward"] for u in frozen_metrics["per_user"]]
        retrained_rewards = [u["cum_reward"] for u in retrained_metrics["per_user"]]
        frozen_iers = [u["ier"] for u in frozen_metrics["per_user"]]
        retrained_iers = [u["ier"] for u in retrained_metrics["per_user"]]

        tt_reward = paired_ttest(frozen_rewards, retrained_rewards,
                                 "frozen_vs_retrained_reward")
        tt_ier = paired_ttest(frozen_iers, retrained_iers,
                              "frozen_vs_retrained_ier")

        print(f"    Reward Δ = {tt_reward['mean_diff']:+.2f}, p = {tt_reward['p_value']:.4f}")
        print(f"    IER Δ    = {tt_ier['mean_diff']:+.4f}, p = {tt_ier['p_value']:.4f}")

        results["arms"][algo] = {
            "train": {
                "metrics": {k: v for k, v in train_metrics.items() if k != "per_user"},
                "wall_time_s": round(train_time, 1),
            },
            "frozen_rollout": {
                "metrics": frozen_metrics,
                "wall_time_s": round(frozen_time, 1),
            },
            "retrained_rollout": {
                "metrics": retrained_metrics,
                "wall_time_s": round(retrained_time, 1),
            },
            "ttest_frozen_vs_retrained": {
                "reward": tt_reward,
                "ier": tt_ier,
            },
            "posteriors_file": f"coldstart_posteriors_{algo}.json",
        }

    # ── Save main result ──────────────────────────────────────────────────
    out_path = RESULTS_DIR / "coldstart_60day_seed42.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)

    print(f"\n{'='*64}")
    print(f"  ✓ Results saved: {out_path.name}")
    print(f"{'='*64}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n  {'Method':<10} {'Condition':<12} {'Reward':>8} {'IER':>8} {'Acc':>8}")
    print(f"  {'─'*50}")
    for algo in ["saps", "linucb"]:
        arm = results["arms"][algo]
        tr = arm["train"]["metrics"]
        fr = arm["frozen_rollout"]["metrics"]
        rt = arm["retrained_rollout"]["metrics"]
        print(f"  {algo.upper():<10} {'Train(ACL+RC)':<12} "
              f"{tr['cum_reward_mean']:>8.2f} {tr['ier_mean']:>8.4f} "
              f"{tr['acceptance_rate_mean']:>8.4f}")
        print(f"  {'':<10} {'Frozen(BP)':<12} "
              f"{fr['cum_reward_mean']:>8.2f} {fr['ier_mean']:>8.4f} "
              f"{fr['acceptance_rate_mean']:>8.4f}")
        print(f"  {'':<10} {'Retrained(BP)':<12} "
              f"{rt['cum_reward_mean']:>8.2f} {rt['ier_mean']:>8.4f} "
              f"{rt['acceptance_rate_mean']:>8.4f}")
