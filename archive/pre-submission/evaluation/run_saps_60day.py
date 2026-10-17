"""T4: SAPS 60-day full run — formal verification.

Verification criteria (R2):
  - IER ≤ LinUCB (LinUCB 60-day IER baseline required)
  - Cum. Reward gap vs LinUCB < 1.0
  - SVR = 0.000

Output schema matches committed evaluation/results/saps_60day.json:
  {method, n_days, n_users, n_interactions, seed, wiring_fix,
   aggregate_metrics, ci95_reward, posteriors}

Also dumps per-user raw data to saps_60day_per_user.json.
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
from scipy import stats as sp_stats

N_DAYS = 60
SEED = 42
WIRING_FIX = "S2_pain_change_threshold_1.0 + set_safe_ids + ungated_IER + five_rule_accepted_SVR"


def run_method(algo: str, extra_overrides: dict | None = None):
    """Run a single method and return (metrics_dict, pipeline_object)."""
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

    metrics = {
        "method": algo,
        "n_days": N_DAYS,
        "n_users": len(user_logs),
        "n_interactions": len(logs),
        "cum_reward_mean": float(np.mean(cum_rewards)),
        "cum_reward_std": float(np.std(cum_rewards)),
        "ier_mean": float(np.mean(ier_vals)),
        "acceptance_rate": float(np.mean(acc_vals)),
        "svr": float(np.mean(svr_vals)),
        "diversity_mean": float(np.mean(div_vals)),
        "per_user_cum_rewards": [float(x) for x in cum_rewards],
        "per_user_ier": [float(x) for x in ier_vals],
    }
    return metrics, pipe


def extract_posteriors(pipe):
    """Extract SAPS Beta posteriors from the pipeline's model."""
    model = pipe.recommender.model
    alpha_dict = dict(getattr(model, "safe_alpha", {}))
    beta_dict = dict(getattr(model, "safe_beta", {}))
    a0 = getattr(model, "_a0", 1.0)
    b0 = getattr(model, "_b0", 1.0)

    posteriors = {}
    for arm_id in sorted(alpha_dict.keys(), key=lambda x: int(x[1:])):
        a = float(alpha_dict[arm_id])
        b = float(beta_dict.get(arm_id, b0))
        # Skip arms still at prior (never chosen)
        if a == a0 and b == b0:
            continue
        posteriors[arm_id] = {
            "alpha": a,
            "beta": b,
            "mean": round(a / (a + b), 4),
        }
    return posteriors


def main():
    parser = argparse.ArgumentParser(description="SAPS 60-day formal verification")
    parser.add_argument("--out", type=str, default=None,
                        help="Override output path for saps_60day.json")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[T4] SAPS vs LinUCB 60-day comparison")
    print(f"     Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    t0 = time.time()

    # ── Run LinUCB baseline ──────────────────────────────────────────────
    print("\n▶ Running LinUCB baseline (60 days)...")
    t1 = time.time()
    linucb, _ = run_method("linucb")
    print(f"  Done in {time.time()-t1:.1f}s")

    # ── Run SAPS ─────────────────────────────────────────────────────────
    print("\n▶ Running SAPS (60 days)...")
    t2 = time.time()
    saps, saps_pipe = run_method("saps")
    print(f"  Done in {time.time()-t2:.1f}s")

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s")

    # ── T4 verification ─────────────────────────────────────────────────
    ier_ok = saps["ier_mean"] <= linucb["ier_mean"]
    reward_gap = abs(saps["cum_reward_mean"] - linucb["cum_reward_mean"])
    reward_ok = reward_gap < 1.0
    svr_ok = saps["svr"] == 0.0

    print("\n" + "=" * 60)
    print("  SAPS 60-day results:")
    print(f"    Cum. Reward (mean): {saps['cum_reward_mean']:.2f} ± {saps['cum_reward_std']:.2f}")
    print(f"    IER (mean):         {saps['ier_mean']:.4f}")
    print(f"    Acceptance Rate:    {saps['acceptance_rate']:.4f}")
    print(f"    SVR:                {saps['svr']:.4f}")
    print(f"    Diversity:          {saps['diversity_mean']:.4f}")
    print()
    print("  LinUCB 60-day baseline:")
    print(f"    Cum. Reward (mean): {linucb['cum_reward_mean']:.2f} ± {linucb['cum_reward_std']:.2f}")
    print(f"    IER (mean):         {linucb['ier_mean']:.4f}")
    print(f"    Acceptance Rate:    {linucb['acceptance_rate']:.4f}")
    print(f"    SVR:                {linucb['svr']:.4f}")
    print()
    print("  T4 Verification:")
    print(f"    {'✅' if ier_ok else '❌'} IER ≤ LinUCB:      SAPS {saps['ier_mean']:.4f} vs LinUCB {linucb['ier_mean']:.4f}")
    print(f"    {'✅' if reward_ok else '❌'} Reward gap < 1.0:  |{saps['cum_reward_mean']:.2f} - {linucb['cum_reward_mean']:.2f}| = {reward_gap:.2f}")
    print(f"    {'✅' if svr_ok else '❌'} SVR = 0.000:       {saps['svr']:.4f}")
    print()

    # ── Paired t-test ────────────────────────────────────────────────────
    for metric_name, saps_vals, linucb_vals in [
        ("Cum.Reward", saps["per_user_cum_rewards"], linucb["per_user_cum_rewards"]),
        ("IER", saps["per_user_ier"], linucb["per_user_ier"]),
    ]:
        t_stat, p_val = sp_stats.ttest_rel(saps_vals, linucb_vals)
        delta = np.mean(saps_vals) - np.mean(linucb_vals)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  Paired t-test ({metric_name}): Δ={delta:+.4f}, p={p_val:.4f} {sig}")

    # ── CI95 (t-interval, full precision) ────────────────────────────────
    per_user = saps["per_user_cum_rewards"]
    n = len(per_user)
    ci_low, ci_high = sp_stats.t.interval(
        0.95, df=n - 1,
        loc=np.mean(per_user),
        scale=sp_stats.sem(per_user),
    )
    print(f"\n  CI95 Reward: [{ci_low:.4f}, {ci_high:.4f}]")

    # ── Extract SAPS posteriors ──────────────────────────────────────────
    posteriors = extract_posteriors(saps_pipe)
    print(f"  Arms w/ posteriors: {len(posteriors)}")

    # ── Save enriched result (committed schema) ──────────────────────────
    result = {
        "method": "saps",
        "n_days": N_DAYS,
        "n_users": saps["n_users"],
        "n_interactions": saps["n_interactions"],
        "seed": SEED,
        "wiring_fix": WIRING_FIX,
        "aggregate_metrics": {
            "cum_reward_mean": saps["cum_reward_mean"],
            "cum_reward_std": saps["cum_reward_std"],
            "ier_mean": saps["ier_mean"],
            "acceptance_rate": saps["acceptance_rate"],
            "svr": saps["svr"],
            "diversity_mean": saps["diversity_mean"],
        },
        "ci95_reward": [float(ci_low), float(ci_high)],
        "posteriors": posteriors,
    }

    out_path = Path(args.out) if args.out else results_dir / "saps_60day.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ── Save per-user raw data (provenance dump) ─────────────────────────
    per_user_data = {
        "saps_per_user_cum_rewards": saps["per_user_cum_rewards"],
        "saps_per_user_ier": saps["per_user_ier"],
        "linucb_per_user_cum_rewards": linucb["per_user_cum_rewards"],
        "linucb_per_user_ier": linucb["per_user_ier"],
    }
    per_user_path = results_dir / "saps_60day_per_user.json"
    with open(per_user_path, "w") as f:
        json.dump(per_user_data, f, indent=2)
    print(f"Saved: {per_user_path}")


if __name__ == "__main__":
    main()
