"""T4: SAPS 60-day full run — formal verification.

Verification criteria (R2):
  - IER ≤ LinUCB (LinUCB 60-day IER baseline required)
  - Cum. Reward gap vs LinUCB < 1.0
  - SVR = 0.000
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

# ── Helper: run one method ───────────────────────────────────────────────
def run_method(algo: str, extra_overrides: dict | None = None) -> dict:
    overrides = {"recommender": {"algorithm": algo}}
    if extra_overrides:
        overrides.update(extra_overrides)
    pipe = NarraCarePipeline("configs/default.yaml", config_overrides=overrides)
    logs = pipe.run_simulation(n_days=N_DAYS)

    user_logs = defaultdict(list)
    for ix in logs:
        user_logs[ix["user_id"]].append(ix)

    cum_rewards, ier_vals, acc_vals, svr_vals = [], [], [], []
    for uid, ulogs in user_logs.items():
        n = len(ulogs)
        cr = sum(ix.get("reward", 0) for ix in ulogs)
        cum_rewards.append(cr)
        acc = sum(1 for ix in ulogs if ix.get("action") == "accept") / n if n else 0
        acc_vals.append(acc)
        ier_per = [ix.get("ier", 0) for ix in ulogs if ix.get("ier") is not None]
        ier_vals.append(np.mean(ier_per) if ier_per else 0)
        svr_per = [1 for ix in ulogs if ix.get("safety_violation", False)]
        svr_vals.append(len(svr_per) / n if n else 0)

    return {
        "method": algo,
        "n_days": N_DAYS,
        "n_users": len(user_logs),
        "n_interactions": len(logs),
        "cum_reward_mean": float(np.mean(cum_rewards)),
        "cum_reward_std": float(np.std(cum_rewards)),
        "ier_mean": float(np.mean(ier_vals)),
        "acceptance_rate": float(np.mean(acc_vals)),
        "svr": float(np.mean(svr_vals)),
        "per_user_cum_rewards": [float(x) for x in cum_rewards],
        "per_user_ier": [float(x) for x in ier_vals],
    }


print("=" * 60)
print(f"[T4] SAPS vs LinUCB 60-day comparison")
print(f"     Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

t0 = time.time()

# ── Run LinUCB baseline ──────────────────────────────────────────────────
print("\n▶ Running LinUCB baseline (60 days)...")
t1 = time.time()
linucb = run_method("linucb")
print(f"  Done in {time.time()-t1:.1f}s")

# ── Run SAPS ─────────────────────────────────────────────────────────────
print("\n▶ Running SAPS (60 days)...")
t2 = time.time()
saps = run_method("saps")
print(f"  Done in {time.time()-t2:.1f}s")

elapsed = time.time() - t0
print(f"\nTotal wall time: {elapsed:.1f}s")

# ── T4 verification ─────────────────────────────────────────────────────
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

# ── Paired t-test ────────────────────────────────────────────────────────
from scipy import stats as sp_stats

for metric_name, saps_vals, linucb_vals in [
    ("Cum.Reward", saps["per_user_cum_rewards"], linucb["per_user_cum_rewards"]),
    ("IER", saps["per_user_ier"], linucb["per_user_ier"]),
]:
    t_stat, p_val = sp_stats.ttest_rel(saps_vals, linucb_vals)
    delta = np.mean(saps_vals) - np.mean(linucb_vals)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  Paired t-test ({metric_name}): Δ={delta:+.4f}, p={p_val:.4f} {sig}")

# ── Save results ─────────────────────────────────────────────────────────
results_dir = PROJECT_ROOT / "evaluation" / "results"
results_dir.mkdir(parents=True, exist_ok=True)

result = {
    "saps": {k: v for k, v in saps.items() if not k.startswith("per_user")},
    "linucb_baseline": {k: v for k, v in linucb.items() if not k.startswith("per_user")},
    "verification": {
        "ier_leq_linucb": bool(ier_ok),
        "reward_gap_lt_1": bool(reward_ok),
        "reward_gap": round(reward_gap, 4),
        "svr_zero": bool(svr_ok),
    },
    "wall_time_s": round(elapsed, 1),
}

out_path = results_dir / "saps_60day.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved: {out_path.relative_to(PROJECT_ROOT)}")
