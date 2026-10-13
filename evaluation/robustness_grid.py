"""R2.6 T1 — Robustness Grid: one-drop ablation on PF_R1–PF_R5.

For each rule, drops it from the pre-filter and runs LinUCB 60-day (seed=42).
Computes IER / post-hoc SVR delta vs baseline + paired t-test.
Enriches grid rows with solo_hit_count + note from nosafety_svr_posthoc_full.json.
"""
import sys, json, time, csv, logging, argparse, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logging.basicConfig(level=logging.WARNING)

from src.pipeline import NarraCarePipeline
from src.agents.safety_agent import SafetyAgent

N_DAYS = 60
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RULES = ["PF_R1", "PF_R2", "PF_R3", "PF_R4", "PF_R5"]


# ═════════════════════════════════════════════════════════════════════════
#  Per-rule post-hoc SVR check (same as Q1–Q4 fact-check)
# ═════════════════════════════════════════════════════════════════════════
def posthoc_svr_check(task, state, profile, agent):
    pain = state.get("pain_level", 0)
    fatigue = state.get("fatigue_level", 0)
    rom = state.get("rom", 180)
    user_stage = profile.get("current_stage", profile.get("stage", ""))
    user_contras = set(profile.get("contraindications", []))
    category = task.get("category", "").lower()
    difficulty = task.get("difficulty", 1)
    task_contras = set(task.get("contraindications", []))
    task_stages = task.get("applicable_stages", [])
    return {
        "PF_R1": bool(user_contras & task_contras),
        "PF_R2": bool(task_stages and user_stage and user_stage not in task_stages),
        "PF_R3": (pain >= agent.pain_threshold_high and difficulty >= 4),
        "PF_R4": (fatigue >= agent.fatigue_threshold_high
                   and category not in ("rest", "flexibility")),
        "PF_R5": (rom < 60 and (difficulty >= 3 or category == "cardio")),
    }


def run_one(label: str, drop_rule: Optional[str]) -> dict:
    print(f"\n▶ [{label}] drop={drop_rule or 'none (baseline)'}")
    t0 = time.time()
    overrides = {"recommender": {"algorithm": "linucb"}}
    if drop_rule:
        overrides["safety"] = {"drop_rule": drop_rule}
    pipe = NarraCarePipeline("configs/default.yaml", config_overrides=overrides)
    logs = pipe.run_simulation(n_days=N_DAYS)
    elapsed = time.time() - t0
    print(f"  {len(logs)} interactions, {elapsed:.1f}s")

    # Full safety agent for post-hoc SVR
    full_agent = SafetyAgent({"safety": {"enable": True}})

    user_logs = defaultdict(list)
    for ix in logs:
        user_logs[ix["user_id"]].append(ix)

    per_user = []
    for uid in sorted(user_logs.keys()):
        ul = user_logs[uid]
        n = len(ul)
        cr = sum(x.get("reward", 0) for x in ul)
        ier_vals = [x.get("ier", 0) for x in ul if x.get("ier") is not None]
        ier = float(np.mean(ier_vals)) if ier_vals else 0.0
        acc = sum(1 for x in ul if x.get("action") == "accept") / n if n else 0.0

        # Post-hoc SVR
        accepted = [x for x in ul if x.get("action") == "accept"]
        svr_count = 0
        for ix in accepted:
            profile = {"condition": ix.get("user_condition", ""),
                       "current_stage": ix.get("user_stage", ""),
                       "contraindications": ix.get("user_contraindications", [])}
            rules = posthoc_svr_check(ix.get("chosen_task", {}),
                                       ix.get("state", {}), profile, full_agent)
            if any(rules.values()):
                svr_count += 1
        svr = svr_count / len(accepted) if accepted else 0.0

        per_user.append({
            "user_id": uid,
            "cum_reward": round(cr, 4),
            "ier": round(ier, 4),
            "svr": round(svr, 4),
            "acceptance_rate": round(acc, 4),
        })

    return {
        "label": label,
        "drop_rule": drop_rule,
        "n_days": N_DAYS,
        "per_user": per_user,
        "agg": {
            "cum_reward": round(float(np.mean([r["cum_reward"] for r in per_user])), 4),
            "ier": round(float(np.mean([r["ier"] for r in per_user])), 6),
            "svr": round(float(np.mean([r["svr"] for r in per_user])), 6),
            "acceptance_rate": round(float(np.mean([r["acceptance_rate"]
                                                     for r in per_user])), 4),
        },
        "wall_s": round(elapsed, 1),
    }


# ═════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("[R2.6 T1] Robustness Grid — 5-row one-drop ablation")
print(f"  Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 64)

# Baseline (no drop)
baseline = run_one("baseline", None)

# One-drop rows
rows = {}
for rule in RULES:
    rows[rule] = run_one(f"drop_{rule}", rule)

# ═════════════════════════════════════════════════════════════════════════
#  Compute deltas + paired t-tests vs baseline
# ═════════════════════════════════════════════════════════════════════════
grid_rows = []
print(f"\n{'='*64}")
print(f"  {'Row':<12} {'IER':>8} {'ΔIER':>8} {'p(IER)':>10}"
      f"  {'SVR':>8} {'ΔSVR':>8} {'p(SVR)':>10}")
print("-" * 64)

bl_ier = [u["ier"] for u in baseline["per_user"]]
bl_svr = [u["svr"] for u in baseline["per_user"]]
bl_rew = [u["cum_reward"] for u in baseline["per_user"]]

for rule in RULES:
    r = rows[rule]
    r_ier = [u["ier"] for u in r["per_user"]]
    r_svr = [u["svr"] for u in r["per_user"]]
    r_rew = [u["cum_reward"] for u in r["per_user"]]

    d_ier = float(np.mean(r_ier) - np.mean(bl_ier))
    d_svr = float(np.mean(r_svr) - np.mean(bl_svr))
    d_rew = float(np.mean(r_rew) - np.mean(bl_rew))

    _, p_ier = sp_stats.ttest_rel(r_ier, bl_ier)
    _, p_svr = sp_stats.ttest_rel(r_svr, bl_svr)
    _, p_rew = sp_stats.ttest_rel(r_rew, bl_rew)

    sig_ier = "***" if p_ier < .001 else "**" if p_ier < .01 else "*" if p_ier < .05 else "ns"
    sig_svr = "***" if p_svr < .001 else "**" if p_svr < .01 else "*" if p_svr < .05 else "ns"

    print(f"  {rule:<12} {r['agg']['ier']:>8.4f} {d_ier:>+8.4f} {p_ier:>9.4f} {sig_ier}"
          f"  {r['agg']['svr']:>8.4f} {d_svr:>+8.4f} {p_svr:>9.4f} {sig_svr}")

    grid_rows.append({
        "rule": rule,
        "ier": r["agg"]["ier"],
        "ier_delta": round(d_ier, 6),
        "ier_p": None if math.isnan(p_ier) else round(float(p_ier), 6),
        "svr": r["agg"]["svr"],
        "svr_delta": round(d_svr, 6),
        "svr_p": None if math.isnan(p_svr) else round(float(p_svr), 6),
        "reward": r["agg"]["cum_reward"],
        "reward_delta": round(d_rew, 4),
        "reward_p": None if math.isnan(p_rew) else round(float(p_rew), 6),
        "acceptance": r["agg"]["acceptance_rate"],
    })

print(f"  {'baseline':<12} {baseline['agg']['ier']:>8.4f} {'—':>8} {'—':>10}"
      f"  {baseline['agg']['svr']:>8.4f} {'—':>8} {'—':>10}")
print("=" * 64)

# ═════════════════════════════════════════════════════════════════════════
#  Enrich with solo_hit_count + note from nosafety_svr_posthoc_full.json
# ═════════════════════════════════════════════════════════════════════════
nosafety_path = RESULTS_DIR / "nosafety_svr_posthoc_full.json"
if nosafety_path.exists():
    with open(nosafety_path) as f:
        nosafety = json.load(f)
    per_rule_hits = nosafety.get("per_rule_hits", {})
    overlap_matrix = nosafety.get("overlap_matrix", {})

    for row in grid_rows:
        rule = row["rule"]
        total = per_rule_hits.get(rule, 0)
        # Sum all pairwise overlaps involving this rule
        overlap_sum = 0
        for pair_key, count in overlap_matrix.items():
            if rule in pair_key.split("&"):
                overlap_sum += count
        solo = total - overlap_sum
        row["solo_hit_count"] = solo

        # Generate note matching committed format
        if total == 0:
            row["note"] = "0 catches (structural zero, see \u00a77.2.1)"
        elif overlap_sum == 0:
            row["note"] = f"{total} catches, 0 overlap -> solo={solo}"
        else:
            # Find which rules overlap
            partners = []
            for pair_key, count in overlap_matrix.items():
                if count > 0 and rule in pair_key.split("&"):
                    other = [r for r in pair_key.split("&") if r != rule][0]
                    partners.append((other, count))
            partner_str = ", ".join(f"{c} overlap with {r}" for r, c in partners)
            if solo <= 3:
                suffix = "(redundantly protected)"
            elif solo >= 100:
                suffix = "(dominant unique source)"
            else:
                suffix = ""
            row["note"] = f"{total} total catches, {partner_str} -> solo={solo}" + (f" {suffix}" if suffix else "")

    print(f"\n✓ Enriched grid with solo_hit_count + note from {nosafety_path.name}")
else:
    print(f"\n⚠ {nosafety_path.name} not found — skipping solo_hit_count enrichment")

# ═════════════════════════════════════════════════════════════════════════
#  Save
# ═════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Robustness grid")
parser.add_argument("--out", type=str, default=None,
                    help="Override output path for robustness_grid_5row.json")
cli_args, _ = parser.parse_known_args()

result = {
    "experiment": "robustness_grid_one_drop",
    "n_days": N_DAYS,
    "seed": 42,
    "baseline": baseline["agg"],
    "grid": grid_rows,
    "per_user_data": {
        "baseline": baseline["per_user"],
        **{rule: rows[rule]["per_user"] for rule in RULES},
    },
}

out_path = Path(cli_args.out) if cli_args.out else RESULTS_DIR / "robustness_grid_5row.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2,
              default=lambda o: bool(o) if isinstance(o, (np.bool_,))
              else float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
print(f"\n✓ Saved: {out_path}")

# CSV for easy copy-paste into report table
csv_path = RESULTS_DIR / "robustness_grid_5row.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=grid_rows[0].keys())
    w.writeheader()
    w.writerows(grid_rows)
print(f"✓ Saved: {csv_path.name}")
