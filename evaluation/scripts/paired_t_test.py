#!/usr/bin/env python3
"""Paired t-test between two policies on per-user cumulative reward and IER.

Usage examples:
    # SAPS vs LinUCB (per-user data already in saps_60day_per_user.json)
    python evaluation/scripts/paired_t_test.py \
        --comparison "SAPS vs LinUCB" \
        --a-label saps --b-label linucb \
        --per-user-file evaluation/results/saps_60day_per_user.json \
        --a-reward-key saps_per_user_cum_rewards \
        --b-reward-key linucb_per_user_cum_rewards \
        --a-ier-key saps_per_user_ier \
        --b-ier-key linucb_per_user_ier \
        --out evaluation/results/paired_t_saps_vs_linucb.json

    # SAPS vs NeuralUCB-postfix (NeuralUCB per-user extracted from logs)
    python evaluation/scripts/paired_t_test.py \
        --comparison "SAPS vs NeuralUCB-postfix" \
        --a-label saps --b-label neuralucb \
        --per-user-file evaluation/results/saps_60day_per_user.json \
        --a-reward-key saps_per_user_cum_rewards \
        --a-ier-key saps_per_user_ier \
        --b-logs-file evaluation/results/logs/NeuralUCB_60day_logs.json \
        --out evaluation/results/paired_t_saps_vs_neuralucb.json
"""
import argparse, json, sys, datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats as sp_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def extract_per_user_from_logs(logs_path: Path):
    """Extract per-user cumulative reward and mean IER from raw interaction logs."""
    with open(logs_path) as f:
        logs = json.load(f)
    user_logs = defaultdict(list)
    for ix in logs:
        user_logs[ix["user_id"]].append(ix)
    cum_rewards, ier_vals = [], []
    for uid in sorted(user_logs.keys()):
        ulogs = user_logs[uid]
        cum_rewards.append(sum(ix.get("reward", 0) for ix in ulogs))
        ier_per = [ix.get("ier", 0) for ix in ulogs if ix.get("ier") is not None]
        ier_vals.append(float(np.mean(ier_per)) if ier_per else 0.0)
    return cum_rewards, ier_vals


def paired_t(a_vals, b_vals, label):
    """Compute paired t-test and return dict with all statistics."""
    a = np.array(a_vals, dtype=float)
    b = np.array(b_vals, dtype=float)
    assert len(a) == len(b), f"Unequal lengths: {len(a)} vs {len(b)}"
    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    se = std_diff / np.sqrt(n)
    t_stat, p_val = sp_stats.ttest_rel(a, b)
    ci = sp_stats.t.interval(0.95, df=n - 1, loc=mean_diff, scale=se)
    cohen_d = mean_diff / std_diff if std_diff > 0 else float("inf")
    return {
        "a_mean": round(float(np.mean(a)), 6),
        "b_mean": round(float(np.mean(b)), 6),
        "mean_diff": round(mean_diff, 6),
        "std_diff": round(std_diff, 6),
        "t_statistic": round(float(t_stat), 6),
        "df": n - 1,
        "p_value_two_sided": float(p_val),
        "ci_95_lower": round(float(ci[0]), 6),
        "ci_95_upper": round(float(ci[1]), 6),
        "cohen_d_paired": round(cohen_d, 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Paired t-test between two policies")
    parser.add_argument("--comparison", required=True, help="Label, e.g. 'SAPS vs LinUCB'")
    parser.add_argument("--a-label", required=True)
    parser.add_argument("--b-label", required=True)
    parser.add_argument("--per-user-file", required=True,
                        help="JSON with per-user arrays (at least for policy A)")
    parser.add_argument("--a-reward-key", required=True)
    parser.add_argument("--a-ier-key", required=True)
    parser.add_argument("--b-reward-key", default=None,
                        help="Key in per-user-file for B rewards (if B data is in same file)")
    parser.add_argument("--b-ier-key", default=None,
                        help="Key in per-user-file for B IER (if B data is in same file)")
    parser.add_argument("--b-logs-file", default=None,
                        help="Raw interaction logs for B (used if --b-reward-key is not set)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    per_user_path = PROJECT_ROOT / args.per_user_file
    with open(per_user_path) as f:
        pu = json.load(f)

    a_rewards = pu[args.a_reward_key]
    a_ier = pu[args.a_ier_key]

    source_files = [args.per_user_file]

    if args.b_reward_key and args.b_reward_key in pu:
        b_rewards = pu[args.b_reward_key]
        b_ier = pu[args.b_ier_key]
    elif args.b_logs_file:
        logs_path = PROJECT_ROOT / args.b_logs_file
        b_rewards, b_ier = extract_per_user_from_logs(logs_path)
        source_files.append(args.b_logs_file)
    else:
        print("ERROR: Must provide either --b-reward-key or --b-logs-file", file=sys.stderr)
        sys.exit(1)

    n = len(a_rewards)
    assert len(b_rewards) == n, f"User count mismatch: A={n}, B={len(b_rewards)}"

    reward_result = paired_t(a_rewards, b_rewards, "reward")
    # Rename keys for clarity
    reward_out = {
        f"{args.a_label}_mean": reward_result.pop("a_mean"),
        f"{args.b_label}_mean": reward_result.pop("b_mean"),
        **reward_result,
    }

    ier_result = paired_t(a_ier, b_ier, "ier")
    ier_out = {
        f"{args.a_label}_mean": ier_result.pop("a_mean"),
        f"{args.b_label}_mean": ier_result.pop("b_mean"),
        **ier_result,
    }

    output = {
        "comparison": args.comparison,
        "pairing_unit": "user",
        "sample_size_users": n,
        "seed": args.seed,
        "reward": reward_out,
        "ier": ier_out,
        "source_files": source_files,
        "computed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "script": "evaluation/scripts/paired_t_test.py",
    }

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"=== {args.comparison} (n={n} users, paired t-test) ===")
    print(f"Reward: Δ = {reward_out['mean_diff']:.4f}, "
          f"t({reward_out['df']}) = {reward_out['t_statistic']:.4f}, "
          f"p = {reward_out['p_value_two_sided']:.2e}, "
          f"95% CI [{reward_out['ci_95_lower']:.4f}, {reward_out['ci_95_upper']:.4f}], "
          f"Cohen's d = {reward_out['cohen_d_paired']:.4f}")
    print(f"IER:    Δ = {ier_out['mean_diff']:.6f}, "
          f"t({ier_out['df']}) = {ier_out['t_statistic']:.4f}, "
          f"p = {ier_out['p_value_two_sided']:.2e}, "
          f"95% CI [{ier_out['ci_95_lower']:.6f}, {ier_out['ci_95_upper']:.6f}], "
          f"Cohen's d = {ier_out['cohen_d_paired']:.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
