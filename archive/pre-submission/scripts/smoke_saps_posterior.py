#!/usr/bin/env python3
"""Smoke / medium-scale test for SAPS Beta-posterior wiring (R2.8 §3.1 / §3.1.5).

Usage:
    # §3.1 smoke (3 days, 1 user)
    python scripts/smoke_saps_posterior.py --user U003 --days 3

    # §3.1.5 medium-scale (60 days, 1 user)
    python scripts/smoke_saps_posterior.py --user U003 --days 60

    # Full 20 users × 60 days (for rerun)
    python scripts/smoke_saps_posterior.py --days 60 --all_users

Checks:
  §3.1   — at least 3 arms with (α, β) ≠ (1, 1)
  §3.1.5 — at least 5 arms with |α/(α+β) − 0.5| > 0.3
           AND at least 3 arms with α + β > 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on sys.path
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from src.pipeline import NarraCarePipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="SAPS posterior smoke test")
    ap.add_argument("--user", default="U003", help="Single user ID for smoke")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--all_users", action="store_true",
                    help="Run all 20 users (ignores --user)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None,
                    help="Write posteriors JSON to this path")
    args = ap.parse_args()

    # Load pipeline with SAPS policy
    overrides = {
        "recommender": {"algorithm": "saps"},
        "simulation": {"n_days": args.days},
        "narrator": {"enable": False},
        "narrator_v2": {"enable": False},
    }

    pipeline = NarraCarePipeline(
        config_path="configs/default.yaml",
        config_overrides=overrides,
    )

    # Restrict to single user unless --all_users
    if not args.all_users:
        pipeline.simulator.profiles = {
            args.user: pipeline.simulator.profiles[args.user]
        }

    pipeline.run_simulation(n_days=args.days)

    # Extract posteriors
    model = getattr(pipeline.recommender, "model", None)
    if model is None:
        print("ERROR: recommender has no .model attribute")
        sys.exit(1)

    alpha_dict = dict(getattr(model, "safe_alpha", {}))
    beta_dict = dict(getattr(model, "safe_beta", {}))
    a0 = getattr(model, "_a0", 1.0)
    b0 = getattr(model, "_b0", 1.0)

    all_arms = sorted(set(alpha_dict.keys()) | set(beta_dict.keys()))

    posteriors = {}
    for arm in all_arms:
        a = alpha_dict.get(arm, a0)
        b = beta_dict.get(arm, b0)
        posteriors[arm] = {"alpha": a, "beta": b}

    # Print summary
    print(f"\n{'='*60}")
    print(f"SAPS Beta Posteriors — {args.user if not args.all_users else 'ALL'} "
          f"× {args.days} days, seed={args.seed}")
    print(f"{'='*60}")

    non_prior = 0
    deviated_arms = 0      # |mean - 0.5| > 0.3
    concentrated_arms = 0  # α + β > 5

    for arm in sorted(posteriors.keys()):
        a = posteriors[arm]["alpha"]
        b = posteriors[arm]["beta"]
        mean = a / (a + b) if (a + b) > 0 else 0.5
        total = a + b
        is_prior = (a == a0 and b == b0)
        if not is_prior:
            non_prior += 1
        dev = abs(mean - 0.5)
        if dev > 0.3:
            deviated_arms += 1
        if total > 5:
            concentrated_arms += 1
        flag = "" if is_prior else " *"
        print(f"  {arm:8s}  α={a:6.1f}  β={b:6.1f}  "
              f"mean={mean:.3f}  n={total:.0f}{flag}")

    print(f"\n--- Summary ---")
    print(f"  Arms with (α,β) ≠ prior:        {non_prior}")
    print(f"  Arms with |mean-0.5| > 0.3:      {deviated_arms}")
    print(f"  Arms with α+β > 5:               {concentrated_arms}")

    # §3.1 check
    if args.days <= 5:
        verdict = "PASS" if non_prior >= 3 else "FAIL"
        print(f"\n§3.1 Smoke ({non_prior} ≥ 3?): {verdict}")
    else:
        # §3.1.5 check
        v1 = deviated_arms >= 5
        v2 = concentrated_arms >= 3
        verdict = "PASS" if (v1 and v2) else "FAIL"
        print(f"\n§3.1.5 Medium-scale:")
        print(f"  |mean-0.5| > 0.3 ({deviated_arms} ≥ 5?): {'PASS' if v1 else 'FAIL'}")
        print(f"  α+β > 5          ({concentrated_arms} ≥ 3?): {'PASS' if v2 else 'FAIL'}")
        print(f"  Overall: {verdict}")

    if verdict == "FAIL":
        print("\n⚠️  STOPPING — criterion not met.")

    # Dump JSON
    out_data = {
        "user": args.user if not args.all_users else "ALL",
        "days": args.days,
        "seed": args.seed,
        "non_prior_count": non_prior,
        "deviated_count": deviated_arms,
        "concentrated_count": concentrated_arms,
        "verdict": verdict,
        "posteriors": posteriors,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nPosteriors written to {args.out}")
    else:
        # Also print to /tmp by default
        default_out = f"/tmp/saps_posterior_{args.days}d.json"
        with open(default_out, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nPosteriors written to {default_out}")


if __name__ == "__main__":
    main()
