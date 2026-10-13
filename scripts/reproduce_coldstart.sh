#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# reproduce_coldstart.sh — reproduce §7.4 cold-start experiment
#
# Usage:
#   cd NarraCare
#   bash scripts/reproduce_coldstart.sh
#
# Expected wall-time: ~30 sec
# Outputs:
#   evaluation/results/coldstart_60day_seed42.json
#   evaluation/results/coldstart_posteriors_saps.json
#   evaluation/results/coldstart_posteriors_linucb.json
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================"
echo " NarraCare — Cold-Start Transfer (§7.4)"
echo " Train: U001-U010,U016-U020 (10 ACL + 5 RC)"
echo " Rollout: U011-U015 (5 Back-Pain)"
echo " Seed: 42 | Days: 60"
echo "========================================"
echo ""

python evaluation/run_coldstart.py

echo ""
echo " ✓ Cold-start artefacts saved to evaluation/results/"
