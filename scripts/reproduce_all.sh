#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# reproduce_all.sh — reproduce ALL 60-day evaluation artefacts
#
# Usage:
#   cd NarraCare
#   bash scripts/reproduce_all.sh
#
# Expected wall-time: ~5 min (seed 42, no GPU required)
# All outputs land in evaluation/results/
# ──────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================"
echo " NarraCare — Full Reproduction Pipeline"
echo " Seed: 42 | Days: 60 | Users: 20"
echo "========================================"
echo ""

# 1. Main method comparison (SAPS + LinUCB)
echo "[1/7] SAPS + LinUCB 60-day formal run..."
python evaluation/run_saps_60day.py
echo ""

# 2. Baselines (Popularity, RuleBased, EpsilonGreedy, LinUCB-NoSafety)
echo "[2/7] Baselines 60-day..."
python evaluation/run_baselines_60day.py
echo ""

# 3. NeuralUCB (post-fix)
echo "[3/7] NeuralUCB 60-day..."
python evaluation/run_neuralucb_60day.py
echo ""

# 4. NeuralUCB pre-fix ablation
echo "[4/7] NeuralUCB pre-fix ablation..."
python evaluation/run_neuralucb_prefix_60day.py
echo ""

# 5. Post-hoc SVR + narrator invariants
echo "[5/7] Post-hoc SVR + narrator invariants..."
python evaluation/svr_narrator_invariants.py
python evaluation/q1q4_factcheck.py
python evaluation/nosafety_svr_full.py
echo ""

# 6. Robustness grid
echo "[6/7] Safety robustness grid..."
python evaluation/robustness_grid.py
echo ""

# 7. Narrative sampler (scoring requires OpenAI key — skip judge)
echo "[7/7] Narrative sampler..."
python evaluation/narrative_sampler.py
echo ""

echo "========================================"
echo " ✓ All artefacts regenerated."
echo "   Results: evaluation/results/"
echo "========================================"
echo ""

# ──────────────────────────────────────────────────────────────
#  Verification: diff regenerated vs committed artefacts
# ──────────────────────────────────────────────────────────────
echo "========================================"
echo " Verification: diff regenerated vs committed"
echo "========================================"
echo ""

PASS=0
FAIL=0
WARN=0

# Deterministic artifacts — must be bit-identical after JSON normalisation
DETERMINISTIC_ARTIFACTS=(
    "evaluation/results/saps_60day.json"
    "evaluation/results/saps_60day_per_user.json"
    "evaluation/results/robustness_grid_5row.json"
    "evaluation/results/nosafety_svr_posthoc_full.json"
    "evaluation/results/narrative_samples_60.json"
)

for f in "${DETERMINISTIC_ARTIFACTS[@]}"; do
    if git diff --exit-code -- "$f" > /dev/null 2>&1; then
        echo "  ✅ PASS  $f"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL  $f (diff detected)"
        FAIL=$((FAIL + 1))
    fi
done

# Timing-tolerant artifacts — ignore wall_time_s diffs only
TIMING_ARTIFACTS=(
    "evaluation/results/linucb_nosafety_60day_seed42.json"
    "evaluation/results/popularity_60day_seed42.json"
    "evaluation/results/rule_based_60day_seed42.json"
    "evaluation/results/epsilon_greedy_60day_seed42.json"
    "evaluation/results/random_60day.json"
    "evaluation/results/coldstart_60day_seed42.json"
)

for f in "${TIMING_ARTIFACTS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  ⚠️  SKIP  $f (not found)"
        WARN=$((WARN + 1))
        continue
    fi
    # Check if only wall_time_s differs
    DIFF_LINES=$(git diff -- "$f" | grep '^[+-]' | grep -v '^[+-][+-][+-]' | grep -cv 'wall_time_s' || true)
    if [ "$DIFF_LINES" -eq 0 ]; then
        echo "  ✅ PASS  $f (timing-only diff OK)"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL  $f (non-timing diff detected)"
        FAIL=$((FAIL + 1))
    fi
done

# NeuralUCB — nondeterministic (PyTorch CPU); check within tolerance
NEURAL_ARTIFACTS=(
    "evaluation/results/neuralucb_60day.json"
    "evaluation/results/neural_ucb_prefix_60day_seed42.json"
)

for f in "${NEURAL_ARTIFACTS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "  ⚠️  SKIP  $f (not found)"
        WARN=$((WARN + 1))
        continue
    fi
    python3 -c "
import json, sys
with open('$f') as fh:
    regen = json.load(fh)
regen_r = regen.get('cum_reward_mean', regen.get('aggregate_metrics',{}).get('cum_reward_mean', 0))
# Read committed version
import subprocess
committed = json.loads(subprocess.check_output(['git', 'show', 'HEAD:$f']))
comm_r = committed.get('cum_reward_mean', committed.get('aggregate_metrics',{}).get('cum_reward_mean', 0))
delta = abs(regen_r - comm_r)
if delta <= 1.0:
    print(f'  ✅ PASS  $f (Δ reward = {delta:.2f}, within ±1.0 tolerance)')
else:
    print(f'  ❌ FAIL  $f (Δ reward = {delta:.2f}, exceeds ±1.0 tolerance)')
    sys.exit(1)
" && PASS=$((PASS + 1)) || FAIL=$((FAIL + 1))
done

TOTAL=$((PASS + FAIL + WARN))
echo ""
echo "========================================"
echo " Verification Summary: $PASS/$TOTAL passed, $FAIL failed, $WARN skipped"
if [ "$FAIL" -eq 0 ]; then
    echo " ✓ All deterministic artifacts bit-identical."
    echo "   NeuralUCB within CPU nondeterminism tolerance."
else
    echo " ✗ $FAIL artifact(s) have unexpected diffs — inspect above."
fi
echo "========================================"
echo ""
echo "NOTE: LLM-as-Judge scoring (evaluation/llm_judge_colab.py) requires"
echo "      an OpenAI API key and is designed to run in Google Colab."
echo "      See evaluation/llm_judge_colab.ipynb for instructions."
echo ""
echo "NOTE: Cold-start evaluation runs separately via:"
echo "      bash scripts/reproduce_coldstart.sh"
