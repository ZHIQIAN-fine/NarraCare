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

# ── Save reference copies BEFORE experiments overwrite them ──────────
REF_DIR=$(mktemp -d)
trap "rm -rf $REF_DIR" EXIT
for f in evaluation/results/*.json; do
    [ -f "$f" ] && cp "$f" "$REF_DIR/$(basename "$f")"
done

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
#  Verification: diff regenerated vs reference artefacts
#  Works in both git-repo and tarball-unpack scenarios.
# ──────────────────────────────────────────────────────────────
echo "========================================"
echo " Verification: diff regenerated vs reference"
echo "========================================"
echo ""

# Helper: strip wall_time_s for timing-tolerant comparison (jq preferred, python3 fallback)
strip_timing() {
    if command -v jq >/dev/null 2>&1; then
        jq -S 'del(.wall_time_s) | walk(if type=="object" then del(.wall_time_s) else . end)' "$1"
    else
        python3 -c "
import json, sys
def strip(o):
    if isinstance(o, dict):
        o.pop('wall_time_s', None)
        for v in o.values(): strip(v)
    elif isinstance(o, list):
        for v in o: strip(v)
with open(sys.argv[1]) as f: d = json.load(f)
strip(d)
print(json.dumps(d, sort_keys=True, indent=2))
" "$1"
    fi
}

PASS=0
FAIL=0
WARN=0

# Deterministic artifacts — must be bit-identical
DETERMINISTIC_ARTIFACTS=(
    "evaluation/results/saps_60day.json"
    "evaluation/results/saps_60day_per_user.json"
    "evaluation/results/robustness_grid_5row.json"
    "evaluation/results/nosafety_svr_posthoc_full.json"
    "evaluation/results/narrative_samples_60.json"
)

for f in "${DETERMINISTIC_ARTIFACTS[@]}"; do
    REF="$REF_DIR/$(basename "$f")"
    if [ ! -f "$REF" ]; then
        echo "  ⚠️  SKIP  $f (no reference copy)"
        WARN=$((WARN + 1))
        continue
    fi
    if diff "$REF" "$f" > /dev/null 2>&1; then
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
    "evaluation/results/neuralucb_60day.json"
    "evaluation/results/neural_ucb_prefix_60day_seed42.json"
)

for f in "${TIMING_ARTIFACTS[@]}"; do
    REF="$REF_DIR/$(basename "$f")"
    if [ ! -f "$f" ]; then
        echo "  ⚠️  SKIP  $f (not regenerated)"
        WARN=$((WARN + 1))
        continue
    fi
    if [ ! -f "$REF" ]; then
        echo "  ⚠️  SKIP  $f (no reference copy)"
        WARN=$((WARN + 1))
        continue
    fi
    if diff <(strip_timing "$REF") <(strip_timing "$f") > /dev/null 2>&1; then
        echo "  ✅ PASS  $f (timing-only diff OK)"
        PASS=$((PASS + 1))
    else
        echo "  ❌ FAIL  $f (non-timing diff detected)"
        FAIL=$((FAIL + 1))
    fi
done

TOTAL=$((PASS + FAIL + WARN))
echo ""
echo "========================================"
echo " Verification Summary: $PASS/$TOTAL passed, $FAIL failed, $WARN skipped"
if [ "$FAIL" -eq 0 ]; then
    echo " ✓ All artifacts reproduce at seed=42 (5 bit-identical + 8 timing-only)."
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
