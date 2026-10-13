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
echo " ✓ All artefacts reproduced."
echo "   Results: evaluation/results/"
echo "========================================"
echo ""
echo "NOTE: LLM-as-Judge scoring (evaluation/llm_judge_colab.py) requires"
echo "      an OpenAI API key and is designed to run in Google Colab."
echo "      See evaluation/llm_judge_colab.ipynb for instructions."
