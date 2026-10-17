#!/usr/bin/env bash
# Reproduce every final report artifact; also works from the submitted tarball.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/narracare-matplotlib}"
OUT_DIR=$(mktemp -d)
trap 'rm -rf "$OUT_DIR"' EXIT
python -m evaluation.run_submission --output-dir "$OUT_DIR" --reference evaluation/results/final
