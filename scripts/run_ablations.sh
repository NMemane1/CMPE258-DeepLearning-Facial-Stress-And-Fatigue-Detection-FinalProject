#!/usr/bin/env bash
# Run all six ablation variants sequentially.
# Usage:  bash scripts/run_ablations.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="ablation_results.json"
echo "Running full ablation matrix → $OUT"
python -m src.evaluation.ablation --base-config src/config/config.yaml --out "$OUT"
echo "Done. Results in $OUT"
