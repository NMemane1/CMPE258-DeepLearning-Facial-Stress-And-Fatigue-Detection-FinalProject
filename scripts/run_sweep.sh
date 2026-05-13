#!/usr/bin/env bash
# Launch a W&B sweep over the hyperparameter space defined in src/config/sweep.yaml
# Usage:  bash scripts/run_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Creating sweep…"
SWEEP_ID=$(wandb sweep src/config/sweep.yaml 2>&1 | grep "wandb agent" | awk '{print $NF}')
echo "Sweep ID: $SWEEP_ID"

echo "Launching agent (will run until run_cap is hit)…"
wandb agent "$SWEEP_ID"
