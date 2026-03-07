#!/usr/bin/env bash
set -euo pipefail

# Run terazi evaluation against a model
# Usage: ./scripts/run_eval.sh <model_name> [backend] [categories]

MODEL="${1:?Usage: $0 <model_name> [backend] [categories]}"
BACKEND="${2:-hf}"
CATEGORIES="${3:-core,tool,fin,legal}"

echo "Running terazi evaluation..."
echo "  Model: $MODEL"
echo "  Backend: $BACKEND"
echo "  Categories: $CATEGORIES"
echo ""

terazi eval --model "$MODEL" --backend "$BACKEND" --categories "$CATEGORIES"

echo ""
echo "Results:"
terazi results --format table
