#!/usr/bin/env bash
set -euo pipefail

# Generate all terazi benchmark data
# Usage: ./scripts/generate_all.sh [num_examples_per_category]

NUM_EXAMPLES="${1:-500}"
BATCH_SIZE="${2:-5}"

echo "Generating terazi benchmark data..."
echo "  Examples per category: $NUM_EXAMPLES"
echo "  Batch size: $BATCH_SIZE"
echo ""

for category in core tool fin legal; do
    echo "=== terazi-$category ==="
    terazi generate --category "$category" --num-examples "$NUM_EXAMPLES" --batch-size "$BATCH_SIZE"
    echo ""
done

echo "Done. All data saved to data/"
