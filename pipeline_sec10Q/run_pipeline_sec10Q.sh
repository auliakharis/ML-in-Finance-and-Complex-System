#!/bin/bash
# Run the compiler pipeline on the SEC 10-Q dataset.
# Produces datasets/sec10Q/output_sec10Q/random_questions_sec10Q.csv
#
# Usage:
#   bash pipeline_sec10Q/run_pipeline_sec10Q.sh
#   N=90 bash pipeline_sec10Q/run_pipeline_sec10Q.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../datasets/sec10Q/output_sec10Q"
CSV="$SCRIPT_DIR/../datasets/sec10Q/financial_spreadsheet.csv"

# --- defaults (overridable via env) ---
N="${N:-90}"
DEPTH_MIN="${DEPTH_MIN:-0}"
DEPTH_MAX="${DEPTH_MAX:-4}"
DERIVED_PROB_MIN="${DERIVED_PROB_MIN:-0.1}"
DERIVED_PROB_MAX="${DERIVED_PROB_MAX:-0.6}"
SEED="${SEED:-2345}"

mkdir -p "$OUT_DIR"

echo "=== Step 1: Build sec10Q atoms ==="
python3 "$SCRIPT_DIR/build_atoms_sec10Q.py" \
    --csv "$CSV" \
    --output "$OUT_DIR/atoms_data.json"

echo ""
echo "=== Step 2: Sample expression trees ==="
python3 "$SCRIPT_DIR/step3_tree_sampler_sec10Q.py" \
    --depth "$DEPTH_MAX" \
    --seed "$SEED" \
    --derived-prob 0.4 \
    --output "$OUT_DIR/generated_operator_tree.json"

echo ""
echo "=== Step 3: Generate questions ==="
python3 "$SCRIPT_DIR/step5_make_questions_sec10Q.py" \
    --csv "$CSV" \
    --n "$N" \
    --depth-min "$DEPTH_MIN" \
    --depth-max "$DEPTH_MAX" \
    --derived-prob-min "$DERIVED_PROB_MIN" \
    --derived-prob-max "$DERIVED_PROB_MAX" \
    --seed "$SEED" \
    --output "$OUT_DIR/random_questions_sec10Q.csv"

echo ""
echo "=== Done ==="
echo "Output: $OUT_DIR/random_questions_sec10Q.csv"
