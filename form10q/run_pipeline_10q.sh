#!/bin/bash
# Run the compiler pipeline on the 10-Q dataset.
# Produces output_form10q/random_questions_10q.csv (and .json via post-step).
#
# Usage:
#   cd 10q && bash run_pipeline_10q.sh
#   bash run_pipeline_10q.sh --n 500 --depth-max 4 --seed 42

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/../compiler_pipeline"
OUT_DIR="$SCRIPT_DIR/output_10q"
CSV="$SCRIPT_DIR/financial_spreadsheet.csv"

# --- defaults (overridable via env or args passed through) ---
N="${N:-500}"
DEPTH_MIN="${DEPTH_MIN:-0}"
DEPTH_MAX="${DEPTH_MAX:-4}"
DERIVED_PROB_MIN="${DERIVED_PROB_MIN:-0.1}"
DERIVED_PROB_MAX="${DERIVED_PROB_MAX:-0.6}"
SEED="${SEED:-2345}"

mkdir -p "$OUT_DIR"

echo "=== Step 1: Build 10-Q atoms ==="
python3 "$SCRIPT_DIR/build_atoms_10q.py" \
    --csv "$CSV" \
    --output "$OUT_DIR/atoms_data.json"

echo ""
echo "=== Step 2: Sample expression trees ==="
python3 "$PIPELINE_DIR/3.fixed_tree_sampler.py" \
    --depth "$DEPTH_MAX" \
    --seed "$SEED" \
    --derived-prob 0.4 \
    --output "$OUT_DIR/generated_operator_tree.json"

echo ""
echo "=== Step 3: Generate questions ==="
python3 "$PIPELINE_DIR/5.make_random_questions.py" \
    --csv "$CSV" \
    --atoms-module "$SCRIPT_DIR/build_atoms_10q.py" \
    --n "$N" \
    --depth-min "$DEPTH_MIN" \
    --depth-max "$DEPTH_MAX" \
    --derived-prob-min "$DERIVED_PROB_MIN" \
    --derived-prob-max "$DERIVED_PROB_MAX" \
    --seed "$SEED" \
    --output "$OUT_DIR/random_questions_10q.csv"

echo ""
echo "=== Done ==="
echo "Output: $OUT_DIR/random_questions_10q.csv"
