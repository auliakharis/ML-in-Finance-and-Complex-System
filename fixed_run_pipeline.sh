#!/usr/bin/env bash
set -euo pipefail

python fixed_build_atoms.py --csv financial_spreadsheet.csv --output output/atoms_data.json
python fixed_tree_sampler.py --output output/tree_templates.json
python fixed_generate_90_questions.py \
  --compiler Victorie_tree_with_depth.py \
  --atoms output/atoms_data.json \
  --templates output/tree_templates.json \
  --output output/final_90_questions.json \
  --n 90
