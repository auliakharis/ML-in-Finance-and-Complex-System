#!/bin/bash

mkdir -p output && \
python 1.synthetic_data_seen_by_LLM.py && \
python 2.fixed_building_atoms.py --csv output/financial_spreadsheet.csv --output output/atoms_data.json && \
python 3.fixed_tree_sampler.py --depth 3 --seed 37 --derived-prob 0.5 --output output/generated_operator_tree.json && \
python 4.tree_to_language.py --atoms output/atoms_data.json --tree output/generated_operator_tree.json --seed 131 --output output/concrete_expression.json && \
python 5.make_random_questions.py \
  --csv output/financial_spreadsheet.csv \
  --n 90 \
  --depth-min 0 \
  --depth-max 3 \
  --derived-prob-min 0.3 \
  --derived-prob-max 0.7 \
  --seed 2345 \
  --output output/random_questions_90.csv