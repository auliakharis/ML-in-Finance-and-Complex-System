# compiler_pipeline

A five-step pipeline that generates synthetic financial data, samples random expression trees, and compiles them into natural-language Q&A pairs.

## Pipeline Steps

| Step | Script | Description |
|---|---|---|
| 1 | `1.synthetic_data_seen_by_LLM.py` | Generate a synthetic financial spreadsheet (companies × years × metrics) |
| 2 | `2.fixed_building_atoms.py` | Extract atomic data lookups ("atoms") from the spreadsheet |
| 3 | `3.fixed_tree_sampler.py` | Sample random expression trees (operators + leaves + derived concepts) |
| 4 | `4.tree_to_language.py` | Bind atoms to tree leaves and render concrete expressions |
| 5 | `5.make_random_questions.py` | Generate a large batch of random questions from the pipeline |

## Run the Full Pipeline

```bash
bash run_pipeline.sh
```

This creates all intermediate and final outputs under `output/`.

### Custom Parameters

```bash
# Step 3: control tree depth and derived concept probability
python3 3.fixed_tree_sampler.py --depth 3 --seed 37 --derived-prob 0.5 --output output/generated_operator_tree.json

# Step 5: generate 1000 questions with depths 0–3
python3 5.make_random_questions.py \
  --csv output/financial_spreadsheet.csv \
  --n 1000 \
  --depth-min 0 --depth-max 3 \
  --derived-prob-min 0.3 --derived-prob-max 0.7 \
  --seed 2345 \
  --output output/random_questions.csv
```

## Supported Operators

- **Arithmetic:** `add`, `subtract`, `mul`, `divide`
- **Aliases:** `sum`, `diff`, `ratio`
- **Time aggregations:** `min`, `max`, `average` / `avg`
- **Finance-specific:** `growth` (year-over-year % change)
- **Derived concepts:** named formulas like `gross_profit`, `current_assets`, etc.

## Output

See [output/](output/) for the artifacts produced by the last pipeline run.
