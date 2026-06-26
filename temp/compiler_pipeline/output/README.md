# compiler_pipeline/output

Artifacts produced by the last run of `run_pipeline.sh`.

| File | Produced by | Description |
|---|---|---|
| `financial_spreadsheet.csv` | Step 1 | Synthetic financial spreadsheet (companies × years × metrics) |
| `atoms_data.json` | Step 2 | All atomic data lookups extracted from the spreadsheet |
| `generated_operator_tree.json` | Step 3 | A single sampled expression tree template |
| `concrete_expression.json` | Step 4 | The tree with atoms bound to leaves (concrete expression) |
| `schema.json` | Step 1 | Column schema for the spreadsheet |
| `random_questions_90_alicia.csv` | Step 5 | Large batch of generated Q&A pairs |

These files are intermediate/output artifacts — regenerate them by re-running `run_pipeline.sh`.
