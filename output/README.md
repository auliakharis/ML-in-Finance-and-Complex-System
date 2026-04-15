# output

Artifacts from the 10q pipeline run — financial data and Q&A pairs for ~10 companies.

| File | Description |
|---|---|
| `final_qa_dataset.json/csv` | Q&A pairs generated from the 10q reports |
| `financial_spreadsheet.json/csv` | Tabular financial data for context in eval prompts |
| `expression_tree.json` | Expression tree used to generate compound questions |
| `operations.json` | Operations/question templates applied to the data |
| `sampled_executed.json` | Executed expression trees with bound values |
| `schema.json` | Column schema for the financial spreadsheet |
| `company_index.json` | Index of companies in the dataset |
| `companies/` | Per-company financial data files |

These files are generated artifacts — re-run `10q/run_pipeline.py` to regenerate.
