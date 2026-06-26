# 10q

Generates realistic but fictional 10-Q (quarterly) financial reports and a small Q&A dataset from them.

## Contents

| File / Folder | Description |
|---|---|
| `generate.py` | Main report generator — picks a random company, quarter, and financial ratios to produce a full `FinancialReport` |
| `financial_report.py` | Dataclass definitions for `FinancialStatements` and `FinancialReport` |
| `run_pipeline.py` | End-to-end pipeline: generate reports → build spreadsheet → create Q&A pairs |
| `operations_10q.py` | Defines financial operations/questions applicable to 10-Q data |
| `rewrite_10q.py` | Rewrites generated questions for naturalness |
| `generate_csv.py` | Exports the financial data to CSV format |
| `schema.json` | JSON schema describing the financial spreadsheet structure |
| `statements/` | Dataclass modules for each financial statement |
| `notes/` | Dataclass modules for each footnote section |
| `industry_analysis/` | Fiscal year distribution analysis across industries |
| `final_qa_dataset.json/csv` | Generated Q&A pairs used for LLM evaluation |
| `financial_spreadsheet.json/csv` | Tabular financial data (context for eval prompts) |
| `top10_qa.json/csv` | Q&A subset for top-10 companies |

## Usage

```bash
# Generate a single report (for inspection)
python -c "from generate import generate_report; import json; print(json.dumps(generate_report(), default=str, indent=2))"

# Run the full pipeline
python run_pipeline.py
```

## Submodules

- **[statements/](statements/)** — balance sheet, cash flows, statement of operations, comprehensive income, shareholders equity
- **[notes/](notes/)** — debt, equity, revenue, segments, commitments
- **[industry_analysis/](industry_analysis/)** — distribution of fiscal years across generated companies
