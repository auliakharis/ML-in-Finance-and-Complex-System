# notes

Dataclass modules for the footnote disclosures included in a 10-Q report. Each module covers one category of notes.

| File | Notes Section |
|---|---|
| `note_debt.py` | Debt obligations — long-term debt, credit facilities, maturities |
| `note_equity.py` | Equity disclosures — share repurchases, dividends, stock compensation |
| `note_revenue.py` | Revenue disaggregation by product/geography/timing |
| `note_segments.py` | Segment reporting — revenue and profit by business segment |
| `note_commitments.py` | Commitments and contingencies — leases, legal proceedings |
| `notes.py` | Aggregates all note sections into a single `Notes` object |
| `__init__.py` | Re-exports all note classes |

These are combined with the financial statements in `generate.py` to form a complete `FinancialReport`.
