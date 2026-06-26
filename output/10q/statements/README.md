# statements

Dataclass modules for the five core financial statements in a 10-Q report. Each module defines the fields and logic for one statement.

| File | Statement |
|---|---|
| `statement_of_operations.py` | Income statement — revenue, cost of sales, operating expenses, EPS |
| `balance_sheet.py` | Assets, liabilities, and shareholders' equity at a point in time |
| `cash_flows.py` | Cash flows from operating, investing, and financing activities |
| `comprehensive_income.py` | Net income plus other comprehensive income items |
| `shareholders_equity.py` | Rollforward of equity from period to period |
| `__init__.py` | Re-exports all statement classes for convenient importing |

These modules are consumed by `generate.py` to assemble a complete `FinancialReport`.
