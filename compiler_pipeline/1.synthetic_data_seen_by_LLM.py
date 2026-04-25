"""
STEP 1: Generate Financial Spreadsheet
========================================
Creates a simplified synthetic financial spreadsheet for benchmarking
LLM reasoning over structured financial data. Numbers are randomly generated but
modeled on realistic corporate ratios (e.g. COGS as 30-75% of revenue). The
accounting identities (e.g. assets = liabilities + equity) are enforced by construction.
Many real line items (retained earnings, PP&E, long-term debt, etc.) are omitted.

Schema: one row per (company, year), 15 companies x 6 years = 90 rows.

Columns:
  - Categorical: company_name, ticker, sector, country, exchange, credit_rating
  - Income statement: revenue, cost_of_goods_sold, operating_expenses,
                      non_operating_expenses, income_tax (rate, not amount)
  - Balance sheet: total_assets, total_liabilities, total_equity,
                   cash, accounts_receivable, inventories,
                   short_term_investments, current_liabilities
  - Cash flow: capex, dividends_paid
  - Market data: stock_price, shares_outstanding
  - Company profile: employees

Output: output/financial_spreadsheet.csv  -> consumed by steps 2 and 5
        output/schema.json               -> human-readable column metadata, not used by the pipeline
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

random.seed(42)

YEARS = list(range(2020, 2026))

# Categorical columns (NOT numeric — can only use = or ≠)
CATEGORICAL_COLS = {
    "company_name":  {"type": "categorical", "desc": "Company name"},
    "ticker":        {"type": "categorical", "desc": "Stock ticker symbol"},
    "sector":        {"type": "categorical", "desc": "Industry sector"},
    "country":       {"type": "categorical", "desc": "Country of headquarters"},
    "exchange":      {"type": "categorical", "desc": "Stock exchange"},
    "credit_rating": {"type": "ordinal",     "desc": "Credit rating",
                      "order": ["CCC", "B", "BB", "BBB", "A", "AA", "AAA"]},
}

# Numeric columns (per-year) — these can use all math operations
YEARLY_NUMERIC_COLS = {
    "year":                   {},
    "revenue":                {"unit": "M_USD",   "range": (5000, 500000)},
    "cost_of_goods_sold":     {"unit": "M_USD"},
    "operating_expenses":     {"unit": "M_USD"},
    "non_operating_expenses": {"unit": "M_USD"},
    "income_tax":             {"unit": "ratio",   "range": (0.10, 0.30)},

    "total_assets":           {"unit": "M_USD"},
    "total_liabilities":      {"unit": "M_USD"},
    "total_equity":           {"unit": "M_USD"},
    "cash":                   {"unit": "M_USD"},
    "accounts_receivable":    {"unit": "M_USD"},
    "inventories":            {"unit": "M_USD"},
    "short_term_investments": {"unit": "M_USD"},
    "current_liabilities":    {"unit": "M_USD"},

    # Requires multi-period equity / retained earnings (interesting to check if LLM can infer these on its own later!)
    "capex":                  {"unit": "M_USD"},
    "dividends_paid":         {"unit": "M_USD"},

    "shares_outstanding":     {"unit": "M_shares","range": (100, 15000)},
    "stock_price":            {"unit": "USD",     "range": (15, 800)},
    "employees":              {"unit": "count",   "range": (5000, 500000)},
}

# Company templates
COMPANIES = [
    ("Apex Dynamics Corp",       "APEX", "Technology",     "USA",         "NYSE"),
    ("Meridian Systems Inc",     "MSYS", "Technology",     "USA",         "NASDAQ"),
    ("Quantum Bridge Holdings",  "QBHD", "Healthcare",     "USA",         "NYSE"),
    ("NovaStar Technologies",    "NVST", "Technology",     "Germany",     "XETRA"),
    ("Atlas Global Industries",  "ATGL", "Industrials",    "UK",          "LSE"),
    ("Pinnacle Health Group",    "PNHG", "Healthcare",     "USA",         "NYSE"),
    ("Ironclad Manufacturing",   "IRCL", "Industrials",    "Japan",       "TSE"),
    ("SilverLine Retail Inc",    "SLRI", "Consumer",       "USA",         "NASDAQ"),
    ("Pacific Crest Energy",     "PCEN", "Energy",         "Canada",      "TSX"),
    ("Orion Financial Group",    "ORFG", "Financials",     "UK",          "LSE"),
    ("Titan Consumer Brands",    "TICB", "Consumer",       "France",      "Euronext"),
    ("Cobalt Semiconductor",     "CBSM", "Technology",     "Taiwan",      "TWSE"),
    ("Helix BioSciences",        "HLXB", "Healthcare",     "USA",         "NASDAQ"),
    ("ClearPath Logistics",      "CPTH", "Industrials",    "USA",         "NYSE"),
    ("Crimson Pharmaceuticals",  "CRPH", "Healthcare",     "Switzerland", "SIX"),
]

OUTPUT_DIR = Path("output")
CSV_PATH = OUTPUT_DIR / "financial_spreadsheet.csv"
SCHEMA_PATH = OUTPUT_DIR / "schema.json"


def ensure_output_dir(path: Path) -> None:
    # Ensure downstream writes never fail on missing folders.
    path.mkdir(parents=True, exist_ok=True)


def generate_company_row(company_tuple: Tuple[str, str, str, str, str]) -> List[Dict[str, Any]]:
    """Generate one row of the spreadsheet for one company."""
    # Unpack static company attributes used in every yearly row.
    name, ticker, sector, country, exchange = company_tuple
    #creates a unique row prefix for each company
    row_prefix = {
        "company_name": name,
        "ticker": ticker,
        "sector": sector,
        "country": country,
        "exchange": exchange,
        "credit_rating": random.choice(CATEGORICAL_COLS["credit_rating"]["order"]),
    }

    # Generate base revenue, then derive everything else consistently
    base_revenue = random.uniform(*YEARLY_NUMERIC_COLS["revenue"]["range"])

    # Collect one record per year for this company.
    rows = []
    for year in YEARS:
        row = row_prefix.copy()

        # Revenue grows/shrinks slightly each year
        growth = random.uniform(-0.08, 0.20)
        revenue = round(base_revenue * (1 + growth))
        base_revenue = revenue  # next year starts from here

        # Derive income-statement drivers from revenue.
        cogs = round(revenue * random.uniform(0.30, 0.75))
        opex = round(revenue * random.uniform(0.05, 0.25))
        nonopex = round(revenue * random.uniform(0.05, 0.25))
        income_tax = random.uniform(*YEARLY_NUMERIC_COLS["income_tax"]["range"])

        # Derive balance-sheet items tied to revenue scale.
        # Asset turnover (revenue / assets) derives total assets from revenue.
        asset_turnover = random.uniform(0.3, 1.5)
        total_assets = round(revenue / asset_turnover)
        # Debt-to-equity ratio splits assets into equity and liabilities; enforces assets = liabilities + equity.
        de_ratio = random.uniform(0.3, 3.0)
        total_equity = round(total_assets / (1 + de_ratio))
        total_liabilities = total_assets - total_equity

        # Cash: 3-25% of total assets (liquidity buffer).
        cash = round(total_assets * random.uniform(0.03, 0.25))
        # Accounts receivable: 4-14% of revenue (~15-51 days sales outstanding).
        ar = round(revenue * random.uniform(0.04, 0.14))
        # Inventories: 2-15% of COGS (days inventory outstanding anchor).
        inv = round(cogs * random.uniform(0.02, 0.15))
        # Short-term investments: 1-15% of total assets (excess cash parked in securities).
        short_term_investments = round(total_assets * random.uniform(0.01, 0.15))
        # Current liabilities: 25-50% of total liabilities (remainder is long-term).
        cl = round(total_liabilities * random.uniform(0.25, 0.50))

        # Derive cash-flow and market/profile fields.
        net_income = round((revenue - cogs - opex - nonopex) * (1 - income_tax))
        dividends = round(max(0, net_income * random.uniform(0.0, 0.40)))
        capex = round(revenue * random.uniform(0.02, 0.10))
        
        shares = round(random.uniform(*YEARLY_NUMERIC_COLS["shares_outstanding"]["range"]))
        price = round(random.uniform(*YEARLY_NUMERIC_COLS["stock_price"]["range"]), 2)
        employees = round(random.uniform(*YEARLY_NUMERIC_COLS["employees"]["range"]))

        # Persist normalized row-level columns used by later pipeline steps.
        row["year"] = year
        row["revenue"] = revenue
        row["cost_of_goods_sold"] = cogs
        row["operating_expenses"] = opex
        row["non_operating_expenses"] = nonopex
        row["income_tax"] = income_tax

        row["total_assets"] = total_assets
        row["total_liabilities"] = total_liabilities
        row["total_equity"] = total_equity
        row["cash"] = cash
        row["accounts_receivable"] = ar
        row["inventories"] = inv
        row["short_term_investments"] = short_term_investments
        row["current_liabilities"] = cl
        
        row["capex"] = capex
        row["dividends_paid"] = dividends

        row["shares_outstanding"] = shares
        row["stock_price"] = price
        row["employees"] = employees
        rows.append(row)

    return rows


def build_schema(columns: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Build a metadata dict for each CSV column and write it to schema.json.

    Each column gets a small record describing its type, unit, and how the year
    dimension is encoded. Three kinds of columns are handled:

      Categorical/ordinal  -> type, desc, year=null (ordinals also get an order list)
      "year" column        -> type=numeric, unit="year", year=null
      Numeric concepts     -> type=numeric, unit from YEARLY_NUMERIC_COLS, year="row_level"

    year=null means the column is not time-indexed.
    year="row_level" means the year lives in the separate "year" column, not in the column name.
    """
    schema: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        # Map categorical columns directly from predefined metadata.
        if col in CATEGORICAL_COLS:
            info = CATEGORICAL_COLS[col]
            schema[col] = {"type": info["type"], "desc": info["desc"], "year": None}
            if info["type"] == "ordinal":
                schema[col]["order"] = info["order"]
            continue

        # In this pipeline the numeric dataset is flattened to one row per year,
        # so numeric column names are plain concept names (not revenue_2024 style).
        if col == "year":
            schema[col] = {"type": "numeric", "base_name": "year", "year": None, "unit": "year", "desc": "reporting year"}
            continue

        # Map numeric concepts to unit/type metadata.
        if col in YEARLY_NUMERIC_COLS:
            unit = YEARLY_NUMERIC_COLS[col].get("unit", "M_USD")
            schema[col] = {
                "type": "numeric",
                "base_name": col,
                "year": "row_level",
                "unit": unit,
                "desc": col.replace("_", " "),
            }
    return schema


def print_preview(rows: List[Dict[str, Any]], columns: Sequence[str], schema: Dict[str, Dict[str, Any]]) -> None:
    # Print compact dataset stats for quick sanity checks.
    print(f"Generated spreadsheet rows: {len(rows)}")
    print(f"  Categorical columns: {len(CATEGORICAL_COLS)}")
    print(
        f"  Numeric columns: {len(columns) - len(CATEGORICAL_COLS)} "
        f"({len(YEARLY_NUMERIC_COLS)} metrics across {len(YEARS)} years)"
    )

    print("\nPreview (first 3 rows):")
    for row in rows[:3]:
        print(
            f"  {row['ticker']:>5} | {row['company_name']:<30} | {row['sector']:<15} | "
            f"Year: {row['year']} | Rev: ${row['revenue']:>10,} | Tax rate: {row['income_tax']:.2%}"
        )

    print("\nAll columns:")
    for i, col in enumerate(columns):
        ctype = schema.get(col, {}).get("type", "?")
        print(f"  {i:>2}. {col:<35} [{ctype}]")


def main() -> None:
    # Generate all company-year rows.
    rows = []
    for c in COMPANIES:
        rows.extend(generate_company_row(c))

    if not rows:
        raise ValueError("No rows were generated from company templates.")

    # Preserve stable column order from generated dictionaries.
    columns = list(rows[0].keys())

    # Write the synthetic dataset CSV.
    ensure_output_dir(OUTPUT_DIR)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # Write and print schema/preview artifacts for inspection.
    schema = build_schema(columns)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print_preview(rows, columns, schema)
    print(f"\nSaved: {CSV_PATH}")
    print(f"Saved: {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
