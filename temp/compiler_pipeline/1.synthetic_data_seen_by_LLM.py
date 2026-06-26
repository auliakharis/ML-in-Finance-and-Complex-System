"""
STEP 1: Generate Financial Spreadsheet
========================================
Creates a typical Excel-style financial dataset.

The data has:
  - Multiple companies (rows)
  - Multiple years (columns grouped by year)
  - Mix of numeric and categorical columns
  - Both monetary values, percentages, counts, and labels

Output: financial_spreadsheet.csv + financial_spreadsheet.xlsx
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
    "cost_of_goods_sold":     {"unit": "M_USD",   "range": (2000, 350000)},
    "operating_expenses":     {"unit": "M_USD",   "range": (500, 80000)},
    "non_operating_expenses": {"unit": "M_USD",   "range": (-1000, 1000)},
    "income_tax":             {"unit": "M_USD",   "range": (200, 80000)},      

    "total_assets":           {"unit": "M_USD",   "range": (10000, 1000000)},
    "total_liabilities":      {"unit": "M_USD",   "range": (5000, 700000)},
    "total_equity":           {"unit": "M_USD",   "range": (3000, 400000)},
    "cash":                   {"unit": "M_USD",   "range": (1000, 150000)},
    "accounts_receivable":    {"unit": "M_USD",   "range": (500, 80000)},
    "inventories":            {"unit": "M_USD",   "range": (200, 50000)},
    "short_term_investments": {"unit": "M_USD",   "range": (10, 5000)},
    "current_liabilities":    {"unit": "M_USD",   "range": (2000, 200000)},

    # Requires multi-period equity / retained earnings (interesting to check if LLM can infer these on its own later!)
    "capex":                  {"unit": "M_USD",   "range": (500, 60000)},
    "dividends_paid":         {"unit": "M_USD",   "range": (0, 20000)},
    
    "shares_outstanding":     {"unit": "millions","range": (100, 15000)},
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

RATINGS = ["CCC", "B", "BB", "BBB", "A", "AA", "AAA"]
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
        "credit_rating": random.choice(RATINGS),
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
        cogs_pct = random.uniform(0.30, 0.75)
        cogs = round(revenue * cogs_pct)

        opex = round(revenue * random.uniform(0.05, 0.25))
        nonopex = round(revenue * random.uniform(0.05, 0.25))
        income_tax = random.uniform(0.10, 0.30)

        # Derive balance-sheet items tied to revenue scale.
        asset_turnover = random.uniform(0.3, 1.5)
        total_assets = round(revenue / asset_turnover)
        de_ratio = random.uniform(0.3, 3.0)
        total_equity = round(total_assets / (1 + de_ratio))
        total_liabilities = total_assets - total_equity

        cash = round(total_assets * random.uniform(0.03, 0.25))
        ar = round(revenue * random.uniform(0.04, 0.14))
        inv = round(cogs * random.uniform(0.02, 0.15))
        short_term_investments = round(total_assets * random.uniform(0.01, 0.15))
        cl = round(total_liabilities * random.uniform(0.25, 0.50))

        # Derive cash-flow and market/profile fields.
        net_income = round(revenue - cogs - opex * (1 - income_tax))
        dividends = round(max(0, net_income * random.uniform(0.0, 0.40)))
        capex = round(revenue * random.uniform(0.02, 0.10))
        
        shares = round(random.uniform(100, 15000))
        price = round(random.uniform(15, 800), 2)
        employees = round(random.uniform(5000, 500000))

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
    """Describe each CSV column in a small JSON-friendly record for tools and humans.

    The synthetic spreadsheet is **one row per (company, year)** with **flat column names**
    (e.g. ``revenue``, ``cash``), not wide ``revenue_2024``-style names. This function walks
    ``columns`` in display order and attaches metadata so consumers can tell what each
    column means without parsing the generator code.

    **Output shape:** ``schema[column_name]`` is a dict with at least ``type`` and usually
    ``desc``. Optional keys depend on the column kind:

    - **Categorical** (keys in ``CATEGORICAL_COLS``): ``type`` is ``categorical`` or
      ``ordinal``, ``desc`` is human text, ``year`` is always ``None``. Ordinal columns
      also get ``order`` (allowed rating ladder).
    - **``year``**: treated as numeric reporting year; ``base_name`` is ``"year"``,
      ``unit`` is ``"year"``, ``year`` field in schema is ``None`` (the *column* is the
      year dimension, not a year suffix).
    - **Other numerics** (keys in ``YEARLY_NUMERIC_COLS``): ``type`` is ``numeric``,
      ``base_name`` matches the column name (the metric id), ``unit`` comes from the
      generator spec (e.g. ``M_USD``, ``USD``), ``desc`` is a spaced label. ``year`` is
      set to the string ``"row_level"`` to mean: the year is **not** encoded in the
      column name; it lives in the separate ``year`` column on each row.

    Columns that appear in ``columns`` but are not in ``CATEGORICAL_COLS``, not
    ``year``, and not in ``YEARLY_NUMERIC_COLS`` are **skipped** (no entry). In normal
    runs every generated column should be covered by one of these branches.

    Args:
        columns: Ordered column names, typically ``list(rows[0].keys())`` from generated rows.

    Returns:
        Mapping from column name to metadata dict, suitable for ``json.dump`` to ``schema.json``.
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

    # Preserve stable column order from generated dictionaries.
    columns = list(rows[0].keys())

    if not rows:
        raise ValueError("No rows were generated from company templates.")

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
