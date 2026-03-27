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
import os

random.seed(42)

# ── Define the spreadsheet schema ───────────────────────────
# Each column has a name, data type, and how to generate it.
# This is exactly what you'd see in a real financial Excel file.

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
    "non_operating":          {"unit": "M_USD",   "range": (-1000, 1000)},
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


def generate_company_row(company_tuple):
    """Generate one row of the spreadsheet for one company."""
    name, ticker, sector, country, exchange = company_tuple

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

    rows = []
    for year in YEARS:
        row = row_prefix.copy()

        # Revenue grows/shrinks slightly each year
        growth = random.uniform(-0.08, 0.20)
        revenue = round(base_revenue * (1 + growth))
        base_revenue = revenue  # next year starts from here

        # Derive other values from revenue (realistic ratios)
        cogs_pct = random.uniform(0.30, 0.75)
        cogs = round(revenue * cogs_pct)

        opex = round(revenue * random.uniform(0.05, 0.25))
        nonopex = round(revenue * random.uniform(0.05, 0.25))
        income_tax = random.uniform(0.10, 0.30)

        # Balance sheet
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

        net_income = round(revenue - cogs - opex * (1 - income_tax))
        dividends = round(max(0, net_income * random.uniform(0.0, 0.40)))
        capex = round(revenue * random.uniform(0.02, 0.10))
        
        shares = round(random.uniform(100, 15000))
        price = round(random.uniform(15, 800), 2)
        employees = round(random.uniform(5000, 500000))

        # Store with year suffix: revenue_2022, revenue_2023, etc.
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


def main():
    # Generate all rows
    rows = []
    for c in COMPANIES:
        rows.extend(generate_company_row(c))

    # Get column order
    columns = list(rows[0].keys())

    # ── Save CSV ──
    csv_path = "output/financial_spreadsheet.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # ── Save schema (for the tree builder to know types) ──
    schema = {}
    for col in columns:
        if col in CATEGORICAL_COLS:
            info = CATEGORICAL_COLS[col]
            schema[col] = {
                "type": info["type"],
                "desc": info["desc"],
                "year": None,
            }
            if info["type"] == "ordinal":
                schema[col]["order"] = info["order"]
        else:
            # Parse: "revenue_2024" → base="revenue", year=2024
            parts = col.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]
                year = int(parts[1])
                unit = YEARLY_NUMERIC_COLS.get(base_name, {}).get("unit", "M_USD")
                schema[col] = {
                    "type": "numeric",
                    "base_name": base_name,
                    "year": year,
                    "unit": unit,
                    "desc": f"{base_name.replace('_', ' ')} in {year}",
                }

    with open("output/schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    # ── Print preview ──
    print(f"✓ Generated spreadsheet: {len(rows)} companies × {len(columns)} columns\n")
    print(f"  Categorical columns: {len(CATEGORICAL_COLS)}")
    print(f"  Numeric columns:     {len(columns) - len(CATEGORICAL_COLS)} ({len(YEARLY_NUMERIC_COLS)} metrics × {len(YEARS)} years)\n")

    # Show a few rows
    print(f"  Preview (first 3 companies):")
    for row in rows[:3]:
        print(f"    {row['ticker']:>5} | {row['company_name']:<30} | {row['sector']:<15} | "
              f"Rev 2024: ${row.get('revenue_2024', 0):>10,}M | NI 2024: ${row.get('net_income_2024', 0):>10,}M")

    print(f"\n  All columns:")
    for i, col in enumerate(columns):
        info = schema.get(col, {})
        ctype = info.get("type", "?")
        print(f"    {i:>2}. {col:<35} [{ctype}]")

    print(f"\n✓ Saved: {csv_path}")
    print(f"✓ Saved: output/schema.json")


if __name__ == "__main__":
    main()
