"""
Generate Financial Spreadsheet + Per-Company Financial Sheets
==============================================================
Creates:
  1. financial_spreadsheet.csv          — Full dataset (all companies, all years)
  2. financial_spreadsheet.json         — Same data as JSON
  3. schema.json                        — Column type metadata
  4. company_index.json                 — Lookup: company_name → company_id, file paths
  5. companies/<company_id>.json        — Per-company structured financial data
  6. companies/<company_id>_sheet.txt   — Per-company human-readable financial sheet
                                          (this is what you feed to the LLM as context)
"""

import csv
import json
import random
import os
import textwrap

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

random.seed(42)

# ── Schema ──────────────────────────────────────────────────
YEARS = [2022, 2023, 2024]

CATEGORICAL_COLS = {
    "company_name":  {"type": "categorical", "desc": "Company name"},
    "ticker":        {"type": "categorical", "desc": "Stock ticker symbol"},
    "sector":        {"type": "categorical", "desc": "Industry sector"},
    "country":       {"type": "categorical", "desc": "Country of headquarters"},
    "exchange":      {"type": "categorical", "desc": "Stock exchange"},
    "credit_rating": {"type": "ordinal",     "desc": "Credit rating",
                      "order": ["CCC", "B", "BB", "BBB", "A", "AA", "AAA"]},
}

YEARLY_NUMERIC_COLS = {
    "revenue":              {"unit": "M_USD",   "range": (5000, 500000)},
    "cost_of_goods_sold":   {"unit": "M_USD",   "range": (2000, 350000)},
    "gross_profit":         {"unit": "M_USD",   "range": (1000, 200000)},
    "operating_expenses":   {"unit": "M_USD",   "range": (500, 80000)},
    "operating_income":     {"unit": "M_USD",   "range": (500, 120000)},
    "net_income":           {"unit": "M_USD",   "range": (200, 80000)},
    "total_assets":         {"unit": "M_USD",   "range": (10000, 1000000)},
    "total_liabilities":    {"unit": "M_USD",   "range": (5000, 700000)},
    "total_equity":         {"unit": "M_USD",   "range": (3000, 400000)},
    "current_assets":       {"unit": "M_USD",   "range": (3000, 300000)},
    "current_liabilities":  {"unit": "M_USD",   "range": (2000, 200000)},
    "cash":                 {"unit": "M_USD",   "range": (1000, 150000)},
    "long_term_debt":       {"unit": "M_USD",   "range": (1000, 300000)},
    "accounts_receivable":  {"unit": "M_USD",   "range": (500, 80000)},
    "inventories":          {"unit": "M_USD",   "range": (200, 50000)},
    "capex":                {"unit": "M_USD",   "range": (500, 60000)},
    "dividends_paid":       {"unit": "M_USD",   "range": (0, 20000)},
    "shares_outstanding":   {"unit": "millions","range": (100, 15000)},
    "stock_price":          {"unit": "USD",     "range": (15, 800)},
    "employees":            {"unit": "count",   "range": (5000, 500000)},
}

COMPANIES = [
    ("Apex Dynamics Corp",       "APEX", "Technology",     "USA",     "NYSE"),
    ("Meridian Systems Inc",     "MSYS", "Technology",     "USA",     "NASDAQ"),
    ("Quantum Bridge Holdings",  "QBHD", "Healthcare",    "USA",     "NYSE"),
    ("NovaStar Technologies",    "NVST", "Technology",     "Germany", "XETRA"),
    ("Atlas Global Industries",  "ATGL", "Industrials",   "UK",      "LSE"),
    ("Pinnacle Health Group",    "PNHG", "Healthcare",    "USA",     "NYSE"),
    ("Ironclad Manufacturing",   "IRCL", "Industrials",   "Japan",   "TSE"),
    ("SilverLine Retail Inc",    "SLRI", "Consumer",      "USA",     "NASDAQ"),
    ("Pacific Crest Energy",     "PCEN", "Energy",        "Canada",  "TSX"),
    ("Orion Financial Group",    "ORFG", "Financials",    "UK",      "LSE"),
    ("Titan Consumer Brands",    "TICB", "Consumer",      "France",  "Euronext"),
    ("Cobalt Semiconductor",     "CBSM", "Technology",    "Taiwan",  "TWSE"),
    ("Helix BioSciences",        "HLXB", "Healthcare",    "USA",     "NASDAQ"),
    ("ClearPath Logistics",      "CPTH", "Industrials",   "USA",     "NYSE"),
    ("Crimson Pharmaceuticals",  "CRPH", "Healthcare",    "Switzerland","SIX"),
]

RATINGS = ["CCC", "B", "BB", "BBB", "A", "AA", "AAA"]


def generate_company_row(company_tuple):
    """Generate one row of the spreadsheet for one company."""
    name, ticker, sector, country, exchange = company_tuple

    row = {
        "company_name": name,
        "ticker": ticker,
        "sector": sector,
        "country": country,
        "exchange": exchange,
        "credit_rating": random.choice(RATINGS),
    }

    base_revenue = random.uniform(*YEARLY_NUMERIC_COLS["revenue"]["range"])

    for year in YEARS:
        growth = random.uniform(-0.08, 0.20)
        revenue = round(base_revenue * (1 + growth))
        base_revenue = revenue

        cogs_pct = random.uniform(0.30, 0.75)
        cogs = round(revenue * cogs_pct)
        gross_profit = revenue - cogs

        opex = round(revenue * random.uniform(0.05, 0.25))
        operating_income = gross_profit - opex
        tax_rate = random.uniform(0.10, 0.30)
        net_income = round(operating_income * (1 - tax_rate))

        asset_turnover = random.uniform(0.3, 1.5)
        total_assets = round(revenue / asset_turnover)
        de_ratio = random.uniform(0.3, 3.0)
        total_equity = round(total_assets / (1 + de_ratio))
        total_liabilities = total_assets - total_equity

        current_ratio = random.uniform(0.8, 3.0)
        cl = round(total_liabilities * random.uniform(0.25, 0.50))
        ca = round(cl * current_ratio)

        cash = round(total_assets * random.uniform(0.03, 0.25))
        ar = round(revenue * random.uniform(0.04, 0.14))
        inv = round(cogs * random.uniform(0.02, 0.15))
        lt_debt = round(total_liabilities * random.uniform(0.30, 0.70))

        capex = round(revenue * random.uniform(0.02, 0.10))
        dividends = round(max(0, net_income * random.uniform(0.0, 0.40)))
        shares = round(random.uniform(100, 15000))
        price = round(random.uniform(15, 800), 2)
        employees = round(random.uniform(5000, 500000))

        row[f"revenue_{year}"] = revenue
        row[f"cost_of_goods_sold_{year}"] = cogs
        row[f"gross_profit_{year}"] = gross_profit
        row[f"operating_expenses_{year}"] = opex
        row[f"operating_income_{year}"] = operating_income
        row[f"net_income_{year}"] = net_income
        row[f"total_assets_{year}"] = total_assets
        row[f"total_liabilities_{year}"] = total_liabilities
        row[f"total_equity_{year}"] = total_equity
        row[f"current_assets_{year}"] = ca
        row[f"current_liabilities_{year}"] = cl
        row[f"cash_{year}"] = cash
        row[f"long_term_debt_{year}"] = lt_debt
        row[f"accounts_receivable_{year}"] = ar
        row[f"inventories_{year}"] = inv
        row[f"capex_{year}"] = capex
        row[f"dividends_paid_{year}"] = dividends
        row[f"shares_outstanding_{year}"] = shares
        row[f"stock_price_{year}"] = price
        row[f"employees_{year}"] = employees

    return row


def build_company_structured(company_id, row):
    """
    Build a structured per-company JSON with financial data
    organized by statement type and year — easy for LLM to parse.
    """
    profile = {
        "company_id": company_id,
        "company_name": row["company_name"],
        "ticker": row["ticker"],
        "sector": row["sector"],
        "country": row["country"],
        "exchange": row["exchange"],
        "credit_rating": row["credit_rating"],
    }

    financials = {}
    for year in YEARS:
        y = str(year)
        financials[y] = {
            "income_statement": {
                "revenue": row[f"revenue_{year}"],
                "cost_of_goods_sold": row[f"cost_of_goods_sold_{year}"],
                "gross_profit": row[f"gross_profit_{year}"],
                "operating_expenses": row[f"operating_expenses_{year}"],
                "operating_income": row[f"operating_income_{year}"],
                "net_income": row[f"net_income_{year}"],
            },
            "balance_sheet": {
                "total_assets": row[f"total_assets_{year}"],
                "total_liabilities": row[f"total_liabilities_{year}"],
                "total_equity": row[f"total_equity_{year}"],
                "current_assets": row[f"current_assets_{year}"],
                "current_liabilities": row[f"current_liabilities_{year}"],
                "cash": row[f"cash_{year}"],
                "long_term_debt": row[f"long_term_debt_{year}"],
                "accounts_receivable": row[f"accounts_receivable_{year}"],
                "inventories": row[f"inventories_{year}"],
            },
            "cash_flow_and_other": {
                "capex": row[f"capex_{year}"],
                "dividends_paid": row[f"dividends_paid_{year}"],
            },
            "market_data": {
                "shares_outstanding_millions": row[f"shares_outstanding_{year}"],
                "stock_price_usd": row[f"stock_price_{year}"],
                "employees": row[f"employees_{year}"],
            },
        }

    return {"profile": profile, "financials": financials}


def build_human_readable_sheet(company_data):
    """
    Build a human-readable text 'financial sheet' — this is what you
    feed into the LLM prompt as context for answering questions.
    """
    p = company_data["profile"]
    f = company_data["financials"]

    lines = []
    lines.append("=" * 72)
    lines.append(f"  FINANCIAL REPORT: {p['company_name']}")
    lines.append(f"  Ticker: {p['ticker']}  |  Sector: {p['sector']}  |  "
                 f"Country: {p['country']}  |  Exchange: {p['exchange']}")
    lines.append(f"  Credit Rating: {p['credit_rating']}")
    lines.append("=" * 72)
    lines.append("")

    # ── Income Statement ──
    lines.append("-" * 72)
    lines.append("  INCOME STATEMENT (in millions USD)")
    lines.append("-" * 72)
    header = f"  {'Metric':<30} " + "  ".join(f"{'FY' + y:>12}" for y in sorted(f.keys()))
    lines.append(header)
    lines.append("  " + "-" * 68)

    is_keys = ["revenue", "cost_of_goods_sold", "gross_profit",
               "operating_expenses", "operating_income", "net_income"]
    for key in is_keys:
        label = key.replace("_", " ").title()
        vals = "  ".join(f"{f[y]['income_statement'][key]:>12,}" for y in sorted(f.keys()))
        lines.append(f"  {label:<30} {vals}")
    lines.append("")

    # ── Balance Sheet ──
    lines.append("-" * 72)
    lines.append("  BALANCE SHEET (in millions USD)")
    lines.append("-" * 72)
    lines.append(header)
    lines.append("  " + "-" * 68)

    bs_keys = ["total_assets", "total_liabilities", "total_equity",
               "current_assets", "current_liabilities", "cash",
               "long_term_debt", "accounts_receivable", "inventories"]
    for key in bs_keys:
        label = key.replace("_", " ").title()
        vals = "  ".join(f"{f[y]['balance_sheet'][key]:>12,}" for y in sorted(f.keys()))
        lines.append(f"  {label:<30} {vals}")
    lines.append("")

    # ── Cash Flow & Other ──
    lines.append("-" * 72)
    lines.append("  CASH FLOW & OTHER DATA")
    lines.append("-" * 72)
    lines.append(header)
    lines.append("  " + "-" * 68)

    cf_keys = ["capex", "dividends_paid"]
    for key in cf_keys:
        label = key.replace("_", " ").title()
        vals = "  ".join(f"{f[y]['cash_flow_and_other'][key]:>12,}" for y in sorted(f.keys()))
        lines.append(f"  {label:<30} {vals}")

    md_display = [
        ("shares_outstanding_millions", "Shares Outstanding (M)"),
        ("stock_price_usd", "Stock Price (USD)"),
        ("employees", "Employees"),
    ]
    for key, label in md_display:
        vals = "  ".join(f"{f[y]['market_data'][key]:>12,}" for y in sorted(f.keys()))
        lines.append(f"  {label:<30} {vals}")
    lines.append("")
    lines.append("=" * 72)

    return "\n".join(lines)


def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "companies"), exist_ok=True)

    rows = [generate_company_row(c) for c in COMPANIES]
    columns = list(rows[0].keys())

    # ── 1. Save master CSV ──
    csv_path = os.path.join(OUTPUT_DIR, "financial_spreadsheet.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # ── 2. Save master JSON ──
    json_path = os.path.join(OUTPUT_DIR, "financial_spreadsheet.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # ── 3. Save schema ──
    schema = {}
    for col in columns:
        if col in CATEGORICAL_COLS:
            info = CATEGORICAL_COLS[col]
            schema[col] = {"type": info["type"], "desc": info["desc"], "year": None}
            if info["type"] == "ordinal":
                schema[col]["order"] = info["order"]
        else:
            parts = col.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_name = parts[0]
                year = int(parts[1])
                unit = YEARLY_NUMERIC_COLS.get(base_name, {}).get("unit", "M_USD")
                schema[col] = {
                    "type": "numeric", "base_name": base_name,
                    "year": year, "unit": unit,
                    "desc": f"{base_name.replace('_', ' ')} in {year}",
                }
    with open(os.path.join(OUTPUT_DIR, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    # ── 4. Generate per-company files + index ──
    company_index = {}

    for i, row in enumerate(rows):
        company_id = f"COMP-{i+1:04d}"
        company_name = row["company_name"]
        ticker = row["ticker"]

        # Build structured data
        company_data = build_company_structured(company_id, row)

        # Save per-company JSON
        json_file = f"companies/{company_id}.json"
        with open(os.path.join(OUTPUT_DIR, json_file), "w") as f:
            json.dump(company_data, f, indent=2)

        # Save per-company human-readable sheet (LLM context)
        txt_file = f"companies/{company_id}_sheet.txt"
        sheet_text = build_human_readable_sheet(company_data)
        with open(os.path.join(OUTPUT_DIR, txt_file), "w") as f:
            f.write(sheet_text)

        # Add to index (supports lookup by name, ticker, or ID)
        company_index[company_name] = {
            "company_id": company_id,
            "ticker": ticker,
            "sector": row["sector"],
            "json_file": json_file,
            "sheet_file": txt_file,
        }

    # Also add ticker-based lookup
    index_with_aliases = {
        "by_name": company_index,
        "by_ticker": {v["ticker"]: {"company_name": k, **v} for k, v in company_index.items()},
        "by_id": {v["company_id"]: {"company_name": k, **v} for k, v in company_index.items()},
    }

    with open(os.path.join(OUTPUT_DIR, "company_index.json"), "w") as f:
        json.dump(index_with_aliases, f, indent=2)

    # ── Print summary ──
    print(f"✓ Generated {len(rows)} companies × {len(columns)} columns\n")
    print(f"Output files:")
    print(f"  ../output/financial_spreadsheet.csv   — Full flat dataset")
    print(f"  ../output/financial_spreadsheet.json   — Full dataset as JSON")
    print(f"  ../output/schema.json                  — Column metadata")
    print(f"  ../output/company_index.json           — Lookup index (by name/ticker/id)")
    print(f"  ../output/companies/COMP-XXXX.json     — Per-company structured data (×{len(rows)})")
    print(f"  ../output/companies/COMP-XXXX_sheet.txt— Per-company readable sheet (×{len(rows)})")
    print()

    # Show example
    print("Example: Meridian Systems Inc")
    print("-" * 40)
    msys = [r for r in rows if r["company_name"] == "Meridian Systems Inc"][0]
    msys_id = company_index["Meridian Systems Inc"]["company_id"]
    print(f"  ID: {msys_id}")
    print(f"  inventories_2024 = {msys['inventories_2024']}")
    print(f"  Files: companies/{msys_id}.json, companies/{msys_id}_sheet.txt")
    print()

    # Show what the LLM would see
    print("=" * 72)
    print("PREVIEW: What the LLM sees as context (first company)")
    print("=" * 72)
    first_id = company_index[rows[0]["company_name"]]["company_id"]
    with open(os.path.join(OUTPUT_DIR, "companies", f"{first_id}_sheet.txt")) as f:
        print(f.read())


if __name__ == "__main__":
    main()