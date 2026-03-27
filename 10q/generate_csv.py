"""
generate_csv.py
================
Generates the financial spreadsheet and schema from 10-Q reports.
Combines to_csv.py and generate_schema.py into one file.

Creates:
  financial_spreadsheet.csv  — one row per company, columns named field_PREFIX_Date
  schema.json                — column metadata (type, unit, base_name, period_tag)

Usage:
    python generate_csv.py
    python generate_csv.py --industry pharma --quarter 1 --year 2024 --n 20
"""

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from generate import generate_report, report_to_dict


# ---------------------------------------------------------------------------
# Field metadata — units and descriptions
# ---------------------------------------------------------------------------

FIELD_META = {
    # Identity
    "company_name":               {"type": "categorical", "unit": None,         "desc": "Company name"},
    "form_type":                  {"type": "categorical", "unit": None,         "desc": "SEC form type"},
    "industry":                   {"type": "categorical", "unit": None,         "desc": "Industry"},
    "fiscal_year":                {"type": "categorical", "unit": None,         "desc": "Fiscal year"},
    "quarter":                    {"type": "categorical", "unit": None,         "desc": "Quarter number"},
    "period":                     {"type": "categorical", "unit": None,         "desc": "Period label"},
    "fiscal_year_start":          {"type": "categorical", "unit": None,         "desc": "Fiscal year start date"},
    # Income statement
    "revenues":                   {"type": "numeric", "unit": "M_USD",    "desc": "Total revenues"},
    "membership_and_other_income":{"type": "numeric", "unit": "M_USD",    "desc": "Membership and other income"},
    "cost_of_sales":              {"type": "numeric", "unit": "M_USD",    "desc": "Cost of sales"},
    "rd_expense":                 {"type": "numeric", "unit": "M_USD",    "desc": "R&D expense"},
    "sga_expense":                {"type": "numeric", "unit": "M_USD",    "desc": "SG&A expense"},
    "total_costs":                {"type": "numeric", "unit": "M_USD",    "desc": "Total costs and expenses"},
    "gross_profit":               {"type": "numeric", "unit": "M_USD",    "desc": "Gross profit"},
    "operating_income":           {"type": "numeric", "unit": "M_USD",    "desc": "Operating income"},
    "other_income":               {"type": "numeric", "unit": "M_USD",    "desc": "Other income/(expense), net"},
    "pretax_income":              {"type": "numeric", "unit": "M_USD",    "desc": "Income before taxes"},
    "tax":                        {"type": "numeric", "unit": "M_USD",    "desc": "Provision for income taxes"},
    "net_income":                 {"type": "numeric", "unit": "M_USD",    "desc": "Net income"},
    "eps_basic":                  {"type": "numeric", "unit": "USD",      "desc": "EPS basic"},
    "eps_diluted":                {"type": "numeric", "unit": "USD",      "desc": "EPS diluted"},
    "shares_basic":               {"type": "numeric", "unit": "thousands","desc": "Shares basic"},
    "shares_diluted":             {"type": "numeric", "unit": "thousands","desc": "Shares diluted"},
    # Comprehensive income
    "ci_net_income":              {"type": "numeric", "unit": "M_USD",    "desc": "Net income (CI statement)"},
    "ci_fx_translation":          {"type": "numeric", "unit": "M_USD",    "desc": "FX translation adjustment"},
    "ci_unrealized_gains":        {"type": "numeric", "unit": "M_USD",    "desc": "Unrealized gains/losses on securities"},
    "ci_total_oci":               {"type": "numeric", "unit": "M_USD",    "desc": "Total other comprehensive income/(loss)"},
    "comprehensive_income":       {"type": "numeric", "unit": "M_USD",    "desc": "Total comprehensive income"},
    # Balance sheet
    "cash":                       {"type": "numeric", "unit": "M_USD",    "desc": "Cash and cash equivalents"},
    "st_investments":             {"type": "numeric", "unit": "M_USD",    "desc": "Short-term investments"},
    "accounts_receivable":        {"type": "numeric", "unit": "M_USD",    "desc": "Accounts receivable, net"},
    "inventories":                {"type": "numeric", "unit": "M_USD",    "desc": "Inventories"},
    "prepaid":                    {"type": "numeric", "unit": "M_USD",    "desc": "Prepaid expenses and other"},
    "current_assets":             {"type": "numeric", "unit": "M_USD",    "desc": "Total current assets"},
    "lt_investments":             {"type": "numeric", "unit": "M_USD",    "desc": "Long-term marketable securities"},
    "ppe":                        {"type": "numeric", "unit": "M_USD",    "desc": "PP&E, net"},
    "goodwill":                   {"type": "numeric", "unit": "M_USD",    "desc": "Goodwill"},
    "other_nca":                  {"type": "numeric", "unit": "M_USD",    "desc": "Other non-current assets"},
    "total_assets":               {"type": "numeric", "unit": "M_USD",    "desc": "Total assets"},
    "accounts_payable":           {"type": "numeric", "unit": "M_USD",    "desc": "Accounts payable"},
    "deferred_revenue":           {"type": "numeric", "unit": "M_USD",    "desc": "Deferred revenue"},
    "accrued_expenses":           {"type": "numeric", "unit": "M_USD",    "desc": "Accrued expenses and other"},
    "current_debt":               {"type": "numeric", "unit": "M_USD",    "desc": "Current portion of long-term debt"},
    "current_liabilities":        {"type": "numeric", "unit": "M_USD",    "desc": "Total current liabilities"},
    "long_term_debt":             {"type": "numeric", "unit": "M_USD",    "desc": "Long-term debt"},
    "other_ncl":                  {"type": "numeric", "unit": "M_USD",    "desc": "Other non-current liabilities"},
    "total_liabilities":          {"type": "numeric", "unit": "M_USD",    "desc": "Total liabilities"},
    "common_stock_apic":          {"type": "numeric", "unit": "M_USD",    "desc": "Common stock and APIC"},
    "retained_earnings":          {"type": "numeric", "unit": "M_USD",    "desc": "Retained earnings"},
    "aoci":                       {"type": "numeric", "unit": "M_USD",    "desc": "Accumulated other comprehensive income/(loss)"},
    "total_equity":               {"type": "numeric", "unit": "M_USD",    "desc": "Total shareholders equity"},
    # Cash flows
    "cf_net_income":              {"type": "numeric", "unit": "M_USD",    "desc": "Net income (cash flows)"},
    "cf_da":                      {"type": "numeric", "unit": "M_USD",    "desc": "Depreciation and amortization"},
    "cf_sbc":                     {"type": "numeric", "unit": "M_USD",    "desc": "Stock-based compensation"},
    "cf_operating":               {"type": "numeric", "unit": "M_USD",    "desc": "Net cash from operating activities"},
    "cf_capex":                   {"type": "numeric", "unit": "M_USD",    "desc": "Capital expenditures"},
    "cf_investing":               {"type": "numeric", "unit": "M_USD",    "desc": "Net cash from investing activities"},
    "cf_repurchases":             {"type": "numeric", "unit": "M_USD",    "desc": "Share repurchases"},
    "cf_dividends":               {"type": "numeric", "unit": "M_USD",    "desc": "Dividends paid"},
    "cf_financing":               {"type": "numeric", "unit": "M_USD",    "desc": "Net cash from financing activities"},
    "cf_net_change":              {"type": "numeric", "unit": "M_USD",    "desc": "Net change in cash"},
    "cf_closing_cash":            {"type": "numeric", "unit": "M_USD",    "desc": "Cash at end of period"},
}

IDENTITY_COLS = {"company_name", "form_type", "industry", "fiscal_year",
                 "quarter", "period", "fiscal_year_start"}


# ---------------------------------------------------------------------------
# Schema: parse a column name into metadata
# ---------------------------------------------------------------------------

def parse_column(col: str) -> dict:
    """Parse column name into schema metadata."""
    if col in IDENTITY_COLS:
        meta = FIELD_META.get(col, {"type": "categorical", "unit": None, "desc": col})
        return {"type": meta["type"], "unit": meta.get("unit"), "desc": meta["desc"],
                "base_name": col, "period_tag": None, "prefix": None, "date": None}

    # Q or YTD: field_Q_Mon_DD_YYYY or field_YTD_Mon_DD_YYYY
    m = re.match(r'^(.+?)_(Q|YTD)_([A-Z][a-z]{2}_\d{2}_\d{4})$', col)
    if m:
        field, prefix, date = m.group(1), m.group(2), m.group(3)
        meta = FIELD_META.get(field, {"type": "numeric", "unit": "M_USD",
                                      "desc": field.replace("_", " ")})
        return {"type": meta["type"], "unit": meta.get("unit"),
                "desc": f"{meta['desc']} ({prefix} ended {date.replace('_', ' ')})",
                "base_name": field, "period_tag": f"{prefix}_{date}",
                "prefix": prefix, "date": date}

    # Balance sheet: field_Mon_DD_YYYY
    m2 = re.match(r'^(.+?)_([A-Z][a-z]{2}_\d{2}_\d{4})$', col)
    if m2:
        field, date = m2.group(1), m2.group(2)
        meta = FIELD_META.get(field, {"type": "numeric", "unit": "M_USD",
                                      "desc": field.replace("_", " ")})
        return {"type": meta["type"], "unit": meta.get("unit"),
                "desc": f"{meta['desc']} (as of {date.replace('_', ' ')})",
                "base_name": field, "period_tag": f"BS_{date}",
                "prefix": "BS", "date": date}

    return {"type": "categorical", "unit": None, "desc": col,
            "base_name": col, "period_tag": None, "prefix": None, "date": None}


def build_schema(columns: list) -> dict:
    """Build schema dict from list of column names."""
    return {col: parse_column(col) for col in columns}


# ---------------------------------------------------------------------------
# Date formatting
# ---------------------------------------------------------------------------

def _fmt_date(period_str: str) -> str:
    """'Three Months Ended November 30, 2025' → 'Nov_30_2025'"""
    from datetime import datetime
    date_part = re.sub(
        r'^(Three|Six|Nine|Twelve)\s+Months\s+Ended\s+', "", period_str
    ).strip()
    try:
        return datetime.strptime(date_part, "%B %d, %Y").strftime("%b_%d_%Y")
    except ValueError:
        return date_part.replace(",", "").replace(" ", "_")


# ---------------------------------------------------------------------------
# Row flattener
# ---------------------------------------------------------------------------

def report_to_row(report, industry: str = "") -> dict:
    """Flatten a FinancialReport into a flat CSV row."""
    if not isinstance(report, dict):
        report = report_to_dict(report)

    fs  = report["financial_statements"]
    inc = fs["income_statement"]
    ci  = fs["comprehensive_income"]
    bs  = fs["balance_sheet"]
    cf  = fs["cash_flows"]

    def _date(ops_dict):
        if ops_dict is None: return None
        return _fmt_date(ops_dict.get("period", ""))

    cq_date  = _date(inc["current_quarter"])
    pq_date  = _date(inc["prior_year_quarter"])
    ytd_date = _date(inc["current_ytd"] or inc["current_quarter"])
    pyt_date = _date(inc["prior_year_ytd"] or inc["prior_year_quarter"])
    bs_date  = _fmt_date(bs["current"].get("as_of", "")) if bs["current"] else cq_date
    pfy_date = _fmt_date(bs["prior_fiscal_year_end"].get("as_of", "")) \
               if bs["prior_fiscal_year_end"] else None
    cf_date  = _date(cf["current_ytd"])
    cf_p_date= _date(cf["prior_ytd"])

    row = {
        "company_name":      report["company_name"],
        "form_type":         report["form_type"],
        "industry":          industry,
        "fiscal_year":       report["fiscal_year"],
        "quarter":           report["quarter"],
        "period":            report["period"].replace(",", ""),
        "fiscal_year_start": report["fiscal_year_start"].replace(",", ""),
    }

    def _ops(ops, date, prefix):
        if ops is None or date is None: return {}
        tag = f"{prefix}_{date}"
        rev = ops.get("revenue", {})

        # Revenue — total only, plus any named sub-lines
        result = {}
        result[f"revenues_{tag}"] = rev.get("total_revenues")
        for lbl, val in rev.get("other_components", {}).items():
            safe = lbl.lower().replace(" ", "_").replace("/", "_")
            result[f"{safe}_{tag}"] = val
        for lbl, val in rev.get("other_operating_revenue", {}).items():
            safe = lbl.lower().replace(" ", "_").replace("/", "_")
            result[f"{safe}_{tag}"] = val

        # Costs
        cos = ops.get("costs_and_expenses", {}).get("cost_of_sales")
        result[f"cost_of_sales_{tag}"]  = cos.get("total") if isinstance(cos, dict) else cos
        opex = ops.get("costs_and_expenses", {}).get("operating_expenses", {})
        result[f"rd_expense_{tag}"]     = opex.get("research_and_development")
        result[f"sga_expense_{tag}"]    = opex.get("selling_general_and_administrative")
        result[f"total_costs_{tag}"]    = ops.get("costs_and_expenses", {}).get("total_costs_and_expenses")

        # P&L
        result[f"gross_profit_{tag}"]    = ops.get("gross_profit")
        result[f"operating_income_{tag}"]= ops.get("operating_income")
        result[f"other_income_{tag}"]    = ops.get("other_income_expense_net")
        result[f"pretax_income_{tag}"]   = ops.get("income_before_taxes")
        result[f"tax_{tag}"]             = ops.get("provision_for_income_taxes")
        result[f"net_income_{tag}"]      = ops.get("net_income")

        # EPS & shares
        eps = ops.get("earnings_per_share") or {}
        result[f"eps_basic_{tag}"]    = eps.get("basic")
        result[f"eps_diluted_{tag}"]  = eps.get("diluted")
        sh = ops.get("shares_used_in_computing_eps") or {}
        result[f"shares_basic_{tag}"]   = sh.get("basic")
        result[f"shares_diluted_{tag}"] = sh.get("diluted")

        return {k: v for k, v in result.items() if v is not None}

    def _ci(ci_dict, date, prefix):
        if ci_dict is None or date is None: return {}
        tag = f"{prefix}_{date}"
        oci = ci_dict.get("other_comprehensive_income") or {}
        result = {
            f"ci_net_income_{tag}":       ci_dict.get("net_income"),
            f"ci_fx_translation_{tag}":   oci.get("foreign_currency_translation"),
            f"ci_unrealized_gains_{tag}": oci.get("unrealized_gains_losses_on_securities"),
            f"ci_total_oci_{tag}":        oci.get("total") if isinstance(oci, dict) else None,
            f"comprehensive_income_{tag}":ci_dict.get("total_comprehensive_income"),
        }
        return {k: v for k, v in result.items() if v is not None}

    def _bs(bs_dict, date):
        if bs_dict is None or date is None: return {}
        tag = date
        ca  = bs_dict.get("current_assets") or {}
        nca = bs_dict.get("non_current_assets") or {}
        cl  = bs_dict.get("current_liabilities") or {}
        ncl = bs_dict.get("non_current_liabilities") or {}
        eq  = bs_dict.get("shareholders_equity") or {}
        result = {
            f"cash_{tag}":                ca.get("cash_and_cash_equivalents"),
            f"st_investments_{tag}":      ca.get("short_term_investments"),
            f"accounts_receivable_{tag}": ca.get("accounts_receivable_net"),
            f"inventories_{tag}":         ca.get("inventories"),
            f"prepaid_{tag}":             ca.get("prepaid_expenses_and_other"),
            f"current_assets_{tag}":      ca.get("total"),
            f"lt_investments_{tag}":      nca.get("long_term_marketable_securities"),
            f"ppe_{tag}":                 nca.get("property_plant_and_equipment_net"),
            f"goodwill_{tag}":            nca.get("goodwill"),
            f"other_nca_{tag}":           nca.get("other_non_current_assets"),
            f"total_assets_{tag}":        bs_dict.get("total_assets"),
            f"accounts_payable_{tag}":    cl.get("accounts_payable"),
            f"deferred_revenue_{tag}":    cl.get("deferred_revenue"),
            f"accrued_expenses_{tag}":    cl.get("accrued_expenses_and_other"),
            f"current_debt_{tag}":        cl.get("current_portion_of_long_term_debt"),
            f"current_liabilities_{tag}": cl.get("total"),
            f"long_term_debt_{tag}":      ncl.get("long_term_debt"),
            f"other_ncl_{tag}":           ncl.get("other_non_current_liabilities"),
            f"total_liabilities_{tag}":   bs_dict.get("total_liabilities"),
            f"common_stock_apic_{tag}":   eq.get("common_stock_and_apic"),
            f"retained_earnings_{tag}":   eq.get("retained_earnings"),
            f"aoci_{tag}":                eq.get("accumulated_other_comprehensive_loss"),
            f"total_equity_{tag}":        eq.get("total"),
        }
        return {k: v for k, v in result.items() if v is not None}

    def _cf(cf_dict, date):
        if cf_dict is None or date is None: return {}
        tag = f"YTD_{date}"
        oa = cf_dict.get("operating_activities") or {}
        ia = cf_dict.get("investing_activities") or {}
        fa = cf_dict.get("financing_activities") or {}
        result = {
            f"cf_net_income_{tag}":   oa.get("net_income"),
            f"cf_da_{tag}":           oa.get("depreciation_and_amortization"),
            f"cf_sbc_{tag}":          oa.get("stock_based_compensation"),
            f"cf_operating_{tag}":    oa.get("total"),
            f"cf_capex_{tag}":        ia.get("capital_expenditures"),
            f"cf_investing_{tag}":    ia.get("total"),
            f"cf_repurchases_{tag}":  fa.get("share_repurchases"),
            f"cf_dividends_{tag}":    fa.get("dividends_paid"),
            f"cf_financing_{tag}":    fa.get("total"),
            f"cf_net_change_{tag}":   cf_dict.get("net_change_in_cash"),
            f"cf_closing_cash_{tag}": cf_dict.get("closing_cash"),
        }
        return {k: v for k, v in result.items() if v is not None}

    # Income statement — 4 columns
    row.update(_ops(inc["current_quarter"],    cq_date,  "Q"))
    row.update(_ops(inc["prior_year_quarter"], pq_date,  "Q"))
    if inc["current_ytd"]:
        row.update(_ops(inc["current_ytd"],    ytd_date, "YTD"))
    if inc["prior_year_ytd"]:
        row.update(_ops(inc["prior_year_ytd"], pyt_date, "YTD"))

    # Comprehensive income — 4 columns
    row.update(_ci(ci["current_quarter"],    cq_date,  "Q"))
    row.update(_ci(ci["prior_year_quarter"], pq_date,  "Q"))
    if ci["current_ytd"]:
        row.update(_ci(ci["current_ytd"],    ytd_date, "YTD"))
    if ci["prior_year_ytd"]:
        row.update(_ci(ci["prior_year_ytd"], pyt_date, "YTD"))

    # Balance sheet — 2 columns
    row.update(_bs(bs["current"],            bs_date))
    if pfy_date:
        row.update(_bs(bs["prior_fiscal_year_end"], pfy_date))

    # Cash flows — 2 columns
    row.update(_cf(cf["current_ytd"], cf_date))
    row.update(_cf(cf["prior_ytd"],   cf_p_date))

    return row


# ---------------------------------------------------------------------------
# Generate dataset
# ---------------------------------------------------------------------------

def generate_csv(
    n_companies: int = 15,
    industry: str = "retail",
    quarter: int = 3,
    filing_year: int = 2025,
    base_seed: int = 42,
    csv_path: str = "financial_spreadsheet.csv",
    schema_path: str = "schema.json",
) -> tuple:
    """
    Generate n_companies 10-Q reports (all same industry/quarter/year)
    and save to CSV + schema.json.

    Returns (csv_path, schema_path).
    """
    # Generate reports
    rows = []
    for i in range(n_companies):
        r = generate_report(
            industry=industry,
            quarter=quarter,
            filing_year=filing_year,
            seed=base_seed + i,
        )
        rows.append(report_to_row(r, industry=industry))

    # Collect all columns in order
    all_cols = []
    seen = set()
    for row in rows:
        for col in row:
            if col not in seen:
                all_cols.append(col)
                seen.add(col)

    # Save CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in all_cols})

    # Save schema
    schema = build_schema(all_cols)
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return csv_path, schema_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--industry", default="retail")
    parser.add_argument("--quarter",  type=int, default=3)
    parser.add_argument("--year",     type=int, default=2025)
    parser.add_argument("--n",        type=int, default=15)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--csv",      default="financial_spreadsheet.csv")
    parser.add_argument("--schema",   default="schema.json")
    args = parser.parse_args()

    print(f"Generating {args.n} companies: {args.industry} Q{args.quarter} FY{args.year}...")
    csv_path, schema_path = generate_csv(
        n_companies=args.n,
        industry=args.industry,
        quarter=args.quarter,
        filing_year=args.year,
        base_seed=args.seed,
        csv_path=args.csv,
        schema_path=args.schema,
    )

    # Check for empty columns
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = reader.fieldnames

    empty = [c for c in cols if any(rows[i].get(c, "") == "" for i in range(len(rows)))]
    print(f"Saved: {csv_path}  ({len(rows)} rows × {len(cols)} columns)")
    print(f"Saved: {schema_path}")
    print(f"Empty columns: {len(empty)}" + (f" — {empty[:5]}" if empty else " ✓"))