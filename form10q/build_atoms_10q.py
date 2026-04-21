"""
build_atoms_10q.py
==================
Drop-in replacement for compiler_pipeline/2.fixed_building_atoms.py,
adapted for the 10-Q spreadsheet schema.

Column naming convention:
  field_Q_Mon_DD_YYYY    — quarterly income/comprehensive income
  field_YTD_Mon_DD_YYYY  — year-to-date income/cash flow
  field_Mon_DD_YYYY      — balance sheet (no prefix)

Concept naming in atoms:
  Q columns   → concept = "{field}_q"
  YTD columns → concept = "{field}_ytd"
  BS columns  → concept = "{field}"

This keeps Q and YTD as distinct concepts so the tree sampler never
mixes them in a growth operation (Q vs YTD would be meaningless).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Human-readable labels (mirrors rewrite_10q.LABELS)
# ---------------------------------------------------------------------------

LABELS: Dict[str, str] = {
    "net_sales":                   "net sales",
    "revenues":                    "total revenues",
    "membership_and_other_income": "membership and other income",
    "cost_of_sales":               "cost of sales",
    "rd_expense":                  "research and development expense",
    "sga_expense":                 "selling general and administrative expense",
    "total_costs":                 "total costs and expenses",
    "gross_profit":                "gross profit",
    "operating_income":            "operating income",
    "other_income":                "other income/(expense), net",
    "pretax_income":               "income before taxes",
    "tax":                         "provision for income taxes",
    "net_income":                  "net income",
    "eps_basic":                   "basic earnings per share",
    "eps_diluted":                 "diluted earnings per share",
    "shares_basic":                "basic shares used in computing EPS",
    "shares_diluted":              "diluted shares used in computing EPS",
    "ci_net_income":               "net income (CI)",
    "ci_fx_translation":           "change in foreign currency translation",
    "ci_unrealized_gains":         "change in unrealized gains/losses on securities",
    "ci_total_oci":                "total other comprehensive income/(loss)",
    "comprehensive_income":        "total comprehensive income",
    "cash":                        "cash and cash equivalents",
    "st_investments":              "short-term investments",
    "accounts_receivable":         "accounts receivable, net",
    "inventories":                 "inventories",
    "prepaid":                     "prepaid expenses and other",
    "current_assets":              "total current assets",
    "lt_investments":              "long-term marketable securities",
    "ppe":                         "property, plant and equipment, net",
    "goodwill":                    "goodwill",
    "other_nca":                   "other non-current assets",
    "total_assets":                "total assets",
    "accounts_payable":            "accounts payable",
    "deferred_revenue":            "deferred revenue",
    "accrued_expenses":            "accrued expenses and other",
    "current_debt":                "current portion of long-term debt",
    "current_liabilities":         "total current liabilities",
    "long_term_debt":              "long-term debt",
    "other_ncl":                   "other non-current liabilities",
    "total_liabilities":           "total liabilities",
    "common_stock_apic":           "common stock and additional paid-in capital",
    "retained_earnings":           "retained earnings",
    "aoci":                        "accumulated other comprehensive income/(loss)",
    "total_equity":                "total shareholders equity",
    "cf_net_income":               "net income (cash flow)",
    "cf_da":                       "depreciation and amortization",
    "cf_sbc":                      "stock-based compensation",
    "cf_operating":                "net cash from operating activities",
    "cf_capex":                    "capital expenditures",
    "cf_investing":                "net cash from investing activities",
    "cf_repurchases":              "share repurchases",
    "cf_dividends":                "dividends paid",
    "cf_financing":                "net cash from financing activities",
    "cf_fx_effect":                "cash effect of exchange rate changes",
    "cf_net_change":               "net change in cash",
    "cf_closing_cash":             "cash at end of period",
}

# ---------------------------------------------------------------------------
# Concept metadata: semantic_type, unit, parent_concept, role
# ---------------------------------------------------------------------------

CONCEPT_METADATA: Dict[str, Dict[str, Any]] = {
    # --- Income statement ---
    "net_sales":                   {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "component"},
    "revenues":                    {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "total"},
    "membership_and_other_income": {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "component"},
    "cost_of_sales":               {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "component"},
    "rd_expense":                  {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "component"},
    "sga_expense":                 {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "component"},
    "total_costs":                 {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "total"},
    "gross_profit":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "total"},
    "operating_income":            {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "total"},
    "other_income":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "component"},
    "pretax_income":               {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "total"},
    "tax":                         {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "component"},
    "net_income":                  {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": "total"},
    "eps_basic":                   {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": None},
    "eps_diluted":                 {"semantic_type": "amount", "unit": "USD", "parent_concept": "income_statement", "role": None},
    "shares_basic":                {"semantic_type": "count",  "unit": "shares", "parent_concept": "income_statement", "role": None},
    "shares_diluted":              {"semantic_type": "count",  "unit": "shares", "parent_concept": "income_statement", "role": None},
    # --- Comprehensive income ---
    "ci_net_income":               {"semantic_type": "amount", "unit": "USD", "parent_concept": "comprehensive_income", "role": "component"},
    "ci_fx_translation":           {"semantic_type": "amount", "unit": "USD", "parent_concept": "comprehensive_income", "role": "component"},
    "ci_unrealized_gains":         {"semantic_type": "amount", "unit": "USD", "parent_concept": "comprehensive_income", "role": "component"},
    "ci_total_oci":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "comprehensive_income", "role": "total"},
    "comprehensive_income":        {"semantic_type": "amount", "unit": "USD", "parent_concept": "comprehensive_income", "role": "total"},
    # --- Balance sheet: assets ---
    "cash":                        {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets",  "role": "component"},
    "st_investments":              {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets",  "role": "component"},
    "accounts_receivable":         {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets",  "role": "component"},
    "inventories":                 {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets",  "role": "component"},
    "prepaid":                     {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets",  "role": "component"},
    "current_assets":              {"semantic_type": "amount", "unit": "USD", "parent_concept": "balance_sheet",   "role": "total"},
    "lt_investments":              {"semantic_type": "amount", "unit": "USD", "parent_concept": "long_term_assets","role": "component"},
    "ppe":                         {"semantic_type": "amount", "unit": "USD", "parent_concept": "long_term_assets","role": "component"},
    "goodwill":                    {"semantic_type": "amount", "unit": "USD", "parent_concept": "long_term_assets","role": "component"},
    "other_nca":                   {"semantic_type": "amount", "unit": "USD", "parent_concept": "long_term_assets","role": "component"},
    "total_assets":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "balance_sheet",   "role": "total"},
    # --- Balance sheet: liabilities ---
    "accounts_payable":            {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_liabilities",  "role": "component"},
    "deferred_revenue":            {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_liabilities",  "role": "component"},
    "accrued_expenses":            {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_liabilities",  "role": "component"},
    "current_debt":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_liabilities",  "role": "component"},
    "current_liabilities":         {"semantic_type": "amount", "unit": "USD", "parent_concept": "balance_sheet",        "role": "total"},
    "long_term_debt":              {"semantic_type": "amount", "unit": "USD", "parent_concept": "long_term_liabilities", "role": "component"},
    "other_ncl":                   {"semantic_type": "amount", "unit": "USD", "parent_concept": "long_term_liabilities", "role": "component"},
    "total_liabilities":           {"semantic_type": "amount", "unit": "USD", "parent_concept": "balance_sheet",        "role": "total"},
    # --- Balance sheet: equity ---
    "common_stock_apic":           {"semantic_type": "amount", "unit": "USD", "parent_concept": "equity", "role": "component"},
    "retained_earnings":           {"semantic_type": "amount", "unit": "USD", "parent_concept": "equity", "role": "component"},
    "aoci":                        {"semantic_type": "amount", "unit": "USD", "parent_concept": "equity", "role": "component"},
    "total_equity":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "equity", "role": "total"},
    # --- Cash flow (YTD only in the data) ---
    "cf_net_income":               {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "component"},
    "cf_da":                       {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "component"},
    "cf_sbc":                      {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "component"},
    "cf_operating":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "total"},
    "cf_capex":                    {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "component"},
    "cf_investing":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "total"},
    "cf_repurchases":              {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "component"},
    "cf_dividends":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "component"},
    "cf_financing":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "total"},
    "cf_fx_effect":                {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "component"},
    "cf_net_change":               {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "total"},
    "cf_closing_cash":             {"semantic_type": "amount", "unit": "USD", "parent_concept": "cash_flow_statement", "role": "total"},
}

# Metadata-only columns to skip (not numeric financial data).
SKIP_COLUMNS = {"company_name", "form_type", "industry", "fiscal_year", "quarter",
                "period", "fiscal_year_start"}

# ---------------------------------------------------------------------------
# Column parsing
# ---------------------------------------------------------------------------

_Q_RE   = re.compile(r'^(.+?)_Q_([A-Z][a-z]{2}_\d{2}_\d{4})$')
_YTD_RE = re.compile(r'^(.+?)_YTD_([A-Z][a-z]{2}_\d{2}_\d{4})$')
_BS_RE  = re.compile(r'^(.+?)_([A-Z][a-z]{2}_\d{2}_\d{4})$')


def _parse_column(col: str) -> Optional[Tuple[str, str, str]]:
    """Return (field, prefix_type, date_readable) or None if not a financial column."""
    m = _Q_RE.match(col)
    if m:
        return m.group(1), "q", m.group(2).replace("_", " ")
    m = _YTD_RE.match(col)
    if m:
        return m.group(1), "ytd", m.group(2).replace("_", " ")
    m = _BS_RE.match(col)
    if m:
        return m.group(1), "bs", m.group(2).replace("_", " ")
    return None


def _period_string(prefix_type: str, date_readable: str) -> str:
    if prefix_type == "q":
        return f"Q {date_readable}"
    if prefix_type == "ytd":
        return f"YTD {date_readable}"
    return date_readable  # balance sheet: plain date


def _concept_name(field: str, prefix_type: str) -> str:
    if prefix_type == "q":
        return f"{field}_q"
    if prefix_type == "ytd":
        return f"{field}_ytd"
    return field  # balance sheet


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_atoms(df: pd.DataFrame) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []

    financial_cols = [c for c in df.columns if c not in SKIP_COLUMNS]

    for row_idx, row in df.iterrows():
        company = row["company_name"]

        for col in financial_cols:
            parsed = _parse_column(col)
            if parsed is None:
                continue
            field, prefix_type, date_readable = parsed

            if field not in CONCEPT_METADATA:
                continue  # skip unknown fields

            raw_val = row.get(col)
            if pd.isna(raw_val):
                continue

            try:
                value = float(raw_val)
            except (TypeError, ValueError):
                continue

            meta    = CONCEPT_METADATA[field]
            concept = _concept_name(field, prefix_type)
            period  = _period_string(prefix_type, date_readable)
            label   = LABELS.get(field, field.replace("_", " "))

            atoms.append({
                "key":            f"{row_idx}_{col}",
                "concept":        concept,
                "semantic_type":  meta["semantic_type"],
                "label":          label,
                "entity":         company,
                "period":         period,
                "unit":           meta["unit"],
                "value":          value,
                "depth":          0,
                "parent_concept": meta["parent_concept"],
                "role":           meta["role"],
            })

    return atoms


# ---------------------------------------------------------------------------
# CLI (mirrors 2.fixed_building_atoms.py interface)
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build 10-Q atoms JSON from CSV.")
    parser.add_argument("--csv",    default="financial_spreadsheet.csv")
    parser.add_argument("--output", default="output_10q/atoms_data.json")
    args = parser.parse_args()

    csv_path    = Path(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df    = pd.read_csv(csv_path)
    atoms = build_atoms(df)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(atoms, f, indent=2, ensure_ascii=False)

    concepts = len({a["concept"] for a in atoms})
    print(f"Saved {len(atoms)} atoms ({concepts} distinct concepts) to {output_path}")


if __name__ == "__main__":
    main()
