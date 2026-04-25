from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_CONCEPTS = [
    "revenue",
    "cost_of_goods_sold",
    "operating_expenses",
    "non_operating_expenses",
    "income_tax",
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash",
    "accounts_receivable",
    "inventories",
    "short_term_investments",
    "current_liabilities",
    "capex",
    "dividends_paid",
    "shares_outstanding",
    "stock_price",
    "employees",
]
#ADDED new metadata: statement, section, agregation parent
# =========================================================
# Metadata fields
# =========================================================
#
# semantic_type  = what kind of number this is
#   "amount"     -> monetary value, supports all arithmetic ops
#   "rate"       -> dimensionless ratio (e.g. tax rate), range 0-1
#   "count"      -> integer count (employees, shares)
#   "price"      -> price per unit (stock price)
#
# unit           = what it is measured in
#   "M_USD"      -> millions of dollars
#   "USD"        -> dollars (stock price)
#   "ratio"      -> dimensionless rate
#   "shares"     -> number of shares (millions)
#   "employees"  -> headcount
#
# parent_concept = named subtotal this concept rolls into when summed with siblings
#   Only set when there is a meaningful intermediate aggregate below the statement level.
#   Used by step 4 to detect natural rollups (e.g. cash + ar + inv + sti = current_assets).
#   Examples: cash -> current_assets, current_liabilities -> total_liabilities
#   None for top-level line items (revenue, total_assets, etc.)
#
# statement      = which financial document this concept comes from
#   "income_statement", "balance_sheet", "cash_flow_statement",
#   "market_data", "company_profile"
#
# role           = structural role relative to its parent_concept group
#   "total"      -> a roll-up total (total_assets, revenue)
#   "component"  -> a part of a named subtotal (cash, inventories)
#   None         -> standalone metric with no group role
#=====
CONCEPT_METADATA: Dict[str, Dict[str, Any]] = {
    "revenue": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "income_statement",
        "role": "total",
    },
    "cost_of_goods_sold": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "income_statement",
        "role": "component",
    },
    "operating_expenses": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "income_statement",
        "role": "component",
    },
    "non_operating_expenses": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "income_statement",
        "role": "component",
    },
    "income_tax": {
        "semantic_type": "rate",
        "unit": "ratio",
        "parent_concept": None,
        "statement": "income_statement",
        "role": None,
    },
    "total_assets": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "balance_sheet",
        "role": "total",
    },
    "total_liabilities": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "balance_sheet",
        "role": "total",
    },
    "total_equity": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "balance_sheet",
        "role": "total",
    },
    "cash": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": "current_assets",
        "statement": "balance_sheet",
        "role": "component",
    },
    "accounts_receivable": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": "current_assets",
        "statement": "balance_sheet",
        "role": "component",
    },
    "inventories": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": "current_assets",
        "statement": "balance_sheet",
        "role": "component",
    },
    "short_term_investments": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": "current_assets",
        "statement": "balance_sheet",
        "role": "component",
    },
    "current_liabilities": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": "total_liabilities",
        "statement": "balance_sheet",
        "role": "component",
    },
    "capex": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "cash_flow_statement",
        "role": "component",
    },
    "dividends_paid": {
        "semantic_type": "amount",
        "unit": "M_USD",
        "parent_concept": None,
        "statement": "cash_flow_statement",
        "role": "component",
    },
    "shares_outstanding": {
        "semantic_type": "count",
        "unit": "shares",
        "parent_concept": None,
        "statement": "market_data",
        "role": None,
    },
    "stock_price": {
        "semantic_type": "price",
        "unit": "USD",
        "parent_concept": None,
        "statement": "market_data",
        "role": None,
    },
    "employees": {
        "semantic_type": "count",
        "unit": "employees",
        "parent_concept": None,
        "statement": "company_profile",
        "role": None,
    },
}

def normalize_value(value: Any) -> float:
    # Reject NaN/empty spreadsheet cells before float coercion.
    if pd.isna(value):
        raise ValueError("Missing value in spreadsheet.")
    return float(value)

def validate_dataframe(df: pd.DataFrame) -> None:
    # Ensure all concepts required by BASE_CONCEPTS are present.
    required = {"company_name", "year", *BASE_CONCEPTS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def build_atoms(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Fail early if the input spreadsheet shape is not compatible.
    validate_dataframe(df)
    atoms: List[Dict[str, Any]] = []

    # Expand each company-year row into one atom per concept.
    for idx, row in df.iterrows():
        for concept in BASE_CONCEPTS:
            meta = CONCEPT_METADATA[concept]
            atoms.append(
                {
                    "key": f"{idx}_{concept}",
                    "concept": concept,
                    "semantic_type": meta["semantic_type"],
                    "label": concept.replace("_", " "),
                    "entity": row["company_name"],
                    "period": str(row["year"]),
                    "unit": meta["unit"],
                    "value": normalize_value(row[concept]),
                    "depth": 0,
                    "parent_concept": meta["parent_concept"],
                    "statement": meta["statement"],
                    "role": meta["role"],
                }
            )

    return atoms


def main() -> None:
    # Parse file paths for input CSV and output JSON.
    parser = argparse.ArgumentParser(description="Build financial atoms JSON from CSV.")
    parser.add_argument("--csv", default="financial_spreadsheet.csv")
    parser.add_argument("--output", default="output/atoms_data.json")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    # Read source data, build atom list, and serialize to disk.
    df = pd.read_csv(csv_path)
    atoms = build_atoms(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(atoms, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(atoms)} atoms to {output_path}")


if __name__ == "__main__":
    main()