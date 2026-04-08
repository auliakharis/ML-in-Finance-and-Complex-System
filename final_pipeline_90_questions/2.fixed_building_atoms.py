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
# What the metadata fields mean
# =========================================================
#
# semantic_type:
#   The broad numeric type of the concept.
#   tells the compiler what kinds of operations are valid.
#   Examples:
#     - "amount","rate", 
#     -"count"   -> can appear in "per employee" or "per share" style ratios
#     - "price"   -> can combine with shares_outstanding to form market cap
#
# unit:
#   Examples:
#     - "USD", "ratio","shares", "employees"...
#
# parent_concept:
#   A broad semantic parent for the concept.
#   This says what bigger category the metric belongs to.
#   It is mostly useful for meaning and hierarchy, not strict arithmetic.
#   Examples:
#     - cash -> current_assets
#     - total_assets -> balance_sheet
#     - stock_price -> market_data
#
# aggregation_parent:
#   The subtotal/group that this concept can naturally roll up into.
#   This is more operational than parent_concept.
#   It helps the compiler detect when several leaves are sibling components
#   of the same aggregate and therefore can be rendered as a natural subtotal.
#
#   Examples:
#     - cash, accounts_receivable, inventories, short_term_investments
#       all have aggregation_parent = "current_assets" (technically overlaps with derived concept for this example)
#
#   Intuition:
#   "If I sum this with its siblings, what named bucket do they form?"
#
# statement:
#   The major financial statement or data area this concept comes from.
#   This helps the compiler prefer financially coherent combinations.
#
#   Examples:
#     - "income_statement"
#     - "balance_sheet"
#     - "cash_flow_statement"
#     - "market_data"
#     - "company_profile"
#
#   Intuition:
#   "Which part of the company’s reported information does this belong to?"
#
# section:
#   A finer-grained grouping inside a statement.
#   This gives the compiler more local structure than statement alone.
#
#   Examples:
#     - cash -> current_assets
#     - capex -> investing_activities
#     - employees -> operating_scale
#     - stock_price -> valuation
#
#   Intuition:
#   "Which subsection inside the broader statement does this belong to?"
#
# role:
#   The structural role of the concept relative to a group.
#   This is useful for telling totals apart from components.
#
#   Examples:
#     - total_assets -> "total"
#     - cash -> "component"
#     - inventories -> "component"
#     - stock_price -> None
#
#   Intuition:
#   "Is this a roll-up total, a component of one, or neither?"
#
#
# =========================================================
# Why these fields help question generation
# =========================================================
#
# With only semantic_type + unit:
#   the compiler knows what is mathematically allowed.
#
# With parent_concept + aggregation_parent + statement + section + role:
#   the compiler starts to know what is financially natural.
#
# That helps it:
#   - sum sibling components into named aggregates
#   - avoid uglier cross-statement combinations
#   - phrase questions more naturally
#   - detect patterns like:
#       * component / total        -> "share of"
#       * amount / employees       -> "per employee"
#       * stock_price * shares     -> "market capitalization"
#       * same concept across years-> "growth over time"
#
#
# =========================================================
# Short version
# =========================================================
#
# semantic_type   = what kind of number this is
# unit            = what it is measured in
# parent_concept  = broad semantic parent
# aggregation_parent = named subtotal/group it rolls into
# statement       = which financial statement / data area it comes from
# section         = finer subsection inside that statement
# role            = whether it is a total, a component, or neither
#=====
CONCEPT_METADATA: Dict[str, Dict[str, Any]] = {
    "revenue": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "income_statement",
        "aggregation_parent": None,
        "statement": "income_statement",
        "section": "revenue",
        "role": "total",
    },
    "cost_of_goods_sold": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "income_statement",
        "aggregation_parent": "gross_profit_inputs",
        "statement": "income_statement",
        "section": "direct_costs",
        "role": "component",
    },
    "operating_expenses": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "income_statement",
        "aggregation_parent": "operating_costs",
        "statement": "income_statement",
        "section": "operating_costs",
        "role": "component",
    },
    "non_operating_expenses": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "income_statement",
        "aggregation_parent": "non_operating_items",
        "statement": "income_statement",
        "section": "non_operating",
        "role": "component",
    },
    "income_tax": {
        "semantic_type": "rate",
        "unit": "ratio",
        "parent_concept": "taxation",
        "aggregation_parent": None,
        "statement": "income_statement",
        "section": "tax",
        "role": None,
    },
    "total_assets": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "balance_sheet",
        "aggregation_parent": None,
        "statement": "balance_sheet",
        "section": "assets",
        "role": "total",
    },
    "total_liabilities": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "balance_sheet",
        "aggregation_parent": None,
        "statement": "balance_sheet",
        "section": "liabilities",
        "role": "total",
    },
    "total_equity": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "balance_sheet",
        "aggregation_parent": None,
        "statement": "balance_sheet",
        "section": "equity",
        "role": "total",
    },
    "cash": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "current_assets",
        "aggregation_parent": "current_assets",
        "statement": "balance_sheet",
        "section": "current_assets",
        "role": "component",
    },
    "accounts_receivable": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "current_assets",
        "aggregation_parent": "current_assets",
        "statement": "balance_sheet",
        "section": "current_assets",
        "role": "component",
    },
    "inventories": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "current_assets",
        "aggregation_parent": "current_assets",
        "statement": "balance_sheet",
        "section": "current_assets",
        "role": "component",
    },
    "short_term_investments": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "current_assets",
        "aggregation_parent": "current_assets",
        "statement": "balance_sheet",
        "section": "current_assets",
        "role": "component",
    },
    "current_liabilities": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "total_liabilities",
        "aggregation_parent": "current_liabilities",
        "statement": "balance_sheet",
        "section": "current_liabilities",
        "role": "component",
    },
    "capex": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "cash_flow_statement",
        "aggregation_parent": "investing_activities",
        "statement": "cash_flow_statement",
        "section": "investing_activities",
        "role": "component",
    },
    "dividends_paid": {
        "semantic_type": "amount",
        "unit": "USD",
        "parent_concept": "cash_flow_statement",
        "aggregation_parent": "financing_activities",
        "statement": "cash_flow_statement",
        "section": "financing_activities",
        "role": "component",
    },
    "shares_outstanding": {
        "semantic_type": "count",
        "unit": "shares",
        "parent_concept": "market_data",
        "aggregation_parent": None,
        "statement": "market_data",
        "section": "capital_structure",
        "role": None,
    },
    "stock_price": {
        "semantic_type": "price",
        "unit": "USD",
        "parent_concept": "market_data",
        "aggregation_parent": None,
        "statement": "market_data",
        "section": "valuation",
        "role": None,
    },
    "employees": {
        "semantic_type": "count",
        "unit": "employees",
        "parent_concept": "company_profile",
        "aggregation_parent": None,
        "statement": "company_profile",
        "section": "operating_scale",
        "role": None,
    },
}

#makes sure every atom value is actually usable and that the imported csv is clean
def normalize_value(value: Any) -> float:
    if pd.isna(value):
        raise ValueError("Missing value in spreadsheet.")
    return float(value)

def validate_dataframe(df: pd.DataFrame) -> None:
    required = {"company_name", "year", *BASE_CONCEPTS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


def build_atoms(df: pd.DataFrame) -> List[Dict[str, Any]]:
    validate_dataframe(df)
    atoms: List[Dict[str, Any]] = []

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
                    "aggregation_parent": meta["aggregation_parent"],
                    "statement": meta["statement"],
                    "section": meta["section"],
                    "role": meta["role"],
                }
            )

    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Build financial atoms JSON from CSV.")
    parser.add_argument("--csv", default="financial_spreadsheet.csv")
    parser.add_argument("--output", default="output/atoms_data.json")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    df = pd.read_csv(csv_path)
    atoms = build_atoms(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(atoms, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(atoms)} atoms to {output_path}")


if __name__ == "__main__":
    main()