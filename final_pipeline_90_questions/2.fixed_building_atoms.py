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

CONCEPT_METADATA: Dict[str, Dict[str, Any]] = {
    "revenue": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": None},
    "cost_of_goods_sold": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": None},
    "operating_expenses": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": None},
    "non_operating_expenses": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": None},
    "income_tax": {"semantic_type": "rate", "unit": "ratio", "parent_concept": None, "role": None},
    "total_assets": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": "total"},
    "total_liabilities": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": "total"},
    "total_equity": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": "total"},
    "cash": {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets", "role": "component"},
    "accounts_receivable": {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets", "role": "component"},
    "inventories": {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets", "role": "component"},
    "short_term_investments": {"semantic_type": "amount", "unit": "USD", "parent_concept": "current_assets", "role": "component"},
    "current_liabilities": {"semantic_type": "amount", "unit": "USD", "parent_concept": "total_liabilities", "role": "component"},
    "capex": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": None},
    "dividends_paid": {"semantic_type": "amount", "unit": "USD", "parent_concept": None, "role": None},
    "shares_outstanding": {"semantic_type": "count", "unit": "shares", "parent_concept": None, "role": None},
    "stock_price": {"semantic_type": "price", "unit": "USD", "parent_concept": None, "role": None},
    "employees": {"semantic_type": "count", "unit": "employees", "parent_concept": None, "role": None},
}


def normalize_value(value: Any, semantic_type: str) -> float:
    if pd.isna(value):
        raise ValueError("Missing value in spreadsheet.")
    return float(value)


def build_atoms(df: pd.DataFrame) -> List[Dict[str, Any]]:
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
                    "value": row[concept],
                    "depth": 0,
                    "parent_concept": meta["parent_concept"],
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