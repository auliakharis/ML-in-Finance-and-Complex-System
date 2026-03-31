from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


BASE_CONCEPTS = [
    'revenue',
    'cost_of_goods_sold',
    'operating_expenses',
    'non_operating_expenses',
    'income_tax',
    'total_assets',
    'total_liabilities',
    'total_equity',
    'cash',
    'accounts_receivable',
    'inventories',
    'short_term_investments',
    'current_liabilities',
    'capex',
    'dividends_paid',
    'shares_outstanding',
    'stock_price',
]

CONCEPT_METADATA: Dict[str, Dict[str, Any]] = {
    'revenue': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'cost_of_goods_sold': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'operating_expenses': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'non_operating_expenses': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'income_tax': {'semantic_type': 'rate', 'unit': 'ratio', 'parent_concept': None, 'role': None},
    'total_assets': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': 'total'},
    'total_liabilities': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': 'total'},
    'total_equity': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': 'total'},
    'cash': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': 'current_assets', 'role': 'component'},
    'accounts_receivable': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': 'current_assets', 'role': 'component'},
    'inventories': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': 'current_assets', 'role': 'component'},
    'short_term_investments': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': 'current_assets', 'role': 'component'},
    'current_liabilities': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': 'total_liabilities', 'role': 'component'},
    'capex': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'dividends_paid': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'shares_outstanding': {'semantic_type': 'count', 'unit': 'shares', 'parent_concept': None, 'role': None},
    'stock_price': {'semantic_type': 'price', 'unit': 'USD/share', 'parent_concept': None, 'role': None},
    'gross_profit': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'operating_income': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'pretax_income': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'income_tax_expense': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'net_income': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': None, 'role': None},
    'current_assets': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': 'total_assets', 'role': 'component'},
    'longterm_assets': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': 'total_assets', 'role': 'component'},
    'longterm_liabilities': {'semantic_type': 'amount', 'unit': 'USD', 'parent_concept': 'total_liabilities', 'role': 'component'},
}


REQUIRED_COLUMNS = [
    'company_name', 'year', *BASE_CONCEPTS,
]


def normalize_value(value: Any, semantic_type: str) -> Any:
    if pd.isna(value):
        return None
    value = float(value)
    if semantic_type == 'rate':
        return round(value, 6)
    if semantic_type == 'price':
        return round(value, 2)
    rounded = round(value, 2)
    return int(rounded) if rounded.is_integer() else rounded


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_atoms(df: pd.DataFrame) -> List[Dict[str, Any]]:
    gross_profit = df['revenue'] - df['cost_of_goods_sold']
    operating_income = gross_profit - df['operating_expenses']
    pretax_income = operating_income - df['non_operating_expenses']
    income_tax_expense = pretax_income * df['income_tax']
    net_income = pretax_income - income_tax_expense
    current_assets = (
        df['cash']
        + df['accounts_receivable']
        + df['inventories']
        + df['short_term_investments']
    )
    longterm_assets = df['total_assets'] - current_assets
    longterm_liabilities = df['total_liabilities'] - df['current_liabilities']

    derived = {
        'gross_profit': {'series': gross_profit, 'depth': 1},
        'operating_income': {'series': operating_income, 'depth': 2},
        'pretax_income': {'series': pretax_income, 'depth': 3},
        'income_tax_expense': {'series': income_tax_expense, 'depth': 4},
        'net_income': {'series': net_income, 'depth': 4},
        'current_assets': {'series': current_assets, 'depth': 1},
        'longterm_assets': {'series': longterm_assets, 'depth': 2},
        'longterm_liabilities': {'series': longterm_liabilities, 'depth': 1},
    }

    atoms: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        for concept in BASE_CONCEPTS:
            meta = CONCEPT_METADATA[concept]
            atoms.append({
                'key': f'{idx}_{concept}',
                'concept': concept,
                'semantic_type': meta['semantic_type'],
                'label': concept.replace('_', ' '),
                'entity': row['company_name'],
                'period': str(row['year']),
                'unit': meta['unit'],
                'value': normalize_value(row[concept], meta['semantic_type']),
                'depth': 0,
                'parent_concept': meta['parent_concept'],
                'role': meta['role'],
            })

        for concept, spec in derived.items():
            meta = CONCEPT_METADATA[concept]
            atoms.append({
                'key': f'{idx}_{concept}',
                'concept': concept,
                'semantic_type': meta['semantic_type'],
                'label': concept.replace('_', ' '),
                'entity': row['company_name'],
                'period': str(row['year']),
                'unit': meta['unit'],
                'value': normalize_value(spec['series'].iloc[idx], meta['semantic_type']),
                'depth': spec['depth'],
                'parent_concept': meta['parent_concept'],
                'role': meta['role'],
            })
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description='Build financial atoms JSON from CSV.')
    parser.add_argument('--csv', default='financial_spreadsheet.csv')
    parser.add_argument('--output', default='output/atoms_data.json')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    df = pd.read_csv(csv_path)
    validate_columns(df)
    atoms = build_atoms(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(atoms, f, indent=2, ensure_ascii=False)

    print(f'Saved {len(atoms)} atoms to {output_path}')


if __name__ == '__main__':
    main()
