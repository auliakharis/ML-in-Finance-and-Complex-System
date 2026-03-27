from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import json, csv
import pandas as pd
import numpy as np

df_data = pd.read_csv('output/financial_spreadsheet.csv')

# Atom concepts
concepts = ['revenue',
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

derived_concepts = {
    "gross_profit": {'calc_rule': df_data['revenue'] - df_data['cost_of_goods_sold'], 'depth': 1},
    "operating_income": {'calc_rule': df_data['revenue'] - df_data['operating_expenses'], 'depth': 1},
    "net_income": {'calc_rule': df_data['revenue'] - df_data['operating_expenses'] - df_data['non_operating_expenses'] - df_data['income_tax'], 'depth': 3},
    "current_assets": {'calc_rule': df_data['cash'] + df_data['accounts_receivable'] + df_data['inventories'] + df_data['short_term_investments'], 'depth': 3},
    "longterm_assets": {'calc_rule': df_data['total_assets'] - (df_data['cash'] + df_data['accounts_receivable'] + df_data['inventories'] + df_data['short_term_investments']), 'depth': 4},
    "longterm_liabilities": {'calc_rule': df_data['total_liabilities'] - df_data['current_liabilities'], 'depth': 1},
}

atoms_data = list()
for idx, row in df_data.iterrows():
    print(row)
    for i, concept in enumerate(concepts):
        atom = {
            "key": idx + i,
            "concept": concept,
            "semantic_type": "amount",     # For now everything is an "amount"
            "label": " ".join(concept.split('_')),
            "entity": row['company_name'],
            "period": row['year'],
            "unit": "USD",                 # TODO: update
            "value": row[concept],
            "depth": 0,
        }
        atoms_data.append(atom)
    for i, concept in enumerate(derived_concepts.keys()):
        atom = {
            "key": idx + (len(df_data) * len(concepts)),
            "concept": concept,
            "semantic_type": "amount",     # For now everything is an "amount"
            "label": " ".join(concept.split('_')),
            "entity": row['company_name'],
            "period": row['year'],
            "unit": "USD",                 # TODO: update
            "value": int(derived_concepts[concept]['calc_rule'].iloc[idx]),
            "depth": derived_concepts[concept]['depth'],
        }
        atoms_data.append(atom)

# save as json
with open('output/atoms_data.json', 'w') as f:
    json.dump(atoms_data, f, indent=4)