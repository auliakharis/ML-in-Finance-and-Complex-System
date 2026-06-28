"""Ensure every pipeline concept has a useless_info template family."""

from __future__ import annotations

import json
from pathlib import Path

LEAF_CONCEPTS = [
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

DERIVED_CONCEPTS = [
    "gross_profit",
    "operating_income",
    "pretax_income",
    "income_tax_expense",
    "net_income",
    "current_assets",
    "longterm_assets",
    "longterm_liabilities",
]

# New family -> copy/adapt from existing family
FAMILY_SEED: dict[str, str] = {
    "non_operating_expenses": "operating_expenses",
    "total_equity": "total_liabilities",
    "accounts_receivable": "total_assets",
    "inventories": "cost_of_goods_sold",
    "short_term_investments": "cash",
    "current_liabilities": "total_liabilities",
    "capex": "total_assets",
    "dividends_paid": "total_equity",
    "shares_outstanding": "total_equity",
    "stock_price": "revenue",
    "employees": "operating_expenses",
    "current_assets": "total_assets",
    "longterm_assets": "total_assets",
    "longterm_liabilities": "total_liabilities",
}

LABEL_REPLACEMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "non_operating_expenses": (
        ("operating expenses", "non operating expenses"),
        ("operating expense", "non operating expense"),
        ("expense reductions", "non operating charge reductions"),
        ("operating income", "pretax income"),
        ("operating profit", "pretax income"),
        ("opex", "non operating expenses"),
    ),
    "total_equity": (
        ("liabilities", "equity"),
        ("liability", "equity"),
        ("debt", "equity"),
        ("debt load", "equity base"),
        ("refinancing", "equity issuance"),
        ("debt exchange", "share repurchase"),
        ("covenant", "capital return policy"),
    ),
    "accounts_receivable": (
        ("total assets", "accounts receivable"),
        ("assets", "receivables"),
        ("asset sale", "receivable collection"),
        ("asset valuation", "allowance for doubtful accounts"),
        ("impairment charge", "bad debt reserve"),
        ("production capacity", "customer collections"),
    ),
    "inventories": (
        ("cost of goods sold", "inventories"),
        ("input costs", "inventory carrying costs"),
        ("raw material", "inventory"),
        ("supplier", "warehouse"),
        ("vendor contracts", "inventory purchases"),
        ("freight costs", "storage costs"),
    ),
    "short_term_investments": (
        ("cash", "short term investments"),
        ("liquidity", "investment portfolio"),
        ("financing", "portfolio rebalancing"),
        ("credit facility", "treasury investments"),
        ("cash position", "short term investment balance"),
        ("cash-management", "treasury management"),
    ),
    "current_liabilities": (
        ("total liabilities", "current liabilities"),
        ("liabilities", "current liabilities"),
        ("debt", "short term obligations"),
        ("debt restructuring", "working capital adjustment"),
        ("lease reclassification", "payables reclassification"),
    ),
    "capex": (
        ("total assets", "capital expenditures"),
        ("assets", "capex"),
        ("asset sale", "project deferral"),
        ("impairment", "project write-down"),
        ("production capacity", "capital projects"),
        ("acquisition", "facility investment"),
    ),
    "dividends_paid": (
        ("equity", "dividends paid"),
        ("shareholder", "dividend"),
        ("share repurchase", "dividend payout"),
        ("dilution", "dividend policy"),
        ("dividend", "dividends paid"),
    ),
    "shares_outstanding": (
        ("equity", "shares outstanding"),
        ("shareholder", "share count"),
        ("share repurchase", "share issuance"),
        ("dilution", "share count change"),
        ("dividend", "shares outstanding"),
    ),
    "stock_price": (
        ("revenue", "stock price"),
        ("sales", "share price"),
        ("pricing", "market price"),
        ("selling prices", "trading price"),
        ("revenue recognition", "market valuation"),
    ),
    "employees": (
        ("operating expenses", "employee costs"),
        ("headcount", "employees"),
        ("expense reductions", "workforce reductions"),
        ("restructuring", "hiring freeze"),
        ("consulting costs", "compensation costs"),
    ),
    "current_assets": (
        ("total assets", "current assets"),
        ("assets", "current assets"),
        ("asset sale", "working capital release"),
        ("impairment", "inventory write-down"),
    ),
    "longterm_assets": (
        ("total assets", "long term assets"),
        ("assets", "long term assets"),
        ("depreciation", "useful life"),
        ("impairment", "long lived asset impairment"),
        ("asset valuation", "PP&E review"),
    ),
    "longterm_liabilities": (
        ("total liabilities", "long term liabilities"),
        ("liabilities", "long term liabilities"),
        ("debt", "long term debt"),
        ("refinancing", "bond refinancing"),
    ),
}


def adapt_clauses(concept: str, seed_clauses: list[str]) -> list[str]:
    replacements = LABEL_REPLACEMENTS.get(concept, ())
    adapted: list[str] = []
    for clause in seed_clauses:
        text = clause
        for old, new in replacements:
            text = text.replace(old, new)
        adapted.append(text)
    return adapted


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "useless_info_templates.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    generic = data.get("generic", [])

    for concept in LEAF_CONCEPTS + DERIVED_CONCEPTS:
        if concept in data and data[concept]:
            continue
        seed_name = FAMILY_SEED.get(concept, "revenue")
        seed = data.get(seed_name) or generic
        data[concept] = adapt_clauses(concept, list(seed))[:10]

    if "generic" not in data:
        data["generic"] = generic

    ordered: dict[str, list[str]] = {}
    for concept in LEAF_CONCEPTS + DERIVED_CONCEPTS:
        ordered[concept] = data[concept]
    ordered["generic"] = data["generic"]

    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(ordered)} families to {path}")


if __name__ == "__main__":
    main()
