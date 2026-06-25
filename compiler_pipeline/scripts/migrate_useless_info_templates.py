"""One-off migration: theme families -> per-concept useless_info families."""

from __future__ import annotations

import json
import re
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

ALL_CONCEPTS = LEAF_CONCEPTS + DERIVED_CONCEPTS

# Keyword -> concept routing for auto-assignment
CONCEPT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "income_tax": ("tax", "taxable", "effective tax", "windfall tax", "tax expense", "tax adjustment"),
    "income_tax_expense": ("tax expense", "tax adjustment", "taxable income"),
    "pretax_income": ("pretax", "pre-tax", "before tax"),
    "net_income": ("net income", "earnings", "profit growth", "reported profits"),
    "gross_profit": ("gross profit", "margin", "cogs", "cost of goods"),
    "operating_income": ("operating income", "operating profit", "operating margin"),
    "revenue": ("revenue", "sales", "reported sales", "unit sales", "pricing", "demand"),
    "cost_of_goods_sold": ("cost of goods", "cogs", "input costs"),
    "operating_expenses": ("operating expense", "opex", "marketing spending", "headcount"),
    "non_operating_expenses": ("non-operating", "non operating"),
    "total_assets": ("total assets", "asset sale", "production capacity", "acquisition"),
    "total_liabilities": ("liabilities", "liability", "debt", "covenant", "leverage"),
    "total_equity": ("equity", "shareholder", "dilution", "dividend"),
    "cash": ("cash position", "liquidity", "financing", "emergency financing"),
    "accounts_receivable": ("receivable", "doubtful accounts"),
    "inventories": ("inventory", "inventories"),
    "short_term_investments": ("short-term investment", "investment"),
    "current_liabilities": ("current liabilities", "payment terms", "suppliers"),
    "current_assets": ("current assets",),
    "longterm_assets": ("long-term asset", "depreciation", "impairment", "useful life"),
    "longterm_liabilities": ("long-term liabilities", "lease obligations"),
    "capex": ("capital expenditure", "capex", "capitalized"),
    "dividends_paid": ("dividend",),
    "shares_outstanding": ("shares outstanding", "share count", "dilution"),
    "stock_price": ("stock", "market capitalization", "valuation multiple"),
    "employees": ("employees", "workforce", "headcount", "layoffs"),
}


def _matches(concept: str, text: str) -> bool:
    lower = text.lower()
    label = concept.replace("_", " ")
    if label in lower:
        return True
    for kw in CONCEPT_KEYWORDS.get(concept, ()):
        if kw in lower:
            return True
    return False


def route_clause(clause: str) -> list[str]:
    hits = [c for c in ALL_CONCEPTS if _matches(c, clause)]
    if hits:
        return hits
    return []


def pick_slice(clauses: list[str], n: int = 8) -> list[str]:
    if len(clauses) <= n:
        return list(clauses)
    step = max(1, len(clauses) // n)
    return [clauses[i] for i in range(0, len(clauses), step)][:n]


def main() -> None:
    src = Path(__file__).resolve().parents[1] / "config" / "useless_info_templates.json"
    with open(src, encoding="utf-8") as f:
        old = json.load(f)

    # Fix corrupted key
    if "ç" in old:
        old["explicit_ignore_numeric"] = old.pop("ç")

    by_concept: dict[str, list[str]] = {c: [] for c in ALL_CONCEPTS}
    generic: list[str] = []

    theme_to_generic = {
        "governance_management",
        "performance_sentiment",
        "peer_comparison",
        "temporal_irrelevant",
    }

    for family, clauses in old.items():
        if family in theme_to_generic:
            generic.extend(clauses)
            continue
        for clause in clauses:
            targets = route_clause(clause)
            if targets:
                for t in targets:
                    if clause not in by_concept[t]:
                        by_concept[t].append(clause)
            elif family == "tax_discount_policy":
                for t in ("income_tax", "income_tax_expense", "pretax_income", "net_income"):
                    if clause not in by_concept[t]:
                        by_concept[t].append(clause)
            elif family == "explicit_ignore_numeric":
                if "revenue" in clause.lower() or "sales" in clause.lower():
                    by_concept["revenue"].append(clause)
                elif "tax" in clause.lower():
                    by_concept["income_tax"].append(clause)
                elif "expense" in clause.lower() or "cost" in clause.lower():
                    by_concept["operating_expenses"].append(clause)
                elif "asset" in clause.lower():
                    by_concept["total_assets"].append(clause)
                else:
                    generic.append(clause)
            elif family in ("bankruptcy_distress",):
                for t in ("cash", "total_liabilities", "current_liabilities"):
                    if clause not in by_concept[t]:
                        by_concept[t].append(clause)
            elif family in ("inflation_macro", "sector_country_context"):
                generic.append(clause)
            elif family in ("future_numeric_adjustments", "accounting_adjustment"):
                for t in route_clause(clause) or ["revenue"]:
                    if clause not in by_concept[t]:
                        by_concept[t].append(clause)
            else:
                generic.append(clause)

    # Ensure every concept has at least 5 clauses
    tax_clauses = old.get("tax_discount_policy", [])
    ignore_clauses = old.get("explicit_ignore_numeric", [])
    for concept in ALL_CONCEPTS:
        if len(by_concept[concept]) >= 5:
            by_concept[concept] = pick_slice(by_concept[concept], 10)
            continue
        pool = list(by_concept[concept])
        if "tax" in concept or concept in ("pretax_income", "net_income", "income_tax_expense"):
            pool.extend(tax_clauses)
            pool.extend(c for c in ignore_clauses if "tax" in c.lower())
        if concept == "revenue":
            pool.extend(c for c in ignore_clauses if "revenue" in c.lower())
            pool.extend(c for c in old.get("future_numeric_adjustments", []) if "revenue" in c.lower())
        if concept in ("gross_profit", "operating_income"):
            pool.extend(c for c in old.get("accounting_adjustment", []) if "expense" in c.lower() or "margin" in c.lower())
        if concept in ("total_assets", "total_liabilities", "cash"):
            pool.extend(old.get("bankruptcy_distress", [])[:15])
        if not pool:
            pool.extend(pick_slice(generic, 5))
        seen: set[str] = set()
        deduped: list[str] = []
        for c in pool:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        by_concept[concept] = pick_slice(deduped, 10) if deduped else pick_slice(generic, 8)

    min_clauses = 5
    generic_unique = list(dict.fromkeys(generic))

    def pad_concept(clauses: list[str]) -> list[str]:
        seen = set(clauses)
        result = list(clauses)
        for extra in generic_unique:
            if len(result) >= min_clauses:
                break
            if extra not in seen:
                seen.add(extra)
                result.append(extra)
        return pick_slice(result, 10) if result else pick_slice(generic_unique, min_clauses)

    out: dict[str, list[str]] = {}
    for concept in ALL_CONCEPTS:
        out[concept] = pad_concept(by_concept[concept])
    out["generic"] = pick_slice(generic_unique, 30)

    dst = src
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {len(out)} families to {dst}")
    for k, v in out.items():
        print(f"  {k}: {len(v)} clauses")


if __name__ == "__main__":
    main()
