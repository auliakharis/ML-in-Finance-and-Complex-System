"""
Data preparation for the 10Q compiler pipeline.

Replaces the CSV-based data_prep.py from compiler_pipeline_refactored.
Instead of reading a flat spreadsheet, it generates FinancialReport objects
and atomizes them directly — no intermediate CSV needed.

Flow:
    generate_report()  ->  FinancialReport  ->  atomize_report()  ->  [Atom, ...]
"""

import json
import sys
import os
from typing import Optional

from script.compiler_pipeline_refactored_10Q.report_generation.generate import generate_report
from script.compiler_pipeline_refactored_10Q.report_generation.financial_report import FinancialReport, FinancialStatements
from script.compiler_pipeline_adversarial.tree import Atom, SemanticType

CONCEPT_METADATA_FILE = os.path.join(os.path.dirname(__file__), "config", "concept_metadata_10q.json")


# ---------------------------------------------------------------------------
# Field accessors — map concept name -> value on each statement type
# ---------------------------------------------------------------------------

OPS_ACCESSORS = {
    "total_revenues":                   lambda ops: ops.revenue.total_revenues,
    "cost_of_sales":                    lambda ops: ops.costs_and_expenses.cost_of_sales_total,
    "gross_profit":                     lambda ops: ops.gross_profit,
    "research_and_development":         lambda ops: (
        ops.costs_and_expenses.operating_expenses.research_and_development
        if hasattr(ops.costs_and_expenses.operating_expenses, "research_and_development")
        else None
    ),
    "selling_general_and_administrative": lambda ops: (
        ops.costs_and_expenses.operating_expenses.selling_general_and_administrative
        if hasattr(ops.costs_and_expenses.operating_expenses, "selling_general_and_administrative")
        else None
    ),
    "total_costs_and_expenses":         lambda ops: ops.costs_and_expenses.total_costs_and_expenses,
    "operating_income":                 lambda ops: ops.operating_income,
    "other_income_expense_net":         lambda ops: ops._other_income_expense,
    "provision_for_income_taxes":       lambda ops: ops.provision_for_income_taxes,
    "net_income":                       lambda ops: ops.net_income,
    "earnings_per_share_basic":         lambda ops: ops.earnings_per_share.basic if ops.earnings_per_share else None,
    "earnings_per_share_diluted":       lambda ops: ops.earnings_per_share.diluted if ops.earnings_per_share else None,
    "shares_used_basic":                lambda ops: ops.shares_used_in_computing_eps.basic if ops.shares_used_in_computing_eps else None,
    "shares_used_diluted":              lambda ops: ops.shares_used_in_computing_eps.diluted if ops.shares_used_in_computing_eps else None,
}

BS_ACCESSORS = {
    "cash_and_cash_equivalents":        lambda bs: bs.current_assets.cash_and_cash_equivalents,
    "short_term_investments":           lambda bs: bs.current_assets.short_term_investments,
    "accounts_receivable_net":          lambda bs: bs.current_assets.accounts_receivable_net,
    "inventories":                      lambda bs: bs.current_assets.inventories,
    "prepaid_expenses_and_other":       lambda bs: bs.current_assets.prepaid_expenses_and_other,
    "total_current_assets":             lambda bs: bs.total_current_assets,
    "long_term_marketable_securities":  lambda bs: bs.non_current_assets.long_term_marketable_securities,
    "property_plant_and_equipment_net": lambda bs: bs.non_current_assets.property_plant_and_equipment_net,
    "goodwill":                         lambda bs: bs.non_current_assets.goodwill,
    "other_non_current_assets":         lambda bs: bs.non_current_assets.other_non_current_assets,
    "total_non_current_assets":         lambda bs: bs.total_non_current_assets,
    "total_assets":                     lambda bs: bs.total_assets,
    "accounts_payable":                 lambda bs: bs.current_liabilities.accounts_payable,
    "deferred_revenue_current":         lambda bs: bs.current_liabilities.deferred_revenue_current,
    "accrued_expenses_and_other":       lambda bs: bs.current_liabilities.accrued_expenses_and_other,
    "current_portion_of_long_term_debt":lambda bs: bs.current_liabilities.current_portion_of_long_term_debt,
    "total_current_liabilities":        lambda bs: bs.total_current_liabilities,
    "long_term_debt":                   lambda bs: bs.non_current_liabilities.long_term_debt,
    "other_non_current_liabilities":    lambda bs: bs.non_current_liabilities.other_non_current_liabilities,
    "total_non_current_liabilities":    lambda bs: bs.total_non_current_liabilities,
    "total_liabilities":                lambda bs: bs.total_liabilities,
    "common_stock_and_additional_paid_in_capital": lambda bs: bs.shareholders_equity.common_stock_and_additional_paid_in_capital,
    "retained_earnings":                lambda bs: bs.shareholders_equity.retained_earnings,
    "accumulated_other_comprehensive_income_loss": lambda bs: bs.shareholders_equity.accumulated_other_comprehensive_income_loss,
    "total_shareholders_equity":        lambda bs: bs.total_shareholders_equity,
}

CF_ACCESSORS = {
    "net_cash_from_operating":                lambda cf: cf.net_cash_from_operating,
    "depreciation_and_amortization":          lambda cf: cf.operating_activities.depreciation_and_amortization,
    "stock_based_compensation":               lambda cf: cf.operating_activities.stock_based_compensation,
    "change_in_accounts_receivable":          lambda cf: cf.operating_activities.change_in_accounts_receivable,
    "change_in_inventories":                  lambda cf: cf.operating_activities.change_in_inventories,
    "change_in_accounts_payable":             lambda cf: cf.operating_activities.change_in_accounts_payable,
    "change_in_other_working_capital":        lambda cf: cf.operating_activities.change_in_other_working_capital,
    "capital_expenditures":                   lambda cf: cf.investing_activities.capital_expenditures,
    "purchases_of_marketable_securities":     lambda cf: cf.investing_activities.purchases_of_marketable_securities,
    "proceeds_from_maturities_of_securities": lambda cf: cf.investing_activities.proceeds_from_maturities_of_securities,
    "net_cash_from_investing":                lambda cf: cf.net_cash_from_investing,
    "repayments_of_debt":                     lambda cf: cf.financing_activities.repayments_of_debt,
    "share_repurchases":                      lambda cf: cf.financing_activities.share_repurchases,
    "proceeds_from_stock_option_exercises":   lambda cf: cf.financing_activities.proceeds_from_stock_option_exercises,
    "dividends_paid":                         lambda cf: cf.financing_activities.dividends_paid,
    "net_cash_from_financing":                lambda cf: cf.net_cash_from_financing,
    "opening_cash_and_equivalents":           lambda cf: cf.opening_cash_and_equivalents,
    "net_change_in_cash":                     lambda cf: cf.net_change_in_cash,
    "effect_of_exchange_rate_on_cash":        lambda cf: cf.effect_of_exchange_rate_on_cash,
}

# period_type -> slot getter on FinancialStatements
OPS_PERIOD_SLOTS = {
    "current_quarter":    lambda fs: fs.ops_current_quarter,
    "prior_year_quarter": lambda fs: fs.ops_prior_year_quarter,
    "current_ytd":        lambda fs: fs.ops_current_ytd,
    "prior_year_ytd":     lambda fs: fs.ops_prior_year_ytd,
}

BS_PERIOD_SLOTS = {
    "current_quarter_end": lambda fs: fs.bs_current,
    "prior_fy_end":        lambda fs: fs.bs_prior_fy_end,
}

CF_PERIOD_SLOTS = {
    "current_ytd": lambda fs: fs.cf_current_ytd,
    "prior_ytd":   lambda fs: fs.cf_prior_ytd,
}

CI_ACCESSORS = {
    "foreign_currency_translation":          lambda ci: ci.other_comprehensive_income.foreign_currency_translation,
    "unrealized_gains_losses_on_securities": lambda ci: ci.other_comprehensive_income.unrealized_gains_losses_on_securities,
    "total_other_comprehensive_income":      lambda ci: ci.total_other_comprehensive_income,
    "comprehensive_income":                  lambda ci: ci.comprehensive_income,
}

CI_PERIOD_SLOTS = {
    "current_quarter":    lambda fs: fs.ci_current_quarter,
    "prior_year_quarter": lambda fs: fs.ci_prior_year_quarter,
    "current_ytd":        lambda fs: fs.ci_current_ytd,
    "prior_year_ytd":     lambda fs: fs.ci_prior_year_ytd,
}


# ---------------------------------------------------------------------------
# Period label helpers
# ---------------------------------------------------------------------------

def _period_label(report: FinancialReport, period_type: str) -> str:
    """Human-readable period string for an atom."""
    fs = report.financial_statements
    labels = {
        "current_quarter":    fs.ops_current_quarter.period,
        "prior_year_quarter": fs.ops_prior_year_quarter.period,
        "current_ytd":        fs.ops_current_ytd.period if fs.ops_current_ytd else None,
        "prior_year_ytd":     fs.ops_prior_year_ytd.period if fs.ops_prior_year_ytd else None,
        "current_quarter_end":fs.bs_current.as_of,
        "prior_fy_end":       fs.bs_prior_fy_end.as_of,
        "current_ytd_cf":     fs.cf_current_ytd.period,
        "prior_ytd":          fs.cf_prior_ytd.period,
    }
    # cash flow ytd uses same key as income ytd — disambiguate
    if period_type == "current_ytd" and fs.ops_current_ytd is None:
        return fs.cf_current_ytd.period
    return labels.get(period_type, period_type)


# ---------------------------------------------------------------------------
# Helpers for dynamic breakdown extraction
# ---------------------------------------------------------------------------

def _label_to_concept(label: str) -> str:
    """Convert a filing label to a snake_case concept name."""
    import re
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _atomize_breakdown(
    items: dict,
    parent_concept: str,
    parent_meta: dict,
    entity: str,
    period: str,
    period_type: str,
    idx: int,
) -> list[Atom]:
    """Emit one atom per entry in a dynamic breakdown dict, inheriting metadata from parent."""
    is_cos = parent_concept == "cost_of_sales"
    atoms = []
    for label, value in items.items():
        if value is None:
            continue
        concept = ("cos_" if is_cos else "") + _label_to_concept(label)
        display_label = f"cost of {label.lower()}" if is_cos else label
        atoms.append(Atom(
            key=f"{idx}_{concept}_{period_type}",
            concept=concept,
            semantic_type=SemanticType(parent_meta["semantic_type"]),
            label=display_label,
            entity=entity,
            period=period,
            unit=parent_meta["unit"],
            value=float(value),
            depth=0,
            parent_concept=parent_concept,
            role="component",
        ))
        idx += 1
    return atoms


# ---------------------------------------------------------------------------
# Atomizer
# ---------------------------------------------------------------------------

def atomize_report(
    report: FinancialReport,
    concept_metadata: dict,
    atom_index_offset: int = 0,
) -> list[Atom]:
    """
    Walk a FinancialReport and produce a flat list of Atoms.
    Extracts both named fields (from concept_metadata) and dynamic industry
    breakdown lines (NetSales components, other_components, etc.).
    Skips atoms where the value is None.
    """
    fs = report.financial_statements
    entity = report.ticker or report.company_name or "unknown"
    atoms: list[Atom] = []
    idx = atom_index_offset

    # ── Named fields from concept_metadata ──────────────────────────────────
    for concept, meta in concept_metadata["define"].items():
        stmt = meta["statement"]
        ytd_quarters = meta.get("ytd_requires_quarter", [2, 3])
        allowed_period_types = meta["allowed_period_types"]

        if stmt == "income_statement":
            accessors = OPS_ACCESSORS
            slots = OPS_PERIOD_SLOTS
        elif stmt == "balance_sheet":
            accessors = BS_ACCESSORS
            slots = BS_PERIOD_SLOTS
        elif stmt == "cash_flow_statement":
            accessors = CF_ACCESSORS
            slots = CF_PERIOD_SLOTS
        elif stmt == "comprehensive_income_statement":
            accessors = CI_ACCESSORS
            slots = CI_PERIOD_SLOTS
        else:
            continue

        accessor = accessors.get(concept)
        if accessor is None:
            continue

        for period_type in allowed_period_types:
            if period_type in ("current_ytd", "prior_year_ytd") and stmt in ("income_statement", "comprehensive_income_statement"):
                if report.quarter not in ytd_quarters:
                    continue

            slot_getter = slots.get(period_type)
            if slot_getter is None:
                continue

            stmt_obj = slot_getter(fs)
            if stmt_obj is None:
                continue

            try:
                value = accessor(stmt_obj)
            except Exception:
                continue

            if value is None:
                continue

            period = _period_label(report, period_type)
            atoms.append(Atom(
                key=f"{idx}_{concept}_{period_type}",
                concept=concept,
                semantic_type=SemanticType(meta["semantic_type"]),
                label=concept.replace("_", " "),
                entity=entity,
                period=period,
                unit=meta["unit"],
                value=float(value),
                depth=0,
                parent_concept=meta.get("parent_concept"),
                role=meta.get("role"),
            ))
            idx += 1

    # ── Dynamic revenue/cost breakdowns ─────────────────────────────────────
    revenues_meta = concept_metadata["define"]["total_revenues"]
    cos_meta      = concept_metadata["define"]["cost_of_sales"]

    ops_period_slots = [
        ("current_quarter",    fs.ops_current_quarter),
        ("prior_year_quarter", fs.ops_prior_year_quarter),
    ]
    if report.quarter > 1:
        ops_period_slots += [
            ("current_ytd",    fs.ops_current_ytd),
            ("prior_year_ytd", fs.ops_prior_year_ytd),
        ]

    for period_type, ops in ops_period_slots:
        if ops is None:
            continue
        period = _period_label(report, period_type)

        # NetSales component breakdown (e.g. Products / Services)
        ns = ops.revenue.net_sales
        if hasattr(ns, "components") and ns.components:
            breakdown_atoms = _atomize_breakdown(
                ns.components, "total_revenues", revenues_meta, entity, period, period_type, idx
            )
            atoms.extend(breakdown_atoms)
            idx += len(breakdown_atoms)

        # other_components (e.g. Passenger revenue / Cargo for airlines)
        if ops.revenue.other_components:
            breakdown_atoms = _atomize_breakdown(
                ops.revenue.other_components, "total_revenues", revenues_meta, entity, period, period_type, idx
            )
            atoms.extend(breakdown_atoms)
            idx += len(breakdown_atoms)

        # other_operating_revenue (e.g. Membership income for retail)
        if ops.revenue.other_operating_revenue:
            breakdown_atoms = _atomize_breakdown(
                ops.revenue.other_operating_revenue, "total_revenues", revenues_meta, entity, period, period_type, idx
            )
            atoms.extend(breakdown_atoms)
            idx += len(breakdown_atoms)

        # CostOfSales component breakdown
        cos = ops.costs_and_expenses.cost_of_sales
        if hasattr(cos, "components") and cos.components:
            breakdown_atoms = _atomize_breakdown(
                cos.components, "cost_of_sales", cos_meta, entity, period, period_type, idx
            )
            atoms.extend(breakdown_atoms)
            idx += len(breakdown_atoms)

    return atoms


# ---------------------------------------------------------------------------
# Multi-report generation
# ---------------------------------------------------------------------------

def generate_atoms(
    n_reports: int = 50,
    seed: Optional[int] = None,
    concept_metadata_path: str = CONCEPT_METADATA_FILE,
) -> dict[str, Atom]:
    """
    Generate n_reports random 10-Q filings and atomize all of them.
    Returns a flat atom dict keyed by atom.key.
    """
    import random
    rng = random.Random(seed)

    with open(concept_metadata_path, "r", encoding="utf-8") as f:
        concept_metadata = json.load(f)

    atoms: dict[str, Atom] = {}
    offset = 0
    used_tickers: set[str] = set()
    attempts = 0
    max_attempts = n_reports * 20

    while len(used_tickers) < n_reports and attempts < max_attempts:
        report_seed = rng.randint(0, 10_000_000)
        report = generate_report(seed=report_seed)
        attempts += 1
        if report.ticker in used_tickers:
            continue
        used_tickers.add(report.ticker)
        report_atoms = atomize_report(report, concept_metadata, atom_index_offset=offset)
        for atom in report_atoms:
            atoms[atom.key] = atom
        offset += len(report_atoms)

    if len(used_tickers) < n_reports:
        print(f"[WARN] Only generated {len(used_tickers)} unique-ticker reports after {max_attempts} attempts")

    return atoms


# ---------------------------------------------------------------------------
# Main — optional snapshot to atoms.json for inspection
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="Number of 10-Q reports to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/atoms_10q.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Generating {args.n} 10-Q reports...")
    atoms = generate_atoms(n_reports=args.n, seed=args.seed)
    print(f"  -> {len(atoms)} atoms produced")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([a.model_dump() for a in atoms.values()], f, indent=2)
    print(f"  -> saved to {args.output}")


if __name__ == "__main__":
    main()
