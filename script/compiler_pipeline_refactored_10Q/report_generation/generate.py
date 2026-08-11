"""
Random 10-Q Financial Report Generator

Generates a realistic but fictional FinancialReport by:
  1. Picking a company identity (name, ticker, industry)
  2. Picking a fiscal year structure and quarter
  3. Picking a revenue scale
  4. Applying industry-realistic financial ratios with randomness
  5. Populating all five statements and core notes

Usage:
    from generate import generate_report
    report = generate_report()           # fully random
    report = generate_report(industry="tech", quarter=1)
    report = generate_report(seed=42)    # reproducible
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .statements import (
    NetSales, Revenue, CostOfSales, OperatingExpenses, CostsAndExpenses,
    EarningsPerShare, SharesUsed, StatementOfOperations,
    OtherComprehensiveIncome, StatementOfComprehensiveIncome,
    CurrentAssets, NonCurrentAssets, CurrentLiabilities, NonCurrentLiabilities,
    ShareholdersEquity, BalanceSheet,
    EquityRollforwardPeriod, StatementOfShareholdersEquity,
    OperatingActivities, InvestingActivities, FinancingActivities,
    StatementOfCashFlows,
)
from .notes import Notes
from .financial_report import FinancialStatements, FinancialReport


# ---------------------------------------------------------------------------
# Industry profiles
# Each profile defines realistic ratio ranges as (min, max) tuples.
# All ratios are expressed as a fraction of net sales unless noted.
# ---------------------------------------------------------------------------

@dataclass
class IndustryProfile:
    name: str
    # Revenue
    splits_net_sales: bool                        # whether net_sales is broken into categories
    net_sales_categories: list[str]               # e.g. ["Products", "Services"] or ["Passenger revenue", "Cargo and other"]
    has_other_operating_revenue: bool           # e.g. membership income
    # Margins
    gross_profit_margin: tuple[float, float]    # gross profit / net sales
    rd_ratio: tuple[float, float]               # R&D / net sales
    sga_ratio: tuple[float, float]              # SG&A / net sales
    # Below the line
    other_income_ratio: tuple[float, float]     # other income / net sales
    tax_rate: tuple[float, float]               # effective tax rate
    # Balance sheet
    cash_ratio: tuple[float, float]             # cash / net sales (annualised)
    inventory_ratio: tuple[float, float]        # inventory / net sales
    ppe_ratio: tuple[float, float]              # PP&E net / net sales
    goodwill_ratio: tuple[float, float]         # goodwill / net sales
    debt_ratio: tuple[float, float]             # long-term debt / net sales
    # Cash flows
    capex_ratio: tuple[float, float]            # capex / net sales
    da_ratio: tuple[float, float]               # D&A / net sales
    # Share count range (millions of shares)
    shares_range: tuple[float, float]
    # Scale: annual revenue range in millions
    revenue_range: tuple[float, float]
    # Ticker suffix pool
    ticker_pool: list[str]


INDUSTRIES: dict[str, IndustryProfile] = {
    "tech": IndustryProfile(
        name="Technology",
        splits_net_sales=True,
        net_sales_categories=["Products", "Services"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.55, 0.75),
        rd_ratio=(0.10, 0.18),
        sga_ratio=(0.08, 0.14),
        other_income_ratio=(0.00, 0.02),
        tax_rate=(0.14, 0.20),
        cash_ratio=(0.15, 0.35),
        inventory_ratio=(0.02, 0.06),
        ppe_ratio=(0.05, 0.15),
        goodwill_ratio=(0.10, 0.40),
        debt_ratio=(0.10, 0.40),
        capex_ratio=(0.03, 0.07),
        da_ratio=(0.03, 0.06),
        shares_range=(500, 16_000),
        revenue_range=(2_000, 400_000),
        ticker_pool=["SYS", "NXT", "CRX", "VLT", "DNX", "PLX", "QRK"],
    ),
    "retail": IndustryProfile(
        name="Retail",
        splits_net_sales=False,
        net_sales_categories=["Net sales"],
        has_other_operating_revenue=True,
        gross_profit_margin=(0.22, 0.35),
        rd_ratio=(0.00, 0.01),
        sga_ratio=(0.18, 0.26),
        other_income_ratio=(-0.01, 0.00),
        tax_rate=(0.22, 0.28),
        cash_ratio=(0.04, 0.10),
        inventory_ratio=(0.10, 0.20),
        ppe_ratio=(0.20, 0.40),
        goodwill_ratio=(0.05, 0.20),
        debt_ratio=(0.15, 0.35),
        capex_ratio=(0.02, 0.05),
        da_ratio=(0.02, 0.04),
        shares_range=(200, 9_000),
        revenue_range=(5_000, 600_000),
        ticker_pool=["MRT", "GRV", "TRD", "BZR", "CRG", "NVX", "PLZ"],
    ),
    "pharma": IndustryProfile(
        name="Pharmaceuticals",
        splits_net_sales=True,
        net_sales_categories=["Products", "Royalties and other"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.60, 0.80),
        rd_ratio=(0.15, 0.25),
        sga_ratio=(0.20, 0.30),
        other_income_ratio=(-0.02, 0.01),
        tax_rate=(0.10, 0.18),
        cash_ratio=(0.10, 0.30),
        inventory_ratio=(0.05, 0.12),
        ppe_ratio=(0.08, 0.20),
        goodwill_ratio=(0.20, 0.60),
        debt_ratio=(0.20, 0.50),
        capex_ratio=(0.04, 0.08),
        da_ratio=(0.04, 0.08),
        shares_range=(400, 6_000),
        revenue_range=(1_000, 100_000),
        ticker_pool=["MXP", "VRX", "GNT", "CLR", "BPH", "ZNX", "TPH"],
    ),
    "energy": IndustryProfile(
        name="Energy",
        splits_net_sales=False,
        net_sales_categories=["Revenues"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.25, 0.45),
        rd_ratio=(0.00, 0.02),
        sga_ratio=(0.04, 0.10),
        other_income_ratio=(-0.02, 0.02),
        tax_rate=(0.20, 0.35),
        cash_ratio=(0.05, 0.15),
        inventory_ratio=(0.03, 0.08),
        ppe_ratio=(0.40, 0.70),
        goodwill_ratio=(0.02, 0.15),
        debt_ratio=(0.20, 0.45),
        capex_ratio=(0.06, 0.12),
        da_ratio=(0.05, 0.10),
        shares_range=(500, 5_000),
        revenue_range=(5_000, 400_000),
        ticker_pool=["PXE", "CRD", "VLX", "GLP", "TXN", "BRX", "NRG"],
    ),
    "airline": IndustryProfile(
        name="Airlines",
        splits_net_sales=False,
        net_sales_categories=["Passenger revenue", "Cargo and other"],
        has_other_operating_revenue=True,
        gross_profit_margin=(0.25, 0.40),
        rd_ratio=(0.00, 0.01),
        sga_ratio=(0.06, 0.12),
        other_income_ratio=(-0.04, -0.01),
        tax_rate=(0.20, 0.26),
        cash_ratio=(0.08, 0.20),
        inventory_ratio=(0.01, 0.03),
        ppe_ratio=(0.40, 0.65),
        goodwill_ratio=(0.02, 0.10),
        debt_ratio=(0.40, 0.70),
        capex_ratio=(0.06, 0.12),
        da_ratio=(0.04, 0.08),
        shares_range=(200, 800),
        revenue_range=(2_000, 60_000),
        ticker_pool=["AVX", "SKY", "FLT", "JTX", "CRZ", "WNG", "HRZ"],
    ),
    "wholesale": IndustryProfile(
        name="Wholesale",
        splits_net_sales=False,
        net_sales_categories=["Net sales"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.18, 0.28),
        rd_ratio=(0.00, 0.01),
        sga_ratio=(0.10, 0.18),
        other_income_ratio=(-0.01, 0.01),
        tax_rate=(0.20, 0.26),
        cash_ratio=(0.03, 0.08),
        inventory_ratio=(0.12, 0.22),
        ppe_ratio=(0.08, 0.18),
        goodwill_ratio=(0.05, 0.20),
        debt_ratio=(0.10, 0.30),
        capex_ratio=(0.01, 0.03),
        da_ratio=(0.01, 0.03),
        shares_range=(100, 3_000),
        revenue_range=(2_000, 200_000),
        ticker_pool=["WHL", "DST", "TRD", "SUP", "GBL", "MRK", "DXW"],
    ),
    "manufacturing": IndustryProfile(
        name="Manufacturing",
        splits_net_sales=False,
        net_sales_categories=["Net sales"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.25, 0.45),
        rd_ratio=(0.02, 0.06),
        sga_ratio=(0.08, 0.15),
        other_income_ratio=(-0.02, 0.01),
        tax_rate=(0.18, 0.25),
        cash_ratio=(0.04, 0.12),
        inventory_ratio=(0.10, 0.20),
        ppe_ratio=(0.20, 0.40),
        goodwill_ratio=(0.05, 0.25),
        debt_ratio=(0.15, 0.35),
        capex_ratio=(0.03, 0.07),
        da_ratio=(0.03, 0.06),
        shares_range=(100, 5_000),
        revenue_range=(1_000, 150_000),
        ticker_pool=["MFG", "IND", "PRD", "FAB", "CRF", "BLD", "MTX"],
    ),
    "finance": IndustryProfile(
        name="Finance & Insurance",
        splits_net_sales=False,
        net_sales_categories=["Net interest income", "Non-interest income"],
        has_other_operating_revenue=True,
        gross_profit_margin=(0.55, 0.75),
        rd_ratio=(0.00, 0.01),
        sga_ratio=(0.30, 0.45),
        other_income_ratio=(-0.02, 0.02),
        tax_rate=(0.18, 0.25),
        cash_ratio=(0.10, 0.30),
        inventory_ratio=(0.00, 0.01),
        ppe_ratio=(0.02, 0.08),
        goodwill_ratio=(0.05, 0.20),
        debt_ratio=(0.30, 0.60),
        capex_ratio=(0.01, 0.03),
        da_ratio=(0.01, 0.03),
        shares_range=(200, 8_000),
        revenue_range=(2_000, 120_000),
        ticker_pool=["FNB", "INS", "CAP", "FIN", "BNK", "TRU", "AST"],
    ),
    "healthcare": IndustryProfile(
        name="Healthcare & Social Assistance",
        splits_net_sales=True,
        net_sales_categories=["Patient service revenue", "Other revenue"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.35, 0.55),
        rd_ratio=(0.02, 0.08),
        sga_ratio=(0.12, 0.22),
        other_income_ratio=(-0.01, 0.01),
        tax_rate=(0.18, 0.25),
        cash_ratio=(0.05, 0.15),
        inventory_ratio=(0.02, 0.06),
        ppe_ratio=(0.15, 0.30),
        goodwill_ratio=(0.10, 0.35),
        debt_ratio=(0.15, 0.40),
        capex_ratio=(0.03, 0.07),
        da_ratio=(0.03, 0.06),
        shares_range=(100, 4_000),
        revenue_range=(500, 80_000),
        ticker_pool=["HLT", "MED", "CRX", "HCS", "WCR", "VTL", "NVH"],
    ),
    "construction": IndustryProfile(
        name="Construction",
        splits_net_sales=False,
        net_sales_categories=["Contract revenues"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.15, 0.25),
        rd_ratio=(0.00, 0.01),
        sga_ratio=(0.06, 0.12),
        other_income_ratio=(-0.01, 0.01),
        tax_rate=(0.20, 0.26),
        cash_ratio=(0.04, 0.10),
        inventory_ratio=(0.03, 0.08),
        ppe_ratio=(0.10, 0.25),
        goodwill_ratio=(0.02, 0.12),
        debt_ratio=(0.10, 0.30),
        capex_ratio=(0.02, 0.05),
        da_ratio=(0.02, 0.04),
        shares_range=(50, 1_500),
        revenue_range=(500, 50_000),
        ticker_pool=["BLD", "CNS", "STR", "GRD", "INF", "CIV", "ENG"],
    ),
    "professional_services": IndustryProfile(
        name="Professional, Scientific & Technical Services",
        splits_net_sales=True,
        net_sales_categories=["Consulting revenue", "Technology services"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.30, 0.50),
        rd_ratio=(0.02, 0.08),
        sga_ratio=(0.15, 0.25),
        other_income_ratio=(-0.01, 0.01),
        tax_rate=(0.20, 0.26),
        cash_ratio=(0.08, 0.20),
        inventory_ratio=(0.00, 0.02),
        ppe_ratio=(0.03, 0.10),
        goodwill_ratio=(0.10, 0.35),
        debt_ratio=(0.05, 0.25),
        capex_ratio=(0.01, 0.04),
        da_ratio=(0.02, 0.05),
        shares_range=(50, 2_000),
        revenue_range=(200, 30_000),
        ticker_pool=["PST", "CNS", "ADV", "TCS", "PRF", "SCI", "KNW"],
    ),
    "information": IndustryProfile(
        name="Information",
        splits_net_sales=True,
        net_sales_categories=["Subscription revenue", "Advertising revenue"],
        has_other_operating_revenue=False,
        gross_profit_margin=(0.45, 0.70),
        rd_ratio=(0.08, 0.18),
        sga_ratio=(0.15, 0.28),
        other_income_ratio=(-0.01, 0.02),
        tax_rate=(0.15, 0.22),
        cash_ratio=(0.10, 0.25),
        inventory_ratio=(0.00, 0.02),
        ppe_ratio=(0.08, 0.20),
        goodwill_ratio=(0.10, 0.40),
        debt_ratio=(0.10, 0.35),
        capex_ratio=(0.04, 0.10),
        da_ratio=(0.04, 0.08),
        shares_range=(100, 6_000),
        revenue_range=(500, 80_000),
        ticker_pool=["INF", "MDI", "STR", "BCT", "DTA", "NET", "CLD"],
    ),
}


# ---------------------------------------------------------------------------
# Company name generator
# ---------------------------------------------------------------------------

PREFIXES = ["Apex", "Nova", "Vex", "Crest", "Zion", "Alto", "Mira",
            "Plex", "Orion", "Trek", "Volt", "Axon", "Core", "Peak"]
SUFFIXES = ["Corp", "Inc", "Group", "Holdings", "Technologies",
            "Enterprises", "Solutions", "Industries", "Systems"]


def _generate_company_name(rng: random.Random) -> tuple[str, str]:
    """Returns (company_name, ticker)."""
    prefix = rng.choice(PREFIXES)
    suffix = rng.choice(SUFFIXES)
    name = f"{prefix} {suffix}"
    ticker = prefix[:3].upper()
    return name, ticker


# ---------------------------------------------------------------------------
# Fiscal year helpers
# ---------------------------------------------------------------------------

# (fiscal_year_start_month, fiscal_year_start_day) for each industry archetype
FISCAL_YEAR_STARTS = {
    "tech":                 (10, 1),  # Apple-like: starts October
    "retail":               (1,  1),  # calendar year: starts January
    "pharma":               (1,  1),  # calendar year
    "energy":               (1,  1),  # calendar year
    "airline":              (1,  1),  # calendar year
    "wholesale":            (1,  1),  # calendar year
    "manufacturing":        (1,  1),  # calendar year
    "finance":              (1,  1),  # calendar year
    "healthcare":           (1,  1),  # calendar year
    "construction":         (1,  1),  # calendar year
    "professional_services":(1,  1),  # calendar year
    "information":          (1,  1),  # calendar year
}

QUARTER_END_MONTHS = [3, 6, 9, 12]  # months into fiscal year each quarter ends


def _fiscal_year_dates(
    industry: str,
    quarter: int,
    filing_year: int,
) -> tuple[date, date, date, str, str]:
    """
    Returns:
      quarter_end_date, prior_fy_end_date, prior_year_quarter_end_date,
      period_label, as_of_label
    """
    start_month, start_day = FISCAL_YEAR_STARTS[industry]

    # Fiscal year start for the filing year
    fy_start = date(filing_year, start_month, start_day)

    # Quarter end = fy_start + 3*quarter months, minus 1 day
    q_month = (start_month - 1 + 3 * quarter) % 12 + 1
    q_year = filing_year + ((start_month - 1 + 3 * quarter) // 12)
    # Use last day of the quarter-end month
    next_month = q_month % 12 + 1
    next_year = q_year if next_month > 1 else q_year + 1
    quarter_end = date(next_year, next_month, 1) - timedelta(days=1)

    # Prior fiscal year end = day before fy_start
    prior_fy_end = fy_start - timedelta(days=1)

    # Prior year same quarter end
    prior_quarter_end = date(
        quarter_end.year - 1, quarter_end.month, quarter_end.day
    )

    period_label = f"Three Months Ended {quarter_end.strftime('%B %d, %Y')}"
    as_of_label = quarter_end.strftime("%B %d, %Y")

    return quarter_end, prior_fy_end, prior_quarter_end, period_label, as_of_label


# ---------------------------------------------------------------------------
# Ratio helpers
# ---------------------------------------------------------------------------

def _r(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def _annualised_to_quarterly(annual: float, quarter: int) -> float:
    """Rough quarterly figure: divide annual by 4, add some seasonality."""
    base = annual / 4
    # Q4 tends to be strongest for retail/tech; keep simple here
    return base


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_report(
    industry: Optional[str] = None,
    quarter: Optional[int] = None,
    filing_year: int = 2025,
    seed: Optional[int] = None,
    include_prior_year_quarter_bs: Optional[bool] = None,
) -> FinancialReport:
    """
    Generate a random but internally consistent FinancialReport.

    Args:
        industry:    one of "tech", "retail", "pharma", "energy", "airline"
                     (random if omitted)
        quarter:     1, 2, or 3 (random if omitted)
        filing_year: calendar year of the filing (default 2025)
        seed:        random seed for reproducibility
        include_prior_year_quarter_bs: whether to include the optional 3rd
                     balance sheet column (prior year same quarter). If None,
                     defaults to True for large accelerated filers and False
                     for smaller reporting companies (randomly assigned).
    """
    rng = random.Random(seed)

    # ── Identity ──────────────────────────────────────────────────────────
    if industry is None:
        industry = rng.choice(list(INDUSTRIES.keys()))
    profile = INDUSTRIES[industry]

    if quarter is None:
        quarter = rng.randint(1, 3)

    company_name, ticker = _generate_company_name(rng)
    ticker = rng.choice(profile.ticker_pool)

    # ── Scale ─────────────────────────────────────────────────────────────
    annual_revenue = _r(rng, *profile.revenue_range)
    quarterly_revenue = _annualised_to_quarterly(annual_revenue, quarter)

    # ── Dates ─────────────────────────────────────────────────────────────
    (quarter_end, prior_fy_end, prior_q_end,
     period_label, as_of_label) = _fiscal_year_dates(industry, quarter, filing_year)

    fy_start_month, fy_start_day = FISCAL_YEAR_STARTS[industry]
    fiscal_year_start = date(filing_year, fy_start_month, fy_start_day).strftime("%B %d, %Y")

    # ── Income Statement ──────────────────────────────────────────────────
    net_sales_total = round(quarterly_revenue, 0)
    gross_profit_margin = _r(rng, *profile.gross_profit_margin)
    cost_of_sales_total = round(net_sales_total * (1 - gross_profit_margin), 0)

    rd = round(net_sales_total * _r(rng, *profile.rd_ratio), 0) if profile.rd_ratio[1] > 0 else None
    sga = round(net_sales_total * _r(rng, *profile.sga_ratio), 0)

    other_income = round(net_sales_total * _r(rng, *profile.other_income_ratio), 0)
    tax_rate = _r(rng, *profile.tax_rate)

    # Revenue — three cases:
    # 1. splits_net_sales=True  → NetSales with breakdown, no other_op_rev from categories
    # 2. splits_net_sales=False, has_other_operating_revenue=False → plain float net_sales
    # 3. splits_net_sales=False, has_other_operating_revenue=True  → net_sales=None,
    #    all revenue goes into other_operating_revenue (airlines, banks)

    other_op_rev: dict = {}
    other_components: dict = {}

    if profile.splits_net_sales and len(profile.net_sales_categories) == 2:
        # Case 1: net_sales broken into named categories (Apple, Delta won't use this)
        split = _r(rng, 0.55, 0.90)
        cat_a = round(net_sales_total * split, 0)
        cat_b = round(net_sales_total * (1 - split), 0)
        net_sales = NetSales(components={
            profile.net_sales_categories[0]: cat_a,
            profile.net_sales_categories[1]: cat_b,
        })
        cost_of_sales = CostOfSales(components={
            profile.net_sales_categories[0]: round(cat_a * (1 - gross_profit_margin * _r(rng, 0.9, 1.1)), 0),
            profile.net_sales_categories[1]: round(cat_b * (1 - gross_profit_margin * _r(rng, 0.7, 1.0)), 0),
        })

    elif profile.has_other_operating_revenue and len(profile.net_sales_categories) >= 2:
        # Case 3: no net_sales — primary revenue lines go into other_components (airlines, banks)
        net_sales = None
        cost_of_sales = cost_of_sales_total
        split = _r(rng, 0.55, 0.90)
        other_components = {
            profile.net_sales_categories[0]: round(net_sales_total * split, 0),
            profile.net_sales_categories[1]: round(net_sales_total * (1 - split), 0),
        }

    else:
        # Case 2: single net_sales line (retail, energy, manufacturing, etc.)
        net_sales = net_sales_total
        cost_of_sales = cost_of_sales_total
        if profile.has_other_operating_revenue:
            other_op_rev = {"Membership and other income": round(net_sales_total * _r(rng, 0.005, 0.015), 0)}

    revenue = Revenue(net_sales=net_sales, other_components=other_components, other_operating_revenue=other_op_rev)

    opex = OperatingExpenses(
        research_and_development=rd,
        selling_general_and_administrative=sga,
    )
    costs = CostsAndExpenses(cost_of_sales=cost_of_sales, operating_expenses=opex)

    # Derive pre-tax income to back into tax provision
    gross_profit = revenue.net_sales_total - costs.cost_of_sales_total
    op_income = revenue.total_revenues - costs.total_costs_and_expenses
    pretax = op_income + other_income
    tax_provision = round(max(pretax * tax_rate, 0), 0)

    shares_basic = round(_r(rng, *profile.shares_range), 0)
    shares_diluted = round(shares_basic * _r(rng, 1.001, 1.015), 0)

    ops = StatementOfOperations(
        period=period_label,
        revenue=revenue,
        costs_and_expenses=costs,
        other_income_expense_net=other_income,
        provision_for_income_taxes=tax_provision,
        earnings_per_share=EarningsPerShare(
            # net_income in millions, shares_basic in millions → EPS in dollars per share
            basic=round((pretax - tax_provision) / shares_basic, 2) if shares_basic else 0,
            diluted=round((pretax - tax_provision) / shares_diluted, 2) if shares_diluted else 0,
        ),
        shares_used_in_computing_eps=SharesUsed(
            basic=shares_basic * 1000,    # millions → thousands (as reported on 10-Q)
            diluted=shares_diluted * 1000,
        ),
    )

    # ── Comprehensive Income ──────────────────────────────────────────────
    oci = round(net_sales_total * _r(rng, -0.01, 0.01), 0)
    comp_income = StatementOfComprehensiveIncome(
        period=period_label,
        net_income=ops.net_income,
        other_comprehensive_income=OtherComprehensiveIncome(
            foreign_currency_translation=round(oci * 0.6, 0),
            unrealized_gains_losses_on_securities=round(oci * 0.4, 0),
        ),
    )

    # ── Balance Sheet ─────────────────────────────────────────────────────
    ann = annual_revenue  # use annual for balance sheet ratios
    cash = round(ann * _r(rng, *profile.cash_ratio) / 4, 0)
    ar = round(net_sales_total * _r(rng, 0.08, 0.18), 0)
    inventory = round(ann * _r(rng, *profile.inventory_ratio) / 4, 0)
    prepaid = round(net_sales_total * _r(rng, 0.02, 0.06), 0)
    st_investments = round(cash * _r(rng, 0.3, 1.5), 0)

    ppe = round(ann * _r(rng, *profile.ppe_ratio) / 4, 0)
    goodwill = round(ann * _r(rng, *profile.goodwill_ratio) / 4, 0)
    other_nca = round(ann * _r(rng, 0.02, 0.08) / 4, 0)
    lt_investments = round(cash * _r(rng, 0.5, 2.0), 0)

    ap = round(cost_of_sales_total * _r(rng, 0.3, 0.7), 0)
    accrued = round(net_sales_total * _r(rng, 0.04, 0.10), 0)
    deferred_rev_curr = round(net_sales_total * _r(rng, 0.02, 0.08), 0)
    current_debt = round(ann * _r(rng, 0.02, 0.06) / 4, 0)

    lt_debt = round(ann * _r(rng, *profile.debt_ratio) / 4, 0)
    other_ncl = round(ann * _r(rng, 0.02, 0.08) / 4, 0)

    total_assets = (cash + ar + inventory + prepaid + st_investments +
                    ppe + goodwill + other_nca + lt_investments)
    total_liabilities = ap + accrued + deferred_rev_curr + current_debt + lt_debt + other_ncl
    total_equity = total_assets - total_liabilities

    apic = round(total_equity * _r(rng, 0.4, 0.9), 0)
    aoci = round(total_equity * _r(rng, -0.15, 0.05), 0)
    retained = total_equity - apic - aoci

    balance_sheet = BalanceSheet(
        as_of=as_of_label,
        current_assets=CurrentAssets(
            cash_and_cash_equivalents=cash,
            short_term_investments=st_investments,
            accounts_receivable_net=ar,
            inventories=inventory,
            prepaid_expenses_and_other=prepaid,
        ),
        non_current_assets=NonCurrentAssets(
            long_term_marketable_securities=lt_investments,
            property_plant_and_equipment_net=ppe,
            goodwill=goodwill,
            other_non_current_assets=other_nca,
        ),
        current_liabilities=CurrentLiabilities(
            accounts_payable=ap,
            accrued_expenses_and_other=accrued,
            deferred_revenue_current=deferred_rev_curr,
            current_portion_of_long_term_debt=current_debt,
        ),
        non_current_liabilities=NonCurrentLiabilities(
            long_term_debt=lt_debt,
            other_non_current_liabilities=other_ncl,
        ),
        shareholders_equity=ShareholdersEquity(
            common_stock_and_additional_paid_in_capital=apic,
            retained_earnings=retained,
            accumulated_other_comprehensive_income_loss=aoci,
        ),
    )

    # ── YTD label (needed for cash flows period) ─────────────────────────
    ytd_label = period_label if quarter == 1 else (
        f"Six Months Ended {quarter_end.strftime('%B %d, %Y')}" if quarter == 2
        else f"Nine Months Ended {quarter_end.strftime('%B %d, %Y')}")
    prior_ytd_label = ytd_label.replace(str(quarter_end.year), str(quarter_end.year-1))
    ops_ytd = None  # will be set in assemble block; used below for cf_net_income

    # ── Cash Flows ────────────────────────────────────────────────────────
    sbc      = round(net_sales_total * _r(rng, 0.01, 0.03), 0)
    dividends= round(-ops.net_income * _r(rng, 0.10, 0.40), 0)
    buybacks = round(-net_sales_total * _r(rng, 0.01, 0.05), 0)
    # Scale cash flow items by quarter for YTD (Q1=1x, Q2=2x, Q3=3x)
    cf_scale = quarter
    da = round(net_sales_total * _r(rng, *profile.da_ratio) * cf_scale, 0)
    capex = round(-net_sales_total * _r(rng, *profile.capex_ratio) * cf_scale, 0)
    change_ar = round(-(ar * _r(rng, 0.05, 0.20)) * cf_scale, 0)
    change_inv = round(inventory * _r(rng, -0.10, 0.10) * cf_scale, 0)
    change_ap = round(ap * _r(rng, -0.10, 0.15) * cf_scale, 0)
    opening_cash = round(cash * _r(rng, 0.7, 1.3), 0)

    sec_purchases = round(-lt_investments * _r(rng, 0.1, 0.3), 0)
    sec_maturities = round(lt_investments * _r(rng, 0.05, 0.20), 0)
    debt_repaid = round(-current_debt * _r(rng, 0.3, 0.8), 0)

    notes = Notes()

    # ── Assemble ──────────────────────────────────────────────────────────
    # Build simple prior-year comparatives by scaling current figures
    growth = _r(rng, 0.85, 1.15)
    def _scale_ops(o, factor, period_override=None):
        scaled_rev   = o.revenue.net_sales_total / factor
        scaled_cos   = o.costs_and_expenses.cost_of_sales_total / factor
        scaled_other = o.other_income_expense_net / factor
        scaled_tax_rate = o.provision_for_income_taxes / max(o.income_before_taxes, 1)

        # Preserve OperatingExpenses breakdown if available, else use total float
        orig_opex = o.costs_and_expenses.operating_expenses
        if isinstance(orig_opex, OperatingExpenses):
            pq_opex = OperatingExpenses(
                research_and_development=(
                    round(orig_opex.research_and_development / factor, 0)
                    if orig_opex.research_and_development is not None else None),
                selling_general_and_administrative=(
                    round(orig_opex.selling_general_and_administrative / factor, 0)
                    if orig_opex.selling_general_and_administrative is not None else None),
                depreciation_and_amortization=(
                    round(orig_opex.depreciation_and_amortization / factor, 0)
                    if orig_opex.depreciation_and_amortization is not None else None),
            )
        else:
            pq_opex = round(float(orig_opex) / factor, 0)

        # Scale net_sales
        pq_ns = o.revenue.net_sales
        if pq_ns is None:
            pq_net_sales = None
        elif isinstance(pq_ns, NetSales):
            pq_net_sales = NetSales(components={
                k: round(v/factor, 0) for k, v in pq_ns.components.items()
            })
        else:
            pq_net_sales = round(float(pq_ns) / factor, 0)

        # Scale cost_of_sales
        pq_cos_obj = o.costs_and_expenses.cost_of_sales
        if isinstance(pq_cos_obj, CostOfSales):
            pq_cost_of_sales = CostOfSales(components={
                k: round(v/factor, 0) for k, v in pq_cos_obj.components.items()
            })
        else:
            pq_cost_of_sales = round(float(pq_cos_obj) / factor, 0)

        pq_revenue = Revenue(
            net_sales=pq_net_sales,
            other_components={k: round(v/factor,0)
                              for k,v in o.revenue.other_components.items()},
            other_operating_revenue={k: round(v/factor,0)
                                     for k,v in o.revenue.other_operating_revenue.items()})

        pq_costs = CostsAndExpenses(
            cost_of_sales=pq_cost_of_sales,
            operating_expenses=pq_opex)

        pretax = pq_revenue.total_revenues - pq_costs.total_costs_and_expenses + scaled_other
        tax    = round(max(pretax * scaled_tax_rate, 0), 0)

        # Replace the calendar year embedded in the period string (not filing_year,
        # which may differ e.g. Apple FY2025 Q1 ends January 2026)
        if period_override:
            period = period_override
        else:
            # Find the year actually in the period string and decrement it
            years_in_period = re.findall(r'\d{4}', o.period)
            period = o.period
            if years_in_period:
                cal_year = years_in_period[-1]  # use last year found (end date)
                period = o.period.replace(cal_year, str(int(cal_year) - 1))
        return StatementOfOperations(
            period=period,
            revenue=pq_revenue, costs_and_expenses=pq_costs,
            other_income_expense_net=round(scaled_other, 0),
            provision_for_income_taxes=tax,
            earnings_per_share=EarningsPerShare(
                basic=round((pretax-tax)/shares_basic,2) if shares_basic else 0,
                diluted=round((pretax-tax)/shares_diluted,2) if shares_diluted else 0),
            shares_used_in_computing_eps=SharesUsed(
                basic=round(shares_basic*1000*_r(rng,0.95,1.05),0),
                diluted=round(shares_diluted*1000*_r(rng,0.95,1.05),0)),
        )

    ops_pq = _scale_ops(ops, growth)
    oci_pq = round(net_sales_total/_r(rng,1,10) * _r(rng,-0.01,0.01), 0)
    ci_pq = StatementOfComprehensiveIncome(
        period=ops_pq.period, net_income=ops_pq.net_income,
        other_comprehensive_income=OtherComprehensiveIncome(
            foreign_currency_translation=round(oci_pq*0.6,0),
            unrealized_gains_losses_on_securities=round(oci_pq*0.4,0)))

    # YTD versions for Q2/Q3
    ops_ytd = ops_pq_ytd = ci_ytd = ci_pq_ytd = None

    if quarter > 1:
        ops_ytd    = _scale_ops(ops,    1/quarter, period_override=ytd_label)
        ops_pq_ytd = _scale_ops(ops_pq, 1/quarter, period_override=prior_ytd_label)
        oci_ytd = round(oci_pq * quarter, 0)
        ci_ytd = StatementOfComprehensiveIncome(
            period=ytd_label, net_income=ops_ytd.net_income,
            other_comprehensive_income=OtherComprehensiveIncome(
                foreign_currency_translation=round(oci_ytd*0.6,0),
                unrealized_gains_losses_on_securities=round(oci_ytd*0.4,0)))
        ci_pq_ytd = StatementOfComprehensiveIncome(
            period=prior_ytd_label, net_income=ops_pq_ytd.net_income,
            other_comprehensive_income=OtherComprehensiveIncome(
                foreign_currency_translation=round(oci_ytd*0.6/_r(rng,0.8,1.2),0),
                unrealized_gains_losses_on_securities=round(oci_ytd*0.4/_r(rng,0.8,1.2),0)))


    # ── Cash Flows (built here so ops_ytd is available) ──────────────────
    cf_net_income = ops_ytd.net_income if (quarter > 1 and ops_ytd is not None) else ops.net_income
    cf_period     = ytd_label

    cash_flows = StatementOfCashFlows(
        period=cf_period,
        operating_activities=OperatingActivities(
            net_income=cf_net_income,
            depreciation_and_amortization=da,
            stock_based_compensation=sbc,
            change_in_accounts_receivable=change_ar,
            change_in_inventories=change_inv,
            change_in_accounts_payable=change_ap,
            change_in_other_working_capital=round(net_sales_total * _r(rng, -0.02, 0.02), 0),
        ),
        investing_activities=InvestingActivities(
            capital_expenditures=capex,
            purchases_of_marketable_securities=sec_purchases,
            proceeds_from_maturities_of_securities=sec_maturities,
        ),
        financing_activities=FinancingActivities(
            share_repurchases=buybacks,
            dividends_paid=dividends,
            repayments_of_debt=debt_repaid,
            proceeds_from_stock_option_exercises=round(sbc * _r(rng, 0.1, 0.3), 0),
        ),
        opening_cash_and_equivalents=opening_cash,
        effect_of_exchange_rate_on_cash=round(net_sales_total * _r(rng, -0.002, 0.002), 0),
    )

    # Determine whether to include optional 3rd balance sheet column
    if include_prior_year_quarter_bs is None:
        include_prior_year_quarter_bs = False  # default: omit it

    # Prior fiscal year-end balance sheet
    s = _r(rng, 0.85, 1.10)
    prior_fy_label = prior_fy_end.strftime("%B %d, %Y")
    prior_q_end_label = prior_q_end.strftime("%B %d, %Y")
    def _scale_bs(bs, factor, as_of):
        ca = bs.current_assets; nca = bs.non_current_assets
        cl = bs.current_liabilities; ncl = bs.non_current_liabilities

        # Round each component first
        s_cash   = round((ca.cash_and_cash_equivalents or 0) * factor, 0)
        s_st_inv = round((ca.short_term_investments or 0) * factor, 0)
        s_ar     = round((ca.accounts_receivable_net or 0) * factor, 0)
        s_inv    = round((ca.inventories or 0) * factor, 0)
        s_prep   = round((ca.prepaid_expenses_and_other or 0) * factor, 0)

        s_lt_inv = round((nca.long_term_marketable_securities or 0) * factor, 0)
        s_ppe    = round((nca.property_plant_and_equipment_net or 0) * factor, 0)
        s_gw     = round((nca.goodwill or 0) * factor, 0)
        s_onca   = round((nca.other_non_current_assets or 0) * factor, 0)

        s_ap     = round((cl.accounts_payable or 0) * factor, 0)
        s_ae     = round((cl.accrued_expenses_and_other or 0) * factor, 0)
        s_dr     = round((cl.deferred_revenue_current or 0) * factor, 0)
        s_cltd   = round((cl.current_portion_of_long_term_debt or 0) * factor, 0)

        s_ltd    = round((ncl.long_term_debt or 0) * factor, 0)
        s_oncl   = round((ncl.other_non_current_liabilities or 0) * factor, 0)

        # Derive totals from rounded components — guarantees identity
        s_total_ca  = s_cash + s_st_inv + s_ar + s_inv + s_prep
        s_total_nca = s_lt_inv + s_ppe + s_gw + s_onca
        s_total_assets = s_total_ca + s_total_nca

        s_total_cl  = s_ap + s_ae + s_dr + s_cltd
        s_total_ncl = s_ltd + s_oncl
        s_total_liab = s_total_cl + s_total_ncl

        # Equity is derived from assets - liabilities — guarantees balance
        s_total_eq   = s_total_assets - s_total_liab
        s_apic       = round(s_total_eq * 0.7, 0)
        s_re         = s_total_eq - s_apic  # exact remainder, no rounding

        return BalanceSheet(
            as_of=as_of,
            current_assets=CurrentAssets(
                cash_and_cash_equivalents=s_cash,
                short_term_investments=s_st_inv,
                accounts_receivable_net=s_ar,
                inventories=s_inv,
                prepaid_expenses_and_other=s_prep),
            non_current_assets=NonCurrentAssets(
                long_term_marketable_securities=s_lt_inv,
                property_plant_and_equipment_net=s_ppe,
                goodwill=s_gw,
                other_non_current_assets=s_onca),
            current_liabilities=CurrentLiabilities(
                accounts_payable=s_ap,
                accrued_expenses_and_other=s_ae,
                deferred_revenue_current=s_dr,
                current_portion_of_long_term_debt=s_cltd),
            non_current_liabilities=NonCurrentLiabilities(
                long_term_debt=s_ltd,
                other_non_current_liabilities=s_oncl),
            shareholders_equity=ShareholdersEquity(
                common_stock_and_additional_paid_in_capital=s_apic,
                retained_earnings=s_re,
                accumulated_other_comprehensive_income_loss=0),
        )

    bs_pfy = _scale_bs(balance_sheet, s, prior_fy_label)
    bs_pyq = _scale_bs(balance_sheet, _r(rng,0.80,1.05), prior_q_end_label)         if include_prior_year_quarter_bs else None

    # Prior-year cash flows
    cf_prior_ni = ops_pq_ytd.net_income if (quarter > 1 and ops_pq_ytd is not None) else ops_pq.net_income
    cf_prior = StatementOfCashFlows(
        period=prior_ytd_label,
        operating_activities=OperatingActivities(
            net_income=cf_prior_ni,
            depreciation_and_amortization=round(da/growth,0),
            stock_based_compensation=round(sbc/growth,0),
            change_in_accounts_receivable=round(change_ar/growth,0),
            change_in_inventories=round(change_inv/growth,0),
            change_in_accounts_payable=round(change_ap/growth,0),
            change_in_other_working_capital=round(net_sales_total*_r(rng,-0.02,0.02)/growth,0)),
        investing_activities=InvestingActivities(
            capital_expenditures=round(capex/growth,0),
            purchases_of_marketable_securities=round(sec_purchases/growth,0),
            proceeds_from_maturities_of_securities=round(sec_maturities/growth,0)),
        financing_activities=FinancingActivities(
            share_repurchases=round(buybacks/growth,0),
            dividends_paid=round(dividends/growth,0),
            repayments_of_debt=round(debt_repaid/growth,0),
            proceeds_from_stock_option_exercises=round(sbc*_r(rng,0.1,0.3)/growth,0)),
        opening_cash_and_equivalents=round(opening_cash/growth,0),
        effect_of_exchange_rate_on_cash=round(net_sales_total*_r(rng,-0.002,0.002),0),
    )

    oci_q = round(net_sales_total * _r(rng, -0.01, 0.01), 0)
    ci_q = StatementOfComprehensiveIncome(
        period=period_label, net_income=ops.net_income,
        other_comprehensive_income=OtherComprehensiveIncome(
            foreign_currency_translation=round(oci_q*0.6,0),
            unrealized_gains_losses_on_securities=round(oci_q*0.4,0)))

    return FinancialReport(
        financial_statements=FinancialStatements(
            ops_current_quarter=ops,
            ops_prior_year_quarter=ops_pq,
            ops_current_ytd=ops_ytd,
            ops_prior_year_ytd=ops_pq_ytd,
            ci_current_quarter=ci_q,
            ci_prior_year_quarter=ci_pq,
            ci_current_ytd=ci_ytd,
            ci_prior_year_ytd=ci_pq_ytd,
            bs_current=balance_sheet,
            bs_prior_fy_end=bs_pfy,
            bs_prior_year_quarter=bs_pyq,
            cf_current_ytd=cash_flows,
            cf_prior_ytd=cf_prior,
        ),
        notes=notes,
        company_name=company_name,
        ticker=ticker,
        period=period_label,
        fiscal_year=str(filing_year),
        quarter=quarter,
        fiscal_year_start=fiscal_year_start,
        form_type="10-Q",
    )


# ---------------------------------------------------------------------------
# Multi-column printer
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], width: int = 14) -> str:
    if v is None:
        return " " * (width - 1) + "—"
    return f"{v:>{width},.0f}"

def _row(label: str, *values, indent: int = 2) -> None:
    pad = " " * indent
    cols = "".join(_fmt(v) for v in values)
    print(f"{pad}{label:<42}{cols}")

def _section(title: str) -> None:
    print(f"\n  {'─' * 70}")
    print(f"  {title}")
    print(f"  {'─' * 70}")

def _header(*labels, col_width: int = 14) -> None:
    pad = " " * 44
    cols = "".join(f"{l:>{col_width}}" for l in labels)
    print(f"{pad}{cols}")
    print(f"  {'─' * (42 + col_width * len(labels))}")

def _short_date(period: str) -> str:
    """Extract a short readable date label from a full period string.
    e.g. "Three Months Ended January 31, 2026" -> "Jan 31, 2026"
         "Nine Months Ended November 30, 2025" -> "Nov 30, 2025"
    """
    # Period strings end with the date: "... Month DD, YYYY"
    # Find the last occurrence of a month name
    months = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
    abbr   = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    for i, m in enumerate(months):
        if m in period:
            idx = period.index(m)
            date_part = period[idx:]          # e.g. "January 31, 2026"
            parts = date_part.split()
            if len(parts) >= 3:
                return f"{abbr[i]} {parts[1]} {parts[2]}"
    return period[-12:]


def print_report(report: FinancialReport, show_header: bool = True) -> None:
    fs = report.financial_statements
    q  = report.quarter
    is_q1 = (q == 1)

    W = 80
    if show_header:
        print("━" * W)
        print(f"  {report.company_name}  |  {report.form_type}  |  FY{report.fiscal_year}  |  Q{q}")
        print(f"  Current quarter: {report.period}")
        print(f"  Fiscal year start: {report.fiscal_year_start}")
        print(f"  (Amounts in millions, except per-share data)")
        print("━" * W)

    # ── Income Statement ────────────────────────────────────────────────
    _section("CONDENSED CONSOLIDATED STATEMENT OF OPERATIONS")
    ops_cq  = fs.ops_current_quarter
    ops_pq  = fs.ops_prior_year_quarter
    ops_cy  = fs.ops_current_ytd
    ops_py  = fs.ops_prior_year_ytd

    if is_q1:
        _header(_short_date(ops_cq.period), _short_date(ops_pq.period))
    else:
        # For 4 columns use "Mon YYYY" to keep headers compact
        def _moy(p): return _short_date(p)[:3] + " " + _short_date(p)[-4:]
        _header("Q "+_moy(ops_cq.period), "Q "+_moy(ops_pq.period),
                "YTD "+_moy(ops_cy.period), "YTD "+_moy(ops_py.period))

    def ops_row(label, fn, indent=2):
        cq = fn(ops_cq); pq = fn(ops_pq)
        if is_q1:
            _row(label, cq, pq, indent=indent)
        else:
            _row(label, cq, pq, fn(ops_cy), fn(ops_py), indent=indent)

    print("  Revenues:")
    # net_sales — label, indented components, indented total
    if ops_cq.revenue.net_sales is not None:
        if isinstance(ops_cq.revenue.net_sales, NetSales):
            print("    Net sales:")
            for lbl in ops_cq.revenue.net_sales.components:
                ops_row(lbl, lambda o, l=lbl: o.revenue.net_sales.components.get(l, 0)
                        if isinstance(o.revenue.net_sales, NetSales) else None, indent=6)
            ops_row("Total net sales", lambda o: o.revenue.net_sales_total, indent=8)
        else:
            ops_row("Net sales", lambda o: o.revenue.net_sales_total, indent=4)
    # other_components — primary revenue lines for companies without net_sales
    for lbl in ops_cq.revenue.other_components:
        ops_row(lbl, lambda o, l=lbl: o.revenue.other_components.get(l, 0), indent=4)
    # other_operating_revenue — ancillary lines (membership fees etc.)
    for lbl in ops_cq.revenue.other_operating_revenue:
        ops_row(lbl, lambda o, l=lbl: o.revenue.other_operating_revenue.get(l, 0), indent=4)
    ops_row("Total revenues", lambda o: o.revenue.total_revenues, indent=4)

    print("  Costs and expenses:")
    cq_cos = ops_cq.costs_and_expenses.cost_of_sales
    if isinstance(cq_cos, CostOfSales) and cq_cos.components:
        print("    Cost of sales:")
        for lbl in cq_cos.components:
            ops_row(lbl, lambda o, l=lbl: o.costs_and_expenses.cost_of_sales.components.get(l, 0)
                    if isinstance(o.costs_and_expenses.cost_of_sales, CostOfSales) else None, indent=6)
        ops_row("Total cost of sales", lambda o: o.costs_and_expenses.cost_of_sales_total, indent=8)
    else:
        ops_row("Cost of sales", lambda o: o.costs_and_expenses.cost_of_sales_total, indent=4)
    cq_opex = ops_cq.costs_and_expenses.operating_expenses
    if hasattr(cq_opex, "research_and_development") and cq_opex.research_and_development:
        ops_row("Research and development",
                lambda o: o.costs_and_expenses.operating_expenses.research_and_development
                          if hasattr(o.costs_and_expenses.operating_expenses, "research_and_development")
                          else None, indent=4)
    if hasattr(cq_opex, "selling_general_and_administrative") and cq_opex.selling_general_and_administrative:
        ops_row("Selling, general and administrative",
                lambda o: o.costs_and_expenses.operating_expenses.selling_general_and_administrative
                          if hasattr(o.costs_and_expenses.operating_expenses, "selling_general_and_administrative")
                          else None, indent=4)
    ops_row("Total costs and expenses", lambda o: o.costs_and_expenses.total_costs_and_expenses, indent=4)

    print()
    # Only print Gross profit if net_sales exists
    if ops_cq.gross_profit is not None:
        ops_row("Gross profit", lambda o: o.gross_profit)
    ops_row("Operating income", lambda o: o.operating_income)
    ops_row("Other income/(expense), net", lambda o: o.other_income_expense_net)
    ops_row("Income before taxes", lambda o: o.income_before_taxes)
    ops_row("Provision for income taxes", lambda o: o.provision_for_income_taxes)
    ops_row("Net income", lambda o: o.net_income)
    print()
    def _eps_row(label, fn):
        """EPS rows use 2 decimal places instead of integer formatting."""
        vals = [fn(ops_cq), fn(ops_pq),
                fn(ops_cy) if ops_cy else None,
                fn(ops_py) if ops_py else None]
        pad = " " * 6
        cols = "".join(
            f"{v:>14.2f}" if v is not None else " " * 13 + "—"
            for v in vals[:2 if is_q1 else 4]
        )
        print(f"{pad}{label:<38}{cols}")

    print("  Earnings per share:")
    _eps_row("Basic", lambda o: o.earnings_per_share.basic)
    _eps_row("Diluted", lambda o: o.earnings_per_share.diluted)
    print()
    def _shares_row(label, fn):
        """Shares rows use integer formatting."""
        vals = [fn(ops_cq), fn(ops_pq),
                fn(ops_cy) if ops_cy else None,
                fn(ops_py) if ops_py else None]
        pad = " " * 6
        cols = "".join(
            f"{v:>14,.0f}" if v is not None else " " * 13 + "—"
            for v in vals[:2 if is_q1 else 4]
        )
        print(f"{pad}{label:<38}{cols}")

    print("  Shares used in computing earnings per share:")
    _shares_row("Basic", lambda o: o.shares_used_in_computing_eps.basic)
    _shares_row("Diluted", lambda o: o.shares_used_in_computing_eps.diluted)

    # ── Comprehensive Income ────────────────────────────────────────────
    _section("CONDENSED CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME")
    ci_cq = fs.ci_current_quarter; ci_pq = fs.ci_prior_year_quarter
    ci_cy = fs.ci_current_ytd;     ci_py = fs.ci_prior_year_ytd

    if is_q1:
        _header(_short_date(ci_cq.period), _short_date(ci_pq.period))
    else:
        def _moy(p): return _short_date(p)[:3] + " " + _short_date(p)[-4:]
        _header("Q "+_moy(ci_cq.period), "Q "+_moy(ci_pq.period),
                "YTD "+_moy(ci_cy.period), "YTD "+_moy(ci_py.period))

    def ci_row(label, fn, indent=2):
        cq = fn(ci_cq); pq = fn(ci_pq)
        if is_q1:
            _row(label, cq, pq, indent=indent)
        else:
            _row(label, cq, pq, fn(ci_cy), fn(ci_py), indent=indent)

    ci_row("Net income", lambda c: c.net_income)
    print("  Other comprehensive income/(loss):")
    ci_row("Change in foreign currency translation, net of tax",
           lambda c: c.other_comprehensive_income.foreign_currency_translation, indent=4)
    ci_row("Change in unrealized gains/(losses) on securities, net of tax",
           lambda c: c.other_comprehensive_income.unrealized_gains_losses_on_securities, indent=4)
    ci_row("Total other comprehensive income/(loss)",
           lambda c: c.total_other_comprehensive_income, indent=6)
    print()
    ci_row("Total comprehensive income", lambda c: c.comprehensive_income)

    # ── Balance Sheet ───────────────────────────────────────────────────
    _section("CONDENSED CONSOLIDATED BALANCE SHEET")
    bc = fs.bs_current; bp = fs.bs_prior_fy_end; bpq = fs.bs_prior_year_quarter

    if bpq is not None:
        _header(_short_date(bc.as_of), _short_date(bp.as_of), _short_date(bpq.as_of))
    else:
        _header(_short_date(bc.as_of), _short_date(bp.as_of))

    def bs_row(label, fn, indent=2):
        if bpq is not None:
            _row(label, fn(bc), fn(bp), fn(bpq), indent=indent)
        else:
            _row(label, fn(bc), fn(bp), indent=indent)

    print("  Current assets:")
    if bc.current_assets.cash_and_cash_equivalents:
        bs_row("Cash and cash equivalents", lambda b: b.current_assets.cash_and_cash_equivalents, indent=4)
    if bc.current_assets.short_term_investments:
        bs_row("Short-term investments", lambda b: b.current_assets.short_term_investments, indent=4)
    if bc.current_assets.accounts_receivable_net:
        bs_row("Accounts receivable, net", lambda b: b.current_assets.accounts_receivable_net, indent=4)
    if bc.current_assets.inventories:
        bs_row("Inventories", lambda b: b.current_assets.inventories, indent=4)
    if bc.current_assets.prepaid_expenses_and_other:
        bs_row("Prepaid expenses and other", lambda b: b.current_assets.prepaid_expenses_and_other, indent=4)
    bs_row("Total current assets", lambda b: b.total_current_assets, indent=4)

    print("  Non-current assets:")
    if bc.non_current_assets.long_term_marketable_securities:
        bs_row("Long-term marketable securities", lambda b: b.non_current_assets.long_term_marketable_securities, indent=4)
    if bc.non_current_assets.property_plant_and_equipment_net:
        bs_row("Property, plant and equipment, net", lambda b: b.non_current_assets.property_plant_and_equipment_net, indent=4)
    if bc.non_current_assets.goodwill:
        bs_row("Goodwill", lambda b: b.non_current_assets.goodwill, indent=4)
    if bc.non_current_assets.other_non_current_assets:
        bs_row("Other non-current assets", lambda b: b.non_current_assets.other_non_current_assets, indent=4)
    bs_row("Total non-current assets", lambda b: b.total_non_current_assets, indent=4)
    bs_row("Total assets", lambda b: b.total_assets)

    print("  Current liabilities:")
    if bc.current_liabilities.accounts_payable:
        bs_row("Accounts payable", lambda b: b.current_liabilities.accounts_payable, indent=4)
    if bc.current_liabilities.deferred_revenue_current:
        bs_row("Deferred revenue", lambda b: b.current_liabilities.deferred_revenue_current, indent=4)
    if bc.current_liabilities.accrued_expenses_and_other:
        bs_row("Accrued expenses and other", lambda b: b.current_liabilities.accrued_expenses_and_other, indent=4)
    if bc.current_liabilities.current_portion_of_long_term_debt:
        bs_row("Current portion of long-term debt", lambda b: b.current_liabilities.current_portion_of_long_term_debt, indent=4)
    bs_row("Total current liabilities", lambda b: b.total_current_liabilities, indent=4)

    print("  Non-current liabilities:")
    if bc.non_current_liabilities.long_term_debt:
        bs_row("Long-term debt", lambda b: b.non_current_liabilities.long_term_debt, indent=4)
    if bc.non_current_liabilities.other_non_current_liabilities:
        bs_row("Other non-current liabilities", lambda b: b.non_current_liabilities.other_non_current_liabilities, indent=4)
    bs_row("Total non-current liabilities", lambda b: b.total_non_current_liabilities, indent=4)
    bs_row("Total liabilities", lambda b: b.total_liabilities)

    print("  Shareholders equity:")
    if bc.shareholders_equity.common_stock_and_additional_paid_in_capital:
        bs_row("Common stock and APIC", lambda b: b.shareholders_equity.common_stock_and_additional_paid_in_capital, indent=4)
    if bc.shareholders_equity.retained_earnings is not None:
        bs_row("Retained earnings/(accumulated deficit)", lambda b: b.shareholders_equity.retained_earnings, indent=4)
    if bc.shareholders_equity.accumulated_other_comprehensive_income_loss:
        bs_row("Accumulated other comprehensive loss", lambda b: b.shareholders_equity.accumulated_other_comprehensive_income_loss, indent=4)
    bs_row("Total shareholders equity", lambda b: b.total_shareholders_equity, indent=4)
    bs_row("Total liabilities and equity", lambda b: b.total_liabilities_and_equity)



    # ── Cash Flows ──────────────────────────────────────────────────────
    _section("CONDENSED CONSOLIDATED STATEMENT OF CASH FLOWS")
    cf_c = fs.cf_current_ytd; cf_p = fs.cf_prior_ytd
    def _cf_header_label(period: str) -> str:
        if period.startswith("Three Months"):
            return _short_date(period)
        return "YTD " + _short_date(period)
    _header(_cf_header_label(cf_c.period), _cf_header_label(cf_p.period), col_width=18)

    def cf_row(label, fn, indent=2):
        _row(label, fn(cf_c), fn(cf_p), indent=indent)

    print("  Operating activities:")
    cf_row("Net income", lambda c: c.operating_activities.net_income, indent=4)
    cf_row("Depreciation and amortization", lambda c: c.operating_activities.depreciation_and_amortization, indent=4)
    cf_row("Stock-based compensation", lambda c: c.operating_activities.stock_based_compensation, indent=4)
    cf_row("Change in accounts receivable", lambda c: c.operating_activities.change_in_accounts_receivable, indent=4)
    cf_row("Change in inventories", lambda c: c.operating_activities.change_in_inventories, indent=4)
    cf_row("Change in accounts payable", lambda c: c.operating_activities.change_in_accounts_payable, indent=4)
    if cf_c.operating_activities.change_in_deferred_revenue:
        cf_row("Change in deferred revenue", lambda c: c.operating_activities.change_in_deferred_revenue, indent=4)
    if cf_c.operating_activities.change_in_other_working_capital:
        cf_row("Change in other working capital", lambda c: c.operating_activities.change_in_other_working_capital, indent=4)
    for lbl in cf_c.operating_activities.other_lines:
        cf_row(lbl, lambda c, l=lbl: c.operating_activities.other_lines.get(l, 0), indent=4)
    cf_row("Net cash from operating", lambda c: c.net_cash_from_operating, indent=4)

    print("  Investing activities:")
    cf_row("Capital expenditures", lambda c: c.investing_activities.capital_expenditures, indent=4)
    cf_row("Purchases of marketable securities", lambda c: c.investing_activities.purchases_of_marketable_securities, indent=4)
    cf_row("Proceeds from maturities", lambda c: c.investing_activities.proceeds_from_maturities_of_securities, indent=4)
    cf_row("Net cash from investing", lambda c: c.net_cash_from_investing, indent=4)

    print("  Financing activities:")
    if cf_c.financing_activities.proceeds_from_issuance_of_debt:
        cf_row("Proceeds from issuance of debt", lambda c: c.financing_activities.proceeds_from_issuance_of_debt, indent=4)
    if cf_c.financing_activities.repayments_of_debt:
        cf_row("Repayments of debt", lambda c: c.financing_activities.repayments_of_debt, indent=4)
    if cf_c.financing_activities.share_repurchases:
        cf_row("Share repurchases", lambda c: c.financing_activities.share_repurchases, indent=4)
    if cf_c.financing_activities.proceeds_from_stock_option_exercises:
        cf_row("Proceeds from stock option exercises", lambda c: c.financing_activities.proceeds_from_stock_option_exercises, indent=4)
    if cf_c.financing_activities.dividends_paid:
        cf_row("Dividends paid", lambda c: c.financing_activities.dividends_paid, indent=4)
    for lbl in cf_c.financing_activities.other_lines:
        cf_row(lbl, lambda c, l=lbl: c.financing_activities.other_lines.get(l, 0), indent=4)
    cf_row("Net cash from financing", lambda c: c.net_cash_from_financing, indent=4)

    print()
    cf_row("Effect of exchange rate on cash", lambda c: c.effect_of_exchange_rate_on_cash)
    cf_row("Net change in cash", lambda c: c.net_change_in_cash)
    cf_row("Cash at beginning of period", lambda c: c.opening_cash_and_equivalents)
    cf_row("Cash at end of period", lambda c: c.closing_cash_and_equivalents)

    print("\n" + "━" * W)


def _center(text: str, width: int = 80) -> str:
    return text.center(width)

def _title_page(report: FinancialReport) -> list[str]:
    """Returns lines for a realistic SEC-style 10-Q cover page."""
    import random
    W = 80
    lines = []

    # SEC header
    lines.append(_center("UNITED STATES", W))
    lines.append(_center("SECURITIES AND EXCHANGE COMMISSION", W))
    lines.append(_center("Washington, D.C. 20549", W))
    lines.append("")
    lines.append(_center("FORM 10-Q", W))
    lines.append("")
    lines.append("(Mark One)")
    lines.append("")

    # Checkboxes
    period_end = report.period.replace("Three Months Ended ", "").replace(
                 "Six Months Ended ", "").replace("Nine Months Ended ", "")
    lines.append("[X] QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE")
    lines.append("    SECURITIES EXCHANGE ACT OF 1934")
    lines.append("")
    lines.append(f"    For the quarterly period ended {period_end}.")
    lines.append("")
    lines.append("[ ] TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE")
    lines.append("    SECURITIES EXCHANGE ACT OF 1934")
    lines.append("")
    lines.append(f"    For the transition period from ______ to ______.")
    lines.append("")

    # Commission file number (fictional)
    rng = random.Random(hash(report.company_name))
    file_no = f"{rng.randint(0,9)}-{rng.randint(10000,99999)}"
    lines.append(_center(f"Commission File Number {file_no}", W))
    lines.append("")

    # Company name (large, centered)
    lines.append(_center(report.company_name, W))
    lines.append(_center("(Exact name of registrant as specified in its charter)", W))
    lines.append("")

    # State of incorporation and EIN (fictional)
    states = ["Delaware", "Nevada", "California", "New York", "Texas",
              "Maryland", "Ohio", "Illinois"]
    state = states[rng.randint(0, len(states)-1)]
    ein = f"{rng.randint(10,99)}-{rng.randint(1000000,9999999)}"
    left  = f"{state}"
    right = f"{ein}"
    mid   = " " * (W - len(left) - len(right))
    lines.append(left + mid + right)
    lines.append("(State or other jurisdiction of incorporation or organization)" +
                 " " * 10 + "(I.R.S. Employer Identification No.)")
    lines.append("")

    # Address (fictional)
    streets = ["1 Corporate Drive", "100 Main Street", "500 Technology Way",
               "200 Commerce Blvd", "One Financial Center", "10 Industrial Park"]
    cities  = [("New York, NY", "10001"), ("Chicago, IL", "60601"),
               ("Houston, TX", "77001"), ("Atlanta, GA", "30301"),
               ("Boston, MA", "02101"), ("Seattle, WA", "98101"),
               ("Dallas, TX", "75201"), ("Los Angeles, CA", "90001")]
    street = streets[rng.randint(0, len(streets)-1)]
    city, zipcode = cities[rng.randint(0, len(cities)-1)]
    left  = street
    right = zipcode
    mid   = " " * (W - len(left) - len(right))
    lines.append(left + mid + right)
    lines.append(city)
    lines.append("(Address of principal executive offices)" + " " * 25 + "(Zip Code)")
    lines.append("")

    # Phone
    area = rng.randint(200, 999)
    phone = f"({area}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"
    lines.append(f"Registrant's telephone number, including area code: {phone}")
    lines.append("")
    lines.append("Former name, former address and former fiscal year, if changed since last report: N/A")
    lines.append("")

    # Securities registered
    lines.append("Securities registered pursuant to Section 12(b) of the Act:")
    lines.append("")
    lines.append(f"  {'Title of each class':<35} {'Trading Symbol':<15} {'Exchange'}")
    lines.append(f"  {'-'*33} {'-'*13} {'-'*25}")
    ticker = report.company_name.split()[0][:4].upper()
    lines.append(f"  {'Common Stock, par value $0.01 per share':<35} {ticker:<15} {'New York Stock Exchange'}")
    lines.append("")

    # Filer status checkboxes
    lines.append("Indicate by check mark whether the registrant is a large accelerated filer,")
    lines.append("an accelerated filer, a non-accelerated filer, a smaller reporting company,")
    lines.append("or an emerging growth company.")
    lines.append("")
    lines.append("  Large Accelerated Filer  [X]    Accelerated Filer          [ ]")
    lines.append("  Non-Accelerated Filer    [ ]    Smaller Reporting Company  [ ]")
    lines.append("                                  Emerging Growth Company    [ ]")
    lines.append("")

    # Shares outstanding — actual count as of period end, distinct from
    # weighted average used in EPS. Slightly different due to buybacks/issuances.
    ops = report.financial_statements.ops_current_quarter
    weighted_avg = ops.shares_used_in_computing_eps.basic
    # Actual shares outstanding differs from weighted average by a small amount
    import random as _random
    _rng = _random.Random(hash(report.company_name + report.period))
    actual_shares = int(round(weighted_avg * _rng.uniform(0.97, 1.03), -3))
    lines.append(f"The registrant had {actual_shares:,} shares of common stock outstanding")
    lines.append(f"as of {period_end}.")
    lines.append("")
    lines.append("=" * W)
    lines.append("")
    lines.append("PART I — FINANCIAL INFORMATION")
    lines.append("")
    lines.append("Item 1.  Financial Statements (Unaudited)")
    lines.append("")
    lines.append("=" * W)
    return lines


def save_report(report: FinancialReport, filename: str = None) -> str:
    """
    Save the report to a .txt file by capturing print_report output.
    Returns the filename written.
    """
    import io, sys

    if filename is None:
        safe_name = report.company_name.replace(" ", "_").lower()
        filename = f"{safe_name}_Q{report.quarter}_{report.fiscal_year}.txt"

    # Capture print_report output (suppress header — title page already has it)
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_report(report, show_header=False)
    sys.stdout = old_stdout
    report_text = buf.getvalue()

    # Build title page
    title_lines = _title_page(report)
    title_text = "\n".join(title_lines) + "\n\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(title_text)
        f.write(report_text)

    return filename


def _ops_to_dict(o) -> dict:
    """Serialize a StatementOfOperations to a flat dict."""
    if o is None:
        return None
    ns = o.revenue.net_sales
    return {
        "period": o.period,
        "revenue": {
            "net_sales": (
                {"components": ns.components, "total": ns.total}
                if isinstance(ns, NetSales)
                else ns
            ),
            "other_components": dict(o.revenue.other_components),
            "other_operating_revenue": dict(o.revenue.other_operating_revenue),
            "total_revenues": o.revenue.total_revenues,
        },
        "costs_and_expenses": {
            "cost_of_sales": (
                {"components": o.costs_and_expenses.cost_of_sales.components,
                 "total": o.costs_and_expenses.cost_of_sales.total}
                if isinstance(o.costs_and_expenses.cost_of_sales, CostOfSales)
                else o.costs_and_expenses.cost_of_sales
            ),
            "operating_expenses": {
                "research_and_development": (
                    o.costs_and_expenses.operating_expenses.research_and_development
                    if isinstance(o.costs_and_expenses.operating_expenses, OperatingExpenses)
                    else None
                ),
                "selling_general_and_administrative": (
                    o.costs_and_expenses.operating_expenses.selling_general_and_administrative
                    if isinstance(o.costs_and_expenses.operating_expenses, OperatingExpenses)
                    else None
                ),
                "total": o.costs_and_expenses.operating_expenses_total,
            },
            "total_costs_and_expenses": o.costs_and_expenses.total_costs_and_expenses,
        },
        "gross_profit": o.gross_profit,
        "operating_income": o.operating_income,
        "other_income_expense_net": o.other_income_expense_net,
        "income_before_taxes": o.income_before_taxes,
        "provision_for_income_taxes": o.provision_for_income_taxes,
        "net_income": o.net_income,
        "earnings_per_share": {
            "basic": o.earnings_per_share.basic if o.earnings_per_share else None,
            "diluted": o.earnings_per_share.diluted if o.earnings_per_share else None,
        },
        "shares_used_in_computing_eps": {
            "basic": o.shares_used_in_computing_eps.basic if o.shares_used_in_computing_eps else None,
            "diluted": o.shares_used_in_computing_eps.diluted if o.shares_used_in_computing_eps else None,
        },
    }


def _ci_to_dict(c) -> dict:
    if c is None:
        return None
    return {
        "period": c.period,
        "net_income": c.net_income,
        "other_comprehensive_income": {
            "foreign_currency_translation": c.other_comprehensive_income.foreign_currency_translation,
            "unrealized_gains_losses_on_securities": c.other_comprehensive_income.unrealized_gains_losses_on_securities,
            "total": c.total_other_comprehensive_income,
        },
        "total_comprehensive_income": c.comprehensive_income,
    }


def _bs_to_dict(b) -> dict:
    if b is None:
        return None
    ca = b.current_assets; nca = b.non_current_assets
    cl = b.current_liabilities; ncl = b.non_current_liabilities
    eq = b.shareholders_equity
    return {
        "as_of": b.as_of,
        "current_assets": {
            "cash_and_cash_equivalents": ca.cash_and_cash_equivalents,
            "short_term_investments": ca.short_term_investments,
            "accounts_receivable_net": ca.accounts_receivable_net,
            "inventories": ca.inventories,
            "prepaid_expenses_and_other": ca.prepaid_expenses_and_other,
            "total": b.total_current_assets,
        },
        "non_current_assets": {
            "long_term_marketable_securities": nca.long_term_marketable_securities,
            "property_plant_and_equipment_net": nca.property_plant_and_equipment_net,
            "goodwill": nca.goodwill,
            "other_non_current_assets": nca.other_non_current_assets,
            "total": b.total_non_current_assets,
        },
        "total_assets": b.total_assets,
        "current_liabilities": {
            "accounts_payable": cl.accounts_payable,
            "deferred_revenue": cl.deferred_revenue_current,
            "accrued_expenses_and_other": cl.accrued_expenses_and_other,
            "current_portion_of_long_term_debt": cl.current_portion_of_long_term_debt,
            "total": b.total_current_liabilities,
        },
        "non_current_liabilities": {
            "long_term_debt": ncl.long_term_debt,
            "other_non_current_liabilities": ncl.other_non_current_liabilities,
            "total": b.total_non_current_liabilities,
        },
        "total_liabilities": b.total_liabilities,
        "shareholders_equity": {
            "common_stock_and_apic": eq.common_stock_and_additional_paid_in_capital,
            "retained_earnings": eq.retained_earnings,
            "accumulated_other_comprehensive_loss": eq.accumulated_other_comprehensive_income_loss,
            "total": b.total_shareholders_equity,
        },
        "total_liabilities_and_equity": b.total_liabilities_and_equity,
    }


def _cf_to_dict(c) -> dict:
    if c is None:
        return None
    oa = c.operating_activities; ia = c.investing_activities; fa = c.financing_activities
    return {
        "period": c.period,
        "operating_activities": {
            "net_income": oa.net_income,
            "depreciation_and_amortization": oa.depreciation_and_amortization,
            "stock_based_compensation": oa.stock_based_compensation,
            "change_in_accounts_receivable": oa.change_in_accounts_receivable,
            "change_in_inventories": oa.change_in_inventories,
            "change_in_accounts_payable": oa.change_in_accounts_payable,
            "change_in_other_working_capital": oa.change_in_other_working_capital,
            "total": c.net_cash_from_operating,
        },
        "investing_activities": {
            "capital_expenditures": ia.capital_expenditures,
            "purchases_of_marketable_securities": ia.purchases_of_marketable_securities,
            "proceeds_from_maturities": ia.proceeds_from_maturities_of_securities,
            "total": c.net_cash_from_investing,
        },
        "financing_activities": {
            "share_repurchases": fa.share_repurchases,
            "dividends_paid": fa.dividends_paid,
            "repayments_of_debt": fa.repayments_of_debt,
            "proceeds_from_stock_option_exercises": fa.proceeds_from_stock_option_exercises,
            "total": c.net_cash_from_financing,
        },
        "effect_of_exchange_rate_on_cash": c.effect_of_exchange_rate_on_cash,
        "net_change_in_cash": c.net_change_in_cash,
        "opening_cash": c.opening_cash_and_equivalents,
        "closing_cash": c.closing_cash_and_equivalents,
    }


def report_to_dict(report: FinancialReport) -> dict:
    """Serialize a FinancialReport to a JSON-serializable dict."""
    fs = report.financial_statements
    return {
        "company_name": report.company_name,
        "form_type": report.form_type,
        "fiscal_year": report.fiscal_year,
        "quarter": report.quarter,
        "period": report.period,
        "fiscal_year_start": report.fiscal_year_start,
        "financial_statements": {
            "income_statement": {
                "current_quarter": _ops_to_dict(fs.ops_current_quarter),
                "prior_year_quarter": _ops_to_dict(fs.ops_prior_year_quarter),
                "current_ytd": _ops_to_dict(fs.ops_current_ytd),
                "prior_year_ytd": _ops_to_dict(fs.ops_prior_year_ytd),
            },
            "comprehensive_income": {
                "current_quarter": _ci_to_dict(fs.ci_current_quarter),
                "prior_year_quarter": _ci_to_dict(fs.ci_prior_year_quarter),
                "current_ytd": _ci_to_dict(fs.ci_current_ytd),
                "prior_year_ytd": _ci_to_dict(fs.ci_prior_year_ytd),
            },
            "balance_sheet": {
                "current": _bs_to_dict(fs.bs_current),
                "prior_fiscal_year_end": _bs_to_dict(fs.bs_prior_fy_end),
                "prior_year_quarter": _bs_to_dict(fs.bs_prior_year_quarter),
            },
            "cash_flows": {
                "current_ytd": _cf_to_dict(fs.cf_current_ytd),
                "prior_ytd": _cf_to_dict(fs.cf_prior_ytd),
            },
        },
    }


def save_report_json(report: FinancialReport, filename: str = None) -> str:
    """
    Save the report as a JSON file.
    Returns the filename written.
    """
    import json
    if filename is None:
        safe_name = report.company_name.replace(" ", "_").lower()
        filename = f"{safe_name}_Q{report.quarter}_{report.fiscal_year}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(report_to_dict(report), f, indent=2)

    return filename


if __name__ == "__main__":
    print("── Tech company, Q1, seed=42 ───────────────────────────────")
    r1 = generate_report(industry="tech", quarter=1, seed=42)
    print_report(r1)
    path1 = save_report(r1)
    json1 = save_report_json(r1)
    print(f"  → Saved to: {path1}, {json1}")

    print("\n\n── Retail company, Q3, seed=7 ──────────────────────────────")
    r2 = generate_report(industry="retail", quarter=3, seed=7)
    print_report(r2)
    path2 = save_report(r2)
    json2 = save_report_json(r2)
    print(f"  → Saved to: {path2}, {json2}")