"""
1. Make a very good template from the balance sheet, let's say the 10K Balance Sheet
2. Make a template so that we can easily read some variables from the balance sheet
3. List clear allowed operator like get_value(), sum(), diff(x,y) make lots of variational questions from that
4. generate many variational questions by using that function, but because it's still using computer language, we need to transform it into economical languange
5. translate it into economical language using a very good compiler

Final step by step : 

a. Structured template → Extract a canonical schema from 10-K balance sheets
b. Variable binding → Map line items to readable variables (e.g., total_assets, current_liabilities)
c. Formal query language → Define operators like get_value(total_assets, 2024), diff(revenue, 2024, 2023), ratio(net_income, total_equity) to compose questions programmatically
d. Combinatorial generation → Automatically combine operators + variables to produce thousands of question-answer pairs with verifiable ground truth (since the answer is computed deterministically)
e. Natural language compilation → Transform the formal expressions into fluent, natural-sounding economics/finance questions

goals : verifiable ground truth and high quality economics dataset

"""

"""
Synthetic Financial QA Dataset Generator
=========================================
Generates verifiable question-answer pairs from structured 10-K financial statement templates.

Pipeline:
  1. Load template schema
  2. Generate realistic synthetic financial data for a fictional company
  3. Compose formal queries (operator expressions) at configurable difficulty
  4. Execute formal queries to get deterministic ground truth answers
  5. Compile formal queries into natural language questions
  6. Output verified (question, formal_program, answer, metadata) tuples
"""

import json
import random
import math
import csv
import os
import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from datetime import datetime


# ============================================================
# 1. DATA GENERATION — Realistic Synthetic Financial Data
# ============================================================

class FinancialDataGenerator:
    """Generates a realistic, internally-consistent set of financial statements."""

    # Industry profiles control the shape of the generated data
    INDUSTRY_PROFILES = {
        "tech_large": {
            "label": "Large-cap Technology",
            "revenue_range": (80000, 400000),
            "gross_margin": (0.55, 0.75),
            "operating_margin": (0.20, 0.40),
            "rd_pct": (0.10, 0.25),
            "sga_pct": (0.08, 0.18),
            "tax_rate": (0.12, 0.22),
            "debt_to_equity": (0.3, 1.5),
            "current_ratio": (1.2, 3.0),
            "capex_pct": (0.04, 0.12),
            "dividend_pct": (0.0, 0.30),
            "buyback_pct": (0.0, 0.50),
            "depreciation_pct": (0.03, 0.08),
            "goodwill_pct": (0.0, 0.25),
            "cash_pct_assets": (0.10, 0.35),
        },
        "industrial": {
            "label": "Industrial / Manufacturing",
            "revenue_range": (30000, 150000),
            "gross_margin": (0.25, 0.45),
            "operating_margin": (0.08, 0.18),
            "rd_pct": (0.02, 0.06),
            "sga_pct": (0.08, 0.15),
            "tax_rate": (0.18, 0.26),
            "debt_to_equity": (0.5, 2.5),
            "current_ratio": (1.0, 2.0),
            "capex_pct": (0.04, 0.10),
            "dividend_pct": (0.20, 0.50),
            "buyback_pct": (0.0, 0.20),
            "depreciation_pct": (0.04, 0.10),
            "goodwill_pct": (0.05, 0.30),
            "cash_pct_assets": (0.03, 0.12),
        },
        "consumer": {
            "label": "Consumer Goods / Retail",
            "revenue_range": (50000, 500000),
            "gross_margin": (0.20, 0.55),
            "operating_margin": (0.03, 0.15),
            "rd_pct": (0.00, 0.03),
            "sga_pct": (0.12, 0.25),
            "tax_rate": (0.20, 0.28),
            "debt_to_equity": (0.5, 3.0),
            "current_ratio": (0.8, 1.8),
            "capex_pct": (0.02, 0.06),
            "dividend_pct": (0.25, 0.60),
            "buyback_pct": (0.05, 0.30),
            "depreciation_pct": (0.02, 0.06),
            "goodwill_pct": (0.05, 0.35),
            "cash_pct_assets": (0.02, 0.10),
        },
        "healthcare": {
            "label": "Healthcare / Pharma",
            "revenue_range": (20000, 200000),
            "gross_margin": (0.55, 0.80),
            "operating_margin": (0.15, 0.35),
            "rd_pct": (0.12, 0.25),
            "sga_pct": (0.15, 0.28),
            "tax_rate": (0.10, 0.20),
            "debt_to_equity": (0.3, 2.0),
            "current_ratio": (1.2, 2.5),
            "capex_pct": (0.03, 0.08),
            "dividend_pct": (0.10, 0.40),
            "buyback_pct": (0.05, 0.25),
            "depreciation_pct": (0.03, 0.07),
            "goodwill_pct": (0.10, 0.40),
            "cash_pct_assets": (0.05, 0.20),
        },
        "financial": {
            "label": "Financial Services",
            "revenue_range": (20000, 150000),
            "gross_margin": (0.50, 0.90),
            "operating_margin": (0.20, 0.45),
            "rd_pct": (0.00, 0.05),
            "sga_pct": (0.15, 0.35),
            "tax_rate": (0.18, 0.25),
            "debt_to_equity": (2.0, 10.0),
            "current_ratio": (0.8, 1.5),
            "capex_pct": (0.01, 0.04),
            "dividend_pct": (0.20, 0.50),
            "buyback_pct": (0.05, 0.30),
            "depreciation_pct": (0.01, 0.04),
            "goodwill_pct": (0.05, 0.25),
            "cash_pct_assets": (0.05, 0.15),
        },
    }

    COMPANY_NAMES = [
        "Apex Dynamics Corp", "Meridian Systems Inc", "Quantum Bridge Holdings",
        "NovaStar Technologies", "Atlas Global Industries", "Pinnacle Health Group",
        "Vertex Capital Partners", "Ironclad Manufacturing", "SilverLine Retail Inc",
        "Pacific Crest Energy", "Orion Financial Group", "Titan Consumer Brands",
        "Cobalt Semiconductor", "Evergreen Solutions Corp", "Helix BioSciences",
        "Summit Infrastructure", "ClearPath Logistics", "Nexus Digital Holdings",
        "BluePeak Resources", "Vanguard Medical Systems", "Cascade Innovations",
        "Sterling Aerospace", "Crimson Pharmaceuticals", "Obsidian Data Corp",
        "Magellan Ventures", "Keystone Industries", "Sapphire Cloud Inc",
        "Redwood Capital Group", "TerraFirma Holdings", "Zenith Communications",
    ]

    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.rng = random

    def _rand(self, lo: float, hi: float) -> float:
        return self.rng.uniform(lo, hi)

    def _rand_int(self, lo: int, hi: int) -> int:
        return self.rng.randint(lo, hi)

    def _round(self, val: float, decimals: int = 0) -> float:
        return round(val, decimals)

    def generate_company(self, industry: str = None, fiscal_year: int = None) -> dict:
        """Generate a complete set of financial statements for a fictional company."""
        if industry is None:
            industry = self.rng.choice(list(self.INDUSTRY_PROFILES.keys()))
        if fiscal_year is None:
            fiscal_year = self.rng.choice([2022, 2023, 2024])

        profile = self.INDUSTRY_PROFILES[industry]
        company_name = self.rng.choice(self.COMPANY_NAMES)
        ticker = "".join([w[0] for w in company_name.split()[:3]]).upper()

        # Generate 3 years of data with realistic YoY trends
        years = [fiscal_year - 2, fiscal_year - 1, fiscal_year]
        all_data = {}

        # Base year parameters
        base_revenue = self._rand(*profile["revenue_range"])
        revenue_growth_1 = self._rand(-0.05, 0.20)
        revenue_growth_2 = self._rand(-0.05, 0.20)

        for i, year in enumerate(years):
            if i == 0:
                revenue = base_revenue
            elif i == 1:
                revenue = base_revenue * (1 + revenue_growth_1)
            else:
                revenue = base_revenue * (1 + revenue_growth_1) * (1 + revenue_growth_2)

            all_data[year] = self._generate_single_year(
                revenue, profile, year, 
                # slight variations per year
                margin_drift=self._rand(-0.03, 0.03)
            )

        # Ensure balance sheet consistency across years
        self._ensure_cross_year_consistency(all_data, years)

        return {
            "company": {
                "name": company_name,
                "ticker": ticker,
                "industry": industry,
                "industry_label": profile["label"],
                "fiscal_year_end": fiscal_year,
            },
            "periods": {str(y): all_data[y] for y in years},
            "years": [str(y) for y in years],
        }

    def _generate_single_year(self, revenue: float, profile: dict, year: int,
                               margin_drift: float = 0.0) -> dict:
        """Generate internally-consistent financial data for one fiscal year."""
        r = self._round

        # === INCOME STATEMENT ===
        total_revenue = r(revenue)
        gross_margin = self._rand(*profile["gross_margin"]) + margin_drift
        gross_margin = max(0.10, min(0.95, gross_margin))
        cogs = r(total_revenue * (1 - gross_margin))
        gross_profit = r(total_revenue - cogs)

        rd = r(total_revenue * self._rand(*profile["rd_pct"]))
        sga = r(total_revenue * self._rand(*profile["sga_pct"]))
        da = r(total_revenue * self._rand(*profile["depreciation_pct"]))
        other_opex = r(self._rand(-500, 500))
        total_opex = r(rd + sga + da + other_opex)
        operating_income = r(gross_profit - total_opex)

        # Non-operating
        interest_expense = r(self._rand(200, max(300, total_revenue * 0.02)))
        interest_income = r(self._rand(50, max(100, total_revenue * 0.005)))
        other_income = r(self._rand(-300, 300))

        income_before_tax = r(operating_income - interest_expense + interest_income + other_income)
        tax_rate = self._rand(*profile["tax_rate"])
        if income_before_tax > 0:
            tax_expense = r(income_before_tax * tax_rate)
        else:
            tax_expense = r(income_before_tax * tax_rate * 0.3)  # tax benefit
        net_income = r(income_before_tax - tax_expense)

        shares_basic = r(self._rand(500, 15000))
        shares_diluted = r(shares_basic * self._rand(1.005, 1.03))
        eps_basic = round(net_income / shares_basic, 2) if shares_basic > 0 else 0
        eps_diluted = round(net_income / shares_diluted, 2) if shares_diluted > 0 else 0

        # === BALANCE SHEET ===
        # Work backwards from reasonable ratios
        target_de = self._rand(*profile["debt_to_equity"])
        target_cr = self._rand(*profile["current_ratio"])

        # Derive total assets from revenue and a reasonable asset turnover
        asset_turnover = self._rand(0.3, 1.5)
        total_assets = r(total_revenue / asset_turnover)

        # Equity and liabilities from D/E ratio
        total_equity = r(total_assets / (1 + target_de))
        total_liabilities = r(total_assets - total_equity)

        # Current vs non-current split
        current_liab_pct = self._rand(0.25, 0.50)
        total_current_liabilities = r(total_liabilities * current_liab_pct)
        total_non_current_liabilities = r(total_liabilities - total_current_liabilities)

        total_current_assets = r(total_current_liabilities * target_cr)
        total_non_current_assets = r(total_assets - total_current_assets)

        # Break down current assets
        cash_pct = self._rand(*profile["cash_pct_assets"])
        cash = r(total_assets * cash_pct)
        short_term_inv = r(self._rand(0, cash * 0.5))
        ar = r(total_revenue * self._rand(0.05, 0.15))  # ~30-55 days DSO
        inventories = r(cogs * self._rand(0.02, 0.15)) if cogs > 0 else 0  # inventory days
        prepaid = r(max(0, total_current_assets - cash - short_term_inv - ar - inventories))

        # Adjust to match total
        ca_sum = cash + short_term_inv + ar + inventories + prepaid
        if ca_sum > 0:
            scale = total_current_assets / ca_sum
            cash = r(cash * scale)
            short_term_inv = r(short_term_inv * scale)
            ar = r(ar * scale)
            inventories = r(inventories * scale)
            prepaid = r(prepaid * scale)

        # Break down non-current assets
        goodwill_pct = self._rand(*profile["goodwill_pct"])
        goodwill = r(total_assets * goodwill_pct)
        ppe = r(total_assets * self._rand(0.10, 0.40))
        intangibles = r(total_assets * self._rand(0.02, 0.12))
        lt_investments = r(self._rand(0, total_assets * 0.10))
        deferred_tax_a = r(self._rand(0, total_assets * 0.03))
        other_nca = r(max(0, total_non_current_assets - goodwill - ppe - intangibles - lt_investments - deferred_tax_a))

        nca_sum = goodwill + ppe + intangibles + lt_investments + deferred_tax_a + other_nca
        if nca_sum > 0:
            scale = total_non_current_assets / nca_sum
            goodwill = r(goodwill * scale)
            ppe = r(ppe * scale)
            intangibles = r(intangibles * scale)
            lt_investments = r(lt_investments * scale)
            deferred_tax_a = r(deferred_tax_a * scale)
            other_nca = r(other_nca * scale)

        # Break down current liabilities
        ap = r(cogs * self._rand(0.05, 0.15))
        accrued = r(total_revenue * self._rand(0.03, 0.10))
        short_debt = r(self._rand(0, total_current_liabilities * 0.3))
        deferred_rev_c = r(self._rand(0, total_revenue * 0.05))
        taxes_payable = r(max(0, tax_expense * self._rand(0.05, 0.25)))
        cl_sum = ap + accrued + short_debt + deferred_rev_c + taxes_payable
        if cl_sum > 0:
            scale = total_current_liabilities / cl_sum
            ap = r(ap * scale)
            accrued = r(accrued * scale)
            short_debt = r(short_debt * scale)
            deferred_rev_c = r(deferred_rev_c * scale)
            taxes_payable = r(taxes_payable * scale)

        # Break down non-current liabilities
        lt_debt = r(total_non_current_liabilities * self._rand(0.40, 0.80))
        deferred_rev_nc = r(self._rand(0, total_revenue * 0.03))
        deferred_tax_l = r(self._rand(0, total_assets * 0.03))
        pension = r(self._rand(0, total_assets * 0.03))
        other_ncl = r(max(0, total_non_current_liabilities - lt_debt - deferred_rev_nc - deferred_tax_l - pension))

        # Break down equity
        common_stock = r(self._rand(1, 80))
        apic = r(total_equity * self._rand(0.20, 0.60))
        aoci = r(self._rand(-total_equity * 0.10, total_equity * 0.03))
        treasury = r(-abs(self._rand(0, total_equity * 0.30)))
        retained = r(total_equity - common_stock - apic - aoci - treasury)

        # === CASH FLOW STATEMENT ===
        cfo_adjustments = r(da + self._rand(500, max(1000, total_revenue * 0.03)))  # D&A + SBC + WC changes
        sbc = r(self._rand(200, max(500, total_revenue * 0.04)))
        wc_changes = r(self._rand(-total_revenue * 0.03, total_revenue * 0.03))
        other_cf_adj = r(self._rand(-500, 500))
        cash_from_ops = r(net_income + da + sbc + wc_changes + other_cf_adj)

        capex = r(total_revenue * self._rand(*profile["capex_pct"]))
        acquisitions = r(self._rand(0, total_revenue * 0.05))
        inv_purchases = r(self._rand(0, total_revenue * 0.05))
        inv_proceeds = r(self._rand(0, total_revenue * 0.04))
        cash_from_inv = r(-capex - acquisitions - inv_purchases + inv_proceeds)

        debt_issued = r(self._rand(0, total_revenue * 0.05))
        debt_repaid = r(self._rand(0, total_revenue * 0.04))
        buybacks = r(net_income * self._rand(*profile["buyback_pct"])) if net_income > 0 else 0
        dividends = r(net_income * self._rand(*profile["dividend_pct"])) if net_income > 0 else 0
        cash_from_fin = r(debt_issued - debt_repaid - buybacks - dividends)

        net_cash_change = r(cash_from_ops + cash_from_inv + cash_from_fin)

        # Recompute totals to ensure perfect consistency
        total_current_assets = r(cash + short_term_inv + ar + inventories + prepaid)
        total_non_current_assets = r(goodwill + ppe + intangibles + lt_investments + deferred_tax_a + other_nca)
        total_assets = r(total_current_assets + total_non_current_assets)

        total_current_liabilities = r(ap + accrued + short_debt + deferred_rev_c + taxes_payable)
        total_non_current_liabilities = r(lt_debt + deferred_rev_nc + deferred_tax_l + pension + other_ncl)
        total_liabilities = r(total_current_liabilities + total_non_current_liabilities)

        total_equity = r(common_stock + apic + retained + aoci + treasury)
        # Force balance: adjust retained earnings to make A = L + E
        gap = total_assets - total_liabilities - total_equity
        retained = r(retained + gap)
        total_equity = r(common_stock + apic + retained + aoci + treasury)

        return {
            # Income Statement
            "total_revenue": total_revenue,
            "cost_of_goods_sold": cogs,
            "gross_profit": gross_profit,
            "research_and_development": rd,
            "selling_general_admin": sga,
            "depreciation_amortization": da,
            "other_operating_expenses": other_opex,
            "total_operating_expenses": total_opex,
            "operating_income": operating_income,
            "interest_expense": interest_expense,
            "interest_income": interest_income,
            "other_income_expense": other_income,
            "income_before_tax": income_before_tax,
            "income_tax_expense": tax_expense,
            "net_income": net_income,
            "eps_basic": eps_basic,
            "eps_diluted": eps_diluted,
            "shares_basic": shares_basic,
            "shares_diluted": shares_diluted,

            # Balance Sheet — Current Assets
            "cash_and_equivalents": cash,
            "short_term_investments": short_term_inv,
            "accounts_receivable_net": ar,
            "inventories": inventories,
            "prepaid_expenses": prepaid,
            "total_current_assets": total_current_assets,

            # Balance Sheet — Non-Current Assets
            "property_plant_equipment_net": ppe,
            "goodwill": goodwill,
            "intangible_assets_net": intangibles,
            "long_term_investments": lt_investments,
            "deferred_tax_assets": deferred_tax_a,
            "other_non_current_assets": other_nca,
            "total_non_current_assets": total_non_current_assets,
            "total_assets": total_assets,

            # Balance Sheet — Current Liabilities
            "accounts_payable": ap,
            "accrued_expenses": accrued,
            "short_term_debt": short_debt,
            "deferred_revenue_current": deferred_rev_c,
            "income_taxes_payable": taxes_payable,
            "total_current_liabilities": total_current_liabilities,

            # Balance Sheet — Non-Current Liabilities
            "long_term_debt": lt_debt,
            "deferred_revenue_non_current": deferred_rev_nc,
            "deferred_tax_liabilities": deferred_tax_l,
            "pension_obligations": pension,
            "other_non_current_liabilities": other_ncl,
            "total_non_current_liabilities": total_non_current_liabilities,
            "total_liabilities": total_liabilities,

            # Balance Sheet — Equity
            "common_stock": common_stock,
            "additional_paid_in_capital": apic,
            "retained_earnings": retained,
            "accumulated_other_comprehensive_income": aoci,
            "treasury_stock": treasury,
            "total_stockholders_equity": total_equity,
            "total_liabilities_and_equity": r(total_liabilities + total_equity),

            # Cash Flow
            "cash_from_operations": cash_from_ops,
            "stock_based_compensation": sbc,
            "changes_in_working_capital": wc_changes,
            "other_operating_adjustments": other_cf_adj,
            "capital_expenditures": capex,
            "acquisitions": acquisitions,
            "purchases_of_investments": inv_purchases,
            "proceeds_from_investments": inv_proceeds,
            "cash_from_investing": cash_from_inv,
            "debt_issued": debt_issued,
            "debt_repaid": debt_repaid,
            "share_repurchases": buybacks,
            "dividends_paid": dividends,
            "cash_from_financing": cash_from_fin,
            "net_change_in_cash": net_cash_change,
        }

    def _ensure_cross_year_consistency(self, all_data: dict, years: list):
        """Minor adjustments across years for realism."""
        # Ensure shares don't fluctuate wildly
        base_shares = all_data[years[0]]["shares_basic"]
        for y in years[1:]:
            drift = random.uniform(-0.03, 0.01)  # shares usually decrease (buybacks)
            base_shares = round(base_shares * (1 + drift))
            all_data[y]["shares_basic"] = base_shares
            all_data[y]["shares_diluted"] = round(base_shares * random.uniform(1.005, 1.03))
            if all_data[y]["shares_basic"] > 0:
                all_data[y]["eps_basic"] = round(all_data[y]["net_income"] / all_data[y]["shares_basic"], 2)
            if all_data[y]["shares_diluted"] > 0:
                all_data[y]["eps_diluted"] = round(all_data[y]["net_income"] / all_data[y]["shares_diluted"], 2)


# ============================================================
# 2. FORMAL QUERY COMPOSITION
# ============================================================

@dataclass
class FormalQuery:
    """Represents a composable formal query over financial data."""
    expression: str          # e.g. "ratio(get_value(net_income, 2024), get_value(total_revenue, 2024))"
    steps: list              # decomposed execution steps
    variables_used: list     # which financial variables are referenced
    periods_used: list       # which periods are referenced
    difficulty: int          # 1-5
    concept_name: str = None # named financial concept if applicable
    category: str = None     # liquidity, leverage, profitability, etc.


class QueryComposer:
    """Composes formal queries at various difficulty levels."""

    # Variable labels for NL generation
    VAR_LABELS = {
        "total_assets": "total assets",
        "total_current_assets": "total current assets",
        "total_non_current_assets": "total non-current assets",
        "total_liabilities": "total liabilities",
        "total_current_liabilities": "total current liabilities",
        "total_non_current_liabilities": "total non-current liabilities",
        "total_stockholders_equity": "total stockholders' equity",
        "total_revenue": "total revenue",
        "cost_of_goods_sold": "cost of goods sold",
        "gross_profit": "gross profit",
        "operating_income": "operating income",
        "net_income": "net income",
        "income_before_tax": "income before taxes",
        "income_tax_expense": "income tax expense",
        "cash_and_equivalents": "cash and cash equivalents",
        "short_term_investments": "short-term investments",
        "accounts_receivable_net": "accounts receivable",
        "inventories": "inventories",
        "prepaid_expenses": "prepaid expenses",
        "property_plant_equipment_net": "property, plant and equipment",
        "goodwill": "goodwill",
        "intangible_assets_net": "intangible assets",
        "long_term_investments": "long-term investments",
        "deferred_tax_assets": "deferred tax assets",
        "other_non_current_assets": "other non-current assets",
        "accounts_payable": "accounts payable",
        "accrued_expenses": "accrued expenses",
        "short_term_debt": "short-term debt",
        "deferred_revenue_current": "current deferred revenue",
        "income_taxes_payable": "income taxes payable",
        "long_term_debt": "long-term debt",
        "deferred_revenue_non_current": "non-current deferred revenue",
        "deferred_tax_liabilities": "deferred tax liabilities",
        "pension_obligations": "pension obligations",
        "other_non_current_liabilities": "other non-current liabilities",
        "common_stock": "common stock",
        "additional_paid_in_capital": "additional paid-in capital",
        "retained_earnings": "retained earnings",
        "accumulated_other_comprehensive_income": "accumulated other comprehensive income",
        "treasury_stock": "treasury stock",
        "research_and_development": "research and development expense",
        "selling_general_admin": "selling, general and administrative expense",
        "depreciation_amortization": "depreciation and amortization",
        "interest_expense": "interest expense",
        "interest_income": "interest income",
        "other_income_expense": "other income/expense",
        "cash_from_operations": "net cash from operating activities",
        "cash_from_investing": "net cash from investing activities",
        "cash_from_financing": "net cash from financing activities",
        "capital_expenditures": "capital expenditures",
        "dividends_paid": "dividends paid",
        "share_repurchases": "share repurchases",
        "debt_issued": "debt issued",
        "debt_repaid": "debt repaid",
        "stock_based_compensation": "stock-based compensation",
        "eps_basic": "basic earnings per share",
        "eps_diluted": "diluted earnings per share",
        "shares_basic": "basic shares outstanding",
        "shares_diluted": "diluted shares outstanding",
        "net_change_in_cash": "net change in cash",
        "total_operating_expenses": "total operating expenses",
        "other_operating_expenses": "other operating expenses",
    }

    # All variables that can be looked up
    LOOKUPABLE_VARS = list(VAR_LABELS.keys())

    # Named concept definitions (formula pattern → concept)
    CONCEPTS = {
        "current_ratio": {
            "category": "liquidity",
            "formula_fn": lambda d, y: d[y]["total_current_assets"] / d[y]["total_current_liabilities"] if d[y]["total_current_liabilities"] != 0 else None,
            "formal": "ratio(get_value(total_current_assets, {y}), get_value(total_current_liabilities, {y}))",
            "vars": ["total_current_assets", "total_current_liabilities"],
            "nl_templates": [
                "What was the current ratio in {y}?",
                "What was {company}'s current ratio for fiscal year {y}?",
                "In {y}, for every dollar of current liabilities, how many dollars of current assets did {company} have?",
            ],
        },
        "quick_ratio": {
            "category": "liquidity",
            "formula_fn": lambda d, y: (d[y]["total_current_assets"] - d[y]["inventories"]) / d[y]["total_current_liabilities"] if d[y]["total_current_liabilities"] != 0 else None,
            "formal": "ratio(diff(get_value(total_current_assets, {y}), get_value(inventories, {y})), get_value(total_current_liabilities, {y}))",
            "vars": ["total_current_assets", "inventories", "total_current_liabilities"],
            "nl_templates": [
                "What was the quick ratio in {y}?",
                "Excluding inventory, what was {company}'s liquidity ratio in {y}?",
            ],
        },
        "working_capital": {
            "category": "liquidity",
            "formula_fn": lambda d, y: d[y]["total_current_assets"] - d[y]["total_current_liabilities"],
            "formal": "diff(get_value(total_current_assets, {y}), get_value(total_current_liabilities, {y}))",
            "vars": ["total_current_assets", "total_current_liabilities"],
            "nl_templates": [
                "What was the working capital in {y}?",
                "How much net working capital did {company} have in {y}?",
            ],
        },
        "debt_to_equity": {
            "category": "leverage",
            "formula_fn": lambda d, y: d[y]["total_liabilities"] / d[y]["total_stockholders_equity"] if d[y]["total_stockholders_equity"] != 0 else None,
            "formal": "ratio(get_value(total_liabilities, {y}), get_value(total_stockholders_equity, {y}))",
            "vars": ["total_liabilities", "total_stockholders_equity"],
            "nl_templates": [
                "What was the debt-to-equity ratio in {y}?",
                "How leveraged was {company} in {y}?",
                "For every dollar of equity, how much debt did {company} carry in {y}?",
            ],
        },
        "debt_to_assets": {
            "category": "leverage",
            "formula_fn": lambda d, y: d[y]["total_liabilities"] / d[y]["total_assets"] if d[y]["total_assets"] != 0 else None,
            "formal": "ratio(get_value(total_liabilities, {y}), get_value(total_assets, {y}))",
            "vars": ["total_liabilities", "total_assets"],
            "nl_templates": [
                "What was the debt-to-assets ratio in {y}?",
                "What percentage of {company}'s total assets was financed by debt in {y}?",
            ],
        },
        "interest_coverage": {
            "category": "leverage",
            "formula_fn": lambda d, y: d[y]["operating_income"] / d[y]["interest_expense"] if d[y]["interest_expense"] != 0 else None,
            "formal": "ratio(get_value(operating_income, {y}), get_value(interest_expense, {y}))",
            "vars": ["operating_income", "interest_expense"],
            "nl_templates": [
                "What was the interest coverage ratio in {y}?",
                "How many times could {company} cover its interest expense with operating income in {y}?",
            ],
        },
        "gross_margin": {
            "category": "profitability",
            "formula_fn": lambda d, y: d[y]["gross_profit"] / d[y]["total_revenue"] if d[y]["total_revenue"] != 0 else None,
            "formal": "ratio(get_value(gross_profit, {y}), get_value(total_revenue, {y}))",
            "vars": ["gross_profit", "total_revenue"],
            "nl_templates": [
                "What was the gross profit margin in {y}?",
                "What percentage of revenue was retained after cost of goods sold in {y}?",
            ],
        },
        "operating_margin": {
            "category": "profitability",
            "formula_fn": lambda d, y: d[y]["operating_income"] / d[y]["total_revenue"] if d[y]["total_revenue"] != 0 else None,
            "formal": "ratio(get_value(operating_income, {y}), get_value(total_revenue, {y}))",
            "vars": ["operating_income", "total_revenue"],
            "nl_templates": [
                "What was the operating profit margin in {y}?",
                "What percentage of {company}'s revenue became operating income in {y}?",
            ],
        },
        "net_profit_margin": {
            "category": "profitability",
            "formula_fn": lambda d, y: d[y]["net_income"] / d[y]["total_revenue"] if d[y]["total_revenue"] != 0 else None,
            "formal": "ratio(get_value(net_income, {y}), get_value(total_revenue, {y}))",
            "vars": ["net_income", "total_revenue"],
            "nl_templates": [
                "What was the net profit margin in {y}?",
                "How much of each dollar of revenue was net income for {company} in {y}?",
            ],
        },
        "return_on_assets": {
            "category": "profitability",
            "formula_fn": lambda d, y: d[y]["net_income"] / ((d[y]["total_assets"] + d[str(int(y)-1)]["total_assets"]) / 2) if str(int(y)-1) in d and ((d[y]["total_assets"] + d[str(int(y)-1)]["total_assets"]) / 2) != 0 else None,
            "formal": "ratio(get_value(net_income, {y}), avg(get_value(total_assets, {y}), get_value(total_assets, {y_prev})))",
            "vars": ["net_income", "total_assets"],
            "needs_prior_year": True,
            "nl_templates": [
                "What was the return on assets (ROA) in {y}?",
                "How efficiently did {company} use its assets to generate profit in {y}?",
            ],
        },
        "return_on_equity": {
            "category": "profitability",
            "formula_fn": lambda d, y: d[y]["net_income"] / ((d[y]["total_stockholders_equity"] + d[str(int(y)-1)]["total_stockholders_equity"]) / 2) if str(int(y)-1) in d and ((d[y]["total_stockholders_equity"] + d[str(int(y)-1)]["total_stockholders_equity"]) / 2) != 0 else None,
            "formal": "ratio(get_value(net_income, {y}), avg(get_value(total_stockholders_equity, {y}), get_value(total_stockholders_equity, {y_prev})))",
            "vars": ["net_income", "total_stockholders_equity"],
            "needs_prior_year": True,
            "nl_templates": [
                "What was the return on equity (ROE) in {y}?",
                "What return did {company}'s equity investors earn in {y}?",
            ],
        },
        "effective_tax_rate": {
            "category": "profitability",
            "formula_fn": lambda d, y: d[y]["income_tax_expense"] / d[y]["income_before_tax"] if d[y]["income_before_tax"] != 0 else None,
            "formal": "ratio(get_value(income_tax_expense, {y}), get_value(income_before_tax, {y}))",
            "vars": ["income_tax_expense", "income_before_tax"],
            "nl_templates": [
                "What was the effective tax rate in {y}?",
                "What percentage of pre-tax income did {company} pay in taxes in {y}?",
            ],
        },
        "free_cash_flow": {
            "category": "cash_flow",
            "formula_fn": lambda d, y: d[y]["cash_from_operations"] - d[y]["capital_expenditures"],
            "formal": "diff(get_value(cash_from_operations, {y}), get_value(capital_expenditures, {y}))",
            "vars": ["cash_from_operations", "capital_expenditures"],
            "nl_templates": [
                "What was the free cash flow in {y}?",
                "How much cash was left after capital expenditures in {y}?",
            ],
        },
        "rd_to_revenue": {
            "category": "efficiency",
            "formula_fn": lambda d, y: d[y]["research_and_development"] / d[y]["total_revenue"] if d[y]["total_revenue"] != 0 else None,
            "formal": "ratio(get_value(research_and_development, {y}), get_value(total_revenue, {y}))",
            "vars": ["research_and_development", "total_revenue"],
            "nl_templates": [
                "What percentage of revenue was spent on R&D in {y}?",
                "What was {company}'s R&D intensity in {y}?",
            ],
        },
        "capex_to_revenue": {
            "category": "efficiency",
            "formula_fn": lambda d, y: d[y]["capital_expenditures"] / d[y]["total_revenue"] if d[y]["total_revenue"] != 0 else None,
            "formal": "ratio(get_value(capital_expenditures, {y}), get_value(total_revenue, {y}))",
            "vars": ["capital_expenditures", "total_revenue"],
            "nl_templates": [
                "What was capital expenditure as a percentage of revenue in {y}?",
                "How capital-intensive was {company} in {y}?",
            ],
        },
    }

    # Growth/change concepts (need two periods)
    GROWTH_CONCEPTS = {
        "revenue_growth": {
            "category": "growth",
            "var": "total_revenue",
            "nl_templates": [
                "What was the revenue growth rate from {y1} to {y2}?",
                "By what percentage did {company}'s revenue change from {y1} to {y2}?",
                "What was {company}'s top-line growth?",
            ],
        },
        "net_income_growth": {
            "category": "growth",
            "var": "net_income",
            "nl_templates": [
                "What was the net income growth rate from {y1} to {y2}?",
                "By what percentage did {company}'s earnings change from {y1} to {y2}?",
            ],
        },
        "asset_growth": {
            "category": "growth",
            "var": "total_assets",
            "nl_templates": [
                "By what percentage did total assets grow from {y1} to {y2}?",
                "What was the year-over-year growth in {company}'s asset base?",
            ],
        },
        "equity_growth": {
            "category": "growth",
            "var": "total_stockholders_equity",
            "nl_templates": [
                "How did shareholders' equity change from {y1} to {y2}?",
                "What was the growth rate in {company}'s book value?",
            ],
        },
        "operating_income_growth": {
            "category": "growth",
            "var": "operating_income",
            "nl_templates": [
                "What was the change in operating income from {y1} to {y2}?",
                "By what percentage did operating profit grow?",
            ],
        },
    }

    def compose_level1(self, company_data: dict) -> list:
        """Level 1: Simple value retrieval."""
        queries = []
        years = company_data["years"]
        company_name = company_data["company"]["name"]

        vars_to_query = random.sample(self.LOOKUPABLE_VARS, min(15, len(self.LOOKUPABLE_VARS)))
        for var in vars_to_query:
            year = random.choice(years)
            label = self.VAR_LABELS.get(var, var)
            templates = [
                f"What was the {label} in {year}?",
                f"How much did {company_name} report for {label} in fiscal year {year}?",
                f"What was {company_name}'s {label} as of {year}?",
            ]
            queries.append(FormalQuery(
                expression=f"get_value({var}, {year})",
                steps=[f"get_value({var}, {year})"],
                variables_used=[var],
                periods_used=[year],
                difficulty=1,
                concept_name=None,
                category="lookup",
            ))
        return queries

    def compose_level2(self, company_data: dict) -> list:
        """Level 2: Single arithmetic operation."""
        queries = []
        years = company_data["years"]

        # YoY differences
        diff_vars = ["total_revenue", "net_income", "total_assets", "total_liabilities",
                      "total_stockholders_equity", "cash_and_equivalents", "operating_income",
                      "long_term_debt", "gross_profit"]
        for var in random.sample(diff_vars, min(6, len(diff_vars))):
            if len(years) >= 2:
                y1, y2 = years[-2], years[-1]
                queries.append(FormalQuery(
                    expression=f"diff(get_value({var}, {y2}), get_value({var}, {y1}))",
                    steps=[f"get_value({var}, {y2})", f"get_value({var}, {y1})", f"diff(#0, #1)"],
                    variables_used=[var],
                    periods_used=[y1, y2],
                    difficulty=2,
                    category="change",
                ))

        # Simple ratios (not named concepts)
        ratio_pairs = [
            ("cash_and_equivalents", "total_assets"),
            ("long_term_debt", "total_assets"),
            ("accounts_receivable_net", "total_revenue"),
            ("inventories", "cost_of_goods_sold"),
            ("selling_general_admin", "total_revenue"),
        ]
        for num, den in random.sample(ratio_pairs, min(4, len(ratio_pairs))):
            year = random.choice(years)
            queries.append(FormalQuery(
                expression=f"ratio(get_value({num}, {year}), get_value({den}, {year}))",
                steps=[f"get_value({num}, {year})", f"get_value({den}, {year})", f"ratio(#0, #1)"],
                variables_used=[num, den],
                periods_used=[year],
                difficulty=2,
                category="ratio",
            ))

        return queries

    def compose_level3(self, company_data: dict) -> list:
        """Level 3: Named financial concepts."""
        queries = []
        years = company_data["years"]

        concepts_to_use = random.sample(list(self.CONCEPTS.keys()), min(10, len(self.CONCEPTS)))
        for concept_name in concepts_to_use:
            concept = self.CONCEPTS[concept_name]
            needs_prior = concept.get("needs_prior_year", False)

            if needs_prior and len(years) < 2:
                continue

            year = years[-1] if needs_prior else random.choice(years)
            y_prev = years[-2] if needs_prior else None

            queries.append(FormalQuery(
                expression=concept["formal"].format(y=year, y_prev=y_prev or ""),
                steps=[],  # will be filled during execution
                variables_used=concept["vars"],
                periods_used=[year] + ([y_prev] if y_prev else []),
                difficulty=3,
                concept_name=concept_name,
                category=concept["category"],
            ))

        return queries

    def compose_level4(self, company_data: dict) -> list:
        """Level 4: Multi-step reasoning (e.g., change in a ratio across years)."""
        queries = []
        years = company_data["years"]
        if len(years) < 2:
            return queries

        y1, y2 = years[-2], years[-1]

        # Change in named ratios across years
        ratio_concepts = ["current_ratio", "debt_to_equity", "gross_margin", "operating_margin", "net_profit_margin"]
        for concept_name in random.sample(ratio_concepts, min(4, len(ratio_concepts))):
            concept = self.CONCEPTS[concept_name]
            queries.append(FormalQuery(
                expression=f"diff({concept_name}({y2}), {concept_name}({y1}))",
                steps=[f"compute_{concept_name}({y1})", f"compute_{concept_name}({y2})", "diff(#1, #0)"],
                variables_used=concept["vars"],
                periods_used=[y1, y2],
                difficulty=4,
                concept_name=f"change_in_{concept_name}",
                category=concept["category"],
            ))

        # Growth concepts
        for gname, gdef in random.sample(list(self.GROWTH_CONCEPTS.items()), min(3, len(self.GROWTH_CONCEPTS))):
            queries.append(FormalQuery(
                expression=f"pct_change({gdef['var']}, {y1}, {y2})",
                steps=[f"get_value({gdef['var']}, {y1})", f"get_value({gdef['var']}, {y2})", "pct_change(#0, #1)"],
                variables_used=[gdef["var"]],
                periods_used=[y1, y2],
                difficulty=4,
                concept_name=gname,
                category=gdef["category"],
            ))

        return queries

    def compose_level5(self, company_data: dict) -> list:
        """Level 5: Comparative / boolean / multi-concept questions."""
        queries = []
        years = company_data["years"]
        if len(years) < 2:
            return queries

        y1, y2 = years[-2], years[-1]

        # "Did X improve?"
        for concept_name in random.sample(["gross_margin", "operating_margin", "current_ratio", "debt_to_equity"], 2):
            queries.append(FormalQuery(
                expression=f"compare_gt({concept_name}({y2}), {concept_name}({y1}))",
                steps=[f"compute_{concept_name}({y1})", f"compute_{concept_name}({y2})", "compare_gt(#1, #0)"],
                variables_used=self.CONCEPTS[concept_name]["vars"],
                periods_used=[y1, y2],
                difficulty=5,
                concept_name=f"did_{concept_name}_improve",
                category=self.CONCEPTS[concept_name]["category"],
            ))

        # FCF margin change
        queries.append(FormalQuery(
            expression=f"diff(ratio(diff(cash_from_operations, capital_expenditures), total_revenue, {y2}), ratio(diff(cash_from_operations, capital_expenditures), total_revenue, {y1}))",
            steps=["compute_fcf_margin(y1)", "compute_fcf_margin(y2)", "diff(#1, #0)"],
            variables_used=["cash_from_operations", "capital_expenditures", "total_revenue"],
            periods_used=[y1, y2],
            difficulty=5,
            concept_name="change_in_fcf_margin",
            category="cash_flow",
        ))

        return queries

    def compose_all(self, company_data: dict, max_per_level: int = None) -> list:
        """Generate queries at all difficulty levels."""
        all_queries = []
        for level_fn in [self.compose_level1, self.compose_level2, self.compose_level3,
                         self.compose_level4, self.compose_level5]:
            queries = level_fn(company_data)
            if max_per_level and len(queries) > max_per_level:
                queries = random.sample(queries, max_per_level)
            all_queries.extend(queries)
        return all_queries


# ============================================================
# 3. EXECUTION ENGINE — Deterministic Ground Truth
# ============================================================

class ExecutionEngine:
    """Executes formal queries against financial data to produce ground truth answers."""

    def __init__(self, company_data: dict):
        self.data = company_data["periods"]
        self.years = company_data["years"]
        self.company_name = company_data["company"]["name"]

    def get_value(self, var: str, year: str) -> float:
        if year not in self.data:
            raise ValueError(f"Year {year} not in data")
        if var not in self.data[year]:
            raise ValueError(f"Variable {var} not in data for year {year}")
        return self.data[year][var]

    def execute(self, query: FormalQuery) -> dict:
        """Execute a formal query and return the result with full trace."""
        try:
            result = self._dispatch(query)
            return {
                "success": True,
                "answer": result["value"],
                "answer_formatted": self._format_answer(result["value"], result.get("type", "number")),
                "computation_trace": result.get("trace", ""),
                "type": result.get("type", "number"),
            }
        except Exception as e:
            return {
                "success": False,
                "answer": None,
                "error": str(e),
            }

    def _dispatch(self, query: FormalQuery) -> dict:
        concept = query.concept_name
        periods = query.periods_used

        # Level 1: simple lookup
        if query.difficulty == 1:
            var = query.variables_used[0]
            year = periods[0]
            val = self.get_value(var, year)
            return {"value": val, "type": "monetary", "trace": f"{var}({year}) = {val}"}

        # Level 2: single operation
        if query.difficulty == 2:
            if query.category == "change":
                var = query.variables_used[0]
                y1, y2 = periods[0], periods[1]
                v1 = self.get_value(var, y1)
                v2 = self.get_value(var, y2)
                diff = round(v2 - v1, 2)
                return {"value": diff, "type": "monetary",
                        "trace": f"{var}({y2}) - {var}({y1}) = {v2} - {v1} = {diff}"}
            elif query.category == "ratio":
                num_var, den_var = query.variables_used[0], query.variables_used[1]
                year = periods[0]
                num = self.get_value(num_var, year)
                den = self.get_value(den_var, year)
                if den == 0:
                    return {"value": None, "type": "ratio", "trace": "Division by zero"}
                val = round(num / den, 4)
                return {"value": val, "type": "ratio",
                        "trace": f"{num_var}({year}) / {den_var}({year}) = {num} / {den} = {val}"}

        # Level 3: named concepts
        if query.difficulty == 3 and concept in QueryComposer.CONCEPTS:
            cdef = QueryComposer.CONCEPTS[concept]
            y = periods[0]
            val = cdef["formula_fn"](self.data, y)
            if val is not None:
                val = round(val, 4)
            return {"value": val, "type": "ratio",
                    "trace": f"{concept}({y}) = {val}"}

        # Level 4: growth / change in ratio
        if query.difficulty == 4:
            if concept and concept.startswith("change_in_"):
                base_concept = concept.replace("change_in_", "")
                if base_concept in QueryComposer.CONCEPTS:
                    cdef = QueryComposer.CONCEPTS[base_concept]
                    y1, y2 = periods[0], periods[1]
                    v1 = cdef["formula_fn"](self.data, y1)
                    v2 = cdef["formula_fn"](self.data, y2)
                    if v1 is not None and v2 is not None:
                        diff = round(v2 - v1, 4)
                        return {"value": diff, "type": "ratio_change",
                                "trace": f"{base_concept}({y2}) - {base_concept}({y1}) = {round(v2,4)} - {round(v1,4)} = {diff}"}

            if concept in QueryComposer.GROWTH_CONCEPTS:
                gdef = QueryComposer.GROWTH_CONCEPTS[concept]
                var = gdef["var"]
                y1, y2 = periods[0], periods[1]
                v1 = self.get_value(var, y1)
                v2 = self.get_value(var, y2)
                if v1 == 0:
                    return {"value": None, "type": "percentage", "trace": "Division by zero (base=0)"}
                pct = round((v2 - v1) / abs(v1) * 100, 2)
                return {"value": pct, "type": "percentage",
                        "trace": f"({var}({y2}) - {var}({y1})) / |{var}({y1})| * 100 = ({v2} - {v1}) / {abs(v1)} * 100 = {pct}%"}

        # Level 5: comparisons
        if query.difficulty == 5:
            if concept and concept.startswith("did_") and concept.endswith("_improve"):
                base_concept = concept.replace("did_", "").replace("_improve", "")
                if base_concept in QueryComposer.CONCEPTS:
                    cdef = QueryComposer.CONCEPTS[base_concept]
                    y1, y2 = periods[0], periods[1]
                    v1 = cdef["formula_fn"](self.data, y1)
                    v2 = cdef["formula_fn"](self.data, y2)
                    if v1 is not None and v2 is not None:
                        # For debt_to_equity, lower is "better"
                        if base_concept in ["debt_to_equity"]:
                            improved = v2 < v1
                        else:
                            improved = v2 > v1
                        change = round(v2 - v1, 4)
                        return {"value": improved, "type": "boolean",
                                "trace": f"{base_concept}({y1})={round(v1,4)}, {base_concept}({y2})={round(v2,4)}, improved={improved}, change={change}"}

            if concept == "change_in_fcf_margin":
                y1, y2 = periods[0], periods[1]
                fcf1 = self.get_value("cash_from_operations", y1) - self.get_value("capital_expenditures", y1)
                fcf2 = self.get_value("cash_from_operations", y2) - self.get_value("capital_expenditures", y2)
                rev1 = self.get_value("total_revenue", y1)
                rev2 = self.get_value("total_revenue", y2)
                if rev1 == 0 or rev2 == 0:
                    return {"value": None, "type": "ratio_change"}
                m1 = fcf1 / rev1
                m2 = fcf2 / rev2
                change = round(m2 - m1, 4)
                return {"value": change, "type": "ratio_change",
                        "trace": f"FCF_margin({y2}) - FCF_margin({y1}) = {round(m2,4)} - {round(m1,4)} = {change}"}

        return {"value": None, "type": "unknown", "trace": "Could not execute"}

    def _format_answer(self, value, answer_type: str) -> str:
        if value is None:
            return "N/A"
        if answer_type == "boolean":
            return "Yes" if value else "No"
        if answer_type == "percentage":
            return f"{value}%"
        if answer_type == "monetary":
            if abs(value) >= 1000:
                return f"${value:,.0f} million"
            return f"${value:,.2f} million"
        if answer_type in ("ratio", "ratio_change"):
            return f"{value:.4f}"
        return str(value)


# ============================================================
# 4. NATURAL LANGUAGE COMPILER
# ============================================================

class NLCompiler:
    """Compiles formal queries into natural language questions."""

    def __init__(self, company_name: str):
        self.company = company_name

    def compile(self, query: FormalQuery) -> str:
        """Convert a formal query to a natural language question."""
        concept = query.concept_name
        periods = query.periods_used
        vars_used = query.variables_used

        # Level 1: lookup
        if query.difficulty == 1:
            return self._compile_lookup(vars_used[0], periods[0])

        # Level 2: single op
        if query.difficulty == 2:
            if query.category == "change":
                return self._compile_change(vars_used[0], periods[0], periods[1])
            elif query.category == "ratio":
                return self._compile_simple_ratio(vars_used[0], vars_used[1], periods[0])

        # Level 3: named concept
        if query.difficulty == 3 and concept in QueryComposer.CONCEPTS:
            cdef = QueryComposer.CONCEPTS[concept]
            templates = cdef["nl_templates"]
            template = random.choice(templates)
            return template.format(y=periods[0], company=self.company, y_prev=periods[1] if len(periods) > 1 else "")

        # Level 4: growth or ratio change
        if query.difficulty == 4:
            if concept and concept.startswith("change_in_"):
                base = concept.replace("change_in_", "")
                base_label = base.replace("_", " ")
                templates = [
                    f"How did the {base_label} change from {periods[0]} to {periods[1]}?",
                    f"What was the change in {self.company}'s {base_label} between {periods[0]} and {periods[1]}?",
                    f"By how much did the {base_label} change from fiscal year {periods[0]} to {periods[1]}?",
                ]
                return random.choice(templates)

            if concept in QueryComposer.GROWTH_CONCEPTS:
                gdef = QueryComposer.GROWTH_CONCEPTS[concept]
                templates = gdef["nl_templates"]
                template = random.choice(templates)
                return template.format(y1=periods[0], y2=periods[1], company=self.company)

        # Level 5: comparative
        if query.difficulty == 5:
            if concept and concept.startswith("did_") and concept.endswith("_improve"):
                base = concept.replace("did_", "").replace("_improve", "")
                base_label = base.replace("_", " ")
                templates = [
                    f"Did {self.company}'s {base_label} improve from {periods[0]} to {periods[1]}?",
                    f"Was there an improvement in the {base_label} between {periods[0]} and {periods[1]}?",
                    f"Did the {base_label} get better from {periods[0]} to {periods[1]}, and by how much?",
                ]
                return random.choice(templates)

            if concept == "change_in_fcf_margin":
                return f"How did {self.company}'s free cash flow margin change from {periods[0]} to {periods[1]}?"

        # Fallback
        return f"[Could not compile: {query.expression}]"

    def _compile_lookup(self, var: str, year: str) -> str:
        label = QueryComposer.VAR_LABELS.get(var, var)
        templates = [
            f"What was the {label} in {year}?",
            f"How much did {self.company} report for {label} in fiscal year {year}?",
            f"What was {self.company}'s {label} as of {year}?",
            f"According to the {year} financial statements, what was the {label}?",
        ]
        return random.choice(templates)

    def _compile_change(self, var: str, y1: str, y2: str) -> str:
        label = QueryComposer.VAR_LABELS.get(var, var)
        templates = [
            f"What was the change in {label} from {y1} to {y2}?",
            f"By how much did {label} increase or decrease between {y1} and {y2}?",
            f"What was the year-over-year change in {self.company}'s {label} from {y1} to {y2}?",
            f"How did {label} change from fiscal year {y1} to {y2}?",
        ]
        return random.choice(templates)

    def _compile_simple_ratio(self, num_var: str, den_var: str, year: str) -> str:
        num_label = QueryComposer.VAR_LABELS.get(num_var, num_var)
        den_label = QueryComposer.VAR_LABELS.get(den_var, den_var)
        templates = [
            f"What was {num_label} as a proportion of {den_label} in {year}?",
            f"What was the ratio of {num_label} to {den_label} in {year}?",
            f"For every dollar of {den_label}, how much {num_label} did {self.company} have in {year}?",
            f"What percentage of {den_label} was {num_label} in {year}?",
        ]
        return random.choice(templates)


# ============================================================
# 5. DATASET GENERATOR — Full Pipeline
# ============================================================

@dataclass
class QAPair:
    """A single verified question-answer pair."""
    id: str
    company_name: str
    company_ticker: str
    industry: str
    question: str
    answer: Any
    answer_formatted: str
    formal_program: str
    computation_trace: str
    difficulty: int
    concept_name: Optional[str]
    category: str
    variables_used: list
    periods_used: list
    answer_type: str


class DatasetGenerator:
    """Orchestrates the full pipeline: data → queries → execution → NL → verified pairs."""

    def __init__(self, seed: int = 42):
        self.data_gen = FinancialDataGenerator(seed=seed)
        self.composer = QueryComposer()
        self.seed = seed

    def generate_dataset(self, num_companies: int = 10, max_per_level: int = 8) -> list:
        """Generate a full dataset of verified QA pairs."""
        all_pairs = []
        pair_id = 0

        for i in range(num_companies):
            # Generate company data
            company = self.data_gen.generate_company()
            company_name = company["company"]["name"]
            company_ticker = company["company"]["ticker"]
            industry = company["company"]["industry_label"]

            # Compose queries at all levels
            queries = self.composer.compose_all(company, max_per_level=max_per_level)

            # Execute and compile
            engine = ExecutionEngine(company)
            compiler = NLCompiler(company_name)

            for query in queries:
                # Execute for ground truth
                result = engine.execute(query)
                if not result["success"] or result["answer"] is None:
                    continue

                # Compile to natural language
                nl_question = compiler.compile(query)
                if nl_question.startswith("[Could not compile"):
                    continue

                pair_id += 1
                qa = QAPair(
                    id=f"QA-{pair_id:05d}",
                    company_name=company_name,
                    company_ticker=company_ticker,
                    industry=industry,
                    question=nl_question,
                    answer=result["answer"],
                    answer_formatted=result["answer_formatted"],
                    formal_program=query.expression,
                    computation_trace=result["computation_trace"],
                    difficulty=query.difficulty,
                    concept_name=query.concept_name,
                    category=query.category,
                    variables_used=query.variables_used,
                    periods_used=query.periods_used,
                    answer_type=result["type"],
                )
                all_pairs.append(qa)

        return all_pairs

    def export_csv(self, pairs: list, filepath: str):
        """Export QA pairs to CSV."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "company", "ticker", "industry", "difficulty", "category",
                "concept", "question", "answer", "answer_formatted", "answer_type",
                "formal_program", "computation_trace", "variables_used", "periods_used"
            ])
            for qa in pairs:
                writer.writerow([
                    qa.id, qa.company_name, qa.company_ticker, qa.industry,
                    qa.difficulty, qa.category, qa.concept_name or "",
                    qa.question, qa.answer, qa.answer_formatted, qa.answer_type,
                    qa.formal_program, qa.computation_trace,
                    "|".join(qa.variables_used), "|".join(qa.periods_used),
                ])

    def export_jsonl(self, pairs: list, filepath: str):
        """Export QA pairs to JSONL."""
        with open(filepath, "w", encoding="utf-8") as f:
            for qa in pairs:
                f.write(json.dumps(asdict(qa), ensure_ascii=False) + "\n")

    def print_stats(self, pairs: list):
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print(f"  SYNTHETIC FINANCIAL QA DATASET — STATISTICS")
        print(f"{'='*60}")
        print(f"  Total QA pairs: {len(pairs)}")
        print()

        # By difficulty
        by_diff = {}
        for p in pairs:
            by_diff.setdefault(p.difficulty, []).append(p)
        print("  By difficulty level:")
        for d in sorted(by_diff):
            labels = {1: "Lookup", 2: "Single Op", 3: "Named Concept", 4: "Multi-Step", 5: "Comparative"}
            print(f"    Level {d} ({labels.get(d, '?')}): {len(by_diff[d])} questions")

        # By category
        by_cat = {}
        for p in pairs:
            by_cat.setdefault(p.category, []).append(p)
        print("\n  By category:")
        for cat in sorted(by_cat):
            print(f"    {cat}: {len(by_cat[cat])}")

        # By answer type
        by_type = {}
        for p in pairs:
            by_type.setdefault(p.answer_type, []).append(p)
        print("\n  By answer type:")
        for t in sorted(by_type):
            print(f"    {t}: {len(by_type[t])}")

        # Companies
        companies = set(p.company_name for p in pairs)
        print(f"\n  Unique companies: {len(companies)}")

        # Sample questions
        print(f"\n{'='*60}")
        print(f"  SAMPLE QA PAIRS")
        print(f"{'='*60}")
        samples = random.sample(pairs, min(10, len(pairs)))
        for qa in samples:
            print(f"\n  [{qa.id}] Difficulty: {qa.difficulty} | Category: {qa.category}")
            print(f"  Company: {qa.company_name} ({qa.company_ticker})")
            print(f"  Q: {qa.question}")
            print(f"  A: {qa.answer_formatted}")
            print(f"  Program: {qa.formal_program}")
            print(f"  Trace: {qa.computation_trace}")
        print()


# ============================================================
# 6. MAIN — Run the pipeline
# ============================================================

if __name__ == "__main__":
    print("Initializing Synthetic Financial QA Generator...")
    generator = DatasetGenerator(seed=42)

    print("Generating dataset...")
    pairs = generator.generate_dataset(num_companies=20, max_per_level=10)

    # Print stats
    generator.print_stats(pairs)

    # Export
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "financial_qa_dataset.csv")
    jsonl_path = os.path.join(output_dir, "financial_qa_dataset.jsonl")

    generator.export_csv(pairs, csv_path)
    generator.export_jsonl(pairs, jsonl_path)

    print(f"\nExported {len(pairs)} QA pairs:")
    print(f"  CSV:   {csv_path}")
    print(f"  JSONL: {jsonl_path}")
    print("\nDone!")

