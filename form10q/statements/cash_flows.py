"""
FORM 10-Q — CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS

Three sections: Operating, Investing, Financing.
All section totals and the net change in cash are @property.

Convention: inflows are positive, outflows are negative.
  e.g. capital_expenditures = -3_000  (cash out)
       proceeds_from_debt    = +5_000  (cash in)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Operating activities
# ---------------------------------------------------------------------------

@dataclass
class OperatingActivities:
    """
    Cash flows from operating activities (indirect method).

    Starts from net income, adds back non-cash items, then captures
    working capital changes. All fields optional.
    other_lines: company-specific items keyed by filing label.
    """
    net_income: Optional[float] = None

    # Non-cash adjustments
    depreciation_and_amortization: Optional[float] = None
    stock_based_compensation: Optional[float] = None
    deferred_income_taxes: Optional[float] = None
    other_non_cash_items: Optional[float] = None

    # Working capital changes (positive = source of cash, negative = use)
    change_in_accounts_receivable: Optional[float] = None
    change_in_inventories: Optional[float] = None
    change_in_accounts_payable: Optional[float] = None
    change_in_deferred_revenue: Optional[float] = None
    change_in_other_working_capital: Optional[float] = None

    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.net_income,
                self.depreciation_and_amortization,
                self.stock_based_compensation,
                self.deferred_income_taxes,
                self.other_non_cash_items,
                self.change_in_accounts_receivable,
                self.change_in_inventories,
                self.change_in_accounts_payable,
                self.change_in_deferred_revenue,
                self.change_in_other_working_capital,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


# ---------------------------------------------------------------------------
# Investing activities
# ---------------------------------------------------------------------------

@dataclass
class InvestingActivities:
    """
    Cash flows from investing activities.
    Outflows (purchases, capex) should be entered as negative values.
    other_lines: company-specific items keyed by filing label.
    """
    capital_expenditures: Optional[float] = None                    # negative
    purchases_of_marketable_securities: Optional[float] = None      # negative
    proceeds_from_maturities_of_securities: Optional[float] = None  # positive
    proceeds_from_sales_of_securities: Optional[float] = None       # positive
    acquisitions_net_of_cash: Optional[float] = None                # negative
    proceeds_from_divestitures: Optional[float] = None              # positive
    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.capital_expenditures,
                self.purchases_of_marketable_securities,
                self.proceeds_from_maturities_of_securities,
                self.proceeds_from_sales_of_securities,
                self.acquisitions_net_of_cash,
                self.proceeds_from_divestitures,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


# ---------------------------------------------------------------------------
# Financing activities
# ---------------------------------------------------------------------------

@dataclass
class FinancingActivities:
    """
    Cash flows from financing activities.
    Outflows (repurchases, repayments, dividends) should be negative.
    other_lines: company-specific items keyed by filing label.
    """
    proceeds_from_issuance_of_debt: Optional[float] = None          # positive
    repayments_of_debt: Optional[float] = None                      # negative
    proceeds_from_commercial_paper: Optional[float] = None          # positive
    repayments_of_commercial_paper: Optional[float] = None          # negative
    share_repurchases: Optional[float] = None                       # negative
    proceeds_from_stock_option_exercises: Optional[float] = None    # positive
    dividends_paid: Optional[float] = None                          # negative
    finance_lease_payments: Optional[float] = None                  # negative
    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.proceeds_from_issuance_of_debt,
                self.repayments_of_debt,
                self.proceeds_from_commercial_paper,
                self.repayments_of_commercial_paper,
                self.share_repurchases,
                self.proceeds_from_stock_option_exercises,
                self.dividends_paid,
                self.finance_lease_payments,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


# ---------------------------------------------------------------------------
# Statement of cash flows
# ---------------------------------------------------------------------------

@dataclass
class StatementOfCashFlows:
    """
    Condensed Consolidated Statement of Cash Flows.

    net_change_in_cash and closing_cash are @property — never constructor args.
    Use validate() to confirm opening + net change = closing.
    """
    period: str                             # e.g. "Three Months Ended December 27, 2025"

    operating_activities: OperatingActivities
    investing_activities: InvestingActivities
    financing_activities: FinancingActivities

    # ---- Cash position (raw inputs) ----
    opening_cash_and_equivalents: float = 0.0
    effect_of_exchange_rate_on_cash: float = 0.0    # FX impact on cash balances

    currency: str = "USD"
    unit: str = "millions"

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def net_cash_from_operating(self) -> float:
        return self.operating_activities.total

    @property
    def net_cash_from_investing(self) -> float:
        return self.investing_activities.total

    @property
    def net_cash_from_financing(self) -> float:
        return self.financing_activities.total

    @property
    def net_change_in_cash(self) -> float:
        return (
            self.net_cash_from_operating
            + self.net_cash_from_investing
            + self.net_cash_from_financing
            + self.effect_of_exchange_rate_on_cash
        )

    @property
    def closing_cash_and_equivalents(self) -> float:
        return self.opening_cash_and_equivalents + self.net_change_in_cash

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, tolerance: float = 1.0) -> list[str]:
        """
        Checks that all three sections sum to net_change_in_cash.
        Returns a list of warning strings; empty = all good.
        """
        warnings: list[str] = []
        expected = (
            self.net_cash_from_operating
            + self.net_cash_from_investing
            + self.net_cash_from_financing
            + self.effect_of_exchange_rate_on_cash
        )
        diff = abs(expected - self.net_change_in_cash)
        if diff > tolerance:
            warnings.append(
                f"Cash flow does not reconcile: "
                f"expected net change={expected:,.1f}, "
                f"actual={self.net_change_in_cash:,.1f}"
            )
        return warnings

    def __repr__(self) -> str:
        return (
            f"StatementOfCashFlows("
            f"period={self.period!r}, "
            f"operating={self.net_cash_from_operating:+,.0f}, "
            f"investing={self.net_cash_from_investing:+,.0f}, "
            f"financing={self.net_cash_from_financing:+,.0f}, "
            f"net_change={self.net_change_in_cash:+,.0f}, "
            f"closing_cash={self.closing_cash_and_equivalents:,.0f})"
        )