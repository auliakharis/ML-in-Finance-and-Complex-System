"""
FORM 10-Q — CONDENSED CONSOLIDATED STATEMENTS OF SHAREHOLDERS' EQUITY

Tracks the rollforward of each equity component across one or more periods.
Each period captures opening balances, all movements, and derives closing balances
as @property values — never constructor arguments.

Common movements:
  - Net income
  - Other comprehensive income / (loss)
  - Share repurchases (buybacks)
  - Stock-based compensation
  - Dividends declared
  - Common stock issuances (options, RSUs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EquityRollforwardPeriod:
    """
    Equity activity for a single reporting period (quarter or YTD).

    Opening balances are raw inputs. Every closing balance is a @property.
    other_movements: escape hatch for company-specific equity transactions,
    keyed by filing label (positive = increase, negative = decrease).
    """
    period: str                                         # e.g. "Three Months Ended December 27, 2025"

    # ---- Opening balances (raw inputs) ----
    opening_common_stock_and_apic: float = 0.0
    opening_retained_earnings: float = 0.0
    opening_accumulated_oci: float = 0.0
    opening_treasury_stock: float = 0.0                 # typically negative
    opening_noncontrolling_interest: float = 0.0

    # ---- Movements (raw inputs, all optional) ----
    net_income: Optional[float] = None
    other_comprehensive_income: Optional[float] = None
    dividends_declared: Optional[float] = None          # negative (reduces retained earnings)
    share_repurchases: Optional[float] = None           # negative (increases treasury stock)
    stock_based_compensation: Optional[float] = None    # increases APIC
    common_stock_issued: Optional[float] = None         # options/RSU exercises
    other_movements: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Closing balances — all @property
    # ------------------------------------------------------------------

    @property
    def closing_common_stock_and_apic(self) -> float:
        delta = sum([
            v for v in [self.stock_based_compensation, self.common_stock_issued]
            if v is not None
        ])
        delta += sum(
            v for k, v in self.other_movements.items()
            if "apic" in k.lower() or "stock" in k.lower() or "issu" in k.lower()
        )
        return self.opening_common_stock_and_apic + delta

    @property
    def closing_retained_earnings(self) -> float:
        delta = sum([
            v for v in [self.net_income, self.dividends_declared]
            if v is not None
        ])
        delta += sum(
            v for k, v in self.other_movements.items()
            if "retained" in k.lower() or "dividend" in k.lower()
        )
        return self.opening_retained_earnings + delta

    @property
    def closing_accumulated_oci(self) -> float:
        delta = self.other_comprehensive_income or 0.0
        return self.opening_accumulated_oci + delta

    @property
    def closing_treasury_stock(self) -> float:
        delta = self.share_repurchases or 0.0
        return self.opening_treasury_stock + delta

    @property
    def closing_noncontrolling_interest(self) -> float:
        # NCI changes are less standardized; route through other_movements
        delta = sum(
            v for k, v in self.other_movements.items()
            if "noncontrolling" in k.lower() or "nci" in k.lower()
        )
        return self.opening_noncontrolling_interest + delta

    @property
    def opening_total_equity(self) -> float:
        return (
            self.opening_common_stock_and_apic
            + self.opening_retained_earnings
            + self.opening_accumulated_oci
            + self.opening_treasury_stock
            + self.opening_noncontrolling_interest
        )

    @property
    def closing_total_equity(self) -> float:
        return (
            self.closing_common_stock_and_apic
            + self.closing_retained_earnings
            + self.closing_accumulated_oci
            + self.closing_treasury_stock
            + self.closing_noncontrolling_interest
        )

    @property
    def net_change_in_equity(self) -> float:
        return self.closing_total_equity - self.opening_total_equity

    def __repr__(self) -> str:
        return (
            f"EquityRollforwardPeriod("
            f"period={self.period!r}, "
            f"opening_equity={self.opening_total_equity:,.0f}, "
            f"net_change={self.net_change_in_equity:+,.0f}, "
            f"closing_equity={self.closing_total_equity:,.0f})"
        )


@dataclass
class StatementOfShareholdersEquity:
    """
    Condensed Consolidated Statement of Shareholders' Equity.

    Contains one EquityRollforwardPeriod per reported column.
    10-Qs typically show one quarter (or YTD); annual reports show full year.
    """
    periods: List[EquityRollforwardPeriod]
    currency: str = "USD"
    unit: str = "millions"

    @property
    def latest(self) -> EquityRollforwardPeriod:
        return self.periods[-1]

    @property
    def closing_total_equity(self) -> float:
        return self.latest.closing_total_equity

    def __repr__(self) -> str:
        period_reprs = ", ".join(p.period for p in self.periods)
        return (
            f"StatementOfShareholdersEquity("
            f"periods=[{period_reprs}], "
            f"closing_equity={self.closing_total_equity:,.0f})"
        )