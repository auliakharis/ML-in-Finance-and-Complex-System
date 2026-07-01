"""
FORM 10-Q — CONDENSED CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME

Comprehensive income = net income + other comprehensive income (OCI).
OCI captures gains/losses that bypass the income statement:
  - Foreign currency translation adjustments
  - Unrealized gains/losses on available-for-sale securities
  - Pension / post-retirement benefit adjustments
  - Cash flow hedge adjustments

All totals are @property — never constructor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class OtherComprehensiveIncome:
    """
    Other Comprehensive Income / (Loss) section.
    Each line is optional — only populate what the filing reports.
    Losses are negative values.

    other_lines: escape hatch for any company-specific OCI items,
    keyed by the label used in the filing.
    """
    foreign_currency_translation: Optional[float] = None
    unrealized_gains_losses_on_securities: Optional[float] = None
    pension_and_postretirement_adjustments: Optional[float] = None
    cash_flow_hedge_adjustments: Optional[float] = None
    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.foreign_currency_translation,
                self.unrealized_gains_losses_on_securities,
                self.pension_and_postretirement_adjustments,
                self.cash_flow_hedge_adjustments,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


@dataclass
class StatementOfComprehensiveIncome:
    """
    Condensed Consolidated Statement of Comprehensive Income.

    net_income must be passed in — it comes from StatementOfOperations.
    All totals are derived @property values.

    For companies that combine this with the income statement (single-statement
    presentation), set other_comprehensive_income with only the relevant lines.
    """
    period: str
    net_income: float                           # from StatementOfOperations.net_income
    other_comprehensive_income: OtherComprehensiveIncome = field(
        default_factory=OtherComprehensiveIncome
    )

    # Noncontrolling interest split (optional)
    oci_attributable_to_noncontrolling_interest: Optional[float] = None

    currency: str = "USD"
    unit: str = "millions"

    @property
    def total_other_comprehensive_income(self) -> float:
        return self.other_comprehensive_income.total

    @property
    def comprehensive_income(self) -> float:
        return self.net_income + self.total_other_comprehensive_income

    @property
    def comprehensive_income_attributable_to_parent(self) -> Optional[float]:
        if self.oci_attributable_to_noncontrolling_interest is None:
            return None
        return self.comprehensive_income - self.oci_attributable_to_noncontrolling_interest

    def __repr__(self) -> str:
        return (
            f"StatementOfComprehensiveIncome("
            f"period={self.period!r}, "
            f"net_income={self.net_income:,.0f}, "
            f"oci={self.total_other_comprehensive_income:,.0f}, "
            f"comprehensive_income={self.comprehensive_income:,.0f})"
        )