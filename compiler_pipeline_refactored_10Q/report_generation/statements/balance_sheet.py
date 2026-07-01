"""
FORM 10-Q — CONDENSED CONSOLIDATED BALANCE SHEETS

Assets = Liabilities + Shareholders' Equity.
All section totals and the balance check are @property — never constructor arguments.

Structure:
    Assets
      CurrentAssets        → total_current_assets
      NonCurrentAssets     → total_non_current_assets
                           → total_assets
    Liabilities
      CurrentLiabilities   → total_current_liabilities
      NonCurrentLiabilities→ total_non_current_liabilities
                           → total_liabilities
    ShareholdersEquity     → total_shareholders_equity
                           → total_liabilities_and_equity  (must equal total_assets)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

@dataclass
class CurrentAssets:
    """
    Current assets section (expected to be converted to cash within 12 months).
    All fields optional — populate only what the filing reports.
    other_lines: company-specific line items keyed by filing label.
    """
    cash_and_cash_equivalents: Optional[float] = None
    short_term_investments: Optional[float] = None           # marketable securities
    accounts_receivable_net: Optional[float] = None
    inventories: Optional[float] = None
    vendor_non_trade_receivables: Optional[float] = None     # Apple-specific
    prepaid_expenses_and_other: Optional[float] = None
    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.cash_and_cash_equivalents,
                self.short_term_investments,
                self.accounts_receivable_net,
                self.inventories,
                self.vendor_non_trade_receivables,
                self.prepaid_expenses_and_other,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


@dataclass
class NonCurrentAssets:
    """
    Non-current (long-term) assets section.
    other_lines: company-specific line items keyed by filing label.
    """
    long_term_marketable_securities: Optional[float] = None
    property_plant_and_equipment_net: Optional[float] = None
    operating_lease_right_of_use_assets: Optional[float] = None
    goodwill: Optional[float] = None
    intangible_assets_net: Optional[float] = None
    deferred_tax_assets: Optional[float] = None
    other_non_current_assets: Optional[float] = None
    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.long_term_marketable_securities,
                self.property_plant_and_equipment_net,
                self.operating_lease_right_of_use_assets,
                self.goodwill,
                self.intangible_assets_net,
                self.deferred_tax_assets,
                self.other_non_current_assets,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


# ---------------------------------------------------------------------------
# Liabilities
# ---------------------------------------------------------------------------

@dataclass
class CurrentLiabilities:
    """
    Current liabilities section (due within 12 months).
    other_lines: company-specific line items keyed by filing label.
    """
    accounts_payable: Optional[float] = None
    deferred_revenue_current: Optional[float] = None
    accrued_expenses_and_other: Optional[float] = None
    short_term_debt: Optional[float] = None
    current_portion_of_long_term_debt: Optional[float] = None
    operating_lease_liabilities_current: Optional[float] = None
    income_taxes_payable: Optional[float] = None
    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.accounts_payable,
                self.deferred_revenue_current,
                self.accrued_expenses_and_other,
                self.short_term_debt,
                self.current_portion_of_long_term_debt,
                self.operating_lease_liabilities_current,
                self.income_taxes_payable,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


@dataclass
class NonCurrentLiabilities:
    """
    Non-current (long-term) liabilities section.
    other_lines: company-specific line items keyed by filing label.
    """
    long_term_debt: Optional[float] = None
    operating_lease_liabilities_non_current: Optional[float] = None
    deferred_revenue_non_current: Optional[float] = None
    deferred_tax_liabilities: Optional[float] = None
    other_non_current_liabilities: Optional[float] = None
    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.long_term_debt,
                self.operating_lease_liabilities_non_current,
                self.deferred_revenue_non_current,
                self.deferred_tax_liabilities,
                self.other_non_current_liabilities,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


# ---------------------------------------------------------------------------
# Shareholders' equity
# ---------------------------------------------------------------------------

@dataclass
class ShareholdersEquity:
    """
    Shareholders' equity section.
    other_lines: company-specific line items keyed by filing label.
    """
    common_stock_and_additional_paid_in_capital: Optional[float] = None
    retained_earnings: Optional[float] = None                # or accumulated deficit
    accumulated_other_comprehensive_income_loss: Optional[float] = None
    treasury_stock: Optional[float] = None                   # negative value
    noncontrolling_interest: Optional[float] = None
    other_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.common_stock_and_additional_paid_in_capital,
                self.retained_earnings,
                self.accumulated_other_comprehensive_income_loss,
                self.treasury_stock,
                self.noncontrolling_interest,
            ]
            if v is not None
        ]
        known += list(self.other_lines.values())
        return sum(known)


# ---------------------------------------------------------------------------
# Balance sheet
# ---------------------------------------------------------------------------

@dataclass
class BalanceSheet:
    """
    Condensed Consolidated Balance Sheet as of a specific date.

    total_assets, total_liabilities, total_liabilities_and_equity are all
    @property — never constructor arguments.

    Use validate() to confirm Assets = Liabilities + Equity.
    """
    as_of: str                              # e.g. "December 27, 2025"

    current_assets: CurrentAssets
    non_current_assets: NonCurrentAssets

    current_liabilities: CurrentLiabilities
    non_current_liabilities: NonCurrentLiabilities

    shareholders_equity: ShareholdersEquity

    currency: str = "USD"
    unit: str = "millions"

    # ------------------------------------------------------------------
    # Derived totals
    # ------------------------------------------------------------------

    @property
    def total_current_assets(self) -> float:
        return self.current_assets.total

    @property
    def total_non_current_assets(self) -> float:
        return self.non_current_assets.total

    @property
    def total_assets(self) -> float:
        return self.total_current_assets + self.total_non_current_assets

    @property
    def total_current_liabilities(self) -> float:
        return self.current_liabilities.total

    @property
    def total_non_current_liabilities(self) -> float:
        return self.non_current_liabilities.total

    @property
    def total_liabilities(self) -> float:
        return self.total_current_liabilities + self.total_non_current_liabilities

    @property
    def total_shareholders_equity(self) -> float:
        return self.shareholders_equity.total

    @property
    def total_liabilities_and_equity(self) -> float:
        return self.total_liabilities + self.total_shareholders_equity

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, tolerance: float = 1.0) -> list[str]:
        """
        Checks the fundamental balance sheet identity: Assets = Liabilities + Equity.
        Returns a list of warning strings; empty = balanced.
        """
        warnings: list[str] = []
        diff = abs(self.total_assets - self.total_liabilities_and_equity)
        if diff > tolerance:
            warnings.append(
                f"Balance sheet does not balance: "
                f"total_assets={self.total_assets:,.1f}, "
                f"total_liabilities_and_equity={self.total_liabilities_and_equity:,.1f}, "
                f"diff={diff:,.1f}"
            )
        return warnings

    def validate_strict(self, tolerance: float = 1.0) -> None:
        issues = self.validate(tolerance)
        if issues:
            raise ValueError("\n".join(issues))

    def __repr__(self) -> str:
        return (
            f"BalanceSheet("
            f"as_of={self.as_of!r}, "
            f"total_assets={self.total_assets:,.0f}, "
            f"total_liabilities={self.total_liabilities:,.0f}, "
            f"total_equity={self.total_shareholders_equity:,.0f})"
        )