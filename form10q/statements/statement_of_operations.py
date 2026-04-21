"""
FORM 10-Q — CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS
Template supporting both detailed breakdowns and high-level totals.

Key design decisions:
- Revenue and CostsAndExpenses are their own classes.
- ALL derived quantities (totals, gross_profit, operating_income, net_income, etc.)
  are @property values — never constructor arguments. There is no way to enter
  an inconsistent value; the math is always correct by construction.
- net_sales and cost_of_sales accept either a detailed object or a plain float.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Union


# ---------------------------------------------------------------------------
# Net sales breakdown (optional — pass a plain float to Revenue for one line)
# ---------------------------------------------------------------------------

@dataclass
class NetSales:
    """
    Breakdown of net sales by category.
    Pass a plain float to Revenue instead if the form shows a single line.

    components: keyed by the label used in the filing.
      e.g. {"Products": 113_743, "Services": 30_013}                        
    """
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.components.values())

    @classmethod
    def from_dict(cls, d: dict) -> "NetSales":
        return cls(components=d.get("components", {}))


# ---------------------------------------------------------------------------
# Revenue  (net sales + any other operating revenue lines)
# ---------------------------------------------------------------------------

@dataclass
class Revenue:
    """
    Full revenues section as reported on the income statement.

    Three separate fields — all optional, any combination is valid:

    - net_sales: a NetSales object (with component breakdown), a plain float,
      or None. Printed with a "Net sales" subtotal line.
        Apple:   NetSales(components={"Products": 113_743, "Services": 30_013})
        Walmart: 177_769  (plain float)
        Delta:   None     (Delta has no "Net sales" concept)

    - other_components: additional revenue lines without a subtotal. Used for companies whose
      primary revenue is not called "Net sales".
        Delta:   {"Passenger": 9_100, "Cargo and other": 800}
        Bank:    {"Net interest income": 5_000, "Non-interest income": 2_000}

    - other_operating_revenue: ancillary revenue lines without a subtotal.
        Walmart: {"Membership and other income": 1_727}

    total_revenues = net_sales_total + sum(other_components) + sum(other_operating_revenue)
    gross_profit   = net_sales_total - cost_of_sales  (None when net_sales is None)
    """
    net_sales: Optional[Union[NetSales, float]] = None
    other_components: Dict[str, float] = field(default_factory=dict)
    other_operating_revenue: Dict[str, float] = field(default_factory=dict)

    @property
    def net_sales_total(self) -> float:
        if self.net_sales is None:
            return 0.0
        if isinstance(self.net_sales, NetSales):
            return self.net_sales.total
        return self.net_sales

    @property
    def total_other_components(self) -> float:
        return sum(self.other_components.values())

    @property
    def total_other_operating_revenue(self) -> float:
        return sum(self.other_operating_revenue.values())

    @property
    def total_revenues(self) -> float:
        return (self.net_sales_total
                + self.total_other_components
                + self.total_other_operating_revenue)

    @classmethod
    def from_dict(cls, d: dict) -> "Revenue":
        ns = d.get("net_sales")
        if isinstance(ns, dict):
            ns = NetSales.from_dict(ns)
        return cls(
            net_sales=ns,
            other_components=d.get("other_components", {}),
            other_operating_revenue=d.get("other_operating_revenue", {}),
        )


# ---------------------------------------------------------------------------
# Cost of sales breakdown (optional — pass a plain float for single-line cost)
# ---------------------------------------------------------------------------

@dataclass
class CostOfSales:
    """
    Breakdown of cost of sales by category.
    Pass a plain float to CostsAndExpenses if the form does not break it out.

    components: keyed by the label used in the filing.
      e.g. {"Products": 67_478, "Services": 7_047}  (Apple)
    """
    components: Dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.components.values())

    @classmethod
    def from_dict(cls, d: dict) -> "CostOfSales":
        return cls(
            components=d.get("components", {}),
        )


# ---------------------------------------------------------------------------
# Operating expenses
# ---------------------------------------------------------------------------

@dataclass
class OperatingExpenses:
    """
    Standard operating expense lines.
    Add company-specific lines via CostsAndExpenses.other_expense_lines.
    """
    research_and_development: Optional[float] = None
    selling_general_and_administrative: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None

    @property
    def total(self) -> float:
        known = [
            v for v in [
                self.research_and_development,
                self.selling_general_and_administrative,
                self.depreciation_and_amortization,
            ]
            if v is not None
        ]
        return sum(known)

    @classmethod
    def from_dict(cls, d: dict) -> "OperatingExpenses":
        return cls(
            research_and_development=d.get("research_and_development"),
            selling_general_and_administrative=d.get("selling_general_and_administrative"),
            depreciation_and_amortization=d.get("depreciation_and_amortization"),
        )


# ---------------------------------------------------------------------------
# Costs and expenses  (cost_of_sales + operating_expenses + any extras)
# ---------------------------------------------------------------------------

@dataclass
class CostsAndExpenses:
    """
    Full costs and expenses section as reported on the income statement.

    - cost_of_sales: pass a CostOfSales object or a plain float.
    - operating_expenses: pass an OperatingExpenses object or a plain float.
    - other_expense_lines: any additional lines keyed by filing label.
      e.g. {"Restructuring charges": 120.0}

    total_costs_and_expenses is always auto-derived — never set manually.
    """
    cost_of_sales: Union[CostOfSales, float]
    operating_expenses: Union[OperatingExpenses, float]
    other_expense_lines: Dict[str, float] = field(default_factory=dict)

    @property
    def cost_of_sales_total(self) -> float:
        if isinstance(self.cost_of_sales, CostOfSales):
            return self.cost_of_sales.total
        return self.cost_of_sales

    @property
    def operating_expenses_total(self) -> float:
        if isinstance(self.operating_expenses, OperatingExpenses):
            return self.operating_expenses.total
        return self.operating_expenses

    @property
    def total_costs_and_expenses(self) -> float:
        return (
            self.cost_of_sales_total
            + self.operating_expenses_total
            + sum(self.other_expense_lines.values())
        )

    @classmethod
    def from_dict(cls, d: dict) -> "CostsAndExpenses":
        cos = d.get("cost_of_sales")
        if isinstance(cos, dict):
            cos = CostOfSales.from_dict(cos)
        opex = d.get("operating_expenses")
        if isinstance(opex, dict):
            opex = OperatingExpenses.from_dict(opex)
        return cls(
            cost_of_sales=cos,
            operating_expenses=opex,
            other_expense_lines=d.get("other_expense_lines", {}),
        )


# ---------------------------------------------------------------------------
# EPS / share counts
# ---------------------------------------------------------------------------

@dataclass
class EarningsPerShare:
    basic: float
    diluted: float

    def validate(self) -> None:
        if self.diluted > self.basic:
            raise ValueError(
                f"Diluted EPS ({self.diluted}) cannot exceed basic EPS ({self.basic})."
            )


@dataclass
class SharesUsed:
    """Shares used in computing EPS (in thousands, as reported)."""
    basic: float
    diluted: float

    def validate(self) -> None:
        if self.diluted < self.basic:
            raise ValueError(
                f"Diluted shares ({self.diluted}) cannot be less than basic shares ({self.basic})."
            )


# ---------------------------------------------------------------------------
# Primary statement
# ---------------------------------------------------------------------------

@dataclass
class StatementOfOperations:
    """
    Condensed Consolidated Statement of Operations for a single period.

    RAW INPUTS ONLY — every derived quantity is a @property:
        gross_profit                     = net_sales - cost_of_sales
        operating_income                 = total_revenues - total_costs_and_expenses
        income_before_taxes              = operating_income + other_income_expense_net
        net_income                       = income_before_taxes - provision_for_income_taxes
        net_income_attributable_to_parent = net_income - noncontrolling_interest

    Works for both Apple-style (net_sales == total_revenues) and
    Walmart-style (total_revenues = net_sales + membership income) filings.
    """

    period: str                         # e.g. "Three Months Ended October 31, 2025"

    # ---- Revenue and costs (the only structural inputs) ----
    revenue: Revenue
    costs_and_expenses: CostsAndExpenses

    # ---- Below-the-line items ----
    # For a single net figure use other_income_expense_net.
    # For a detailed breakdown (Walmart-style) use other_income_expense_lines;
    # other_income_expense_net will be ignored if lines are provided.
    other_income_expense_net: float = 0.0
    other_income_expense_lines: Dict[str, float] = field(default_factory=dict)

    # ---- Taxes (raw input) ----
    provision_for_income_taxes: float = 0.0

    # ---- Noncontrolling interest (raw input, optional) ----
    net_income_attributable_to_noncontrolling_interest: Optional[float] = None

    # ---- Per-share data (raw inputs, optional) ----
    earnings_per_share: Optional[EarningsPerShare] = None
    shares_used_in_computing_eps: Optional[SharesUsed] = None

    # ---- Dividends (optional) ----
    dividends_per_share: Optional[float] = None

    # ---- Metadata ----
    currency: str = "USD"
    unit: str = "millions"

    # ------------------------------------------------------------------
    # Derived quantities — all @property, never constructor arguments
    # ------------------------------------------------------------------

    @property
    def gross_profit(self) -> Optional[float]:
        """
        net_sales - cost_of_sales. Returns None when net_sales is None
        (e.g. airlines, banks — gross profit is not meaningful for them).
        """
        if self.revenue.net_sales is None:
            return None
        return (
            self.revenue.net_sales_total
            - self.costs_and_expenses.cost_of_sales_total
        )

    @property
    def operating_income(self) -> float:
        """total_revenues - total_costs_and_expenses."""
        return (
            self.revenue.total_revenues
            - self.costs_and_expenses.total_costs_and_expenses
        )

    @property
    def _other_income_expense(self) -> float:
        """Resolve to detailed lines sum if provided, else the net field."""
        if self.other_income_expense_lines:
            return sum(self.other_income_expense_lines.values())
        return self.other_income_expense_net

    @property
    def income_before_taxes(self) -> float:
        return self.operating_income + self._other_income_expense

    @property
    def net_income(self) -> float:
        return self.income_before_taxes - self.provision_for_income_taxes

    @property
    def net_income_attributable_to_parent(self) -> Optional[float]:
        if self.net_income_attributable_to_noncontrolling_interest is None:
            return None
        return self.net_income - self.net_income_attributable_to_noncontrolling_interest

    # ------------------------------------------------------------------
    # Margin ratios
    # ------------------------------------------------------------------

    @property
    def gross_profit_pct(self) -> Optional[float]:
        if self.revenue.net_sales_total:
            return self.gross_profit / self.revenue.net_sales_total * 100
        return None

    @property
    def operating_margin_pct(self) -> Optional[float]:
        if self.revenue.net_sales_total:
            return self.operating_income / self.revenue.net_sales_total * 100
        return None

    @property
    def net_margin_pct(self) -> Optional[float]:
        if self.revenue.net_sales_total:
            return self.net_income / self.revenue.net_sales_total * 100
        return None

    # ------------------------------------------------------------------
    # Validation — EPS / share sanity only; arithmetic is always correct
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """
        Since all P&L quantities are derived by formula, the only things
        left to validate are EPS ordering and share count ordering.
        Returns a list of warning strings; empty = all good.
        """
        warnings: list[str] = []
        try:
            if self.earnings_per_share:
                self.earnings_per_share.validate()
        except ValueError as e:
            warnings.append(str(e))
        try:
            if self.shares_used_in_computing_eps:
                self.shares_used_in_computing_eps.validate()
        except ValueError as e:
            warnings.append(str(e))
        return warnings

    def validate_strict(self) -> None:
        """Like validate() but raises ValueError if any issue is found."""
        issues = self.validate()
        if issues:
            raise ValueError(
                "Statement of Operations failed validation:\n"
                + "\n".join(f"  • {w}" for w in issues)
            )

    # ------------------------------------------------------------------
    # YoY comparison
    # ------------------------------------------------------------------

    def yoy_delta(self, prior: "StatementOfOperations") -> Dict[str, Optional[float]]:
        """Absolute and % changes vs. a prior-period statement."""

        def pct(cur: float, prev: float) -> Optional[float]:
            return (cur - prev) / abs(prev) * 100 if prev != 0 else None

        return {
            "net_sales_delta":         self.revenue.net_sales_total - prior.revenue.net_sales_total,
            "net_sales_pct":           pct(self.revenue.net_sales_total, prior.revenue.net_sales_total),
            "total_revenues_delta":    self.revenue.total_revenues - prior.revenue.total_revenues,
            "total_revenues_pct":      pct(self.revenue.total_revenues, prior.revenue.total_revenues),
            "gross_profit_delta":      self.gross_profit - prior.gross_profit,
            "gross_profit_pct_change": pct(self.gross_profit, prior.gross_profit),
            "operating_income_delta":  self.operating_income - prior.operating_income,
            "operating_income_pct":    pct(self.operating_income, prior.operating_income),
            "net_income_delta":        self.net_income - prior.net_income,
            "net_income_pct":          pct(self.net_income, prior.net_income),
        }

    def __repr__(self) -> str:
        return (
            f"StatementOfOperations("
            f"period={self.period!r}, "
            f"net_sales={self.revenue.net_sales_total:,.0f}, "
            f"total_revenues={self.revenue.total_revenues:,.0f}, "
            f"gross_profit={self.gross_profit:,.0f} ({self.gross_profit_pct:.1f}%), "
            f"operating_income={self.operating_income:,.0f} ({self.operating_margin_pct:.1f}%), "
            f"net_income={self.net_income:,.0f} ({self.net_margin_pct:.1f}%)"
            f")"
        )