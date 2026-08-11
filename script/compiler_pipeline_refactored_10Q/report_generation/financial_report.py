"""
10-Q Financial Report — top-level assembly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
10-Q PRIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS A 10-Q?
  A quarterly report filed with the SEC by every US public company.
  Contains condensed (unaudited) financial statements for the quarter.

FISCAL YEAR vs CALENDAR YEAR
  A fiscal year is a company's 12-month accounting period. It does not
  have to start on January 1 — companies choose their own start date.

    Apple:    fiscal year starts late September  -> Q1 = Oct-Dec
    Walmart:  fiscal year starts early February  -> Q1 = Feb-Apr
    Most banks/tech: fiscal year starts January 1 (calendar year)

  Consequence: two companies filing for "Three Months Ended December 2025"
  may be at completely different points in their fiscal year. Apple would
  be in Q1; a calendar-year company would be in Q4.

WHAT IS YTD?
  Year-to-date — the cumulative period from the fiscal year start to the
  current quarter end.

    Q1:  quarter = 3 months,  YTD = 3 months  (identical)
    Q2:  quarter = 3 months,  YTD = 6 months
    Q3:  quarter = 3 months,  YTD = 9 months

THE FIVE REQUIRED STATEMENTS (Regulation S-X Rule 10-01)
  Each statement measures something different, so the SEC requires
  different time periods for each:

  1. Income statement
       Measures activity (revenue, expenses) over a period of time.
       Required periods: current quarter + YTD, each with a prior-year
       comparative. This gives up to 4 columns:
         [Q this year] [Q last year] [YTD this year] [YTD last year]
       Exception: in Q1 the quarter and YTD are identical (both 3 months),
       so they collapse to 2 columns — as seen in Apple's Q1 form.

  2. Statement of comprehensive income
       Extends the income statement by adding other comprehensive income
       (OCI) — gains/losses that bypass the income statement, such as
       foreign currency translation and unrealized securities gains.
       Uses the same periods as the income statement.

  3. Balance sheet
       Measures a stock — what the company owns and owes at a single
       point in time (not a period). Reported "as of" a date, not
       "for the period ended."

       Required columns (always 2):
         [current quarter-end]  [prior fiscal year-end]

       Optional 3rd column: same quarter of the prior year. Companies
       add this voluntarily for comparability — it is never required by
       the SEC. Whether a company includes it is purely a presentation
       choice; some always do, some never do, regardless of which quarter
       is being filed. For example, Walmart's Q3 shows 3 columns while
       Delta's Q3 shows only 2.

       Deriving the quarter number: the form does not explicitly label
       itself "Q1", "Q2", or "Q3" on the face of the financial statements.
       You derive it by comparing the period end date against the company's
       fiscal year start date. This is why fiscal_year_start is a field on
       FinancialReport.

  4. Statement of shareholders' equity
       Tracks how equity components changed over time (net income,
       dividends, buybacks, stock compensation). Presented as a YTD
       rollforward with subtotals for each individual quarter within
       the YTD — so a Q3 filing shows Q1, Q2, Q3 running totals for
       both the current and prior year.

  5. Statement of cash flows
       Tracks cash inflows and outflows across operating, investing,
       and financing activities. YTD only — two columns: current YTD
       + prior YTD. No per-quarter subtotals required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .statements import (
    StatementOfOperations,
    StatementOfComprehensiveIncome,
    BalanceSheet,
    StatementOfShareholdersEquity,
    StatementOfCashFlows,
)
from .notes import Notes


@dataclass
class FinancialStatements:
    """
    The five required statements per SEC Regulation S-X Rule 10-01,
    each holding all required columns as separate objects.

    Income statement / Comprehensive income (up to 4 columns):
      - current_quarter      : always required
      - prior_year_quarter   : always required
      - current_ytd          : None for Q1 (identical to current_quarter)
      - prior_year_ytd       : None for Q1 (identical to prior_year_quarter)

    Balance sheet (2 required + 1 optional):
      - current              : current quarter-end (always required)
      - prior_fy_end         : prior fiscal year-end (always required)
      - prior_year_quarter   : same quarter prior year (optional, company choice)

    Shareholders' equity (2 YTD rollforwards):
      - current_ytd          : current year YTD with per-quarter subtotals
      - prior_ytd            : prior year YTD with per-quarter subtotals

    Cash flows (2 YTD columns):
      - current_ytd          : current year YTD
      - prior_ytd            : prior year YTD
    """

    # ── Income statement ──────────────────────────────────────
    ops_current_quarter: StatementOfOperations
    ops_prior_year_quarter: StatementOfOperations
    ops_current_ytd: Optional[StatementOfOperations]        # None for Q1
    ops_prior_year_ytd: Optional[StatementOfOperations]     # None for Q1

    # ── Comprehensive income ───────────────────────────────────
    ci_current_quarter: StatementOfComprehensiveIncome
    ci_prior_year_quarter: StatementOfComprehensiveIncome
    ci_current_ytd: Optional[StatementOfComprehensiveIncome]
    ci_prior_year_ytd: Optional[StatementOfComprehensiveIncome]

    # ── Balance sheet ──────────────────────────────────────────
    bs_current: BalanceSheet
    bs_prior_fy_end: BalanceSheet
    bs_prior_year_quarter: Optional[BalanceSheet]           # voluntary 3rd column

    # ── Cash flows ─────────────────────────────────────────────
    cf_current_ytd: StatementOfCashFlows
    cf_prior_ytd: StatementOfCashFlows

    # ── Shareholders' equity (optional) ──────────────────────────
    se_current_ytd: Optional[StatementOfShareholdersEquity] = None
    se_prior_ytd: Optional[StatementOfShareholdersEquity] = None

    def validate(self, tolerance: float = 1.0) -> dict[str, list[str]]:
        """
        Run validation on all populated statements.
        Returns {statement_name: [warning, ...]}; empty lists = clean.
        """
        results: dict[str, list[str]] = {}
        for name, stmt in [
            ("ops_current_quarter",    self.ops_current_quarter),
            ("ops_prior_year_quarter", self.ops_prior_year_quarter),
            ("ops_current_ytd",        self.ops_current_ytd),
            ("ops_prior_year_ytd",     self.ops_prior_year_ytd),
        ]:
            if stmt is not None:
                w = stmt.validate()
                if w:
                    results[name] = w

        for name, bs in [
            ("bs_current",           self.bs_current),
            ("bs_prior_fy_end",      self.bs_prior_fy_end),
            ("bs_prior_year_quarter", self.bs_prior_year_quarter),
        ]:
            if bs is not None:
                w = bs.validate(tolerance)
                if w:
                    results[name] = w

        for name, cf in [
            ("cf_current_ytd", self.cf_current_ytd),
            ("cf_prior_ytd",   self.cf_prior_ytd),
        ]:
            w = cf.validate(tolerance)
            if w:
                results[name] = w

        return results


@dataclass
class FinancialReport:
    """
    Complete 10-Q financial report for one filing period.

    company_name:         e.g. "Apple Inc."
    period:               e.g. "Three Months Ended December 27, 2025"
    fiscal_year:          e.g. "2026"
    quarter:              1, 2, or 3
    fiscal_year_start:    e.g. "September 28, 2025"
    form_type:            "10-Q" or "10-K"
    """
    financial_statements: FinancialStatements
    notes: Notes

    company_name: Optional[str] = None
    ticker: Optional[str] = None
    period: Optional[str] = None
    fiscal_year: Optional[str] = None
    quarter: Optional[int] = None
    fiscal_year_start: Optional[str] = None
    form_type: str = "10-Q"

    def validate(self, tolerance: float = 1.0) -> dict[str, list[str]]:
        return self.financial_statements.validate(tolerance)

    def __repr__(self) -> str:
        return (
            f"FinancialReport("
            f"company={self.company_name!r}, "
            f"period={self.period!r}, "
            f"Q{self.quarter}, "
            f"form={self.form_type!r})"
        )