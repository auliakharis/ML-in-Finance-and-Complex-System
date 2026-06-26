"""
FORM 10-Q — NOTE: EQUITY

Covers share repurchase programs, dividends declared, and stock-based
compensation disclosures that accompany the Statement of Shareholders' Equity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ShareRepurchaseProgram:
    """
    Details of a board-authorized share repurchase program.
    Amounts in the same unit as the financial statements (typically millions).
    Shares in thousands.
    """
    authorization_amount: Optional[float] = None        # total authorized
    shares_repurchased_period: Optional[float] = None   # shares repurchased this period (thousands)
    amount_repurchased_period: Optional[float] = None   # $ value this period
    shares_repurchased_ytd: Optional[float] = None
    amount_repurchased_ytd: Optional[float] = None
    remaining_authorization: Optional[float] = None
    program_description: Optional[str] = None           # e.g. "March 2023 ASR Program"


@dataclass
class DividendDeclaration:
    """A single dividend declaration."""
    declaration_date: Optional[str] = None
    record_date: Optional[str] = None
    payment_date: Optional[str] = None
    amount_per_share: Optional[float] = None
    total_amount: Optional[float] = None


@dataclass
class StockBasedCompensation:
    """
    Stock-based compensation disclosure.
    expense: total SBC recognized in the period (matches cash flow add-back).
    unrecognized_compensation: remaining expense to be recognized.
    weighted_average_recognition_period: in years.
    """
    expense: Optional[float] = None
    unrecognized_compensation: Optional[float] = None
    weighted_average_recognition_period: Optional[float] = None     # years
    other_fields: Dict[str, float] = field(default_factory=dict)


@dataclass
class NoteEquity:
    """
    Note — Equity.

    Aggregates share repurchase programs, dividend declarations, and
    stock-based compensation disclosures.
    """
    repurchase_programs: List[ShareRepurchaseProgram] = field(default_factory=list)
    dividends: List[DividendDeclaration] = field(default_factory=list)
    stock_based_compensation: Optional[StockBasedCompensation] = None
    disclosures: Dict[str, str] = field(default_factory=dict)

    @property
    def total_repurchased_period(self) -> float:
        return sum(
            p.amount_repurchased_period or 0.0
            for p in self.repurchase_programs
        )