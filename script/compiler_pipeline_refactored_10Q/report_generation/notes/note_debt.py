"""
FORM 10-Q — NOTE: DEBT

Covers term debt, commercial paper, revolving credit facilities, and
finance leases. Each instrument is a DebtInstrument; the note aggregates
all of them and derives totals as @property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DebtInstrument:
    """
    A single debt instrument or tranche.

    Examples:
      - Fixed-rate note: name="2.500% Notes due Feb 2025", face_value=2_000,
        carrying_value=1_998, maturity_date="February 2025", interest_rate=2.5
      - Revolving credit: name="Revolving Credit Facility", face_value=None,
        carrying_value=0, capacity=10_000
    """
    name: str
    carrying_value: float                   # book value after discount/premium/issuance costs
    face_value: Optional[float] = None      # par / principal amount
    interest_rate: Optional[float] = None   # annual rate in %
    maturity_date: Optional[str] = None     # e.g. "February 2025"
    is_current: bool = False                # True if due within 12 months
    capacity: Optional[float] = None        # for revolving facilities
    unamortized_discount_premium: Optional[float] = None
    unamortized_issuance_costs: Optional[float] = None
    other_fields: Dict[str, float] = field(default_factory=dict)


@dataclass
class NoteDebt:
    """
    Note — Debt.

    instruments: all individual debt tranches / facilities.
    Totals for current and non-current portions are @property.

    commercial_paper_outstanding: reported separately by many companies.
    """
    instruments: List[DebtInstrument] = field(default_factory=list)
    commercial_paper_outstanding: Optional[float] = None

    # Narrative disclosures (e.g. covenant descriptions, fair value disclosures)
    disclosures: Dict[str, str] = field(default_factory=dict)

    @property
    def total_debt(self) -> float:
        return sum(i.carrying_value for i in self.instruments)

    @property
    def current_portion(self) -> float:
        return sum(i.carrying_value for i in self.instruments if i.is_current)

    @property
    def non_current_portion(self) -> float:
        return sum(i.carrying_value for i in self.instruments if not i.is_current)

    def get(self, name: str) -> Optional[DebtInstrument]:
        for i in self.instruments:
            if i.name == name:
                return i
        return None