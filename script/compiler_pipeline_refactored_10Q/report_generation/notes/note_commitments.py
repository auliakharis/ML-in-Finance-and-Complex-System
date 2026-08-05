"""
FORM 10-Q — NOTE: COMMITMENTS AND CONTINGENCIES

Covers operating lease obligations, purchase commitments, legal contingencies,
and guarantees. These are disclosures rather than calculated values, so most
fields are raw inputs or narrative strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LeaseMaturitySchedule:
    """
    Maturity schedule for operating or finance lease obligations.
    Keys are period labels as used in the filing.
    e.g. {"Remainder of 2025": 500, "2026": 1_800, "2027": 1_600,
          "2028": 1_400, "2029": 1_200, "Thereafter": 5_000}
    """
    lease_type: str                                     # "Operating" or "Finance"
    future_payments: Dict[str, float] = field(default_factory=dict)
    less_imputed_interest: Optional[float] = None
    weighted_average_remaining_term: Optional[float] = None   # years
    weighted_average_discount_rate: Optional[float] = None    # %

    @property
    def total_undiscounted_payments(self) -> float:
        return sum(self.future_payments.values())

    @property
    def present_value_of_lease_liabilities(self) -> Optional[float]:
        if self.less_imputed_interest is None:
            return None
        return self.total_undiscounted_payments - self.less_imputed_interest


@dataclass
class PurchaseCommitment:
    """A contractual purchase obligation (e.g. supply agreements, take-or-pay)."""
    description: str
    total_committed: Optional[float] = None
    due_within_one_year: Optional[float] = None
    due_one_to_three_years: Optional[float] = None
    due_three_to_five_years: Optional[float] = None
    due_after_five_years: Optional[float] = None


@dataclass
class LegalContingency:
    """A pending legal matter or regulatory proceeding."""
    description: str
    estimated_loss_range_low: Optional[float] = None
    estimated_loss_range_high: Optional[float] = None
    accrued_liability: Optional[float] = None
    outcome: Optional[str] = None   # e.g. "cannot be estimated", "reasonably possible"


@dataclass
class NoteCommitments:
    """
    Note — Commitments and Contingencies.

    lease_schedules: one per lease type (Operating, Finance).
    purchase_commitments: contractual obligations with suppliers.
    legal_contingencies: pending litigation and regulatory matters.
    disclosures: free-form narrative items keyed by topic.
    """
    lease_schedules: List[LeaseMaturitySchedule] = field(default_factory=list)
    purchase_commitments: List[PurchaseCommitment] = field(default_factory=list)
    legal_contingencies: List[LegalContingency] = field(default_factory=list)
    disclosures: Dict[str, str] = field(default_factory=dict)

    def get_lease_schedule(self, lease_type: str) -> Optional[LeaseMaturitySchedule]:
        for s in self.lease_schedules:
            if s.lease_type.lower() == lease_type.lower():
                return s
        return None