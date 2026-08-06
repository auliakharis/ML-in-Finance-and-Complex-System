"""
FORM 10-Q — NOTE: SEGMENT INFORMATION

Segment reporting under ASC 280. Each operating segment reports its own
revenue, operating income (or the metric management uses), and assets.
The reconciliation back to consolidated totals is a @property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Segment:
    """
    Financial data for a single reportable segment.

    net_sales and operating_income are the most common metrics.
    other_metrics: any additional segment-level KPIs the company discloses
    (e.g. "Comparable sales growth", "eCommerce net sales").
    """
    name: str                                           # e.g. "Walmart U.S."
    period: str
    net_sales: Optional[float] = None
    operating_income: Optional[float] = None
    total_assets: Optional[float] = None
    depreciation_and_amortization: Optional[float] = None
    capital_expenditures: Optional[float] = None
    other_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentReconciliation:
    """
    Reconciliation of segment totals to consolidated totals.
    Unallocated items (corporate overhead, eliminations) are the plug.
    All consolidated totals are @property.
    """
    period: str
    segments: List[Segment] = field(default_factory=list)

    # Reconciling items between segment totals and consolidated figures
    # e.g. {"Corporate and other": -1_200, "Intercompany eliminations": -50}
    reconciling_items_net_sales: Dict[str, float] = field(default_factory=dict)
    reconciling_items_operating_income: Dict[str, float] = field(default_factory=dict)

    @property
    def total_segment_net_sales(self) -> float:
        return sum(s.net_sales or 0.0 for s in self.segments)

    @property
    def consolidated_net_sales(self) -> float:
        return self.total_segment_net_sales + sum(self.reconciling_items_net_sales.values())

    @property
    def total_segment_operating_income(self) -> float:
        return sum(s.operating_income or 0.0 for s in self.segments)

    @property
    def consolidated_operating_income(self) -> float:
        return self.total_segment_operating_income + sum(
            self.reconciling_items_operating_income.values()
        )

    def get_segment(self, name: str) -> Optional[Segment]:
        for s in self.segments:
            if s.name == name:
                return s
        return None


@dataclass
class NoteSegments:
    """
    Note — Segment Information.

    reconciliations: one SegmentReconciliation per reported period.
    disclosures: narrative items (segment descriptions, basis of measurement).
    """
    reconciliations: List[SegmentReconciliation] = field(default_factory=list)
    disclosures: Dict[str, str] = field(default_factory=dict)

    def get_period(self, period: str) -> Optional[SegmentReconciliation]:
        for r in self.reconciliations:
            if r.period == period:
                return r
        return None