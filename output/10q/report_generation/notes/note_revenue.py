"""
FORM 10-Q — NOTE: REVENUE (Disaggregated Net Sales)

Supports both Apple-style (fixed product categories) and
Walmart-style (free-form merchandise categories per segment) disaggregation.

Apple Note 2:
    iPhone / Mac / iPad / Wearables, Home and Accessories / Services
    + deferred revenue disclosure

Walmart Disaggregated Revenues:
    Per segment (e.g. Walmart U.S.) broken into merchandise categories
    (Grocery / General merchandise / Health and wellness / Other)
    + eCommerce callout per segment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Apple-style: named product/service categories
# ---------------------------------------------------------------------------

@dataclass
class DisaggregatedNetSales:
    """
    Net sales disaggregated by product/service category.
    Use for companies with a consistent, named category breakdown (Apple).

    categories: keyed by the label used in the filing.
      e.g. {"iPhone": 85_269, "Mac": 8_386, "iPad": 8_595,
             "Wearables, Home and Accessories": 11_493, "Services": 30_013}

    total is always derived — never a constructor argument.
    """
    period: str
    categories: Dict[str, float] = field(default_factory=dict)

    # Deferred revenue disclosure (Apple Note 2)
    deferred_revenue_included_in_sales: Optional[float] = None

    @property
    def total(self) -> float:
        return sum(self.categories.values())

    def __repr__(self) -> str:
        cats = ", ".join(f"{k}={v:,.0f}" for k, v in self.categories.items())
        return f"DisaggregatedNetSales(period={self.period!r}, total={self.total:,.0f}, [{cats}])"


# ---------------------------------------------------------------------------
# Walmart-style: per-segment, free-form merchandise categories
# ---------------------------------------------------------------------------

@dataclass
class SegmentDisaggregatedRevenue:
    """
    Net sales for a single segment, disaggregated by merchandise category.
    Use for multi-segment retailers (Walmart U.S., Walmart International, Sam's Club).

    categories: keyed by the label used in the filing.
      e.g. {"Grocery": 71_713, "General merchandise": 27_366,
             "Health and wellness": 18_379, "Other categories": 3_220}

    ecommerce_net_sales: optional callout of eCommerce within this segment.
    total is always derived.
    """
    segment_name: str                               # e.g. "Walmart U.S."
    period: str
    categories: Dict[str, float] = field(default_factory=dict)
    ecommerce_net_sales: Optional[float] = None     # subset of total, not additive

    @property
    def total(self) -> float:
        return sum(self.categories.values())

    def __repr__(self) -> str:
        return (
            f"SegmentDisaggregatedRevenue("
            f"segment={self.segment_name!r}, "
            f"period={self.period!r}, "
            f"total={self.total:,.0f})"
        )


# ---------------------------------------------------------------------------
# Note container
# ---------------------------------------------------------------------------

@dataclass
class NoteRevenue:
    """
    Note — Revenue (disaggregated net sales).

    Supports two presentation styles:
    1. Single-entity breakdown (Apple): populate `disaggregated_net_sales`.
    2. Multi-segment breakdown (Walmart): populate `segment_revenues`.

    Both can be populated for companies that present both views.
    Multiple periods (current quarter + YTD) can be stored as separate entries.
    """
    # Apple-style: one DisaggregatedNetSales per reported period
    disaggregated_net_sales: List[DisaggregatedNetSales] = field(default_factory=list)

    # Walmart-style: one SegmentDisaggregatedRevenue per segment per period
    segment_revenues: List[SegmentDisaggregatedRevenue] = field(default_factory=list)

    # Free-form narrative disclosures (e.g. deferred revenue policy, recognition timing)
    disclosures: Dict[str, str] = field(default_factory=dict)

    def get_period(self, period: str) -> Optional[DisaggregatedNetSales]:
        """Return the DisaggregatedNetSales entry matching a period string."""
        for entry in self.disaggregated_net_sales:
            if entry.period == period:
                return entry
        return None

    def get_segment(self, segment_name: str, period: str) -> Optional[SegmentDisaggregatedRevenue]:
        """Return the SegmentDisaggregatedRevenue for a given segment and period."""
        for entry in self.segment_revenues:
            if entry.segment_name == segment_name and entry.period == period:
                return entry
        return None