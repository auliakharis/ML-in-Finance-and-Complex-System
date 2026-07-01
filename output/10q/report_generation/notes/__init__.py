from .note_revenue import (
    DisaggregatedNetSales,
    SegmentDisaggregatedRevenue,
    NoteRevenue,
)
from .note_debt import (
    DebtInstrument,
    NoteDebt,
)
from .note_equity import (
    ShareRepurchaseProgram,
    DividendDeclaration,
    StockBasedCompensation,
    NoteEquity,
)
from .note_segments import (
    Segment,
    SegmentReconciliation,
    NoteSegments,
)
from .note_commitments import (
    LeaseMaturitySchedule,
    PurchaseCommitment,
    LegalContingency,
    NoteCommitments,
)
from .notes import Notes

__all__ = [
    # revenue
    "DisaggregatedNetSales", "SegmentDisaggregatedRevenue", "NoteRevenue",
    # debt
    "DebtInstrument", "NoteDebt",
    # equity
    "ShareRepurchaseProgram", "DividendDeclaration", "StockBasedCompensation", "NoteEquity",
    # segments
    "Segment", "SegmentReconciliation", "NoteSegments",
    # commitments
    "LeaseMaturitySchedule", "PurchaseCommitment", "LegalContingency", "NoteCommitments",
    # aggregator
    "Notes",
]