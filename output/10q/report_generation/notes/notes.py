"""
Notes aggregator — combines all individual note templates into a single
Notes dataclass that can be attached to a FinancialReport.

All note fields are Optional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .note_revenue import NoteRevenue
from .note_debt import NoteDebt
from .note_equity import NoteEquity
from .note_segments import NoteSegments
from .note_commitments import NoteCommitments


@dataclass
class Notes:
    """
    All notes to the condensed consolidated financial statements.

    Standard notes included here:
      note_revenue      — disaggregated net sales (Note 2 in Apple, separate table in Walmart)
      note_debt         — term debt, commercial paper, credit facilities
      note_equity       — share repurchases, dividends, stock-based compensation
      note_segments     — segment information and reconciliation to consolidated totals
      note_commitments  — leases, purchase obligations, legal contingencies

    other_notes: escape hatch for any additional company-specific notes,
    keyed by the note label used in the filing (e.g. "Note 7 — Income Taxes").
    """
    note_revenue: Optional[NoteRevenue] = None
    note_debt: Optional[NoteDebt] = None
    note_equity: Optional[NoteEquity] = None
    note_segments: Optional[NoteSegments] = None
    note_commitments: Optional[NoteCommitments] = None

    # Catch-all for any note not covered by the standard templates.
    # Values are free-form strings (narrative) or dicts for structured data.
    other_notes: Dict[str, object] = field(default_factory=dict)