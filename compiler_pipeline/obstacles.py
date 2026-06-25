"""Toggleable obstacles for the compiler pipeline."""

from __future__ import annotations

import json
import random
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from semantic_analyzer import AnalysisResult

if TYPE_CHECKING:
    from data_prep import YearlyNumericColumn, YearlyNumericColumns
from tree import Atom, DerivedExpr, Expr, Node, TimeAgg

OBSTACLE_NAMES: tuple[str, ...] = (
    "big_numbers",
    "useless_info",
    "unit_scale_change",
    "negation",
    "conditional",
    "balanced_tree",
)

QUESTION_OBSTACLES: frozenset[str] = frozenset(
    {"useless_info", "unit_scale_change", "negation", "conditional"}
)
DATA_PREP_OBSTACLES: frozenset[str] = frozenset({"big_numbers"})
GENERATION_OBSTACLES: frozenset[str] = frozenset({"balanced_tree"})

BIG_NUMBERS_SCALE_FACTOR: float = 100.0
RATIO_COLUMNS: frozenset[str] = frozenset({"income_tax"})

USELESS_INFO_TEMPLATES_FILE = "config/useless_info_templates.json"
M_USD_UNIT: str = "M_USD"

# Leaf + derived concept ids that must have template families (plus "generic").
REQUIRED_USELESS_INFO_CONCEPTS: frozenset[str] = frozenset(
    {
        "revenue",
        "cost_of_goods_sold",
        "operating_expenses",
        "non_operating_expenses",
        "income_tax",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "cash",
        "accounts_receivable",
        "inventories",
        "short_term_investments",
        "current_liabilities",
        "capex",
        "dividends_paid",
        "shares_outstanding",
        "stock_price",
        "employees",
        "gross_profit",
        "operating_income",
        "pretax_income",
        "income_tax_expense",
        "net_income",
        "current_assets",
        "longterm_assets",
        "longterm_liabilities",
    }
)

TAX_USELESS_INFO_CONCEPTS: tuple[str, ...] = (
    "income_tax",
    "income_tax_expense",
    "pretax_income",
    "net_income",
)

# (display unit label, divisor applied to M_USD amounts in financial data)
UNIT_SCALE_OPTIONS: tuple[tuple[str, float], ...] = (
    ("thousands", 1_000.0),
    ("millions", 1_000_000.0),
    ("billions", 1_000_000_000.0),
    ("trillions", 1_000_000_000_000.0),
)

NEGATION_PREFIXES: tuple[str, ...] = (
    "Do not apply any adjustment; simply compute: ",
    "It is not the case that you should ignore the following — determine: ",
    "Without changing the definition, find: ",
)

NEGATION_SUFFIXES: tuple[str, ...] = (
    " (compute the value as stated, without sign changes.)",
    " — answer using the figures exactly as given.",
)


def validate_obstacle_name(name: str | None) -> None:
    if name is not None and name not in OBSTACLE_NAMES:
        raise ValueError(
            f"Unknown obstacle {name!r}. Choose one of: {', '.join(OBSTACLE_NAMES)}"
        )


def validate_big_numbers_factor(factor: float) -> None:
    if factor <= 0:
        raise ValueError(f"--big-numbers-factor must be > 0, got {factor}")


def pick_unit_scale() -> tuple[str, float]:
    """Return (unit label, factor) for unit_scale_change (data is shown as value / factor)."""
    return random.choice(UNIT_SCALE_OPTIONS)


def m_usd_concepts_from_metadata(
    concept_metadata: dict[str, dict[str, object]],
) -> frozenset[str]:
    return frozenset(
        name
        for name, meta in concept_metadata.items()
        if meta.get("unit") == M_USD_UNIT
    )


def scale_m_usd_atom_values(
    atoms: dict[str, Atom],
    factor: float,
) -> dict[str, Atom]:
    """Return atoms with M_USD leaf values divided by factor (display units)."""
    if factor <= 0:
        raise ValueError(f"unit scale factor must be > 0, got {factor}")
    scaled: dict[str, Atom] = {}
    for key, atom in atoms.items():
        if atom.unit == M_USD_UNIT:
            scaled[key] = atom.model_copy(update={"value": atom.value / factor})
        else:
            scaled[key] = atom
    return scaled


def _scale_numeric_field_value(value: str, factor: float) -> str:
    try:
        num = float(value)
    except ValueError:
        return value
    scaled = num / factor
    if scaled == int(scaled):
        return str(int(scaled))
    return str(scaled)


def scale_csv_rows(
    rows: list[dict[str, str]],
    m_usd_columns: frozenset[str],
    factor: float,
) -> list[dict[str, str]]:
    scaled_rows: list[dict[str, str]] = []
    for row in rows:
        new_row = dict(row)
        for col in m_usd_columns:
            if col in new_row and new_row[col]:
                new_row[col] = _scale_numeric_field_value(new_row[col], factor)
        scaled_rows.append(new_row)
    return scaled_rows


def scale_financial_spreadsheet_rows(
    rows: list[dict[str, str]],
    m_usd_columns: frozenset[str],
    factor: float,
) -> list[dict[str, str]]:
    return scale_csv_rows(rows, m_usd_columns, factor)


def unit_scale_question_prefix(unit_label: str) -> str:
    return (
        f"The numbers in the financial_data are all given in {unit_label}. "
        f"Please answer in numbers, not {unit_label}. "
    )


def scale_answer_from_display_units(
    answer: float,
    *,
    semantic_type: object,
    factor: float,
) -> float:
    """Convert an evaluator result on display-unit data to raw-number ground truth."""
    from tree import SemanticType

    if semantic_type == SemanticType.amount:
        return answer * factor
    return answer


def scale_numeric_ranges(
    yearly_numeric_cols: YearlyNumericColumns,
    factor: float = BIG_NUMBERS_SCALE_FACTOR,
) -> YearlyNumericColumns:
    """Return a copy of YearlyNumericColumns with amount ranges scaled up."""
    scaled_define: dict[str, YearlyNumericColumn] = {}
    for col_name, col_def in yearly_numeric_cols.define.items():
        col_copy = deepcopy(col_def)
        if col_name in RATIO_COLUMNS:
            scaled_define[col_name] = col_copy
            continue
        if col_copy.range is not None:
            col_copy.range = {
                "min": col_copy.range["min"] * factor,
                "max": col_copy.range["max"] * factor,
            }
        scaled_define[col_name] = col_copy
    return yearly_numeric_cols.model_copy(update={"define": scaled_define})


def load_financial_spreadsheet(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return payload


def load_useless_info_clauses(path: Path) -> dict[str, list[str]]:
    """Load clause families: category name -> list of irrelevant sentences."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    families: dict[str, list[str]] = {}
    for family, clauses in payload.items():
        if not isinstance(family, str) or not isinstance(clauses, list):
            raise ValueError(f"Invalid family entry in {path}: {family!r}")
        if not all(isinstance(c, str) for c in clauses):
            raise ValueError(f"Family {family!r} must contain only strings")
        if not clauses:
            raise ValueError(f"Family {family!r} must not be empty")
        families[family] = clauses
    if not families:
        raise ValueError(f"No clause families found in {path}")
    return families


def useless_info_family_names(path: Path) -> tuple[str, ...]:
    return tuple(sorted(load_useless_info_clauses(path)))


def validate_useless_info_family(family: str | None, path: Path) -> None:
    if family is None:
        return
    families = load_useless_info_clauses(path)
    if family not in families:
        raise ValueError(
            f"Unknown useless_info family {family!r}. "
            f"Choose one of: {', '.join(sorted(families))}"
        )


def validate_useless_info_template_coverage(path: Path) -> None:
    """Ensure every required concept has a non-empty useless_info family."""
    families = load_useless_info_clauses(path)
    missing = sorted(REQUIRED_USELESS_INFO_CONCEPTS - set(families))
    if missing:
        raise ValueError(
            "useless_info_templates.json is missing families for: "
            f"{', '.join(missing)}. "
            "Run: python scripts/ensure_useless_info_coverage.py"
        )
    empty = sorted(
        c for c in REQUIRED_USELESS_INFO_CONCEPTS if not families.get(c)
    )
    if empty:
        raise ValueError(
            f"useless_info_templates.json has empty families for: {', '.join(empty)}"
        )
    if "generic" not in families or not families["generic"]:
        raise ValueError('useless_info_templates.json must include a non-empty "generic" family')


def select_useless_info_families(
    families: dict[str, list[str]],
    family: str | None = None,
) -> dict[str, list[str]]:
    """Restrict to one family, or return all families when family is None."""
    if family is None:
        return families
    if family not in families:
        raise ValueError(
            f"Unknown useless_info family {family!r}. "
            f"Choose one of: {', '.join(sorted(families))}"
        )
    return {family: families[family]}


def _collect_derived_names(expr: Expr, out: set[str]) -> None:
    if isinstance(expr, DerivedExpr):
        out.add(expr.name)
        if expr.expr is not None:
            _collect_derived_names(expr.expr, out)
    elif isinstance(expr, Node):
        _collect_derived_names(expr.left, out)
        _collect_derived_names(expr.right, out)
    elif isinstance(expr, TimeAgg) and expr.expr is not None:
        _collect_derived_names(expr.expr, out)


def concepts_in_expr(expr: Expr, leaf_atoms: list[Atom]) -> frozenset[str]:
    """Concept ids present in a bound expression (leaf atoms + derived node names)."""
    atom_by_key = {a.key: a for a in leaf_atoms}
    concepts: set[str] = set()
    for leaf in expr.flatten_leaves():
        if leaf.key and leaf.key in atom_by_key:
            concepts.add(atom_by_key[leaf.key].concept)
    derived: set[str] = set()
    _collect_derived_names(expr, derived)
    concepts |= derived
    return frozenset(concepts)


def _concept_label(concept: str) -> str:
    return concept.replace("_", " ")


def _score_concept_in_question(concept: str, question_lc: str) -> int:
    """Higher score = stronger match between concept id and question wording."""
    label = _concept_label(concept)
    if not question_lc:
        return 0
    if label in question_lc:
        return 100 + len(label)
    words = [w for w in label.split() if len(w) > 2]
    if not words:
        return 0
    word_hits = sum(1 for w in words if w in question_lc)
    if word_hits == len(words):
        return 50 + word_hits
    return word_hits


def pick_useless_info_family(
    families: dict[str, list[str]],
    concepts: frozenset[str],
    *,
    forced_family: str | None = None,
    question: str | None = None,
) -> str:
    if forced_family is not None:
        return forced_family
    eligible = [c for c in concepts if c in families and c != "generic"]
    if eligible:
        question_lc = (question or "").lower()
        if "tax" in question_lc:
            for preferred in TAX_USELESS_INFO_CONCEPTS:
                if preferred in eligible:
                    return preferred
        return max(eligible, key=lambda c: (_score_concept_in_question(c, question_lc), c))
    if "generic" in families:
        return "generic"
    return random.choice(list(families.keys()))


def pick_useless_info_clause(
    families: dict[str, list[str]],
    *,
    family_name: str,
) -> str:
    if family_name not in families:
        raise ValueError(
            f"Unknown useless_info family {family_name!r}. "
            f"Choose one of: {', '.join(sorted(families))}"
        )
    clauses = families[family_name]
    if not clauses:
        raise ValueError(f"useless_info family {family_name!r} has no clauses.")
    return random.choice(clauses)


def random_useless_info_placeholder_value(name: str) -> str:
    """Return a random formatted value for a useless-info template placeholder."""
    lower = name.lower()
    if lower == "multiplier":
        return f"{random.uniform(1.05, 1.4):.2f}"
    if lower == "margin_pts" or lower.endswith("_pts"):
        return f"{random.randint(3, 12)} percentage points"
    if lower == "decade":
        return f"{random.choice(range(1960, 2000, 10))}s"
    if lower == "months":
        return f"{random.choice(range(1,28))}"
    if lower == "amount":
        return f"{random.choice(range(5000,34000))}"
    if lower == "year" or lower.startswith("year"):
        return str(random.randint(1900, 2018))
    if lower == "rate" or lower.startswith("rate"):
        return str(random.randint(1, 12))
    if "pct" in lower or lower.endswith("_percent"):
        return str(random.randint(5, 45))
    return str(random.randint(5, 45))


def build_useless_info_mapping(template: str) -> dict[str, str]:
    placeholders = re.findall(r"\{(\w+)\}", template)
    mapping: dict[str, str] = {}
    for name in placeholders:
        if name not in mapping:
            mapping[name] = random_useless_info_placeholder_value(name)
    return mapping


def format_useless_info_clause(clause: str) -> str:
    return format_template(clause, build_useless_info_mapping(clause))


def format_template(template: str, mapping: dict[str, str]) -> str:
    result = template
    for key, value in mapping.items():
        result = result.replace("{" + key + "}", value)
    missing = re.findall(r"\{(\w+)\}", result)
    if missing:
        raise ValueError(f"Unresolved placeholders in template: {missing}")
    return result


def _row_matches_entity(row: dict[str, str], entity: str) -> bool:
    return row.get("company_name") == entity or row.get("ticker") == entity


@dataclass
class ObstacleContext:
    question: str
    analysis: AnalysisResult
    expr: Expr
    leaf_atoms: list[Atom]
    spreadsheet_rows: list[dict[str, str]]
    base_dir: Path
    unit_scale_label: str | None = None
    useless_info_family: str | None = None
    useless_info_family_used: str | None = None

    def entities(self) -> set[str]:
        entities: set[str] = set()
        meaning = self.analysis.meaning
        if meaning.entity:
            entities.add(meaning.entity)
        for atom in self.leaf_atoms:
            if atom.entity:
                entities.add(atom.entity)
        return entities

    def _prepend_or_append(self, snippet: str) -> str:
        snippet = snippet.strip()
        if not snippet.endswith((".", "!", "?")):
            snippet = snippet + "."
        if random.random() < 0.5:
            return f"{snippet} {self.question}"
        return f"{self.question} {snippet}"

    def _pick_spreadsheet_row(
        self,
        *,
        prefer_entity: str | None = None,
        exclude_entities: set[str] | None = None,
    ) -> dict[str, str]:
        exclude = exclude_entities or set()
        if prefer_entity:
            candidates = [
                r
                for r in self.spreadsheet_rows
                if _row_matches_entity(r, prefer_entity)
                and r.get("company_name") not in exclude
            ]
            if candidates:
                return random.choice(candidates)
        candidates = [
            r for r in self.spreadsheet_rows if r.get("company_name") not in exclude
        ]
        if not candidates:
            candidates = self.spreadsheet_rows
        return random.choice(candidates)

    def apply_useless_info(self) -> str:
        templates_path = self.base_dir / USELESS_INFO_TEMPLATES_FILE
        all_families = load_useless_info_clauses(templates_path)
        concepts = concepts_in_expr(self.expr, self.leaf_atoms)
        family_name = pick_useless_info_family(
            all_families,
            concepts,
            forced_family=self.useless_info_family,
            question=self.question,
        )
        self.useless_info_family_used = family_name
        families = select_useless_info_families(all_families, family_name)
        snippet = format_useless_info_clause(
            pick_useless_info_clause(families, family_name=family_name)
        )
        return self._prepend_or_append(snippet)

    def apply_unit_scale_change(self) -> str:
        if not self.unit_scale_label:
            raise ValueError(
                "unit_scale_label is required for unit_scale_change obstacle"
            )
        return unit_scale_question_prefix(self.unit_scale_label) + self.question

    def apply_negation(self) -> str:
        if random.random() < 0.5:
            return random.choice(NEGATION_PREFIXES) + self.question
        return self.question + random.choice(NEGATION_SUFFIXES)

    def apply_conditional(self) -> str:
        entities = self.entities()
        entity = next(iter(entities)) if entities else None
        row = self._pick_spreadsheet_row(prefer_entity=entity)
        entity_name = row.get("company_name", entity or "the company")
        year = row.get("year", "2020")
        revenue_str = row.get("revenue", "0")
        try:
            revenue = float(revenue_str)
            threshold = int(revenue * random.uniform(0.3, 0.7))
        except ValueError:
            threshold = 1000
        prefix = f"If revenue for {entity_name} in {year} exceeds {threshold:,}, "
        if not self.question:
            return prefix
        return prefix + self.question[0].lower() + self.question[1:]

    def apply(self, name: str) -> str:
        """Apply a question-level obstacle and return the modified question."""
        validate_obstacle_name(name)
        if name in DATA_PREP_OBSTACLES:
            raise ValueError(
                f"Obstacle {name!r} applies during data prep, not to question text. "
                "Regenerate data with data_prep --obstacle big_numbers."
            )
        if name not in QUESTION_OBSTACLES:
            raise ValueError(f"Obstacle {name!r} is not a question-level obstacle.")
        handler = getattr(self, f"apply_{name}")
        return handler()


def apply_question_obstacle(name: str, ctx: ObstacleContext) -> str:
    """Apply a question-level obstacle (delegates to ``ctx.apply``)."""
    return ctx.apply(name)


def obstacle_requires_spreadsheet(obstacle: str | None) -> bool:
    return obstacle == "conditional"
