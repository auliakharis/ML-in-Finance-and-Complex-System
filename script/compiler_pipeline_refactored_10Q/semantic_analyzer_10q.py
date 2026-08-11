"""Semantic analysis for bound financial expression trees — 10-Q version."""
from __future__ import annotations

import sys
import os

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from script.compiler_pipeline_adversarial.tree import (
    Atom,
    DerivedExpr,
    Expr,
    Leaf,
    Literal,
    Node,
    Operation,
    SemanticError,
    SemanticType,
)


def clean_label(label: str | None) -> str:
    if not label:
        return "value"
    return label.replace("_", " ")


def question_copula(label: str | None) -> str:
    cleaned = clean_label(label).strip().lower()
    if not cleaned:
        return "is"
    if cleaned.endswith("s") and not cleaned.endswith("ss"):
        return "are"
    return "is"


def shared_entity_period_context(
    left_meaning: "Meaning", right_meaning: "Meaning"
) -> tuple[str, str] | None:
    if (
        left_meaning.entity
        and right_meaning.entity
        and left_meaning.period
        and right_meaning.period
        and left_meaning.entity == right_meaning.entity
        and left_meaning.period == right_meaning.period
    ):
        return left_meaning.entity, left_meaning.period
    return None


def strip_entity_period_suffix(phrase: str, entity: str, period: str) -> str:
    suffix = f" for {entity} in {period}"
    if phrase.endswith(suffix):
        return phrase[: -len(suffix)]
    return phrase


def both_have_entity_period_suffix(left: str, right: str, entity: str, period: str) -> bool:
    suffix = f" for {entity} in {period}"
    return left.endswith(suffix) and right.endswith(suffix)


def _parse_period_date(period: str) -> datetime:
    """Extract the end date from a 10-Q period string for chronological sorting."""
    date_str = period.split("Ended ")[-1].strip() if "Ended " in period else period.strip()
    try:
        return datetime.strptime(date_str, "%B %d, %Y")
    except ValueError:
        return datetime.min


def sort_periods(periods) -> list[str]:
    """Sort period strings chronologically by their end date."""
    return sorted(periods, key=_parse_period_date)


class Meaning(BaseModel):
    kind: str
    semantic_type: SemanticType
    text: str
    entity: str | None = None
    period: str | None = None
    unit: str | None = None
    concept: str | None = None
    label: str | None = None
    from_period: str | None = None
    to_period: str | None = None
    components: list[str] | None = None
    target_concept: str | None = None
    derivation: str | None = None

    def same_context(self, other: Meaning) -> bool:
        return self.entity == other.entity and self.period == other.period and self.unit == other.unit

    def same_entity_and_unit(self, other: Meaning) -> bool:
        return self.entity == other.entity and self.unit == other.unit

    def is_metric_like_amount(self) -> bool:
        return self.semantic_type == SemanticType.amount and self.kind in {
            "leaf_metric",
            "aggregate_components",
            "derived_metric",
            "min_over_time",
            "max_over_time",
            "avg_over_time",
            "avg_over_all_periods",
        }

    def is_metric_like_ratio(self) -> bool:
        return self.semantic_type == SemanticType.rate and self.kind == "leaf_metric"

    def same_amount_context(self, other: Meaning) -> bool:
        return (
            self.semantic_type == SemanticType.amount
            and other.semantic_type == SemanticType.amount
            and self.entity == other.entity
            and self.unit == other.unit
        )

    def same_amount_timeseries_metric(self, other: Meaning) -> bool:
        return (
            self.same_amount_context(other)
            and self.concept == other.concept
            and self.period != other.period
            and self.is_metric_like_amount()
            and other.is_metric_like_amount()
        )

    def period_tokens(self) -> list[str]:
        return [p for p in (self.period, self.from_period, self.to_period) if p is not None]


def base_metric_name_from_child(meaning: Meaning) -> str:
    label = meaning.label or meaning.concept or "value"
    for prefix in ("minimum ", "maximum ", "average "):
        if label.startswith(prefix):
            return clean_label(label[len(prefix):])
    return clean_label(label)


def base_metric_name(meaning: Meaning) -> str:
    label = meaning.label or meaning.concept or "value"
    for prefix in ("growth rate of ", "minimum ", "maximum ", "average "):
        if label.startswith(prefix):
            label = label[len(prefix):]
    return clean_label(label)


@dataclass
class AnalysisResult:
    expr: Expr
    meaning: Meaning
    children: list[AnalysisResult] = field(default_factory=list)
    depth: int = 0


class SemanticAnalyzer:
    def __init__(self, atoms: dict[str, Atom]):
        self.atoms = atoms

    def atom(self, key: str) -> Atom:
        if key not in self.atoms:
            raise SemanticError(f"Unknown leaf key: {key}")
        return self.atoms[key]

    def get_atom(self, key: str) -> Atom:
        return self.atom(key)

    def analyze_leaf(self, expr: Expr) -> AnalysisResult:
        if isinstance(expr, Leaf):
            atom = self.atom(expr.key)
            meaning = Meaning(
                kind="leaf_metric",
                semantic_type=atom.semantic_type,
                text=f"{atom.label} for {atom.entity} in {atom.period}",
                entity=atom.entity,
                period=atom.period,
                unit=atom.unit,
                concept=atom.concept,
                label=atom.label,
                target_concept=atom.parent_concept,
                components=[atom.label] if atom.role == "component" else None,
                derivation=f"leaf {atom.key}",
            )
            return AnalysisResult(expr=expr, meaning=meaning, children=[], depth=0)

        if isinstance(expr, Literal):
            return AnalysisResult(
                expr=expr,
                meaning=Meaning(
                    kind="literal_scalar",
                    semantic_type=SemanticType.ratio,
                    text=str(expr.value),
                    derivation="numeric literal",
                ),
                children=[],
                depth=0,
            )

        if isinstance(expr, DerivedExpr):
            inner = self.analyze(expr.expr)
            inner_meaning = inner.meaning
            meaning = Meaning(
                kind="derived_metric",
                semantic_type=inner_meaning.semantic_type,
                text=f"{expr.name.replace('_', ' ')} for {inner_meaning.entity} in {inner_meaning.period}",
                entity=inner_meaning.entity,
                period=inner_meaning.period,
                unit=inner_meaning.unit,
                concept=expr.name,
                label=expr.name,
                from_period=inner_meaning.from_period,
                to_period=inner_meaning.to_period,
                target_concept=expr.name,
                derivation=f"named derived concept {expr.name}",
            )
            return AnalysisResult(expr=expr, meaning=meaning, children=[inner], depth=expr.expr_depth())

        raise SemanticError(f"Unsupported expression type: {type(expr)!r}")

    def analyze(self, expr: Expr) -> AnalysisResult:
        if isinstance(expr, (Leaf, Literal, DerivedExpr)):
            return self.analyze_leaf(expr)

        assert isinstance(expr, Node)
        left_result = self.analyze(expr.left)
        right_result = self.analyze(expr.right)

        match expr.op:
            case Operation.sum:
                meaning = self.analyze_sum(expr, left_result, right_result)
            case Operation.diff:
                meaning = self.analyze_diff(expr, left_result, right_result)
            case Operation.ratio:
                meaning = self.analyze_ratio(expr, left_result, right_result)
            case Operation.mul:
                meaning = self.analyze_mul(expr, left_result, right_result)
            case Operation.growth:
                meaning = self.analyze_growth(expr, left_result, right_result)
            case Operation.min | Operation.max:
                meaning = self.analyze_time_aggregate(expr, left_result, right_result)
            case _:
                raise Exception(f"Unsupported op: {expr.op}")

        node_depth = expr.depth if isinstance(expr, Node) and expr.depth is not None else 1 + max(left_result.depth, right_result.depth)
        return AnalysisResult(expr=expr, meaning=meaning, children=[left_result, right_result], depth=node_depth)

    def analyze_sum(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning
        leaves = expr.flatten_sum()

        if leaves:
            atoms = [self.atom(leaf.key) for leaf in leaves]
            first = atoms[0]
            total_expected = (
                len(
                    {
                        a.concept
                        for a in self.atoms.values()
                        if a.parent_concept == first.parent_concept and a.role == "component"
                    }
                )
                if (first.parent_concept is not None and first.role == "component")
                else 0
            )
            if (
                len(atoms) >= 2
                and first.semantic_type == SemanticType.amount
                and first.parent_concept is not None
                and first.role == "component"
                and len(atoms) == total_expected
                and all(
                    a.semantic_type == SemanticType.amount
                    and a.parent_concept == first.parent_concept
                    and a.role == "component"
                    and a.entity == first.entity
                    and a.period == first.period
                    and a.unit == first.unit
                    for a in atoms[1:]
                )
            ):
                component_labels = [a.label for a in atoms]
                return Meaning(
                    kind="aggregate_components",
                    semantic_type=SemanticType.amount,
                    text=f"{first.parent_concept.replace('_', ' ')} for {first.entity} in {first.period}",
                    entity=first.entity,
                    period=first.period,
                    unit=first.unit,
                    concept=first.parent_concept,
                    label=first.parent_concept,
                    target_concept=first.parent_concept,
                    components=component_labels,
                    derivation=f"sum of sibling components: {', '.join(component_labels)}",
                )

        if (
            lm.semantic_type == SemanticType.amount
            and rm.semantic_type == SemanticType.amount
            and lm.entity == rm.entity
            and lm.period == rm.period
            and lm.unit == rm.unit
        ):
            left_label = lm.label or lm.concept or "value"
            right_label = rm.label or rm.concept or "value"
            composite_label = f"{left_label} plus {right_label}"
            return Meaning(
                kind="sum_amount",
                semantic_type=SemanticType.amount,
                text=f"sum of {left_label} and {right_label} for {lm.entity} in {lm.period}",
                entity=lm.entity,
                period=lm.period,
                unit=lm.unit,
                concept=None,
                label=composite_label,
                derivation=f"sum({lm.kind}, {rm.kind})",
            )

        if (
            lm.semantic_type == SemanticType.amount
            and rm.semantic_type == SemanticType.amount
            and lm.entity == rm.entity
        ):
            left_label = lm.label or lm.concept or "value"
            right_label = rm.label or rm.concept or "value"
            return Meaning(
                kind="sum_amounts",
                semantic_type=SemanticType.amount,
                text=f"sum of two amount expressions for {lm.entity}",
                entity=lm.entity,
                period=lm.period if lm.period == rm.period else None,
                unit=lm.unit if lm.unit == rm.unit else None,
                concept=None,
                label=f"{left_label} plus {right_label}",
                derivation=f"sum({lm.kind}, {rm.kind}) fallback",
            )

        raise SemanticError(
            f"Cannot sum these meanings: {lm.kind}/{lm.semantic_type} and {rm.kind}/{rm.semantic_type}"
        )

    def analyze_diff(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning
        if lm.semantic_type == rm.semantic_type == SemanticType.amount and lm.same_context(rm):
            left_label = lm.label or lm.concept or "value"
            right_label = rm.label or rm.concept or "value"
            composite_label = f"{left_label} minus {right_label}"
            return Meaning(
                kind="difference_amount",
                semantic_type=SemanticType.amount,
                text=f"difference between {left_label} and {right_label} for {lm.entity} in {lm.period}",
                entity=lm.entity,
                period=lm.period,
                unit=lm.unit,
                concept=None,
                label=composite_label,
                derivation=f"diff({lm.kind}, {rm.kind})",
            )
        raise SemanticError(f"Cannot subtract these meanings: {lm.kind} and {rm.kind}")

    def analyze_ratio(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning

        if isinstance(expr.right, Literal):
            n = float(expr.right.value)
            n_int = int(round(n))
            if n_int >= 2 and abs(n - n_int) < 1e-9:
                leaves = expr.left.flatten_leaves()
                if len(leaves) == n_int:
                    atoms = [self.atom(leaf.key) for leaf in leaves]
                    if (
                        len({a.entity for a in atoms}) == 1
                        and len({a.concept for a in atoms}) == 1
                        and all(a.semantic_type == SemanticType.amount for a in atoms)
                    ):
                        periods = sort_periods({a.period for a in atoms})
                        first = atoms[0]
                        return Meaning(
                            kind="avg_over_all_periods",
                            semantic_type=SemanticType.amount,
                            text=(
                                f"average of {first.label} for {first.entity} "
                                f"from {periods[0]} to {periods[-1]}"
                            ),
                            entity=first.entity,
                            unit=first.unit,
                            concept=first.concept,
                            label=first.label,
                            from_period=periods[0],
                            to_period=periods[-1],
                            period=periods[-1],
                            derivation=f"arithmetic mean of {n_int} period values",
                        )

        if lm.same_amount_context(rm):
            left_label = lm.label or lm.concept or "value"
            right_label = rm.label or rm.concept or "value"
            return Meaning(
                kind="ratio_amounts",
                semantic_type=SemanticType.ratio,
                text=f"ratio of {left_label} to {right_label} for {lm.entity}",
                entity=lm.entity,
                period=lm.period if lm.period == rm.period else None,
                unit=lm.unit,
                concept=None,
                label=f"{left_label} to {right_label}",
                derivation=f"ratio({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot divide these meanings: {lm.kind} and {rm.kind}")

    def analyze_mul(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning

        if lm.semantic_type == SemanticType.amount and rm.semantic_type in {
            SemanticType.ratio,
            SemanticType.rate,
        } and lm.entity == rm.entity:
            amount_label = lm.label or lm.concept or "amount"
            ratio_label = rm.label or rm.concept or "ratio"
            return Meaning(
                kind="scaled_amount",
                semantic_type=SemanticType.amount,
                text=f"{amount_label} adjusted by {ratio_label} for {lm.entity}",
                entity=lm.entity,
                period=lm.period or rm.period,
                unit=lm.unit,
                concept=None,
                label=f"{amount_label} adjusted by {ratio_label}",
                target_concept=ratio_label,
                derivation=f"mul({lm.kind}, {rm.kind})",
            )

        if rm.semantic_type == SemanticType.amount and lm.semantic_type in {
            SemanticType.ratio,
            SemanticType.rate,
        } and lm.entity == rm.entity:
            amount_label = rm.label or rm.concept or "amount"
            ratio_label = lm.label or lm.concept or "ratio"
            return Meaning(
                kind="scaled_amount",
                semantic_type=SemanticType.amount,
                text=f"{amount_label} adjusted by {ratio_label} for {rm.entity}",
                entity=rm.entity,
                period=rm.period or lm.period,
                unit=rm.unit,
                concept=None,
                label=f"{amount_label} adjusted by {ratio_label}",
                target_concept=ratio_label,
                derivation=f"mul({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot multiply these meanings: {lm.kind} and {rm.kind}")

    def analyze_growth(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning
        if lm.same_amount_timeseries_metric(rm):
            periods = sort_periods([lm.period, rm.period])
            assert periods[0] is not None and periods[1] is not None
            return Meaning(
                kind="growth_rate",
                semantic_type=SemanticType.ratio,
                text=f"growth rate of {lm.label or lm.concept} for {lm.entity} from {periods[0]} to {periods[1]}",
                entity=lm.entity,
                unit=lm.unit,
                concept=lm.concept,
                label=f"growth rate of {lm.label or lm.concept}",
                from_period=periods[0],
                to_period=periods[1],
                derivation="explicit growth operator",
            )

        if lm.same_amount_context(rm):
            periods = [p for p in [lm.period, rm.period] if p is not None]
            ordered = sort_periods(periods) if len(periods) == 2 and len(set(periods)) == 2 else periods
            from_period = ordered[0] if ordered else None
            to_period = ordered[-1] if len(ordered) >= 2 else None
            return Meaning(
                kind="growth_rate_composed",
                semantic_type=SemanticType.ratio,
                text=f"growth between two amount expressions for {lm.entity}",
                entity=lm.entity,
                unit=lm.unit,
                concept=None,
                label=None,
                from_period=from_period,
                to_period=to_period,
                derivation="explicit growth operator over composed amount expressions",
            )
        raise SemanticError(f"Cannot compute growth for these meanings: {lm.kind} and {rm.kind}")

    def analyze_time_aggregate(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning
        op_human = {"min": "minimum", "max": "maximum"}[expr.op.value]

        if expr.op in {Operation.min, Operation.max}:
            span_periods = sort_periods(set(lm.period_tokens() + rm.period_tokens()))
            if (
                lm.semantic_type == SemanticType.amount
                and rm.semantic_type == SemanticType.amount
                and lm.entity == rm.entity
                and lm.unit == rm.unit
                and lm.concept is not None
                and lm.concept == rm.concept
                and len(span_periods) >= 2
            ):
                base_label = (lm.label or lm.concept or "value").replace("minimum ", "").replace("maximum ", "")
                return Meaning(
                    kind=f"{expr.op.value}_over_time",
                    semantic_type=SemanticType.amount,
                    text=(
                        f"{op_human} of {base_label} for {lm.entity} "
                        f"from {span_periods[0]} to {span_periods[-1]}"
                    ),
                    entity=lm.entity,
                    unit=lm.unit,
                    concept=lm.concept,
                    label=f"{op_human} {base_label}",
                    from_period=span_periods[0],
                    to_period=span_periods[-1],
                    period=span_periods[-1],
                    derivation=f"{expr.op.value} over {len(span_periods)} periods",
                )

        if lm.same_amount_timeseries_metric(rm):
            periods = sort_periods([lm.period, rm.period])
            assert periods[0] is not None and periods[1] is not None
            base_label = lm.label or lm.concept or "value"
            return Meaning(
                kind=f"{expr.op.value}_over_time",
                semantic_type=SemanticType.amount,
                text=f"{op_human} of {base_label} for {lm.entity} from {periods[0]} to {periods[1]}",
                entity=lm.entity,
                unit=lm.unit,
                concept=lm.concept,
                label=f"{op_human} {base_label}",
                from_period=periods[0],
                to_period=periods[1],
                period=periods[-1],
                derivation=f"{expr.op.value} over time",
            )

        if lm.same_amount_context(rm):
            return Meaning(
                kind=f"{expr.op.value}_amounts",
                semantic_type=SemanticType.amount,
                text=f"{op_human} of two amount expressions for {lm.entity}",
                entity=lm.entity,
                unit=lm.unit,
                concept=None,
                label=None,
                from_period=lm.period,
                to_period=rm.period,
                derivation=f"{expr.op.value} over composed amount expressions",
            )

        raise SemanticError(f"Cannot compute {expr.op} for these meanings: {lm.kind} and {rm.kind}")
