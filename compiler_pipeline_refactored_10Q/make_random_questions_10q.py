"""
Generate random financial Q&A pairs from synthetic 10-Q filings.

Flow:
    generate_atoms()          -> dict[str, Atom]   (from data_prep_10q.py)
    Store.store_from_atoms()  -> Store              (indexes atoms for tree sampler)
    Expr.sample_tree_with_rejection() + instantiate_typed_tree()  -> bound Expr
    SemanticAnalyzer.analyze()  -> AnalysisResult
    Evaluator.eval()            -> float
    QuestionRenderer.render()   -> str
    -> output CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REFACTORED = os.path.join(_HERE, "..", "compiler_pipeline_refactored")
sys.path.insert(0, _HERE)
sys.path.insert(0, _REFACTORED)

from tree import (
    Atom,
    BindEnv,
    DerivedConcept,
    Expr,
    Operation,
    SemanticType,
    Store,
)
from evaluator import Evaluator
from data_prep_10q import generate_atoms
from semantic_analyzer_10q import SemanticAnalyzer
from question_renderer_10q import QuestionRenderer

T = TypeVar("T")

# ── Store10Q — limits time-aggregation depth and excludes outflow concepts ───

# Cash outflow concepts are stored as negative numbers; min/max over them
# produces semantically confusing questions ("maximum share repurchases = -969").
_OUTFLOW_CONCEPTS = frozenset({"share_repurchases", "dividends_paid"})


def _period_type(period: str) -> str:
    """Classify a 10-Q period string into a comparable bucket."""
    if "Three Months" in period:
        return "quarter"
    if "Six Months" in period:
        return "ytd_q2"
    if "Nine Months" in period:
        return "ytd_q3"
    return "balance_sheet"


def _period_group_key(period: str) -> str:
    """Finer grouping that includes calendar month for flow periods.

    Prevents mixing April quarters with October quarters when a company
    appears in multiple filings with different quarter-end months.
    """
    ptype = _period_type(period)
    if ptype == "balance_sheet":
        return ptype
    m = period.split("Ended ")
    month = m[1].split()[0] if len(m) > 1 else "unknown"
    return f"{ptype}_{month}"


@dataclass
class Store10Q(Store):
    """Store subclass that caps time-aggregation windows and skips outflow concepts."""
    max_agg_periods: int = 4

    def _agg_amount_concepts(self) -> list[str]:
        # Exclude any concept where at least one atom has a negative value —
        # min/max of negative numbers produces semantically confusing questions.
        negative_concepts = {
            a.concept for a in self._atoms
            if a.semantic_type == SemanticType.amount and a.value < 0
        }
        blocked = _OUTFLOW_CONCEPTS | negative_concepts
        eligible = [c for c in self.amount_concepts() if c not in blocked]
        return eligible or self.amount_concepts()

    def pick_entity_concept_all_periods(
        self, env: BindEnv, *, purpose: str
    ) -> tuple[str, str, list[str]]:
        entity = env.entity or random.choice(self.entities)
        if env.concept and env.concept not in _OUTFLOW_CONCEPTS:
            concept = env.concept
        else:
            concept = random.choice(self._agg_amount_concepts())

        atoms = self.filter_atoms(
            semantic_types=[SemanticType.amount], concept=concept, entity=entity
        )
        all_periods = sorted({a.period for a in atoms})
        if len(all_periods) < 2:
            raise Exception(
                f"Need at least two periods for concept={concept}, entity={entity} ({purpose})."
            )
        # Group by period type + calendar month — prevents mixing April and
        # October quarters when a company appears in multiple filings.
        by_type: dict[str, list[str]] = {}
        for p in all_periods:
            by_type.setdefault(_period_group_key(p), []).append(p)
        eligible_types = [t for t, ps in by_type.items() if len(ps) >= 2]
        if not eligible_types:
            raise Exception(
                f"No period group with 2+ periods for concept={concept}, entity={entity} ({purpose})."
            )
        chosen_type = random.choice(eligible_types)
        periods = by_type[chosen_type]
        # Cap to avoid trees deeper than depth_max
        if len(periods) > self.max_agg_periods:
            start = random.randint(0, len(periods) - self.max_agg_periods)
            periods = periods[start : start + self.max_agg_periods]
        return entity, concept, periods


# ── Derived concepts for 10-Q ────────────────────────────────────────────────

DERIVED_CONCEPTS_10Q: dict[str, DerivedConcept] = {
    # Income statement — protected because these are also direct atoms
    "gross_profit": DerivedConcept(
        name="gross_profit",
        op=Operation.diff,
        family="amount",
        concept_dept=1,
        args=["total_revenues", "cost_of_sales"],
        protected=True,
    ),
    "operating_income": DerivedConcept(
        name="operating_income",
        op=Operation.diff,
        family="amount",
        concept_dept=2,
        args=["total_revenues", "total_costs_and_expenses"],
        protected=True,
    ),
    "net_income": DerivedConcept(
        name="net_income",
        op=Operation.diff,
        family="amount",
        concept_dept=3,
        args=["operating_income", "provision_for_income_taxes"],
        protected=True,
    ),
    # Balance sheet — protected
    "total_current_assets": DerivedConcept(
        name="total_current_assets",
        op=Operation.sum,
        family="amount",
        concept_dept=1,
        args=[
            "cash_and_cash_equivalents",
            "short_term_investments",
            "accounts_receivable_net",
            "inventories",
            "prepaid_expenses_and_other",
        ],
        protected=True,
    ),
    "total_assets": DerivedConcept(
        name="total_assets",
        op=Operation.sum,
        family="amount",
        concept_dept=2,
        args=["total_current_assets", "total_non_current_assets"],
        protected=True,
    ),
    "total_current_liabilities": DerivedConcept(
        name="total_current_liabilities",
        op=Operation.sum,
        family="amount",
        concept_dept=1,
        args=[
            "accounts_payable",
            "deferred_revenue_current",
            "accrued_expenses_and_other",
            "current_portion_of_long_term_debt",
        ],
        protected=True,
    ),
    "total_liabilities": DerivedConcept(
        name="total_liabilities",
        op=Operation.sum,
        family="amount",
        concept_dept=2,
        args=["total_current_liabilities", "total_non_current_liabilities"],
        protected=True,
    ),
    # Unprotected derived concepts (not direct atoms)
    "free_cash_flow": DerivedConcept(
        name="free_cash_flow",
        op=Operation.diff,
        family="amount",
        concept_dept=1,
        args=["net_cash_from_operating", "capital_expenditures"],
        protected=False,
    ),
    "working_capital": DerivedConcept(
        name="working_capital",
        op=Operation.diff,
        family="amount",
        concept_dept=2,
        args=["total_current_assets", "total_current_liabilities"],
        protected=False,
    ),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def validate_args(args: argparse.Namespace) -> None:
    if args.n <= 0:
        raise ValueError("--n must be > 0.")
    if args.depth_min < 0 or args.depth_max < 0:
        raise ValueError("--depth-min/--depth-max must be >= 0.")
    if args.depth_min > args.depth_max:
        raise ValueError("--depth-min must be <= --depth-max.")
    if not (0.0 <= args.derived_prob_min <= 1.0 and 0.0 <= args.derived_prob_max <= 1.0):
        raise ValueError("--derived-prob-min and --derived-prob-max must be in [0, 1].")
    if args.derived_prob_min > args.derived_prob_max:
        raise ValueError("--derived-prob-min must be <= --derived-prob-max.")


def flatten_leaf_keys(expr: Expr) -> list[str]:
    return [leaf.key for leaf in expr.flatten_leaves() if leaf.key is not None]


def with_random_seed(seed: int, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    previous_state = random.getstate()
    random.seed(seed)
    try:
        return fn(*args, **kwargs)
    finally:
        random.setstate(previous_state)


def build_row(
    i: int,
    expr: Expr,
    question: str,
    answer: float,
    expr_json: dict[str, Any],
    expr_str: str,
    tree_payload: Expr,
    tree_seed: int,
    bind_seed: int,
    master_seed: int,
    derived_prob: float,
    depth: int,
    template_stats: dict[str, int],
    leaf_keys: list[str],
    atoms: dict[str, Atom],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": i + 1,
        "depth": expr.expr_depth(),
        "question": question,
        "expression": expr_str,
        "expression_json": json.dumps(expr_json, ensure_ascii=False),
        "template_expression": json.dumps(tree_payload.expr_to_json(), ensure_ascii=False),
        "answer": answer,
        "template_depth_requested": depth,
        "template_actual_depth": Expr.actual_tree_depth(tree_payload),
        "bound_expression_depth": expr.expr_depth(),
        "tree_seed": tree_seed,
        "binding_seed": bind_seed,
        "master_seed": master_seed,
        "derived_prob": derived_prob,
        "template_internal_nodes": template_stats["internal_nodes"],
        "template_leaf_slots": template_stats["leaves"],
        "bound_leaf_count": len(leaf_keys),
    }

    for j, leaf_key in enumerate(leaf_keys, start=1):
        atom = atoms[leaf_key]
        row[f"leaf_{j}_key"] = atom.key
        row[f"leaf_{j}_concept"] = atom.concept
        row[f"leaf_{j}_label"] = atom.label
        row[f"leaf_{j}_entity"] = atom.entity
        row[f"leaf_{j}_period"] = atom.period
        row[f"leaf_{j}_unit"] = atom.unit
        row[f"leaf_{j}_value"] = atom.value

    return row


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random financial questions from synthetic 10-Q filings."
    )
    parser.add_argument("--n-reports", type=int, default=50, help="Number of 10-Q reports to generate")
    parser.add_argument("--n-questions", type=int, default=90, help="Number of questions to generate")
    parser.add_argument("--depth-min", type=int, default=1)
    parser.add_argument("--depth-max", type=int, default=4)
    parser.add_argument("--derived-prob-min", type=float, default=0.10)
    parser.add_argument("--derived-prob-max", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=None, help="Master seed; default is random")
    parser.add_argument("--data-seed", type=int, default=42, help="Seed for report generation")
    parser.add_argument(
        "--output",
        default=os.path.join(_HERE, "output", "random_questions_10q.csv"),
    )
    args = parser.parse_args()
    args.n = args.n_questions
    validate_args(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.n_reports} 10-Q reports...")
    atoms = generate_atoms(n_reports=args.n_reports, seed=args.data_seed)
    print(f"  -> {len(atoms)} atoms produced")

    _base = Store.store_from_atoms(atoms)
    store = Store10Q(
        concepts=_base.concepts,
        entities=_base.entities,
        periods=_base.periods,
        _atoms=_base._atoms,
        max_agg_periods=4,
    )
    analyzer = SemanticAnalyzer(atoms)
    evaluator = Evaluator(atoms)
    renderer = QuestionRenderer()

    master_seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 10**9)
    master_rng = random.Random(master_seed)

    rows: list[dict[str, Any]] = []
    max_leaf_count = 0

    for i in range(args.n_questions):
        last_error: Exception | None = None
        for attempt in range(1, 1001):
            tree_seed = master_rng.randint(0, 10**9)
            bind_seed = master_rng.randint(0, 10**9)
            depth = master_rng.randint(args.depth_min, args.depth_max)
            derived_prob = round(
                master_rng.uniform(args.derived_prob_min, args.derived_prob_max),
                3,
            )

            try:
                tree_payload, _ = Expr.sample_tree_with_rejection(
                    max_depth=depth,
                    rng=random.Random(tree_seed),
                    derived_prob=derived_prob,
                    derived_registry=DERIVED_CONCEPTS_10Q,
                )
                expr = with_random_seed(
                    bind_seed,
                    Expr.instantiate_typed_tree,
                    tree_payload,
                    store,
                    BindEnv(),
                    DERIVED_CONCEPTS_10Q,
                )
                if expr.expr_depth() > args.depth_max:
                    raise ValueError(f"Expression depth {expr.expr_depth()} exceeds depth_max {args.depth_max}")
                leaf_keys_check = flatten_leaf_keys(expr)
                if len(leaf_keys_check) != len(set(leaf_keys_check)):
                    raise ValueError("Duplicate atom keys in expression — degenerate reuse")
                leaf_periods = [atoms[k].period for k in leaf_keys_check]
                flow_types = {
                    _period_group_key(p) for p in leaf_periods
                    if _period_type(p) != "balance_sheet"
                }
                if len(flow_types) > 1:
                    raise ValueError(
                        f"Mixed flow period groups {flow_types} — cannot combine different quarter months or quarter/YTD"
                    )
                analysis = analyzer.analyze(expr)
                answer = evaluator.eval(expr)
                if abs(answer) < 1e-9:
                    raise ValueError("Degenerate answer: 0")
                question = with_random_seed(bind_seed, renderer.render, analysis)
                expr_json = expr.expr_to_json()
                expr_str = expr.show_expr()
                template_stats = Expr.count_nodes(tree_payload)
                leaf_keys = flatten_leaf_keys(expr)
                max_leaf_count = max(max_leaf_count, len(leaf_keys))
                break
            except Exception as err:
                last_error = err
                if attempt == 1000:
                    raise RuntimeError(
                        f"Failed to generate question {i + 1} after 1000 attempts. "
                        f"Last error: {type(last_error).__name__}: {last_error}"
                    ) from last_error

        row = build_row(
            i=i,
            expr=expr,
            question=question,
            answer=answer,
            expr_json=expr_json,
            expr_str=expr_str,
            tree_payload=tree_payload,
            tree_seed=tree_seed,
            bind_seed=bind_seed,
            master_seed=master_seed,
            derived_prob=derived_prob,
            depth=depth,
            template_stats=template_stats,
            leaf_keys=leaf_keys,
            atoms=atoms,
        )
        rows.append(row)

    fieldnames = [
        "question_id", "depth", "question", "expression", "expression_json",
        "template_expression", "answer", "template_depth_requested",
        "template_actual_depth", "bound_expression_depth", "tree_seed",
        "binding_seed", "master_seed", "derived_prob", "template_internal_nodes",
        "template_leaf_slots", "bound_leaf_count",
    ]
    for j in range(1, max_leaf_count + 1):
        fieldnames.extend([
            f"leaf_{j}_key", f"leaf_{j}_concept", f"leaf_{j}_label",
            f"leaf_{j}_entity", f"leaf_{j}_period", f"leaf_{j}_unit", f"leaf_{j}_value",
        ])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} questions to {output_path}")
    print(f"Master seed: {master_seed}")
    print(f"Max bound leaf count: {max_leaf_count}")


if __name__ == "__main__":
    main()
