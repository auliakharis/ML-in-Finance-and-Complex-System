"""
Financial expression tree sampler.

Generates a random typed expression template — a tree of operators and
placeholder leaves — that step 4 will bind to real spreadsheet atoms.

Template components:
1. raw leaves        -> blank slots, each bound to one atom by step 4
2. operator nodes    -> sum / diff / mul / ratio (binary, type-checked)
3. growth            -> year-over-year change of one concept (ratio)
4. time aggregations -> min / max / avg of one concept across multiple years;
                        step 4 picks the min/max or computes avg as sum / count
5. derived concepts  -> named formulas such as gross_profit or net_income,
                        expanded into their full formula by step 4

This file only builds the template structure — no data, companies, years,
or concept values are assigned here.

Depth terminology:
- template depth = recursive depth budget passed to the sampler
- actual depth   = realized operator depth of the sampled tree
- concept depth  = expansion depth of a derived concept's internal formula
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# 1. Supported operators
# ---------------------------------------------------------------------
# sum      : amount + amount -> amount
# diff     : amount - amount -> amount
# ratio    : amount / amount -> ratio
# mul      : amount * ratio  -> amount
# growth   : amount(current year) vs amount(previous year) -> ratio
# min/max/avg are represented as "time_agg" nodes, not binary nodes
OPS = ("sum", "diff", "ratio", "mul", "growth", "min", "max", "avg")
AMOUNT_BINARY_OPS = ("sum", "diff", "mul")
TIME_AGG_OPS = ("min", "max", "avg")
RATIO_OPS = ("ratio", "growth")


# ---------------------------------------------------------------------
# 2. Registry of named derived financial concepts
# ---------------------------------------------------------------------
DERIVED_CONCEPTS: Dict[str, Dict[str, Any]] = {
    "gross_profit": {
        "family": "amount",
        "concept_depth": 1,
        "formula": {"op": "diff", "args": ["revenue", "cost_of_goods_sold"]},
        "protected": True,
    },
    "operating_income": {
        "family": "amount",
        "concept_depth": 2,
        "formula": {"op": "diff", "args": ["gross_profit", "operating_expenses"]},
        "protected": True,
    },
    "pretax_income": {
        "family": "amount",
        "concept_depth": 3,
        "formula": {"op": "diff", "args": ["operating_income", "non_operating_expenses"]},
        "protected": True,
    },
    "income_tax_expense": {
        "family": "amount",
        "concept_depth": 4,
        "formula": {"op": "mul", "args": ["pretax_income", "income_tax"]},
        "protected": True,
    },
    "net_income": {
        "family": "amount",
        "concept_depth": 5,
        "formula": {"op": "diff", "args": ["pretax_income", "income_tax_expense"]},
        "protected": True,
    },
    "current_assets": {
        "family": "amount",
        "concept_depth": 1,
        "formula": {
            "op": "sum",
            "args": ["cash", "accounts_receivable", "inventories", "short_term_investments"],
        },
        "protected": True,
    },
    "longterm_assets": {
        "family": "amount",
        "concept_depth": 2,
        "formula": {"op": "diff", "args": ["total_assets", "current_assets"]},
        "protected": True,
    },
    "longterm_liabilities": {
        "family": "amount",
        "concept_depth": 1,
        "formula": {"op": "diff", "args": ["total_liabilities", "current_liabilities"]},
        "protected": True,
    },
}


# ---------------------------------------------------------------------
# 3. Constructors for template objects
# ---------------------------------------------------------------------
def make_leaf(family: str) -> Dict[str, Any]:
    if family == "amount":
        return {"kind": "leaf", "semantic_type_in": ["amount"]}
    if family == "ratio":
        return {"kind": "leaf", "semantic_type_in": ["rate"]}
    raise ValueError(f"Unsupported family: {family}")


def make_derived_concept(name: str, family: str, concept_depth: int) -> Dict[str, Any]:
    return {
        "kind": "derived_concept",
        "name": name,
        "concept_depth": concept_depth,
    }


def make_node(
    op: str,
    family: str,
    left: Dict[str, Any],
    right: Dict[str, Any],
    depth: int,
) -> Dict[str, Any]:
    """Build a binary operator node: ``kind`` is ``node`` with ``left`` and ``right``.

    Used for ``sum``, ``diff``, ``mul``, ``ratio``, and ``growth`` only. Time-style
    aggregates (``min``/``max``/``avg`` over years) use :func:`make_time_agg` instead,
    which has no child subtrees.
    """
    return {
        "kind": "node",
        "op": op,
        "depth": depth,
        "left": left,
        "right": right,
    }


def make_time_agg(op: str, family: str, depth: int) -> Dict[str, Any]:
    """Build a non-binary ``time_agg`` node (min/max/avg over multiple years).

    Unlike :func:`make_node`, this has no ``left``/``right`` children. Step 4
    picks the entity and concept randomly from the atom index at binding time.
    """
    return {
        "kind": "time_agg",
        "op": op,
        "depth": depth,
    }


# ---------------------------------------------------------------------
# 4. Helpers for derived concepts
# ---------------------------------------------------------------------
def eligible_derived_concepts(depth: int, family: str) -> List[str]:
    return [
        name
        for name, spec in DERIVED_CONCEPTS.items()
        if spec["family"] == family and spec["concept_depth"] <= depth
    ]


# chooses the next amount operator under current constraints (whether time aggregations are allowed in this part of the tree).
def choose_amount_op(rng: random.Random, allow_time_aggregates: bool) -> str:
    ops = list(AMOUNT_BINARY_OPS)
    if allow_time_aggregates:
        ops.extend(TIME_AGG_OPS)
    return rng.choice(ops)


# samples a terminal amount leaf or derived concept node.
def sample_amount_terminal(
    rng: random.Random,
    depth: int,
    derived_prob: float,
) -> Dict[str, Any]:
    eligible = eligible_derived_concepts(depth=depth, family="amount")
    if eligible and rng.random() < derived_prob:
        name = rng.choice(eligible)
        spec = DERIVED_CONCEPTS[name]
        return make_derived_concept(
            name=name,
            family=spec["family"],
            concept_depth=spec["concept_depth"],
        )
    return make_leaf(family="amount")


# ---------------------------------------------------------------------
# 5. Tree builders
# ---------------------------------------------------------------------
# builds a pair of amount trees that share the same concept but are bound
# to different years by step 4 (used by growth nodes).
def build_time_series_amount_pair(
    depth: int,
    rng: random.Random,
    derived_prob: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    left = build_amount_tree(
        depth=depth,
        rng=rng,
        derived_prob=derived_prob,
        allow_time_aggregates=False,
    )
    right = build_amount_tree(
        depth=depth,
        rng=rng,
        derived_prob=derived_prob,
        allow_time_aggregates=False,
    )
    return left, right


# builds a ratio tree that is used to calculate the ratio of two amount trees.
def build_ratio_tree(
    depth: int,
    rng: random.Random,
    derived_prob: float,
) -> Dict[str, Any]:
    # Terminal ratio leaves stop recursion at depth 0.
    if depth == 0:
        return make_leaf(family="ratio")

    # Ratio-family nodes are either direct ratio or growth over time.
    op = rng.choice(RATIO_OPS)

    if op == "ratio":
        left = build_amount_tree(depth=depth - 1, rng=rng, derived_prob=derived_prob)
        right = build_amount_tree(depth=depth - 1, rng=rng, derived_prob=derived_prob)
        return make_node(op="ratio", family="ratio", left=left, right=right, depth=depth)

    if op == "growth":
        # Force both sides to share the same concept across different years.
        left, right = build_time_series_amount_pair(depth=depth - 1, rng=rng, derived_prob=derived_prob)
        return make_node(op="growth", family="ratio", left=left, right=right, depth=depth)

    raise ValueError(f"Unsupported ratio op: {op}")


# builds an amount tree that is used to calculate the sum, difference, multiplication, or time aggregation of two amount trees.
def build_amount_tree(
    depth: int,
    rng: random.Random,
    derived_prob: float = 0.30,
    allow_time_aggregates: bool = True,
) -> Dict[str, Any]:
    # Base case: only terminals at depth 0.
    if depth == 0:
        return sample_amount_terminal(rng=rng, depth=0, derived_prob=0.0)

    # Random early-stop to mix shallow and deep structures.
    if rng.random() < 0.35:
        return sample_amount_terminal(rng=rng, depth=depth, derived_prob=derived_prob)

    # Choose next amount operator under current constraints.
    op = choose_amount_op(rng, allow_time_aggregates)

    if op == "sum":
        left = build_amount_tree(depth=depth - 1, rng=rng, derived_prob=derived_prob, allow_time_aggregates=allow_time_aggregates)
        right = build_amount_tree(depth=depth - 1, rng=rng, derived_prob=derived_prob, allow_time_aggregates=allow_time_aggregates)
        return make_node(op="sum", family="amount", left=left, right=right, depth=depth)

    if op == "diff":
        left = build_amount_tree(depth=depth - 1, rng=rng, derived_prob=derived_prob, allow_time_aggregates=allow_time_aggregates)
        right = build_amount_tree(depth=depth - 1, rng=rng, derived_prob=derived_prob, allow_time_aggregates=allow_time_aggregates)
        return make_node(op="diff", family="amount", left=left, right=right, depth=depth)

    if op == "mul":
        # Multiplication combines amount branch with ratio branch.
        # If both sides were arbitrary amount trees, we would often get nonsense units (e.g. dollars x dollars)
        left = build_amount_tree(depth=depth - 1, rng=rng, derived_prob=derived_prob, allow_time_aggregates=allow_time_aggregates)
        right = build_ratio_tree(depth=depth - 1, rng=rng, derived_prob=derived_prob)
        return make_node(op="mul", family="amount", left=left, right=right, depth=depth)

    if op in TIME_AGG_OPS:
        # Time aggregate: step 4 picks entity and concept at binding time.
        return make_time_agg(op=op, family="amount", depth=depth)

    raise ValueError(f"Unsupported amount op: {op}")


# ---------------------------------------------------------------------
# 6. Formula expansion helpers
# ---------------------------------------------------------------------
# expands a derived concept name to a tuple of the operation and its arguments.
def expand_formula_reference(name: str) -> Any:
    if name not in DERIVED_CONCEPTS:
        return name

    spec = DERIVED_CONCEPTS[name]
    op = spec["formula"]["op"]
    args = spec["formula"]["args"]
    expanded_args = [expand_formula_reference(arg) for arg in args]
    return (op, tuple(expanded_args))


# normalizes a symbolic representation of a tree by sorting the arguments of the sum operation.
def normalize_symbolic(expr: Any) -> Any:
    if isinstance(expr, str):
        return expr

    op, args = expr
    norm_args = tuple(normalize_symbolic(arg) for arg in args)

    if op == "sum":
        norm_args = tuple(sorted(norm_args, key=repr))

    return (op, norm_args)


# converts a tree to a symbolic representation that can be used to compare trees.
def symbolic_from_tree(tree: Dict[str, Any]) -> Any:
    kind = tree["kind"]

    if kind == "leaf":
        return "LEAF"

    if kind == "derived_concept":
        return normalize_symbolic(expand_formula_reference(tree["name"]))

    if kind == "time_agg":
        return ("time_agg", tree["op"])

    if kind == "node":
        left = symbolic_from_tree(tree["left"])
        right = symbolic_from_tree(tree["right"])
        return normalize_symbolic((tree["op"], (left, right)))

    raise ValueError(f"Unknown tree kind: {kind}")


def protected_signatures() -> Dict[str, Any]:
    """Map each protected derived concept to its fully expanded formula signature.

    Used by violates_protected_canonical_form to detect any node in the tree
    (not just the root) whose structure silently duplicates a named concept without
    using a derived_concept node. For example, raw leaves computing
    revenue - COGS would match gross_profit and be rejected.
    """
    out: Dict[str, Any] = {}
    for name, spec in DERIVED_CONCEPTS.items():
        if spec.get("protected", False):
            out[name] = normalize_symbolic(expand_formula_reference(name))
    return out


def contains_named_derived(tree: Dict[str, Any], concept_name: str) -> bool:
    kind = tree["kind"]

    if kind == "derived_concept":
        return tree["name"] == concept_name

    if kind in {"leaf", "time_agg"}:
        return False

    return contains_named_derived(tree["left"], concept_name) or contains_named_derived(
        tree["right"], concept_name
    )


_PROTECTED_SIGNATURES: Dict[str, Any] = protected_signatures()


def violates_protected_canonical_form(tree: Dict[str, Any]) -> Optional[str]:
    if tree["kind"] in {"leaf", "time_agg", "derived_concept"}:
        return None

    tree_sig = symbolic_from_tree(tree)
    for concept_name, concept_sig in _PROTECTED_SIGNATURES.items():
        if tree_sig == concept_sig and not contains_named_derived(tree, concept_name):
            return concept_name

    left_violation = violates_protected_canonical_form(tree["left"])
    if left_violation is not None:
        return left_violation
    return violates_protected_canonical_form(tree["right"])


# ---------------------------------------------------------------------
# 7. Counting and diagnostics
# ---------------------------------------------------------------------
# counts the number of internal nodes, leaves, and total nodes in a tree.
def count_nodes(tree: Dict[str, Any]) -> Dict[str, int]:
    if tree["kind"] in {"leaf", "derived_concept", "time_agg"}:
        return {"internal_nodes": 0, "leaves": 1, "total_nodes": 1}

    left = count_nodes(tree["left"])
    right = count_nodes(tree["right"])
    return {
        "internal_nodes": 1 + left["internal_nodes"] + right["internal_nodes"],
        "leaves": left["leaves"] + right["leaves"],
        "total_nodes": 1 + left["total_nodes"] + right["total_nodes"],
    }


# counts the depth of a tree by recursively counting the depth of the left and right subtrees.
def actual_tree_depth(tree: Dict[str, Any]) -> int:
    if tree["kind"] in {"leaf", "derived_concept", "time_agg"}:
        return 0
    return 1 + max(actual_tree_depth(tree["left"]), actual_tree_depth(tree["right"]))


# ---------------------------------------------------------------------
# 8. Sampling wrapper with rejection
# ---------------------------------------------------------------------
# samples a tree with rejection to prevent the tree from violating the protected form rules.
def sample_tree_with_rejection(
    max_depth: int,
    rng: random.Random,
    derived_prob: float,
    max_attempts: int = 200,
) -> Tuple[Dict[str, Any], Optional[str]]:
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0.")
    if not (0.0 <= derived_prob <= 1.0):
        raise ValueError("derived_prob must be within [0, 1].")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0.")

    last_reason: Optional[str] = None

    for _ in range(max_attempts):
        tree = build_amount_tree(depth=max_depth, rng=rng, derived_prob=derived_prob)
        violation = violates_protected_canonical_form(tree)
        if violation is None:
            return tree, last_reason
        last_reason = f"Rejected because tree duplicated protected concept: {violation}"

    raise RuntimeError(
        f"Failed to sample a valid tree after {max_attempts} attempts. "
        f"Last rejection reason: {last_reason}"
    )


# ---------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one typed financial operator tree with raw leaves, derived concepts, growth, and min/max/avg across years."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Target recursive depth budget of the sampled tree.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--derived-prob",
        type=float,
        default=0.30,
        help="Probability of sampling a derived concept at a terminal amount position.",
    )
    parser.add_argument(
        "--output",
        default="output/generated_operator_tree.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    tree, rejection_note = sample_tree_with_rejection(
        max_depth=args.depth,
        rng=rng,
        derived_prob=args.derived_prob,
    )
    stats = count_nodes(tree)
    realized_depth = actual_tree_depth(tree)

    payload = {
        "max_depth": args.depth,
        "actual_depth": realized_depth,
        "seed": args.seed,
        "derived_prob": args.derived_prob,
        "ops": list(OPS),
        "derived_concepts": {
            name: {
                "family": spec["family"],
                "concept_depth": spec["concept_depth"],
                "formula": spec["formula"],
                "protected": spec["protected"],
            }
            for name, spec in DERIVED_CONCEPTS.items()
        },
        "stats": stats,
        "tree": tree,
    }

    if rejection_note is not None:
        payload["note"] = rejection_note

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote typed operator tree to {output_path}")
    print(json.dumps(stats, indent=2))
    print(f"Requested max depth: {args.depth}")
    print(f"Actual realized depth: {realized_depth}")


if __name__ == "__main__":
    main()
