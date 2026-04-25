from __future__ import annotations

"""
Typed tree sampler for sec10Q financial reasoning templates.

Fork of compiler_pipeline/step3_tree_sampler.py adapted for SEC 10-Q data:
- DERIVED_CONCEPTS use sec10Q concept names (revenues_q, cost_of_sales_q, etc.)
- Balance-sheet derived concepts use plain names (current_assets, total_assets, etc.)
- All other tree-sampling logic is identical to the annual version.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# 1. Supported operators
# ---------------------------------------------------------------------
OPS = ("sum", "diff", "ratio", "mul", "growth", "min", "max", "avg")
AMOUNT_BINARY_OPS = ("sum", "diff", "mul")
TIME_AGG_OPS = ("min", "max", "avg")
RATIO_OPS = ("ratio", "growth")


# ---------------------------------------------------------------------
# 2. Registry of named derived financial concepts (sec10Q)
# ---------------------------------------------------------------------
DERIVED_CONCEPTS: Dict[str, Dict[str, Any]] = {
    # --- Quarterly income statement ---
    "gross_profit_q": {
        "family": "amount",
        "concept_depth": 1,
        "formula": {"op": "diff", "args": ["revenues_q", "cost_of_sales_q"]},
        "protected": True,
    },
    "net_income_q": {
        "family": "amount",
        "concept_depth": 2,
        "formula": {"op": "diff", "args": ["pretax_income_q", "tax_q"]},
        "protected": True,
    },
    # --- Balance sheet ---
    "net_working_capital": {
        "family": "amount",
        "concept_depth": 1,
        "formula": {"op": "diff", "args": ["current_assets", "current_liabilities"]},
        "protected": True,
    },
    "net_assets": {
        "family": "amount",
        "concept_depth": 1,
        "formula": {"op": "diff", "args": ["total_assets", "total_liabilities"]},
        "protected": True,
    },
}


# ---------------------------------------------------------------------
# 3. ID generator
# ---------------------------------------------------------------------
class IdGen:
    def __init__(self) -> None:
        self.node_idx = 0
        self.leaf_idx = 0
        self.group_idx = 0

    def node_id(self) -> str:
        out = f"N{self.node_idx}"
        self.node_idx += 1
        return out

    def leaf_name(self) -> str:
        out = f"L{self.leaf_idx}"
        self.leaf_idx += 1
        return out

    def group(self, prefix: str) -> str:
        out = f"{prefix}{self.group_idx}"
        self.group_idx += 1
        return out


# ---------------------------------------------------------------------
# 4. Constructors for template objects
# ---------------------------------------------------------------------
def make_leaf(
    idgen: IdGen,
    family: str,
    context_group: Optional[str],
    entity_group: Optional[str],
    time_series_group: Optional[str] = None,
    concept_group: Optional[str] = None,
    statement_group: Optional[str] = None,
    section_group: Optional[str] = None,
    aggregation_group: Optional[str] = None,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "kind": "leaf",
        "name": idgen.leaf_name(),
        "family": family,
    }

    if family == "amount":
        spec["semantic_type_in"] = ["amount"]
        if context_group is not None:
            spec["context_group"] = context_group
        if entity_group is not None:
            spec["entity_group"] = entity_group
        if time_series_group is not None:
            spec["time_series_group"] = time_series_group
        if concept_group is not None:
            spec["concept_group"] = concept_group
        if statement_group is not None:
            spec["statement_group"] = statement_group
        if section_group is not None:
            spec["section_group"] = section_group
        if aggregation_group is not None:
            spec["aggregation_group"] = aggregation_group

    elif family == "ratio":
        spec["semantic_type_in"] = ["ratio", "rate"]
        if entity_group is not None:
            spec["entity_group"] = entity_group

    else:
        raise ValueError(f"Unsupported family: {family}")

    return spec


def make_derived_concept(
    name: str,
    family: str,
    concept_depth: int,
    context_group: Optional[str],
    entity_group: Optional[str],
    time_series_group: Optional[str] = None,
    concept_group: Optional[str] = None,
    statement_group: Optional[str] = None,
    section_group: Optional[str] = None,
    aggregation_group: Optional[str] = None,
) -> Dict[str, Any]:
    spec: Dict[str, Any] = {
        "kind": "derived_concept",
        "name": name,
        "family": family,
        "concept_depth": concept_depth,
    }
    if context_group is not None:
        spec["context_group"] = context_group
    if entity_group is not None:
        spec["entity_group"] = entity_group
    if time_series_group is not None:
        spec["time_series_group"] = time_series_group
    if concept_group is not None:
        spec["concept_group"] = concept_group
    if statement_group is not None:
        spec["statement_group"] = statement_group
    if section_group is not None:
        spec["section_group"] = section_group
    if aggregation_group is not None:
        spec["aggregation_group"] = aggregation_group
    return spec


def make_node(
    idgen: IdGen,
    op: str,
    family: str,
    left: Dict[str, Any],
    right: Dict[str, Any],
    depth: int,
    over_years_group: Optional[str] = None,
) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "kind": "node",
        "node_id": idgen.node_id(),
        "op": op,
        "family": family,
        "depth": depth,
        "left": left,
        "right": right,
    }
    if over_years_group is not None:
        node["over_years_group"] = over_years_group
    return node


def make_time_agg(
    idgen: IdGen,
    op: str,
    family: str,
    depth: int,
    entity_group: str,
    concept_group: str,
    over_years_group: str,
    statement_group: Optional[str] = None,
    section_group: Optional[str] = None,
    aggregation_group: Optional[str] = None,
) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "kind": "time_agg",
        "node_id": idgen.node_id(),
        "op": op,
        "family": family,
        "depth": depth,
        "entity_group": entity_group,
        "concept_group": concept_group,
        "over_years_group": over_years_group,
    }
    if statement_group is not None:
        node["statement_group"] = statement_group
    if section_group is not None:
        node["section_group"] = section_group
    if aggregation_group is not None:
        node["aggregation_group"] = aggregation_group
    return node


# ---------------------------------------------------------------------
# 5. Helpers for derived concepts
# ---------------------------------------------------------------------
def eligible_derived_concepts(depth: int, family: str) -> List[str]:
    return [
        name
        for name, spec in DERIVED_CONCEPTS.items()
        if spec["family"] == family and spec["concept_depth"] <= depth
    ]


def choose_amount_op(rng: random.Random, allow_time_aggregates: bool) -> str:
    ops = list(AMOUNT_BINARY_OPS)
    if allow_time_aggregates:
        ops.extend(TIME_AGG_OPS)
    return rng.choice(ops)


def sample_amount_terminal(
    rng: random.Random,
    idgen: IdGen,
    depth: int,
    context_group: Optional[str],
    entity_group: Optional[str],
    derived_prob: float,
    time_series_group: Optional[str] = None,
    concept_group: Optional[str] = None,
    statement_group: Optional[str] = None,
    section_group: Optional[str] = None,
    aggregation_group: Optional[str] = None,
) -> Dict[str, Any]:
    eligible = eligible_derived_concepts(depth=depth, family="amount")
    if eligible and rng.random() < derived_prob:
        name = rng.choice(eligible)
        spec = DERIVED_CONCEPTS[name]
        return make_derived_concept(
            name=name,
            family=spec["family"],
            concept_depth=spec["concept_depth"],
            context_group=context_group,
            entity_group=entity_group,
            time_series_group=time_series_group,
            concept_group=concept_group,
            statement_group=statement_group,
            section_group=section_group,
            aggregation_group=aggregation_group,
        )
    return make_leaf(
        idgen=idgen,
        family="amount",
        context_group=context_group,
        entity_group=entity_group,
        time_series_group=time_series_group,
        concept_group=concept_group,
        statement_group=statement_group,
        section_group=section_group,
        aggregation_group=aggregation_group,
    )


# ---------------------------------------------------------------------
# 6. Tree builders
# ---------------------------------------------------------------------
def build_time_series_amount_pair(
    depth: int,
    rng: random.Random,
    idgen: IdGen,
    entity_group: str,
    derived_prob: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    time_series_group = idgen.group("Y")
    concept_group = idgen.group("T")

    left = build_amount_tree(
        depth=depth,
        rng=rng,
        idgen=idgen,
        context_group=None,
        entity_group=entity_group,
        derived_prob=derived_prob,
        forced_time_series_group=time_series_group,
        forced_concept_group=concept_group,
        allow_time_aggregates=False,
    )
    right = build_amount_tree(
        depth=depth,
        rng=rng,
        idgen=idgen,
        context_group=None,
        entity_group=entity_group,
        derived_prob=derived_prob,
        forced_time_series_group=time_series_group,
        forced_concept_group=concept_group,
        allow_time_aggregates=False,
    )
    return left, right, time_series_group


def build_ratio_tree(
    depth: int,
    rng: random.Random,
    idgen: IdGen,
    context_group: str,
    entity_group: str,
    derived_prob: float,
) -> Dict[str, Any]:
    if depth == 0:
        return make_leaf(
            idgen=idgen,
            family="ratio",
            context_group=None,
            entity_group=entity_group,
        )

    op = rng.choice(RATIO_OPS)

    if op == "ratio":
        section_group = idgen.group("S")
        left = build_amount_tree(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
            forced_section_group=section_group,
        )
        right = build_amount_tree(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
            forced_section_group=section_group,
        )
        return make_node(
            idgen=idgen,
            op="ratio",
            family="ratio",
            left=left,
            right=right,
            depth=depth,
        )

    if op == "growth":
        left, right, years_group = build_time_series_amount_pair(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            entity_group=entity_group,
            derived_prob=derived_prob,
        )
        return make_node(
            idgen=idgen,
            op="growth",
            family="ratio",
            left=left,
            right=right,
            depth=depth,
            over_years_group=years_group,
        )

    raise ValueError(f"Unsupported ratio op: {op}")


def build_amount_tree(
    depth: int,
    rng: random.Random,
    idgen: IdGen,
    context_group: Optional[str] = None,
    entity_group: Optional[str] = None,
    derived_prob: float = 0.30,
    forced_time_series_group: Optional[str] = None,
    forced_concept_group: Optional[str] = None,
    allow_time_aggregates: bool = True,
    forced_statement_group: Optional[str] = None,
    forced_section_group: Optional[str] = None,
    forced_aggregation_group: Optional[str] = None,
) -> Dict[str, Any]:
    if context_group is None:
        context_group = idgen.group("C")
    if entity_group is None:
        entity_group = idgen.group("E")

    if depth == 0:
        return sample_amount_terminal(
            rng=rng,
            idgen=idgen,
            depth=0,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=0.0,
            time_series_group=forced_time_series_group,
            concept_group=forced_concept_group,
            statement_group=forced_statement_group,
            section_group=forced_section_group,
            aggregation_group=forced_aggregation_group,
        )

    if rng.random() < 0.35:
        return sample_amount_terminal(
            rng=rng,
            idgen=idgen,
            depth=depth,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
            time_series_group=forced_time_series_group,
            concept_group=forced_concept_group,
            statement_group=forced_statement_group,
            section_group=forced_section_group,
            aggregation_group=forced_aggregation_group,
        )

    op = choose_amount_op(rng, allow_time_aggregates)

    if op == "sum":
        agg_group = idgen.group("A")
        section_group = idgen.group("S")

        left = build_amount_tree(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
            forced_time_series_group=forced_time_series_group,
            forced_concept_group=forced_concept_group,
            allow_time_aggregates=allow_time_aggregates,
            forced_statement_group=forced_statement_group,
            forced_section_group=section_group,
            forced_aggregation_group=agg_group,
        )
        right = build_amount_tree(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
            forced_time_series_group=forced_time_series_group,
            forced_concept_group=forced_concept_group,
            allow_time_aggregates=allow_time_aggregates,
            forced_statement_group=forced_statement_group,
            forced_section_group=section_group,
            forced_aggregation_group=agg_group,
        )
        return make_node(
            idgen=idgen,
            op="sum",
            family="amount",
            left=left,
            right=right,
            depth=depth,
        )

    if op == "diff":
        left = build_amount_tree(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
            forced_time_series_group=forced_time_series_group,
            forced_concept_group=forced_concept_group,
            allow_time_aggregates=allow_time_aggregates,
            forced_statement_group=forced_statement_group,
            forced_section_group=forced_section_group,
            forced_aggregation_group=forced_aggregation_group,
        )
        right = build_amount_tree(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
            forced_time_series_group=forced_time_series_group,
            forced_concept_group=forced_concept_group,
            allow_time_aggregates=allow_time_aggregates,
            forced_statement_group=forced_statement_group,
            forced_section_group=forced_section_group,
            forced_aggregation_group=forced_aggregation_group,
        )
        return make_node(
            idgen=idgen,
            op="diff",
            family="amount",
            left=left,
            right=right,
            depth=depth,
        )

    if op == "mul":
        left = build_amount_tree(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
            forced_time_series_group=forced_time_series_group,
            forced_concept_group=forced_concept_group,
            allow_time_aggregates=allow_time_aggregates,
            forced_statement_group=forced_statement_group,
            forced_section_group=forced_section_group,
            forced_aggregation_group=forced_aggregation_group,
        )
        right = build_ratio_tree(
            depth=depth - 1,
            rng=rng,
            idgen=idgen,
            context_group=context_group,
            entity_group=entity_group,
            derived_prob=derived_prob,
        )
        return make_node(
            idgen=idgen,
            op="mul",
            family="amount",
            left=left,
            right=right,
            depth=depth,
        )

    if op in TIME_AGG_OPS:
        years_group = idgen.group("Y")
        concept_group = forced_concept_group or idgen.group("T")
        return make_time_agg(
            idgen=idgen,
            op=op,
            family="amount",
            depth=depth,
            entity_group=entity_group,
            concept_group=concept_group,
            over_years_group=years_group,
            statement_group=forced_statement_group,
            section_group=forced_section_group,
            aggregation_group=forced_aggregation_group,
        )

    raise ValueError(f"Unsupported amount op: {op}")


# ---------------------------------------------------------------------
# 7. Formula expansion helpers
# ---------------------------------------------------------------------
def expand_formula_reference(name: str) -> Any:
    if name not in DERIVED_CONCEPTS:
        return name

    spec = DERIVED_CONCEPTS[name]
    op = spec["formula"]["op"]
    args = spec["formula"]["args"]
    expanded_args = [expand_formula_reference(arg) for arg in args]
    return (op, tuple(expanded_args))


def normalize_symbolic(expr: Any) -> Any:
    if isinstance(expr, str):
        return expr

    op, args = expr
    norm_args = tuple(normalize_symbolic(arg) for arg in args)

    if op == "sum":
        norm_args = tuple(sorted(norm_args, key=repr))

    return (op, norm_args)


def symbolic_from_tree(tree: Dict[str, Any]) -> Any:
    kind = tree["kind"]

    if kind == "leaf":
        return f"LEAF:{tree['name']}"

    if kind == "derived_concept":
        return normalize_symbolic(expand_formula_reference(tree["name"]))

    if kind == "time_agg":
        return (
            "time_agg",
            (
                tree["op"],
                tree["entity_group"],
                tree["concept_group"],
                tree["over_years_group"],
            ),
        )

    if kind == "node":
        left = symbolic_from_tree(tree["left"])
        right = symbolic_from_tree(tree["right"])
        return normalize_symbolic((tree["op"], (left, right)))

    raise ValueError(f"Unknown tree kind: {kind}")


def protected_signatures() -> Dict[str, Any]:
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


def violates_protected_canonical_form(tree: Dict[str, Any], protected: Dict[str, Any]) -> Optional[str]:
    if tree["kind"] == "time_agg":
        return None

    tree_sig = symbolic_from_tree(tree)

    for concept_name, concept_sig in protected.items():
        if tree_sig == concept_sig and not contains_named_derived(tree, concept_name):
            return concept_name

    return None


# ---------------------------------------------------------------------
# 8. Counting and diagnostics
# ---------------------------------------------------------------------
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


def actual_tree_depth(tree: Dict[str, Any]) -> int:
    if tree["kind"] in {"leaf", "derived_concept", "time_agg"}:
        return 0
    return 1 + max(actual_tree_depth(tree["left"]), actual_tree_depth(tree["right"]))


# ---------------------------------------------------------------------
# 9. Sampling wrapper with rejection
# ---------------------------------------------------------------------
def sample_tree_with_rejection(
    max_depth: int,
    rng: random.Random,
    idgen: IdGen,
    derived_prob: float,
    max_attempts: int = 200,
) -> Tuple[Dict[str, Any], Optional[str]]:
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0.")
    if not (0.0 <= derived_prob <= 1.0):
        raise ValueError("derived_prob must be within [0, 1].")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0.")

    protected = protected_signatures()
    last_reason: Optional[str] = None

    for _ in range(max_attempts):
        tree = build_amount_tree(
            depth=max_depth,
            rng=rng,
            idgen=idgen,
            derived_prob=derived_prob,
        )
        violation = violates_protected_canonical_form(tree, protected)
        if violation is None:
            return tree, last_reason
        last_reason = f"Rejected because tree duplicated protected concept: {violation}"

    raise RuntimeError(
        f"Failed to sample a valid tree after {max_attempts} attempts. "
        f"Last rejection reason: {last_reason}"
    )


# ---------------------------------------------------------------------
# 10. Main entry point
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one typed financial operator tree for sec10Q data."
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--derived-prob", type=float, default=0.30)
    parser.add_argument("--output", default="output_sec10Q/generated_operator_tree.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    idgen = IdGen()

    tree, rejection_note = sample_tree_with_rejection(
        max_depth=args.depth,
        rng=rng,
        idgen=idgen,
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
