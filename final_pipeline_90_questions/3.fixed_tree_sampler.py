from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

OPS = ("sum", "diff", "ratio", "mul")


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


def make_leaf(idgen: IdGen, family: str, context_group: Optional[str], entity_group: Optional[str]) -> Dict[str, Any]:
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
    elif family == "ratio":
        spec["semantic_type_in"] = ["ratio", "rate"]
        if entity_group is not None:
            spec["entity_group"] = entity_group
    else:
        raise ValueError(f"Unsupported family: {family}")
    return spec


def make_node(idgen: IdGen, op: str, family: str, left: Dict[str, Any], right: Dict[str, Any], depth: int) -> Dict[str, Any]:
    return {
        "kind": "node",
        "node_id": idgen.node_id(),
        "op": op,
        "family": family,
        "depth": depth,
        "left": left,
        "right": right,
    }


def build_ratio_tree(depth: int, rng: random.Random, idgen: IdGen, context_group: str, entity_group: str) -> Dict[str, Any]:
    if depth == 0:
        return make_leaf(idgen, "ratio", context_group=None, entity_group=entity_group)

    left = build_amount_tree(depth - 1, rng, idgen, context_group=context_group, entity_group=entity_group)
    right = build_amount_tree(depth - 1, rng, idgen, context_group=context_group, entity_group=entity_group)
    return make_node(idgen, "ratio", "ratio", left, right, depth)


def build_amount_tree(depth: int, rng: random.Random, idgen: IdGen, context_group: Optional[str] = None, entity_group: Optional[str] = None) -> Dict[str, Any]:
    if context_group is None:
        context_group = idgen.group("C")
    if entity_group is None:
        entity_group = idgen.group("E")

    if depth == 0:
        return make_leaf(idgen, "amount", context_group=context_group, entity_group=entity_group)

    op = rng.choice(OPS)
    if op in {"sum", "diff"}:
        left = build_amount_tree(depth - 1, rng, idgen, context_group=context_group, entity_group=entity_group)
        right = build_amount_tree(depth - 1, rng, idgen, context_group=context_group, entity_group=entity_group)
        return make_node(idgen, op, "amount", left, right, depth)

    if op == "ratio":
        left = build_amount_tree(depth - 1, rng, idgen, context_group=context_group, entity_group=entity_group)
        right = build_amount_tree(depth - 1, rng, idgen, context_group=context_group, entity_group=entity_group)
        return make_node(idgen, "ratio", "ratio", left, right, depth)

    left = build_amount_tree(depth - 1, rng, idgen, context_group=context_group, entity_group=entity_group)
    right = build_ratio_tree(depth - 1, rng, idgen, context_group=context_group, entity_group=entity_group)
    return make_node(idgen, "mul", "amount", left, right, depth)


def count_nodes(tree: Dict[str, Any]) -> Dict[str, int]:
    if tree["kind"] == "leaf":
        return {"internal_nodes": 0, "leaves": 1, "total_nodes": 1}
    left = count_nodes(tree["left"])
    right = count_nodes(tree["right"])
    return {
        "internal_nodes": 1 + left["internal_nodes"] + right["internal_nodes"],
        "leaves": left["leaves"] + right["leaves"],
        "total_nodes": 1 + left["total_nodes"] + right["total_nodes"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one typed full binary operator tree.")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="output/generated_operator_tree.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    idgen = IdGen()
    tree = build_amount_tree(args.depth, rng, idgen)
    stats = count_nodes(tree)

    payload = {
        "max_depth": args.depth,
        "seed": args.seed,
        "ops": list(OPS),
        "stats": stats,
        "tree": tree,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote typed operator tree to {output_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
