"""
STEP 3: Build Expression Tree
===============================
Enumerates ALL valid combinations of (operation, operand_a, operand_b)
based on the type compatibility rules from Step 2.

The tree has 3 depths:
  Depth 0 (leaves):  Raw column values          → "revenue_2024"
  Depth 1:           Single operations           → "revenue_2024 - cost_of_goods_sold_2024"
  Depth 2:           Composed operations          → "(revenue_2024 - revenue_2023) / revenue_2023"

Each node stores:
  - The expression (human-readable)
  - The operation used
  - The operands (column names or sub-expressions)
  - Whether it's valid (type-checked)

Output: expression_tree.json (all valid nodes)
"""

import json
from step2_operations import OPERATIONS, can_combine

# ── Load schema from Step 1 ────────────────────────────────

def load_schema():
    with open("output/schema.json") as f:
        return json.load(f)


# ── Tree Node ───────────────────────────────────────────────

def make_leaf(col_name, col_info):
    """Depth-0 node: a raw column reference."""
    return {
        "depth": 0,
        "expression": col_name,
        "operation": None,
        "operands": [col_name],
        "result_type": col_info["type"],
        "unit": col_info.get("unit"),
        "base_name": col_info.get("base_name", col_name),
        "year": col_info.get("year"),
        "desc": col_info.get("desc", col_name),
    }


def make_node(operation_name, left, right=None):
    """Depth-1+ node: an operation applied to operands."""
    op = OPERATIONS[operation_name]
    symbol = op["symbol"]

    if right is None:
        # Unary (aggregation)
        expr = f"{symbol}({left['expression']})"
    else:
        if operation_name in ("change", "pct_change"):
            expr = f"{symbol}({left['base_name']}, {left['year']}→{right['year']})"
        elif operation_name == "ratio":
            expr = f"ratio({left['expression']}, {right['expression']})"
        else:
            expr = f"({left['expression']} {symbol} {right['expression']})"

    result_type = op["result_type"]

    # Figure out the result unit
    if result_type == "numeric" and operation_name in ("add", "subtract", "change"):
        unit = left.get("unit")
    elif result_type == "ratio":
        unit = "ratio"
    elif result_type == "percentage":
        unit = "percentage"
    elif result_type == "boolean":
        unit = "boolean"
    else:
        unit = "derived"

    return {
        "depth": max(left["depth"], right["depth"] if right else 0) + 1,
        "expression": expr,
        "operation": operation_name,
        "operands": [left["expression"]] + ([right["expression"]] if right else []),
        "result_type": result_type,
        "unit": unit,
        "base_name": None,
        "year": None,
        "desc": f"{op['description']}: {expr}",
    }


# ── Build All Valid Depth-1 Combinations ────────────────────

def build_depth1(schema):
    """
    For every pair of columns, check every operation.
    Only keep valid combinations.
    """
    leaves = {col: make_leaf(col, info) for col, info in schema.items()}
    col_names = list(schema.keys())
    numeric_cols = [c for c in col_names if schema[c]["type"] == "numeric"]

    nodes = []
    seen = set()

    # ── Binary operations ──
    binary_ops = [
        "add", "subtract", "multiply", "divide",
        "greater_than", "less_than", "greater_equal", "less_equal",
        "equals", "not_equals",
        "ratio", "change", "pct_change",
    ]

    for op_name in binary_ops:
        op = OPERATIONS[op_name]

        # Choose candidate columns based on operation type
        if op["left_type"] == "numeric":
            left_candidates = numeric_cols
        else:
            left_candidates = col_names

        if op["right_type"] == "numeric":
            right_candidates = numeric_cols
        else:
            right_candidates = col_names

        for col_a in left_candidates:
            for col_b in right_candidates:
                if col_a == col_b:
                    continue

                valid, _ = can_combine(op_name, schema[col_a], schema[col_b])
                if not valid:
                    continue

                node = make_node(op_name, leaves[col_a], leaves[col_b])

                # Dedup: skip if we've seen this exact expression
                if node["expression"] not in seen:
                    seen.add(node["expression"])
                    nodes.append(node)

    # ── Unary operations (aggregations) ──
    unary_ops = ["sum_agg", "avg_agg", "max_agg", "min_agg", "count_agg"]
    for op_name in unary_ops:
        if op_name == "count_agg":
            candidates = col_names
        else:
            candidates = numeric_cols

        for col in candidates:
            valid, _ = can_combine(op_name, schema[col])
            if not valid:
                continue
            node = make_node(op_name, leaves[col])
            if node["expression"] not in seen:
                seen.add(node["expression"])
                nodes.append(node)

    return leaves, nodes


# ── Build Depth-2: Compose depth-1 results ──────────────────

def build_depth2(schema, depth1_nodes, max_depth2=500):
    """
    Take interesting depth-1 results and compose them.
    Example: ratio(revenue_2024 - cogs_2024, revenue_2024) = gross margin
    """
    import random
    random.seed(42)

    depth2_nodes = []
    seen = set()

    # Only compose from numeric depth-1 results
    numeric_d1 = [n for n in depth1_nodes if n["result_type"] in ("numeric", "ratio", "percentage")]

    # Also use leaves as potential second operands
    leaves = {col: make_leaf(col, info) for col, info in schema.items()}
    numeric_leaves = [leaves[c] for c in leaves if schema[c]["type"] == "numeric"]

    composable_ops = ["divide", "ratio", "greater_than", "less_than", "subtract"]

    candidates = []

    # Depth-1 OP leaf (e.g., (revenue - cogs) / revenue)
    for d1_node in numeric_d1:
        for leaf in numeric_leaves:
            for op_name in composable_ops:
                # Type check the combination
                d1_fake_info = {"type": "numeric", "unit": d1_node.get("unit"), "base_name": None, "year": None}
                leaf_info = schema.get(leaf["expression"], {})
                if leaf_info.get("type") != "numeric":
                    continue
                valid, _ = can_combine(op_name, d1_fake_info, leaf_info)
                if op_name in ("divide", "ratio"):
                    valid = True  # allow derived / numeric
                if not valid:
                    continue
                candidates.append((op_name, d1_node, leaf))

    # Sample to keep manageable
    if len(candidates) > max_depth2:
        candidates = random.sample(candidates, max_depth2)

    for op_name, left, right in candidates:
        node = make_node(op_name, left, right)
        node["depth"] = 2
        if node["expression"] not in seen:
            seen.add(node["expression"])
            depth2_nodes.append(node)

    return depth2_nodes


# ── Main ────────────────────────────────────────────────────

def main():
    schema = load_schema()

    print("Building expression tree...\n")

    # Depth 0: leaves
    leaves = {col: make_leaf(col, info) for col, info in schema.items()}
    print(f"  Depth 0 (leaves):        {len(leaves):>6} nodes")

    # Depth 1: single operations
    _, depth1 = build_depth1(schema)
    print(f"  Depth 1 (single ops):    {len(depth1):>6} nodes")

    # Depth 2: composed operations
    depth2 = build_depth2(schema, depth1, max_depth2=500)
    print(f"  Depth 2 (composed):      {len(depth2):>6} nodes")

    total = len(leaves) + len(depth1) + len(depth2)
    print(f"  ─────────────────────────────")
    print(f"  Total valid nodes:       {total:>6}")

    # ── Stats by operation ──
    op_counts = {}
    for node in depth1 + depth2:
        op = node["operation"]
        op_counts[op] = op_counts.get(op, 0) + 1

    print(f"\n  By operation:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        symbol = OPERATIONS[op]["symbol"]
        print(f"    {symbol:>5} ({op:<16}): {count:>5}")

    # ── Stats by result type ──
    type_counts = {}
    for node in depth1 + depth2:
        t = node["result_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  By result type:")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<15}: {count:>5}")

    # ── Save ──
    all_nodes = list(leaves.values()) + depth1 + depth2
    with open("output/expression_tree.json", "w") as f:
        json.dump(all_nodes, f, indent=2)

    print(f"\n  Samples (depth 1):")
    for node in depth1[:8]:
        print(f"    {node['expression']}")

    print(f"\n  Samples (depth 2):")
    for node in depth2[:5]:
        print(f"    {node['expression']}")

    print(f"\n✓ Saved {len(all_nodes)} nodes to output/expression_tree.json")


if __name__ == "__main__":
    main()