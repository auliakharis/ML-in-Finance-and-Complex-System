"""
STEP 4: Sample & Execute
==========================
Randomly sample nodes from the expression tree, then execute each
against the actual spreadsheet data to get verifiable ground truth.

For each sampled node, we get:
  - The expression
  - Which company (row) it applies to
  - The computed answer
  - A full computation trace

Output: sampled_executed.json
"""

import json
import csv
import random

from step2_operations import OPERATIONS

random.seed(42)


# ── Load everything ─────────────────────────────────────────

def load_data():
    with open("output/financial_spreadsheet.csv") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric strings to numbers
            clean = {}
            for k, v in row.items():
                try:
                    clean[k] = float(v) if "." in v else int(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return rows


def load_tree():
    with open("output/expression_tree.json") as f:
        return json.load(f)


def load_schema():
    with open("output/schema.json") as f:
        return json.load(f)


# ── Evaluate an expression against one row ──────────────────

def evaluate(node, row, all_rows=None):
    """
    Evaluate a tree node against a data row.
    Returns (value, trace_string) or (None, error_string).
    """
    op = node["operation"]
    operands = node["operands"]

    # ── Depth 0: leaf (column lookup) ──
    if node["depth"] == 0:
        col = node["expression"]
        val = row.get(col)
        return val, f"{col} = {val}"

    # ── Aggregations (work across all rows) ──
    if op in ("sum_agg", "avg_agg", "max_agg", "min_agg", "count_agg"):
        if all_rows is None:
            return None, "Need all rows for aggregation"
        col = operands[0]
        values = [r.get(col) for r in all_rows if isinstance(r.get(col), (int, float))]

        if op == "count_agg":
            val = len(values)
        elif not values:
            return None, "No numeric values"
        elif op == "sum_agg":
            val = round(sum(values), 2)
        elif op == "avg_agg":
            val = round(sum(values) / len(values), 2)
        elif op == "max_agg":
            val = max(values)
        elif op == "min_agg":
            val = min(values)

        return val, f"{OPERATIONS[op]['symbol']}({col}) = {val}"

    # ── Binary operations ──
    if len(operands) < 2:
        return None, "Not enough operands"

    col_a, col_b = operands[0], operands[1]
    a = row.get(col_a)
    b = row.get(col_b)

    if a is None or b is None:
        return None, f"Missing value: {col_a}={a}, {col_b}={b}"

    # Handle categorical comparisons
    if op == "equals":
        val = (a == b)
        return val, f"{col_a}({a}) = {col_b}({b}) → {val}"
    if op == "not_equals":
        val = (a != b)
        return val, f"{col_a}({a}) ≠ {col_b}({b}) → {val}"

    # Numeric operations
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None, f"Non-numeric: {col_a}={a}, {col_b}={b}"

    try:
        if op == "add":
            val = round(a + b, 2)
            trace = f"{col_a}({a}) + {col_b}({b}) = {val}"
        elif op == "subtract":
            val = round(a - b, 2)
            trace = f"{col_a}({a}) - {col_b}({b}) = {val}"
        elif op == "multiply":
            val = round(a * b, 2)
            trace = f"{col_a}({a}) × {col_b}({b}) = {val}"
        elif op in ("divide", "ratio"):
            if b == 0:
                return None, "Division by zero"
            val = round(a / b, 4)
            trace = f"{col_a}({a}) / {col_b}({b}) = {val}"
        elif op == "greater_than":
            val = a > b
            trace = f"{col_a}({a}) > {col_b}({b}) → {val}"
        elif op == "less_than":
            val = a < b
            trace = f"{col_a}({a}) < {col_b}({b}) → {val}"
        elif op == "greater_equal":
            val = a >= b
            trace = f"{col_a}({a}) ≥ {col_b}({b}) → {val}"
        elif op == "less_equal":
            val = a <= b
            trace = f"{col_a}({a}) ≤ {col_b}({b}) → {val}"
        elif op == "change":
            val = round(b - a, 2)  # convention: newer - older
            trace = f"Δ = {col_b}({b}) - {col_a}({a}) = {val}"
        elif op == "pct_change":
            if a == 0:
                return None, "Base value is zero"
            val = round((b - a) / abs(a) * 100, 2)
            trace = f"%Δ = ({col_b}({b}) - {col_a}({a})) / |{a}| × 100 = {val}%"
        else:
            return None, f"Unknown op: {op}"

        return val, trace

    except Exception as e:
        return None, str(e)


def evaluate_depth2(node, row, all_rows, tree_lookup):
    """Evaluate a depth-2 node by first evaluating its sub-expressions."""
    operands = node["operands"]
    op = node["operation"]

    # Evaluate left operand (should be a depth-1 expression)
    left_expr = operands[0]
    left_node = tree_lookup.get(left_expr)
    if left_node is None:
        return None, f"Cannot find sub-expression: {left_expr}"

    left_val, left_trace = evaluate(left_node, row, all_rows)
    if left_val is None:
        return None, f"Left failed: {left_trace}"

    # Right operand (leaf or depth-1)
    if len(operands) > 1:
        right_expr = operands[1]
        right_node = tree_lookup.get(right_expr)
        if right_node is None:
            # Try as raw column
            right_val = row.get(right_expr)
            right_trace = f"{right_expr} = {right_val}"
        else:
            right_val, right_trace = evaluate(right_node, row, all_rows)

        if right_val is None:
            return None, f"Right failed: {right_trace}"

        # Now apply the outer operation
        if op in ("divide", "ratio"):
            if right_val == 0:
                return None, "Division by zero"
            val = round(left_val / right_val, 4) if isinstance(left_val, (int, float)) else None
            trace = f"({left_trace}) / ({right_trace}) = {val}"
        elif op == "subtract":
            val = round(left_val - right_val, 2)
            trace = f"({left_trace}) - ({right_trace}) = {val}"
        elif op == "greater_than":
            val = left_val > right_val
            trace = f"({left_trace}) > ({right_trace}) → {val}"
        elif op == "less_than":
            val = left_val < right_val
            trace = f"({left_trace}) < ({right_trace}) → {val}"
        else:
            val = None
            trace = f"Unhandled depth-2 op: {op}"

        return val, trace

    return None, "Depth-2 with insufficient operands"


# ── Main ────────────────────────────────────────────────────

def main():
    rows = load_data()
    tree = load_tree()
    schema = load_schema()

    # Build lookup by expression
    tree_lookup = {n["expression"]: n for n in tree}

    # Separate by depth
    depth0 = [n for n in tree if n["depth"] == 0]
    depth1 = [n for n in tree if n["depth"] == 1]
    depth2 = [n for n in tree if n["depth"] == 2]

    # ── Sample nodes ──
    N_SAMPLES = 500

    # Proportional sampling: more depth-1 (the sweet spot)
    sampled = (
        random.sample(depth0, min(50, len(depth0))) +
        random.sample(depth1, min(300, len(depth1))) +
        random.sample(depth2, min(150, len(depth2)))
    )
    random.shuffle(sampled)
    sampled = sampled[:N_SAMPLES]

    # ── Execute each sample against a random company ──
    results = []
    success = 0
    fail = 0

    for node in sampled:
        # Pick a random company
        row = random.choice(rows)
        company_name = row["company_name"]

        # Execute
        if node["depth"] <= 1:
            val, trace = evaluate(node, row, rows)
        else:
            val, trace = evaluate_depth2(node, row, rows, tree_lookup)

        if val is not None:
            success += 1
            results.append({
                "company": company_name,
                "expression": node["expression"],
                "operation": node["operation"],
                "depth": node["depth"],
                "result_type": node["result_type"],
                "answer": val,
                "trace": trace,
                "operands": node["operands"],
            })
        else:
            fail += 1

    # ── Save ──
    with open("output/sampled_executed.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Sampled {len(sampled)} nodes, executed against random companies\n")
    print(f"  Successful: {success}")
    print(f"  Failed:     {fail}\n")

    by_depth = {}
    for r in results:
        by_depth[r["depth"]] = by_depth.get(r["depth"], 0) + 1
    print(f"  By depth:")
    for d in sorted(by_depth):
        print(f"    Depth {d}: {by_depth[d]}")

    by_type = {}
    for r in results:
        by_type[r["result_type"]] = by_type.get(r["result_type"], 0) + 1
    print(f"\n  By result type:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {t:<15}: {c}")

    print(f"\n  Samples:")
    for r in results[:10]:
        print(f"    [{r['depth']}] {r['expression']}")
        print(f"        Company: {r['company']}")
        print(f"        Answer:  {r['answer']}")
        print(f"        Trace:   {r['trace']}")
        print()

    print(f"✓ Saved {len(results)} results to output/sampled_executed.json")


if __name__ == "__main__":
    main()