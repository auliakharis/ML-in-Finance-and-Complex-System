"""
run_pipeline.py
================
Runs the full Q&A generation pipeline on the 10-Q financial spreadsheet.
Adapted from their Steps 2-5 to work with 10-Q column naming.

Steps:
  1. generate_schema.py  → schema.json
  2. operations_10q.py   → defines valid operations
  3. Build expression tree (inline)
  4. Sample & execute against CSV data
  5. rewrite_10q.py      → natural language questions

Output: final_qa_dataset.json, final_qa_dataset.csv

Usage:
  python run_pipeline.py
  python run_pipeline.py --csv financial_spreadsheet.csv --n 500
"""

import argparse
import csv
import json
import random
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from generate_csv   import build_schema, parse_column
from operations_10q  import OPERATIONS, can_combine
from rewrite_10q     import rewrite_as_question, format_value


# ---------------------------------------------------------------------------
# Step 1: Load data and schema
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            clean = {}
            for k, v in row.items():
                try:
                    clean[k] = float(v)
                except (ValueError, TypeError):
                    clean[k] = v
            rows.append(clean)
    return rows


# ---------------------------------------------------------------------------
# Step 3: Build expression tree
# ---------------------------------------------------------------------------

def make_leaf(col: str, info: dict) -> dict:
    return {
        "depth": 0,
        "expression": col,
        "operation": None,
        "operands": [col],
        "result_type": info["type"],
        "unit": info.get("unit"),
        "base_name": info.get("base_name", col),
        "period_tag": info.get("period_tag"),
        "prefix": info.get("prefix"),
        "date": info.get("date"),
        "desc": info.get("desc", col),
    }


def make_node(op_name: str, left: dict, right: dict = None) -> dict:
    op = OPERATIONS[op_name]
    symbol = op["symbol"]

    if right is None:
        expr = f"{symbol}({left['expression']})"
    elif op_name in ("change", "pct_change"):
        expr = (f"{symbol}({left['base_name']}, "
                f"{left['period_tag']} → {right['period_tag']})")
    elif op_name == "ratio":
        expr = f"ratio({left['expression']}, {right['expression']})"
    else:
        expr = f"({left['expression']} {symbol} {right['expression']})"

    result_type = op["result_type"]
    if result_type == "numeric" and op_name in ("add", "subtract", "change"):
        unit = left.get("unit")
    elif result_type in ("ratio", "percentage", "boolean"):
        unit = result_type
    else:
        unit = "derived"

    return {
        "depth": max(left["depth"], right["depth"] if right else 0) + 1,
        "expression": expr,
        "operation": op_name,
        "operands": [left["expression"]] + ([right["expression"]] if right else []),
        "result_type": result_type,
        "unit": unit,
        "base_name": None,
        "period_tag": None,
        "prefix": None,
        "date": None,
        "desc": f"{op['description']}: {expr}",
    }


def build_tree(schema: dict) -> tuple:
    leaves = {col: make_leaf(col, info) for col, info in schema.items()}
    numeric_cols = [c for c in schema if schema[c]["type"] == "numeric"]
    all_cols = list(schema.keys())

    depth1 = []
    seen = set()

    binary_ops = [
        "add", "subtract", "multiply", "divide",
        "greater_than", "less_than", "greater_equal", "less_equal",
        "equals", "not_equals", "ratio", "change", "pct_change",
    ]
    unary_ops = ["sum_agg", "avg_agg", "max_agg", "min_agg", "count_agg"]

    for op_name in binary_ops:
        op = OPERATIONS[op_name]
        left_cands  = numeric_cols if op["left_type"] == "numeric" else all_cols
        right_cands = numeric_cols if op["right_type"] == "numeric" else all_cols

        for col_a in left_cands:
            for col_b in right_cands:
                if col_a == col_b:
                    continue
                valid, _ = can_combine(op_name, schema[col_a], schema[col_b])
                if not valid:
                    continue
                node = make_node(op_name, leaves[col_a], leaves[col_b])
                if node["expression"] not in seen:
                    seen.add(node["expression"])
                    depth1.append(node)

    for op_name in unary_ops:
        cands = numeric_cols if op_name != "count_agg" else all_cols
        for col in cands:
            valid, _ = can_combine(op_name, schema[col])
            if not valid:
                continue
            node = make_node(op_name, leaves[col])
            if node["expression"] not in seen:
                seen.add(node["expression"])
                depth1.append(node)

    # Depth 2: compose depth-1 numeric results with leaves
    # Only use depth-1 nodes that have a single consistent date
    import re as _re
    def _single_date(expr):
        dates = _re.findall(r'[A-Z][a-z]{2}_\d{2}_\d{4}', expr)
        return len(set(dates)) <= 1

    numeric_d1 = [n for n in depth1
                  if n["result_type"] in ("numeric", "ratio", "percentage")
                  and _single_date(n["expression"])]
    numeric_leaves = [leaves[c] for c in numeric_cols]
    composable_ops = ["divide", "ratio", "greater_than", "less_than", "subtract"]

    depth2 = []
    seen2 = set()
    candidates = []
    for d1 in numeric_d1:
        for leaf in numeric_leaves:
            for op_name in composable_ops:
                d1_info = {"type": "numeric", "unit": d1.get("unit"),
                           "base_name": None, "period_tag": None, "prefix": None}
                leaf_info = schema.get(leaf["expression"], {})
                if leaf_info.get("type") != "numeric":
                    continue
                valid, _ = can_combine(op_name, d1_info, leaf_info)
                if op_name in ("divide", "ratio"):
                    valid = True
                if valid:
                    candidates.append((op_name, d1, leaf))

    random.shuffle(candidates)
    for op_name, left, right in candidates[:500]:
        node = make_node(op_name, left, right)
        node["depth"] = 2
        if node["expression"] not in seen2:
            seen2.add(node["expression"])
            depth2.append(node)

    return leaves, depth1, depth2


# ---------------------------------------------------------------------------
# Step 4: Execute
# ---------------------------------------------------------------------------

def evaluate(node: dict, row: dict, all_rows: list = None):
    op = node["operation"]
    operands = node["operands"]

    if node["depth"] == 0:
        col = node["expression"]
        val = row.get(col)
        return val, f"{col} = {val}"

    if op in ("sum_agg", "avg_agg", "max_agg", "min_agg", "count_agg"):
        if all_rows is None:
            return None, "Need all rows"
        col = operands[0]
        vals = [r.get(col) for r in all_rows if isinstance(r.get(col), (int, float))]
        if op == "count_agg":
            val = len(vals)
        elif not vals:
            return None, "No values"
        elif op == "sum_agg": val = round(sum(vals), 2)
        elif op == "avg_agg": val = round(sum(vals) / len(vals), 2)
        elif op == "max_agg": val = max(vals)
        elif op == "min_agg": val = min(vals)
        return val, f"{OPERATIONS[op]['symbol']}({col}) = {val}"

    if len(operands) < 2:
        return None, "Not enough operands"

    col_a, col_b = operands[0], operands[1]
    a = row.get(col_a)
    b = row.get(col_b)

    if a is None or b is None:
        return None, f"Missing: {col_a}={a}, {col_b}={b}"

    if op == "equals":
        return (a == b), f"{col_a}({a}) = {col_b}({b})"
    if op == "not_equals":
        return (a != b), f"{col_a}({a}) ≠ {col_b}({b})"

    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None, f"Non-numeric: {a}, {b}"

    try:
        if op == "add":        val, tr = round(a+b,2), f"{a} + {b} = {round(a+b,2)}"
        elif op == "subtract": val, tr = round(a-b,2), f"{a} - {b} = {round(a-b,2)}"
        elif op == "multiply": val, tr = round(a*b,2), f"{a} × {b} = {round(a*b,2)}"
        elif op in ("divide","ratio"):
            if b == 0: return None, "Division by zero"
            val, tr = round(a/b,4), f"{a} / {b} = {round(a/b,4)}"
        elif op == "greater_than":  val, tr = a>b,  f"{a} > {b} → {a>b}"
        elif op == "less_than":     val, tr = a<b,  f"{a} < {b} → {a<b}"
        elif op == "greater_equal": val, tr = a>=b, f"{a} ≥ {b} → {a>=b}"
        elif op == "less_equal":    val, tr = a<=b, f"{a} ≤ {b} → {a<=b}"
        elif op == "change":
            val, tr = round(b-a,2), f"{b} - {a} = {round(b-a,2)}"
        elif op == "pct_change":
            if a == 0: return None, "Base is zero"
            val = round((b-a)/abs(a)*100, 2)
            tr  = f"({b} - {a}) / |{a}| × 100 = {val}%"
        else:
            return None, f"Unknown op: {op}"
        return val, tr
    except Exception as e:
        return None, str(e)


def evaluate_depth2(node: dict, row: dict, all_rows: list, lookup: dict):
    operands = node["operands"]
    op = node["operation"]

    left_node = lookup.get(operands[0])
    if left_node is None:
        return None, "Left sub-expression not found"
    left_val, left_tr = evaluate(left_node, row, all_rows)
    if left_val is None:
        return None, f"Left failed: {left_tr}"

    if len(operands) > 1:
        right_node = lookup.get(operands[1])
        right_val = row.get(operands[1]) if right_node is None else None
        right_tr  = f"{operands[1]} = {right_val}"
        if right_node:
            right_val, right_tr = evaluate(right_node, row, all_rows)
        if right_val is None:
            return None, f"Right failed: {right_tr}"

        if op in ("divide", "ratio"):
            if right_val == 0: return None, "Division by zero"
            val = round(left_val / right_val, 4)
            return val, f"({left_tr}) / ({right_tr}) = {val}"
        elif op == "subtract":
            val = round(left_val - right_val, 2)
            return val, f"({left_tr}) - ({right_tr}) = {val}"
        elif op == "greater_than":
            return left_val > right_val, f"({left_tr}) > ({right_tr})"
        elif op == "less_than":
            return left_val < right_val, f"({left_tr}) < ({right_tr})"

    return None, "Insufficient operands"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    default="financial_spreadsheet.csv")
    parser.add_argument("--n",      type=int, default=500)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", default="final_qa_dataset")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading data...")
    rows   = load_csv(args.csv)
    schema = build_schema(csv.DictReader(open(args.csv)).fieldnames)
    print(f"  {len(rows)} companies, {len(schema)} columns")

    print("Building expression tree...")
    leaves, depth1, depth2 = build_tree(schema)
    all_nodes = list(leaves.values()) + depth1 + depth2
    print(f"  Depth 0: {len(leaves)}  Depth 1: {len(depth1)}  Depth 2: {len(depth2)}")
    print(f"  Total:   {len(all_nodes)}")

    print(f"\nSampling {args.n} expressions and executing...")
    d0 = list(leaves.values())
    sampled = (
        random.sample(d0,     min(50,          len(d0)))     +
        random.sample(depth1, min(args.n*3//5, len(depth1))) +
        random.sample(depth2, min(args.n*2//5, len(depth2)))
    )[:args.n]
    random.shuffle(sampled)

    lookup = {n["expression"]: n for n in all_nodes}
    results = []
    fail = 0

    for node in sampled:
        row = random.choice(rows)
        company = row.get("company_name", "Unknown")

        if node["depth"] <= 1:
            val, trace = evaluate(node, row, rows)
        else:
            val, trace = evaluate_depth2(node, row, rows, lookup)

        if val is not None:
            question = rewrite_as_question({
                "company": company, "operation": node["operation"],
                "depth": node["depth"], "operands": node["operands"],
                "expression": node["expression"],
                "result_type": node["result_type"],
            })
            if not question.startswith("["):
                results.append({
                    "id":                f"FQ-{len(results)+1:05d}",
                    "company":           company,
                    "depth":             node["depth"],
                    "operation":         node["operation"],
                    "expression":        node["expression"],
                    "question":          question,
                    "answer":            val,
                    "answer_formatted":  format_value(val, node["result_type"]),
                    "result_type":       node["result_type"],
                    "computation_trace": trace,
                })
        else:
            fail += 1

    print(f"  Success: {len(results)}  Failed: {fail}")

    # Save JSON
    with open(f"{args.output}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save CSV
    with open(f"{args.output}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id","company","depth","operation","expression",
                         "question","answer","answer_formatted",
                         "result_type","computation_trace"])
        for d in results:
            writer.writerow([d["id"],d["company"],d["depth"],d["operation"],
                             d["expression"],d["question"],d["answer"],
                             d["answer_formatted"],d["result_type"],
                             d["computation_trace"]])

    print(f"\n✓ Saved {len(results)} QA pairs → {args.output}.json / .csv")

    # Summary
    by_depth = {}
    for r in results:
        by_depth[r["depth"]] = by_depth.get(r["depth"], 0) + 1
    print("\n  By depth:")
    for d in sorted(by_depth):
        print(f"    Depth {d}: {by_depth[d]}")

    by_op = {}
    for r in results:
        by_op[r["operation"] or "lookup"] = by_op.get(r["operation"] or "lookup", 0) + 1
    print("\n  By operation:")
    for op, c in sorted(by_op.items(), key=lambda x: -x[1])[:8]:
        print(f"    {op:<20}: {c}")

    print("\n  Sample QA pairs:")
    for depth in [0, 1, 2]:
        subset = [r for r in results if r["depth"] == depth]
        if subset:
            s = random.choice(subset)
            print(f"\n  [{s['id']}] Depth {s['depth']} | {s['operation'] or 'lookup'}")
            print(f"  Q: {s['question']}")
            print(f"  A: {s['answer_formatted']}")
            print(f"  Trace: {s['computation_trace']}")


if __name__ == "__main__":
    main()