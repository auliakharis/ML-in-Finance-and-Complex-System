"""
STEP 5: Rewrite as Formal Questions
=====================================
Takes each executed expression and converts it into a very formal,
precise natural language question.

The questions are intentionally FORMAL and EXPLICIT — they specify
exactly what data, what operation, and what time period.

Examples:
  get_value(revenue_2024) for Apex →
    "What was the total revenue reported by Apex Dynamics Corp for the fiscal year ending 2024?"

  ratio(net_income_2024, total_equity_2024) for Cobalt →
    "What is the ratio of net income to total stockholders' equity for Cobalt Semiconductor
     as reported in the fiscal year 2024 financial statements?"

  pct_change(revenue, 2023, 2024) for Atlas →
    "What is the percentage change in total revenue for Atlas Global Industries
     from fiscal year 2023 to fiscal year 2024?"

Output: final_qa_dataset.json, final_qa_dataset.csv
"""

import json
import csv
import random
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

random.seed(42)

# ── Human-readable labels for column base names ─────────────

LABELS = {
    "revenue":              "total revenue",
    "cost_of_goods_sold":   "cost of goods sold",
    "gross_profit":         "gross profit",
    "operating_expenses":   "total operating expenses",
    "operating_income":     "operating income",
    "net_income":           "net income",
    "total_assets":         "total assets",
    "total_liabilities":    "total liabilities",
    "total_equity":         "total stockholders' equity",
    "current_assets":       "total current assets",
    "current_liabilities":  "total current liabilities",
    "cash":                 "cash and cash equivalents",
    "long_term_debt":       "long-term debt",
    "accounts_receivable":  "net accounts receivable",
    "inventories":          "total inventories",
    "capex":                "capital expenditures",
    "dividends_paid":       "cash dividends paid",
    "shares_outstanding":   "total shares outstanding",
    "stock_price":          "closing stock price",
    "employees":            "total number of employees",
    "company_name":         "company name",
    "ticker":               "ticker symbol",
    "sector":               "industry sector",
    "country":              "country of headquarters",
    "exchange":             "stock exchange listing",
    "credit_rating":        "credit rating",
}


def col_label(col_name):
    """Turn a column name like 'revenue_2024' into 'total revenue (FY 2024)'."""
    # Check if it ends with a year
    parts = col_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base, year = parts[0], parts[1]
        label = LABELS.get(base, base.replace("_", " "))
        return label, year
    else:
        label = LABELS.get(col_name, col_name.replace("_", " "))
        return label, None


def format_value(val, result_type):
    """Format a value for the answer."""
    if val is None:
        return "N/A"
    if result_type == "boolean":
        return "Yes" if val else "No"
    if result_type == "percentage":
        return f"{val}%"
    if result_type == "count":
        return str(int(val))
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        if result_type == "ratio":
            return f"{val:.4f}"
        if abs(val) >= 1000:
            return f"{val:,.0f}"
        return f"{val:,.2f}"
    return str(val)


# ── Question Templates by Operation Type ────────────────────

def rewrite_leaf(company, col):
    """Depth 0: simple lookup."""
    label, year = col_label(col)
    if year:
        return (
            f"What was the {label} reported by {company} "
            f"for the fiscal year ending {year}?"
        )
    else:
        return f"What is the {label} of {company}?"


def rewrite_binary(company, op, col_a, col_b, result_type):
    """Depth 1: binary operation between two columns."""
    label_a, year_a = col_label(col_a)
    label_b, year_b = col_label(col_b)

    # Pick the year context
    if year_a and year_b and year_a == year_b:
        year_ctx = f"for fiscal year {year_a}"
    elif year_a and year_b:
        year_ctx = f"from fiscal year {year_a} to fiscal year {year_b}"
    elif year_a:
        year_ctx = f"for fiscal year {year_a}"
    else:
        year_ctx = ""

    if op == "add":
        return (
            f"What is the sum of {label_a} and {label_b} "
            f"for {company} {year_ctx}?"
        )

    if op == "subtract":
        return (
            f"What is the difference between {label_a} and {label_b} "
            f"(i.e., {label_a} minus {label_b}) for {company} {year_ctx}?"
        )

    if op == "multiply":
        return (
            f"What is the product of {label_a} and {label_b} "
            f"for {company} {year_ctx}?"
        )

    if op in ("divide", "ratio"):
        return (
            f"What is the ratio of {label_a} to {label_b} "
            f"for {company} {year_ctx}?"
        )

    if op == "greater_than":
        return (
            f"Is {label_a} strictly greater than {label_b} "
            f"for {company} {year_ctx}?"
        )

    if op == "less_than":
        return (
            f"Is {label_a} strictly less than {label_b} "
            f"for {company} {year_ctx}?"
        )

    if op == "greater_equal":
        return (
            f"Is {label_a} greater than or equal to {label_b} "
            f"for {company} {year_ctx}?"
        )

    if op == "less_equal":
        return (
            f"Is {label_a} less than or equal to {label_b} "
            f"for {company} {year_ctx}?"
        )

    if op == "equals":
        return (
            f"Is the {label_a} equal to {label_b} for {company}?"
        )

    if op == "not_equals":
        return (
            f"Is the {label_a} different from {label_b} for {company}?"
        )

    if op == "change":
        return (
            f"What is the absolute change in {label_a} for {company} "
            f"from fiscal year {year_a} to fiscal year {year_b}?"
        )

    if op == "pct_change":
        return (
            f"What is the percentage change in {label_a} for {company} "
            f"from fiscal year {year_a} to fiscal year {year_b}?"
        )

    return f"[Unhandled operation: {op}]"


def rewrite_aggregation(op, col):
    """Unary aggregation across all companies."""
    label, year = col_label(col)
    year_ctx = f"for fiscal year {year}" if year else ""

    symbol_map = {
        "sum_agg": "total sum",
        "avg_agg": "average (arithmetic mean)",
        "max_agg": "maximum value",
        "min_agg": "minimum value",
        "count_agg": "count of all companies with data",
    }
    op_label = symbol_map.get(op, op)

    return (
        f"What is the {op_label} of {label} "
        f"across all companies in the dataset {year_ctx}?"
    )


def rewrite_depth2(company, node_expr, operands, op):
    """Depth 2: composed expression — build from the expression string."""
    # For depth 2, we describe what's being computed more abstractly
    label_parts = []
    for operand in operands:
        lab, yr = col_label(operand)
        if yr:
            label_parts.append(f"{lab} (FY {yr})")
        else:
            label_parts.append(lab)

    op_label_map = {
        "divide": "divided by",
        "ratio":  "as a ratio to",
        "subtract": "minus",
        "greater_than": "greater than",
        "less_than": "less than",
    }
    op_word = op_label_map.get(op, op)

    if len(label_parts) >= 2:
        # Try to make it read well
        inner = node_expr.split(")")[0] + ")"  # the inner expression
        return (
            f"For {company}, what is the result of computing {node_expr}? "
            f"Specifically, evaluate the sub-expression first, then apply the outer operation."
        )
    return f"For {company}, evaluate: {node_expr}"


# ── Master rewrite function ─────────────────────────────────

def rewrite_as_question(result):
    """Convert an executed result into a formal natural language question."""
    company = result["company"]
    op = result["operation"]
    depth = result["depth"]
    operands = result["operands"]
    expr = result["expression"]

    if depth == 0:
        return rewrite_leaf(company, operands[0])

    if depth == 1:
        if op in ("sum_agg", "avg_agg", "max_agg", "min_agg", "count_agg"):
            return rewrite_aggregation(op, operands[0])
        if len(operands) >= 2:
            return rewrite_binary(company, op, operands[0], operands[1], result["result_type"])

    if depth == 2:
        return rewrite_depth2(company, expr, operands, op)

    return f"[Could not rewrite: {expr}]"


# ── Main ────────────────────────────────────────────────────

def main():
    with open(os.path.join(OUTPUT_DIR, "sampled_executed.json")) as f:
        results = json.load(f)

    dataset = []
    for i, r in enumerate(results):
        question = rewrite_as_question(r)
        if question.startswith("["):
            continue

        dataset.append({
            "id": f"FQ-{i+1:05d}",
            "company": r["company"],
            "depth": r["depth"],
            "operation": r["operation"],
            "expression": r["expression"],
            "question": question,
            "answer": r["answer"],
            "answer_formatted": format_value(r["answer"], r["result_type"]),
            "result_type": r["result_type"],
            "computation_trace": r["trace"],
        })

    # ── Save JSON ──
    with open(os.path.join(OUTPUT_DIR, "final_qa_dataset.json"), "w") as f:
        json.dump(dataset, f, indent=2, default=str, ensure_ascii=False)

    # ── Save CSV ──
    with open(os.path.join(OUTPUT_DIR, "final_qa_dataset.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "company", "depth", "operation", "expression",
                         "question", "answer", "answer_formatted", "result_type",
                         "computation_trace"])
        for d in dataset:
            writer.writerow([
                d["id"], d["company"], d["depth"], d["operation"], d["expression"],
                d["question"], d["answer"], d["answer_formatted"], d["result_type"],
                d["computation_trace"],
            ])

    # ── Stats ──
    print(f"{'='*65}")
    print(f"  FINAL QA DATASET — FORMAL QUESTIONS")
    print(f"{'='*65}")
    print(f"  Total QA pairs: {len(dataset)}\n")

    by_depth = {}
    for d in dataset:
        by_depth[d["depth"]] = by_depth.get(d["depth"], 0) + 1
    print(f"  By depth:")
    depth_labels = {0: "Lookup", 1: "Single Op", 2: "Composed"}
    for depth in sorted(by_depth):
        print(f"    Depth {depth} ({depth_labels.get(depth, '?'):<12}): {by_depth[depth]:>4}")

    by_op = {}
    for d in dataset:
        by_op[d["operation"] or "lookup"] = by_op.get(d["operation"] or "lookup", 0) + 1
    print(f"\n  By operation:")
    for op, count in sorted(by_op.items(), key=lambda x: -x[1]):
        print(f"    {op:<20}: {count:>4}")

    by_type = {}
    for d in dataset:
        by_type[d["result_type"]] = by_type.get(d["result_type"], 0) + 1
    print(f"\n  By answer type:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {t:<15}: {c:>4}")

    # ── Samples ──
    print(f"\n{'='*65}")
    print(f"  SAMPLE QA PAIRS")
    print(f"{'='*65}")

    for depth in [0, 1, 2]:
        subset = [d for d in dataset if d["depth"] == depth]
        if not subset:
            continue
        samples = random.sample(subset, min(3, len(subset)))
        for qa in samples:
            print(f"\n  [{qa['id']}] Depth {qa['depth']} | {qa['operation'] or 'lookup'}")
            print(f"  Expression: {qa['expression']}")
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer_formatted']}")
            print(f"  Trace: {qa['computation_trace']}")

    print(f"\n{'='*65}")
    print(f"✓ Saved {len(dataset)} QA pairs to {os.path.join(OUTPUT_DIR, 'final_qa_dataset.json')}")
    print(f"✓ Saved {len(dataset)} QA pairs to {os.path.join(OUTPUT_DIR, 'final_qa_dataset.csv')}")


if __name__ == "__main__":
    main()