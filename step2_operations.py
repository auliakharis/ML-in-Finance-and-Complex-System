"""
STEP 2: Define Operations
===========================
All the operations we can perform on spreadsheet data, with strict
rules about what types of data each operation accepts.

Key insight: not everything can be combined with everything.
  - You can't add "USA" + "Technology" 
  - You can't divide employees by a country
  - You CAN compare two monetary values from different years
  - You CAN compute ratios of compatible numeric types

Operations are grouped by what they do:
  ARITHMETIC:    +, -, ×, ÷
  COMPARISON:    =, ≠, <, >, ≤, ≥
  RATIO:         a/b (same as ÷ but semantically "what fraction")
  AGGREGATION:   sum, avg, max, min, count (across rows)
  TEMPORAL:      change, pct_change, growth (across years)

Output: operations.json
"""

import json


# ── Type Compatibility Matrix ───────────────────────────────
# Which operations are valid for which data type combinations?

OPERATIONS = {

    # ═══ ARITHMETIC (two numeric operands → numeric result) ═══

    "add": {
        "symbol": "+",
        "description": "Add two values",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "same_unit",  # can only add USD + USD, not USD + count
        "example": "revenue_2024 + net_income_2024",
    },
    "subtract": {
        "symbol": "-",
        "description": "Subtract right from left",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "same_unit",
        "example": "total_assets_2024 - total_liabilities_2024",
    },
    "multiply": {
        "symbol": "×",
        "description": "Multiply two values",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "any_numeric",  # shares × price = market cap
        "example": "shares_outstanding_2024 × stock_price_2024",
    },
    "divide": {
        "symbol": "÷",
        "description": "Divide left by right",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "any_numeric",  # revenue / employees = rev per employee
        "example": "net_income_2024 ÷ total_equity_2024",
    },

    # ═══ COMPARISON — NUMERIC (two numbers → boolean) ═══

    "greater_than": {
        "symbol": ">",
        "description": "Is left greater than right?",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "boolean",
        "constraint": "same_base_or_unit",  # compare revenue_2023 > revenue_2022
        "example": "revenue_2024 > revenue_2023",
    },
    "less_than": {
        "symbol": "<",
        "description": "Is left less than right?",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "boolean",
        "constraint": "same_base_or_unit",
        "example": "total_liabilities_2024 < total_assets_2024",
    },
    "greater_equal": {
        "symbol": "≥",
        "description": "Is left greater than or equal to right?",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "boolean",
        "constraint": "same_base_or_unit",
        "example": "current_assets_2024 ≥ current_liabilities_2024",
    },
    "less_equal": {
        "symbol": "≤",
        "description": "Is left less than or equal to right?",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "boolean",
        "constraint": "same_base_or_unit",
        "example": "long_term_debt_2024 ≤ total_equity_2024",
    },

    # ═══ COMPARISON — EQUALITY (any type → boolean) ═══

    "equals": {
        "symbol": "=",
        "description": "Are two values equal?",
        "left_type": "any",
        "right_type": "any",
        "result_type": "boolean",
        "constraint": "same_type",  # categorical = categorical, numeric = numeric
        "example": "sector = 'Technology'",
    },
    "not_equals": {
        "symbol": "≠",
        "description": "Are two values not equal?",
        "left_type": "any",
        "right_type": "any",
        "result_type": "boolean",
        "constraint": "same_type",
        "example": "country ≠ 'USA'",
    },

    # ═══ RATIO (a/b but semantically a proportion/fraction) ═══

    "ratio": {
        "symbol": "ratio",
        "description": "Ratio of numerator to denominator",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "ratio",
        "constraint": "any_numeric",
        "example": "ratio(current_assets_2024, current_liabilities_2024)",
    },

    # ═══ TEMPORAL (same metric, different years) ═══

    "change": {
        "symbol": "Δ",
        "description": "Absolute change from year1 to year2",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "same_base_different_year",
        "example": "Δ(revenue, 2023, 2024)",
    },
    "pct_change": {
        "symbol": "%Δ",
        "description": "Percentage change from year1 to year2",
        "left_type": "numeric",
        "right_type": "numeric",
        "result_type": "percentage",
        "constraint": "same_base_different_year",
        "example": "%Δ(revenue, 2023, 2024)",
    },

    # ═══ AGGREGATION (across companies) ═══

    "sum_agg": {
        "symbol": "SUM",
        "description": "Sum of a column across all (or filtered) rows",
        "left_type": "numeric",
        "right_type": None,
        "result_type": "numeric",
        "constraint": "single_numeric_col",
        "example": "SUM(revenue_2024)",
    },
    "avg_agg": {
        "symbol": "AVG",
        "description": "Average of a column across all (or filtered) rows",
        "left_type": "numeric",
        "right_type": None,
        "result_type": "numeric",
        "constraint": "single_numeric_col",
        "example": "AVG(net_income_2024)",
    },
    "max_agg": {
        "symbol": "MAX",
        "description": "Maximum value in a column",
        "left_type": "numeric",
        "right_type": None,
        "result_type": "numeric",
        "constraint": "single_numeric_col",
        "example": "MAX(total_assets_2024)",
    },
    "min_agg": {
        "symbol": "MIN",
        "description": "Minimum value in a column",
        "left_type": "numeric",
        "right_type": None,
        "result_type": "numeric",
        "constraint": "single_numeric_col",
        "example": "MIN(stock_price_2024)",
    },
    "count_agg": {
        "symbol": "COUNT",
        "description": "Count of rows matching a condition",
        "left_type": "any",
        "right_type": None,
        "result_type": "count",
        "constraint": "single_col",
        "example": "COUNT(sector = 'Technology')",
    },
}

# ── Unit Compatibility Rules ────────────────────────────────
# Which units can be combined with which operations?

UNIT_GROUPS = {
    "monetary": ["M_USD"],                  # can add/subtract within group
    "share":    ["millions"],               # share counts
    "price":    ["USD"],                    # per-share price
    "headcount":["count"],                  # employee counts
}

# For addition/subtraction: operands must be in the same unit group
# For multiplication/division: any numeric pair is fine (creates derived units)
# For comparison: must be same base metric or same unit group


def can_combine(op_name, col_a_info, col_b_info=None):
    """
    Check if an operation is valid for given column(s).
    Returns (is_valid, reason).
    """
    op = OPERATIONS[op_name]
    constraint = op["constraint"]

    # Unary operations (aggregations)
    if col_b_info is None:
        if constraint == "single_numeric_col":
            return col_a_info["type"] == "numeric", "needs numeric column"
        if constraint == "single_col":
            return True, "any column"
        return False, "unknown constraint"

    # Binary operations
    a_type = col_a_info["type"]
    b_type = col_b_info["type"]

    if constraint == "same_type":
        return a_type == b_type, f"both must be {a_type}"

    if constraint == "same_unit":
        if a_type != "numeric" or b_type != "numeric":
            return False, "both must be numeric"
        return col_a_info.get("unit") == col_b_info.get("unit"), "same unit required"

    if constraint == "any_numeric":
        return a_type == "numeric" and b_type == "numeric", "both must be numeric"

    if constraint == "same_base_or_unit":
        if a_type != "numeric" or b_type != "numeric":
            return False, "both must be numeric"
        # Same base metric (e.g. revenue_2023 vs revenue_2024) OR same unit
        same_base = col_a_info.get("base_name") == col_b_info.get("base_name")
        same_unit = col_a_info.get("unit") == col_b_info.get("unit")
        return same_base or same_unit, "same metric or unit"

    if constraint == "same_base_different_year":
        if a_type != "numeric" or b_type != "numeric":
            return False, "both must be numeric"
        same_base = col_a_info.get("base_name") == col_b_info.get("base_name")
        diff_year = col_a_info.get("year") != col_b_info.get("year")
        return same_base and diff_year, "same metric, different years"

    return False, "unknown constraint"


def main():
    with open("output/operations.json", "w") as f:
        json.dump(OPERATIONS, f, indent=2)

    print(f"✓ Defined {len(OPERATIONS)} operations\n")

    groups = {
        "Arithmetic":  ["add", "subtract", "multiply", "divide"],
        "Comparison":  ["greater_than", "less_than", "greater_equal", "less_equal", "equals", "not_equals"],
        "Ratio":       ["ratio"],
        "Temporal":    ["change", "pct_change"],
        "Aggregation": ["sum_agg", "avg_agg", "max_agg", "min_agg", "count_agg"],
    }

    for group_name, ops in groups.items():
        print(f"  {group_name}:")
        for op_name in ops:
            op = OPERATIONS[op_name]
            print(f"    {op['symbol']:>5}  {op['description']:<45} constraint: {op['constraint']}")
        print()

    print("  Type Compatibility Rules:")
    print("    + / -     : same unit only (USD+USD ok, USD+count no)")
    print("    × / ÷     : any two numeric columns")
    print("    > < ≥ ≤   : same base metric or same unit group")
    print("    = ≠       : same type (categorical=categorical, numeric=numeric)")
    print("    Δ / %Δ    : same metric, different years")
    print("    SUM/AVG.. : any single numeric column")
    print("    COUNT     : any column")

    print(f"\n✓ Saved to output/operations.json")


if __name__ == "__main__":
    main()
