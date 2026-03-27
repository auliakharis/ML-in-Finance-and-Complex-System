"""
operations_10q.py
==================
Operations adapted for 10-Q column structure.
Replaces their step2_operations.py.

Key difference from original:
  - Columns are named field_PREFIX_Date instead of field_year
  - "same metric different period" means same base_name, different period_tag
  - "same unit" still applies within M_USD group
"""

OPERATIONS = {
    "add": {
        "symbol": "+",
        "description": "Add two values",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "same_unit",
        "example": "revenues_Q_Oct_31_2025 + membership_and_other_income_Q_Oct_31_2025",
    },
    "subtract": {
        "symbol": "-",
        "description": "Subtract right from left",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "same_unit",
        "example": "total_assets_Oct_31_2025 - total_liabilities_Oct_31_2025",
    },
    "multiply": {
        "symbol": "×",
        "description": "Multiply two values",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "any_numeric",
        "example": "shares_basic_Q_Oct_31_2025 × eps_basic_Q_Oct_31_2025",
    },
    "divide": {
        "symbol": "÷",
        "description": "Divide left by right",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "any_numeric",
        "example": "net_income_Q_Oct_31_2025 ÷ revenues_Q_Oct_31_2025",
    },
    "greater_than": {
        "symbol": ">",
        "description": "Is left greater than right?",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "boolean",
        "constraint": "same_unit",
        "example": "revenues_Q_Oct_31_2025 > revenues_Q_Oct_31_2024",
    },
    "less_than": {
        "symbol": "<",
        "description": "Is left less than right?",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "boolean",
        "constraint": "same_unit",
        "example": "total_liabilities_Oct_31_2025 < total_assets_Oct_31_2025",
    },
    "greater_equal": {
        "symbol": "≥",
        "description": "Is left greater than or equal to right?",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "boolean",
        "constraint": "same_unit",
        "example": "current_assets_Oct_31_2025 ≥ current_liabilities_Oct_31_2025",
    },
    "less_equal": {
        "symbol": "≤",
        "description": "Is left less than or equal to right?",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "boolean",
        "constraint": "same_unit",
        "example": "long_term_debt_Oct_31_2025 ≤ total_equity_Oct_31_2025",
    },
    "equals": {
        "symbol": "=",
        "description": "Are two values equal?",
        "left_type": "any", "right_type": "any",
        "result_type": "boolean",
        "constraint": "same_metric_or_categorical",
        "example": "industry = 'retail'  OR  revenues_Q_Oct_31_2025 = revenues_Q_Oct_31_2024",
    },
    "not_equals": {
        "symbol": "≠",
        "description": "Are two values not equal?",
        "left_type": "any", "right_type": "any",
        "result_type": "boolean",
        "constraint": "same_metric_or_categorical",
        "example": "company_name ≠ 'Volt Inc'  OR  net_income_Q_Oct_31_2025 ≠ net_income_Q_Oct_31_2024",
    },
    "ratio": {
        "symbol": "ratio",
        "description": "Ratio of numerator to denominator",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "ratio",
        "constraint": "any_numeric",
        "example": "ratio(current_assets_Oct_31_2025, current_liabilities_Oct_31_2025)",
    },
    "change": {
        "symbol": "Δ",
        "description": "Absolute change from prior period to current",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "numeric",
        "constraint": "same_base_different_period",
        "example": "Δ(revenues, Q_Oct_31_2024 → Q_Oct_31_2025)",
    },
    "pct_change": {
        "symbol": "%Δ",
        "description": "Percentage change from prior period to current",
        "left_type": "numeric", "right_type": "numeric",
        "result_type": "percentage",
        "constraint": "same_base_different_period",
        "example": "%Δ(net_income, Q_Oct_31_2024 → Q_Oct_31_2025)",
    },
    "sum_agg": {
        "symbol": "SUM",
        "description": "Sum across all companies",
        "left_type": "numeric", "right_type": None,
        "result_type": "numeric",
        "constraint": "single_numeric_col",
        "example": "SUM(revenues_Q_Oct_31_2025)",
    },
    "avg_agg": {
        "symbol": "AVG",
        "description": "Average across all companies",
        "left_type": "numeric", "right_type": None,
        "result_type": "numeric",
        "constraint": "single_numeric_col",
        "example": "AVG(net_income_Q_Oct_31_2025)",
    },
    "max_agg": {
        "symbol": "MAX",
        "description": "Maximum value across all companies",
        "left_type": "numeric", "right_type": None,
        "result_type": "numeric",
        "constraint": "single_numeric_col",
        "example": "MAX(total_assets_Oct_31_2025)",
    },
    "min_agg": {
        "symbol": "MIN",
        "description": "Minimum value across all companies",
        "left_type": "numeric", "right_type": None,
        "result_type": "numeric",
        "constraint": "single_numeric_col",
        "example": "MIN(revenues_Q_Oct_31_2025)",
    },
    "count_agg": {
        "symbol": "COUNT",
        "description": "Count of companies",
        "left_type": "any", "right_type": None,
        "result_type": "count",
        "constraint": "single_col",
        "example": "COUNT(company_name)",
    },
}


def can_combine(op_name: str, col_a: dict, col_b: dict = None) -> tuple:
    """
    Check if an operation is valid for the given column(s).
    Returns (is_valid, reason).
    col_a and col_b are schema entries with keys: type, unit, base_name, period_tag, prefix, date
    """
    op = OPERATIONS[op_name]
    constraint = op["constraint"]

    # Unary aggregations
    if col_b is None:
        if constraint == "single_numeric_col":
            return col_a["type"] == "numeric", "needs numeric column"
        if constraint == "single_col":
            return True, "any column"
        return False, "unknown constraint"

    a_type = col_a["type"]
    b_type = col_b["type"]

    if constraint == "same_metric_or_categorical":
        if a_type != b_type:
            return False, "both must be same type"
        # Categorical columns (company_name, industry, etc.) — always valid
        if a_type == "categorical":
            return True, "categorical comparison"
        # Numeric columns — only allow same base metric (across different periods)
        if a_type == "numeric":
            same_base   = col_a.get("base_name") == col_b.get("base_name")
            diff_period = col_a.get("period_tag") != col_b.get("period_tag")
            same_prefix = col_a.get("prefix") == col_b.get("prefix")
            return same_base and diff_period and same_prefix,                    "numeric equals: same metric, same prefix, different period only"
        return True, "same type"

    if constraint == "same_unit":
        if a_type != "numeric" or b_type != "numeric":
            return False, "both must be numeric"
        same_unit   = col_a.get("unit") == col_b.get("unit")
        same_date   = col_a.get("date") == col_b.get("date")
        same_base   = col_a.get("base_name") == col_b.get("base_name")
        same_prefix = col_a.get("prefix") == col_b.get("prefix")
        # add/subtract: must be same statement type (prefix) AND same unit
        # cross-date allowed only if same base metric
        return same_unit and same_prefix and (same_date or same_base),                "same unit, same statement type, same date or same metric required"

    if constraint == "any_numeric":
        if a_type != "numeric" or b_type != "numeric":
            return False, "both must be numeric"
        same_date = col_a.get("date") == col_b.get("date")
        same_base = col_a.get("base_name") == col_b.get("base_name")
        # Allow: same date (any two metrics) OR same metric across dates
        return same_date or same_base,                "must share same date or same base metric"

    if constraint == "same_base_different_period":
        if a_type != "numeric" or b_type != "numeric":
            return False, "both must be numeric"
        same_base   = col_a.get("base_name") == col_b.get("base_name")
        diff_period = col_a.get("period_tag") != col_b.get("period_tag")
        # Also require same prefix type (Q vs Q, YTD vs YTD, BS vs BS)
        same_prefix = col_a.get("prefix") == col_b.get("prefix")
        return same_base and diff_period and same_prefix, \
               "same metric, same prefix type, different period"

    return False, "unknown constraint"