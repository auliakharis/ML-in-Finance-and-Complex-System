"""
rewrite_10q.py
===============
Converts executed expressions into natural language questions,
adapted for 10-Q column naming (field_PREFIX_Date).

Replaces their step5_rewrite_questions.py.
"""

import re

# ---------------------------------------------------------------------------
# Human-readable labels for field base names
# ---------------------------------------------------------------------------

LABELS = {
    "revenues":                    "total revenues",
    "membership_and_other_income": "membership and other income",
    "cost_of_sales":               "cost of sales",
    "rd_expense":                  "research and development expense",
    "sga_expense":                 "selling general and administrative expense",
    "total_costs":                 "total costs and expenses",
    "gross_profit":                "gross profit",
    "operating_income":            "operating income",
    "other_income":                "other income/(expense), net",
    "pretax_income":               "income before taxes",
    "tax":                         "provision for income taxes",
    "net_income":                  "net income",
    "eps_basic":                   "basic earnings per share",
    "eps_diluted":                 "diluted earnings per share",
    "shares_basic":                "basic shares used in computing EPS",
    "shares_diluted":              "diluted shares used in computing EPS",
    "ci_net_income":               "net income",
    "ci_fx_translation":           "change in foreign currency translation",
    "ci_unrealized_gains":         "change in unrealized gains/losses on securities",
    "ci_total_oci":                "total other comprehensive income/(loss)",
    "comprehensive_income":        "total comprehensive income",
    "cash":                        "cash and cash equivalents",
    "st_investments":              "short-term investments",
    "accounts_receivable":         "accounts receivable, net",
    "inventories":                 "inventories",
    "prepaid":                     "prepaid expenses and other",
    "current_assets":              "total current assets",
    "lt_investments":              "long-term marketable securities",
    "ppe":                         "property, plant and equipment, net",
    "goodwill":                    "goodwill",
    "other_nca":                   "other non-current assets",
    "total_assets":                "total assets",
    "accounts_payable":            "accounts payable",
    "deferred_revenue":            "deferred revenue",
    "accrued_expenses":            "accrued expenses and other",
    "current_debt":                "current portion of long-term debt",
    "current_liabilities":         "total current liabilities",
    "long_term_debt":              "long-term debt",
    "other_ncl":                   "other non-current liabilities",
    "total_liabilities":           "total liabilities",
    "common_stock_apic":           "common stock and additional paid-in capital",
    "retained_earnings":           "retained earnings",
    "aoci":                        "accumulated other comprehensive income/(loss)",
    "total_equity":                "total shareholders equity",
    "cf_net_income":               "net income",
    "cf_da":                       "depreciation and amortization",
    "cf_sbc":                      "stock-based compensation",
    "cf_operating":                "net cash from operating activities",
    "cf_capex":                    "capital expenditures",
    "cf_investing":                "net cash from investing activities",
    "cf_repurchases":              "share repurchases",
    "cf_dividends":                "dividends paid",
    "cf_financing":                "net cash from financing activities",
    "cf_net_change":               "net change in cash",
    "cf_closing_cash":             "cash at end of period",
    "company_name":                "company name",
    "industry":                    "industry",
    "fiscal_year":                 "fiscal year",
    "quarter":                     "quarter",
}

PREFIX_LABELS = {
    "Q":   "for the quarter ended",
    "YTD": "for the year-to-date period ended",
    "BS":  "as of",
}


def parse_col(col: str) -> tuple:
    """
    Parse column name into (field_label, period_description).
    Returns (human_label, period_desc).
    """
    # Q or YTD columns: field_Q_Mon_DD_YYYY or field_YTD_Mon_DD_YYYY
    m = re.match(r'^(.+?)_(Q|YTD)_([A-Z][a-z]{2}_\d{2}_\d{4})$', col)
    if m:
        field, prefix, date = m.group(1), m.group(2), m.group(3)
        label = LABELS.get(field, field.replace("_", " "))
        date_str = date.replace("_", " ")
        period = f"{PREFIX_LABELS[prefix]} {date_str}"
        return label, period

    # Balance sheet columns: field_Mon_DD_YYYY
    m2 = re.match(r'^(.+?)_([A-Z][a-z]{2}_\d{2}_\d{4})$', col)
    if m2:
        field, date = m2.group(1), m2.group(2)
        label = LABELS.get(field, field.replace("_", " "))
        date_str = date.replace("_", " ")
        period = f"as of {date_str}"
        return label, period

    # Identity columns
    label = LABELS.get(col, col.replace("_", " "))
    return label, None


def format_value(val, result_type: str) -> str:
    if val is None:
        return "N/A"
    if result_type == "boolean":
        return "Yes" if val else "No"
    if result_type == "percentage":
        return f"{val:.2f}%"
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


def rewrite_as_question(result: dict) -> str:
    """Convert an executed result into a natural language question."""
    company  = result["company"]
    op       = result["operation"]
    depth    = result["depth"]
    operands = result["operands"]
    expr     = result["expression"]

    # Depth 0: simple lookup
    if depth == 0:
        col = operands[0]
        label, period = parse_col(col)
        if period:
            return f"What was the {label} reported by {company} {period}?"
        return f"What is the {label} of {company}?"

    # Aggregations
    if op in ("sum_agg", "avg_agg", "max_agg", "min_agg", "count_agg"):
        col = operands[0]
        label, period = parse_col(col)
        op_map = {
            "sum_agg":   "total sum",
            "avg_agg":   "average",
            "max_agg":   "maximum",
            "min_agg":   "minimum",
            "count_agg": "count of companies with data for",
        }
        period_str = f" {period}" if period else ""
        return (f"What is the {op_map[op]} of {label}"
                f"{period_str} across all companies in the dataset?")

    # Depth 1 binary
    if depth == 1 and len(operands) >= 2:
        col_a, col_b = operands[0], operands[1]
        label_a, period_a = parse_col(col_a)
        label_b, period_b = parse_col(col_b)

        # Build period context — include both dates if they differ
        same_period = (period_a == period_b)
        if period_a and period_b and not same_period:
            # Different periods — name each operand with its own date
            label_a_full = f"{label_a} ({period_a})"
            label_b_full = f"{label_b} ({period_b})"
            period_ctx   = ""  # already embedded in labels
        else:
            label_a_full = label_a
            label_b_full = label_b
            period_ctx   = f" {period_a or period_b}" if (period_a or period_b) else ""

        if op == "add":
            return (f"What is the sum of {label_a_full} and {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op == "subtract":
            return (f"What is the difference between {label_a_full} and {label_b_full} "
                    f"(i.e., {label_a_full} minus {label_b_full}) for {company}{period_ctx}?")
        if op == "multiply":
            return (f"What is the product of {label_a_full} and {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op in ("divide", "ratio"):
            return (f"What is the ratio of {label_a_full} to {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op == "greater_than":
            return (f"Is {label_a_full} strictly greater than {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op == "less_than":
            return (f"Is {label_a_full} strictly less than {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op == "greater_equal":
            return (f"Is {label_a_full} greater than or equal to {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op == "less_equal":
            return (f"Is {label_a_full} less than or equal to {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op == "equals":
            return (f"Is the {label_a_full} equal to {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op == "not_equals":
            return (f"Is the {label_a_full} different from {label_b_full} "
                    f"for {company}{period_ctx}?")
        if op == "change":
            return (f"What is the absolute change in {label_a} for {company} "
                    f"from {period_b} to {period_a}?")
        if op == "pct_change":
            return (f"What is the percentage change in {label_a} for {company} "
                    f"from {period_b} to {period_a}?")

    # Depth 2
    if depth == 2:
        return (f"For {company}, what is the result of the following computation: "
                f"{expr}? Evaluate any sub-expressions first, then apply the outer operation.")

    return f"[Could not rewrite: {expr}]"