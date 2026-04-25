"""
Unit tests for the sec10Q compiler pipeline:
  pipeline_sec10Q/build_atoms_sec10Q.py   — column parser + atom builder
  pipeline_sec10Q/operations_sec10Q.py   — can_combine validity rules
  pipeline_sec10Q/rewrite_sec10Q.py      — parse_col, format_value, rewrite_as_question
  pipeline_sec10Q/generate_csv.py     — parse_column, build_schema

Run with:
  cd /cluster/home/lturgut/ML-in-Finance-and-Complex-System
  .venv/bin/pytest tests/test_sec10Q_pipeline.py -v
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------
_PIPELINE = Path(__file__).resolve().parent.parent / "pipeline_sec10Q"


def _load(filename: str, name: str, base: Path | None = None, extra_sys_modules: dict | None = None):
    """Load a module by filename, optionally pre-populating sys.modules stubs."""
    if extra_sys_modules:
        for mod_name, mod_obj in extra_sys_modules.items():
            sys.modules[mod_name] = mod_obj
    path = (base or _PIPELINE) / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


atoms_sec10Q = _load("build_atoms_sec10Q.py", "build_atoms_sec10Q")
ops_sec10Q   = _load("operations_sec10Q.py",  "operations_sec10Q")
rewrite      = _load("rewrite_sec10Q.py",      "rewrite_sec10Q")

# generate_csv.py does `from generate import ...` at the top level.
# We inject a stub so it loads without the full report generator.
_generate_stub = types.ModuleType("generate")
_generate_stub.generate_report = MagicMock()
_generate_stub.report_to_dict  = MagicMock()
gen_csv = _load("generate_csv.py", "generate_csv", extra_sys_modules={"generate": _generate_stub})

# ---------------------------------------------------------------------------
# Aliases
# ---------------------------------------------------------------------------
_parse_column_sec10Q = atoms_sec10Q._parse_column
_period_string    = atoms_sec10Q._period_string
_concept_name     = atoms_sec10Q._concept_name
build_atoms_sec10Q   = atoms_sec10Q.build_atoms

can_combine       = ops_sec10Q.can_combine
OPERATIONS        = ops_sec10Q.OPERATIONS

parse_col         = rewrite.parse_col
format_value      = rewrite.format_value
rewrite_as_question = rewrite.rewrite_as_question

parse_column      = gen_csv.parse_column
build_schema      = gen_csv.build_schema


# ===========================================================================
# 1. build_atoms_sec10Q — column parsing helpers
# ===========================================================================

class TestParseColumnSec10q:
    """_parse_column: returns (field, prefix_type, date_readable) or None."""

    def test_q_column(self):
        result = _parse_column_sec10Q("net_sales_Q_Oct_31_2025")
        assert result is not None
        field, prefix_type, date_readable = result
        assert field == "net_sales"
        assert prefix_type == "q"
        assert "Oct" in date_readable

    def test_ytd_column(self):
        result = _parse_column_sec10Q("cf_operating_YTD_Oct_31_2025")
        assert result is not None
        field, prefix_type, _ = result
        assert field == "cf_operating"
        assert prefix_type == "ytd"

    def test_balance_sheet_column(self):
        result = _parse_column_sec10Q("total_assets_Oct_31_2025")
        assert result is not None
        field, prefix_type, date_readable = result
        assert field == "total_assets"
        assert prefix_type == "bs"
        assert "Oct" in date_readable

    def test_unknown_column_returns_none(self):
        assert _parse_column_sec10Q("company_name") is None
        assert _parse_column_sec10Q("industry") is None
        assert _parse_column_sec10Q("random_garbage") is None

    def test_q_date_spaces_replaced(self):
        _, _, date_readable = _parse_column_sec10Q("revenues_Q_Jan_01_2024")
        # Underscores in date should be replaced with spaces
        assert "_" not in date_readable

    def test_ytd_date_spaces_replaced(self):
        _, _, date_readable = _parse_column_sec10Q("net_income_YTD_Jun_30_2024")
        assert "_" not in date_readable

    def test_compound_field_name(self):
        result = _parse_column_sec10Q("membership_and_other_income_Q_Oct_31_2025")
        assert result is not None
        field, prefix_type, _ = result
        assert field == "membership_and_other_income"
        assert prefix_type == "q"


class TestPeriodString:
    def test_q_prefix(self):
        p = _period_string("q", "Oct 31 2025")
        assert p.startswith("Q ")

    def test_ytd_prefix(self):
        p = _period_string("ytd", "Oct 31 2025")
        assert p.startswith("YTD ")

    def test_bs_prefix_plain_date(self):
        p = _period_string("bs", "Oct 31 2025")
        assert p == "Oct 31 2025"  # no prefix added


class TestConceptName:
    def test_q_suffix(self):
        assert _concept_name("net_sales", "q") == "net_sales_q"

    def test_ytd_suffix(self):
        assert _concept_name("cf_operating", "ytd") == "cf_operating_ytd"

    def test_bs_no_suffix(self):
        assert _concept_name("total_assets", "bs") == "total_assets"

    def test_q_and_ytd_are_distinct(self):
        assert _concept_name("net_income", "q") != _concept_name("net_income", "ytd")


# ===========================================================================
# 2. build_atoms_sec10Q — main builder
# ===========================================================================

def _make_sec10Q_df(companies: list[str] | None = None) -> pd.DataFrame:
    """Return a minimal 10-Q style DataFrame with one Q and one BS date."""
    if companies is None:
        companies = ["Volt Inc", "Apex Technologies"]

    rows = []
    for company in companies:
        row = {
            "company_name": company,
            "form_type":    "10-Q",
            "industry":     "retail",
            "fiscal_year":  "2025",
            "quarter":      "3",
            "period":       "Three Months Ended October 31 2025",
            "fiscal_year_start": "January 01 2025",
            # Q income statement columns
            "net_sales_Q_Oct_31_2025":      100_000.0,
            "revenues_Q_Oct_31_2025":       101_000.0,
            "cost_of_sales_Q_Oct_31_2025":   70_000.0,
            "net_income_Q_Oct_31_2025":       5_000.0,
            # YTD cash flow columns
            "cf_operating_YTD_Oct_31_2025":  15_000.0,
            "cf_capex_YTD_Oct_31_2025":      -3_000.0,
            # Balance sheet columns (no prefix)
            "total_assets_Oct_31_2025":     500_000.0,
            "total_liabilities_Oct_31_2025":300_000.0,
            "total_equity_Oct_31_2025":     200_000.0,
            "cash_Oct_31_2025":              50_000.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


class TestBuildAtomsSec10q:
    def setup_method(self):
        self.df = _make_sec10Q_df()
        self.atoms = build_atoms_sec10Q(self.df)

    def test_returns_list(self):
        assert isinstance(self.atoms, list)

    def test_non_empty(self):
        assert len(self.atoms) > 0

    def test_skip_columns_excluded(self):
        concepts = {a["concept"] for a in self.atoms}
        # SKIP_COLUMNS like company_name, industry, form_type should not appear
        assert "company_name" not in concepts
        assert "industry" not in concepts
        assert "fiscal_year" not in concepts

    def test_q_columns_get_q_suffix(self):
        q_atoms = [a for a in self.atoms if a["concept"].endswith("_q")]
        assert len(q_atoms) > 0
        for a in q_atoms:
            assert a["period"].startswith("Q ")

    def test_ytd_columns_get_ytd_suffix(self):
        ytd_atoms = [a for a in self.atoms if a["concept"].endswith("_ytd")]
        assert len(ytd_atoms) > 0
        for a in ytd_atoms:
            assert a["period"].startswith("YTD ")

    def test_bs_columns_have_no_suffix(self):
        bs_atoms = [a for a in self.atoms
                    if a["concept"] in {"total_assets", "total_liabilities",
                                        "total_equity", "cash"}]
        assert len(bs_atoms) > 0
        for a in bs_atoms:
            # Period should be a plain date (no Q/YTD prefix)
            assert not a["period"].startswith("Q ")
            assert not a["period"].startswith("YTD ")

    def test_q_and_ytd_are_distinct_concepts(self):
        # net_income_q and cf_operating_ytd must be distinct concepts
        concepts = {a["concept"] for a in self.atoms}
        assert "net_sales_q" in concepts
        assert "cf_operating_ytd" in concepts
        # They should never collide
        assert "net_sales_ytd" not in concepts or "net_sales_q" in concepts

    def test_required_atom_fields(self):
        required = {"key", "concept", "semantic_type", "label", "entity",
                    "period", "unit", "value", "depth", "parent_concept", "role"}
        for atom in self.atoms:
            assert required.issubset(atom.keys()), f"Atom missing fields: {atom}"

    def test_entity_matches_company_name(self):
        entities = {a["entity"] for a in self.atoms}
        assert "Volt Inc" in entities
        assert "Apex Technologies" in entities

    def test_value_is_float(self):
        for atom in self.atoms:
            assert isinstance(atom["value"], float)

    def test_nan_values_skipped(self):
        df = _make_sec10Q_df(["SingleCo"])
        df.loc[0, "net_sales_Q_Oct_31_2025"] = float("nan")
        atoms = build_atoms_sec10Q(df)
        nan_atoms = [a for a in atoms if a["concept"] == "net_sales_q"
                     and a["entity"] == "SingleCo"]
        assert len(nan_atoms) == 0

    def test_unknown_field_skipped(self):
        df = _make_sec10Q_df(["OneCo"])
        df["mystery_field_Q_Oct_31_2025"] = 999.0
        atoms = build_atoms_sec10Q(df)
        concepts = {a["concept"] for a in atoms}
        assert "mystery_field_q" not in concepts

    def test_key_is_unique(self):
        keys = [a["key"] for a in self.atoms]
        assert len(keys) == len(set(keys))

    def test_semantic_type_amount(self):
        revenue_atoms = [a for a in self.atoms if a["concept"] == "revenues_q"]
        assert all(a["semantic_type"] == "amount" for a in revenue_atoms)

    def test_unit_is_usd(self):
        revenue_atoms = [a for a in self.atoms if a["concept"] == "revenues_q"]
        assert all(a["unit"] == "USD" for a in revenue_atoms)

    def test_depth_is_zero(self):
        for atom in self.atoms:
            assert atom["depth"] == 0

    def test_label_is_human_readable(self):
        rev_atoms = [a for a in self.atoms if a["concept"] == "revenues_q"]
        assert len(rev_atoms) > 0
        for a in rev_atoms:
            # Should be human readable label, not raw field name
            assert a["label"] == "total revenues"

    def test_two_companies_correct_total(self):
        df = _make_sec10Q_df(["CompA", "CompB"])
        atoms = build_atoms_sec10Q(df)
        rev_atoms = [a for a in atoms if a["concept"] == "revenues_q"]
        assert len(rev_atoms) == 2
        entities = {a["entity"] for a in rev_atoms}
        assert entities == {"CompA", "CompB"}


# ===========================================================================
# 3. operations_10q — can_combine
# ===========================================================================

def _col(type_="numeric", unit="M_USD", base_name="revenues",
         period_tag="Q_Oct_31_2025", prefix="Q", date="Oct_31_2025") -> dict:
    return {
        "type":       type_,
        "unit":       unit,
        "base_name":  base_name,
        "period_tag": period_tag,
        "prefix":     prefix,
        "date":       date,
    }


class TestCanCombineUnary:
    def test_sum_agg_requires_numeric(self):
        valid, _ = can_combine("sum_agg", _col(type_="numeric"))
        assert valid is True

    def test_sum_agg_rejects_categorical(self):
        valid, _ = can_combine("sum_agg", _col(type_="categorical"))
        assert valid is False

    def test_avg_agg_requires_numeric(self):
        valid, _ = can_combine("avg_agg", _col(type_="numeric"))
        assert valid is True

    def test_count_agg_allows_any(self):
        valid, _ = can_combine("count_agg", _col(type_="categorical"))
        assert valid is True
        valid2, _ = can_combine("count_agg", _col(type_="numeric"))
        assert valid2 is True

    def test_min_agg_requires_numeric(self):
        valid, _ = can_combine("min_agg", _col(type_="numeric"))
        assert valid is True


class TestCanCombineSameUnit:
    """add / subtract / greater_than / less_than / greater_equal / less_equal"""

    def _same_date_col(self, base="revenues", unit="M_USD"):
        return _col(base_name=base, unit=unit,
                    period_tag="Q_Oct_31_2025", prefix="Q", date="Oct_31_2025")

    def test_add_same_date_same_unit(self):
        a = self._same_date_col("revenues")
        b = self._same_date_col("cost_of_sales")
        valid, _ = can_combine("add", a, b)
        assert valid is True

    def test_add_different_unit_rejected(self):
        a = self._same_date_col("revenues", unit="M_USD")
        b = self._same_date_col("eps_basic", unit="USD")
        valid, _ = can_combine("add", a, b)
        assert valid is False

    def test_subtract_same_context(self):
        a = self._same_date_col("total_assets")
        b = self._same_date_col("total_liabilities")
        valid, _ = can_combine("subtract", a, b)
        assert valid is True

    def test_add_rejects_non_numeric(self):
        a = _col(type_="categorical")
        b = _col(type_="numeric")
        valid, _ = can_combine("add", a, b)
        assert valid is False

    def test_greater_than_same_context(self):
        a = self._same_date_col("revenues")
        b = self._same_date_col("cost_of_sales")
        valid, _ = can_combine("greater_than", a, b)
        assert valid is True

    def test_add_cross_prefix_rejected(self):
        # Q vs YTD — different prefix, not same_base
        a = _col(base_name="net_income", prefix="Q",   period_tag="Q_Oct_31_2025",   date="Oct_31_2025")
        b = _col(base_name="cf_operating", prefix="YTD", period_tag="YTD_Oct_31_2025", date="Oct_31_2025")
        valid, _ = can_combine("add", a, b)
        assert valid is False


class TestCanCombineAnyNumeric:
    """multiply / divide / ratio"""

    def test_ratio_same_date_different_metrics(self):
        a = _col(base_name="net_income",   date="Oct_31_2025")
        b = _col(base_name="revenues",     date="Oct_31_2025")
        valid, _ = can_combine("ratio", a, b)
        assert valid is True

    def test_divide_same_metric_different_date(self):
        a = _col(base_name="revenues", date="Oct_31_2025", period_tag="Q_Oct_31_2025")
        b = _col(base_name="revenues", date="Oct_31_2024", period_tag="Q_Oct_31_2024")
        valid, _ = can_combine("divide", a, b)
        assert valid is True

    def test_ratio_rejects_categorical(self):
        a = _col(type_="categorical")
        b = _col(type_="numeric")
        valid, _ = can_combine("ratio", a, b)
        assert valid is False

    def test_ratio_different_date_different_metric_rejected(self):
        a = _col(base_name="revenues",  date="Oct_31_2025")
        b = _col(base_name="net_income", date="Oct_31_2024")
        valid, _ = can_combine("ratio", a, b)
        assert valid is False


class TestCanCombineSameBaseDifferentPeriod:
    """change / pct_change"""

    def test_change_same_metric_different_period(self):
        a = _col(base_name="revenues", period_tag="Q_Oct_31_2025", prefix="Q", date="Oct_31_2025")
        b = _col(base_name="revenues", period_tag="Q_Oct_31_2024", prefix="Q", date="Oct_31_2024")
        valid, _ = can_combine("change", a, b)
        assert valid is True

    def test_change_different_metric_rejected(self):
        a = _col(base_name="revenues",   period_tag="Q_Oct_31_2025", prefix="Q", date="Oct_31_2025")
        b = _col(base_name="net_income", period_tag="Q_Oct_31_2024", prefix="Q", date="Oct_31_2024")
        valid, _ = can_combine("change", a, b)
        assert valid is False

    def test_change_same_period_rejected(self):
        a = _col(base_name="revenues", period_tag="Q_Oct_31_2025", date="Oct_31_2025")
        b = _col(base_name="revenues", period_tag="Q_Oct_31_2025", date="Oct_31_2025")
        valid, _ = can_combine("change", a, b)
        assert valid is False

    def test_change_cross_prefix_rejected(self):
        # Q vs YTD same base — not meaningful
        a = _col(base_name="revenues", prefix="Q",   period_tag="Q_Oct_31_2025",   date="Oct_31_2025")
        b = _col(base_name="revenues", prefix="YTD", period_tag="YTD_Oct_31_2024", date="Oct_31_2024")
        valid, _ = can_combine("change", a, b)
        assert valid is False

    def test_pct_change_valid(self):
        a = _col(base_name="net_income", period_tag="Q_Oct_31_2025", prefix="Q", date="Oct_31_2025")
        b = _col(base_name="net_income", period_tag="Q_Oct_31_2024", prefix="Q", date="Oct_31_2024")
        valid, _ = can_combine("pct_change", a, b)
        assert valid is True


class TestCanCombineSameMetricOrCategorical:
    """equals / not_equals"""

    def test_categorical_always_valid(self):
        a = {"type": "categorical", "unit": None, "base_name": "industry",
             "period_tag": None, "prefix": None, "date": None}
        b = {"type": "categorical", "unit": None, "base_name": "industry",
             "period_tag": None, "prefix": None, "date": None}
        valid, _ = can_combine("equals", a, b)
        assert valid is True

    def test_numeric_same_base_different_period_valid(self):
        a = _col(base_name="revenues", period_tag="Q_Oct_31_2025", prefix="Q", date="Oct_31_2025")
        b = _col(base_name="revenues", period_tag="Q_Oct_31_2024", prefix="Q", date="Oct_31_2024")
        valid, _ = can_combine("equals", a, b)
        assert valid is True

    def test_numeric_same_period_rejected(self):
        a = _col(base_name="revenues", period_tag="Q_Oct_31_2025")
        b = _col(base_name="revenues", period_tag="Q_Oct_31_2025")
        valid, _ = can_combine("equals", a, b)
        assert valid is False

    def test_mixed_types_rejected(self):
        a = _col(type_="categorical")
        b = _col(type_="numeric")
        valid, _ = can_combine("not_equals", a, b)
        assert valid is False


# ===========================================================================
# 4. rewrite_10q — parse_col
# ===========================================================================

class TestParseCol:
    def test_q_column(self):
        label, period = parse_col("revenues_Q_Oct_31_2025")
        assert label == "total revenues"
        assert "quarter ended" in period.lower()
        assert "Oct" in period

    def test_ytd_column(self):
        label, period = parse_col("cf_operating_YTD_Oct_31_2025")
        assert label == "net cash from operating activities"
        assert "year-to-date" in period.lower()

    def test_bs_column(self):
        label, period = parse_col("total_assets_Oct_31_2025")
        assert label == "total assets"
        assert period.startswith("as of")

    def test_identity_column(self):
        label, period = parse_col("company_name")
        assert label == "company name"
        assert period is None

    def test_unknown_field_fallback(self):
        label, period = parse_col("some_field_Q_Jan_01_2024")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_known_field_uses_label_dict(self):
        label, _ = parse_col("net_income_Q_Oct_31_2025")
        assert label == "net income"

    def test_cf_capex_label(self):
        label, _ = parse_col("cf_capex_YTD_Oct_31_2025")
        assert label == "capital expenditures"


# ===========================================================================
# 5. rewrite_10q — format_value
# ===========================================================================

class TestFormatValue:
    def test_none_returns_na(self):
        assert format_value(None, "numeric") == "N/A"

    def test_boolean_true(self):
        assert format_value(True, "boolean") == "Yes"

    def test_boolean_false(self):
        assert format_value(False, "boolean") == "No"

    def test_boolean_result_type(self):
        assert format_value(1, "boolean") == "Yes"
        assert format_value(0, "boolean") == "No"

    def test_percentage(self):
        result = format_value(12.5, "percentage")
        assert "12.50%" in result

    def test_count(self):
        result = format_value(7.0, "count")
        assert result == "7"

    def test_ratio_four_decimals(self):
        result = format_value(0.12345678, "ratio")
        assert "." in result
        # Should have 4 decimal places
        decimal_part = result.split(".")[1]
        assert len(decimal_part) == 4

    def test_large_float_comma_formatted(self):
        result = format_value(1_234_567.0, "numeric")
        assert "," in result

    def test_small_float_two_decimals(self):
        result = format_value(12.5, "numeric")
        assert "." in result

    def test_string_passthrough(self):
        result = format_value("retail", "categorical")
        assert result == "retail"


# ===========================================================================
# 6. rewrite_10q — rewrite_as_question
# ===========================================================================

def _result(op, depth, operands, expr=None, company="Volt Inc", result_type="numeric"):
    return {
        "company":     company,
        "operation":   op,
        "depth":       depth,
        "operands":    operands,
        "expression":  expr or f"({operands[0]} {op} {operands[1] if len(operands) > 1 else ''})",
        "result_type": result_type,
    }


class TestRewriteAsQuestion:
    # --- depth 0: simple lookup ---
    def test_depth0_q_column(self):
        r = _result(None, 0, ["revenues_Q_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert isinstance(q, str)
        assert "total revenues" in q.lower()
        assert "Volt Inc" in q

    def test_depth0_bs_column(self):
        r = _result(None, 0, ["total_assets_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "total assets" in q.lower()

    def test_depth0_identity_column(self):
        r = _result(None, 0, ["company_name"])
        q = rewrite_as_question(r)
        assert isinstance(q, str)
        assert len(q) > 5

    # --- depth 1: aggregation ops ---
    def test_sum_agg(self):
        r = _result("sum_agg", 1, ["revenues_Q_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "total sum" in q.lower() or "sum" in q.lower()
        assert "all companies" in q.lower()

    def test_avg_agg(self):
        r = _result("avg_agg", 1, ["net_income_Q_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "average" in q.lower()

    def test_max_agg(self):
        r = _result("max_agg", 1, ["total_assets_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "maximum" in q.lower()

    def test_min_agg(self):
        r = _result("min_agg", 1, ["total_assets_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "minimum" in q.lower()

    def test_count_agg(self):
        r = _result("count_agg", 1, ["company_name"])
        q = rewrite_as_question(r)
        assert "count" in q.lower() or "companies" in q.lower()

    # --- depth 1: binary arithmetic ops ---
    def test_add(self):
        r = _result("add", 1,
                    ["revenues_Q_Oct_31_2025", "membership_and_other_income_Q_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "sum" in q.lower() or "add" in q.lower()
        assert "Volt Inc" in q

    def test_subtract(self):
        r = _result("subtract", 1,
                    ["total_assets_Oct_31_2025", "total_liabilities_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "difference" in q.lower() or "minus" in q.lower()

    def test_multiply(self):
        r = _result("multiply", 1,
                    ["shares_basic_Q_Oct_31_2025", "eps_basic_Q_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "product" in q.lower() or "multipl" in q.lower()

    def test_ratio(self):
        r = _result("ratio", 1,
                    ["net_income_Q_Oct_31_2025", "revenues_Q_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "ratio" in q.lower()

    def test_divide(self):
        r = _result("divide", 1,
                    ["net_income_Q_Oct_31_2025", "revenues_Q_Oct_31_2025"])
        q = rewrite_as_question(r)
        assert "ratio" in q.lower() or "divide" in q.lower()

    # --- depth 1: change / pct_change ---
    def test_change(self):
        r = _result("change", 1,
                    ["revenues_Q_Oct_31_2025", "revenues_Q_Oct_31_2024"])
        q = rewrite_as_question(r)
        assert "change" in q.lower()
        assert "Volt Inc" in q

    def test_pct_change(self):
        r = _result("pct_change", 1,
                    ["net_income_Q_Oct_31_2025", "net_income_Q_Oct_31_2024"],
                    result_type="percentage")
        q = rewrite_as_question(r)
        assert "percentage" in q.lower() or "change" in q.lower()

    # --- depth 1: boolean comparisons ---
    def test_greater_than(self):
        r = _result("greater_than", 1,
                    ["revenues_Q_Oct_31_2025", "cost_of_sales_Q_Oct_31_2025"],
                    result_type="boolean")
        q = rewrite_as_question(r)
        assert "greater than" in q.lower()
        assert q.endswith("?")

    def test_less_than(self):
        r = _result("less_than", 1,
                    ["total_liabilities_Oct_31_2025", "total_assets_Oct_31_2025"],
                    result_type="boolean")
        q = rewrite_as_question(r)
        assert "less than" in q.lower()

    def test_greater_equal(self):
        r = _result("greater_equal", 1,
                    ["current_assets_Oct_31_2025", "current_liabilities_Oct_31_2025"],
                    result_type="boolean")
        q = rewrite_as_question(r)
        assert "greater than or equal" in q.lower()

    def test_less_equal(self):
        r = _result("less_equal", 1,
                    ["long_term_debt_Oct_31_2025", "total_equity_Oct_31_2025"],
                    result_type="boolean")
        q = rewrite_as_question(r)
        assert "less than or equal" in q.lower()

    def test_equals(self):
        r = _result("equals", 1,
                    ["revenues_Q_Oct_31_2025", "revenues_Q_Oct_31_2024"],
                    result_type="boolean")
        q = rewrite_as_question(r)
        assert "equal" in q.lower()

    def test_not_equals(self):
        r = _result("not_equals", 1,
                    ["net_income_Q_Oct_31_2025", "net_income_Q_Oct_31_2024"],
                    result_type="boolean")
        q = rewrite_as_question(r)
        assert "different" in q.lower() or "not" in q.lower()

    # --- depth 2 ---
    def test_depth2_fallback(self):
        expr = "(ratio(net_income_Q_Oct_31_2025, revenues_Q_Oct_31_2025) > 0.1)"
        r = {
            "company": "Volt Inc",
            "operation": "greater_than",
            "depth": 2,
            "operands": [expr, "0.1"],
            "expression": expr,
            "result_type": "boolean",
        }
        q = rewrite_as_question(r)
        assert "Volt Inc" in q
        assert isinstance(q, str)

    # --- non-empty output ---
    def test_all_depth1_ops_produce_non_empty(self):
        binary_ops = ["add", "subtract", "multiply", "divide", "ratio",
                      "change", "pct_change", "greater_than", "less_than",
                      "greater_equal", "less_equal", "equals", "not_equals"]
        for op in binary_ops:
            r = _result(op, 1,
                        ["revenues_Q_Oct_31_2025", "revenues_Q_Oct_31_2024"])
            q = rewrite_as_question(r)
            assert isinstance(q, str) and len(q) > 5, f"Empty/short for op={op}: {q!r}"


# ===========================================================================
# 7. generate_csv — parse_column and build_schema
# ===========================================================================

class TestGenerateCsvParseColumn:
    def test_q_column(self):
        info = parse_column("revenues_Q_Oct_31_2025")
        assert info["type"] == "numeric"
        assert info["base_name"] == "revenues"
        assert info["prefix"] == "Q"
        assert info["date"] == "Oct_31_2025"
        assert info["period_tag"] == "Q_Oct_31_2025"

    def test_ytd_column(self):
        info = parse_column("cf_operating_YTD_Oct_31_2025")
        assert info["type"] == "numeric"
        assert info["base_name"] == "cf_operating"
        assert info["prefix"] == "YTD"
        assert info["period_tag"] == "YTD_Oct_31_2025"

    def test_bs_column(self):
        info = parse_column("total_assets_Oct_31_2025")
        assert info["type"] == "numeric"
        assert info["base_name"] == "total_assets"
        assert info["prefix"] == "BS"
        assert info["date"] == "Oct_31_2025"

    def test_identity_column_company_name(self):
        info = parse_column("company_name")
        assert info["type"] == "categorical"
        assert info["base_name"] == "company_name"
        assert info["period_tag"] is None
        assert info["prefix"] is None

    def test_identity_column_industry(self):
        info = parse_column("industry")
        assert info["type"] == "categorical"

    def test_unknown_column_fallback(self):
        info = parse_column("random_col")
        assert info["type"] == "categorical"
        assert info["period_tag"] is None

    def test_unit_preserved(self):
        info = parse_column("revenues_Q_Oct_31_2025")
        assert info["unit"] == "M_USD"

    def test_eps_unit_is_usd(self):
        info = parse_column("eps_basic_Q_Oct_31_2025")
        assert info["unit"] == "USD"

    def test_shares_unit_is_thousands(self):
        info = parse_column("shares_basic_Q_Oct_31_2025")
        assert info["unit"] == "thousands"

    def test_compound_field_ytd(self):
        info = parse_column("membership_and_other_income_YTD_Oct_31_2025")
        assert info["base_name"] == "membership_and_other_income"
        assert info["prefix"] == "YTD"


class TestBuildSchema:
    def test_returns_dict(self):
        cols = ["company_name", "revenues_Q_Oct_31_2025", "total_assets_Oct_31_2025"]
        schema = build_schema(cols)
        assert isinstance(schema, dict)

    def test_all_columns_present(self):
        cols = ["company_name", "revenues_Q_Oct_31_2025",
                "cf_operating_YTD_Oct_31_2025", "total_assets_Oct_31_2025"]
        schema = build_schema(cols)
        assert set(schema.keys()) == set(cols)

    def test_categorical_column_entry(self):
        schema = build_schema(["company_name"])
        assert schema["company_name"]["type"] == "categorical"

    def test_q_numeric_column_entry(self):
        schema = build_schema(["revenues_Q_Oct_31_2025"])
        entry = schema["revenues_Q_Oct_31_2025"]
        assert entry["type"] == "numeric"
        assert entry["prefix"] == "Q"

    def test_bs_numeric_column_entry(self):
        schema = build_schema(["total_assets_Oct_31_2025"])
        entry = schema["total_assets_Oct_31_2025"]
        assert entry["type"] == "numeric"
        assert entry["prefix"] == "BS"

    def test_empty_columns_empty_schema(self):
        schema = build_schema([])
        assert schema == {}
