"""
Unit tests for compiler_pipeline/1.synthetic_data_seen_by_LLM.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PIPELINE = Path(__file__).resolve().parent.parent


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _PIPELINE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen_mod = _load("1.synthetic_data_seen_by_LLM.py", "synthetic_data")

generate_company_row = gen_mod.generate_company_row
build_schema         = gen_mod.build_schema
COMPANIES            = gen_mod.COMPANIES
YEARS                = gen_mod.YEARS
CATEGORICAL_COLS     = gen_mod.CATEGORICAL_COLS
YEARLY_NUMERIC_COLS  = gen_mod.YEARLY_NUMERIC_COLS


class TestGenerateCompanyRow:
    def _sample_company(self):
        return COMPANIES[0]

    def test_returns_one_row_per_year(self):
        rows = generate_company_row(self._sample_company())
        assert len(rows) == len(YEARS)

    def test_each_row_has_expected_columns(self):
        rows = generate_company_row(self._sample_company())
        expected = set(CATEGORICAL_COLS) | set(YEARLY_NUMERIC_COLS)
        for row in rows:
            assert expected.issubset(row.keys())

    def test_year_values_match_years(self):
        rows = generate_company_row(self._sample_company())
        assert [r["year"] for r in rows] == YEARS

    def test_company_name_constant_across_years(self):
        rows = generate_company_row(self._sample_company())
        names = {r["company_name"] for r in rows}
        assert len(names) == 1

    def test_balance_sheet_balances(self):
        rows = generate_company_row(self._sample_company())
        for row in rows:
            assert row["total_assets"] == pytest.approx(
                row["total_liabilities"] + row["total_equity"], abs=1
            )

    def test_revenue_is_positive(self):
        rows = generate_company_row(self._sample_company())
        for row in rows:
            assert row["revenue"] > 0

    def test_cogs_less_than_revenue(self):
        rows = generate_company_row(self._sample_company())
        for row in rows:
            assert row["cost_of_goods_sold"] < row["revenue"]

    def test_income_tax_is_rate(self):
        rows = generate_company_row(self._sample_company())
        for row in rows:
            assert 0.0 < row["income_tax"] < 1.0

    def test_dividends_non_negative(self):
        rows = generate_company_row(self._sample_company())
        for row in rows:
            assert row["dividends_paid"] >= 0

    def test_all_companies_produce_rows(self):
        all_rows = []
        for company in COMPANIES:
            rows = generate_company_row(company)
            all_rows.extend(rows)
        assert len(all_rows) == len(COMPANIES) * len(YEARS)


class TestBuildSchema:
    def _columns(self):
        rows = generate_company_row(COMPANIES[0])
        return list(rows[0].keys())

    def test_all_columns_covered(self):
        cols = self._columns()
        schema = build_schema(cols)
        for col in cols:
            assert col in schema

    def test_categorical_type(self):
        cols = self._columns()
        schema = build_schema(cols)
        assert schema["company_name"]["type"] == "categorical"
        assert schema["sector"]["type"] == "categorical"

    def test_year_column_type(self):
        cols = self._columns()
        schema = build_schema(cols)
        assert schema["year"]["type"] == "numeric"
        assert schema["year"]["unit"] == "year"

    def test_numeric_columns_have_unit(self):
        cols = self._columns()
        schema = build_schema(cols)
        assert "unit" in schema["revenue"]
        assert schema["revenue"]["unit"] == "M_USD"

    def test_numeric_columns_have_row_level_year(self):
        cols = self._columns()
        schema = build_schema(cols)
        assert schema["revenue"]["year"] == "row_level"

    def test_income_tax_is_numeric(self):
        cols = self._columns()
        schema = build_schema(cols)
        assert schema["income_tax"]["type"] == "numeric"

    def test_income_tax_unit_is_ratio(self):
        cols = self._columns()
        schema = build_schema(cols)
        assert schema["income_tax"]["unit"] == "ratio"
