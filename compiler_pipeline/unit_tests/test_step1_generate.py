"""
Unit tests for the v2 data-prep layer, adapted from the v1 step-1 suite.
"""
from __future__ import annotations

from data_prep import (
    YEARS,
    build_schema_for_column,
    generate_company_row_for_given_year,
)


def rows_for_company(sample_company):
    base_revenue = 1_000.0
    rows = []
    for year in YEARS:
        row, base_revenue = generate_company_row_for_given_year(
            company_tuple=sample_company,
            credit_rating="A",
            year=year,
            base_revenue=base_revenue,
            income_tax_range=(0.10, 0.30),
            shares_outstanding_range=(100.0, 1_000.0),
            stock_price_range=(15.0, 120.0),
            employees_range=(5_000.0, 50_000.0),
        )
        rows.append(row)
    return rows


class TestGenerateCompanyRow:
    def test_returns_one_row_per_year(self, sample_company):
        rows = rows_for_company(sample_company)
        assert len(rows) == len(YEARS)

    def test_each_row_has_expected_columns(self, sample_company):
        rows = rows_for_company(sample_company)
        expected = {
            "company_name",
            "ticker",
            "sector",
            "country",
            "exchange",
            "credit_rating",
            "year",
            "revenue",
            "cost_of_goods_sold",
            "operating_expenses",
            "non_operating_expenses",
            "income_tax",
            "total_assets",
            "total_liabilities",
            "total_equity",
            "cash",
            "accounts_receivable",
            "inventories",
            "short_term_investments",
            "current_liabilities",
            "capex",
            "dividends_paid",
            "shares_outstanding",
            "stock_price",
            "employees",
        }
        for row in rows:
            assert expected.issubset(row.model_dump().keys())

    def test_year_values_match_years(self, sample_company):
        rows = rows_for_company(sample_company)
        assert [row.year for row in rows] == YEARS

    def test_company_name_constant_across_years(self, sample_company):
        rows = rows_for_company(sample_company)
        assert {row.company_name for row in rows} == {sample_company.name}

    def test_balance_sheet_balances(self, sample_company):
        rows = rows_for_company(sample_company)
        for row in rows:
            assert row.total_assets == row.total_liabilities + row.total_equity

    def test_revenue_is_positive(self, sample_company):
        rows = rows_for_company(sample_company)
        assert all(row.revenue > 0 for row in rows)

    def test_cogs_less_than_revenue(self, sample_company):
        rows = rows_for_company(sample_company)
        assert all(row.cost_of_goods_sold < row.revenue for row in rows)

    def test_income_tax_is_rate(self, sample_company):
        rows = rows_for_company(sample_company)
        assert all(0.0 < row.income_tax < 1.0 for row in rows)

    def test_dividends_non_negative(self, sample_company):
        rows = rows_for_company(sample_company)
        assert all(row.dividends_paid >= 0 for row in rows)


class TestBuildSchema:
    def schema(self, sample_company, categorical_columns_model, yearly_numeric_columns_model):
        one_row = generate_company_row_for_given_year(
            company_tuple=sample_company,
            credit_rating="A",
            year=YEARS[0],
            base_revenue=1_000.0,
            income_tax_range=(0.10, 0.30),
            shares_outstanding_range=(100.0, 1_000.0),
            stock_price_range=(15.0, 120.0),
            employees_range=(5_000.0, 50_000.0),
        )[0]
        columns = list(one_row.model_dump().keys())
        return {
            col: build_schema_for_column(
                col,
                categorical_columns_model,
                yearly_numeric_columns_model,
            )
            for col in columns
        }

    def test_all_columns_covered(
        self,
        sample_company,
        categorical_columns_model,
        yearly_numeric_columns_model,
    ):
        schema = self.schema(
            sample_company,
            categorical_columns_model,
            yearly_numeric_columns_model,
        )
        expected_columns = list(
            generate_company_row_for_given_year(
                company_tuple=sample_company,
                credit_rating="A",
                year=YEARS[0],
                base_revenue=1_000.0,
                income_tax_range=(0.10, 0.30),
                shares_outstanding_range=(100.0, 1_000.0),
                stock_price_range=(15.0, 120.0),
                employees_range=(5_000.0, 50_000.0),
            )[0].model_dump().keys()
        )
        for col in expected_columns:
            assert col in schema

    def test_categorical_type(
        self,
        sample_company,
        categorical_columns_model,
        yearly_numeric_columns_model,
    ):
        schema = self.schema(
            sample_company,
            categorical_columns_model,
            yearly_numeric_columns_model,
        )
        assert schema["company_name"].col_type == "categorical"
        assert schema["sector"].col_type == "categorical"

    def test_year_column_type(
        self,
        sample_company,
        categorical_columns_model,
        yearly_numeric_columns_model,
    ):
        schema = self.schema(
            sample_company,
            categorical_columns_model,
            yearly_numeric_columns_model,
        )
        assert schema["year"].col_type == "numeric"
        assert schema["year"].unit == "year"

    def test_numeric_columns_have_unit(
        self,
        sample_company,
        categorical_columns_model,
        yearly_numeric_columns_model,
    ):
        schema = self.schema(
            sample_company,
            categorical_columns_model,
            yearly_numeric_columns_model,
        )
        assert schema["revenue"].unit == "M_USD"

    def test_numeric_columns_have_row_level_year(
        self,
        sample_company,
        categorical_columns_model,
        yearly_numeric_columns_model,
    ):
        schema = self.schema(
            sample_company,
            categorical_columns_model,
            yearly_numeric_columns_model,
        )
        assert schema["revenue"].year == "row_level"

    def test_income_tax_is_numeric(
        self,
        sample_company,
        categorical_columns_model,
        yearly_numeric_columns_model,
    ):
        schema = self.schema(
            sample_company,
            categorical_columns_model,
            yearly_numeric_columns_model,
        )
        assert schema["income_tax"].col_type == "numeric"

    def test_income_tax_unit_is_ratio(
        self,
        sample_company,
        categorical_columns_model,
        yearly_numeric_columns_model,
    ):
        schema = self.schema(
            sample_company,
            categorical_columns_model,
            yearly_numeric_columns_model,
        )
        assert schema["income_tax"].unit == "ratio"
