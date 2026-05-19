"""
Data Preparation & Atomization
Used to be files 1 and 2
"""

import pandas as pd
import json
from pydantic import BaseModel
import random

random.seed(42)

YEARS = list(range(2020, 2026))
CATEGORICAL_COLS_FILE = "config/categorical_cols.json"
YEARLY_NUMERIC_COLS_FILE = "config/yearly_numeric_cols.json"
COMPANIES_FILE = "config/companies.json"

# BaseModel from Pydantic is used for:
# Data saving to and from json
# Validation of input data structures
class AtomMetadata(BaseModel):
    key: str
    concept: str
    semantic_type: str
    label: str
    entity: str
    period: str
    unit: str
    value: float
    depth: int
    parent_concept: str | None = None
    statement: str
    role: str | None = None


class ConceptMetadata(BaseModel):
    semantic_type: str
    parent_concept: str | None = None
    unit: str
    statement: str
    role: str | None = None


class OutputColumnSchema(BaseModel):
    col_type: str
    base_name: str
    desc: str
    year: str | None = None
    unit: str | None = None
    order: list[str] | None = None


class CategoricalColumn(BaseModel):
    col_type: str
    desc: str
    order: list[str] | None = None


class CSVRow(BaseModel):
    company_name: str
    ticker: str
    sector: str
    country: str
    exchange: str
    credit_rating: str
    year: int
    revenue: int
    cost_of_goods_sold: int
    operating_expenses: int
    non_operating_expenses: int
    income_tax: float
    total_assets: int
    total_liabilities: int
    total_equity: int
    cash: int
    accounts_receivable: int
    inventories: int
    short_term_investments: int
    current_liabilities: int
    capex: int
    dividends_paid: int
    shares_outstanding: int
    stock_price: float
    employees: int


class YearlyNumericColumn(BaseModel):
    unit: str
    range: dict[str, float] | None = None


class Company(BaseModel):
    name: str
    ticker: str
    sector: str
    country: str
    exchange: str


class YearlyNumericColumns(BaseModel):
    define: dict[str, YearlyNumericColumn]


class CategorialColumns(BaseModel):
    define: dict[str, CategoricalColumn]


class Companies(BaseModel):
    define: list[Company]


class OutputSchema(BaseModel):
    columns: dict[str, OutputColumnSchema]


class ConceptMetadataSchema(BaseModel):
    define: dict[str, ConceptMetadata]

def normalize_range(range: dict[str, float] | None) -> tuple[float, float]:
    if range is None:
        return 100, 1_000_000_000
    if range["min"] > range["max"]:
        raise ValueError(f"Invalid range with min > max: {range}")
    return range["min"], range["max"]

def generate_company_row_for_given_year(
    company_tuple: Company,
    credit_rating: str,
    year: int,
    base_revenue: float,
    income_tax_range: tuple[float, float],
    shares_outstanding_range: tuple[float, float],
    stock_price_range: tuple[float, float],
    employees_range: tuple[float, float],
) -> tuple[CSVRow, float]:
    growth = random.uniform(-0.08, 0.20)
    revenue = round(base_revenue * (1 + growth))

    # Derive income-statement drivers from revenue.
    cogs = round(revenue * random.uniform(0.30, 0.75))
    opex = round(revenue * random.uniform(0.05, 0.25))
    nonopex = round(revenue * random.uniform(0.05, 0.25))
    income_tax = random.uniform(*income_tax_range)

    # Derive balance-sheet items tied to revenue scale.
    # Asset turnover (revenue / assets) derives total assets from revenue.
    asset_turnover = random.uniform(0.3, 1.5)
    total_assets = round(revenue / asset_turnover)
    # Debt-to-equity ratio splits assets into equity and liabilities; enforces assets = liabilities + equity.
    de_ratio = random.uniform(0.3, 3.0)
    total_equity = round(total_assets / (1 + de_ratio))
    total_liabilities = total_assets - total_equity

    # Cash: 3-25% of total assets (liquidity buffer).
    cash = round(total_assets * random.uniform(0.03, 0.25))
    # Accounts receivable: 4-14% of revenue (~15-51 days sales outstanding).
    ar = round(revenue * random.uniform(0.04, 0.14))
    # Inventories: 2-15% of COGS (days inventory outstanding anchor).
    inv = round(cogs * random.uniform(0.02, 0.15))
    # Short-term investments: 1-15% of total assets (excess cash parked in securities).
    short_term_investments = round(total_assets * random.uniform(0.01, 0.15))
    # Current liabilities: 25-50% of total liabilities (remainder is long-term).
    cl = round(total_liabilities * random.uniform(0.25, 0.50))

    # Derive cash-flow and market/profile fields.
    net_income = round((revenue - cogs - opex - nonopex) * (1 - income_tax))
    dividends = round(max(0, net_income * random.uniform(0.0, 0.40)))
    capex = round(revenue * random.uniform(0.02, 0.10))

    shares = round(random.uniform(*shares_outstanding_range))
    price = round(random.uniform(*stock_price_range), 2)
    employees = round(random.uniform(*employees_range))

    row = CSVRow(
        company_name=company_tuple.name,
        ticker=company_tuple.ticker,
        sector=company_tuple.sector,
        country=company_tuple.country,
        exchange=company_tuple.exchange,
        credit_rating=credit_rating,
        year=year,
        revenue=revenue,
        cost_of_goods_sold=cogs,
        operating_expenses=opex,
        non_operating_expenses=nonopex,
        income_tax=income_tax,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        total_equity=total_equity,
        cash=cash,
        accounts_receivable=ar,
        inventories=inv,
        short_term_investments=short_term_investments,
        current_liabilities=cl,
        capex=capex,
        dividends_paid=dividends,
        shares_outstanding=shares,
        stock_price=price,
        employees=employees,
    )
    return row, revenue


def generate_company_row(
    company_tuple: Company,
    categorical_cols: CategorialColumns,
    yearly_numeric_cols: YearlyNumericColumns,
) -> list[CSVRow]:
    """Generate one row of the spreadsheet for one company."""
    credit_rating = random.choice(categorical_cols.define["credit_rating"].order)

    # Generate base revenue, then derive everything else consistently
    range = yearly_numeric_cols.define["revenue"].range

    base_revenue = random.uniform(*normalize_range(range))

    # Collect one record per year for this company.
    rows = [] 
    for year in YEARS:
        csv_row, base_revenue = generate_company_row_for_given_year(
            company_tuple=company_tuple,
            credit_rating=credit_rating,
            year=year,
            base_revenue=base_revenue,
            income_tax_range=normalize_range(yearly_numeric_cols.define["income_tax"].range),
            shares_outstanding_range=normalize_range(yearly_numeric_cols.define["shares_outstanding"].range),
            stock_price_range=normalize_range(yearly_numeric_cols.define["stock_price"].range),
            employees_range=normalize_range(yearly_numeric_cols.define["employees"].range),
        )
        rows.append(csv_row)

    return rows


def build_schema_for_column(
    col: str,
    categorical_cols: CategorialColumns,
    yearly_numeric_cols: YearlyNumericColumns,
) -> OutputColumnSchema:
    match col:
        case _ if col in categorical_cols.define:
            info = categorical_cols.define[col]
            return OutputColumnSchema(
                col_type=info.col_type,
                base_name=col,
                desc=info.desc,
                year=None,
                order=info.order if info.col_type == "ordinal" else None,
            )
        case "year":
            return OutputColumnSchema(
                col_type="numeric",
                base_name=col,
                desc="reporting year",
                year=None,
                unit="year",
            )
        case _ if col in yearly_numeric_cols.define:
            info = yearly_numeric_cols.define[col]
            return OutputColumnSchema(
                col_type="numeric",
                base_name=col,
                desc=col.replace("_", " "),
                year="row_level",
                unit=info.unit if info.unit else "M_USD",
            )
        case _:
            raise ValueError(
                f"Column '{col}' not found in either categorical or yearly numeric definitions."
            )


def build_schema(
    columns: list[str],
    categorical_cols: CategorialColumns,
    yearly_numeric_cols: YearlyNumericColumns,
) -> OutputSchema:
    """Build a metadata dict for each CSV column and write it to schema.json.

    Each column gets a small record describing its type, unit, and how the year
    dimension is encoded. Three kinds of columns are handled:

      Categorical/ordinal  -> type, desc, year=null (ordinals also get an order list)
      "year" column        -> type=numeric, unit="year", year=null
      Numeric concepts     -> type=numeric, unit from YEARLY_NUMERIC_COLS, year="row_level"

    year=null means the column is not time-indexed.
    year="row_level" means the year lives in the separate "year" column, not in the column name.
    """
    schema: dict[str, OutputColumnSchema] = {}
    categorical_cols = categorical_cols
    yearly_numeric_cols = yearly_numeric_cols

    schema = {
        col: build_schema_for_column(col, categorical_cols, yearly_numeric_cols)
        for col in columns
    }
    schema = OutputSchema(columns=schema)

    return schema


def load_concept_metadata_from_json() -> ConceptMetadataSchema:
    with open("config/concept_metadata.json", "r", encoding="utf-8") as f:
        concept_metadata = ConceptMetadataSchema.model_validate(json.load(f))
    return concept_metadata


def load_companies_and_columns_from_jsons() -> (
    tuple[Companies, CategorialColumns, YearlyNumericColumns]
):
    with open(COMPANIES_FILE, "r", encoding="utf-8") as f:
        companies = Companies.model_validate(json.load(f))
    with open(CATEGORICAL_COLS_FILE, "r", encoding="utf-8") as f:
        categorical_cols = CategorialColumns.model_validate(json.load(f))
    with open(YEARLY_NUMERIC_COLS_FILE, "r", encoding="utf-8") as f:
        yearly_numeric_cols = YearlyNumericColumns.model_validate(json.load(f))
    return companies, categorical_cols, yearly_numeric_cols


def generate_atoms(
    concept_metadata: ConceptMetadataSchema, rows: list[CSVRow]
) -> list[AtomMetadata]:
    return [
        AtomMetadata(
            key=f"{idx}_{concept_name}",
            concept=concept_name,
            semantic_type=concept_info.semantic_type,
            label=concept_info.statement,
            entity=row.ticker,
            period=str(row.year),
            unit=concept_info.unit,
            value=getattr(row, concept_name),
            depth=0,
            parent_concept=concept_info.parent_concept,
            statement=concept_info.statement,
            role=concept_info.role,
        )
        for idx, row in enumerate(rows)
        for concept_name, concept_info in concept_metadata.define.items()
    ]


def generate_atoms_and_write_to_json(
    concept_metadata: ConceptMetadataSchema, rows: list[CSVRow], output_path: str
) -> None:
    atoms = generate_atoms(concept_metadata, rows)
    json_data = json.dumps([atom.model_dump() for atom in atoms], indent=2)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_data)


def generate_csv(csv_path: str) -> tuple[list[str], list[CSVRow]]:
    # Generate all company-year rows.
    companies, categorical_cols, yearly_numeric_cols = (
        load_companies_and_columns_from_jsons()
    )
    row = [
        generate_company_row(company, categorical_cols, yearly_numeric_cols)
        for company in companies.define
    ]
    row: list[CSVRow] = [inner_row for sublist in row for inner_row in sublist]

    if not row:
        raise ValueError("No rows were generated from company templates.")
    
    rows_dict = [row.model_dump() for row in row]
    headers = list(rows_dict[0].keys()) if rows_dict else []
    rows_dict = [{k: inner_row[k] for k in headers} for inner_row in rows_dict]

    df = pd.DataFrame(rows_dict)
    df.to_csv(csv_path, index=False)
    return headers, row


def generate_json_schema(schema_path: str, columns: list[str], categorical_cols: CategorialColumns, yearly_numeric_cols: YearlyNumericColumns) -> OutputSchema:
    # Write and print schema/preview artifacts for inspection.
    schema: OutputSchema = build_schema(columns, categorical_cols, yearly_numeric_cols)
    schema = schema.model_dump()["columns"]
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    return schema


def main():
    csv_path = "output/synthetic_company_data.csv"
    schema_path = "output/schema.json"
    atoms_path = "output/atoms.json"
    
    columns, rows = generate_csv(csv_path)
    concept_metadata_schema = load_concept_metadata_from_json()
    _, categorical_cols, yearly_numeric_cols = load_companies_and_columns_from_jsons()
    generate_json_schema(schema_path, columns, categorical_cols, yearly_numeric_cols)

    generate_atoms_and_write_to_json(
        concept_metadata=concept_metadata_schema,
        rows=rows,
        output_path=atoms_path,
    )


if __name__ == "__main__":
    main()
