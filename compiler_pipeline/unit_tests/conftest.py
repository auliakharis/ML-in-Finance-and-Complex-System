from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


V2_DIR = Path(__file__).resolve().parent.parent
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))


@pytest.fixture
def v2_dir() -> Path:
    return V2_DIR


@pytest.fixture
def sample_company():
    from data_prep import Company

    return Company(
        name="Acme Holdings",
        ticker="ACME",
        sector="Technology",
        country="USA",
        exchange="NYSE",
    )


@pytest.fixture
def categorical_columns_model(v2_dir: Path):
    from data_prep import CategorialColumns

    payload = json.loads((v2_dir / "config/categorical_cols.json").read_text(encoding="utf-8"))
    return CategorialColumns.model_validate(payload)


@pytest.fixture
def yearly_numeric_columns_model(v2_dir: Path):
    from data_prep import YearlyNumericColumns

    payload = json.loads((v2_dir / "config/yearly_numeric_cols.json").read_text(encoding="utf-8"))
    payload["define"].pop("year", None)
    return YearlyNumericColumns.model_validate(payload)


@pytest.fixture
def concept_metadata(v2_dir: Path) -> dict[str, dict]:
    payload = json.loads((v2_dir / "config/concept_metadata.json").read_text(encoding="utf-8"))
    return payload["define"]


@pytest.fixture
def minimal_csv_rows(concept_metadata: dict[str, dict]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    concepts = list(concept_metadata.keys())

    for company_index in range(2):
        for year in (2021, 2022, 2023):
            row: dict[str, object] = {
                "company_name": f"Corp{company_index}",
                "ticker": f"C{company_index}",
                "year": str(year),
            }
            for offset, concept in enumerate(concepts, start=1):
                base_value = float((company_index + 1) * 1000 + (year - 2020) * 100 + offset)
                if concept == "income_tax":
                    row[concept] = 0.21
                elif concept == "shares_outstanding":
                    row[concept] = 100.0 + company_index + offset
                elif concept == "stock_price":
                    row[concept] = 50.0 + company_index + offset
                elif concept == "employees":
                    row[concept] = 5000.0 + company_index * 100 + offset
                else:
                    row[concept] = base_value
            rows.append(row)

    return rows


@pytest.fixture
def minimal_fieldnames(minimal_csv_rows: list[dict[str, object]]) -> list[str]:
    return list(minimal_csv_rows[0].keys())


@pytest.fixture
def minimal_atoms(
    minimal_fieldnames: list[str],
    minimal_csv_rows: list[dict[str, object]],
    concept_metadata: dict[str, dict],
):
    from make_random_questions import build_atoms_from_dataframe

    return build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
