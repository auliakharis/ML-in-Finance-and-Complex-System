"""
Unit tests for compiler_pipeline/2.fixed_building_atoms.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_PIPELINE = Path(__file__).resolve().parent.parent / "compiler_pipeline"


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _PIPELINE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


atoms_mod = _load("2.fixed_building_atoms.py", "fixed_building_atoms")

build_atoms       = atoms_mod.build_atoms
validate_dataframe = atoms_mod.validate_dataframe
normalize_value   = atoms_mod.normalize_value
BASE_CONCEPTS     = atoms_mod.BASE_CONCEPTS
CONCEPT_METADATA  = atoms_mod.CONCEPT_METADATA


def _minimal_df(n_companies: int = 2, years: list[int] | None = None) -> pd.DataFrame:
    if years is None:
        years = [2021, 2022, 2023]
    rows = []
    for c in range(n_companies):
        for y in years:
            row = {"company_name": f"Corp{c}", "year": y}
            for concept in BASE_CONCEPTS:
                row[concept] = float((c + 1) * 100 + y - 2020)
            row["income_tax"] = 0.21
            rows.append(row)
    return pd.DataFrame(rows)


class TestNormalizeValue:
    def test_valid_int(self):
        assert normalize_value(42) == 42.0

    def test_valid_float(self):
        assert normalize_value(3.14) == pytest.approx(3.14)

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="Missing value"):
            normalize_value(float("nan"))

    def test_pandas_nan_raises(self):
        with pytest.raises(ValueError, match="Missing value"):
            normalize_value(pd.NA)


class TestValidateDataframe:
    def test_valid_df_passes(self):
        validate_dataframe(_minimal_df())

    def test_missing_concept_column(self):
        df = _minimal_df().drop(columns=["revenue"])
        with pytest.raises(ValueError, match="revenue"):
            validate_dataframe(df)

    def test_missing_company_name(self):
        df = _minimal_df().drop(columns=["company_name"])
        with pytest.raises(ValueError, match="company_name"):
            validate_dataframe(df)

    def test_missing_year(self):
        df = _minimal_df().drop(columns=["year"])
        with pytest.raises(ValueError, match="year"):
            validate_dataframe(df)

    def test_multiple_missing_columns(self):
        df = _minimal_df().drop(columns=["revenue", "cash"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df)


class TestBuildAtoms:
    def test_atom_count(self):
        df = _minimal_df(n_companies=2, years=[2021, 2022])
        atoms = build_atoms(df)
        assert len(atoms) == 2 * 2 * 18

    def test_atom_fields_present(self):
        df = _minimal_df(n_companies=1, years=[2022])
        atoms = build_atoms(df)
        required = {"key", "concept", "semantic_type", "label", "entity",
                    "period", "unit", "value", "depth", "parent_concept",
                    "statement", "role"}
        for atom in atoms:
            assert required.issubset(atom.keys())

    def test_entity_and_period(self):
        df = _minimal_df(n_companies=1, years=[2022])
        atoms = build_atoms(df)
        assert {a["entity"] for a in atoms} == {"Corp0"}
        assert {a["period"] for a in atoms} == {"2022"}

    def test_concept_coverage(self):
        df = _minimal_df(n_companies=1, years=[2021])
        atoms = build_atoms(df)
        assert {a["concept"] for a in atoms} == set(BASE_CONCEPTS)

    def test_value_is_float(self):
        df = _minimal_df(n_companies=1, years=[2021])
        for atom in build_atoms(df):
            assert isinstance(atom["value"], float)

    def test_key_unique(self):
        df = _minimal_df(n_companies=3, years=[2020, 2021, 2022])
        atoms = build_atoms(df)
        keys = [a["key"] for a in atoms]
        assert len(keys) == len(set(keys))

    def test_nan_value_raises(self):
        df = _minimal_df(n_companies=1, years=[2021])
        df.loc[0, "revenue"] = float("nan")
        with pytest.raises(ValueError, match="Missing value"):
            build_atoms(df)

    def test_income_tax_is_rate_type(self):
        df = _minimal_df(n_companies=1, years=[2021])
        atoms = build_atoms(df)
        tax_atoms = [a for a in atoms if a["concept"] == "income_tax"]
        assert all(a["semantic_type"] == "rate" for a in tax_atoms)
        assert all(a["unit"] == "ratio" for a in tax_atoms)

    def test_component_atoms_have_parent_concept(self):
        df = _minimal_df(n_companies=1, years=[2021])
        atoms = build_atoms(df)
        for atom in atoms:
            if atom["role"] == "component":
                assert atom["parent_concept"] is not None, f"{atom['concept']} is component but parent_concept is None"
            else:
                assert atom["parent_concept"] is None, f"{atom['concept']} has role=None but parent_concept={atom['parent_concept']!r}"

    def test_statement_values(self):
        df = _minimal_df(n_companies=1, years=[2021])
        atoms = build_atoms(df)
        by_concept = {a["concept"]: a for a in atoms}
        assert by_concept["revenue"]["statement"] == "income_statement"
        assert by_concept["cash"]["statement"] == "balance_sheet"
        assert by_concept["capex"]["statement"] == "cash_flow_statement"
        assert by_concept["stock_price"]["statement"] == "market_data"
        assert by_concept["employees"]["statement"] == "company_profile"

    def test_all_base_concepts_have_metadata(self):
        assert all(c in CONCEPT_METADATA for c in BASE_CONCEPTS), \
            f"Missing metadata for: {[c for c in BASE_CONCEPTS if c not in CONCEPT_METADATA]}"
