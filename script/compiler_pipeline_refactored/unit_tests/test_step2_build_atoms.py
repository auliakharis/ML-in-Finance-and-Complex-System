"""
Unit tests for the v2 atom-building layer, adapted from the v1 step-2 suite.
"""
from __future__ import annotations

import pytest

from script.compiler_pipeline_refactored.make_random_questions import (
    build_atoms_from_dataframe,
    normalize_value,
    resolve_entity_column,
)
from script.compiler_pipeline_refactored.tree import SemanticType


class TestNormalizeValue:
    def test_valid_int(self):
        assert normalize_value(42) == 42.0

    def test_valid_float(self):
        assert normalize_value(3.14) == pytest.approx(3.14)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Missing value"):
            normalize_value(None)

    def test_blank_string_raises(self):
        with pytest.raises(ValueError, match="Missing value"):
            normalize_value("   ")


class TestResolveEntityColumn:
    def test_prefers_company_name(self):
        fieldnames = ["ticker", "company_name", "year"]
        assert resolve_entity_column(fieldnames) == "company_name"

    def test_accepts_ticker(self):
        fieldnames = ["ticker", "year", "revenue"]
        assert resolve_entity_column(fieldnames) == "ticker"

    def test_raises_without_supported_column(self):
        with pytest.raises(ValueError, match="company_name, ticker, entity"):
            resolve_entity_column(["year", "revenue"])


class TestBuildAtoms:
    def test_atom_count(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        atoms = build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
        assert len(atoms) == len(minimal_csv_rows) * len(concept_metadata)

    def test_atom_fields_present(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        atoms = build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
        required = {
            "key",
            "concept",
            "semantic_type",
            "label",
            "entity",
            "period",
            "unit",
            "value",
            "depth",
            "parent_concept",
            "role",
        }
        for atom in atoms.values():
            assert required.issubset(atom.model_dump().keys())

    def test_entity_and_period(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        atoms = build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
        assert {atom.entity for atom in atoms.values()} == {"Corp0", "Corp1"}
        assert {atom.period for atom in atoms.values()} == {"2021", "2022", "2023"}

    def test_concept_coverage(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        atoms = build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
        assert {atom.concept for atom in atoms.values()} == set(concept_metadata)

    def test_value_is_float(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        atoms = build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
        assert all(isinstance(atom.value, float) for atom in atoms.values())

    def test_key_unique(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        atoms = build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
        assert len(atoms) == len(set(atoms))

    def test_missing_concept_column(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        fieldnames = [name for name in minimal_fieldnames if name != "revenue"]
        rows = [{k: v for k, v in row.items() if k != "revenue"} for row in minimal_csv_rows]
        with pytest.raises(ValueError, match="revenue"):
            build_atoms_from_dataframe(fieldnames, rows, concept_metadata)

    def test_missing_year(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        fieldnames = [name for name in minimal_fieldnames if name != "year"]
        rows = [{k: v for k, v in row.items() if k != "year"} for row in minimal_csv_rows]
        with pytest.raises(ValueError, match="year"):
            build_atoms_from_dataframe(fieldnames, rows, concept_metadata)

    def test_missing_multiple_columns(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        fieldnames = [name for name in minimal_fieldnames if name not in {"revenue", "cash"}]
        rows = [
            {k: v for k, v in row.items() if k not in {"revenue", "cash"}}
            for row in minimal_csv_rows
        ]
        with pytest.raises(ValueError, match="required columns"):
            build_atoms_from_dataframe(fieldnames, rows, concept_metadata)

    def test_blank_value_raises(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        rows = [dict(row) for row in minimal_csv_rows]
        rows[0]["revenue"] = ""
        with pytest.raises(ValueError, match="Missing value"):
            build_atoms_from_dataframe(minimal_fieldnames, rows, concept_metadata)

    def test_income_tax_is_rate_type(self, minimal_fieldnames, minimal_csv_rows, concept_metadata):
        atoms = build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
        tax_atoms = [atom for atom in atoms.values() if atom.concept == "income_tax"]
        assert all(atom.semantic_type == SemanticType.rate for atom in tax_atoms)
        assert all(atom.unit == "ratio" for atom in tax_atoms)

    def test_component_atoms_have_parent_concept(
        self,
        minimal_fieldnames,
        minimal_csv_rows,
        concept_metadata,
    ):
        atoms = build_atoms_from_dataframe(minimal_fieldnames, minimal_csv_rows, concept_metadata)
        for atom in atoms.values():
            if atom.role == "component":
                assert atom.parent_concept is not None
            else:
                assert atom.parent_concept is None
