"""
Unit tests for compiler_pipeline/5.make_random_questions.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

_PIPELINE = Path(__file__).resolve().parent.parent / "compiler_pipeline"


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _PIPELINE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


step5_mod = _load("5.make_random_questions.py", "make_random_questions")

validate_args = step5_mod.validate_args
flatten_leaves = step5_mod.flatten_leaves
resolve_path = step5_mod.resolve_path
build_row = step5_mod.build_row
clean_atom_payload = step5_mod.clean_atom_payload


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        n=10,
        depth_min=1,
        depth_max=3,
        derived_prob_min=0.0,
        derived_prob_max=0.5,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestValidateArgs:
    def test_valid_args_passes(self):
        validate_args(_make_args())

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="--n"):
            validate_args(_make_args(n=0))

    def test_n_negative_raises(self):
        with pytest.raises(ValueError, match="--n"):
            validate_args(_make_args(n=-1))

    def test_negative_depth_min_raises(self):
        with pytest.raises(ValueError, match="--depth-min"):
            validate_args(_make_args(depth_min=-1))

    def test_depth_min_gt_depth_max_raises(self):
        with pytest.raises(ValueError, match="--depth-min"):
            validate_args(_make_args(depth_min=5, depth_max=2))

    def test_derived_prob_out_of_range_raises(self):
        with pytest.raises(ValueError, match="--derived-prob"):
            validate_args(_make_args(derived_prob_min=-0.1))

    def test_derived_prob_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="--derived-prob"):
            validate_args(_make_args(derived_prob_min=0.8, derived_prob_max=0.2))

    def test_equal_depth_bounds_ok(self):
        validate_args(_make_args(depth_min=2, depth_max=2))

    def test_equal_prob_bounds_ok(self):
        validate_args(_make_args(derived_prob_min=0.3, derived_prob_max=0.3))


class TestFlattenLeaves:
    """flatten_leaves in step5 works on bound expression objects (duck-typed)."""

    class _Leaf:
        def __init__(self, key: str):
            self.key = key

    class _Node:
        def __init__(self, left, right):
            self.left = left
            self.right = right

    class _Derived:
        def __init__(self, inner):
            self.expr = inner

    class _Literal:
        def __init__(self, v: float):
            self.value = v

    def test_single_leaf(self):
        leaf = self._Leaf("k1")
        assert flatten_leaves(leaf) == ["k1"]

    def test_binary_node(self):
        node = self._Node(self._Leaf("A"), self._Leaf("B"))
        assert set(flatten_leaves(node)) == {"A", "B"}

    def test_nested_node(self):
        inner = self._Node(self._Leaf("A"), self._Leaf("B"))
        outer = self._Node(inner, self._Leaf("C"))
        assert set(flatten_leaves(outer)) == {"A", "B", "C"}

    def test_derived_expr_delegates(self):
        derived = self._Derived(self._Leaf("X"))
        assert flatten_leaves(derived) == ["X"]

    def test_literal_returns_empty(self):
        assert flatten_leaves(self._Literal(3.0)) == []


class TestResolvePath:
    def test_absolute_path_unchanged(self, tmp_path):
        abs_path = tmp_path / "file.json"
        result = resolve_path(str(abs_path), base_dir=Path("/some/other/dir"))
        assert result == abs_path

    def test_relative_path_joined_to_base(self, tmp_path):
        result = resolve_path("atoms.json", base_dir=tmp_path)
        assert result == tmp_path / "atoms.json"


class TestCleanAtomPayload:
    def test_keeps_allowed_fields(self):
        raw = {
            "key": "k1", "concept": "revenue", "semantic_type": "amount",
            "label": "Revenue", "entity": "Corp0", "period": "2021",
            "unit": "M_USD", "value": 1000.0, "depth": 0,
            "parent_concept": None, "role": "base",
            "aggregation_parent": "total_revenue",
            "statement": "income_statement", "section": "top",
        }
        cleaned = clean_atom_payload(raw)
        assert "key" in cleaned
        assert "concept" in cleaned
        assert "aggregation_parent" not in cleaned
        assert "statement" not in cleaned
        assert "section" not in cleaned

    def test_extra_fields_stripped(self):
        raw = {"key": "k", "concept": "cash", "extra_field": "DROPPED",
               "semantic_type": "amount", "label": "Cash", "entity": "E",
               "period": "2021", "unit": "M_USD", "value": 500.0,
               "depth": 0, "parent_concept": None, "role": "base"}
        cleaned = clean_atom_payload(raw)
        assert "extra_field" not in cleaned
