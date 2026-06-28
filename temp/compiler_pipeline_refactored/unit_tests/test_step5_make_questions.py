"""
Unit tests for the v2 question-generation entrypoint, adapted from the v1 step-5 suite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from temp.compiler_pipeline_refactored.make_random_questions import build_row, flatten_leaf_keys, resolve_path, validate_args
from temp.compiler_pipeline_refactored.tree import DerivedExpr, Leaf, Literal, Node


def make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "n": 10,
        "depth_min": 1,
        "depth_max": 3,
        "derived_prob_min": 0.0,
        "derived_prob_max": 0.5,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestValidateArgs:
    def test_valid_args_passes(self):
        validate_args(make_args())

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="--n"):
            validate_args(make_args(n=0))

    def test_n_negative_raises(self):
        with pytest.raises(ValueError, match="--n"):
            validate_args(make_args(n=-1))

    def test_negative_depth_min_raises(self):
        with pytest.raises(ValueError, match="--depth-min"):
            validate_args(make_args(depth_min=-1))

    def test_depth_min_gt_depth_max_raises(self):
        with pytest.raises(ValueError, match="--depth-min"):
            validate_args(make_args(depth_min=5, depth_max=2))

    def test_derived_prob_out_of_range_raises(self):
        with pytest.raises(ValueError, match="--derived-prob"):
            validate_args(make_args(derived_prob_min=-0.1))

    def test_derived_prob_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="--derived-prob"):
            validate_args(make_args(derived_prob_min=0.8, derived_prob_max=0.2))

    def test_equal_depth_bounds_ok(self):
        validate_args(make_args(depth_min=2, depth_max=2))

    def test_equal_prob_bounds_ok(self):
        validate_args(make_args(derived_prob_min=0.3, derived_prob_max=0.3))


class TestFlattenLeafKeys:
    def test_single_leaf(self):
        assert flatten_leaf_keys(Leaf("k1")) == ["k1"]

    def test_binary_node(self):
        node = Node("sum", Leaf("A"), Leaf("B"))
        assert set(flatten_leaf_keys(node)) == {"A", "B"}

    def test_nested_node(self):
        node = Node("sum", Node("sum", Leaf("A"), Leaf("B")), Leaf("C"))
        assert set(flatten_leaf_keys(node)) == {"A", "B", "C"}

    def test_derived_expr_delegates(self):
        derived = DerivedExpr("gross_profit", Leaf("X"))
        assert flatten_leaf_keys(derived) == ["X"]

    def test_literal_returns_empty(self):
        assert flatten_leaf_keys(Literal(3.0)) == []


class TestResolvePath:
    def test_absolute_path_unchanged(self, tmp_path: Path):
        abs_path = tmp_path / "file.json"
        assert resolve_path(str(abs_path), base_dir=Path("/some/other/dir")) == abs_path

    def test_relative_path_joined_to_base(self, tmp_path: Path):
        assert resolve_path("atoms.json", base_dir=tmp_path) == tmp_path / "atoms.json"


class TestBuildRow:
    def test_build_row_includes_core_fields(self, minimal_atoms):
        expr = Node("sum", Leaf("0_revenue"), Leaf("0_cost_of_goods_sold"))
        row = build_row(
            i=0,
            expr=expr,
            question="What is revenue plus cost of goods sold?",
            answer=42.0,
            expr_json=expr.expr_to_json(),
            expr_str=expr.show_expr(),
            tree_payload=expr,
            tree_seed=11,
            bind_seed=22,
            master_seed=33,
            derived_prob=0.25,
            depth=2,
            template_stats={"internal_nodes": 1, "leaves": 2},
            leaf_keys=["0_revenue", "0_cost_of_goods_sold"],
            atoms=minimal_atoms,
        )

        assert row["question_id"] == 1
        assert row["question"] == "What is revenue plus cost of goods sold?"
        assert row["expression"] == "sum(0_revenue, 0_cost_of_goods_sold)"
        assert row["answer"] == 42.0
        assert json.loads(row["expression_json"])["op"] == "sum"
        assert json.loads(row["template_expression"])["op"] == "sum"

    def test_build_row_populates_leaf_columns(self, minimal_atoms):
        expr = Node("sum", Leaf("0_revenue"), Leaf("0_cost_of_goods_sold"))
        row = build_row(
            i=4,
            expr=expr,
            question="q",
            answer=1.0,
            expr_json=expr.expr_to_json(),
            expr_str=expr.show_expr(),
            tree_payload=expr,
            tree_seed=1,
            bind_seed=2,
            master_seed=3,
            derived_prob=0.1,
            depth=1,
            template_stats={"internal_nodes": 1, "leaves": 2},
            leaf_keys=["0_revenue", "0_cost_of_goods_sold"],
            atoms=minimal_atoms,
        )

        assert row["leaf_1_key"] == "0_revenue"
        assert row["leaf_1_concept"] == "revenue"
        assert row["leaf_2_key"] == "0_cost_of_goods_sold"
        assert row["leaf_2_concept"] == "cost_of_goods_sold"
