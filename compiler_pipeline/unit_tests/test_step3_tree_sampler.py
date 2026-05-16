"""
Unit tests for the v2 tree utilities, adapted from the v1 step-3 suite.
"""
from __future__ import annotations

import json
import random

import pytest

from tree import (
    DERIVED_CONCEPTS,
    DerivedExpr,
    Expr,
    Leaf,
    Literal,
    Node,
    Operation,
    SemanticType,
    TimeAgg,
)


class TestMakeLeafAndNode:
    def test_make_leaf_amount(self):
        leaf = Expr.make_leaf("amount")
        assert isinstance(leaf, Leaf)
        assert leaf.semantic_type_in == [SemanticType.amount]
        assert leaf.key is None

    def test_make_leaf_ratio(self):
        leaf = Expr.make_leaf("ratio")
        assert leaf.semantic_type_in == [SemanticType.rate]

    def test_make_leaf_unsupported_family_raises(self):
        with pytest.raises(ValueError, match="Unsupported family"):
            Expr.make_leaf("price")

    def test_make_node_structure(self):
        left = Expr.make_leaf("amount")
        right = Expr.make_leaf("amount")
        node = Expr.make_node(op="sum", left=left, right=right)
        assert isinstance(node, Node)
        assert node.op == Operation.sum
        assert node.left is left
        assert node.right is right

    def test_make_time_agg(self):
        agg = Expr.make_time_agg(op="min")
        assert isinstance(agg, TimeAgg)
        assert agg.op == Operation.min
        assert agg.expr is None


class TestEligibleDerivedConcepts:
    def test_depth_zero_returns_empty(self):
        assert Expr.eligible_derived_concepts(depth=0, family="amount") == []

    def test_depth_one_includes_shallow(self):
        eligible = Expr.eligible_derived_concepts(depth=1, family="amount")
        assert "gross_profit" in eligible
        assert "net_income" not in eligible

    def test_depth_five_includes_all(self):
        eligible = Expr.eligible_derived_concepts(depth=5, family="amount")
        assert set(DERIVED_CONCEPTS).issubset(set(eligible))

    def test_wrong_family_excluded(self):
        assert Expr.eligible_derived_concepts(depth=5, family="ratio") == []


class TestChooseAmountOp:
    def test_no_time_aggregates(self):
        rng = random.Random(0)
        results = {Expr.choose_amount_op(rng, allow_time_aggregates=False) for _ in range(200)}
        assert results.isdisjoint({Operation.min, Operation.max, Operation.avg})

    def test_with_time_aggregates_can_produce_them(self):
        rng = random.Random(42)
        results = {Expr.choose_amount_op(rng, allow_time_aggregates=True) for _ in range(500)}
        assert results & {Operation.min, Operation.max, Operation.avg}


class TestCountNodesAndDepth:
    def test_leaf_counts(self):
        leaf = Expr.make_leaf("amount")
        assert Expr.count_nodes(leaf) == {"internal_nodes": 0, "leaves": 1, "total_nodes": 1}

    def test_derived_concept_counts_as_leaf(self):
        derived = Expr.make_derived_concept("gross_profit")
        assert Expr.count_nodes(derived) == {"internal_nodes": 0, "leaves": 1, "total_nodes": 1}

    def test_binary_node_counts(self):
        node = Expr.make_node("sum", Expr.make_leaf("amount"), Expr.make_leaf("amount"))
        assert Expr.count_nodes(node) == {"internal_nodes": 1, "leaves": 2, "total_nodes": 3}

    def test_leaf_depth_zero(self):
        assert Expr.actual_tree_depth(Expr.make_leaf("amount")) == 0

    def test_depth_one_node(self):
        node = Expr.make_node("diff", Expr.make_leaf("amount"), Expr.make_leaf("amount"))
        assert Expr.actual_tree_depth(node) == 1

    def test_depth_two_node(self):
        inner = Expr.make_node("sum", Expr.make_leaf("amount"), Expr.make_leaf("amount"))
        outer = Expr.make_node("diff", inner, Expr.make_leaf("amount"))
        assert Expr.actual_tree_depth(outer) == 2


class TestExpandFormula:
    def test_base_concept_returns_name(self):
        assert Expr.expand_formula_reference("revenue") == "revenue"

    def test_gross_profit_expansion(self):
        assert Expr.expand_formula_reference("gross_profit") == (
            "diff",
            ("revenue", "cost_of_goods_sold"),
        )

    def test_operating_income_expansion(self):
        op, args = Expr.expand_formula_reference("operating_income")
        assert op == "diff"
        assert args[0] == ("diff", ("revenue", "cost_of_goods_sold"))

    def test_current_assets_expansion(self):
        op, args = Expr.expand_formula_reference("current_assets")
        assert op == "sum"
        assert set(args) == {"cash", "accounts_receivable", "inventories", "short_term_investments"}


class TestNormalizeSymbolic:
    def test_string_unchanged(self):
        assert Expr.normalize_symbolic("revenue") == "revenue"

    def test_sum_args_sorted(self):
        op, args = Expr.normalize_symbolic(("sum", ("b", "a")))
        assert op == "sum"
        assert args == tuple(sorted(args, key=repr))

    def test_diff_args_not_sorted(self):
        assert Expr.normalize_symbolic(("diff", ("b", "a"))) == ("diff", ("b", "a"))


class TestSymbolicFromTree:
    def test_leaf_node(self):
        assert Expr.symbolic_from_tree(Expr.make_leaf("amount")) == "LEAF"

    def test_derived_concept_node(self):
        derived = Expr.make_derived_concept("gross_profit")
        expected = Expr.normalize_symbolic(Expr.expand_formula_reference("gross_profit"))
        assert Expr.symbolic_from_tree(derived) == expected

    def test_time_agg_node(self):
        agg = Expr.make_time_agg("min")
        assert Expr.symbolic_from_tree(agg) == ("time_agg", "min")


class TestContainsNamedDerived:
    def test_leaf_returns_false(self):
        assert not Expr.contains_named_derived(Expr.make_leaf("amount"), "gross_profit")

    def test_derived_concept_matches(self):
        derived = Expr.make_derived_concept("gross_profit")
        assert Expr.contains_named_derived(derived, "gross_profit")
        assert not Expr.contains_named_derived(derived, "net_income")

    def test_nested_in_binary_node(self):
        derived = Expr.make_derived_concept("net_income")
        node = Expr.make_node("sum", derived, Expr.make_leaf("amount"))
        assert Expr.contains_named_derived(node, "net_income")
        assert not Expr.contains_named_derived(node, "gross_profit")


class TestViolatesProtectedCanonicalForm:
    def test_plain_leaf_does_not_violate(self):
        assert Expr.violates_protected_canonical_form(Expr.make_leaf("amount")) is None

    def test_time_agg_never_violates(self):
        assert Expr.violates_protected_canonical_form(Expr.make_time_agg("min")) is None

    def test_named_derived_does_not_violate(self):
        assert Expr.violates_protected_canonical_form(Expr.make_derived_concept("gross_profit")) is None


class TestSampleTreeWithRejection:
    def test_returns_a_tree(self):
        tree, _ = Expr.sample_tree_with_rejection(
            max_depth=2,
            rng=random.Random(0),
            derived_prob=0.0,
        )
        assert isinstance(tree, (Leaf, Node, DerivedExpr, TimeAgg))

    def test_bad_depth_raises(self):
        with pytest.raises(ValueError, match="max_depth"):
            Expr.sample_tree_with_rejection(-1, random.Random(0), 0.0)

    def test_bad_derived_prob_raises(self):
        with pytest.raises(ValueError, match="derived_prob"):
            Expr.sample_tree_with_rejection(2, random.Random(0), 1.5)

    def test_bad_max_attempts_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            Expr.sample_tree_with_rejection(2, random.Random(0), 0.0, max_attempts=0)

    def test_depth_zero_gives_terminal(self):
        tree, _ = Expr.sample_tree_with_rejection(0, random.Random(7), 0.0)
        assert isinstance(tree, (Leaf, DerivedExpr, TimeAgg))

    def test_reproducible_with_same_seed(self):
        tree1, _ = Expr.sample_tree_with_rejection(3, random.Random(42), 0.3)
        tree2, _ = Expr.sample_tree_with_rejection(3, random.Random(42), 0.3)
        assert json.dumps(tree1.expr_to_json(), sort_keys=True) == json.dumps(
            tree2.expr_to_json(),
            sort_keys=True,
        )

    def test_no_dead_keys_in_payload(self):
        dead_keys = {
            "entity_group",
            "context_group",
            "concept_group",
            "time_series_group",
            "over_years_group",
            "section_group",
            "aggregation_group",
            "statement_group",
            "node_id",
            "family",
            "concept_depth",
        }

        for seed in range(50):
            tree, _ = Expr.sample_tree_with_rejection(3, random.Random(seed), 0.0)
            _check_no_dead_keys(tree.expr_to_json(), dead_keys)


def _check_no_dead_keys(tree_json: dict, dead_keys: set[str]) -> None:
    assert not dead_keys.intersection(tree_json.keys())
    if {"left", "right"}.issubset(tree_json):
        _check_no_dead_keys(tree_json["left"], dead_keys)
        _check_no_dead_keys(tree_json["right"], dead_keys)
