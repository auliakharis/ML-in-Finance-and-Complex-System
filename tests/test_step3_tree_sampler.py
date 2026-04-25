"""
Unit tests for compiler_pipeline/3.fixed_tree_sampler.py
"""
from __future__ import annotations

import importlib.util
import json
import random
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


sampler_mod = _load("3.fixed_tree_sampler.py", "fixed_tree_sampler")

IdGen                          = sampler_mod.IdGen
make_leaf                      = sampler_mod.make_leaf
make_node                      = sampler_mod.make_node
make_time_agg                  = sampler_mod.make_time_agg
make_derived_concept           = sampler_mod.make_derived_concept
eligible_derived_concepts      = sampler_mod.eligible_derived_concepts
choose_amount_op               = sampler_mod.choose_amount_op
count_nodes                    = sampler_mod.count_nodes
actual_tree_depth              = sampler_mod.actual_tree_depth
expand_formula_reference       = sampler_mod.expand_formula_reference
normalize_symbolic             = sampler_mod.normalize_symbolic
symbolic_from_tree             = sampler_mod.symbolic_from_tree
contains_named_derived         = sampler_mod.contains_named_derived
violates_protected_canonical_form = sampler_mod.violates_protected_canonical_form
protected_signatures           = sampler_mod.protected_signatures
sample_tree_with_rejection     = sampler_mod.sample_tree_with_rejection
DERIVED_CONCEPTS               = sampler_mod.DERIVED_CONCEPTS


class TestIdGen:
    def test_node_ids_sequential(self):
        idgen = IdGen()
        assert idgen.node_id() == "N0"
        assert idgen.node_id() == "N1"
        assert idgen.node_id() == "N2"

    def test_leaf_names_sequential(self):
        idgen = IdGen()
        assert idgen.leaf_name() == "L0"
        assert idgen.leaf_name() == "L1"

    def test_group_uses_prefix(self):
        idgen = IdGen()
        g = idgen.group("C")
        assert g.startswith("C") and g[1:].isdigit()

    def test_group_counter_shared(self):
        idgen = IdGen()
        g0 = idgen.group("C")
        g1 = idgen.group("E")
        assert int(g1[1:]) == int(g0[1:]) + 1

    def test_node_and_leaf_independent(self):
        idgen = IdGen()
        idgen.node_id()
        idgen.node_id()
        assert idgen.leaf_name() == "L0"


class TestMakeLeafAndNode:
    def test_make_leaf_amount(self):
        idgen = IdGen()
        leaf = make_leaf(idgen, family="amount", context_group="C0", entity_group="E0")
        assert leaf["kind"] == "leaf"
        assert leaf["family"] == "amount"
        assert leaf["semantic_type_in"] == ["amount"]

    def test_make_leaf_ratio(self):
        idgen = IdGen()
        leaf = make_leaf(idgen, family="ratio", context_group=None, entity_group="E0")
        assert leaf["semantic_type_in"] == ["ratio", "rate"]

    def test_make_leaf_unsupported_family_raises(self):
        idgen = IdGen()
        with pytest.raises(ValueError, match="Unsupported family"):
            make_leaf(idgen, family="price", context_group=None, entity_group="E0")

    def test_make_node_structure(self):
        idgen = IdGen()
        left = make_leaf(idgen, "amount", "C0", "E0")
        right = make_leaf(idgen, "amount", "C0", "E0")
        node = make_node(idgen, op="sum", family="amount", left=left, right=right, depth=1)
        assert node["kind"] == "node"
        assert node["op"] == "sum"
        assert node["left"] is left
        assert node["right"] is right

    def test_make_time_agg(self):
        idgen = IdGen()
        ta = make_time_agg(idgen, op="min", family="amount", depth=2,
                           entity_group="E0", concept_group="T0", over_years_group="Y0")
        assert ta["kind"] == "time_agg"
        assert ta["op"] == "min"


class TestEligibleDerivedConcepts:
    def test_depth_zero_returns_empty(self):
        assert eligible_derived_concepts(depth=0, family="amount") == []

    def test_depth_one_includes_shallow(self):
        eligible = eligible_derived_concepts(depth=1, family="amount")
        assert "gross_profit" in eligible
        assert "net_income" not in eligible

    def test_depth_five_includes_all(self):
        eligible = eligible_derived_concepts(depth=5, family="amount")
        assert set(DERIVED_CONCEPTS.keys()).issubset(set(eligible))

    def test_wrong_family_excluded(self):
        assert eligible_derived_concepts(depth=5, family="ratio") == []


class TestChooseAmountOp:
    def test_no_time_aggregates(self):
        rng = random.Random(0)
        results = {choose_amount_op(rng, allow_time_aggregates=False) for _ in range(200)}
        assert results.isdisjoint({"min", "max", "avg"})

    def test_with_time_aggregates_can_produce_them(self):
        rng = random.Random(42)
        results = {choose_amount_op(rng, allow_time_aggregates=True) for _ in range(500)}
        assert results & {"min", "max", "avg"}


class TestCountNodesAndDepth:
    def test_leaf_counts(self):
        idgen = IdGen()
        leaf = make_leaf(idgen, "amount", "C0", "E0")
        assert count_nodes(leaf) == {"internal_nodes": 0, "leaves": 1, "total_nodes": 1}

    def test_derived_concept_counts_as_leaf(self):
        dc = make_derived_concept("gross_profit", "amount", 1, "C0", "E0")
        assert count_nodes(dc) == {"internal_nodes": 0, "leaves": 1, "total_nodes": 1}

    def test_binary_node_counts(self):
        idgen = IdGen()
        left = make_leaf(idgen, "amount", "C0", "E0")
        right = make_leaf(idgen, "amount", "C0", "E0")
        node = make_node(idgen, "sum", "amount", left, right, depth=1)
        counts = count_nodes(node)
        assert counts == {"internal_nodes": 1, "leaves": 2, "total_nodes": 3}

    def test_leaf_depth_zero(self):
        idgen = IdGen()
        leaf = make_leaf(idgen, "amount", "C0", "E0")
        assert actual_tree_depth(leaf) == 0

    def test_depth_one_node(self):
        idgen = IdGen()
        left = make_leaf(idgen, "amount", "C0", "E0")
        right = make_leaf(idgen, "amount", "C0", "E0")
        node = make_node(idgen, "diff", "amount", left, right, depth=1)
        assert actual_tree_depth(node) == 1

    def test_depth_two_node(self):
        idgen = IdGen()
        l1 = make_leaf(idgen, "amount", "C0", "E0")
        l2 = make_leaf(idgen, "amount", "C0", "E0")
        l3 = make_leaf(idgen, "amount", "C0", "E0")
        inner = make_node(idgen, "sum", "amount", l1, l2, depth=1)
        outer = make_node(idgen, "diff", "amount", inner, l3, depth=2)
        assert actual_tree_depth(outer) == 2


class TestExpandFormula:
    def test_base_concept_returns_name(self):
        assert expand_formula_reference("revenue") == "revenue"

    def test_gross_profit_expansion(self):
        assert expand_formula_reference("gross_profit") == ("diff", ("revenue", "cost_of_goods_sold"))

    def test_operating_income_expansion(self):
        op, args = expand_formula_reference("operating_income")
        assert op == "diff"
        assert args[0] == ("diff", ("revenue", "cost_of_goods_sold"))

    def test_current_assets_expansion(self):
        op, args = expand_formula_reference("current_assets")
        assert op == "sum"
        assert set(args) == {"cash", "accounts_receivable", "inventories", "short_term_investments"}


class TestNormalizeSymbolic:
    def test_string_unchanged(self):
        assert normalize_symbolic("revenue") == "revenue"

    def test_sum_args_sorted(self):
        op, args = normalize_symbolic(("sum", ("b", "a")))
        assert op == "sum"
        assert args == tuple(sorted(args, key=repr))

    def test_diff_args_not_sorted(self):
        assert normalize_symbolic(("diff", ("b", "a"))) == ("diff", ("b", "a"))


class TestSymbolicFromTree:
    def test_leaf_node(self):
        idgen = IdGen()
        leaf = make_leaf(idgen, "amount", "C0", "E0")
        assert symbolic_from_tree(leaf) == f"LEAF:{leaf['name']}"

    def test_derived_concept_node(self):
        dc = make_derived_concept("gross_profit", "amount", 1, "C0", "E0")
        expected = normalize_symbolic(expand_formula_reference("gross_profit"))
        assert symbolic_from_tree(dc) == expected

    def test_time_agg_node(self):
        idgen = IdGen()
        ta = make_time_agg(idgen, "min", "amount", depth=1,
                           entity_group="E0", concept_group="T0", over_years_group="Y0")
        assert symbolic_from_tree(ta)[0] == "time_agg"


class TestContainsNamedDerived:
    def test_leaf_returns_false(self):
        idgen = IdGen()
        leaf = make_leaf(idgen, "amount", "C0", "E0")
        assert not contains_named_derived(leaf, "gross_profit")

    def test_derived_concept_matches(self):
        dc = make_derived_concept("gross_profit", "amount", 1, "C0", "E0")
        assert contains_named_derived(dc, "gross_profit")
        assert not contains_named_derived(dc, "net_income")

    def test_nested_in_binary_node(self):
        idgen = IdGen()
        dc = make_derived_concept("net_income", "amount", 5, "C0", "E0")
        leaf = make_leaf(idgen, "amount", "C0", "E0")
        node = make_node(idgen, "sum", "amount", dc, leaf, depth=1)
        assert contains_named_derived(node, "net_income")
        assert not contains_named_derived(node, "gross_profit")


class TestViolatesProtectedCanonicalForm:
    def test_plain_leaf_does_not_violate(self):
        idgen = IdGen()
        leaf = make_leaf(idgen, "amount", "C0", "E0")
        assert violates_protected_canonical_form(leaf, protected_signatures()) is None

    def test_time_agg_never_violates(self):
        idgen = IdGen()
        ta = make_time_agg(idgen, "min", "amount", depth=1,
                           entity_group="E0", concept_group="T0", over_years_group="Y0")
        assert violates_protected_canonical_form(ta, protected_signatures()) is None

    def test_named_derived_does_not_violate(self):
        dc = make_derived_concept("gross_profit", "amount", 1, "C0", "E0")
        assert violates_protected_canonical_form(dc, protected_signatures()) is None


class TestSampleTreeWithRejection:
    def test_returns_a_tree(self):
        rng = random.Random(0)
        idgen = IdGen()
        tree, _ = sample_tree_with_rejection(max_depth=2, rng=rng, idgen=idgen, derived_prob=0.0)
        assert tree["kind"] in {"leaf", "node", "derived_concept", "time_agg"}

    def test_bad_depth_raises(self):
        with pytest.raises(ValueError, match="max_depth"):
            sample_tree_with_rejection(-1, random.Random(0), IdGen(), 0.0)

    def test_bad_derived_prob_raises(self):
        with pytest.raises(ValueError, match="derived_prob"):
            sample_tree_with_rejection(2, random.Random(0), IdGen(), 1.5)

    def test_bad_max_attempts_raises(self):
        with pytest.raises(ValueError, match="max_attempts"):
            sample_tree_with_rejection(2, random.Random(0), IdGen(), 0.0, max_attempts=0)

    def test_depth_zero_gives_terminal(self):
        tree, _ = sample_tree_with_rejection(0, random.Random(7), IdGen(), 0.0)
        assert tree["kind"] in {"leaf", "derived_concept", "time_agg"}

    def test_reproducible_with_same_seed(self):
        tree1, _ = sample_tree_with_rejection(3, random.Random(42), IdGen(), 0.3)
        tree2, _ = sample_tree_with_rejection(3, random.Random(42), IdGen(), 0.3)
        assert json.dumps(tree1, sort_keys=True) == json.dumps(tree2, sort_keys=True)
