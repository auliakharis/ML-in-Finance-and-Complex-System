"""
Unit tests for compiler_pipeline/4.tree_to_language.py
"""
from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

_PIPELINE = Path(__file__).resolve().parent.parent / "compiler_pipeline"


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, _PIPELINE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


atoms_mod = _load("2.fixed_building_atoms.py", "fixed_building_atoms_4")
sampler_mod = _load("3.fixed_tree_sampler.py", "fixed_tree_sampler_4")
lang_mod = _load("4.tree_to_language.py", "tree_to_language")

build_atoms = atoms_mod.build_atoms
BASE_CONCEPTS = atoms_mod.BASE_CONCEPTS

IdGen = sampler_mod.IdGen
make_leaf = sampler_mod.make_leaf
make_node = sampler_mod.make_node
make_derived_concept = sampler_mod.make_derived_concept
make_time_agg = sampler_mod.make_time_agg
sample_tree_with_rejection = sampler_mod.sample_tree_with_rejection
actual_tree_depth = sampler_mod.actual_tree_depth
DERIVED_CONCEPTS = sampler_mod.DERIVED_CONCEPTS

Atom = lang_mod.Atom
Leaf = lang_mod.Leaf
Node = lang_mod.Node
DerivedExpr = lang_mod.DerivedExpr
Literal = lang_mod.Literal
parse_expr = lang_mod.parse_expr
expr_to_json = lang_mod.expr_to_json
show_expr = lang_mod.show_expr
flatten_sum = lang_mod.flatten_sum
flatten_leaves = lang_mod.flatten_leaves
oxford_join = lang_mod.oxford_join
AtomIndex = lang_mod.AtomIndex
Evaluator = lang_mod.Evaluator
SemanticAnalyzer = lang_mod.SemanticAnalyzer
QuestionRenderer = lang_mod.QuestionRenderer
compile_tree_payload = lang_mod.compile_tree_payload
expr_depth = lang_mod.expr_depth
SemanticError = lang_mod.SemanticError


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


def _minimal_atoms(n_companies: int = 2, years: list[int] | None = None) -> Dict[str, Atom]:
    df = _minimal_df(n_companies, years)
    raw = build_atoms(df)
    return {a["key"]: Atom(**{k: v for k, v in a.items() if k in {
        "key", "concept", "semantic_type", "label", "entity",
        "period", "unit", "value", "depth", "parent_concept", "role"
    }}) for a in raw}


class TestParseExpr:
    def test_parse_leaf(self):
        expr = parse_expr({"leaf": "0_revenue"})
        assert isinstance(expr, Leaf)
        assert expr.key == "0_revenue"

    def test_parse_literal(self):
        expr = parse_expr({"literal": 3.0})
        assert isinstance(expr, Literal)
        assert expr.value == 3.0

    def test_parse_node_sum(self):
        obj = {"op": "sum", "left": {"leaf": "L0"}, "right": {"leaf": "L1"}}
        expr = parse_expr(obj)
        assert isinstance(expr, Node)
        assert expr.op == "sum"
        assert isinstance(expr.left, Leaf)
        assert isinstance(expr.right, Leaf)

    def test_parse_node_depth_check(self):
        obj = {"op": "diff", "left": {"leaf": "L0"}, "right": {"leaf": "L1"}, "depth": 99}
        with pytest.raises(ValueError, match="Depth mismatch"):
            parse_expr(obj)

    def test_parse_derived(self):
        obj = {
            "derived": "gross_profit",
            "expanded": {"op": "diff", "left": {"leaf": "L0"}, "right": {"leaf": "L1"}},
        }
        expr = parse_expr(obj)
        assert isinstance(expr, DerivedExpr)
        assert expr.name == "gross_profit"

    def test_parse_unsupported_op_raises(self):
        with pytest.raises(ValueError, match="Unsupported op"):
            parse_expr({"op": "power", "left": {"leaf": "L0"}, "right": {"leaf": "L1"}})

    def test_parse_invalid_object_raises(self):
        with pytest.raises(ValueError):
            parse_expr({"garbage": True})

    def test_parse_non_dict_raises(self):
        with pytest.raises(ValueError):
            parse_expr("not_a_dict")


class TestExprToJsonRoundtrip:
    def test_leaf_roundtrip(self):
        leaf = Leaf(key="0_revenue")
        j = expr_to_json(leaf)
        assert j["leaf"] == "0_revenue"
        recovered = parse_expr(j)
        assert isinstance(recovered, Leaf)
        assert recovered.key == leaf.key

    def test_literal_roundtrip(self):
        lit = Literal(value=4.0)
        j = expr_to_json(lit)
        assert j["literal"] == 4.0
        recovered = parse_expr(j)
        assert isinstance(recovered, Literal)
        assert recovered.value == lit.value

    def test_node_roundtrip(self):
        node = Node(op="sum", left=Leaf("A"), right=Leaf("B"),
                    depth=1 + max(expr_depth(Leaf("A")), expr_depth(Leaf("B"))))
        j = expr_to_json(node)
        recovered = parse_expr(j)
        assert isinstance(recovered, Node)
        assert recovered.op == "sum"


class TestShowExpr:
    def test_leaf(self):
        assert show_expr(Leaf("mykey")) == "mykey"

    def test_literal(self):
        assert show_expr(Literal(3.0)) == "3.0"

    def test_derived(self):
        inner = Node("diff", Leaf("A"), Leaf("B"), depth=1)
        d = DerivedExpr("gross_profit", inner, depth=1)
        assert show_expr(d) == "gross_profit"

    def test_node(self):
        node = Node("sum", Leaf("A"), Leaf("B"), depth=1)
        assert show_expr(node) == "sum(A, B)"


class TestFlattenHelpers:
    def test_flatten_leaves_single_leaf(self):
        assert flatten_leaves(Leaf("X")) == [Leaf("X")]

    def test_flatten_leaves_literal_empty(self):
        assert flatten_leaves(Literal(3.0)) == []

    def test_flatten_leaves_node(self):
        node = Node("ratio", Leaf("A"), Leaf("B"), depth=1)
        assert {l.key for l in flatten_leaves(node)} == {"A", "B"}

    def test_flatten_sum_chain(self):
        inner = Node("sum", Leaf("A"), Leaf("B"), depth=1)
        outer = Node("sum", inner, Leaf("C"), depth=2)
        assert {l.key for l in flatten_sum(outer)} == {"A", "B", "C"}

    def test_flatten_sum_stops_at_derived(self):
        derived = DerivedExpr("gross_profit", Node("diff", Leaf("A"), Leaf("B"), depth=1), depth=1)
        outer = Node("sum", derived, Leaf("C"), depth=2)
        leaves = flatten_sum(outer)
        assert Leaf("C") in leaves
        assert len(leaves) == 1


class TestOxfordJoin:
    def test_empty(self):
        assert oxford_join([]) == ""

    def test_single(self):
        assert oxford_join(["alpha"]) == "alpha"

    def test_two(self):
        assert oxford_join(["alpha", "beta"]) == "alpha and beta"

    def test_three(self):
        assert oxford_join(["a", "b", "c"]) == "a, b, and c"

    def test_four(self):
        result = oxford_join(["a", "b", "c", "d"])
        assert result.endswith(", and d")


class TestAtomIndex:
    def setup_method(self):
        self.atoms = _minimal_atoms(n_companies=2, years=[2021, 2022, 2023])
        self.index = AtomIndex(self.atoms)

    def test_entities_sorted(self):
        assert self.index.entities == sorted(self.index.entities)

    def test_periods_sorted(self):
        assert self.index.periods == sorted(self.index.periods)

    def test_amount_concepts_present(self):
        assert "revenue" in self.index.amount_concepts

    def test_ratio_concepts_present(self):
        assert "income_tax" in self.index.ratio_concepts

    def test_by_concept_index(self):
        by_rev = self.index.by_concept.get("revenue", [])
        assert len(by_rev) == 6  # 2 companies × 3 years

    def test_filter_by_concept(self):
        results = self.index.filter_atoms(concept="revenue")
        assert all(a.concept == "revenue" for a in results)

    def test_filter_by_entity(self):
        results = self.index.filter_atoms(entity="Corp0")
        assert all(a.entity == "Corp0" for a in results)

    def test_filter_by_period(self):
        results = self.index.filter_atoms(period="2022")
        assert all(a.period == "2022" for a in results)

    def test_filter_combined(self):
        results = self.index.filter_atoms(concept="cash", entity="Corp0", period="2021")
        assert len(results) == 1
        assert results[0].concept == "cash"
        assert results[0].entity == "Corp0"
        assert results[0].period == "2021"

    def test_filter_by_semantic_type(self):
        results = self.index.filter_atoms(semantic_types=["ratio", "rate"])
        assert all(a.semantic_type in {"ratio", "rate"} for a in results)


class TestEvaluator:
    def setup_method(self):
        self.atoms = _minimal_atoms(n_companies=1, years=[2021, 2022])
        self.ev = Evaluator(self.atoms)

    def _leaf(self, concept: str, period: str = "2021") -> Leaf:
        key = next(k for k, a in self.atoms.items()
                   if a.concept == concept and a.period == period)
        return Leaf(key=key)

    def _val(self, concept: str, period: str = "2021") -> float:
        key = next(k for k, a in self.atoms.items()
                   if a.concept == concept and a.period == period)
        return self.atoms[key].value

    def test_eval_leaf(self):
        assert self.ev.eval(self._leaf("revenue")) == self._val("revenue")

    def test_eval_literal(self):
        assert self.ev.eval(Literal(7.0)) == 7.0

    def test_eval_sum(self):
        node = Node("sum", self._leaf("revenue"), self._leaf("cost_of_goods_sold"), depth=1)
        expected = self._val("revenue") + self._val("cost_of_goods_sold")
        assert self.ev.eval(node) == pytest.approx(expected)

    def test_eval_diff(self):
        node = Node("diff", self._leaf("revenue"), self._leaf("cost_of_goods_sold"), depth=1)
        expected = self._val("revenue") - self._val("cost_of_goods_sold")
        assert self.ev.eval(node) == pytest.approx(expected)

    def test_eval_ratio(self):
        node = Node("ratio", self._leaf("revenue"), self._leaf("total_assets"), depth=1)
        expected = self._val("revenue") / self._val("total_assets")
        assert self.ev.eval(node) == pytest.approx(expected)

    def test_eval_ratio_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            self.ev.eval(Node("ratio", Literal(5.0), Literal(0.0), depth=1))

    def test_eval_mul(self):
        node = Node("mul", self._leaf("revenue"), self._leaf("income_tax"), depth=1)
        expected = self._val("revenue") * self._val("income_tax")
        assert self.ev.eval(node) == pytest.approx(expected)

    def test_eval_growth(self):
        node = Node("growth", self._leaf("revenue", "2022"), self._leaf("revenue", "2021"), depth=1)
        base = self._val("revenue", "2021")
        curr = self._val("revenue", "2022")
        assert self.ev.eval(node) == pytest.approx((curr - base) / base)

    def test_eval_growth_zero_base_raises(self):
        with pytest.raises(ZeroDivisionError):
            self.ev.eval(Node("growth", Literal(5.0), Literal(0.0), depth=1))

    def test_eval_min(self):
        assert self.ev.eval(Node("min", Literal(3.0), Literal(7.0), depth=1)) == 3.0

    def test_eval_max(self):
        assert self.ev.eval(Node("max", Literal(3.0), Literal(7.0), depth=1)) == 7.0

    def test_eval_derived_expr(self):
        inner = Node("diff", self._leaf("revenue"), self._leaf("cost_of_goods_sold"), depth=1)
        derived = DerivedExpr("gross_profit", inner, depth=1)
        expected = self._val("revenue") - self._val("cost_of_goods_sold")
        assert self.ev.eval(derived) == pytest.approx(expected)

    def test_eval_unsupported_op_raises(self):
        with pytest.raises(ValueError, match="Unsupported op"):
            self.ev.eval(Node("power", Literal(2.0), Literal(3.0), depth=1))


class TestSemanticAnalyzer:
    def setup_method(self):
        self.atoms = _minimal_atoms(n_companies=2, years=[2021, 2022, 2023])
        self.sa = SemanticAnalyzer(self.atoms)

    def _leaf(self, concept: str, entity: str = "Corp0", period: str = "2021") -> Leaf:
        key = next(k for k, a in self.atoms.items()
                   if a.concept == concept and a.entity == entity and a.period == period)
        return Leaf(key=key)

    def test_analyze_leaf(self):
        result = self.sa.analyze(self._leaf("revenue"))
        assert result.meaning.kind == "leaf_metric"
        assert result.meaning.concept == "revenue"
        assert result.meaning.entity == "Corp0"
        assert result.meaning.period == "2021"
        assert result.depth == 0

    def test_analyze_literal(self):
        result = self.sa.analyze(Literal(3.0))
        assert result.meaning.kind == "literal_scalar"
        assert result.meaning.semantic_type == "ratio"

    def test_analyze_derived_expr(self):
        inner = Node("diff", self._leaf("revenue"), self._leaf("cost_of_goods_sold"), depth=1)
        derived = DerivedExpr("gross_profit", inner, depth=1)
        result = self.sa.analyze(derived)
        assert result.meaning.kind == "derived_metric"
        assert result.meaning.concept == "gross_profit"

    def test_analyze_sum_current_assets(self):
        node = Node("sum",
                    Node("sum",
                         Node("sum",
                              self._leaf("cash"),
                              self._leaf("accounts_receivable"), depth=1),
                         self._leaf("inventories"), depth=2),
                    self._leaf("short_term_investments"), depth=3)
        result = self.sa.analyze(node)
        assert result.meaning.kind == "aggregate_components"
        assert result.meaning.concept == "current_assets"

    def test_analyze_sum_generic(self):
        node = Node("sum", self._leaf("revenue"), self._leaf("total_assets"), depth=1)
        result = self.sa.analyze(node)
        assert result.meaning.semantic_type == "amount"
        assert result.meaning.kind in {"sum_amount", "sum_amounts"}

    def test_analyze_diff_same_period(self):
        node = Node("diff", self._leaf("revenue"), self._leaf("cost_of_goods_sold"), depth=1)
        result = self.sa.analyze(node)
        assert result.meaning.kind == "difference_amount"
        assert result.meaning.semantic_type == "amount"

    def test_analyze_diff_across_periods(self):
        node = Node("diff",
                    self._leaf("revenue", period="2022"),
                    self._leaf("revenue", period="2021"), depth=1)
        result = self.sa.analyze(node)
        assert result.meaning.kind == "change_over_time"
        assert result.meaning.from_period == "2021"
        assert result.meaning.to_period == "2022"

    def test_analyze_ratio_amounts(self):
        node = Node("ratio", self._leaf("revenue"), self._leaf("total_assets"), depth=1)
        result = self.sa.analyze(node)
        assert result.meaning.kind == "ratio_amounts"
        assert result.meaning.semantic_type == "ratio"

    def test_analyze_ratio_avg_over_periods(self):
        leaves = [self._leaf("revenue", period=p) for p in ["2021", "2022", "2023"]]
        sum_expr = Node("sum", Node("sum", leaves[0], leaves[1], depth=1), leaves[2], depth=2)
        avg_node = Node("ratio", sum_expr, Literal(3.0), depth=3)
        result = self.sa.analyze(avg_node)
        assert result.meaning.kind == "avg_over_all_periods"

    def test_analyze_mul(self):
        node = Node("mul", self._leaf("revenue"), self._leaf("income_tax"), depth=1)
        result = self.sa.analyze(node)
        assert result.meaning.kind == "scaled_amount"
        assert result.meaning.semantic_type == "amount"

    def test_analyze_growth(self):
        node = Node("growth",
                    self._leaf("revenue", period="2022"),
                    self._leaf("revenue", period="2021"), depth=1)
        result = self.sa.analyze(node)
        assert result.meaning.kind == "growth_rate"
        assert result.meaning.semantic_type == "ratio"

    def test_analyze_min_over_time(self):
        node = Node("min",
                    self._leaf("revenue", period="2021"),
                    self._leaf("revenue", period="2022"), depth=1)
        result = self.sa.analyze(node)
        assert "min" in result.meaning.kind

    def test_analyze_max_over_time(self):
        node = Node("max",
                    self._leaf("revenue", period="2022"),
                    self._leaf("revenue", period="2023"), depth=1)
        result = self.sa.analyze(node)
        assert "max" in result.meaning.kind

    def test_sum_cross_entity_raises(self):
        node = Node("sum",
                    self._leaf("revenue", entity="Corp0"),
                    self._leaf("revenue", entity="Corp1"), depth=1)
        with pytest.raises(SemanticError):
            self.sa.analyze(node)

    def test_diff_cross_unit_raises(self):
        node = Node("diff", self._leaf("revenue"), self._leaf("income_tax"), depth=1)
        with pytest.raises(SemanticError):
            self.sa.analyze(node)

    def test_unknown_leaf_key_raises(self):
        with pytest.raises(SemanticError, match="Unknown leaf key"):
            self.sa.analyze(Leaf(key="does_not_exist"))


class TestQuestionRenderer:
    def setup_method(self):
        self.atoms = _minimal_atoms(n_companies=2, years=[2021, 2022, 2023])
        self.sa = SemanticAnalyzer(self.atoms)
        self.renderer = QuestionRenderer()

    def _leaf(self, concept: str, entity: str = "Corp0", period: str = "2021") -> Leaf:
        key = next(k for k, a in self.atoms.items()
                   if a.concept == concept and a.entity == entity and a.period == period)
        return Leaf(key=key)

    def test_renders_leaf_question(self):
        result = self.sa.analyze(self._leaf("revenue"))
        q = self.renderer.render(result)
        assert isinstance(q, str) and len(q) > 5
        assert "revenue" in q.lower() or "Corp0" in q or "2021" in q

    def test_renders_growth_question(self):
        node = Node("growth",
                    self._leaf("revenue", period="2022"),
                    self._leaf("revenue", period="2021"), depth=1)
        q = self.renderer.render(self.sa.analyze(node))
        assert isinstance(q, str)
        assert any(w in q.lower() for w in ["growth", "rate"])

    def test_renders_avg_question(self):
        leaves = [self._leaf("cash", period=p) for p in ["2021", "2022", "2023"]]
        sum_e = Node("sum", Node("sum", leaves[0], leaves[1], depth=1), leaves[2], depth=2)
        avg_node = Node("ratio", sum_e, Literal(3.0), depth=3)
        q = self.renderer.render(self.sa.analyze(avg_node))
        assert isinstance(q, str)
        assert "average" in q.lower() or "avg" in q.lower()

    def test_renders_ratio_question(self):
        node = Node("ratio", self._leaf("revenue"), self._leaf("total_assets"), depth=1)
        q = self.renderer.render(self.sa.analyze(node))
        assert isinstance(q, str) and len(q) > 5

    def test_no_empty_question(self):
        for concept in ["revenue", "cash", "total_assets"]:
            q = self.renderer.render(self.sa.analyze(self._leaf(concept)))
            assert q.strip() != ""


class TestCompileTreePayload:
    def setup_method(self):
        self.atoms = _minimal_atoms(n_companies=2, years=[2021, 2022, 2023])

    def _make_payload(self, depth: int, seed: int) -> dict:
        rng = random.Random(seed)
        idgen = IdGen()
        tree, _ = sample_tree_with_rejection(max_depth=depth, rng=rng, idgen=idgen, derived_prob=0.2)
        return {"tree": tree, "derived_concepts": {
            name: {"formula": spec["formula"]}
            for name, spec in DERIVED_CONCEPTS.items()
        }}

    def test_compile_depth_1(self):
        payload = self._make_payload(1, seed=0)
        expr = compile_tree_payload(payload, self.atoms, seed=0)
        assert math.isfinite(Evaluator(self.atoms).eval(expr))

    def test_compile_depth_3(self):
        payload = self._make_payload(3, seed=1)
        expr = compile_tree_payload(payload, self.atoms, seed=1)
        assert math.isfinite(Evaluator(self.atoms).eval(expr))

    def test_question_is_string(self):
        payload = self._make_payload(2, seed=5)
        expr = compile_tree_payload(payload, self.atoms, seed=5)
        q = QuestionRenderer().render(SemanticAnalyzer(self.atoms).analyze(expr))
        assert isinstance(q, str) and len(q) > 0

    def test_different_seeds_produce_finite_values(self):
        payload = self._make_payload(2, seed=10)
        expr_a = compile_tree_payload(payload, self.atoms, seed=10)
        expr_b = compile_tree_payload(payload, self.atoms, seed=99)
        ev = Evaluator(self.atoms)
        assert math.isfinite(ev.eval(expr_a))
        assert math.isfinite(ev.eval(expr_b))
