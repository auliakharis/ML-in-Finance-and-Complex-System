"""
Unit tests for the v2 expression, evaluation, semantic, and rendering layers,
adapted from the v1 step-4 suite.
"""
from __future__ import annotations

import math
import random

import pytest

import temp.compiler_pipeline_refactored.question_renderer as question_renderer_module
from temp.compiler_pipeline_refactored.evaluator import Evaluator
from temp.compiler_pipeline_refactored.question_renderer import QuestionRenderer
from temp.compiler_pipeline_refactored.semantic_analyzer import SemanticAnalyzer
from temp.compiler_pipeline_refactored.tree import (
    AtomIndex,
    DERIVED_CONCEPTS,
    DerivedExpr,
    Expr,
    Leaf,
    Literal,
    Node,
    Operation,
    SemanticError,
    SemanticType,
    compile_tree_payload,
)
from temp.compiler_pipeline_refactored.utils import oxford_join


def leaf_for(atoms, concept: str, entity: str = "Corp0", period: str = "2021") -> Leaf:
    key = next(
        key
        for key, atom in atoms.items()
        if atom.concept == concept and atom.entity == entity and atom.period == period
    )
    return Leaf(key=key)


def value_for(atoms, concept: str, entity: str = "Corp0", period: str = "2021") -> float:
    return atoms[leaf_for(atoms, concept, entity=entity, period=period).key].value


class TestParseExpr:
    def test_parse_leaf(self):
        expr = Expr.parse_expr({"leaf": "0_revenue"})
        assert isinstance(expr, Leaf)
        assert expr.key == "0_revenue"

    def test_parse_literal(self):
        expr = Expr.parse_expr({"literal": 3.0})
        assert isinstance(expr, Literal)
        assert expr.value == 3.0

    def test_parse_node_sum(self):
        expr = Expr.parse_expr({"op": "sum", "left": {"leaf": "L0"}, "right": {"leaf": "L1"}})
        assert isinstance(expr, Node)
        assert expr.op == Operation.sum
        assert isinstance(expr.left, Leaf)
        assert isinstance(expr.right, Leaf)

    def test_parse_node_depth_check(self):
        obj = {"op": "diff", "left": {"leaf": "L0"}, "right": {"leaf": "L1"}, "depth": 99}
        with pytest.raises(ValueError, match="Depth mismatch"):
            Expr.parse_expr(obj)

    def test_parse_derived(self):
        obj = {
            "derived": "gross_profit",
            "expanded": {"op": "diff", "left": {"leaf": "L0"}, "right": {"leaf": "L1"}},
        }
        expr = Expr.parse_expr(obj)
        assert isinstance(expr, DerivedExpr)
        assert expr.name == "gross_profit"

    def test_parse_unsupported_op_raises(self):
        with pytest.raises(ValueError):
            Expr.parse_expr({"op": "power", "left": {"leaf": "L0"}, "right": {"leaf": "L1"}})

    def test_parse_invalid_object_raises(self):
        with pytest.raises(ValueError):
            Expr.parse_expr({"garbage": True})

    def test_parse_non_dict_raises(self):
        with pytest.raises(ValueError):
            Expr.parse_expr("not_a_dict")


class TestExprToJsonRoundtrip:
    def test_leaf_roundtrip(self):
        leaf = Leaf(key="0_revenue")
        recovered = Expr.parse_expr(leaf.expr_to_json())
        assert isinstance(recovered, Leaf)
        assert recovered.key == leaf.key

    def test_literal_roundtrip(self):
        literal = Literal(value=4.0)
        recovered = Expr.parse_expr(literal.expr_to_json())
        assert isinstance(recovered, Literal)
        assert recovered.value == literal.value

    def test_node_roundtrip(self):
        node = Node(op="sum", left=Leaf("A"), right=Leaf("B"))
        recovered = Expr.parse_expr(node.expr_to_json())
        assert isinstance(recovered, Node)
        assert recovered.op == Operation.sum


class TestShowExpr:
    def test_leaf(self):
        assert Leaf("mykey").show_expr() == "mykey"

    def test_literal(self):
        assert Literal(3.0).show_expr() == "3.0"

    def test_derived(self):
        inner = Node("diff", Leaf("A"), Leaf("B"))
        derived = DerivedExpr("gross_profit", inner)
        assert derived.show_expr() == "gross_profit"

    def test_node(self):
        node = Node("sum", Leaf("A"), Leaf("B"))
        assert node.show_expr() == "sum(A, B)"


class TestFlattenHelpers:
    def test_flatten_leaves_single_leaf(self):
        assert [leaf.key for leaf in Leaf("X").flatten_leaves()] == ["X"]

    def test_flatten_leaves_literal_empty(self):
        assert Literal(3.0).flatten_leaves() == []

    def test_flatten_leaves_node(self):
        node = Node("ratio", Leaf("A"), Leaf("B"))
        assert {leaf.key for leaf in node.flatten_leaves()} == {"A", "B"}

    def test_flatten_sum_chain(self):
        inner = Node("sum", Leaf("A"), Leaf("B"))
        outer = Node("sum", inner, Leaf("C"))
        assert {leaf.key for leaf in outer.flatten_sum()} == {"A", "B", "C"}

    def test_flatten_sum_stops_at_derived(self):
        derived = DerivedExpr("gross_profit", Node("diff", Leaf("A"), Leaf("B")))
        outer = Node("sum", derived, Leaf("C"))
        assert [leaf.key for leaf in outer.flatten_sum()] == ["C"]


class TestOxfordJoin:
    def test_empty(self):
        assert oxford_join([]) == ""

    def test_single(self):
        assert oxford_join(["alpha"]) == "alpha"

    def test_two(self):
        assert oxford_join(["alpha", "beta"]) == "alpha and beta"

    def test_three(self):
        assert oxford_join(["a", "b", "c"]) == "a, b, and c"


class TestAtomIndex:
    def test_entities_sorted(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        assert index.entities == sorted(index.entities)

    def test_periods_sorted(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        assert index.periods == sorted(index.periods)

    def test_amount_concepts_present(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        assert "revenue" in index.amount_concepts()

    def test_ratio_concepts_present(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        assert "income_tax" in index.rate_concepts()

    def test_filter_by_concept(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        results = index.filter_atoms(concept="revenue")
        assert len(results) == 6
        assert all(atom.concept == "revenue" for atom in results)

    def test_filter_by_entity(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        results = index.filter_atoms(entity="Corp0")
        assert results and all(atom.entity == "Corp0" for atom in results)

    def test_filter_by_period(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        results = index.filter_atoms(period="2022")
        assert results and all(atom.period == "2022" for atom in results)

    def test_filter_combined(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        results = index.filter_atoms(concept="cash", entity="Corp0", period="2021")
        assert len(results) == 1
        assert results[0].concept == "cash"
        assert results[0].entity == "Corp0"
        assert results[0].period == "2021"

    def test_filter_by_semantic_type(self, minimal_atoms):
        index = AtomIndex.store_from_atoms(minimal_atoms)
        results = index.filter_atoms(semantic_types=[SemanticType.rate])
        assert results and all(atom.semantic_type == SemanticType.rate for atom in results)


class TestEvaluator:
    def test_eval_leaf(self, minimal_atoms):
        evaluator = Evaluator(minimal_atoms)
        assert evaluator.eval(leaf_for(minimal_atoms, "revenue")) == value_for(minimal_atoms, "revenue")

    def test_eval_literal(self, minimal_atoms):
        assert Evaluator(minimal_atoms).eval(Literal(7.0)) == 7.0

    def test_eval_sum(self, minimal_atoms):
        evaluator = Evaluator(minimal_atoms)
        node = Node("sum", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "cost_of_goods_sold"))
        expected = value_for(minimal_atoms, "revenue") + value_for(minimal_atoms, "cost_of_goods_sold")
        assert evaluator.eval(node) == pytest.approx(expected)

    def test_eval_diff(self, minimal_atoms):
        evaluator = Evaluator(minimal_atoms)
        node = Node("diff", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "cost_of_goods_sold"))
        expected = value_for(minimal_atoms, "revenue") - value_for(minimal_atoms, "cost_of_goods_sold")
        assert evaluator.eval(node) == pytest.approx(expected)

    def test_eval_ratio(self, minimal_atoms):
        evaluator = Evaluator(minimal_atoms)
        node = Node("ratio", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "total_assets"))
        expected = value_for(minimal_atoms, "revenue") / value_for(minimal_atoms, "total_assets")
        assert evaluator.eval(node) == pytest.approx(expected)

    def test_eval_ratio_division_by_zero(self, minimal_atoms):
        with pytest.raises(ZeroDivisionError):
            Evaluator(minimal_atoms).eval(Node("ratio", Literal(5.0), Literal(0.0)))

    def test_eval_mul(self, minimal_atoms):
        evaluator = Evaluator(minimal_atoms)
        node = Node("mul", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "income_tax"))
        expected = value_for(minimal_atoms, "revenue") * value_for(minimal_atoms, "income_tax")
        assert evaluator.eval(node) == pytest.approx(expected)

    def test_eval_growth(self, minimal_atoms):
        evaluator = Evaluator(minimal_atoms)
        node = Node(
            "growth",
            leaf_for(minimal_atoms, "revenue", period="2022"),
            leaf_for(minimal_atoms, "revenue", period="2021"),
        )
        curr = value_for(minimal_atoms, "revenue", period="2022")
        base = value_for(minimal_atoms, "revenue", period="2021")
        assert evaluator.eval(node) == pytest.approx((curr - base) / base)

    def test_eval_growth_zero_base_raises(self, minimal_atoms):
        with pytest.raises(ZeroDivisionError):
            Evaluator(minimal_atoms).eval(Node("growth", Literal(5.0), Literal(0.0)))

    def test_eval_min(self, minimal_atoms):
        assert Evaluator(minimal_atoms).eval(Node("min", Literal(3.0), Literal(7.0))) == 3.0

    def test_eval_max(self, minimal_atoms):
        assert Evaluator(minimal_atoms).eval(Node("max", Literal(3.0), Literal(7.0))) == 7.0

    def test_eval_derived_expr(self, minimal_atoms):
        evaluator = Evaluator(minimal_atoms)
        inner = Node("diff", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "cost_of_goods_sold"))
        derived = DerivedExpr("gross_profit", inner)
        expected = value_for(minimal_atoms, "revenue") - value_for(minimal_atoms, "cost_of_goods_sold")
        assert evaluator.eval(derived) == pytest.approx(expected)

class TestSemanticAnalyzer:
    def test_analyze_leaf(self, minimal_atoms):
        result = SemanticAnalyzer(minimal_atoms).analyze(leaf_for(minimal_atoms, "revenue"))
        assert result.meaning.kind == "leaf_metric"
        assert result.meaning.concept == "revenue"
        assert result.meaning.entity == "Corp0"
        assert result.meaning.period == "2021"
        assert result.depth == 0

    def test_analyze_literal(self, minimal_atoms):
        result = SemanticAnalyzer(minimal_atoms).analyze(Literal(3.0))
        assert result.meaning.kind == "literal_scalar"
        assert result.meaning.semantic_type == SemanticType.ratio

    def test_analyze_derived_expr(self, minimal_atoms):
        inner = Node("diff", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "cost_of_goods_sold"))
        result = SemanticAnalyzer(minimal_atoms).analyze(DerivedExpr("gross_profit", inner))
        assert result.meaning.kind == "derived_metric"
        assert result.meaning.concept == "gross_profit"

    def test_analyze_sum_current_assets(self, minimal_atoms):
        node = Node(
            "sum",
            Node(
                "sum",
                Node(
                    "sum",
                    leaf_for(minimal_atoms, "cash"),
                    leaf_for(minimal_atoms, "accounts_receivable"),
                ),
                leaf_for(minimal_atoms, "inventories"),
            ),
            leaf_for(minimal_atoms, "short_term_investments"),
        )
        result = SemanticAnalyzer(minimal_atoms).analyze(node)
        assert result.meaning.kind == "aggregate_components"
        assert result.meaning.concept == "current_assets"

    def test_analyze_sum_generic(self, minimal_atoms):
        node = Node("sum", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "total_assets"))
        result = SemanticAnalyzer(minimal_atoms).analyze(node)
        assert result.meaning.semantic_type == SemanticType.amount
        assert result.meaning.kind in {"sum_amount", "sum_amounts"}

    def test_analyze_diff_same_period(self, minimal_atoms):
        node = Node("diff", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "cost_of_goods_sold"))
        result = SemanticAnalyzer(minimal_atoms).analyze(node)
        assert result.meaning.kind == "difference_amount"
        assert result.meaning.semantic_type == SemanticType.amount

    def test_analyze_diff_across_periods_raises(self, minimal_atoms):
        node = Node(
            "diff",
            leaf_for(minimal_atoms, "revenue", period="2022"),
            leaf_for(minimal_atoms, "revenue", period="2021"),
        )
        with pytest.raises(SemanticError):
            SemanticAnalyzer(minimal_atoms).analyze(node)

    def test_analyze_ratio_amounts(self, minimal_atoms):
        node = Node("ratio", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "total_assets"))
        result = SemanticAnalyzer(minimal_atoms).analyze(node)
        assert result.meaning.kind == "ratio_amounts"
        assert result.meaning.semantic_type == SemanticType.ratio

    def test_analyze_ratio_avg_over_periods(self, minimal_atoms):
        leaves = [leaf_for(minimal_atoms, "revenue", period=period) for period in ("2021", "2022", "2023")]
        sum_expr = Node("sum", Node("sum", leaves[0], leaves[1]), leaves[2])
        avg_node = Node("ratio", sum_expr, Literal(3.0))
        result = SemanticAnalyzer(minimal_atoms).analyze(avg_node)
        assert result.meaning.kind == "avg_over_all_periods"

    def test_analyze_mul(self, minimal_atoms):
        node = Node("mul", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "income_tax"))
        result = SemanticAnalyzer(minimal_atoms).analyze(node)
        assert result.meaning.kind == "scaled_amount"
        assert result.meaning.semantic_type == SemanticType.amount

    def test_analyze_growth(self, minimal_atoms):
        node = Node(
            "growth",
            leaf_for(minimal_atoms, "revenue", period="2022"),
            leaf_for(minimal_atoms, "revenue", period="2021"),
        )
        result = SemanticAnalyzer(minimal_atoms).analyze(node)
        assert result.meaning.kind == "growth_rate"
        assert result.meaning.semantic_type == SemanticType.ratio

    def test_analyze_min_over_time(self, minimal_atoms):
        node = Node(
            "min",
            leaf_for(minimal_atoms, "revenue", period="2021"),
            leaf_for(minimal_atoms, "revenue", period="2022"),
        )
        result = SemanticAnalyzer(minimal_atoms).analyze(node)
        assert "min" in result.meaning.kind

    def test_analyze_max_over_time(self, minimal_atoms):
        node = Node(
            "max",
            leaf_for(minimal_atoms, "revenue", period="2022"),
            leaf_for(minimal_atoms, "revenue", period="2023"),
        )
        result = SemanticAnalyzer(minimal_atoms).analyze(node)
        assert "max" in result.meaning.kind

    def test_sum_cross_entity_raises(self, minimal_atoms):
        node = Node(
            "sum",
            leaf_for(minimal_atoms, "revenue", entity="Corp0"),
            leaf_for(minimal_atoms, "revenue", entity="Corp1"),
        )
        with pytest.raises(SemanticError):
            SemanticAnalyzer(minimal_atoms).analyze(node)

    def test_diff_cross_unit_raises(self, minimal_atoms):
        node = Node("diff", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "income_tax"))
        with pytest.raises(SemanticError):
            SemanticAnalyzer(minimal_atoms).analyze(node)

    def test_unknown_leaf_key_raises(self, minimal_atoms):
        with pytest.raises(SemanticError, match="Unknown leaf key"):
            SemanticAnalyzer(minimal_atoms).analyze(Leaf(key="does_not_exist"))


class TestQuestionRenderer:
    def test_renders_leaf_question(self, minimal_atoms, monkeypatch):
        monkeypatch.setattr(question_renderer_module, "random", lambda: 0.0)
        result = SemanticAnalyzer(minimal_atoms).analyze(leaf_for(minimal_atoms, "revenue"))
        question = QuestionRenderer().render(result)
        assert question == "What is the revenue for Corp0 in 2021"

    def test_renders_growth_question(self, minimal_atoms, monkeypatch):
        monkeypatch.setattr(question_renderer_module, "random", lambda: 0.0)
        node = Node(
            "growth",
            leaf_for(minimal_atoms, "revenue", period="2022"),
            leaf_for(minimal_atoms, "revenue", period="2021"),
        )
        question = QuestionRenderer().render(SemanticAnalyzer(minimal_atoms).analyze(node))
        assert "growth rate" in question.lower()

    def test_renders_avg_question(self, minimal_atoms, monkeypatch):
        monkeypatch.setattr(question_renderer_module, "random", lambda: 0.0)
        leaves = [leaf_for(minimal_atoms, "cash", period=period) for period in ("2021", "2022", "2023")]
        sum_expr = Node("sum", Node("sum", leaves[0], leaves[1]), leaves[2])
        avg_node = Node("ratio", sum_expr, Literal(3.0))
        question = QuestionRenderer().render(SemanticAnalyzer(minimal_atoms).analyze(avg_node))
        assert "average" in question.lower()

    def test_renders_ratio_question(self, minimal_atoms, monkeypatch):
        monkeypatch.setattr(question_renderer_module, "random", lambda: 0.0)
        node = Node("ratio", leaf_for(minimal_atoms, "revenue"), leaf_for(minimal_atoms, "total_assets"))
        question = QuestionRenderer().render(SemanticAnalyzer(minimal_atoms).analyze(node))
        assert "ratio" in question.lower()

    def test_no_empty_question(self, minimal_atoms, monkeypatch):
        monkeypatch.setattr(question_renderer_module, "random", lambda: 0.0)
        analyzer = SemanticAnalyzer(minimal_atoms)
        renderer = QuestionRenderer()
        for concept in ("revenue", "cash", "total_assets"):
            question = renderer.render(analyzer.analyze(leaf_for(minimal_atoms, concept)))
            assert question.strip() != ""


class TestCompileTreePayload:
    def make_payload(self, depth: int, seed: int) -> dict:
        tree, _ = Expr.sample_tree_with_rejection(
            max_depth=depth,
            rng=random.Random(seed),
            derived_prob=0.2,
        )
        return {"tree": tree, "derived_concepts": DERIVED_CONCEPTS}

    def test_compile_depth_1(self, minimal_atoms):
        payload = self.make_payload(1, seed=0)
        expr = compile_tree_payload(payload, minimal_atoms, seed=0)
        assert math.isfinite(Evaluator(minimal_atoms).eval(expr))

    def test_compile_depth_3(self, minimal_atoms):
        payload = self.make_payload(3, seed=1)
        expr = compile_tree_payload(payload, minimal_atoms, seed=1)
        assert math.isfinite(Evaluator(minimal_atoms).eval(expr))

    def test_question_is_string(self, minimal_atoms, monkeypatch):
        monkeypatch.setattr(question_renderer_module, "random", lambda: 0.0)
        payload = self.make_payload(2, seed=5)
        expr = compile_tree_payload(payload, minimal_atoms, seed=5)
        question = QuestionRenderer().render(SemanticAnalyzer(minimal_atoms).analyze(expr))
        assert isinstance(question, str) and len(question) > 0

    def test_different_seeds_produce_finite_values(self, minimal_atoms):
        payload = self.make_payload(2, seed=10)
        expr_a = compile_tree_payload(payload, minimal_atoms, seed=10)
        expr_b = compile_tree_payload(payload, minimal_atoms, seed=99)
        evaluator = Evaluator(minimal_atoms)
        assert math.isfinite(evaluator.eval(expr_a))
        assert math.isfinite(evaluator.eval(expr_b))
