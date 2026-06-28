from __future__ import annotations

import json
from pathlib import Path

from obstacles import (
    GENERATION_OBSTACLES,
    QUESTION_OBSTACLES,
    ObstacleContext,
    concepts_in_expr,
    pick_useless_info_family,
    validate_useless_info_template_coverage,
)
from semantic_analyzer import AnalysisResult, Meaning
from tree import DerivedExpr, Leaf, Node, SemanticType


def _analysis_for(expr) -> AnalysisResult:
    meaning = Meaning(
        kind="leaf_metric",
        semantic_type=SemanticType.amount,
        text="dummy",
        entity="Corp0",
        period="2021",
        unit="M_USD",
        concept="revenue",
        label="revenue",
    )
    return AnalysisResult(expr=expr, meaning=meaning)


def test_concepts_in_expr_collects_leaf_concepts(minimal_atoms):
    expr = Node("ratio", Leaf("0_revenue"), Leaf("0_income_tax"))
    leaf_atoms = [minimal_atoms["0_revenue"], minimal_atoms["0_income_tax"]]

    concepts = concepts_in_expr(expr, leaf_atoms)

    assert "revenue" in concepts
    assert "income_tax" in concepts


def test_concepts_in_expr_collects_derived_names(minimal_atoms):
    inner = Node("diff", Leaf("0_revenue"), Leaf("0_cost_of_goods_sold"))
    expr = DerivedExpr("gross_profit", inner)
    leaf_atoms = [minimal_atoms["0_revenue"], minimal_atoms["0_cost_of_goods_sold"]]

    concepts = concepts_in_expr(expr, leaf_atoms)

    assert "gross_profit" in concepts
    assert "revenue" in concepts
    assert "cost_of_goods_sold" in concepts


def test_pick_useless_info_family_prefers_eligible_concepts():
    families = {"income_tax": ["a"], "revenue": ["b"], "generic": ["c"]}
    concepts = frozenset({"income_tax", "foo"})

    chosen = pick_useless_info_family(families, concepts)

    assert chosen == "income_tax"


def test_pick_useless_info_family_fallbacks_to_generic():
    families = {"revenue": ["b"], "generic": ["c"]}
    concepts = frozenset({"income_tax"})

    chosen = pick_useless_info_family(families, concepts)

    assert chosen == "generic"


def test_templates_cover_all_required_concepts(v2_dir: Path):
    validate_useless_info_template_coverage(
        v2_dir / "config/useless_info_templates.json"
    )


def test_pick_useless_info_family_prefers_question_mentioned_concept():
    families = {
        "current_liabilities": ["a"],
        "short_term_investments": ["b"],
        "generic": ["g"],
    }
    concepts = frozenset({"current_liabilities", "short_term_investments"})

    chosen = pick_useless_info_family(
        families,
        concepts,
        question=(
            "what does the following give: sum of current liabilities and "
            "short term investments for Crimson in 2021?"
        ),
    )

    assert chosen in {"current_liabilities", "short_term_investments"}
    assert chosen != "generic"


def test_pick_useless_info_family_prefers_tax_when_question_mentions_tax():
    families = {"income_tax": ["tax"], "dividends_paid": ["div"], "generic": ["g"]}
    concepts = frozenset({"income_tax", "dividends_paid"})

    chosen = pick_useless_info_family(
        families,
        concepts,
        question="Could you give the result of dividends paid scaled by income tax?",
    )

    assert chosen == "income_tax"


def test_apply_useless_info_matches_income_tax_family(tmp_path: Path, minimal_atoms):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    templates_path = config_dir / "useless_info_templates.json"
    templates = {
        "income_tax": ["Ignore future tax changes and use reported tax values only."],
        "revenue": ["Ignore future revenue scenarios and use reported sales only."],
        "generic": ["Use reported values only."],
    }
    templates_path.write_text(json.dumps(templates), encoding="utf-8")

    expr = Node("ratio", Leaf("0_revenue"), Leaf("0_income_tax"))
    ctx = ObstacleContext(
        question="What is the ratio of revenue to income tax?",
        analysis=_analysis_for(expr),
        expr=expr,
        leaf_atoms=[minimal_atoms["0_revenue"], minimal_atoms["0_income_tax"]],
        spreadsheet_rows=[],
        base_dir=tmp_path,
    )

    result = ctx.apply_useless_info()

    assert "tax" in result.lower()
    assert ctx.useless_info_family_used == "income_tax"


def test_balanced_tree_is_generation_obstacle_not_question_obstacle():
    assert "balanced_tree" in GENERATION_OBSTACLES
    assert "balanced_tree" not in QUESTION_OBSTACLES
