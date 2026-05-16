from __future__ import annotations

from random import random


from semantic_analyzer import (
    AnalysisResult,
    _base_metric_name,
    _base_metric_name_from_child,
    _both_have_entity_period_suffix,
    _clean_label,
    _question_copula,
    _shared_entity_period_context,
    _strip_entity_period_suffix,
)
from tree import DerivedExpr, Leaf, Literal, Operation

QUESTION_TEMPLATES = {
    "question": [
        "What is the {phrase}",
        "What is the value of the {phrase}",
        "What's the value of the {phrase}",
        "Whats the value of the {phrase}",
        "What would be the {phrase}",
        "What do we obtain as the {phrase}",
        "What does the following give: {phrase}",
        "Which value corresponds to the {phrase}",
        "Which result follows from the {phrase}",
        "Which quantity is given by the {phrase}",
        "Could you give the {phrase}",
        "May you calculate the {phrase}",
        "May you give the {phrase}",
    ],
    "imperative": [
        "Give the {phrase}",
        "Provide the {phrase}",
        "State the {phrase}",
        "Determine the {phrase}",
        "Compute the {phrase}",
        "Calculate the {phrase}",
        "Evaluate the {phrase}",
        "You should calculate the {phrase}",
        "You should give the {phrase}",
    ]
}


class QuestionRenderer:
    def render(self, result: AnalysisResult) -> str:
        m = result.meaning

        # --- specialized semantic kinds unchanged except for template insertion support ---
        if m.kind == "aggregate_components":
            copula = _question_copula(m.target_concept)
            phrase = (
                f"total {_clean_label(m.target_concept)} "
                f"for {m.entity} in {m.period}"
            )
            return self._apply_template(phrase=phrase, copula=copula)

        if m.kind == "growth_rate":
            label = _clean_label((m.label or '').replace('growth rate of ', ''))
            phrase = f"growth rate of {label} for {m.entity} from {m.from_period} to {m.to_period}"
            copula = "is"
            return self._apply_template(phrase=phrase, copula=copula)

        if m.kind == "avg_over_all_periods":
            label = _clean_label(m.label or m.concept)
            phrase = f"average {label} for {m.entity} across years {m.from_period} through {m.to_period}"
            copula = "is"
            return self._apply_template(phrase=phrase, copula=copula)

        # --- default ---
        phrase = self._expr_phrase(result, top_level=True)
        copula = _question_copula(m.label or m.concept)
        return self._apply_template(phrase=phrase, copula=copula)

    def _apply_template(self, *, phrase: str, copula: str) -> str:
        # Flatten templates keeping category information
        all_templates = (
            [("question", t) for t in QUESTION_TEMPLATES["question"]] +
            [("imperative", t) for t in QUESTION_TEMPLATES["imperative"]]
        )

        category, tmpl = all_templates[int(random() * len(all_templates))]

        # Perform substitution
        text = tmpl.format(phrase=phrase, copula=copula)

        # Add punctuation
        punct_case = ["none", "space", "normal"][int(random() * 3)]

        if category == "question":
            punct = "?"
        else:
            punct = "."

        stripped = text.rstrip()

        if punct_case == "none":
            result = stripped
        elif punct_case == "space":
            result = stripped + " " + punct
        else:  # "normal"
            result = stripped + punct

        # Randomize capitalization (uniform choice)
        cap_case = ["capitalize", "lower"][int(random() * 2)]

        if result:
            if cap_case == "lower":
                result = result[0].lower() + result[1:]
            else:
                result = result[0].upper() + result[1:]

        return result

    def _expr_phrase(self, result: AnalysisResult, top_level: bool = False) -> str:
        expr = result.expr
        m = result.meaning

        if isinstance(expr, Leaf):
            return f"{_clean_label(m.label)} for {m.entity} in {m.period}"

        if isinstance(expr, DerivedExpr):
            return f"{_clean_label(m.label)} for {m.entity} in {m.period}"

        if isinstance(expr, Literal):
            return str(expr.value)

        left = self._expr_phrase(result.children[0])
        right = self._expr_phrase(result.children[1])
        left_meaning = result.children[0].meaning
        right_meaning = result.children[1].meaning
        match expr.op:
            case Operation.sum:
                shared_context = _shared_entity_period_context(left_meaning, right_meaning)
                if shared_context is not None and _both_have_entity_period_suffix(left, right, *shared_context):
                    entity, period = shared_context
                    left = _strip_entity_period_suffix(left, entity, period)
                    right = _strip_entity_period_suffix(right, entity, period)
                    return f"sum of {left} and {right} for {entity} in {period}"
                return f"sum of {left} and {right}"

            case Operation.diff:
                shared_context = _shared_entity_period_context(left_meaning, right_meaning)
                if shared_context is not None and _both_have_entity_period_suffix(left, right, *shared_context):
                    entity, period = shared_context
                    left = _strip_entity_period_suffix(left, entity, period)
                    right = _strip_entity_period_suffix(right, entity, period)
                    return f"difference between {left} and {right} for {entity} in {period}"
                return f"difference between {left} and {right}"

            case Operation.ratio:
                shared_context = _shared_entity_period_context(left_meaning, right_meaning)
                if shared_context is not None and _both_have_entity_period_suffix(left, right, *shared_context):
                    entity, period = shared_context
                    left = _strip_entity_period_suffix(left, entity, period)
                    right = _strip_entity_period_suffix(right, entity, period)
                    return f"ratio of {left} to {right} for {entity} in {period}"
                return f"ratio of {left} to {right}"

            case Operation.mul:
                shared_context = _shared_entity_period_context(left_meaning, right_meaning)
                if shared_context is not None and _both_have_entity_period_suffix(left, right, *shared_context):
                    entity, period = shared_context
                    left = _strip_entity_period_suffix(left, entity, period)
                    right = _strip_entity_period_suffix(right, entity, period)
                    return f"result of {left} scaled by {right} for {entity} in {period}"
                return f"result of {left} scaled by {right}"

            case Operation.growth:
                if m.kind == "growth_rate" and m.concept is not None:
                    return (
                        f"growth in {_base_metric_name(m)} for {m.entity} "
                        f"from {m.from_period} to {m.to_period}"
                    )
                return f"growth between {left} and {right}"

            case Operation.min | Operation.max:
                op_word = {"min": "minimum", "max": "maximum"}[expr.op.value]

                if m.kind == f"{expr.op.value}_over_time" and m.concept is not None:
                    base_name = _base_metric_name_from_child(result.children[0].meaning)
                    return (
                        f"the {op_word} of {base_name} for {m.entity} "
                        f"using values from {m.from_period} through {m.to_period}"
                    )

                return f"the {op_word} of {left} and {right}"

        return m.text
