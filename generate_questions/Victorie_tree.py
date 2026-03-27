from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import json

# =========================================================
# 1. Data model
# =========================================================

@dataclass(frozen=True)
class Atom: #class named Atom for the leaves
    key: str # unique identifier for the atom
    concept: str #main financial concept this atom represents.
    semantic_type: str # what kind of semantic object this is for the parser
    label: str # Human-readable text for display in questions
    entity: str #Company
    period: str # time period
    unit: str # unit: percent, USD, shares...
    value: float # the numerical value
    parent_concept: Optional[str] = None #optional higher order family it belongs to like for example product revenue \subset of revenue
    role: Optional[str] = None #optional describes role in a larger structure for example numerator, denominator

#define a leaf and an operator with binary input
@dataclass(frozen=True)
class Leaf:
    key: str


@dataclass(frozen=True)
class Node:
    op: str
    left: "Expr"
    right: "Expr"


Expr = Union[Leaf, Node] #expression can be a leaf or a node

#semantic interpretation object:
@dataclass
class Meaning:
    kind: str #tells us what kind of object it is for ex, atom, ratio,.. to use the correct wording in the sentence like percentages... EX difference
    semantic_type: str #“what kind of financial/semantic quantity is this? flow, metric, ratio, currency.. EX currency delta
    text: str #human readable: for example change in Apple revenue in 2024

    #useful when the result still corresponds to a known:
    entity: Optional[str] = None #company
    period: Optional[str] = None # time period
    unit: Optional[str] = None #unit
    concept: Optional[str] = None #concept
    label: Optional[str] = None #human readable label

    # compositional metadata for if by combining it got a new info:
    from_period: Optional[str] = None #beginning of change/comparison
    to_period: Optional[str] = None #end of change/comparison
    components: Optional[List[str]] = None #lists the components/labels that make up the meaning
    target_concept: Optional[str] = None #not quite sure

    # debug / trace
    derivation: Optional[str] = None #explaining where the meaning came from. sum(revenue, net_income)

#object that stores the full analysed tree
@dataclass
class AnalysisResult:
    expr: Expr
    meaning: Meaning
    children: List["AnalysisResult"] #analyzed results for the sub-expressions


# =========================================================
# 2. Parsing
# =========================================================

#parses jason file into a tree
def parse_expr(obj: Any) -> Expr:
    if not isinstance(obj, dict):
        raise ValueError("Expression parts must be JSON objects.")

    if "leaf" in obj:
        return Leaf(key=str(obj["leaf"]))

    if {"op", "left", "right"}.issubset(obj.keys()):
        op = str(obj["op"])
        if op not in {"sum", "diff", "ratio", "mul"}:
            raise ValueError(f"Unsupported op: {op}")
        return Node(
            op=op,
            left=parse_expr(obj["left"]),
            right=parse_expr(obj["right"]),
        )

    raise ValueError("Use either {'leaf': '...'} or {'op': ..., 'left': ..., 'right': ...}.")

#helps print the tree as a readable string
def show_expr(expr: Expr) -> str:
    if isinstance(expr, Leaf):
        return expr.key
    return f"{expr.op}({show_expr(expr.left)}, {show_expr(expr.right)})"

# sum(sum(a+b)+c) nested becomes sum(a+b+c)
def flatten_sum(expr: Expr) -> List[Leaf]:
    if isinstance(expr, Leaf):
        return [expr]
    if isinstance(expr, Node) and expr.op == "sum":
        return flatten_sum(expr.left) + flatten_sum(expr.right)
    return []

# if we in english have a list of 2 elements say a AND b if more say a,b,c AND d
def oxford_join(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# =========================================================
# 3. Semantic analyzer
# =========================================================

#Create a special kind of error for meaning related issues
class SemanticError(Exception):
    pass

#takes two atoms and checks compatibility of meanings
class SemanticAnalyzer:
    def __init__(self, atoms: Dict[str, Atom]): #passe the dictionary of atoms
        self.atoms = atoms

    def atom(self, key: str) -> Atom: #check if the atom exists
        if key not in self.atoms:
            raise SemanticError(f"Unknown leaf key: {key}")
        return self.atoms[key]

#checks whether two Meaning objects match in all three dimensions: entity, period, unit or only in two for more flexibility
    def same_context(self, a: Meaning, b: Meaning) -> bool:
        return a.entity == b.entity and a.period == b.period and a.unit == b.unit

    def same_entity_and_unit(self, a: Meaning, b: Meaning) -> bool:
        return a.entity == b.entity and a.unit == b.unit
#Recursive: walks the expression tree and, for each node, computes a semantic interpretation of that subtree.

    def analyze(self, expr: Expr) -> AnalysisResult:
        if isinstance(expr, Leaf): #if its a leaf take directly all the attributes of the atom and run Meaning on it
            atom = self.atom(expr.key)
            meaning = Meaning(
                kind="leaf_metric",
                semantic_type=atom.semantic_type,
                text=f"{atom.label} for {atom.entity} in {atom.period}",
                entity=atom.entity,
                period=atom.period,
                unit=atom.unit,
                concept=atom.concept,
                label=atom.label,
                target_concept=atom.parent_concept,
                components=[atom.label] if atom.role == "component" else None,
                derivation=f"leaf {atom.key}",
            )
            return AnalysisResult(expr=expr, meaning=meaning, children=[]) #then wrap it into analysis result by saying the expression and its meaning

        left_result = self.analyze(expr.left) #if its not a leaf be recursive and run it in the left subtree and right one
        right_result = self.analyze(expr.right)

        if expr.op == "sum":
            meaning = self._analyze_sum(expr, left_result, right_result) #if operator is ... do the analysis specific to that operator
        elif expr.op == "diff":
            meaning = self._analyze_diff(expr, left_result, right_result)
        elif expr.op == "ratio":
            meaning = self._analyze_ratio(expr, left_result, right_result)
        elif expr.op == "mul":
            meaning = self._analyze_mul(expr, left_result, right_result)
        else:
            raise SemanticError(f"Unsupported op: {expr.op}")

        return AnalysisResult(expr=expr, meaning=meaning, children=[left_result, right_result]) # wrap the result nicely

    def _analyze_sum(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning: #for a sum how do we analyse?
    #A parent metric reconstructed from sibling components like Product revenue + Services revenue = Revenue
    # or A generic combination of two compatible amounts like cash + equivalents
        lm, rm = left.meaning, right.meaning

        leaves = flatten_sum(expr) #flattens everything to not just compare pairwise
        if leaves:
            atoms = [self.atom(leaf.key) for leaf in leaves] #make atoms out of the leaves
            first = atoms[0] #using the first leaf as a reference, compare all the others and check if they match to see if they are siblings
            if (
                len(atoms) >= 2
                and first.semantic_type == "amount"
                and first.parent_concept is not None
                and first.role == "component"
                and all(
                    a.semantic_type == "amount"
                    and a.parent_concept == first.parent_concept
                    and a.role == "component"
                    and a.entity == first.entity
                    and a.period == first.period
                    and a.unit == first.unit
                    for a in atoms[1:]
                )
            ): #if so,return the parent concept as the meaning and say that you just summed up all the components
                component_labels = [a.label for a in atoms]
                return Meaning(
                    kind="aggregate_components",
                    semantic_type="amount",
                    text=f"{first.parent_concept} for {first.entity} in {first.period}", #
                    entity=first.entity,
                    period=first.period,
                    unit=first.unit,
                    concept=first.concept,
                    label=first.parent_concept,
                    target_concept=first.parent_concept,
                    components=component_labels,
                    derivation=f"sum of sibling components: {', '.join(component_labels)}",
                )
# else we are less strict, as long as both are amounts and both have the same context (all 3 match)
        if lm.semantic_type == rm.semantic_type == "amount" and self.same_context(lm, rm):
            concept = None
            if lm.concept == rm.concept:
                concept = lm.concept #keep the concept since it still applies to the sum
                # else the concept gets lost
            label = f"combined {lm.label} and {rm.label}" if lm.label and rm.label else "combined value" #makes the english sentence with the label if it still applies else jsut say combined value
            return Meaning(
                kind="sum_amount",
                semantic_type="amount",
                text=f"combined value of {lm.text} and {rm.text}",
                entity=lm.entity,
                period=lm.period,
                unit=lm.unit,
                concept=concept,
                label=label,
                derivation=f"sum({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot sum these meanings: {lm.kind} and {rm.kind}") #else reject the sum

#other operators similar ...
    def _analyze_diff(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning

        if (
            lm.semantic_type == rm.semantic_type == "amount"
            and lm.concept == "assets"
            and rm.concept == "liabilities"
            and self.same_context(lm, rm)
        ):
            return Meaning(
                kind="named_identity",
                semantic_type="amount",
                text=f"equity of {lm.entity} in {lm.period}",
                entity=lm.entity,
                period=lm.period,
                unit=lm.unit,
                concept="equity",
                label="equity",
                derivation="assets minus liabilities -> equity",
            )

        if (
            lm.semantic_type == rm.semantic_type == "amount"
            and lm.concept == rm.concept
            and self.same_entity_and_unit(lm, rm)
            and lm.period != rm.period
        ):
            return Meaning(
                kind="change_over_time",
                semantic_type="amount",
                text=f"change in {lm.label or lm.concept} for {lm.entity} from {rm.period} to {lm.period}",
                entity=lm.entity,
                unit=lm.unit,
                concept=lm.concept,
                label=lm.label or lm.concept,
                from_period=rm.period,
                to_period=lm.period,
                derivation=f"diff over time: {rm.period} -> {lm.period}",
            )

        if lm.semantic_type == rm.semantic_type == "amount" and self.same_context(lm, rm):
            return Meaning(
                kind="difference_amount",
                semantic_type="amount",
                text=f"difference between {lm.text} and {rm.text}",
                entity=lm.entity,
                period=lm.period,
                unit=lm.unit,
                concept=lm.concept if lm.concept == rm.concept else None,
                label=f"difference between {lm.label} and {rm.label}",
                derivation=f"diff({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot subtract these meanings: {lm.kind} and {rm.kind}")

    def _analyze_ratio(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning

        if (
            lm.kind == "change_over_time"
            and rm.semantic_type == "amount"
            and lm.concept == rm.concept
            and lm.entity == rm.entity
            and lm.unit == rm.unit
            and lm.from_period == rm.period
        ):
            return Meaning(
                kind="growth_rate",
                semantic_type="ratio",
                text=f"growth rate of {rm.label or rm.concept} for {rm.entity} from {lm.from_period} to {lm.to_period}",
                entity=rm.entity,
                unit=rm.unit,
                concept=rm.concept,
                label=f"growth rate of {rm.label or rm.concept}",
                from_period=lm.from_period,
                to_period=lm.to_period,
                derivation="change over time divided by base period",
            )

        if lm.semantic_type == rm.semantic_type == "amount" and self.same_context(lm, rm):
            return Meaning(
                kind="ratio_amounts",
                semantic_type="ratio",
                text=f"ratio of {lm.text} to {rm.text}",
                entity=lm.entity,
                period=lm.period,
                unit=lm.unit,
                concept=lm.concept if lm.concept == rm.concept else None,
                label=f"ratio of {lm.label} to {rm.label}",
                derivation=f"ratio({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot divide these meanings: {lm.kind} and {rm.kind}")

    def _analyze_mul(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning

        if lm.semantic_type == "amount" and rm.semantic_type in {"ratio", "rate"} and lm.entity == rm.entity:
            return Meaning(
                kind="scaled_amount",
                semantic_type="amount",
                text=f"{lm.text} scaled by {rm.text}",
                entity=lm.entity,
                period=lm.period or rm.period,
                unit=lm.unit,
                concept=lm.concept,
                label=lm.label,
                derivation=f"mul({lm.kind}, {rm.kind})",
            )

        if rm.semantic_type == "amount" and lm.semantic_type in {"ratio", "rate"} and lm.entity == rm.entity:
            return Meaning(
                kind="scaled_amount",
                semantic_type="amount",
                text=f"{rm.text} scaled by {lm.text}",
                entity=rm.entity,
                period=rm.period or lm.period,
                unit=rm.unit,
                concept=rm.concept,
                label=rm.label,
                derivation=f"mul({lm.kind}, {rm.kind})",
            )

        if lm.semantic_type == rm.semantic_type == "amount":
            return Meaning(
                kind="product_amounts",
                semantic_type="amount",
                text=f"product of {lm.text} and {rm.text}",
                entity=lm.entity if lm.entity == rm.entity else None,
                unit=lm.unit if lm.unit == rm.unit else None,
                label="product",
                derivation=f"mul({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot multiply these meanings: {lm.kind} and {rm.kind}")
    


# =========================================================
# 4. Renderer + numeric evaluator
# =========================================================

#not relevant for the compiler it just gives the answer following the tree programm
class Evaluator:
    def __init__(self, atoms: Dict[str, Atom]):
        self.atoms = atoms

    def eval(self, expr: Expr) -> float:
        if isinstance(expr, Leaf):
            return self.atoms[expr.key].value

        lv = self.eval(expr.left)
        rv = self.eval(expr.right)

        if expr.op == "sum":
            return lv + rv
        if expr.op == "diff":
            return lv - rv
        if expr.op == "ratio":
            if rv == 0:
                raise ZeroDivisionError("Division by zero.")
            return lv / rv
        if expr.op == "mul":
            return lv * rv

        raise ValueError(f"Unsupported op: {expr.op}")

# makes the questions grammatically correct using the nouns provided by meaning
class QuestionRenderer:
    def render(self, meaning: Meaning) -> str:
        if meaning.kind == "aggregate_components":
            return (
                f"What is the {meaning.target_concept} for {meaning.entity} in {meaning.period} "
                f"from {oxford_join(meaning.components or [])} combined?"
            )

        if meaning.kind == "named_identity":
            return f"What is the equity of {meaning.entity} in {meaning.period}?"

        if meaning.kind == "growth_rate":
            return (
                f"What is the growth rate of {meaning.label.replace('growth rate of ', '')} "
                f"for {meaning.entity} from {meaning.from_period} to {meaning.to_period}?"
            )

        if meaning.kind == "change_over_time":
            return (
                f"What is the change in {meaning.label} for {meaning.entity} "
                f"from {meaning.from_period} to {meaning.to_period}?"
            )

        if meaning.kind == "ratio_amounts":
            return f"What is the ratio of {meaning.text.split('ratio of ',1)[1]}?"

        if meaning.kind == "sum_amount":
            return f"What is the combined value described by this expression?"

        if meaning.kind == "difference_amount":
          return (
              f"What is the difference between "
              f"{meaning.text.split('difference between ',1)[1]}?"
          )

        #if meaning.kind == "difference_amount":
        #    return f"What is the difference described by this expression?"

        if meaning.kind == "scaled_amount":
            return f"What is the scaled amount described by this expression?"

        if meaning.kind == "product_amounts":
            return f"What is the product described by this expression?"

        if meaning.kind == "leaf_metric":
            return f"What is {meaning.text}?"

        raise SemanticError(f"No renderer for meaning kind: {meaning.kind}")
    
# =========================================================
# 5. Trace printer
# =========================================================

def print_analysis_tree(result: AnalysisResult, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}Expr:     {show_expr(result.expr)}")
    print(f"{prefix}Kind:     {result.meaning.kind}")
    print(f"{prefix}Type:     {result.meaning.semantic_type}")
    print(f"{prefix}Meaning:  {result.meaning.text}")
    if result.meaning.derivation:
        print(f"{prefix}How:      {result.meaning.derivation}")
    print()
    for child in result.children:
        print_analysis_tree(child, indent + 1)