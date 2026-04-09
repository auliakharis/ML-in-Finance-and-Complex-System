from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# =========================================================
# 1. Data model
# =========================================================

@dataclass(frozen=True)
class Atom:
    key: str
    concept: str
    semantic_type: str
    label: str
    entity: str
    period: str
    unit: str
    value: float
    depth: int = 0
    parent_concept: Optional[str] = None
    role: Optional[str] = None


@dataclass(frozen=True)
class Leaf:
    key: str


@dataclass(frozen=True)
class Node:
    op: str
    left: "Expr"
    right: "Expr"
    depth: Optional[int] = None


@dataclass(frozen=True)
class DerivedExpr:
    name: str
    expr: "Expr"
    depth: Optional[int] = None


@dataclass(frozen=True)
class Literal:
    """Floating-point constant (e.g. count of years) for ratio(sum, n) → arithmetic mean."""

    value: float


Expr = Union[Leaf, Node, DerivedExpr, Literal]
SUPPORTED_OPS = {"sum", "diff", "ratio", "mul", "growth", "min", "max", "avg"}

# counts the depth of an expression by recursively counting the depth of the left and right subtrees.
def expr_depth(expr: Expr) -> int:
    if isinstance(expr, Literal):
        return 0
    if isinstance(expr, (Leaf, DerivedExpr)):
        return 0 if isinstance(expr, Leaf) else (expr.depth if expr.depth is not None else expr_depth(expr.expr))
    return 1 + max(expr_depth(expr.left), expr_depth(expr.right))


# semantic interpretation object: a Meaning object is created for each expression to describe its meaning.
@dataclass
class Meaning:
    kind: str
    semantic_type: str
    text: str
    entity: Optional[str] = None
    period: Optional[str] = None
    unit: Optional[str] = None
    concept: Optional[str] = None
    label: Optional[str] = None
    from_period: Optional[str] = None
    to_period: Optional[str] = None
    components: Optional[List[str]] = None
    target_concept: Optional[str] = None
    derivation: Optional[str] = None


@dataclass #a AnalysisResult object is created for each expression to describe its analysis, including the expression itself, its meaning, and its children.
class AnalysisResult:
    expr: Expr
    meaning: Meaning
    children: List["AnalysisResult"]
    depth: int


# =========================================================
# 2. Parsing / serialization
# =========================================================
# parses a JSON object into an expression.
def parse_expr(obj: Any) -> Expr:
    if not isinstance(obj, dict):
        raise ValueError("Expression parts must be JSON objects.")

    if "leaf" in obj:
        return Leaf(key=str(obj["leaf"]))

    if "literal" in obj:
        return Literal(value=float(obj["literal"]))

    if "derived" in obj and "expanded" in obj:
        expanded = parse_expr(obj["expanded"])
        return DerivedExpr(
            name=str(obj["derived"]),
            expr=expanded,
            depth=int(obj.get("depth", expr_depth(expanded))),
        )

    if {"op", "left", "right"}.issubset(obj.keys()):
        op = str(obj["op"])
        if op not in SUPPORTED_OPS:
            raise ValueError(f"Unsupported op: {op}")
        left = parse_expr(obj["left"])
        right = parse_expr(obj["right"])
        parsed_depth = obj.get("depth")
        computed_depth = 1 + max(expr_depth(left), expr_depth(right))
        if parsed_depth is not None and int(parsed_depth) != computed_depth:
            raise ValueError(
                f"Depth mismatch for node {op}: provided {parsed_depth}, computed {computed_depth}"
            )
        return Node(op=op, left=left, right=right, depth=computed_depth)

    raise ValueError(
        "Use {'leaf': ...}, {'literal': ...}, {'derived': ..., 'expanded': ...}, "
        "or {'op': ..., 'left': ..., 'right': ...}."
    )

# converts an expression to a JSON object.
def expr_to_json(expr: Expr) -> Dict[str, Any]:
    if isinstance(expr, Leaf):
        return {"leaf": expr.key, "depth": 0}
    if isinstance(expr, Literal):
        return {"literal": expr.value, "depth": 0}
    if isinstance(expr, DerivedExpr):
        return {
            "derived": expr.name,
            "expanded": expr_to_json(expr.expr),
            "depth": expr.depth if expr.depth is not None else expr_depth(expr.expr),
        }
    return {
        "op": expr.op,
        "left": expr_to_json(expr.left),
        "right": expr_to_json(expr.right),
        "depth": expr.depth if expr.depth is not None else expr_depth(expr),
    }

# converts an expression to a string representation.
def show_expr(expr: Expr) -> str:
    if isinstance(expr, Leaf):
        return expr.key
    if isinstance(expr, Literal):
        return str(expr.value)
    if isinstance(expr, DerivedExpr):
        return expr.name
    return f"{expr.op}({show_expr(expr.left)}, {show_expr(expr.right)})"

# flattens a sum expression into a list of leaves.
def flatten_sum(expr: Expr) -> List[Leaf]:
    if isinstance(expr, Leaf):
        return [expr]
    if isinstance(expr, DerivedExpr):
        return []
    if isinstance(expr, Node) and expr.op == "sum":
        return flatten_sum(expr.left) + flatten_sum(expr.right)
    return []

# flattens an expression into a list of leaves.
def flatten_leaves(expr: Expr) -> List[Leaf]:
    if isinstance(expr, Leaf):
        return [expr]
    if isinstance(expr, Literal):
        return []
    if isinstance(expr, DerivedExpr):
        return flatten_leaves(expr.expr)
    return flatten_leaves(expr.left) + flatten_leaves(expr.right)

# joins a list of strings with a comma and "and" or "or" depending on the length of the list.
def oxford_join(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


# =========================================================
# 3. Instantiation from typed sampler trees
# =========================================================
# a SemanticError exception is raised when there is an error in the semantic interpretation of an expression.
class SemanticError(Exception):
    pass


DERIVED_FORMULAS: Dict[str, Dict[str, Any]] = {
    "gross_profit": {"op": "diff", "args": ["revenue", "cost_of_goods_sold"]},
    "operating_income": {"op": "diff", "args": ["gross_profit", "operating_expenses"]},
    "pretax_income": {"op": "diff", "args": ["operating_income", "non_operating_expenses"]},
    "income_tax_expense": {"op": "mul", "args": ["pretax_income", "income_tax"]},
    "net_income": {"op": "diff", "args": ["pretax_income", "income_tax_expense"]},
    "current_assets": {
        "op": "sum",
        "args": ["cash", "accounts_receivable", "inventories", "short_term_investments"],
    },
    "longterm_assets": {"op": "diff", "args": ["total_assets", "current_assets"]},
    "longterm_liabilities": {"op": "diff", "args": ["total_liabilities", "current_liabilities"]},
}

class AtomIndex:
    """Fast lookup over all spreadsheet atoms for template binding.

    Atoms are keyed by ``Atom.key`` in ``self.atoms`` (the authoritative map).
    The index adds **secondary groupings** so binding code can answer questions
    like “which companies exist?”, “which years?”, and “what amount-like concepts
    do we have?” without scanning the full dict every time.

    **Inverted indexes** (each maps a dimension to the list of atoms touching it):

    - ``by_concept`` — all atoms for a metric name (e.g. ``revenue``).
    - ``by_entity`` — all atoms for one company.
    - ``by_period`` — all atoms for one reporting year.

    **Sorted convenience lists** (for random choice when the template does not
    fix entity/concept yet):

    - ``entities``, ``periods``
    - ``amount_concepts`` — concepts whose ``semantic_type`` is ``amount``
    - ``ratio_concepts`` — concepts with ``semantic_type`` in ``ratio`` or ``rate``
      (e.g. effective tax rate for ``mul`` branches)

    Typical use: :meth:`filter_atoms` narrows by any combination of
    ``semantic_types``, ``concept``, ``entity``, and ``period``; then
    ``pick_one(rng, candidates, ...)`` chooses a concrete atom for a leaf.
    """

    def __init__(self, atoms: Dict[str, Atom]) -> None:
        self.atoms = atoms
        self.by_concept: Dict[str, List[Atom]] = {}
        self.by_entity: Dict[str, List[Atom]] = {}
        self.by_period: Dict[str, List[Atom]] = {}
        self.entities: List[str] = sorted({a.entity for a in atoms.values()})
        self.periods: List[str] = sorted({a.period for a in atoms.values()})
        self.amount_concepts: List[str] = sorted({a.concept for a in atoms.values() if a.semantic_type == "amount"})
        self.ratio_concepts: List[str] = sorted({a.concept for a in atoms.values() if a.semantic_type in {"ratio", "rate"}})
        for atom in atoms.values():
            self.by_concept.setdefault(atom.concept, []).append(atom)
            self.by_entity.setdefault(atom.entity, []).append(atom)
            self.by_period.setdefault(atom.period, []).append(atom)

    def filter_atoms(
        self,
        *,
        semantic_types: Optional[Sequence[str]] = None,
        concept: Optional[str] = None,
        entity: Optional[str] = None,
        period: Optional[str] = None,
    ) -> List[Atom]:
        """Return atoms matching all *provided* filters; omitted filters are ignored."""
        pool = list(self.atoms.values())
        if concept is not None:
            pool = [a for a in pool if a.concept == concept]
        if entity is not None:
            pool = [a for a in pool if a.entity == entity]
        if period is not None:
            pool = [a for a in pool if a.period == period]
        if semantic_types is not None:
            allowed = set(semantic_types)
            pool = [a for a in pool if a.semantic_type in allowed]
        return pool


@dataclass #a BindEnv object is created to bind the entity, period, and concept of an expression.
class BindEnv:
    entity: Optional[str] = None
    period: Optional[str] = None
    concept: Optional[str] = None

# loads a JSON file into a dictionary of atoms.
def load_atoms_json(path: str | Path) -> Dict[str, Atom]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    allowed = {
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

    out: Dict[str, Atom] = {}
    for item in raw:
        clean = {k: v for k, v in item.items() if k in allowed}
        atom = Atom(**clean)
        out[atom.key] = atom
    return out

# picks a random item from a list.
def pick_one(rng: random.Random, items: Sequence[Any], what: str) -> Any:
    if not items:
        raise SemanticError(f"No candidates available for {what}.")
    return rng.choice(list(items))

# creates a node with the given operator and left and right subtrees.
def with_depth(op: str, left: Expr, right: Expr) -> Node:
    return Node(op=op, left=left, right=right, depth=1 + max(expr_depth(left), expr_depth(right)))

# folds a list of expressions into a single expression using the given operator.
def fold_nary(op: str, args: List[Expr]) -> Expr:
    if not args:
        raise ValueError("Cannot fold empty argument list.")
    current = args[0]
    for nxt in args[1:]:
        current = with_depth(op, current, nxt)
    return current

# chooses an amount concept from the list of amount concepts.
def choose_amount_concept(index: AtomIndex, rng: random.Random, env: BindEnv) -> str:
    if env.concept is not None:
        return env.concept
    return pick_one(rng, index.amount_concepts, "amount concept")

def choose_ratio_concept(index: AtomIndex, rng: random.Random) -> str:
    if not index.ratio_concepts:
        raise SemanticError("No ratio/rate atoms were found. You need at least one rate-like concept such as income_tax.")
    return pick_one(rng, index.ratio_concepts, "ratio concept")


def choose_entity(index: AtomIndex, rng: random.Random, env: BindEnv) -> str:
    if env.entity is not None:
        return env.entity
    return pick_one(rng, index.entities, "entity")


def choose_period(index: AtomIndex, rng: random.Random, env: BindEnv) -> str:
    if env.period is not None:
        return env.period
    return pick_one(rng, index.periods, "period")

# instantiates a base atom from the given index, random number generator, semantic types, and environment.
# A base atom is a leaf node in the expression tree that corresponds to a single atom from the spreadsheet.
def instantiate_base_atom(
    index: AtomIndex,
    rng: random.Random,
    *,
    semantic_types: Sequence[str],
    env: BindEnv,
) -> Leaf:
    # Resolve entity/period first so all downstream picks stay context-consistent.
    entity = choose_entity(index, rng, env)
    period = choose_period(index, rng, env)

    # Choose concept family-aware: amount concepts vs ratio/rate concepts.
    if any(t in {"amount"} for t in semantic_types):
        concept = choose_amount_concept(index, rng, env)
    else:
        concept = env.concept or choose_ratio_concept(index, rng)

    # Filter to matching atoms and bind one concrete leaf.
    candidates = index.filter_atoms(
        semantic_types=semantic_types,
        concept=concept,
        entity=entity,
        period=period,
    )
    atom = pick_one(rng, candidates, f"atom concept={concept} entity={entity} period={period}")
    return Leaf(key=atom.key)
# instantiates a derived concept from the given name, index, random number generator, environment, derived registry, and preserve named derived flag.
# A derived concept is a node in the expression tree that corresponds to a derived concept from the spreadsheet.
def instantiate_formula_reference(
    name: str,
    index: AtomIndex,
    rng: random.Random,
    env: BindEnv,
    derived_registry: Dict[str, Dict[str, Any]],
    *,
    preserve_named_derived: bool = True,
) -> Expr:
    """Expand a registry derived name into sub-expressions, or bind a base atom if ``name`` is primitive."""
    # One company-year for the whole expansion: operands must refer to the same slice.
    bound_env = BindEnv(
        entity=choose_entity(index, rng, env) if env.entity is None else env.entity,
        period=choose_period(index, rng, env) if env.period is None else env.period,
        concept=env.concept,
    )
    # Formula args like "revenue" are base metrics, not registry keys: bind one atom.
    if name not in derived_registry:
        return instantiate_base_atom(
            index,
            rng,
            semantic_types=["amount"],
            env=BindEnv(
                entity=bound_env.entity,
                period=bound_env.period,
                concept=name,
            ),
        )

    formula = derived_registry[name]
    op = formula["op"]

    args = [
        instantiate_formula_reference(
            arg,
            index,
            rng,
            bound_env,
            derived_registry,
            preserve_named_derived=preserve_named_derived,
        )
        for arg in formula["args"]
    ]

    if op == "sum":
        expanded = fold_nary("sum", args)
    elif op in {"diff", "ratio", "mul"}:
        if len(args) != 2:
            raise ValueError(f"Operator {op} expects exactly 2 args in derived formula {name}.")
        expanded = with_depth(op, args[0], args[1])
    else:
        raise ValueError(f"Unsupported derived formula op: {op}")

    if preserve_named_derived:
        return DerivedExpr(name=name, expr=expanded, depth=expr_depth(expanded))
    return expanded

#used for time aggregates when it is still binary
def _pick_entity_concept_two_periods(
    index: AtomIndex,
    rng: random.Random,
    env: BindEnv,
    *,
    purpose: str,
) -> Tuple[str, str, str, str]:
    """Pick one amount metric and two random distinct years for the same company."""
    entity = choose_entity(index, rng, BindEnv(entity=env.entity))
    concept = choose_amount_concept(index, rng, BindEnv(concept=env.concept))
    concept_period_atoms = index.filter_atoms(semantic_types=["amount"], concept=concept, entity=entity)
    periods = sorted({a.period for a in concept_period_atoms})
    if len(periods) < 2:
        raise SemanticError(
            f"Need at least two periods for concept={concept}, entity={entity} ({purpose})."
        )
    p_left, p_right = rng.sample(periods, 2)
    return entity, concept, p_left, p_right

#built for time aggregates over multiple periods
def _pick_entity_concept_all_periods(
    index: AtomIndex,
    rng: random.Random,
    env: BindEnv,
    *,
    purpose: str,
) -> Tuple[str, str, List[str]]:
    """Pick one amount metric and every reporting year available for that company."""
    entity = choose_entity(index, rng, BindEnv(entity=env.entity))
    concept = choose_amount_concept(index, rng, BindEnv(concept=env.concept))
    concept_period_atoms = index.filter_atoms(semantic_types=["amount"], concept=concept, entity=entity)
    periods = sorted({a.period for a in concept_period_atoms})
    if len(periods) < 2:
        raise SemanticError(
            f"Need at least two periods for concept={concept}, entity={entity} ({purpose})."
        )
    return entity, concept, periods

#"Min over N years" is represented as N−1 nested binary nodes; this helper builds that tree.
def _fold_binary_op(op: str, parts: List[Expr]) -> Expr:
    """Left-fold binary op over two or more sub-expressions (same semantics as nested nodes)."""
    if len(parts) < 2:
        raise ValueError(f"{op} needs at least 2 operands, got {len(parts)}")
    acc = parts[0]
    for nxt in parts[1:]:
        acc = with_depth(op, acc, nxt)
    return acc


def _bind_min_or_max_over_all_periods(
    op: str,
    index: AtomIndex,
    rng: random.Random,
    env: BindEnv,
    *,
    purpose: str,
) -> Expr:
    """Bind min/max over every year that has data for the chosen entity and amount concept."""
    if op not in {"min", "max"}:
        raise ValueError(f"expected min or max, got {op!r}")
    entity, concept, periods = _pick_entity_concept_all_periods(index, rng, env, purpose=purpose)
    leaves: List[Expr] = [
        instantiate_base_atom(
            index,
            rng,
            semantic_types=["amount"],
            env=BindEnv(entity=entity, period=p, concept=concept),
        )
        for p in periods
    ]
    return _fold_binary_op(op, leaves)


def _bind_avg_over_all_periods(
    index: AtomIndex,
    rng: random.Random,
    env: BindEnv,
    *,
    purpose: str,
) -> Expr:
    """Arithmetic mean over every year that has data: sum(values) / n (not nested binary avg)."""
    entity, concept, periods = _pick_entity_concept_all_periods(index, rng, env, purpose=purpose)
    leaves: List[Expr] = [
        instantiate_base_atom(
            index,
            rng,
            semantic_types=["amount"],
            env=BindEnv(entity=entity, period=p, concept=concept),
        )
        for p in periods
    ]
    n = len(leaves)
    sum_expr = fold_nary("sum", leaves)
    return with_depth("ratio", sum_expr, Literal(float(n)))


def instantiate_typed_tree(
    tree: Dict[str, Any],
    index: AtomIndex,
    rng: random.Random,
    env: Optional[BindEnv] = None,
    derived_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Expr:
    # Use default environment/registry when caller does not provide one.
    env = env or BindEnv()
    derived_registry = derived_registry or DERIVED_FORMULAS

    kind = tree.get("kind")
    if kind == "leaf":
        # Leaf nodes map to a single base atom matching semantic constraints.
        semantic_types = tree.get("semantic_type_in") or ([tree["family"]] if "family" in tree else ["amount"])
        return instantiate_base_atom(index, rng, semantic_types=semantic_types, env=env)

    if kind == "derived_concept":
        # Derived nodes expand via formula registry while preserving name wrapper.
        return instantiate_formula_reference(
            tree["name"],
            index,
            rng,
            env,
            derived_registry,
            preserve_named_derived=True,
        )

    # Sampler emits time_agg for min/max/avg without left/right; bind here.
    if kind == "time_agg":
        op = tree.get("op")
        if op in {"min", "max"}:
            return _bind_min_or_max_over_all_periods(op, index, rng, env, purpose=f"time_agg {op}")
        if op == "avg":
            return _bind_avg_over_all_periods(index, rng, env, purpose="time_agg avg")
        raise ValueError(
            f"time_agg supports min, max, avg; got {op!r}. (growth uses a binary node.)"
        )

    if kind != "node":
        raise ValueError(f"Unknown typed tree kind: {kind}")

    op = tree["op"]
    if op not in SUPPORTED_OPS:
        raise ValueError(f"Unsupported sampler op: {op}")

    if op in {"sum", "diff"}:
        # Sum/diff are evaluated in shared entity+period context.
        shared = BindEnv(entity=choose_entity(index, rng, env), period=choose_period(index, rng, env), concept=env.concept)
        left = instantiate_typed_tree(tree["left"], index, rng, shared, derived_registry)
        right = instantiate_typed_tree(tree["right"], index, rng, shared, derived_registry)
        return with_depth(op, left, right)

    if op == "ratio":
        # Ratio sides must also share context for meaningful division.
        shared = BindEnv(entity=choose_entity(index, rng, env), period=choose_period(index, rng, env), concept=env.concept)
        left = instantiate_typed_tree(tree["left"], index, rng, shared, derived_registry)
        right = instantiate_typed_tree(tree["right"], index, rng, shared, derived_registry)
        return with_depth("ratio", left, right)

    if op == "mul":
        # Multiplication keeps same entity/period but allows ratio concept to differ.
        shared_entity = choose_entity(index, rng, env)
        shared_period = choose_period(index, rng, env)
        left = instantiate_typed_tree(tree["left"], index, rng, BindEnv(entity=shared_entity, period=shared_period, concept=env.concept), derived_registry)
        right = instantiate_typed_tree(tree["right"], index, rng, BindEnv(entity=shared_entity, period=shared_period), derived_registry)
        return with_depth("mul", left, right)

    if op == "growth":
        entity, concept, p_left, p_right = _pick_entity_concept_two_periods(
            index, rng, env, purpose="growth"
        )
        left = instantiate_typed_tree(
            tree["left"],
            index,
            rng,
            BindEnv(entity=entity, period=p_left, concept=concept),
            derived_registry,
        )
        right = instantiate_typed_tree(
            tree["right"],
            index,
            rng,
            BindEnv(entity=entity, period=p_right, concept=concept),
            derived_registry,
        )
        return with_depth("growth", left, right)

    if op in {"min", "max"}:
        return _bind_min_or_max_over_all_periods(op, index, rng, env, purpose=f"node {op}")

    if op == "avg":
        return _bind_avg_over_all_periods(index, rng, env, purpose="node avg")

    raise ValueError(f"Unhandled op: {op}")

# compiles a tree payload into an expression.
# A tree payload is a JSON object that contains the tree to be compiled.
def compile_tree_payload(
    tree_payload: Dict[str, Any],
    atoms: Dict[str, Atom],
    seed: int = 0,
) -> Expr:
    # Build runtime index and bind one concrete expression from typed template.
    rng = random.Random(seed)
    index = AtomIndex(atoms)
    derived_registry = {
        name: spec["formula"] if "formula" in spec else spec
        for name, spec in (tree_payload.get("derived_concepts") or {}).items()
    }
    if not derived_registry: #If the derived registry is not provided, use the default derived registry.
        derived_registry = DERIVED_FORMULAS
    typed_tree = tree_payload["tree"] if "tree" in tree_payload else tree_payload
    return instantiate_typed_tree(typed_tree, index, rng, BindEnv(), derived_registry)


# =========================================================
# 4. Semantic analyzer
# =========================================================
# a SemanticAnalyzer object is created to analyze the meaning of an expression.
class SemanticAnalyzer:
    def __init__(self, atoms: Dict[str, Atom]):
        self.atoms = atoms

    def atom(self, key: str) -> Atom:
        if key not in self.atoms:
            raise SemanticError(f"Unknown leaf key: {key}")
        return self.atoms[key]

    def same_context(self, a: Meaning, b: Meaning) -> bool:
        return a.entity == b.entity and a.period == b.period and a.unit == b.unit

    def same_entity_and_unit(self, a: Meaning, b: Meaning) -> bool:
        return a.entity == b.entity and a.unit == b.unit

    def _is_metric_like_amount(self, meaning: Meaning) -> bool:
        return meaning.semantic_type == "amount" and meaning.kind in {
            "leaf_metric",
            "aggregate_components",
            "derived_metric",
            "min_over_time",
            "max_over_time",
            "avg_over_time",
            "avg_over_all_periods",
        }

    def _is_metric_like_ratio(self, meaning: Meaning) -> bool:
        return meaning.semantic_type in {"ratio", "rate"} and meaning.kind == "leaf_metric"

    def _same_amount_context(self, a: Meaning, b: Meaning) -> bool:
        return (
            a.semantic_type == "amount"
            and b.semantic_type == "amount"
            and a.entity == b.entity
            and a.unit == b.unit
        )
    # checks if two meanings have the same amount context, concept, and different periods.
    def _same_amount_timeseries_metric(self, a: Meaning, b: Meaning) -> bool:
        return (
            self._same_amount_context(a, b)
            and a.concept == b.concept
            and a.period != b.period
            and self._is_metric_like_amount(a)
            and self._is_metric_like_amount(b)
        )
# analyzes an expression and returns an AnalysisResult object.
    def analyze(self, expr: Expr) -> AnalysisResult:

        if isinstance(expr, Leaf):
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
            return AnalysisResult(expr=expr, meaning=meaning, children=[], depth=0)

        if isinstance(expr, Literal):
            return AnalysisResult(
                expr=expr,
                meaning=Meaning(
                    kind="literal_scalar",
                    semantic_type="ratio",
                    text=str(expr.value),
                    derivation="numeric literal",
                ),
                children=[],
                depth=0,
            )

        if isinstance(expr, DerivedExpr):
            inner = self.analyze(expr.expr)
            inner_meaning = inner.meaning
            meaning = Meaning(
                kind="derived_metric",
                semantic_type=inner_meaning.semantic_type,
                text=f"{expr.name.replace('_', ' ')} for {inner_meaning.entity} in {inner_meaning.period}",
                entity=inner_meaning.entity,
                period=inner_meaning.period,
                unit=inner_meaning.unit,
                concept=expr.name,
                label=expr.name,
                from_period=inner_meaning.from_period,
                to_period=inner_meaning.to_period,
                target_concept=expr.name,
                derivation=f"named derived concept {expr.name}",
            )
            return AnalysisResult(expr=expr, meaning=meaning, children=[inner], depth=expr_depth(expr))

        left_result = self.analyze(expr.left)
        right_result = self.analyze(expr.right)

        if expr.op == "sum":
            meaning = self._analyze_sum(expr, left_result, right_result)
        elif expr.op == "diff":
            meaning = self._analyze_diff(expr, left_result, right_result)
        elif expr.op == "ratio":
            meaning = self._analyze_ratio(expr, left_result, right_result)
        elif expr.op == "mul":
            meaning = self._analyze_mul(expr, left_result, right_result)
        elif expr.op == "growth":
            meaning = self._analyze_growth(expr, left_result, right_result)
        elif expr.op in {"min", "max", "avg"}:
            meaning = self._analyze_time_aggregate(expr, left_result, right_result)
        else:
            raise SemanticError(f"Unsupported op: {expr.op}")
    # calculates the depth of the expression by recursively counting the depth of the left and right subtrees.
        node_depth = expr.depth if isinstance(expr, Node) and expr.depth is not None else 1 + max(left_result.depth, right_result.depth)
        return AnalysisResult(expr=expr, meaning=meaning, children=[left_result, right_result], depth=node_depth)

#Note: the previous function: analyze is very general and traverses an expression tree + route by operator + assembles the final annotated tree
# Analyze_operator takes the pre analyzed left/ right results and just implements the operator node's effect
    def _analyze_sum(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning
        leaves = flatten_sum(expr)

        # Best case: sibling components that form a named aggregate like current_assets
        if leaves:
            atoms = [self.atom(leaf.key) for leaf in leaves]
            first = atoms[0]
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
            ):
                component_labels = [a.label for a in atoms]
                return Meaning(
                    kind="aggregate_components",
                    semantic_type="amount",
                    text=f"{first.parent_concept.replace('_', ' ')} for {first.entity} in {first.period}",
                    entity=first.entity,
                    period=first.period,
                    unit=first.unit,
                    concept=first.parent_concept,
                    label=first.parent_concept,
                    target_concept=first.parent_concept,
                    components=component_labels,
                    derivation=f"sum of sibling components: {', '.join(component_labels)}",
                )

        # Generic same-context amount sum
        if (
            lm.semantic_type == "amount"
            and rm.semantic_type == "amount"
            and lm.entity == rm.entity
            and lm.period == rm.period
            and lm.unit == rm.unit
        ):
            left_label = lm.label or lm.concept or "value"
            right_label = rm.label or rm.concept or "value"
            composite_label = f"{left_label} plus {right_label}"
            return Meaning(
                kind="sum_amount",
                semantic_type="amount",
                text=f"sum of {left_label} and {right_label} for {lm.entity} in {lm.period}",
                entity=lm.entity,
                period=lm.period,
                unit=lm.unit,
                concept=None,
                label=composite_label,
                derivation=f"sum({lm.kind}, {rm.kind})",
            )

        # Fallback for composed amount expressions that still belong to the same entity
        if (
            lm.semantic_type == "amount"
            and rm.semantic_type == "amount"
            and lm.entity == rm.entity
        ):
            left_label = lm.label or lm.concept or "value"
            right_label = rm.label or rm.concept or "value"
            return Meaning(
                kind="sum_amounts",
                semantic_type="amount",
                text=f"sum of two amount expressions for {lm.entity}",
                entity=lm.entity,
                period=lm.period if lm.period == rm.period else None,
                unit=lm.unit if lm.unit == rm.unit else None,
                concept=None,
                label=f"{left_label} plus {right_label}",
                derivation=f"sum({lm.kind}, {rm.kind}) fallback",
            )

        raise SemanticError(
            f"Cannot sum these meanings: {lm.kind}/{lm.semantic_type} and {rm.kind}/{rm.semantic_type}"
        )
               
    def _analyze_diff(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning

        if self._same_amount_context(lm, rm) and lm.concept == rm.concept and lm.period != rm.period:
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
            left_label = lm.label or lm.concept or "value"
            right_label = rm.label or rm.concept or "value"
            composite_label = f"{left_label} minus {right_label}"
            return Meaning(
                kind="difference_amount",
                semantic_type="amount",
                text=f"difference between {left_label} and {right_label} for {lm.entity} in {lm.period}",
                entity=lm.entity,
                period=lm.period,
                unit=lm.unit,
                concept=None,
                label=composite_label,
                derivation=f"diff({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot subtract these meanings: {lm.kind} and {rm.kind}")

    def _analyze_ratio(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning

        if isinstance(expr.right, Literal):
            n = float(expr.right.value)
            n_int = int(round(n))
            if n_int >= 2 and abs(n - n_int) < 1e-9:
                leaves = flatten_leaves(expr.left)
                if len(leaves) == n_int:
                    atoms = [self.atom(leaf.key) for leaf in leaves]
                    if (
                        len({a.entity for a in atoms}) == 1
                        and len({a.concept for a in atoms}) == 1
                        and all(a.semantic_type == "amount" for a in atoms)
                    ):
                        periods = sorted({a.period for a in atoms})
                        first = atoms[0]
                        return Meaning(
                            kind="avg_over_all_periods",
                            semantic_type="amount",
                            text=(
                                f"average of {first.label} for {first.entity} "
                                f"across years {periods[0]} through {periods[-1]}"
                            ),
                            entity=first.entity,
                            unit=first.unit,
                            concept=first.concept,
                            label=first.label,
                            from_period=periods[0],
                            to_period=periods[-1],
                            period=periods[-1],
                            derivation=f"arithmetic mean of {n_int} yearly values",
                        )

        if (
            lm.kind == "change_over_time"
            and rm.semantic_type == "amount"
            and lm.concept == rm.concept
            and lm.entity == rm.entity
            and lm.unit == rm.unit
            and lm.from_period == rm.period
            and self._is_metric_like_amount(rm)
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

        if self._same_amount_context(lm, rm):
            left_label = lm.label or lm.concept or "value"
            right_label = rm.label or rm.concept or "value"
            return Meaning(
                kind="ratio_amounts",
                semantic_type="ratio",
                text=f"ratio of {left_label} to {right_label} for {lm.entity}",
                entity=lm.entity,
                period=lm.period if lm.period == rm.period else None,
                unit=lm.unit,
                concept=None,
                label=f"{left_label} to {right_label}",
                derivation=f"ratio({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot divide these meanings: {lm.kind} and {rm.kind}")

    def _analyze_mul(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning

        if lm.semantic_type == "amount" and rm.semantic_type in {"ratio", "rate"} and lm.entity == rm.entity:
            amount_label = lm.label or lm.concept or 'amount'
            ratio_label = rm.label or rm.concept or 'ratio'
            return Meaning(
                kind="scaled_amount",
                semantic_type="amount",
                text=f"{amount_label} adjusted by {ratio_label} for {lm.entity}",
                entity=lm.entity,
                period=lm.period or rm.period,
                unit=lm.unit,
                concept=None,
                label=f"{amount_label} adjusted by {ratio_label}",
                target_concept=ratio_label,
                derivation=f"mul({lm.kind}, {rm.kind})",
            )

        if rm.semantic_type == "amount" and lm.semantic_type in {"ratio", "rate"} and lm.entity == rm.entity:
            amount_label = rm.label or rm.concept or 'amount'
            ratio_label = lm.label or lm.concept or 'ratio'
            return Meaning(
                kind="scaled_amount",
                semantic_type="amount",
                text=f"{amount_label} adjusted by {ratio_label} for {rm.entity}",
                entity=rm.entity,
                period=rm.period or lm.period,
                unit=rm.unit,
                concept=None,
                label=f"{amount_label} adjusted by {ratio_label}",
                target_concept=ratio_label,
                derivation=f"mul({lm.kind}, {rm.kind})",
            )

        raise SemanticError(f"Cannot multiply these meanings: {lm.kind} and {rm.kind}")

    def _analyze_growth(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning
        if self._same_amount_timeseries_metric(lm, rm):
            periods = sorted([lm.period, rm.period])
            return Meaning(
                kind="growth_rate",
                semantic_type="ratio",
                text=f"growth rate of {lm.label or lm.concept} for {lm.entity} from {periods[0]} to {periods[1]}",
                entity=lm.entity,
                unit=lm.unit,
                concept=lm.concept,
                label=f"growth rate of {lm.label or lm.concept}",
                from_period=periods[0],
                to_period=periods[1],
                derivation="explicit growth operator",
            )

        if self._same_amount_context(lm, rm):
            periods = [p for p in [lm.period, rm.period] if p is not None]
            ordered = sorted(periods) if len(periods) == 2 and len(set(periods)) == 2 else periods
            from_period = ordered[0] if ordered else None
            to_period = ordered[-1] if len(ordered) >= 2 else None
            return Meaning(
                kind="growth_rate_composed",
                semantic_type="ratio",
                text=f"growth between two amount expressions for {lm.entity}",
                entity=lm.entity,
                unit=lm.unit,
                concept=None,
                label=None,
                from_period=from_period,
                to_period=to_period,
                derivation="explicit growth operator over composed amount expressions",
            )
        raise SemanticError(f"Cannot compute growth for these meanings: {lm.kind} and {rm.kind}")

    def _analyze_time_aggregate(self, expr: Node, left: AnalysisResult, right: AnalysisResult) -> Meaning:
        lm, rm = left.meaning, right.meaning
        op_name = {"min": "minimum", "max": "maximum", "avg": "average"}[expr.op]

        def _period_tokens(m: Meaning) -> List[str]:
            return [p for p in (m.period, m.from_period, m.to_period) if p is not None]

        # Nested min/max folds share one concept; span all years mentioned in either subtree.
        if expr.op in {"min", "max"}:
            span_periods = sorted(set(_period_tokens(lm) + _period_tokens(rm)))
            if (
                lm.semantic_type == "amount"
                and rm.semantic_type == "amount"
                and lm.entity == rm.entity
                and lm.unit == rm.unit
                and lm.concept is not None
                and lm.concept == rm.concept
                and len(span_periods) >= 2
            ):
                base_label = (lm.label or lm.concept or "value").replace("minimum ", "").replace("maximum ", "")
                return Meaning(
                    kind=f"{expr.op}_over_time",
                    semantic_type="amount",
                    text=(
                        f"{op_name} of {base_label} for {lm.entity} "
                        f"across years {span_periods[0]} through {span_periods[-1]}"
                    ),
                    entity=lm.entity,
                    unit=lm.unit,
                    concept=lm.concept,
                    label=f"{op_name} {base_label}",
                    from_period=span_periods[0],
                    to_period=span_periods[-1],
                    period=span_periods[-1],
                    derivation=f"{expr.op} over {len(span_periods)} periods",
                )

        if self._same_amount_timeseries_metric(lm, rm):
            periods = sorted([lm.period, rm.period])
            base_label = lm.label or lm.concept or "value"
            return Meaning(
                kind=f"{expr.op}_over_time",
                semantic_type="amount",
                text=f"{op_name} of {base_label} for {lm.entity} from {periods[0]} to {periods[1]}",
                entity=lm.entity,
                unit=lm.unit,
                concept=lm.concept,
                label=f"{op_name} {base_label}",
                from_period=periods[0],
                to_period=periods[1],
                period=periods[-1],
                derivation=f"{expr.op} over time",
            )

        if self._same_amount_context(lm, rm):
            return Meaning(
                kind=f"{expr.op}_amounts",
                semantic_type="amount",
                text=f"{op_name} of two amount expressions for {lm.entity}",
                entity=lm.entity,
                unit=lm.unit,
                concept=None,
                label=None,
                from_period=lm.period,
                to_period=rm.period,
                derivation=f"{expr.op} over composed amount expressions",
            )

        raise SemanticError(f"Cannot compute {expr.op} for these meanings: {lm.kind} and {rm.kind}")


# =========================================================
# 5. Numeric evaluator + question renderer
# =========================================================
# an Evaluator object is created to evaluate the value of an expression.
class Evaluator:
    def __init__(self, atoms: Dict[str, Atom]):
        self.atoms = atoms

    def eval(self, expr: Expr) -> float:
        # Evaluate bottom-up by recursively computing child values.
        if isinstance(expr, Leaf):
            return float(self.atoms[expr.key].value)
        if isinstance(expr, Literal):
            return float(expr.value)
        if isinstance(expr, DerivedExpr):
            return self.eval(expr.expr)

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
        if expr.op == "growth":
            if rv == 0:
                raise ZeroDivisionError("Division by zero in growth base.")
            return (lv - rv) / rv
        if expr.op == "min":
            return min(lv, rv)
        if expr.op == "max":
            return max(lv, rv)
        if expr.op == "avg":
            return (lv + rv) / 2.0

        raise ValueError(f"Unsupported op: {expr.op}")


class QuestionRenderer:
    def render(self, result: AnalysisResult) -> str:
        # Prefer specialized phrasing for high-signal semantic kinds.
        m = result.meaning

        if m.kind == "aggregate_components":
            return (
                f"What are the total {self._clean_label(m.target_concept)} "
                f"for {m.entity} in {m.period}?"
            )

        if m.kind == "growth_rate":
            return (
                f"What is the growth rate of "
                f"{self._clean_label((m.label or '').replace('growth rate of ', ''))} "
                f"for {m.entity} from {m.from_period} to {m.to_period}?"
            )

        if m.kind == "avg_over_all_periods":
            return (
                f"What is the average {self._clean_label(m.label or m.concept)} "
                f"for {m.entity} across years {m.from_period} through {m.to_period}?"
            )

        phrase = self._expr_phrase(result, top_level=True)
        return f"What is {phrase}?"

    def _expr_phrase(self, result: AnalysisResult, top_level: bool = False) -> str:
        expr = result.expr
        m = result.meaning

        if isinstance(expr, Leaf):
            return f"{self._clean_label(m.label)} for {m.entity} in {m.period}"

        if isinstance(expr, DerivedExpr):
            return f"{self._clean_label(m.label)} for {m.entity} in {m.period}"

        if isinstance(expr, Literal):
            return str(expr.value)

        left = self._expr_phrase(result.children[0])
        right = self._expr_phrase(result.children[1])
        left_meaning = result.children[0].meaning
        right_meaning = result.children[1].meaning

        if expr.op == "sum":
            shared_context = self._shared_entity_period_context(left_meaning, right_meaning)
            if shared_context is not None and self._both_have_entity_period_suffix(left, right, *shared_context):
                entity, period = shared_context
                left = self._strip_entity_period_suffix(left, entity, period)
                right = self._strip_entity_period_suffix(right, entity, period)
                return f"the sum of {left} and {right} for {entity} in {period}"
            return f"the sum of {left} and {right}"

        if expr.op == "diff":
            shared_context = self._shared_entity_period_context(left_meaning, right_meaning)
            if shared_context is not None and self._both_have_entity_period_suffix(left, right, *shared_context):
                entity, period = shared_context
                left = self._strip_entity_period_suffix(left, entity, period)
                right = self._strip_entity_period_suffix(right, entity, period)
                return f"the difference between {left} and {right} for {entity} in {period}"
            return f"the difference between {left} and {right}"

        if expr.op == "ratio":
            shared_context = self._shared_entity_period_context(left_meaning, right_meaning)
            if shared_context is not None and self._both_have_entity_period_suffix(left, right, *shared_context):
                entity, period = shared_context
                left = self._strip_entity_period_suffix(left, entity, period)
                right = self._strip_entity_period_suffix(right, entity, period)
                return f"the ratio of {left} to {right} for {entity} in {period}"
            return f"the ratio of {left} to {right}"

        if expr.op == "mul":
            shared_context = self._shared_entity_period_context(left_meaning, right_meaning)
            if shared_context is not None and self._both_have_entity_period_suffix(left, right, *shared_context):
                entity, period = shared_context
                left = self._strip_entity_period_suffix(left, entity, period)
                right = self._strip_entity_period_suffix(right, entity, period)
                return f"the result of {left} scaled by {right} for {entity} in {period}"
            return f"the result of {left} scaled by {right}"

        if expr.op == "growth":
            if m.kind == "growth_rate" and m.concept is not None:
                return (
                    f"the growth in {self._base_metric_name(m)} for {m.entity} "
                    f"from {m.from_period} to {m.to_period}"
                )
            return f"the growth between {left} and {right}"

        if expr.op in {"min", "max", "avg"}:
            op_word = {
                "min": "minimum",
                "max": "maximum",
                "avg": "average",
            }[expr.op]

            if m.kind == f"{expr.op}_over_time" and m.concept is not None:
                base_name = self._base_metric_name_from_child(result.children[0].meaning)
                return (
                    f"the {op_word} of {base_name} for {m.entity} "
                    f"using values from {m.from_period} through {m.to_period}"
                )

            return f"the {op_word} of {left} and {right}"

        return m.text

    def _base_metric_name_from_child(self, meaning: Meaning) -> str:
        label = meaning.label or meaning.concept or "value"

        for prefix in ("minimum ", "maximum ", "average "):
            if label.startswith(prefix):
                return self._clean_label(label[len(prefix):])

        return self._clean_label(label)

    def _base_metric_name(self, meaning: Meaning) -> str:
        label = meaning.label or meaning.concept or "value"
        for prefix in ("growth rate of ", "minimum ", "maximum ", "average "):
            if label.startswith(prefix):
                label = label[len(prefix):]
        return self._clean_label(label)

    def _clean_label(self, label: Optional[str]) -> str:
        if not label:
            return "value"
        return label.replace("_", " ")

    def _shared_entity_period_context(
        self, left_meaning: Meaning, right_meaning: Meaning
    ) -> Optional[Tuple[str, str]]:
        if (
            left_meaning.entity
            and right_meaning.entity
            and left_meaning.period
            and right_meaning.period
            and left_meaning.entity == right_meaning.entity
            and left_meaning.period == right_meaning.period
        ):
            return left_meaning.entity, left_meaning.period
        return None

    def _strip_entity_period_suffix(self, phrase: str, entity: str, period: str) -> str:
        suffix = f" for {entity} in {period}"
        if phrase.endswith(suffix):
            return phrase[: -len(suffix)]
        return phrase

    def _both_have_entity_period_suffix(self, left: str, right: str, entity: str, period: str) -> bool:
        suffix = f" for {entity} in {period}"
        return left.endswith(suffix) and right.endswith(suffix)

# =========================================================
# 6. Convenience wrappers for the pipeline
# =========================================================

def compile_tree_file(
    tree_json_path: str | Path,
    atoms_json_path: str | Path,
    seed: int = 0,
) -> Expr:
    atoms = load_atoms_json(atoms_json_path)
    with open(tree_json_path, "r", encoding="utf-8") as f:
        tree_payload = json.load(f)
    return compile_tree_payload(tree_payload, atoms, seed=seed)


def compile_and_analyze(
    tree_json_path: str | Path,
    atoms_json_path: str | Path,
    seed: int = 0,
) -> Dict[str, Any]:
    atoms = load_atoms_json(atoms_json_path)
    with open(tree_json_path, "r", encoding="utf-8") as f:
        tree_payload = json.load(f)

    expr = compile_tree_payload(tree_payload, atoms, seed=seed)
    analyzer = SemanticAnalyzer(atoms)
    result = analyzer.analyze(expr)
    value = Evaluator(atoms).eval(expr)
    question = QuestionRenderer().render(result)
    return {
        "expr": expr_to_json(expr),
        "expr_str": show_expr(expr),
        "depth": expr_depth(expr),
        "question": question,
        "answer": value,
        "meaning": result.meaning.__dict__,
    }


def print_analysis_tree(result: AnalysisResult, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}Expr:     {show_expr(result.expr)}")
    print(f"{prefix}Kind:     {result.meaning.kind}")
    print(f"{prefix}Type:     {result.meaning.semantic_type}")
    print(f"{prefix}Depth:    {result.depth}")
    print(f"{prefix}Meaning:  {result.meaning.text}")
    if result.meaning.derivation:
        print(f"{prefix}How:      {result.meaning.derivation}")
    print()
    for child in result.children:
        print_analysis_tree(child, indent + 1)


# =========================================================
# 7. CLI
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate a typed sampled tree into a concrete financial expression.")
    parser.add_argument("--atoms", required=True, help="Path to atoms_data.json")
    parser.add_argument("--tree", required=True, help="Path to generated_operator_tree.json")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for atom binding")
    parser.add_argument("--output", default="output/concrete_expression.json", help="Where to save the compiled expression bundle")
    args = parser.parse_args()

    bundle = compile_and_analyze(args.tree, args.atoms, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)
    bound_png = str(Path(args.output).with_suffix(".png"))

    try:
        subprocess.run(
            [
                sys.executable,
                "tree_visualizer_bound.py",
                "--expression-json",
                args.output,
                "--atoms-json",
                args.atoms,
                "--output",
                bound_png,
            ],
            check=True,
        )
        print(f"Saved bound tree image to {bound_png}")
    except Exception as e:
        print(f"Warning: could not generate bound tree image: {e}")

    print(f"Saved compiled expression bundle to {output_path}")
    print(bundle["expr_str"])
    print(bundle["question"])
    print(bundle["answer"])


if __name__ == "__main__":
    main()
