from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from enum import StrEnum
from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils import pick_random_contiguous_period_window

class SemanticError(Exception):
    """Raised when an expression violates semantic compatibility rules."""

class DerivedConcept(BaseModel):
    name: str
    op: Operation
    family: str
    concept_dept: int
    args: list[str]
    protected: bool = False

    @property
    def concept_depth(self) -> int:
        return self.concept_dept

class Expr(BaseModel, ABC):
    model_config = {
        "arbitrary_types_allowed": True,
    }

    def __getitem__(self, key: str):
        return getattr(self, key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    @abstractmethod
    def expr_depth(self) -> int:
        ...

    @abstractmethod
    def flatten_leaves(self) -> list["Leaf"]:
        ...

    @abstractmethod
    def flatten_sum(self) -> list["Leaf"]:
        ...

    @abstractmethod
    def contains_a_derived_concept(self) -> bool:
        ...

    @staticmethod
    def fold_narry(op: Operation, operands: list[Expr], *, balanced: bool = False) -> Expr:
        if not operands:
            raise ValueError("No operands to fold.")
        if len(operands) == 1:
            return operands[0]
        if not balanced:
            current_expr = operands[0]
            for operand in operands[1:]:
                current_expr = Node(op=op, left=current_expr, right=operand)
            return current_expr
        mid = len(operands) // 2
        left = Expr.fold_narry(op, operands[:mid], balanced=True)
        right = Expr.fold_narry(op, operands[mid:], balanced=True)
        return Node(op=op, left=left, right=right)

    @staticmethod
    def split_balanced_child_depths(depth: int, rng: random.Random) -> tuple[int, int]:
        """Both children get the same remaining depth so the tree is fully balanced."""
        if depth <= 0:
            return 0, 0
        child_depth = depth - 1
        return child_depth, child_depth

    @staticmethod
    def is_height_balanced(expr: Expr | dict) -> bool:
        """True when every internal node has equal-depth left and right subtrees."""
        tree = Expr.coerce_template_expr(expr) if isinstance(expr, dict) else expr
        if isinstance(tree, (Leaf, Literal, TimeAgg)):
            return True
        if isinstance(tree, DerivedExpr):
            if tree.expr is None:
                return True
            return Expr.is_height_balanced(tree.expr)
        if isinstance(tree, Node):
            left_depth = tree.left.expr_depth()
            right_depth = tree.right.expr_depth()
            if left_depth != right_depth:
                return False
            return Expr.is_height_balanced(tree.left) and Expr.is_height_balanced(tree.right)
        return True

    @staticmethod
    def coerce_operation(op: Operation | str) -> Operation:
        if isinstance(op, Operation):
            return op
        return Operation(str(op))

    @staticmethod
    def normalize_derived_registry(
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None,
    ) -> dict[str, DerivedConcept]:
        if derived_registry is None:
            return DERIVED_CONCEPTS
        if isinstance(derived_registry, list):
            return {concept.name: concept for concept in derived_registry}

        normalized: dict[str, DerivedConcept] = {}
        for name, spec in derived_registry.items():
            if isinstance(spec, DerivedConcept):
                normalized[name] = spec
                continue

            formula = spec.get("formula", spec)
            normalized[name] = DerivedConcept(
                name=name,
                op=Expr.coerce_operation(formula["op"]),
                family=str(spec.get("family", "amount")),
                concept_dept=int(spec.get("concept_depth", spec.get("concept_dept", 0))),
                args=[str(arg) for arg in formula["args"]],
                protected=bool(spec.get("protected", False)),
            )
        return normalized

    @staticmethod
    def coerce_template_expr(tree: Expr | dict) -> Expr:
        if isinstance(tree, Expr):
            return tree
        if isinstance(tree, dict) and "kind" in tree:
            return Expr.parse_template_expr(tree)
        if isinstance(tree, dict):
            return Expr.parse_expr(tree)
        raise TypeError(f"Unsupported tree value: {type(tree)!r}")    
    
    @staticmethod
    def instantiate_formula_reference(
        name: str,
        index: Store,
        env: BindEnv,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
        *,
        preserve_named_derived: bool = True,
        balanced: bool = False,
    ) -> Expr:
        """
        Expand a registry derived name into sub-expressions, or bind a base atom if ``name`` is primitive.
        """
        env = env or BindEnv()
        registry = Expr.normalize_derived_registry(derived_registry)

        # One company-year for the whole expansion: operands must refer to the same slice.
        bound_env = BindEnv(
            entity=index.choose_entity(env),
            period=index.choose_period(env),
            concept=env.concept,
        )
        # Formula args like "revenue" are base metrics, not registry keys: bind one atom.
        if name not in registry:
            return index.instantiate_base_atom(
                semantic_types=[SemanticType.amount],
                env=BindEnv(
                    entity=bound_env.entity,
                    period=bound_env.period,
                    concept=name,
                ),
            )

        derived_concept = registry[name]
        op = derived_concept.op

        args = [
            Expr.instantiate_formula_reference(
                arg,
                index,
                bound_env,
                registry,
                preserve_named_derived=preserve_named_derived,
                balanced=balanced,
            )
            for arg in derived_concept.args]
        
        match op:
            case Operation.sum:
                expanded = Expr.fold_narry(Operation.sum, args, balanced=balanced)
            case Operation.diff | Operation.ratio | Operation.mul:
                if len(args) != 2:
                    raise ValueError(f"Operator {op} expects exactly 2 args in derived formula {name}.")
                expanded = Node(op=op, left=args[0], right=args[1])
            case _:
                raise ValueError(f"Unsupported derived formula op: {op}")


        if preserve_named_derived:
            return DerivedExpr(name=name, expr=expanded, depth=expanded.expr_depth())
        return expanded


    @staticmethod
    def instantiate_typed_tree(
        tree: Node | Leaf | DerivedExpr | TimeAgg | Literal | dict,
        index: Store,
        env: BindEnv | None = None,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
        *,
        balanced: bool = False,
    ) -> Expr:
        env = env or BindEnv()
        registry = Expr.normalize_derived_registry(derived_registry)
        tree = Expr.coerce_template_expr(tree)

        match tree:
            case Literal():
                return tree
            case Leaf():
                if tree.key is not None and not tree.semantic_type_in:
                    return tree
                # Leaf nodes map to a single base atom matching semantic constraints.
                semantic_types = tree.semantic_type_in or [SemanticType.amount]
                return index.instantiate_base_atom(semantic_types=semantic_types, env=env)
            case DerivedExpr():
                if tree.expr is not None:
                    return tree
                # Derived nodes expand via formula registry while preserving name wrapper.
                return Expr.instantiate_formula_reference(
                    tree.name,
                    index,
                    env,
                    registry,
                    preserve_named_derived=True,
                    balanced=balanced,
                )
            case TimeAgg():
                op = tree.op
                match op:
                    case Operation.max | Operation.min:
                        return index.bind_min_or_max_over_all_periods(op, env, purpose=f"time_agg {op}")
                    case Operation.avg:
                        return index.bind_avg_over_all_periods(env, purpose="time_agg avg")
                    case _:
                        raise ValueError(f"Unsupported time aggregation op: {op}")
                    
            case Node():
                op = tree.op
                match op:
                    case Operation.sum | Operation.diff:
                        # Sum/diff are evaluated in shared entity+period context.
                        shared = BindEnv(entity=index.choose_entity(env), period=index.choose_period(env), concept=env.concept)
                        left = Expr.instantiate_typed_tree(tree.left, index, shared, registry, balanced=balanced)
                        right = Expr.instantiate_typed_tree(tree.right, index, shared, registry, balanced=balanced)
                        return Node(op=op, left=left, right=right)
                    case Operation.ratio:
                        # Ratio sides must also share context for meaningful division.
                        shared = BindEnv(entity=index.choose_entity(env), period=index.choose_period(env), concept=env.concept)
                        left = Expr.instantiate_typed_tree(tree.left, index, shared, registry, balanced=balanced)
                        right = Expr.instantiate_typed_tree(tree.right, index, shared, registry, balanced=balanced)
                        return Node(op=Operation.ratio, left=left, right=right)
                    case Operation.mul:
                        # Multiplication keeps same entity/period but allows ratio concept to differ.
                        shared_entity = index.choose_entity(env)
                        shared_period = index.choose_period(env)
                        left = Expr.instantiate_typed_tree(tree.left, index, BindEnv(entity=shared_entity, period=shared_period, concept=env.concept), registry, balanced=balanced)
                        right = Expr.instantiate_typed_tree(tree.right, index, BindEnv(entity=shared_entity, period=shared_period), registry, balanced=balanced)
                        return Node(op=Operation.mul, left=left, right=right)
                    case Operation.growth:
                        entity, concept, p_left, p_right = index.pick_entity_concept_two_periods(
                            env, purpose="growth"
                        )
                        left = Expr.instantiate_typed_tree(
                            tree.left,
                            index,
                            BindEnv(entity=entity, period=p_left, concept=concept),
                            registry,
                            balanced=balanced,
                        )
                        right = Expr.instantiate_typed_tree(
                            tree.right,
                            index,
                            BindEnv(entity=entity, period=p_right, concept=concept),
                            registry,
                            balanced=balanced,
                        )
                        return Node(op=Operation.growth, left=left, right=right)
                    case Operation.min | Operation.max:
                        return index.bind_min_or_max_over_all_periods(op, env, purpose=f"node {op}")
                    case Operation.avg:
                        return index.bind_avg_over_all_periods(env, purpose="node avg")
                    case _:
                        raise ValueError(f"Unknown operation: {op}")

    @staticmethod
    def make_leaf(family: str) -> Leaf:
        if family == "amount":
            return Leaf(semantic_type_in=[SemanticType.amount])
        if family == "ratio":
            return Leaf(semantic_type_in=[SemanticType.rate])
        raise ValueError(f"Unsupported family: {family}")

    @staticmethod
    def make_derived_concept(name: str) -> DerivedExpr:
        return DerivedExpr(name=name)

    @staticmethod
    def make_node(op: Operation | str, left: Expr, right: Expr) -> Node:
        return Node(op=Expr.coerce_operation(op), left=left, right=right)

    @staticmethod
    def make_time_agg(op: Operation | str) -> TimeAgg:
        return TimeAgg(op=Expr.coerce_operation(op))

    @staticmethod
    def eligible_derived_concepts(
        depth: int,
        family: str,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
    ) -> list[str]:
        registry = Expr.normalize_derived_registry(derived_registry)
        return [
            name
            for name, spec in registry.items()
            if spec.family == family and spec.concept_depth < depth
        ]

    @staticmethod
    def choose_amount_op( allow_time_aggregates: bool) -> Operation:
        ops = list(AMOUNT_BINARY_OPS)
        if allow_time_aggregates:
            ops.extend(TIME_AGG_OPS)
        return random.choice(ops)
    @staticmethod
    def sample_amount_terminal(
        depth: int,
        derived_prob: float,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
    ) -> Expr:
        eligible = Expr.eligible_derived_concepts(
            depth=depth,
            family="amount",
            derived_registry=derived_registry,
        )
        if eligible and random.uniform(0, 1) < derived_prob:
            return Expr.make_derived_concept(random.choice(eligible))
        return Expr.make_leaf("amount")

    @staticmethod
    def build_time_series_amount_pair(
        depth: int,
        derived_prob: float,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
        *,
        balanced: bool = False,
    ) -> tuple[Expr, Expr]:
        if balanced:
            left_depth, right_depth = depth, depth
        else:
            left_depth = right_depth = depth
        left = Expr.build_amount_tree(

            depth=depth,
            derived_prob=derived_prob,
            allow_time_aggregates=False,
            derived_registry=derived_registry,
            balanced=balanced,
        )
        right = Expr.build_amount_tree(
            depth=depth,
            derived_prob=derived_prob,
            allow_time_aggregates=False,
            derived_registry=derived_registry,
            balanced=balanced,
        )
        return left, right

    @staticmethod
    def build_ratio_tree(
        depth: int,
        derived_prob: float,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
        *,
        balanced: bool = False,
    ) -> Expr:
        if depth == 0:
            return Expr.make_leaf("ratio")


        op = random.choice(RATIO_OPS)
        if op == Operation.ratio:
            left = Expr.build_amount_tree(
                depth=depth - 1,
                derived_prob=derived_prob,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            right = Expr.build_amount_tree(
                depth=depth - 1,
                derived_prob=derived_prob,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            return Expr.make_node(op=Operation.ratio, left=left, right=right)

        if op == Operation.growth:
            left, right = Expr.build_time_series_amount_pair(
                depth=depth - 1,
                derived_prob=derived_prob,
                allow_time_aggregates=False,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            right = Expr.build_amount_tree(
                depth=right_depth,
                rng=rng,
                derived_prob=derived_prob,
                allow_time_aggregates=False,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            return Expr.make_node(op=Operation.growth, left=left, right=right)

        raise ValueError(f"Unsupported ratio op: {op}")

    @staticmethod
    def build_amount_tree(
        depth: int,
        derived_prob: float = 0.30,
        allow_time_aggregates: bool = True,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
        *,
        balanced: bool = False,
    ) -> Expr:
        if depth == 0:
            return Expr.sample_amount_terminal(
                depth=0,
                derived_registry=derived_registry,
            )

        if random.uniform(0, 1) < 0.35:
            return Expr.sample_amount_terminal(
                depth=depth,
                derived_prob=derived_prob,
                derived_registry=derived_registry,
            )


        op = Expr.choose_amount_op(allow_time_aggregates)

        if op in {Operation.sum, Operation.diff}:
            left = Expr.build_amount_tree(
                depth=depth - 1,
                derived_prob=derived_prob,
                allow_time_aggregates=allow_time_aggregates and not balanced,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            right = Expr.build_amount_tree(
                depth=depth - 1,
                derived_prob=derived_prob,
                allow_time_aggregates=allow_time_aggregates and not balanced,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            return Expr.make_node(op=op, left=left, right=right)

        if op == Operation.mul:
            left = Expr.build_amount_tree(
                depth=depth - 1,
                derived_prob=derived_prob,
                allow_time_aggregates=allow_time_aggregates and not balanced,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            right = Expr.build_ratio_tree(
                depth=depth - 1,
                derived_prob=derived_prob,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            return Expr.make_node(op=Operation.mul, left=left, right=right)

        if op in TIME_AGG_OPS:
            return Expr.make_time_agg(op)

        raise ValueError(f"Unsupported amount op: {op}")

    @staticmethod
    def expand_formula_reference(
        name: str,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
    ):
        registry = Expr.normalize_derived_registry(derived_registry)
        if name not in registry:
            return name

        spec = registry[name]
        return (
            spec.op.value,
            tuple(Expr.expand_formula_reference(arg, registry) for arg in spec.args),
        )

    @staticmethod
    def normalize_symbolic(expr):
        if isinstance(expr, str):
            return expr

        op, args = expr
        norm_args = tuple(Expr.normalize_symbolic(arg) for arg in args)
        if op == Operation.sum.value:
            norm_args = tuple(sorted(norm_args, key=repr))
        return op, norm_args

    @staticmethod
    def symbolic_from_tree(
        tree: Expr | dict,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
    ):
        tree = Expr.coerce_template_expr(tree)
        if isinstance(tree, Leaf):
            return "LEAF"
        if isinstance(tree, DerivedExpr):
            if tree.name:
                return Expr.normalize_symbolic(Expr.expand_formula_reference(tree.name, derived_registry))
            if tree.expr is not None:
                return Expr.symbolic_from_tree(tree.expr, derived_registry)
        if isinstance(tree, TimeAgg):
            if tree.op is None:
                raise ValueError("TimeAgg nodes must define an op.")
            return ("time_agg", tree.op.value)
        if isinstance(tree, Literal):
            return str(tree.value)
        if isinstance(tree, Node):
            left = Expr.symbolic_from_tree(tree.left, derived_registry)
            right = Expr.symbolic_from_tree(tree.right, derived_registry)
            return Expr.normalize_symbolic((tree.op.value, (left, right)))
        raise ValueError(f"Unknown tree type: {type(tree)!r}")

    @staticmethod
    def contains_named_derived(tree: Expr | dict, concept_name: str) -> bool:
        tree = Expr.coerce_template_expr(tree)
        if isinstance(tree, DerivedExpr):
            return tree.name == concept_name
        if isinstance(tree, (Leaf, Literal, TimeAgg)):
            return False
        if isinstance(tree, Node):
            return Expr.contains_named_derived(tree.left, concept_name) or Expr.contains_named_derived(tree.right, concept_name)
        return False

    @staticmethod
    def protected_signatures(
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
    ) -> dict[str, object]:
        registry = Expr.normalize_derived_registry(derived_registry)
        out: dict[str, object] = {}
        for name, spec in registry.items():
            if spec.protected:
                out[name] = Expr.normalize_symbolic(Expr.expand_formula_reference(name, registry))
        return out

    @staticmethod
    def violates_protected_canonical_form(
        tree: Expr | dict,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
    ) -> str | None:
        tree = Expr.coerce_template_expr(tree)
        if isinstance(tree, (Leaf, Literal, TimeAgg, DerivedExpr)):
            return None

        signatures = Expr.protected_signatures(derived_registry)
        tree_sig = Expr.symbolic_from_tree(tree, derived_registry)
        for concept_name, concept_sig in signatures.items():
            if tree_sig == concept_sig and not Expr.contains_named_derived(tree, concept_name):
                return concept_name

        left_violation = Expr.violates_protected_canonical_form(tree.left, derived_registry)
        if left_violation is not None:
            return left_violation
        return Expr.violates_protected_canonical_form(tree.right, derived_registry)

    @staticmethod
    def count_nodes(tree: Expr | dict) -> dict[str, int]:
        tree = Expr.coerce_template_expr(tree)
        if isinstance(tree, (Leaf, Literal, DerivedExpr, TimeAgg)):
            return {"internal_nodes": 0, "leaves": 1, "total_nodes": 1}

        left = Expr.count_nodes(tree.left)
        right = Expr.count_nodes(tree.right)
        return {
            "internal_nodes": 1 + left["internal_nodes"] + right["internal_nodes"],
            "leaves": left["leaves"] + right["leaves"],
            "total_nodes": 1 + left["total_nodes"] + right["total_nodes"],
        }

    @staticmethod
    def actual_tree_depth(tree: Expr | dict) -> int:
        tree = Expr.coerce_template_expr(tree)
        if isinstance(tree, (Leaf, Literal, DerivedExpr, TimeAgg)):
            return 0
        return 1 + max(Expr.actual_tree_depth(tree.left), Expr.actual_tree_depth(tree.right))

    @staticmethod
    def parse_expr(obj: dict[str, Any]) -> Expr:
        if not isinstance(obj, dict):
            raise ValueError("Expression parts must be JSON objects.")

        if "kind" in obj:
            return Expr.parse_template_expr(obj)

        if "leaf" in obj:
            return Leaf(key=str(obj["leaf"]))

        if "literal" in obj:
            return Literal(value=float(obj["literal"]))

        if "derived" in obj and "expanded" in obj:
            expanded = Expr.parse_expr(obj["expanded"])
            return DerivedExpr(
                name=str(obj["derived"]),
                expr=expanded,
                depth=int(obj.get("depth", expanded.expr_depth())),
            )

        if {"op", "left", "right"}.issubset(obj.keys()):
            op = Expr.coerce_operation(obj["op"])
            left = Expr.parse_expr(obj["left"])
            right = Expr.parse_expr(obj["right"])
            parsed_depth = obj.get("depth")
            computed_depth = 1 + max(left.expr_depth(), right.expr_depth())
            if parsed_depth is not None and int(parsed_depth) != computed_depth:
                raise ValueError(
                    f"Depth mismatch for node {op}: provided {parsed_depth}, computed {computed_depth}"
                )
            return Node(op=op, left=left, right=right, depth=computed_depth)

        raise ValueError(
            "Use {'leaf': ...}, {'literal': ...}, {'derived': ..., 'expanded': ...}, "
            "or {'op': ..., 'left': ..., 'right': ...}."
        )


    @staticmethod
    def sample_tree_with_rejection(
        max_depth: int,
        derived_prob: float,
        max_attempts: int = 200,
        derived_registry: dict[str, DerivedConcept] | list[DerivedConcept] | None = None,
        require_derived: bool = False,
        *,
        balanced: bool = False,
    ) -> tuple[Expr, str | None]:
        if max_depth < 0:
            raise ValueError("max_depth must be >= 0.")
        if not (0.0 <= derived_prob <= 1.0):
            raise ValueError("derived_prob must be within [0, 1].")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be > 0.")
        if require_derived and max_depth == 0:
            raise ValueError(
                "require_derived=True is unsatisfiable at max_depth=0: no concept "
                "has concept_depth <= 0, so no derived concept can be placed."
            )

        last_reason: str | None = None
        for _ in range(max_attempts):
            tree = Expr.build_amount_tree(
                depth=max_depth,
                derived_prob=derived_prob,
                derived_registry=derived_registry,
                balanced=balanced,
            )
            violation = Expr.violates_protected_canonical_form(tree, derived_registry)
            if violation is not None:
                last_reason = f"Rejected because tree duplicated protected concept: {violation}"
                continue
            if require_derived and not Expr.contains_derived(tree):
                last_reason = "Rejected because tree contained no derived concepts."
                continue
            return tree, last_reason

        raise RuntimeError(
            f"Failed to sample a valid tree after {max_attempts} attempts. "
            f"Last rejection reason: {last_reason}"
        )
    
    def is_template_expr(self) -> bool:
        if isinstance(self, Leaf):
            return self.key is None
        if isinstance(self, DerivedExpr):
            return self.expr is None
        if isinstance(self, TimeAgg):
            return self.expr is None
        if isinstance(self, Literal):
            return False
        return self.left.is_template_expr() or self.right.is_template_expr()

    def expr_to_json(self) -> dict:
        if isinstance(self, Leaf):
            if self.key is not None:
                return {"leaf": self.key, "depth": 0}
            return {
                "kind": "leaf",
                "semantic_type_in": [semantic_type.value for semantic_type in (self.semantic_type_in or [SemanticType.amount])],
                "depth": 0,
            }
        if isinstance(self, Literal):
            return {"literal": self.value, "depth": 0}
        if isinstance(self, DerivedExpr):
            if self.expr is None:
                return {"kind": "derived_concept", "name": self.name, "depth": 0}
            return {
                "derived": self.name,
                "expanded": self.expr.expr_to_json(),
                "depth": self.depth if self.depth is not None else self.expr.expr_depth(),
            }
        if isinstance(self, TimeAgg):
            if self.expr is None:
                if self.op is None:
                    raise ValueError("TimeAgg template nodes must define an op.")
                return {"kind": "time_agg", "op": self.op.value, "depth": 0}
            return self.expr.expr_to_json()

        payload = {
            "op": self.op.value,
            "left": self.left.expr_to_json(),
            "right": self.right.expr_to_json(),
            "depth": self.depth if self.depth is not None else self.expr_depth(),
        }
        if self.is_template_expr():
            payload["kind"] = "node"
        return payload

    def show_expr(self) -> str:
        if isinstance(self, Leaf):
            if self.key is not None:
                return self.key
            semantic_types = ",".join(t.value for t in (self.semantic_type_in or [SemanticType.amount]))
            return f"leaf[{semantic_types}]"
        if isinstance(self, Literal):
            return str(self.value)
        if isinstance(self, DerivedExpr):
            return self.name
        if isinstance(self, TimeAgg):
            if self.expr is not None:
                return self.expr.show_expr()
            if self.op is None:
                return "time_agg(?)"
            return f"time_agg({self.op.value})"
        return f"{self.op.value}({self.left.show_expr()}, {self.right.show_expr()})"


    @staticmethod
    def parse_template_expr(obj: dict[str, Any]) -> Expr:
        kind = obj.get("kind")
        if kind == "leaf":
            raw_semantic_types = obj.get("semantic_type_in", [SemanticType.amount])
            semantic_types = [SemanticType.coerce_semantic_type(value) for value in raw_semantic_types]
            return Leaf(semantic_type_in=semantic_types)
        if kind == "derived_concept":
            return DerivedExpr(name=str(obj["name"]))
        if kind == "time_agg":
            return TimeAgg(op=Expr.coerce_operation(obj["op"]))
        if kind == "node":
            left = Expr.parse_template_expr(obj["left"])
            right = Expr.parse_template_expr(obj["right"])
            return Node(op=Expr.coerce_operation(obj["op"]), left=left, right=right)
        raise ValueError(f"Unknown tree kind: {kind}")


class Operation(StrEnum):
    sum = "sum"
    diff = "diff"
    ratio = "ratio"
    mul = "mul"
    growth = "growth"
    min = "min"
    max = "max"
    avg = "avg"

    def is_time_aggregation(self) -> bool:
        return self in {Operation.min, Operation.max, Operation.avg}

class SemanticType(StrEnum):
    amount = "amount"
    count = "count"
    price = "price"
    rate = "rate"
    ratio = "ratio"

    @staticmethod
    def coerce_semantic_type(value: SemanticType | str) -> SemanticType:
        if isinstance(value, SemanticType):
            return value
        return SemanticType(str(value))

class Atom(BaseModel):
    key: str
    concept: str
    semantic_type: SemanticType
    label: str
    entity: str
    period: str
    unit: str
    value: float
    depth: int = 0
    parent_concept: str | None = None
    role: str | None = None

    def coerce_atom(item: dict[Any, Any] | Atom) -> Atom:
        if isinstance(item, Atom):
            return item
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
        clean = {k: v for k, v in item.items() if k in allowed}
        return Atom(**clean)

class Leaf(Expr, BaseModel):
    key: str | None = None
    semantic_type_in: list[SemanticType] | None = None
    depth: int = 0

    def __init__(self, *args, **data):
        if args:
            if len(args) != 1:
                raise TypeError("Leaf accepts at most one positional argument.")
            data = {"key": args[0], **data}
        super().__init__(**data)

    def expr_depth(self) -> int:
        return 0
    
    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["depth"] = self.expr_depth()
        return data

    def flatten_leaves(self) -> list["Leaf"]:
        return [self] if self.key is not None else []

    def flatten_sum(self) -> list["Leaf"]:
        return [self] if self.key is not None else []
    
    def contains_a_derived_concept(self) -> bool:
        return False
    
class Node(Expr, BaseModel):
    op: Operation
    left: "Expr"
    right: "Expr"
    depth: int | None = None

    def __init__(self, *args, **data):
        if args:
            if len(args) != 3:
                raise TypeError("Node expects op, left, and right positional arguments.")
            op, left, right = args
            data = {"op": op, "left": left, "right": right, **data}
        super().__init__(**data)
        if self.depth is None:
            self.depth = 1 + max(self.left.expr_depth(), self.right.expr_depth())

    def expr_depth(self) -> int:
        return self.depth if self.depth is not None else 1 + max(self.left.expr_depth(), self.right.expr_depth())
    
    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["depth"] = self.depth if self.depth is not None else self.expr_depth()
        return data

    def flatten_leaves(self) -> list["Leaf"]:
        return self.left.flatten_leaves() + self.right.flatten_leaves()
    
    def flatten_sum(self) -> list["Leaf"]:
        if self.op == Operation.sum:
            return self.left.flatten_sum() + self.right.flatten_sum()
        return []
    
    def contains_a_derived_concept(self) -> bool:
        return self.left.contains_a_derived_concept() or self.right.contains_a_derived_concept()
    

class DerivedExpr(Expr, BaseModel):
    name: str
    expr: "Expr | None" = None
    depth: int | None = None

    def __init__(self, *args, **data):
        if args:
            if len(args) == 1:
                data = {"name": args[0], **data}
            elif len(args) == 2:
                data = {"name": args[0], "expr": args[1], **data}
            else:
                raise TypeError("DerivedExpr accepts name and optional expr positional arguments.")
        super().__init__(**data)

    def expr_depth(self) -> int:
        if self.expr is None:
            return 0
        return self.depth if self.depth is not None else self.expr.expr_depth()

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["depth"] = self.depth if self.depth is not None else self.expr_depth()
        return data

    def flatten_leaves(self):
        if self.expr is None:
            return []
        return self.expr.flatten_leaves()

    def flatten_sum(self) -> list["Leaf"]:
        return []
    
    def contains_a_derived_concept(self) -> bool:
        return True

class TimeAgg(Expr, BaseModel):
    name: str | None = None
    expr: "Expr | None" = None
    op: Operation | None = None
    depth: int | None = None

    def __init__(self, *args, **data):
        if args:
            if len(args) == 1:
                data = {"op": args[0], **data}
            elif len(args) == 2:
                data = {"name": args[0], "expr": args[1], **data}
            else:
                raise TypeError("TimeAgg accepts op or name/expr positional arguments.")
        super().__init__(**data)

    def expr_depth(self) -> int:
        if self.expr is None:
            return 0
        return self.depth if self.depth is not None else self.expr.expr_depth()

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["depth"] = self.depth if self.depth is not None else self.expr_depth()
        return data

    def flatten_leaves(self):
        if self.expr is None:
            return []
        return self.expr.flatten_leaves()

    def flatten_sum(self) -> list["Leaf"]:
        return []
    
    def contains_a_derived_concept(self) -> bool:
        if self.expr is None:
            return False
        return self.expr.contains_a_derived_concept()


class Literal(Expr, BaseModel):
    value: float
    depth: int = 0

    def __init__(self, *args, **data):
        if args:
            if len(args) != 1:
                raise TypeError("Literal accepts a single positional value.")
            data = {"value": args[0], **data}
        super().__init__(**data)

    def expr_depth(self) -> int:
        return 0

    def model_dump(self, *args, **kwargs):
        data = super().model_dump(*args, **kwargs)
        data["depth"] = self.depth if self.depth is not None else self.expr_depth()
        return data

    def flatten_leaves(self):
        return []
    
    def flatten_sum(self) -> list["Leaf"]:
        return []
    
    def contains_a_derived_concept(self) -> bool:
        return False

class BindEnv(BaseModel):
    entity: str | None = None
    period: str | None = None
    concept: str | None = None

@dataclass
class Store:
    concepts: list[str]
    entities: list[str]
    periods: list[str]
    _atoms : list[Atom]

    @staticmethod
    def store_from_atoms(atoms: dict[str, Atom] | list[Atom] | list[dict]) -> Store:
        if isinstance(atoms, dict):
            atom_list = [Atom.coerce_atom(atom) for atom in atoms.values()]
        else:
            atom_list = [Atom.coerce_atom(atom) for atom in atoms]
        return Store(
            concepts=sorted({atom.concept for atom in atom_list}),
            entities=sorted({atom.entity for atom in atom_list}),
            periods=sorted({atom.period for atom in atom_list}),
            _atoms=atom_list,
        )

    def choose_entity(self, env: BindEnv | None) -> str:
        if env is not None and env.entity is not None:
            return env.entity
        return random.choice(self.entities)

    def choose_period(self, env: BindEnv | None) -> str:
        if env is not None and env.period is not None:
            return env.period
        return random.choice(self.periods)

    def amount_concepts(self) -> list[str]:
        return sorted({c.concept for c in self._atoms if c.semantic_type == SemanticType.amount})

    def rate_concepts(self) -> list[str]:
        return sorted({c.concept for c in self._atoms if c.semantic_type in {SemanticType.rate, SemanticType.ratio}})

    def filter_atoms(
        self,
        *,
        semantic_types: list[SemanticType] | None = None,
        concept: str | None = None,
        entity: str | None = None,
        period: str | None = None,
    ) -> list[Atom]:
        results = self._atoms
        if semantic_types is not None:
            results = [a for a in results if a.semantic_type in semantic_types]
        if concept is not None:
            results = [a for a in results if a.concept == concept]
        if entity is not None:
            results = [a for a in results if a.entity == entity]
        if period is not None:
            results = [a for a in results if a.period == period]
        return results
    
    def instantiate_base_atom(
    self,
    *,
    semantic_types: list[SemanticType],
    env: BindEnv,
    ) -> Leaf:
        env = env or BindEnv()
        entity = env.entity or random.choice(self.entities)
        period = env.period or random.choice(self.periods)

        # Choose concept family-aware: amount concepts vs ratio/rate concepts.
        if any(t == SemanticType.amount for t in semantic_types):
            concept = env.concept or random.choice(self.amount_concepts())
        else:
            concept = env.concept or random.choice(self.rate_concepts())

        # Filter to matching atoms and bind one concrete leaf.
        candidates = self.filter_atoms(
            semantic_types=semantic_types,
            concept=concept,
            entity=entity,
            period=period,
        )
        if not candidates:
            raise ValueError(
                f"No atoms match semantic_types={semantic_types}, concept={concept}, entity={entity}, period={period}."
            )
        atom = random.choice(candidates)
        return Leaf(key=atom.key)
    

    def pick_entity_concept_two_periods(
    self,
    env: BindEnv,
    *,
    purpose: str,
    ) -> tuple[str, str, str, str]:
        """Pick one amount metric and two random distinct years for the same company."""
        entity, concept, periods = self.pick_entity_concept_all_periods(
            env=env,
            purpose=purpose,
        )
        p_left, p_right = random.sample(periods, 2)
        return entity, concept, p_left, p_right
    
    def pick_entity_concept_all_periods(
    self,
    env: BindEnv,
    *,
    purpose: str,
    ) -> tuple[str, str, list[str]]:
        """Pick one amount metric and every reporting year available for that company."""
        entity = env.entity or random.choice(self.entities)
        concept = env.concept or random.choice(self.amount_concepts())
        concept_period_atoms = self.filter_atoms(
            semantic_types=[SemanticType.amount],
            concept=concept,
            entity=entity,
        )
        periods = sorted({a.period for a in concept_period_atoms})
        if len(periods) < 2:
            raise Exception(
                f"Need at least two periods for concept={concept}, entity={entity} ({purpose})."
            )
        return entity, concept, periods
    
    def bind_op_over_all_periods(
    self,
    env: BindEnv,
    *,
    purpose: str
    ) -> list[Leaf]:
        """Bind sum/diff over a random contiguous year window for one entity+concept."""
        entity, concept, periods = self.pick_entity_concept_all_periods(env, purpose=purpose)
        selected_periods = pick_random_contiguous_period_window(periods)
        leaves = [
            self.instantiate_base_atom(
                semantic_types=[SemanticType.amount],
                env=BindEnv(entity=entity, period=p, concept=concept),
            )
            for p in selected_periods
        ]
        return leaves

    def bind_avg_over_all_periods(
    self,
    env: BindEnv,
    *,
    purpose: str,
    ) -> Expr:
        """Arithmetic mean over a random contiguous year window: sum(values) / n."""
        leaves = self.bind_op_over_all_periods(env, purpose=purpose)
        n = len(leaves)
        sum_expr = Expr.fold_narry(Operation.sum, leaves)
        return Node(op=Operation.ratio, left=sum_expr, right=Literal(float(n)))
    
    def bind_min_or_max_over_all_periods(
    self,
    op: Operation,
    env: BindEnv,
    *,
    purpose: str,
    ) -> Expr:
        """Bind min/max over a random contiguous year window for one entity+concept."""
        if op not in {Operation.min, Operation.max}:
            raise ValueError(f"expected min or max, got {op!r}")
        leaves = self.bind_op_over_all_periods(env, purpose=purpose)
        return Expr.fold_narry(op, leaves)



OPS = tuple(op.value for op in Operation)
AMOUNT_BINARY_OPS = (Operation.sum, Operation.diff, Operation.mul)
TIME_AGG_OPS = (Operation.min, Operation.max, Operation.avg)
RATIO_OPS = (Operation.ratio, Operation.growth)

DERIVED_CONCEPTS: dict[str, DerivedConcept] = {
    "gross_profit": DerivedConcept(
        name="gross_profit",
        op=Operation.diff,
        family="amount",
        concept_dept=1,
        args=["revenue", "cost_of_goods_sold"],
        protected=True,
    ),
    "operating_income": DerivedConcept(
        name="operating_income",
        op=Operation.diff,
        family="amount",
        concept_dept=2,
        args=["gross_profit", "operating_expenses"],
        protected=True,
    ),
    "pretax_income": DerivedConcept(
        name="pretax_income",
        op=Operation.diff,
        family="amount",
        concept_dept=3,
        args=["operating_income", "non_operating_expenses"],
        protected=True,
    ),
    "income_tax_expense": DerivedConcept(
        name="income_tax_expense",
        op=Operation.mul,
        family="amount",
        concept_dept=4,
        args=["pretax_income", "income_tax"],
        protected=True,
    ),
    "net_income": DerivedConcept(
        name="net_income",
        op=Operation.diff,
        family="amount",
        concept_dept=5,
        args=["pretax_income", "income_tax_expense"],
        protected=True,
    ),
    "current_assets": DerivedConcept(
        name="current_assets",
        op=Operation.sum,
        family="amount",
        concept_dept=1,
        args=["cash", "accounts_receivable", "inventories", "short_term_investments"],
        protected=True,
    ),
    "longterm_assets": DerivedConcept(
        name="longterm_assets",
        op=Operation.diff,
        family="amount",
        concept_dept=2,
        args=["total_assets", "current_assets"],
        protected=True,
    ),
    "longterm_liabilities": DerivedConcept(
        name="longterm_liabilities",
        op=Operation.diff,
        family="amount",
        concept_dept=1,
        args=["total_liabilities", "current_liabilities"],
        protected=True,
    ),
}








def load_atoms_json(path: str | Path) -> dict[str, Atom]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        items = raw.values()
    else:
        items = raw

    atoms = [Atom.coerce_atom(item) for item in items]
    return {atom.key: atom for atom in atoms}


AtomIndex = Store
TemplateLeaf = Leaf
TemplateOperatorNode = Node
TemplateDerivedConcept = DerivedExpr
TypedTemplate = Expr



def compile_tree_payload(
    tree_payload: dict,
    atoms: dict[str, Atom] | list[Atom] | list[dict],
    seed: int = 0,
) -> Expr:
    store = Store.store_from_atoms(atoms)
    derived_registry = tree_payload.get("derived_concepts") or DERIVED_CONCEPTS
    typed_tree = tree_payload.get("tree", tree_payload)

    previous_state = random.getstate()
    random.seed(seed)
    try:
        return Expr.instantiate_typed_tree(typed_tree, store, BindEnv(), derived_registry)
    finally:
        random.setstate(previous_state)


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
) -> dict:
    from evaluator import Evaluator
    from question_renderer import QuestionRenderer
    from semantic_analyzer import SemanticAnalyzer

    atoms = load_atoms_json(atoms_json_path)
    with open(tree_json_path, "r", encoding="utf-8") as f:
        tree_payload = json.load(f)

    expr = compile_tree_payload(tree_payload, atoms, seed=seed)
    result = SemanticAnalyzer(atoms).analyze(expr)
    value = Evaluator(atoms).eval(expr)
    question = QuestionRenderer().render(result)
    return {
        "expr": expr.expr_to_json(),
        "expr_str": expr.show_expr(),
        "depth": expr.expr_depth(),
        "question": question,
        "answer": value,
        "meaning": result.meaning.model_dump(),
    }


def print_analysis_tree(result, indent: int = 0) -> None:
    prefix = "  " * indent
    print(f"{prefix}Expr:     {result.expr.show_expr()}")
    print(f"{prefix}Kind:     {result.meaning.kind}")
    print(f"{prefix}Type:     {result.meaning.semantic_type}")
    print(f"{prefix}Depth:    {result.depth}")
    print(f"{prefix}Meaning:  {result.meaning.text}")
    if result.meaning.derivation:
        print(f"{prefix}How:      {result.meaning.derivation}")
    print()
    for child in result.children:
        print_analysis_tree(child, indent + 1)


Leaf.model_rebuild()
Node.model_rebuild()
DerivedExpr.model_rebuild()
TimeAgg.model_rebuild()
