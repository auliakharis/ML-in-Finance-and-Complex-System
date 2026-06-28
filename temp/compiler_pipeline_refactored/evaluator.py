"""Numeric evaluation for bound expression trees."""
from __future__ import annotations


from temp.compiler_pipeline_refactored.tree import Atom, DerivedExpr, Expr, Leaf, Literal, Node, Operation


class Evaluator:
    def __init__(self, atoms: dict[str, Atom]):
        self.atoms = atoms

    def eval(self, expr: Expr) -> float:
        if isinstance(expr, Leaf):
            return float(self.atoms[expr.key].value)
        if isinstance(expr, Literal):
            return float(expr.value)
        if isinstance(expr, DerivedExpr):
            return self.eval(expr.expr)

        assert isinstance(expr, Node)
        lv = self.eval(expr.left)
        rv = self.eval(expr.right)

        if expr.op == Operation.sum:
            return lv + rv
        if expr.op == Operation.diff:
            return lv - rv
        if expr.op == Operation.ratio:
            if rv == 0:
                raise ZeroDivisionError("Division by zero.")
            return lv / rv
        if expr.op == Operation.mul:
            return lv * rv
        if expr.op == Operation.growth:
            if rv == 0:
                raise ZeroDivisionError("Division by zero in growth base.")
            return (lv - rv) / rv
        if expr.op == Operation.min:
            return min(lv, rv)
        if expr.op == Operation.max:
            return max(lv, rv)

        raise ValueError(f"Unsupported op: {expr.op}")
