"""
Interval Newton Contraction Operator

Implements the Interval Newton method for tightening variable bounds
from equality constraints h(x) = 0.

Mathematical Foundation:
Given h(x) = 0 and an interval box [x], the Interval Newton operator:
1. Computes h(x_mid) at the midpoint
2. Computes interval bounds for the Jacobian dh/dx_i
3. Applies Newton: x_i_new = x_mid - h(x_mid) / (dh/dx_i)
4. Intersects with current bounds
5. If intersection is empty -> region contains no root -> EMPTY certificate

This provides tighter bounds than FBBT for smooth equality constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .interval import Interval, IntervalEvaluator, ROUND_EPS
from ..expr_graph import (
    ExpressionGraph,
    ExprNode,
    Variable,
    Constant,
    UnaryOp,
    BinaryOp,
    OpType,
)


class IntervalNewtonStatus(Enum):
    """Status of Interval Newton contraction."""
    CONTRACTED = "contracted"  # Bounds were tightened
    EMPTY = "empty"           # Proved no root exists
    UNCHANGED = "unchanged"   # No significant change
    CONVERGED = "converged"   # Fixed point reached


@dataclass
class IntervalNewtonResult:
    """Result of Interval Newton contraction."""
    lower: np.ndarray
    upper: np.ndarray
    status: IntervalNewtonStatus
    tightened: bool
    empty: bool
    iterations: int
    certificate: Dict[str, Any]


class IntervalNewtonOperator:
    """
    Interval Newton Contraction Operator for h(x) = 0.

    Uses interval arithmetic to compute rigorous enclosures of Newton iterates,
    providing tighter bounds than pure FBBT for equality constraints.
    """

    def __init__(
        self,
        constraint_graph: ExpressionGraph,
        n_vars: int,
        max_iterations: int = 20,
        tol: float = 1e-9,
        min_progress: float = 0.01
    ):
        """
        Initialize Interval Newton for constraint h(x) = 0.

        Args:
            constraint_graph: Expression DAG for h(x)
            n_vars: Number of variables
            max_iterations: Maximum Newton iterations
            tol: Convergence tolerance
            min_progress: Minimum relative progress per iteration
        """
        self.graph = constraint_graph
        self.n_vars = n_vars
        self.max_iterations = max_iterations
        self.tol = tol
        self.min_progress = min_progress

        # Build partial derivative graphs
        self._derivative_graphs: Dict[int, ExpressionGraph] = {}
        self._build_derivative_graphs()

    def _build_derivative_graphs(self):
        """
        Build expression graphs for partial derivatives dh/dx_i.

        Uses automatic differentiation via expression graph traversal.
        """
        for var_idx in range(self.n_vars):
            if var_idx in self.graph.variables:
                deriv_graph = self._differentiate_wrt(var_idx)
                if deriv_graph is not None:
                    self._derivative_graphs[var_idx] = deriv_graph

    def _differentiate_wrt(self, var_idx: int) -> Optional[ExpressionGraph]:
        """
        Compute partial derivative dh/dx_i via reverse-mode AD on the graph.

        Returns a new expression graph representing the derivative.
        """
        if self.graph.output_node is None:
            return None

        # Build derivative using reverse-mode AD
        deriv_graph = ExpressionGraph()

        # Create variables in the derivative graph
        for i in range(self.n_vars):
            deriv_graph.variable(i)

        # Compute derivative via chain rule traversal
        node_derivs = self._compute_node_derivatives(var_idx)

        if node_derivs is None:
            return None

        output_deriv_node = node_derivs.get(self.graph.output_node.node_id)
        if output_deriv_node is None:
            # Derivative is zero
            deriv_graph.set_output(deriv_graph.constant(0.0))
            return deriv_graph

        # Build the derivative expression in the new graph
        deriv_output = self._build_deriv_expr(deriv_graph, output_deriv_node)
        deriv_graph.set_output(deriv_output)

        return deriv_graph

    def _compute_node_derivatives(self, var_idx: int) -> Optional[Dict[int, Any]]:
        """
        Compute symbolic derivatives for each node using forward-mode AD.

        Returns dict mapping node_id -> derivative expression (as tuple representation).
        """
        # Forward-mode AD: d/dx_i of each node
        derivs: Dict[int, Any] = {}

        for node in self.graph.topological_order():
            if isinstance(node, Variable):
                # d(x_j)/d(x_i) = 1 if i==j else 0
                derivs[node.node_id] = ('const', 1.0) if node.var_index == var_idx else ('const', 0.0)

            elif isinstance(node, Constant):
                derivs[node.node_id] = ('const', 0.0)

            elif isinstance(node, UnaryOp):
                child_deriv = derivs.get(node.child.node_id, ('const', 0.0))
                child_expr = ('node', node.child.node_id)
                derivs[node.node_id] = self._deriv_unary(node.op, child_expr, child_deriv)

            elif isinstance(node, BinaryOp):
                left_deriv = derivs.get(node.left.node_id, ('const', 0.0))
                right_deriv = derivs.get(node.right.node_id, ('const', 0.0))
                left_expr = ('node', node.left.node_id)
                right_expr = ('node', node.right.node_id)
                derivs[node.node_id] = self._deriv_binary(
                    node.op, left_expr, right_expr, left_deriv, right_deriv
                )

        return derivs

    def _deriv_unary(self, op: OpType, child_expr: tuple, child_deriv: tuple) -> tuple:
        """Compute derivative of unary operation."""
        if self._is_zero_deriv(child_deriv):
            return ('const', 0.0)

        # d/dx f(g(x)) = f'(g(x)) * g'(x)
        if op == OpType.NEG:
            return ('mul', ('const', -1.0), child_deriv)

        elif op == OpType.SQUARE:
            return ('mul', ('mul', ('const', 2.0), child_expr), child_deriv)

        elif op == OpType.SQRT:
            return ('div', child_deriv, ('mul', ('const', 2.0), ('sqrt', child_expr)))

        elif op == OpType.EXP:
            return ('mul', ('exp', child_expr), child_deriv)

        elif op == OpType.LOG:
            return ('div', child_deriv, child_expr)

        elif op == OpType.SIN:
            return ('mul', ('cos', child_expr), child_deriv)

        elif op == OpType.COS:
            return ('mul', ('neg', ('sin', child_expr)), child_deriv)

        elif op == OpType.ABS:
            return ('mul', ('sign', child_expr), child_deriv)

        else:
            return ('deriv', op.value, child_expr, child_deriv)

    def _deriv_binary(
        self, op: OpType,
        left_expr: tuple, right_expr: tuple,
        left_deriv: tuple, right_deriv: tuple
    ) -> tuple:
        """Compute derivative of binary operation."""
        left_zero = self._is_zero_deriv(left_deriv)
        right_zero = self._is_zero_deriv(right_deriv)

        if left_zero and right_zero:
            return ('const', 0.0)

        if op == OpType.ADD:
            if left_zero:
                return right_deriv
            if right_zero:
                return left_deriv
            return ('add', left_deriv, right_deriv)

        elif op == OpType.SUB:
            if left_zero:
                return ('neg', right_deriv)
            if right_zero:
                return left_deriv
            return ('sub', left_deriv, right_deriv)

        elif op == OpType.MUL:
            if left_zero:
                return ('mul', left_expr, right_deriv)
            if right_zero:
                return ('mul', left_deriv, right_expr)
            term1 = ('mul', left_deriv, right_expr)
            term2 = ('mul', left_expr, right_deriv)
            return ('add', term1, term2)

        elif op == OpType.DIV:
            if left_zero and right_zero:
                return ('const', 0.0)
            num = ('sub', ('mul', left_deriv, right_expr), ('mul', left_expr, right_deriv))
            denom = ('square', right_expr)
            return ('div', num, denom)

        elif op == OpType.POW:
            if right_zero:
                n_minus_1 = ('sub', right_expr, ('const', 1.0))
                return ('mul', ('mul', right_expr, ('pow', left_expr, n_minus_1)), left_deriv)
            else:
                ln_f = ('log', left_expr)
                f_pow_g = ('pow', left_expr, right_expr)
                term1 = ('mul', right_deriv, ln_f)
                term2 = ('div', ('mul', right_expr, left_deriv), left_expr)
                return ('mul', f_pow_g, ('add', term1, term2))

        else:
            return ('const', 0.0)

    def _is_zero_deriv(self, deriv: tuple) -> bool:
        """Check if derivative is zero."""
        return deriv[0] == 'const' and deriv[1] == 0.0

    def _build_deriv_expr(self, graph: ExpressionGraph, deriv_tuple: tuple) -> ExprNode:
        """Build an ExprNode from derivative tuple representation."""
        op = deriv_tuple[0]

        if op == 'const':
            return graph.constant(deriv_tuple[1])

        elif op == 'node':
            var_idx = deriv_tuple[1]
            return graph.variable(var_idx) if var_idx < self.n_vars else graph.constant(0.0)

        elif op == 'add':
            left = self._build_deriv_expr(graph, deriv_tuple[1])
            right = self._build_deriv_expr(graph, deriv_tuple[2])
            return graph.binary(OpType.ADD, left, right)

        elif op == 'sub':
            left = self._build_deriv_expr(graph, deriv_tuple[1])
            right = self._build_deriv_expr(graph, deriv_tuple[2])
            return graph.binary(OpType.SUB, left, right)

        elif op == 'mul':
            left = self._build_deriv_expr(graph, deriv_tuple[1])
            right = self._build_deriv_expr(graph, deriv_tuple[2])
            return graph.binary(OpType.MUL, left, right)

        elif op == 'div':
            left = self._build_deriv_expr(graph, deriv_tuple[1])
            right = self._build_deriv_expr(graph, deriv_tuple[2])
            return graph.binary(OpType.DIV, left, right)

        elif op == 'neg':
            child = self._build_deriv_expr(graph, deriv_tuple[1])
            return graph.unary(OpType.NEG, child)

        elif op == 'square':
            child = self._build_deriv_expr(graph, deriv_tuple[1])
            return graph.unary(OpType.SQUARE, child)

        elif op == 'sqrt':
            child = self._build_deriv_expr(graph, deriv_tuple[1])
            return graph.unary(OpType.SQRT, child)

        elif op == 'exp':
            child = self._build_deriv_expr(graph, deriv_tuple[1])
            return graph.unary(OpType.EXP, child)

        elif op == 'log':
            child = self._build_deriv_expr(graph, deriv_tuple[1])
            return graph.unary(OpType.LOG, child)

        elif op == 'sin':
            child = self._build_deriv_expr(graph, deriv_tuple[1])
            return graph.unary(OpType.SIN, child)

        elif op == 'cos':
            child = self._build_deriv_expr(graph, deriv_tuple[1])
            return graph.unary(OpType.COS, child)

        elif op == 'pow':
            left = self._build_deriv_expr(graph, deriv_tuple[1])
            right = self._build_deriv_expr(graph, deriv_tuple[2])
            return graph.binary(OpType.POW, left, right)

        elif op == 'sign':
            child = self._build_deriv_expr(graph, deriv_tuple[1])
            return child

        else:
            return graph.constant(0.0)

    def contract(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> IntervalNewtonResult:
        """
        Apply Interval Newton contraction for h(x) = 0.

        The Newton update is:
            x_i_new = x_mid - h(x_mid) / (dh/dx_i evaluated over [x])

        The result is intersected with current bounds.
        If intersection is empty, the region contains no root.

        Args:
            lower: Current lower bounds
            upper: Current upper bounds

        Returns:
            IntervalNewtonResult with tightened bounds or EMPTY certificate
        """
        lower = lower.copy()
        upper = upper.copy()

        initial_width = np.sum(upper - lower)
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Midpoint of current box
            x_mid = (lower + upper) / 2

            # Evaluate h(x_mid)
            try:
                h_mid = self.graph.evaluate(x_mid)
            except:
                return IntervalNewtonResult(
                    lower=lower,
                    upper=upper,
                    status=IntervalNewtonStatus.UNCHANGED,
                    tightened=False,
                    empty=False,
                    iterations=iterations,
                    certificate={"type": "eval_failed"}
                )

            # If h(x_mid) is nearly zero, we may have found the root
            if abs(h_mid) < self.tol:
                return IntervalNewtonResult(
                    lower=lower,
                    upper=upper,
                    status=IntervalNewtonStatus.CONVERGED,
                    tightened=initial_width - np.sum(upper - lower) > self.tol,
                    empty=False,
                    iterations=iterations,
                    certificate={
                        "type": "newton_converged",
                        "residual": abs(h_mid)
                    }
                )

            # Apply Newton contraction for each variable
            contracted_any = False

            for var_idx in range(self.n_vars):
                if var_idx not in self._derivative_graphs:
                    continue

                # Evaluate dh/dx_i over the interval
                deriv_graph = self._derivative_graphs[var_idx]
                var_intervals = {
                    i: Interval(lower[i], upper[i])
                    for i in range(self.n_vars)
                }

                try:
                    evaluator = IntervalEvaluator(deriv_graph)
                    deriv_interval, _ = evaluator.evaluate(var_intervals)
                except:
                    continue

                # Check if derivative interval contains zero
                if deriv_interval.contains_zero():
                    continue

                # Newton update: x_new = x_mid - h(x_mid) / dh
                h_mid_interval = Interval.point(h_mid)
                newton_update = h_mid_interval / deriv_interval

                # New interval for x_i
                new_lo = x_mid[var_idx] - newton_update.hi
                new_hi = x_mid[var_idx] - newton_update.lo

                # Intersect with current bounds
                intersect_lo = max(lower[var_idx], new_lo)
                intersect_hi = min(upper[var_idx], new_hi)

                # Check for empty intersection
                if intersect_lo > intersect_hi + ROUND_EPS:
                    return IntervalNewtonResult(
                        lower=lower,
                        upper=upper,
                        status=IntervalNewtonStatus.EMPTY,
                        tightened=True,
                        empty=True,
                        iterations=iterations,
                        certificate={
                            "type": "newton_empty",
                            "variable": var_idx,
                            "newton_interval": [new_lo, new_hi],
                            "current_interval": [lower[var_idx], upper[var_idx]]
                        }
                    )

                # Apply contraction
                if intersect_lo > lower[var_idx] + self.tol:
                    lower[var_idx] = intersect_lo
                    contracted_any = True
                if intersect_hi < upper[var_idx] - self.tol:
                    upper[var_idx] = intersect_hi
                    contracted_any = True

            # Check for sufficient progress
            current_width = np.sum(upper - lower)
            if initial_width > 0:
                progress = (initial_width - current_width) / initial_width
                if not contracted_any or progress < self.min_progress:
                    break
            else:
                break

        # Determine final status
        final_width = np.sum(upper - lower)
        tightened = initial_width - final_width > self.tol

        status = IntervalNewtonStatus.CONTRACTED if tightened else IntervalNewtonStatus.UNCHANGED

        return IntervalNewtonResult(
            lower=lower,
            upper=upper,
            status=status,
            tightened=tightened,
            empty=False,
            iterations=iterations,
            certificate={
                "type": "newton_contracted" if tightened else "newton_unchanged",
                "initial_width": initial_width,
                "final_width": final_width,
                "iterations": iterations
            }
        )


def apply_interval_newton_all_constraints(
    eq_constraints: List[ExpressionGraph],
    n_vars: int,
    lower: np.ndarray,
    upper: np.ndarray,
    max_outer_iterations: int = 5
) -> IntervalNewtonResult:
    """
    Apply Interval Newton for all equality constraints until fixed point.

    Args:
        eq_constraints: List of expression graphs for h_j(x) = 0
        n_vars: Number of variables
        lower: Current lower bounds
        upper: Current upper bounds
        max_outer_iterations: Maximum outer iterations

    Returns:
        IntervalNewtonResult with tightened bounds or EMPTY certificate
    """
    if not eq_constraints:
        return IntervalNewtonResult(
            lower=lower.copy(),
            upper=upper.copy(),
            status=IntervalNewtonStatus.UNCHANGED,
            tightened=False,
            empty=False,
            iterations=0,
            certificate={"type": "no_equality_constraints"}
        )

    lower = lower.copy()
    upper = upper.copy()

    # Create operators for each constraint
    operators = [
        IntervalNewtonOperator(g, n_vars)
        for g in eq_constraints
    ]

    total_iterations = 0
    initial_width = np.sum(upper - lower)

    for outer_iter in range(max_outer_iterations):
        any_tightened = False

        for i, op in enumerate(operators):
            result = op.contract(lower, upper)
            total_iterations += result.iterations

            if result.empty:
                return IntervalNewtonResult(
                    lower=lower,
                    upper=upper,
                    status=IntervalNewtonStatus.EMPTY,
                    tightened=True,
                    empty=True,
                    iterations=total_iterations,
                    certificate={
                        "type": "newton_infeasible",
                        "constraint_index": i,
                        "inner_certificate": result.certificate
                    }
                )

            if result.tightened:
                any_tightened = True
                lower = result.lower
                upper = result.upper

        if not any_tightened:
            break

    final_width = np.sum(upper - lower)
    tightened = initial_width - final_width > 1e-9

    return IntervalNewtonResult(
        lower=lower,
        upper=upper,
        status=IntervalNewtonStatus.CONTRACTED if tightened else IntervalNewtonStatus.UNCHANGED,
        tightened=tightened,
        empty=False,
        iterations=total_iterations,
        certificate={
            "type": "newton_converged",
            "outer_iterations": outer_iter + 1,
            "total_inner_iterations": total_iterations,
            "width_reduction": initial_width - final_width
        }
    )
