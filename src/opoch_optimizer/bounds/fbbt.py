"""
Feasibility-Based Bound Tightening (FBBT) - Tier 2a

The missing Delta-closure for equality and inequality constraints.

FBBT treats constraints as witness-generated refinement operators,
not mere feasibility checks. This is the key to handling curved
manifolds like x^2 + y^2 = 1.

Mathematical Foundation:
Given a region R (box bounds) and constraint c(x), FBBT is a deterministic operator:
    FBBT_c(R) = R'
such that:
    - F intersection R subset of F intersection R' (no feasible point is lost)
    - R' subset of R (tightening)
    - R' is the tightest enclosure reachable by interval forward/backward propagation

Algorithm:
1. Forward pass: Compute interval bounds for all nodes in expression DAG
2. Backward pass: From constraint target, propagate constraints back to variables
3. Iterate until fixed point (no bound changes)

This is exactly "Delta is endogenous": constraints generate new tests
and new tightenings.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

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


@dataclass
class FBBTResult:
    """Result of FBBT propagation."""
    lower: np.ndarray
    upper: np.ndarray
    tightened: bool
    empty: bool  # True if region became infeasible
    iterations: int
    certificate: Dict[str, Any]


class FBBTOperator:
    """
    Feasibility-Based Bound Tightening operator for equality constraints.

    This is the Delta-closure for equality constraints h(x) = 0.
    """

    def __init__(self, constraint_graph: ExpressionGraph, n_vars: int):
        """
        Initialize FBBT for a constraint h(x) = 0.

        Args:
            constraint_graph: Expression DAG for h(x)
            n_vars: Number of variables
        """
        self.graph = constraint_graph
        self.n_vars = n_vars
        self.max_iterations = 100
        self.tol = 1e-9

    def tighten(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> FBBTResult:
        """
        Apply FBBT to tighten variable bounds given h(x) = 0.

        This is the main Delta-closure operation.

        Args:
            lower: Current lower bounds
            upper: Current upper bounds

        Returns:
            FBBTResult with tightened bounds
        """
        lower = lower.copy()
        upper = upper.copy()

        total_tightening = 0.0
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Forward pass: compute intervals for all nodes
            node_intervals = self._forward_pass(lower, upper)

            # Check if constraint can be satisfied (0 must be in output interval)
            output_id = self.graph.output_node.node_id
            output_interval = node_intervals[output_id]

            if not output_interval.contains(0.0):
                # Region is infeasible
                return FBBTResult(
                    lower=lower,
                    upper=upper,
                    tightened=True,
                    empty=True,
                    iterations=iterations,
                    certificate={
                        "type": "fbbt_infeasible",
                        "output_interval": [output_interval.lo, output_interval.hi]
                    }
                )

            # Backward pass: propagate from h(x) = 0
            # The output must be 0, so output interval is [0, 0]
            target_intervals = {output_id: Interval(0.0, 0.0)}

            tightening = self._backward_pass(node_intervals, target_intervals, lower, upper)
            total_tightening += tightening

            # Check for empty regions after tightening
            for i in range(self.n_vars):
                if lower[i] > upper[i] + self.tol:
                    return FBBTResult(
                        lower=lower,
                        upper=upper,
                        tightened=True,
                        empty=True,
                        iterations=iterations,
                        certificate={
                            "type": "fbbt_empty_after_tightening",
                            "variable": i
                        }
                    )

            # Check for convergence
            if tightening < self.tol:
                break

        return FBBTResult(
            lower=lower,
            upper=upper,
            tightened=total_tightening > self.tol,
            empty=False,
            iterations=iterations,
            certificate={
                "type": "fbbt_converged",
                "total_tightening": total_tightening,
                "iterations": iterations
            }
        )

    def _forward_pass(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> Dict[int, Interval]:
        """Forward pass: compute interval bounds for all nodes."""
        var_intervals = {
            i: Interval(lower[i], upper[i])
            for i in range(self.n_vars)
        }

        evaluator = IntervalEvaluator(self.graph)
        _, node_intervals = evaluator.evaluate(var_intervals)

        return node_intervals

    def _backward_pass(
        self,
        node_intervals: Dict[int, Interval],
        target_intervals: Dict[int, Interval],
        lower: np.ndarray,
        upper: np.ndarray
    ) -> float:
        """
        Backward pass: propagate constraints from output back to variables.

        Returns total tightening (sum of bound reductions).
        """
        total_tightening = 0.0

        # Process nodes in reverse topological order
        nodes = list(reversed(self.graph.topological_order()))

        # Initialize target intervals for all nodes
        for node in nodes:
            if node.node_id not in target_intervals:
                target_intervals[node.node_id] = node_intervals[node.node_id]

        for node in nodes:
            target = target_intervals[node.node_id]

            if isinstance(node, Variable):
                # Tighten variable bounds
                old_lower = lower[node.var_index]
                old_upper = upper[node.var_index]

                # Intersect with target
                new_lower = max(old_lower, target.lo)
                new_upper = min(old_upper, target.hi)

                # Apply tightening
                if new_lower > old_lower:
                    lower[node.var_index] = new_lower
                    total_tightening += new_lower - old_lower

                if new_upper < old_upper:
                    upper[node.var_index] = new_upper
                    total_tightening += old_upper - new_upper

            elif isinstance(node, Constant):
                pass  # Nothing to propagate

            elif isinstance(node, UnaryOp):
                child_target = self._backward_unary(
                    node.op,
                    target,
                    node_intervals[node.child.node_id]
                )
                # Intersect with existing target
                current = target_intervals.get(node.child.node_id, node_intervals[node.child.node_id])
                target_intervals[node.child.node_id] = current.intersect(child_target)

            elif isinstance(node, BinaryOp):
                left_target, right_target = self._backward_binary(
                    node.op,
                    target,
                    node_intervals[node.left.node_id],
                    node_intervals[node.right.node_id]
                )
                # Intersect with existing targets
                current_left = target_intervals.get(node.left.node_id, node_intervals[node.left.node_id])
                current_right = target_intervals.get(node.right.node_id, node_intervals[node.right.node_id])
                target_intervals[node.left.node_id] = current_left.intersect(left_target)
                target_intervals[node.right.node_id] = current_right.intersect(right_target)

        return total_tightening

    def _backward_unary(
        self,
        op: OpType,
        target: Interval,
        child_interval: Interval
    ) -> Interval:
        """
        Backward propagation for unary operations.

        Given: w = op(x), and w in target
        Compute: tighter bounds on x
        """
        if op == OpType.NEG:
            # w = -x => x = -w
            return Interval(-target.hi, -target.lo)

        elif op == OpType.SQUARE:
            # w = x^2 => x in [-sqrt(w_hi), -sqrt(w_lo)] union [sqrt(w_lo), sqrt(w_hi)]
            # Intersect with child_interval to handle sign
            lo = max(0, target.lo)
            hi = target.hi

            if hi < 0:
                return Interval.empty()

            sqrt_lo = np.sqrt(lo) if lo >= 0 else 0
            sqrt_hi = np.sqrt(hi) if hi >= 0 else 0

            # Union of positive and negative roots
            if child_interval.hi <= 0:
                # x is negative
                return Interval(-sqrt_hi, -sqrt_lo)
            elif child_interval.lo >= 0:
                # x is positive
                return Interval(sqrt_lo, sqrt_hi)
            else:
                # x can be either sign
                return Interval(-sqrt_hi, sqrt_hi)

        elif op == OpType.SQRT:
            # w = sqrt(x) => x = w^2
            lo = target.lo
            hi = target.hi
            if lo < 0:
                lo = 0
            return Interval(lo * lo - ROUND_EPS, hi * hi + ROUND_EPS)

        elif op == OpType.EXP:
            # w = exp(x) => x = log(w)
            lo = target.lo
            hi = target.hi
            if lo <= 0:
                lo = ROUND_EPS
            if hi <= 0:
                return Interval.empty()
            return Interval(np.log(lo) - ROUND_EPS, np.log(hi) + ROUND_EPS)

        elif op == OpType.LOG:
            # w = log(x) => x = exp(w)
            lo = target.lo
            hi = target.hi
            return Interval(np.exp(lo) - ROUND_EPS, np.exp(hi) + ROUND_EPS)

        elif op == OpType.ABS:
            # w = |x| => x in [-w_hi, -w_lo] union [w_lo, w_hi]
            if target.lo >= 0:
                if child_interval.hi <= 0:
                    return Interval(-target.hi, -target.lo)
                elif child_interval.lo >= 0:
                    return Interval(target.lo, target.hi)
                else:
                    return Interval(-target.hi, target.hi)
            else:
                return Interval(-target.hi, target.hi)

        elif op == OpType.SIN:
            # w = sin(x) => x = arcsin(w) + 2k*pi
            # This is complex; return conservative bounds
            return child_interval

        elif op == OpType.COS:
            # w = cos(x) => x = arccos(w) + 2k*pi
            return child_interval

        else:
            # Default: no tightening
            return child_interval

    def _backward_binary(
        self,
        op: OpType,
        target: Interval,
        left_interval: Interval,
        right_interval: Interval
    ) -> Tuple[Interval, Interval]:
        """
        Backward propagation for binary operations.

        Given: w = op(x, y), and w in target
        Compute: tighter bounds on x and y
        """
        if op == OpType.ADD:
            # w = x + y => x = w - y, y = w - x
            left_target = Interval(
                target.lo - right_interval.hi - ROUND_EPS,
                target.hi - right_interval.lo + ROUND_EPS
            )
            right_target = Interval(
                target.lo - left_interval.hi - ROUND_EPS,
                target.hi - left_interval.lo + ROUND_EPS
            )
            return left_target, right_target

        elif op == OpType.SUB:
            # w = x - y => x = w + y, y = x - w
            left_target = Interval(
                target.lo + right_interval.lo - ROUND_EPS,
                target.hi + right_interval.hi + ROUND_EPS
            )
            right_target = Interval(
                left_interval.lo - target.hi - ROUND_EPS,
                left_interval.hi - target.lo + ROUND_EPS
            )
            return left_target, right_target

        elif op == OpType.MUL:
            # w = x * y
            # x = w / y (if y doesn't contain 0)
            # y = w / x (if x doesn't contain 0)

            left_target = left_interval
            right_target = right_interval

            # Tighten x bounds
            if not right_interval.contains_zero():
                # x = w / y
                div_result = target / right_interval
                left_target = left_interval.intersect(div_result)

            # Tighten y bounds
            if not left_interval.contains_zero():
                # y = w / x
                div_result = target / left_interval
                right_target = right_interval.intersect(div_result)

            return left_target, right_target

        elif op == OpType.DIV:
            # w = x / y => x = w * y, y = x / w
            left_target = target * right_interval

            if not target.contains_zero():
                right_target = left_interval / target
            else:
                right_target = right_interval

            return left_target.intersect(left_interval), right_target.intersect(right_interval)

        elif op == OpType.POW:
            # w = x^y - complex, return conservative
            return left_interval, right_interval

        else:
            return left_interval, right_interval


class FBBTInequalityOperator:
    """
    Feasibility-Based Bound Tightening operator for inequality constraints.

    This is the Delta-closure for inequality constraints g(x) <= 0.

    For g(x) <= 0:
    - Forward pass: compute interval for g(x)
    - If g.lo > 0 -> infeasible (region cannot satisfy constraint)
    - Backward pass: propagate g(x) in (-inf, 0] through DAG
    """

    def __init__(self, constraint_graph: ExpressionGraph, n_vars: int):
        """
        Initialize FBBT for a constraint g(x) <= 0.

        Args:
            constraint_graph: Expression DAG for g(x)
            n_vars: Number of variables
        """
        self.graph = constraint_graph
        self.n_vars = n_vars
        self.max_iterations = 100
        self.tol = 1e-9

    def tighten(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> FBBTResult:
        """
        Apply FBBT to tighten variable bounds given g(x) <= 0.

        The key difference from equality FBBT:
        - Target interval is (-inf, 0] instead of [0, 0]

        Args:
            lower: Current lower bounds
            upper: Current upper bounds

        Returns:
            FBBTResult with tightened bounds
        """
        lower = lower.copy()
        upper = upper.copy()

        total_tightening = 0.0
        iterations = 0

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Forward pass: compute intervals for all nodes
            node_intervals = self._forward_pass(lower, upper)

            # Check if constraint can be satisfied
            output_id = self.graph.output_node.node_id
            output_interval = node_intervals[output_id]

            # For g(x) <= 0: if g.lo > 0, the constraint cannot be satisfied
            if output_interval.lo > self.tol:
                # Region is infeasible
                return FBBTResult(
                    lower=lower,
                    upper=upper,
                    tightened=True,
                    empty=True,
                    iterations=iterations,
                    certificate={
                        "type": "fbbt_ineq_infeasible",
                        "output_interval": [output_interval.lo, output_interval.hi],
                        "reason": "g(x) > 0 for all x in region"
                    }
                )

            # If g.hi <= 0, constraint is always satisfied - no tightening possible
            if output_interval.hi <= 0:
                return FBBTResult(
                    lower=lower,
                    upper=upper,
                    tightened=total_tightening > self.tol,
                    empty=False,
                    iterations=iterations,
                    certificate={
                        "type": "fbbt_ineq_always_satisfied",
                        "output_interval": [output_interval.lo, output_interval.hi]
                    }
                )

            # Backward pass: propagate from g(x) <= 0
            # The output must be in (-inf, 0], so target interval is (-inf, 0]
            target_intervals = {output_id: Interval(float('-inf'), 0.0)}

            tightening = self._backward_pass(node_intervals, target_intervals, lower, upper)
            total_tightening += tightening

            # Check for empty regions after tightening
            for i in range(self.n_vars):
                if lower[i] > upper[i] + self.tol:
                    return FBBTResult(
                        lower=lower,
                        upper=upper,
                        tightened=True,
                        empty=True,
                        iterations=iterations,
                        certificate={
                            "type": "fbbt_ineq_empty_after_tightening",
                            "variable": i
                        }
                    )

            # Check for convergence
            if tightening < self.tol:
                break

        return FBBTResult(
            lower=lower,
            upper=upper,
            tightened=total_tightening > self.tol,
            empty=False,
            iterations=iterations,
            certificate={
                "type": "fbbt_ineq_converged",
                "total_tightening": total_tightening,
                "iterations": iterations
            }
        )

    def _forward_pass(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> Dict[int, Interval]:
        """Forward pass: compute interval bounds for all nodes."""
        var_intervals = {
            i: Interval(lower[i], upper[i])
            for i in range(self.n_vars)
        }

        evaluator = IntervalEvaluator(self.graph)
        _, node_intervals = evaluator.evaluate(var_intervals)

        return node_intervals

    def _backward_pass(
        self,
        node_intervals: Dict[int, Interval],
        target_intervals: Dict[int, Interval],
        lower: np.ndarray,
        upper: np.ndarray
    ) -> float:
        """
        Backward pass: propagate constraints from output back to variables.

        For inequality constraints, the target is (-inf, 0] at the output.
        """
        total_tightening = 0.0

        # Process nodes in reverse topological order
        nodes = list(reversed(self.graph.topological_order()))

        # Initialize target intervals for all nodes
        for node in nodes:
            if node.node_id not in target_intervals:
                target_intervals[node.node_id] = node_intervals[node.node_id]

        for node in nodes:
            target = target_intervals[node.node_id]

            if isinstance(node, Variable):
                # Tighten variable bounds
                old_lower = lower[node.var_index]
                old_upper = upper[node.var_index]

                # Intersect with target
                new_lower = max(old_lower, target.lo)
                new_upper = min(old_upper, target.hi)

                # Apply tightening
                if new_lower > old_lower:
                    lower[node.var_index] = new_lower
                    total_tightening += new_lower - old_lower

                if new_upper < old_upper:
                    upper[node.var_index] = new_upper
                    total_tightening += old_upper - new_upper

            elif isinstance(node, Constant):
                pass  # Nothing to propagate

            elif isinstance(node, UnaryOp):
                child_target = self._backward_unary(
                    node.op,
                    target,
                    node_intervals[node.child.node_id]
                )
                # Intersect with existing target
                current = target_intervals.get(node.child.node_id, node_intervals[node.child.node_id])
                target_intervals[node.child.node_id] = current.intersect(child_target)

            elif isinstance(node, BinaryOp):
                left_target, right_target = self._backward_binary(
                    node.op,
                    target,
                    node_intervals[node.left.node_id],
                    node_intervals[node.right.node_id]
                )
                # Intersect with existing targets
                current_left = target_intervals.get(node.left.node_id, node_intervals[node.left.node_id])
                current_right = target_intervals.get(node.right.node_id, node_intervals[node.right.node_id])
                target_intervals[node.left.node_id] = current_left.intersect(left_target)
                target_intervals[node.right.node_id] = current_right.intersect(right_target)

        return total_tightening

    def _backward_unary(
        self,
        op: OpType,
        target: Interval,
        child_interval: Interval
    ) -> Interval:
        """Backward propagation for unary operations (same as equality FBBT)."""
        if op == OpType.NEG:
            return Interval(-target.hi, -target.lo)

        elif op == OpType.SQUARE:
            lo = max(0, target.lo)
            hi = target.hi

            if hi < 0:
                return Interval.empty()

            sqrt_lo = np.sqrt(lo) if lo >= 0 else 0
            sqrt_hi = np.sqrt(hi) if hi >= 0 else 0

            if child_interval.hi <= 0:
                return Interval(-sqrt_hi, -sqrt_lo)
            elif child_interval.lo >= 0:
                return Interval(sqrt_lo, sqrt_hi)
            else:
                return Interval(-sqrt_hi, sqrt_hi)

        elif op == OpType.SQRT:
            lo = target.lo
            hi = target.hi
            if lo < 0:
                lo = 0
            return Interval(lo * lo - ROUND_EPS, hi * hi + ROUND_EPS)

        elif op == OpType.EXP:
            lo = target.lo
            hi = target.hi
            if lo <= 0:
                lo = ROUND_EPS
            if hi <= 0:
                return Interval.empty()
            return Interval(np.log(lo) - ROUND_EPS, np.log(hi) + ROUND_EPS)

        elif op == OpType.LOG:
            lo = target.lo
            hi = target.hi
            return Interval(np.exp(lo) - ROUND_EPS, np.exp(hi) + ROUND_EPS)

        elif op == OpType.ABS:
            if target.lo >= 0:
                if child_interval.hi <= 0:
                    return Interval(-target.hi, -target.lo)
                elif child_interval.lo >= 0:
                    return Interval(target.lo, target.hi)
                else:
                    return Interval(-target.hi, target.hi)
            else:
                return Interval(-target.hi, target.hi)

        else:
            return child_interval

    def _backward_binary(
        self,
        op: OpType,
        target: Interval,
        left_interval: Interval,
        right_interval: Interval
    ) -> Tuple[Interval, Interval]:
        """Backward propagation for binary operations (same as equality FBBT)."""
        if op == OpType.ADD:
            left_target = Interval(
                target.lo - right_interval.hi - ROUND_EPS,
                target.hi - right_interval.lo + ROUND_EPS
            )
            right_target = Interval(
                target.lo - left_interval.hi - ROUND_EPS,
                target.hi - left_interval.lo + ROUND_EPS
            )
            return left_target, right_target

        elif op == OpType.SUB:
            left_target = Interval(
                target.lo + right_interval.lo - ROUND_EPS,
                target.hi + right_interval.hi + ROUND_EPS
            )
            right_target = Interval(
                left_interval.lo - target.hi - ROUND_EPS,
                left_interval.hi - target.lo + ROUND_EPS
            )
            return left_target, right_target

        elif op == OpType.MUL:
            left_target = left_interval
            right_target = right_interval

            if not right_interval.contains_zero():
                div_result = target / right_interval
                left_target = left_interval.intersect(div_result)

            if not left_interval.contains_zero():
                div_result = target / left_interval
                right_target = right_interval.intersect(div_result)

            return left_target, right_target

        elif op == OpType.DIV:
            left_target = target * right_interval

            if not target.contains_zero():
                right_target = left_interval / target
            else:
                right_target = right_interval

            return left_target.intersect(left_interval), right_target.intersect(right_interval)

        elif op == OpType.POW:
            return left_interval, right_interval

        else:
            return left_interval, right_interval


def apply_fbbt_all_constraints(
    eq_constraints: List[ExpressionGraph] = None,
    ineq_constraints: List[ExpressionGraph] = None,
    n_vars: int = 0,
    lower: np.ndarray = None,
    upper: np.ndarray = None,
    max_iterations: int = 10
) -> FBBTResult:
    """
    Apply FBBT for all constraints (equality and inequality) until fixed point.

    This is the complete Delta-closure for all constraints.

    Args:
        eq_constraints: List of expression graphs for h_j(x) = 0
        ineq_constraints: List of expression graphs for g_i(x) <= 0
        n_vars: Number of variables
        lower: Current lower bounds
        upper: Current upper bounds
        max_iterations: Maximum outer iterations

    Returns:
        FBBTResult with tightened bounds
    """
    eq_constraints = eq_constraints or []
    ineq_constraints = ineq_constraints or []

    if not eq_constraints and not ineq_constraints:
        return FBBTResult(
            lower=lower.copy() if lower is not None else np.array([]),
            upper=upper.copy() if upper is not None else np.array([]),
            tightened=False,
            empty=False,
            iterations=0,
            certificate={"type": "no_constraints"}
        )

    lower = lower.copy()
    upper = upper.copy()

    # Create operators for equality constraints
    eq_operators = [FBBTOperator(g, n_vars) for g in eq_constraints]

    # Create operators for inequality constraints
    ineq_operators = [FBBTInequalityOperator(g, n_vars) for g in ineq_constraints]

    total_tightening = 0.0
    total_iterations = 0
    any_tightened = False

    for outer_iter in range(max_iterations):
        iteration_tightened = False

        # Apply equality FBBT
        for i, op in enumerate(eq_operators):
            result = op.tighten(lower, upper)
            total_iterations += result.iterations

            if result.empty:
                return FBBTResult(
                    lower=lower,
                    upper=upper,
                    tightened=True,
                    empty=True,
                    iterations=total_iterations,
                    certificate={
                        "type": "fbbt_eq_infeasible",
                        "constraint_index": i,
                        "inner_certificate": result.certificate
                    }
                )

            if result.tightened:
                iteration_tightened = True
                any_tightened = True
                lower = result.lower
                upper = result.upper

        # Apply inequality FBBT
        for i, op in enumerate(ineq_operators):
            result = op.tighten(lower, upper)
            total_iterations += result.iterations

            if result.empty:
                return FBBTResult(
                    lower=lower,
                    upper=upper,
                    tightened=True,
                    empty=True,
                    iterations=total_iterations,
                    certificate={
                        "type": "fbbt_ineq_infeasible",
                        "constraint_index": i,
                        "inner_certificate": result.certificate
                    }
                )

            if result.tightened:
                iteration_tightened = True
                any_tightened = True
                lower = result.lower
                upper = result.upper

        if not iteration_tightened:
            break

    return FBBTResult(
        lower=lower,
        upper=upper,
        tightened=any_tightened,
        empty=False,
        iterations=total_iterations,
        certificate={
            "type": "fbbt_converged",
            "outer_iterations": outer_iter + 1,
            "total_inner_iterations": total_iterations,
            "n_eq_constraints": len(eq_constraints),
            "n_ineq_constraints": len(ineq_constraints)
        }
    )
