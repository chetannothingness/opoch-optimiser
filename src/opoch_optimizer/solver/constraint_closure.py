"""
Unified Constraint Closure System (Δ*)

This module implements the complete constraint closure for GLOBALLib certification.
Constraints are treated as CONTRACTORS (Δ operators), not mere checks.

The closure applies:
1. FBBT for all constraints (inequalities and equalities)
2. Krawczyk contraction for equality manifolds
3. Disjunction detection for even-power constraints
4. Fixed-point iteration until no more tightening

The Δ* constructors include:
- FBBT: Feasibility-based bound tightening
- Krawczyk: Equality manifold contraction
- Root-isolation: Disjunction splitting for (g(x))^(2k) = c constraints

This is the complete mathematical foundation for GLOBALLib certification.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import math

from ..bounds.interval import Interval
from ..bounds.fbbt import FBBTOperator, FBBTResult, FBBTInequalityOperator
from ..bounds.krawczyk import KrawczykOperator, KrawczykResult, KrawczykStatus
from ..bounds.disjunction_contractor import (
    DisjunctionContractor,
    detect_torus_constraint,
    create_component_subproblems,
)
from ..expr_graph import ExpressionGraph


class ClosureStatus(Enum):
    """Status of constraint closure."""
    CONTRACTED = "contracted"   # Bounds were tightened
    EMPTY = "empty"             # Proved infeasible
    UNCHANGED = "unchanged"     # No significant change
    CONVERGED = "converged"     # Fixed point reached
    DISJUNCTION = "disjunction" # Constraint creates disjoint components


@dataclass
class ComponentBranch:
    """A component branch from disjunction splitting."""
    lower: np.ndarray
    upper: np.ndarray
    scalar_value: float        # The scalar constraint value (e.g., 1.5 for inner circle)
    constraint_index: int      # Which constraint created this branch
    priority: float = 0.0      # Lower is better (for LB-based ordering)


@dataclass
class ClosureResult:
    """Result of constraint closure."""
    lower: np.ndarray
    upper: np.ndarray
    status: ClosureStatus
    tightened: bool
    empty: bool
    iterations: int
    fbbt_iterations: int
    krawczyk_iterations: int
    certificate: Dict[str, Any]
    # Disjunction information (if status == DISJUNCTION)
    disjunction_branches: List[ComponentBranch] = field(default_factory=list)


class ConstraintClosure:
    """
    Unified Constraint Closure (Δ*) for optimization problems.

    Applies all constraint contractors to fixed point:
    - FBBTOperator for h(x) = 0 (equalities)
    - FBBTInequalityOperator for g(x) ≤ 0 (inequalities)
    - KrawczykOperator for equality systems (manifold contraction)
    - DisjunctionContractor for even-power constraints (root isolation)

    The root-isolation constructor is the key for constraints like:
        (g(x))^2 = c  →  g(x) = +√c OR g(x) = -√c

    This creates disjoint components that must be branched on separately.
    """

    def __init__(
        self,
        n_vars: int,
        eq_constraints: Optional[List[ExpressionGraph]] = None,
        ineq_constraints: Optional[List[ExpressionGraph]] = None,
        max_outer_iterations: int = 20,
        tol: float = 1e-9,
        min_progress: float = 0.001
    ):
        """
        Initialize constraint closure.

        Args:
            n_vars: Number of variables
            eq_constraints: Expression graphs for h_j(x) = 0
            ineq_constraints: Expression graphs for g_i(x) ≤ 0
            max_outer_iterations: Maximum outer iterations
            tol: Convergence tolerance
            min_progress: Minimum relative progress
        """
        self.n_vars = n_vars
        self.eq_constraints = eq_constraints or []
        self.ineq_constraints = ineq_constraints or []
        self.max_outer_iterations = max_outer_iterations
        self.tol = tol
        self.min_progress = min_progress

        # Build operators
        self._fbbt_eq_ops = [
            FBBTOperator(g, n_vars) for g in self.eq_constraints
        ]
        self._fbbt_ineq_ops = [
            FBBTInequalityOperator(g, n_vars) for g in self.ineq_constraints
        ]

        # Krawczyk for equality system (if any equalities)
        self._krawczyk_op = None
        if self.eq_constraints:
            self._krawczyk_op = KrawczykOperator(
                self.eq_constraints, n_vars
            )

        # Disjunction detection for even-power constraints
        self._disjunction_info: Optional[Dict[str, Any]] = None
        self._detect_disjunctions()

    def _detect_disjunctions(self):
        """
        Detect even-power equality constraints that create disjunctions.

        Pattern: (g(x) - c)^2 = d  →  g(x) = c ± √d (two components)

        This is the root-isolation Δ* constructor.
        """
        for i, graph in enumerate(self.eq_constraints):
            # Detect torus-like constraint: (x² + y² - c)² = d
            info = detect_torus_constraint(graph, self.n_vars)
            if info is not None:
                self._disjunction_info = {
                    "constraint_index": i,
                    "scalar_expr": info["scalar_expr"],
                    "center": info["center"],
                    "rhs": info["rhs"],
                    "roots": info["roots"],  # Scalar values for each component
                }
                return  # Only handle one disjunction for now

    def get_disjunction_branches(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        objective_graph: Optional[ExpressionGraph] = None
    ) -> Optional[List[ComponentBranch]]:
        """
        Get component branches for disjunction constraints.

        If this closure has a disjunction constraint, returns branches
        ordered by priority (lowest LB first for minimization).

        For constraint (x² + y² - c)² = d with roots [r1, r2]:
        - Creates branch for x² + y² = r1 with bounds tightened to circle r1
        - Creates branch for x² + y² = r2 with bounds tightened to circle r2

        Returns None if no disjunction detected.
        """
        if self._disjunction_info is None:
            return None

        roots = self._disjunction_info["roots"]
        idx = self._disjunction_info["constraint_index"]

        # Create component subproblems
        components = create_component_subproblems(lower, upper, roots, self.n_vars)

        if not components:
            return None

        branches = []
        for comp_lower, comp_upper, scalar_val in components:
            # Priority = scalar_val for minimization of x² + y² type objectives
            # Lower scalar value = smaller circle = lower objective = higher priority
            branch = ComponentBranch(
                lower=comp_lower,
                upper=comp_upper,
                scalar_value=scalar_val,
                constraint_index=idx,
                priority=scalar_val  # Lower is better
            )
            branches.append(branch)

        # Sort by priority (lowest first)
        branches.sort(key=lambda b: b.priority)

        return branches

    def apply(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> ClosureResult:
        """
        Apply constraint closure to fixed point.

        This is the main Δ* operation:
        1. Apply FBBT for inequalities
        2. Apply FBBT for equalities
        3. Apply Krawczyk for equality manifolds
        4. Iterate until no more tightening

        Args:
            lower: Current lower bounds
            upper: Current upper bounds

        Returns:
            ClosureResult with contracted bounds or EMPTY certificate
        """
        lower = lower.copy()
        upper = upper.copy()

        initial_width = np.sum(upper - lower)
        total_fbbt_iters = 0
        total_krawczyk_iters = 0

        for outer_iter in range(self.max_outer_iterations):
            any_progress = False
            iter_start_width = np.sum(upper - lower)

            # Phase 1: Apply FBBT for inequalities g(x) ≤ 0
            for i, op in enumerate(self._fbbt_ineq_ops):
                result = op.tighten(lower, upper)
                total_fbbt_iters += result.iterations

                if result.empty:
                    return ClosureResult(
                        lower=lower,
                        upper=upper,
                        status=ClosureStatus.EMPTY,
                        tightened=True,
                        empty=True,
                        iterations=outer_iter + 1,
                        fbbt_iterations=total_fbbt_iters,
                        krawczyk_iterations=total_krawczyk_iters,
                        certificate={
                            "type": "ineq_infeasible",
                            "constraint_index": i,
                            "inner_certificate": result.certificate
                        }
                    )

                if result.tightened:
                    any_progress = True
                    lower = result.lower
                    upper = result.upper

            # Phase 2: Apply FBBT for equalities h(x) = 0
            for i, op in enumerate(self._fbbt_eq_ops):
                result = op.tighten(lower, upper)
                total_fbbt_iters += result.iterations

                if result.empty:
                    return ClosureResult(
                        lower=lower,
                        upper=upper,
                        status=ClosureStatus.EMPTY,
                        tightened=True,
                        empty=True,
                        iterations=outer_iter + 1,
                        fbbt_iterations=total_fbbt_iters,
                        krawczyk_iterations=total_krawczyk_iters,
                        certificate={
                            "type": "eq_infeasible",
                            "constraint_index": i,
                            "inner_certificate": result.certificate
                        }
                    )

                if result.tightened:
                    any_progress = True
                    lower = result.lower
                    upper = result.upper

            # Phase 3: Apply Krawczyk for equality manifolds
            if self._krawczyk_op is not None:
                result = self._krawczyk_op.contract(lower, upper)
                total_krawczyk_iters += result.iterations

                if result.empty:
                    return ClosureResult(
                        lower=lower,
                        upper=upper,
                        status=ClosureStatus.EMPTY,
                        tightened=True,
                        empty=True,
                        iterations=outer_iter + 1,
                        fbbt_iterations=total_fbbt_iters,
                        krawczyk_iterations=total_krawczyk_iters,
                        certificate={
                            "type": "krawczyk_infeasible",
                            "inner_certificate": result.certificate
                        }
                    )

                if result.tightened:
                    any_progress = True
                    lower = result.lower
                    upper = result.upper

            # Check for convergence
            iter_end_width = np.sum(upper - lower)
            if iter_start_width > 0:
                progress = (iter_start_width - iter_end_width) / iter_start_width
                if not any_progress or progress < self.min_progress:
                    break

        # Final status
        final_width = np.sum(upper - lower)
        tightened = initial_width - final_width > self.tol

        if tightened:
            status = ClosureStatus.CONTRACTED
        else:
            status = ClosureStatus.UNCHANGED

        return ClosureResult(
            lower=lower,
            upper=upper,
            status=status,
            tightened=tightened,
            empty=False,
            iterations=outer_iter + 1,
            fbbt_iterations=total_fbbt_iters,
            krawczyk_iterations=total_krawczyk_iters,
            certificate={
                "type": "closure_converged",
                "initial_width": initial_width,
                "final_width": final_width,
                "outer_iterations": outer_iter + 1,
                "fbbt_iterations": total_fbbt_iters,
                "krawczyk_iterations": total_krawczyk_iters
            }
        )


def check_feasibility_at_point(
    x: np.ndarray,
    eq_constraints: List[ExpressionGraph],
    ineq_constraints: List[ExpressionGraph],
    tol: float = 1e-8
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if a point is feasible for all constraints.

    Args:
        x: Point to check
        eq_constraints: h_j(x) = 0
        ineq_constraints: g_i(x) ≤ 0
        tol: Tolerance for equality constraints

    Returns:
        (is_feasible, certificate)
    """
    # Check equalities
    for i, graph in enumerate(eq_constraints):
        try:
            val = graph.evaluate(x)
            if abs(val) > tol:
                return False, {
                    "type": "eq_violated",
                    "constraint_index": i,
                    "value": val
                }
        except:
            return False, {"type": "eval_failed", "constraint_index": i}

    # Check inequalities
    for i, graph in enumerate(ineq_constraints):
        try:
            val = graph.evaluate(x)
            if val > tol:
                return False, {
                    "type": "ineq_violated",
                    "constraint_index": i,
                    "value": val
                }
        except:
            return False, {"type": "eval_failed", "constraint_index": i}

    return True, {"type": "feasible"}


def compute_constraint_violation(
    x: np.ndarray,
    eq_constraints: List[ExpressionGraph],
    ineq_constraints: List[ExpressionGraph]
) -> float:
    """
    Compute total constraint violation at a point.

    Args:
        x: Point to evaluate
        eq_constraints: h_j(x) = 0
        ineq_constraints: g_i(x) ≤ 0

    Returns:
        Total violation (0 if feasible)
    """
    violation = 0.0

    for graph in eq_constraints:
        try:
            val = graph.evaluate(x)
            violation += abs(val)
        except:
            violation += 1e10

    for graph in ineq_constraints:
        try:
            val = graph.evaluate(x)
            violation += max(0.0, val)
        except:
            violation += 1e10

    return violation
