"""
Krawczyk Contractor for Equality Systems

The Krawczyk operator is a certified contractor for systems h(x) = 0.
It is more robust than Interval Newton for multi-dimensional systems.

Mathematical Foundation:
Given h: ℝⁿ → ℝᵐ and a box R, the Krawczyk operator is:

    K(R) = m - Y·h(m) + (I - Y·J_h(R))·(R - m)

where:
    - m = midpoint of R
    - J_h(R) = interval Jacobian of h over R
    - Y = preconditioner (approx inverse of point Jacobian at m)

Properties:
    - K(R) ∩ R = ∅  ⟹  no root in R (EMPTY certificate)
    - K(R) ⊆ R      ⟹  unique root in R (existence + uniqueness)
    - K(R) ∩ R ≠ ∅  ⟹  contract to R ∩ K(R)

This is the "kill shot" for circle/manifold constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .interval import Interval, IntervalEvaluator, ROUND_EPS
from ..expr_graph import ExpressionGraph


class KrawczykStatus(Enum):
    """Status of Krawczyk contraction."""
    CONTRACTED = "contracted"       # Bounds were tightened
    EMPTY = "empty"                 # Proved no root exists
    UNIQUE = "unique"               # K(R) ⊆ R, unique root exists
    UNCHANGED = "unchanged"         # No significant change


@dataclass
class KrawczykResult:
    """Result of Krawczyk contraction."""
    lower: np.ndarray
    upper: np.ndarray
    status: KrawczykStatus
    tightened: bool
    empty: bool
    unique_root: bool
    iterations: int
    certificate: Dict[str, Any]


class KrawczykOperator:
    """
    Krawczyk Contractor for equality systems h(x) = 0.

    For a system of m equations in n variables:
        h₁(x) = 0
        h₂(x) = 0
        ...
        hₘ(x) = 0

    The Krawczyk operator provides certified contraction or refutation.
    """

    def __init__(
        self,
        constraint_graphs: List[ExpressionGraph],
        n_vars: int,
        max_iterations: int = 10,
        tol: float = 1e-9,
        min_progress: float = 0.01
    ):
        """
        Initialize Krawczyk for system h(x) = 0.

        Args:
            constraint_graphs: List of expression DAGs for h_j(x)
            n_vars: Number of variables
            max_iterations: Maximum iterations
            tol: Convergence tolerance
            min_progress: Minimum relative progress per iteration
        """
        self.graphs = constraint_graphs
        self.n_vars = n_vars
        self.n_eqs = len(constraint_graphs)
        self.max_iterations = max_iterations
        self.tol = tol
        self.min_progress = min_progress

        # Build Jacobian graphs (∂h_i/∂x_j)
        self._jacobian_graphs: List[List[Optional[ExpressionGraph]]] = []
        self._build_jacobian_graphs()

    def _build_jacobian_graphs(self):
        """Build expression graphs for Jacobian entries ∂h_i/∂x_j."""
        from .interval_newton import IntervalNewtonOperator

        for i, graph in enumerate(self.graphs):
            row = []
            for j in range(self.n_vars):
                # Use IntervalNewtonOperator's differentiation
                op = IntervalNewtonOperator(graph, self.n_vars)
                if j in op._derivative_graphs:
                    row.append(op._derivative_graphs[j])
                else:
                    row.append(None)
            self._jacobian_graphs.append(row)

    def _eval_h(self, x: np.ndarray) -> np.ndarray:
        """Evaluate h(x) at a point."""
        result = np.zeros(self.n_eqs)
        for i, graph in enumerate(self.graphs):
            try:
                result[i] = graph.evaluate(x)
            except:
                result[i] = np.nan
        return result

    def _eval_jacobian_point(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian at a point."""
        J = np.zeros((self.n_eqs, self.n_vars))
        for i in range(self.n_eqs):
            for j in range(self.n_vars):
                graph = self._jacobian_graphs[i][j]
                if graph is not None:
                    try:
                        J[i, j] = graph.evaluate(x)
                    except:
                        J[i, j] = 0.0
        return J

    def _eval_jacobian_interval(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate interval Jacobian over a box.

        Returns (J_lo, J_hi) where J_lo[i,j] ≤ ∂h_i/∂x_j ≤ J_hi[i,j].
        """
        J_lo = np.zeros((self.n_eqs, self.n_vars))
        J_hi = np.zeros((self.n_eqs, self.n_vars))

        var_intervals = {
            k: Interval(lower[k], upper[k])
            for k in range(self.n_vars)
        }

        for i in range(self.n_eqs):
            for j in range(self.n_vars):
                graph = self._jacobian_graphs[i][j]
                if graph is not None:
                    try:
                        evaluator = IntervalEvaluator(graph)
                        interval, _ = evaluator.evaluate(var_intervals)
                        J_lo[i, j] = interval.lo
                        J_hi[i, j] = interval.hi
                    except:
                        J_lo[i, j] = -1e10
                        J_hi[i, j] = 1e10
                else:
                    J_lo[i, j] = 0.0
                    J_hi[i, j] = 0.0

        return J_lo, J_hi

    def _compute_preconditioner(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute preconditioner Y ≈ J(x)⁻¹.

        For rectangular systems (m ≠ n), use pseudoinverse.
        Returns None if computation fails (singular matrix, SVD non-convergence).
        """
        J = self._eval_jacobian_point(x)

        # Check for NaN/Inf in Jacobian
        if not np.isfinite(J).all():
            return None

        try:
            if self.n_eqs == self.n_vars:
                # Square system: try inverse first, fall back to pseudoinverse
                try:
                    Y = np.linalg.inv(J)
                except np.linalg.LinAlgError:
                    Y = np.linalg.pinv(J, rcond=1e-8)
            else:
                # Rectangular: use pseudoinverse
                Y = np.linalg.pinv(J, rcond=1e-8)

            # Check for NaN/Inf in result
            if not np.isfinite(Y).all():
                return None

            return Y

        except np.linalg.LinAlgError:
            # SVD non-convergence or other linear algebra error
            return None

    def _interval_matrix_mult(
        self,
        A_lo: np.ndarray,
        A_hi: np.ndarray,
        x_lo: np.ndarray,
        x_hi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interval matrix-vector multiplication.

        Computes enclosure of A * x where A ∈ [A_lo, A_hi] and x ∈ [x_lo, x_hi].
        """
        m, n = A_lo.shape
        result_lo = np.zeros(m)
        result_hi = np.zeros(m)

        for i in range(m):
            lo = 0.0
            hi = 0.0
            for j in range(n):
                a_lo, a_hi = A_lo[i, j], A_hi[i, j]
                b_lo, b_hi = x_lo[j], x_hi[j]

                # Interval multiplication: [a_lo, a_hi] * [b_lo, b_hi]
                products = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi]
                prod_lo = min(products)
                prod_hi = max(products)

                lo += prod_lo
                hi += prod_hi

            result_lo[i] = lo
            result_hi[i] = hi

        return result_lo, result_hi

    def contract(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> KrawczykResult:
        """
        Apply Krawczyk contraction for system h(x) = 0.

        The Krawczyk operator:
            K(R) = m - Y·h(m) + (I - Y·J_h(R))·(R - m)

        Args:
            lower: Current lower bounds
            upper: Current upper bounds

        Returns:
            KrawczykResult with contracted bounds or certificates
        """
        lower = lower.copy()
        upper = upper.copy()

        initial_width = np.sum(upper - lower)
        iterations = 0
        unique_root = False

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Midpoint
            m = (lower + upper) / 2

            # Evaluate h(m)
            h_m = self._eval_h(m)
            if np.any(np.isnan(h_m)):
                return KrawczykResult(
                    lower=lower,
                    upper=upper,
                    status=KrawczykStatus.UNCHANGED,
                    tightened=False,
                    empty=False,
                    unique_root=False,
                    iterations=iterations,
                    certificate={"type": "eval_failed"}
                )

            # Check if already at root
            if np.max(np.abs(h_m)) < self.tol:
                return KrawczykResult(
                    lower=lower,
                    upper=upper,
                    status=KrawczykStatus.CONTRACTED,
                    tightened=initial_width - np.sum(upper - lower) > self.tol,
                    empty=False,
                    unique_root=True,
                    iterations=iterations,
                    certificate={
                        "type": "krawczyk_root_found",
                        "residual": float(np.max(np.abs(h_m)))
                    }
                )

            # Compute preconditioner Y ≈ J(m)⁻¹
            Y = self._compute_preconditioner(m)

            if Y is None:
                # Cannot compute preconditioner, return unchanged
                return KrawczykResult(
                    lower=lower,
                    upper=upper,
                    status=KrawczykStatus.UNCHANGED,
                    tightened=False,
                    empty=False,
                    unique_root=False,
                    iterations=iterations,
                    certificate={"type": "preconditioner_failed"}
                )

            # Compute Y·h(m)
            Yh = Y @ h_m

            # Compute interval Jacobian
            J_lo, J_hi = self._eval_jacobian_interval(lower, upper)

            # Compute Y·J_h(R) as interval matrix
            # Y is point matrix, J_h(R) is interval matrix
            YJ_lo = np.zeros((self.n_vars, self.n_vars))
            YJ_hi = np.zeros((self.n_vars, self.n_vars))

            for i in range(self.n_vars):
                for j in range(self.n_vars):
                    val_lo = 0.0
                    val_hi = 0.0
                    for k in range(self.n_eqs):
                        # Y[i,k] * J[k,j]
                        y_val = Y[i, k] if i < Y.shape[0] and k < Y.shape[1] else 0.0
                        j_lo, j_hi = J_lo[k, j], J_hi[k, j]

                        if y_val >= 0:
                            val_lo += y_val * j_lo
                            val_hi += y_val * j_hi
                        else:
                            val_lo += y_val * j_hi
                            val_hi += y_val * j_lo

                    YJ_lo[i, j] = val_lo
                    YJ_hi[i, j] = val_hi

            # Compute I - Y·J_h(R)
            IYJ_lo = -YJ_hi.copy()
            IYJ_hi = -YJ_lo.copy()
            for i in range(min(self.n_vars, IYJ_lo.shape[0])):
                IYJ_lo[i, i] += 1.0
                IYJ_hi[i, i] += 1.0

            # Compute (R - m)
            R_minus_m_lo = lower - m
            R_minus_m_hi = upper - m

            # Compute (I - Y·J_h(R))·(R - m)
            term_lo, term_hi = self._interval_matrix_mult(
                IYJ_lo, IYJ_hi,
                R_minus_m_lo, R_minus_m_hi
            )

            # Krawczyk operator: K(R) = m - Y·h(m) + (I - Y·J_h(R))·(R - m)
            K_lo = m[:self.n_vars] - Yh[:self.n_vars] + term_lo[:self.n_vars]
            K_hi = m[:self.n_vars] - Yh[:self.n_vars] + term_hi[:self.n_vars]

            # Check for empty intersection
            new_lo = np.maximum(lower, K_lo)
            new_hi = np.minimum(upper, K_hi)

            if np.any(new_lo > new_hi + ROUND_EPS):
                # K(R) ∩ R = ∅ suggests no root in R
                # BUT: This is only mathematically valid for SQUARE systems (n_eqs == n_vars)
                # For underdetermined systems (n_eqs < n_vars), Krawczyk may give false negatives
                # because the feasible set is a manifold, not isolated points
                if self.n_eqs == self.n_vars:
                    # Square system: EMPTY certificate is valid
                    return KrawczykResult(
                        lower=lower,
                        upper=upper,
                        status=KrawczykStatus.EMPTY,
                        tightened=True,
                        empty=True,
                        unique_root=False,
                        iterations=iterations,
                        certificate={
                            "type": "krawczyk_empty",
                            "K_interval": [K_lo.tolist(), K_hi.tolist()],
                            "R_interval": [lower.tolist(), upper.tolist()],
                            "empty_variable": int(np.argmax(new_lo - new_hi))
                        }
                    )
                else:
                    # Underdetermined system: K(R) ∩ R = ∅ is NOT a valid EMPTY certificate
                    # The feasible set is a manifold of dimension (n_vars - n_eqs)
                    # Krawczyk is designed for isolated roots, not manifolds
                    # Return unchanged to avoid false UNSAT claims
                    return KrawczykResult(
                        lower=lower,
                        upper=upper,
                        status=KrawczykStatus.UNCHANGED,
                        tightened=False,
                        empty=False,
                        unique_root=False,
                        iterations=iterations,
                        certificate={
                            "type": "krawczyk_underdetermined_skip",
                            "n_eqs": self.n_eqs,
                            "n_vars": self.n_vars,
                            "reason": "Krawczyk EMPTY not valid for m < n systems"
                        }
                    )

            # Check for unique root: K(R) ⊆ R
            # Note: For underdetermined systems (n_eqs < n_vars), solutions form a manifold
            # not isolated points, so "unique root" doesn't apply. Only set for square systems.
            if self.n_eqs == self.n_vars:
                if np.all(K_lo >= lower - ROUND_EPS) and np.all(K_hi <= upper + ROUND_EPS):
                    unique_root = True

            # Apply contraction
            lower = new_lo
            upper = new_hi

            # Check for sufficient progress
            current_width = np.sum(upper - lower)
            if initial_width > 0:
                progress = (initial_width - current_width) / initial_width
                if progress < self.min_progress:
                    break

        # Determine final status
        final_width = np.sum(upper - lower)
        tightened = initial_width - final_width > self.tol

        if unique_root:
            status = KrawczykStatus.UNIQUE
        elif tightened:
            status = KrawczykStatus.CONTRACTED
        else:
            status = KrawczykStatus.UNCHANGED

        return KrawczykResult(
            lower=lower,
            upper=upper,
            status=status,
            tightened=tightened,
            empty=False,
            unique_root=unique_root,
            iterations=iterations,
            certificate={
                "type": "krawczyk_contracted" if tightened else "krawczyk_unchanged",
                "initial_width": initial_width,
                "final_width": final_width,
                "unique_root": unique_root,
                "iterations": iterations
            }
        )


def apply_krawczyk_all_constraints(
    eq_constraints: List[ExpressionGraph],
    n_vars: int,
    lower: np.ndarray,
    upper: np.ndarray,
    max_outer_iterations: int = 5
) -> KrawczykResult:
    """
    Apply Krawczyk contraction for all equality constraints.

    Args:
        eq_constraints: List of expression graphs for h_j(x) = 0
        n_vars: Number of variables
        lower: Current lower bounds
        upper: Current upper bounds
        max_outer_iterations: Maximum outer iterations

    Returns:
        KrawczykResult with contracted bounds or certificate
    """
    if not eq_constraints:
        return KrawczykResult(
            lower=lower.copy(),
            upper=upper.copy(),
            status=KrawczykStatus.UNCHANGED,
            tightened=False,
            empty=False,
            unique_root=False,
            iterations=0,
            certificate={"type": "no_equality_constraints"}
        )

    operator = KrawczykOperator(eq_constraints, n_vars)
    return operator.contract(lower, upper)
