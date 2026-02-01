"""
Feasibility Branch-and-Prune Solver

Implements a branch-and-prune algorithm to find feasible points
or prove infeasibility using FBBT + Interval Newton.

This is the core component for:
1. Finding initial feasible UB (certified witness)
2. Proving UNSAT (infeasibility certificate)

Mathematical Foundation:
- Uses FBBT for constraint propagation (equalities + inequalities)
- Uses Interval Newton for tighter contraction on equalities
- Branch-and-prune with deterministic splitting
- Returns FEASIBLE (with witness) or EMPTY (with certificate)
"""

import numpy as np
import heapq
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from ..bounds.fbbt import (
    FBBTOperator,
    FBBTInequalityOperator,
    FBBTResult,
    apply_fbbt_all_constraints,
)
from ..bounds.interval_newton import (
    IntervalNewtonOperator,
    IntervalNewtonResult,
    apply_interval_newton_all_constraints,
)
from ..bounds.interval import interval_evaluate, Interval
from ..expr_graph import ExpressionGraph
from ..contract import Region


class FeasibilityStatus(Enum):
    """Status of feasibility search."""
    FEASIBLE = "feasible"
    EMPTY = "empty"
    UNKNOWN = "unknown"


@dataclass
class FeasibilityResult:
    """Result of feasibility search."""
    status: FeasibilityStatus
    witness: Optional[np.ndarray] = None
    objective_value: Optional[float] = None
    certificate: Dict[str, Any] = field(default_factory=dict)
    nodes_explored: int = 0
    max_depth: int = 0


@dataclass
class FeasibilityRegion:
    """Region state for feasibility search."""
    lower: np.ndarray
    upper: np.ndarray
    depth: int = 0
    fingerprint: str = ""

    def __lt__(self, other: 'FeasibilityRegion') -> bool:
        if self.depth != other.depth:
            return self.depth > other.depth
        return self.fingerprint < other.fingerprint

    @property
    def width(self) -> float:
        return float(np.max(self.upper - self.lower))

    @property
    def volume(self) -> float:
        widths = self.upper - self.lower
        return float(np.prod(widths[widths > 0]))


class FeasibilityBNP:
    """
    Feasibility Branch-and-Prune Solver.

    Uses FBBT + Interval Newton to find feasible points or prove infeasibility.
    """

    def __init__(
        self,
        n_vars: int,
        bounds: List[Tuple[float, float]],
        eq_constraints: List[ExpressionGraph] = None,
        ineq_constraints: List[ExpressionGraph] = None,
        eq_callables: List = None,
        ineq_callables: List = None,
        objective_graph: ExpressionGraph = None,
        objective_callable = None,
        epsilon: float = 1e-6,
        feas_tol: float = 1e-8,
        max_nodes: int = 10000,
        min_box_width: float = 1e-8
    ):
        self.n_vars = n_vars
        self.bounds = bounds
        self.lb = np.array([b[0] for b in bounds])
        self.ub = np.array([b[1] for b in bounds])

        self.eq_graphs = eq_constraints or []
        self.ineq_graphs = ineq_constraints or []
        self.eq_callables = eq_callables or []
        self.ineq_callables = ineq_callables or []

        self.objective_graph = objective_graph
        self.objective_callable = objective_callable

        self.epsilon = epsilon
        self.feas_tol = feas_tol
        self.max_nodes = max_nodes
        self.min_box_width = min_box_width

        self._eq_fbbt_ops = [FBBTOperator(g, n_vars) for g in self.eq_graphs]
        self._ineq_fbbt_ops = [FBBTInequalityOperator(g, n_vars) for g in self.ineq_graphs]
        self._newton_ops = [IntervalNewtonOperator(g, n_vars) for g in self.eq_graphs]

    def find_feasible(
        self,
        region: Region = None,
        upper_bound: float = float('inf')
    ) -> FeasibilityResult:
        """Find a feasible point or prove infeasibility."""
        if region is None:
            lower = self.lb.copy()
            upper = self.ub.copy()
        else:
            lower = region.lower.copy()
            upper = region.upper.copy()

        heap: List[FeasibilityRegion] = []
        root = FeasibilityRegion(
            lower=lower,
            upper=upper,
            depth=0,
            fingerprint=self._fingerprint(lower, upper)
        )
        heapq.heappush(heap, root)

        nodes_explored = 0
        max_depth = 0
        empty_certificates = []

        while heap and nodes_explored < self.max_nodes:
            nodes_explored += 1
            current = heapq.heappop(heap)
            max_depth = max(max_depth, current.depth)

            # Apply FBBT
            fbbt_result = self._apply_full_fbbt(current.lower, current.upper)

            if fbbt_result.empty:
                empty_certificates.append({
                    "region": [current.lower.tolist(), current.upper.tolist()],
                    "certificate": fbbt_result.certificate
                })
                continue

            lower = fbbt_result.lower
            upper = fbbt_result.upper

            # Apply Interval Newton
            if self.eq_graphs:
                newton_result = self._apply_interval_newton(lower, upper)

                if newton_result.empty:
                    empty_certificates.append({
                        "region": [lower.tolist(), upper.tolist()],
                        "certificate": newton_result.certificate
                    })
                    continue

                lower = newton_result.lower
                upper = newton_result.upper

            # Apply UB constraint
            if upper_bound < float('inf') and self.objective_graph:
                ub_result = self._apply_ub_constraint(lower, upper, upper_bound)
                if ub_result is None:
                    empty_certificates.append({
                        "region": [lower.tolist(), upper.tolist()],
                        "certificate": {"type": "ub_exceeded"}
                    })
                    continue
                lower, upper = ub_result

            # Check box size
            box_width = np.max(upper - lower)
            if box_width <= self.min_box_width:
                center = (lower + upper) / 2
                if self._is_feasible(center):
                    obj_val = self._eval_objective(center)
                    return FeasibilityResult(
                        status=FeasibilityStatus.FEASIBLE,
                        witness=center,
                        objective_value=obj_val,
                        certificate={"type": "feasible_witness", "box_width": box_width},
                        nodes_explored=nodes_explored,
                        max_depth=max_depth
                    )

            # Try to find feasible point
            candidate = self._try_find_feasible(lower, upper)
            if candidate is not None:
                obj_val = self._eval_objective(candidate)
                if obj_val <= upper_bound + self.feas_tol:
                    return FeasibilityResult(
                        status=FeasibilityStatus.FEASIBLE,
                        witness=candidate,
                        objective_value=obj_val,
                        certificate={"type": "feasible_witness", "method": "local_search"},
                        nodes_explored=nodes_explored,
                        max_depth=max_depth
                    )

            # Split region
            children = self._split_region(lower, upper, current.depth)
            for child_lower, child_upper in children:
                child = FeasibilityRegion(
                    lower=child_lower,
                    upper=child_upper,
                    depth=current.depth + 1,
                    fingerprint=self._fingerprint(child_lower, child_upper)
                )
                heapq.heappush(heap, child)

        if not heap and len(empty_certificates) > 0:
            return FeasibilityResult(
                status=FeasibilityStatus.EMPTY,
                witness=None,
                certificate={
                    "type": "cover_refutation",
                    "num_regions_refuted": len(empty_certificates),
                    "certificates": empty_certificates[:10]
                },
                nodes_explored=nodes_explored,
                max_depth=max_depth
            )

        return FeasibilityResult(
            status=FeasibilityStatus.UNKNOWN,
            witness=None,
            certificate={
                "type": "budget_exhausted",
                "nodes_explored": nodes_explored,
                "remaining_regions": len(heap)
            },
            nodes_explored=nodes_explored,
            max_depth=max_depth
        )

    def _apply_full_fbbt(self, lower: np.ndarray, upper: np.ndarray) -> FBBTResult:
        return apply_fbbt_all_constraints(
            eq_constraints=self.eq_graphs,
            ineq_constraints=self.ineq_graphs,
            n_vars=self.n_vars,
            lower=lower,
            upper=upper,
            max_iterations=10
        )

    def _apply_interval_newton(self, lower: np.ndarray, upper: np.ndarray) -> IntervalNewtonResult:
        return apply_interval_newton_all_constraints(
            eq_constraints=self.eq_graphs,
            n_vars=self.n_vars,
            lower=lower,
            upper=upper,
            max_outer_iterations=5
        )

    def _apply_ub_constraint(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        upper_bound: float
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.objective_graph is None:
            return lower, upper

        try:
            obj_interval = interval_evaluate(self.objective_graph, lower, upper)
            if obj_interval.lo > upper_bound + self.feas_tol:
                return None
        except:
            pass

        return lower, upper

    def _is_feasible(self, x: np.ndarray) -> bool:
        if np.any(x < self.lb - self.feas_tol) or np.any(x > self.ub + self.feas_tol):
            return False

        for g in self.eq_graphs:
            try:
                val = g.evaluate(x)
                if abs(val) > self.feas_tol:
                    return False
            except:
                return False

        for h in self.eq_callables:
            try:
                val = h(x)
                if abs(val) > self.feas_tol:
                    return False
            except:
                return False

        for g in self.ineq_graphs:
            try:
                val = g.evaluate(x)
                if val > self.feas_tol:
                    return False
            except:
                return False

        for g in self.ineq_callables:
            try:
                val = g(x)
                if val > self.feas_tol:
                    return False
            except:
                return False

        return True

    def _eval_objective(self, x: np.ndarray) -> float:
        if self.objective_graph is not None:
            return self.objective_graph.evaluate(x)
        elif self.objective_callable is not None:
            return self.objective_callable(x)
        return 0.0

    def _try_find_feasible(self, lower: np.ndarray, upper: np.ndarray) -> Optional[np.ndarray]:
        center = (lower + upper) / 2
        if self._is_feasible(center):
            return center

        for i in range(min(16, 2**self.n_vars)):
            corner = np.array([
                lower[j] if (i >> j) & 1 == 0 else upper[j]
                for j in range(self.n_vars)
            ])
            if self._is_feasible(corner):
                return corner

        if self.eq_callables or self.ineq_callables or self.eq_graphs or self.ineq_graphs:
            try:
                from scipy.optimize import minimize

                def violation(x):
                    total = 0.0
                    for g in self.eq_graphs:
                        try:
                            total += g.evaluate(x)**2
                        except:
                            total += 1e10
                    for h in self.eq_callables:
                        try:
                            total += h(x)**2
                        except:
                            total += 1e10
                    for g in self.ineq_graphs:
                        try:
                            val = g.evaluate(x)
                            total += max(0, val)**2
                        except:
                            total += 1e10
                    for g in self.ineq_callables:
                        try:
                            val = g(x)
                            total += max(0, val)**2
                        except:
                            total += 1e10
                    return total

                result = minimize(
                    violation,
                    center,
                    method='L-BFGS-B',
                    bounds=list(zip(lower, upper)),
                    options={'maxiter': 50}
                )

                if result.success and result.fun < self.feas_tol:
                    if self._is_feasible(result.x):
                        return result.x.copy()
            except:
                pass

        return None

    def _split_region(self, lower: np.ndarray, upper: np.ndarray, depth: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        widths = upper - lower
        split_dim = int(np.argmax(widths))
        split_point = (lower[split_dim] + upper[split_dim]) / 2

        child1_lower = lower.copy()
        child1_upper = upper.copy()
        child1_upper[split_dim] = split_point

        child2_lower = lower.copy()
        child2_lower[split_dim] = split_point
        child2_upper = upper.copy()

        return [(child1_lower, child1_upper), (child2_lower, child2_upper)]

    def _fingerprint(self, lower: np.ndarray, upper: np.ndarray) -> str:
        data = f"{lower.tolist()}:{upper.tolist()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


def find_initial_feasible_ub(
    problem,
    max_nodes: int = 5000,
    min_box_width: float = 1e-6
) -> FeasibilityResult:
    """Find initial feasible upper bound using FeasibilityBNP."""
    solver = FeasibilityBNP(
        n_vars=problem.n_vars,
        bounds=problem.bounds,
        eq_constraints=problem._eq_graphs,
        ineq_constraints=problem._ineq_graphs,
        eq_callables=problem.eq_constraints,
        ineq_callables=problem.ineq_constraints,
        objective_graph=problem._obj_graph,
        objective_callable=problem.objective if not isinstance(problem.objective, ExpressionGraph) else None,
        epsilon=problem.epsilon,
        feas_tol=problem.feas_tol,
        max_nodes=max_nodes,
        min_box_width=min_box_width
    )

    return solver.find_feasible()
