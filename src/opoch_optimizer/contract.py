"""
Problem Contract Definition

Defines the optimization problem structure:
- Objective function
- Inequality constraints g_i(x) <= 0
- Equality constraints h_j(x) = 0
- Variable bounds (box domain)
- Precision target epsilon

Also defines Region for branch-and-bound partitioning.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import hashlib
import json

from .expr_graph import ExpressionGraph


def canonical_dumps(obj: Any, indent: int = None) -> str:
    """Canonical JSON serialization with sorted keys."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':') if indent is None else None, indent=indent)


def canonical_hash(obj: Any) -> str:
    """Compute SHA-256 hash of canonical JSON."""
    return hashlib.sha256(canonical_dumps(obj).encode()).hexdigest()


@dataclass
class Bounds:
    """
    Variable bounds defining a box domain.

    Attributes:
        lower: Lower bounds for each variable
        upper: Upper bounds for each variable
    """
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)

        if len(self.lower) != len(self.upper):
            raise ValueError("Lower and upper bounds must have same length")

        if np.any(self.lower > self.upper):
            raise ValueError("Lower bounds must be <= upper bounds")

    @property
    def n_vars(self) -> int:
        return len(self.lower)

    @property
    def widths(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def center(self) -> np.ndarray:
        return (self.lower + self.upper) / 2.0

    @property
    def volume(self) -> float:
        return float(np.prod(self.widths))

    def contains(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a point is within bounds (with tolerance)."""
        return bool(
            np.all(x >= self.lower - tol) and
            np.all(x <= self.upper + tol)
        )

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist()
        }

    @classmethod
    def from_list(cls, bounds: List[Tuple[float, float]]) -> 'Bounds':
        """Create from list of (lower, upper) tuples."""
        lower = [b[0] for b in bounds]
        upper = [b[1] for b in bounds]
        return cls(np.array(lower), np.array(upper))


@dataclass
class Region:
    """
    A region (box) for branch-and-bound.

    Attributes:
        lower: Lower bounds of the region
        upper: Upper bounds of the region
        region_id: Unique identifier for this region
        lb: Certified lower bound on objective in this region
        status: "maybe" or "empty"
        parent_id: ID of parent region (None for root)
        depth: Depth in the B&B tree
    """
    lower: np.ndarray
    upper: np.ndarray
    region_id: int = 0
    lb: float = float('-inf')
    status: str = "maybe"  # "maybe" or "empty"
    parent_id: Optional[int] = None
    depth: int = 0

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)

    @property
    def n_vars(self) -> int:
        return len(self.lower)

    @property
    def widths(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def center(self) -> np.ndarray:
        return (self.lower + self.upper) / 2.0

    @property
    def volume(self) -> float:
        return float(np.prod(self.widths))

    @property
    def max_width(self) -> float:
        return float(np.max(self.widths))

    def longest_dimension(self) -> int:
        """
        Return the index of the longest dimension.
        Ties broken by smallest index.
        """
        widths = self.widths
        max_w = np.max(widths)
        candidates = np.where(np.abs(widths - max_w) < 1e-12)[0]
        return int(candidates[0])

    def split(self, dimension: int = None, point: float = None) -> Tuple['Region', 'Region']:
        """
        Split the region into two children.

        Args:
            dimension: Dimension to split (default: longest)
            point: Split point (default: midpoint)

        Returns:
            Tuple of two child regions
        """
        if dimension is None:
            dimension = self.longest_dimension()

        if point is None:
            point = (self.lower[dimension] + self.upper[dimension]) / 2.0

        # Create child lower bounds
        lower1 = self.lower.copy()
        upper1 = self.upper.copy()
        upper1[dimension] = point

        lower2 = self.lower.copy()
        lower2[dimension] = point
        upper2 = self.upper.copy()

        # Child IDs will be assigned by the solver
        child1 = Region(
            lower=lower1,
            upper=upper1,
            region_id=-1,  # To be assigned
            lb=float('-inf'),
            status="maybe",
            parent_id=self.region_id,
            depth=self.depth + 1
        )

        child2 = Region(
            lower=lower2,
            upper=upper2,
            region_id=-1,
            lb=float('-inf'),
            status="maybe",
            parent_id=self.region_id,
            depth=self.depth + 1
        )

        return child1, child2

    def contains(self, x: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if a point is within the region."""
        return bool(
            np.all(x >= self.lower - tol) and
            np.all(x <= self.upper + tol)
        )

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "lb": self.lb,
            "status": self.status,
            "parent_id": self.parent_id,
            "depth": self.depth
        }

    def fingerprint(self) -> str:
        """Canonical hash for tie-breaking."""
        return canonical_hash({
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist()
        })


@dataclass
class ProblemContract:
    """
    Complete optimization problem specification.

    Represents:
        min f(x)
        s.t. g_i(x) <= 0, i = 1..m
             h_j(x) = 0,  j = 1..p
             x in [lower, upper]

    Attributes:
        objective: The objective function (ExpressionGraph or callable)
        ineq_constraints: List of inequality constraints g_i(x) <= 0
        eq_constraints: List of equality constraints h_j(x) = 0
        bounds: Variable bounds (list of (lo, hi) tuples)
        epsilon: Optimality tolerance
        feas_tol: Feasibility tolerance
        name: Optional problem name
    """
    objective: Union[ExpressionGraph, Callable]
    bounds: List[Tuple[float, float]]
    ineq_constraints: List[Union[ExpressionGraph, Callable]] = field(default_factory=list)
    eq_constraints: List[Union[ExpressionGraph, Callable]] = field(default_factory=list)
    epsilon: float = 1e-6
    feas_tol: float = 1e-8
    name: str = "unnamed"

    # Expression graphs (populated if callables are provided)
    _obj_graph: Optional[ExpressionGraph] = field(default=None, repr=False)
    _ineq_graphs: List[ExpressionGraph] = field(default_factory=list, repr=False)
    _eq_graphs: List[ExpressionGraph] = field(default_factory=list, repr=False)

    def __post_init__(self):
        # Validate bounds
        self._bounds = Bounds.from_list(self.bounds)

        # Convert to expression graphs if needed
        if isinstance(self.objective, ExpressionGraph):
            self._obj_graph = self.objective

    @property
    def n_vars(self) -> int:
        return len(self.bounds)

    @property
    def n_ineq(self) -> int:
        return len(self.ineq_constraints)

    @property
    def n_eq(self) -> int:
        return len(self.eq_constraints)

    @property
    def domain_bounds(self) -> Bounds:
        return self._bounds

    def eval_objective(self, x: np.ndarray) -> float:
        """Evaluate the objective function."""
        if self._obj_graph is not None:
            return self._obj_graph.evaluate(x)
        else:
            return float(self.objective(x))

    def eval_ineq_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all inequality constraints."""
        if not self.ineq_constraints:
            return np.array([])

        values = []
        for i, g in enumerate(self.ineq_constraints):
            if i < len(self._ineq_graphs) and self._ineq_graphs[i] is not None:
                values.append(self._ineq_graphs[i].evaluate(x))
            else:
                values.append(float(g(x)))
        return np.array(values)

    def eval_eq_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all equality constraints."""
        if not self.eq_constraints:
            return np.array([])

        values = []
        for j, h in enumerate(self.eq_constraints):
            if j < len(self._eq_graphs) and self._eq_graphs[j] is not None:
                values.append(self._eq_graphs[j].evaluate(x))
            else:
                values.append(float(h(x)))
        return np.array(values)

    def is_feasible(self, x: np.ndarray) -> bool:
        """Check if a point is feasible."""
        # Check bounds
        if not self._bounds.contains(x, self.feas_tol):
            return False

        # Check inequality constraints
        g_vals = self.eval_ineq_constraints(x)
        if len(g_vals) > 0 and np.any(g_vals > self.feas_tol):
            return False

        # Check equality constraints
        h_vals = self.eval_eq_constraints(x)
        if len(h_vals) > 0 and np.any(np.abs(h_vals) > self.feas_tol):
            return False

        return True

    def get_violations(self, x: np.ndarray) -> Dict[str, Any]:
        """Get detailed constraint violation information."""
        violations = {
            "bound_lower": np.maximum(0, self._bounds.lower - x),
            "bound_upper": np.maximum(0, x - self._bounds.upper),
            "ineq": np.maximum(0, self.eval_ineq_constraints(x)),
            "eq": np.abs(self.eval_eq_constraints(x))
        }
        violations["max_violation"] = max(
            float(np.max(violations["bound_lower"])) if len(violations["bound_lower"]) > 0 else 0,
            float(np.max(violations["bound_upper"])) if len(violations["bound_upper"]) > 0 else 0,
            float(np.max(violations["ineq"])) if len(violations["ineq"]) > 0 else 0,
            float(np.max(violations["eq"])) if len(violations["eq"]) > 0 else 0
        )
        violations["is_feasible"] = violations["max_violation"] <= self.feas_tol
        return violations

    def initial_region(self) -> Region:
        """Create the initial region covering the entire domain."""
        return Region(
            lower=self._bounds.lower.copy(),
            upper=self._bounds.upper.copy(),
            region_id=0,
            lb=float('-inf'),
            status="maybe",
            parent_id=None,
            depth=0
        )

    def to_canonical(self) -> Dict[str, Any]:
        """Convert problem to canonical form for hashing/serialization."""
        return {
            "name": self.name,
            "n_vars": self.n_vars,
            "n_ineq": self.n_ineq,
            "n_eq": self.n_eq,
            "bounds": self.bounds,
            "epsilon": self.epsilon,
            "feas_tol": self.feas_tol,
            "objective_graph": self._obj_graph.to_canonical() if self._obj_graph else None
        }

    @classmethod
    def create(
        cls,
        objective: Callable,
        bounds: List[Tuple[float, float]],
        ineq_constraints: List[Callable] = None,
        eq_constraints: List[Callable] = None,
        epsilon: float = 1e-6,
        name: str = "unnamed"
    ) -> 'ProblemContract':
        """
        Create a problem contract from callables.

        Args:
            objective: Objective function f(x) -> float
            bounds: List of (lower, upper) tuples for each variable
            ineq_constraints: List of g_i(x) <= 0 constraint functions
            eq_constraints: List of h_j(x) = 0 constraint functions
            epsilon: Optimality tolerance
            name: Problem name

        Returns:
            ProblemContract instance
        """
        return cls(
            objective=objective,
            bounds=bounds,
            ineq_constraints=ineq_constraints or [],
            eq_constraints=eq_constraints or [],
            epsilon=epsilon,
            name=name
        )
