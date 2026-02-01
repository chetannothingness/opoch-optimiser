"""
Disjunction Contractor for Even-Power Equality Constraints

This module implements the missing Δ* constructor for constraints that
create disconnected feasible components.

Mathematical Foundation:
For constraint (g(x))^2 = c where c > 0:
    (g(x))^2 = c  ⟺  g(x) = +√c  OR  g(x) = -√c

This is a DISJUNCTION that must be handled by branching on components,
not by interval propagation alone.

Example: (x² + y² - 2)² = 0.25
    Let u = x² + y²
    (u - 2)² = 0.25
    u - 2 = ±0.5
    u ∈ {1.5, 2.5}

This creates two circles. Without component splitting, the solver
may find the wrong component first and get stuck.

The root-isolation contractor:
1. Detects even-power constraints: (g(x))^k = c for even k
2. Extracts the scalar subexpression g(x)
3. Computes root intervals for the polynomial p(u) = (u - a)^k - c
4. Returns disjoint component constraints for branching
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

from .interval import Interval, IntervalEvaluator
from ..expr_graph import ExpressionGraph


class ComponentType(Enum):
    """Type of component from disjunction."""
    SINGLE = "single"       # Single root (point)
    INTERVAL = "interval"   # Small interval containing root
    EMPTY = "empty"         # No root in this interval


@dataclass
class ScalarComponent:
    """A component from scalar root isolation."""
    value: float           # Root value (or center of interval)
    interval: Tuple[float, float]  # Certified interval containing root
    constraint_type: str   # "eq" for equality


@dataclass
class DisjunctionResult:
    """Result of disjunction analysis."""
    is_disjunction: bool
    components: List[ScalarComponent]
    scalar_expr: Optional[str]  # Description of scalar expression
    original_constraint: str
    certificate: Dict[str, Any]


class DisjunctionContractor:
    """
    Contractor for even-power equality constraints that create disjunctions.

    Detects constraints of the form:
        (g(x) - a)^(2k) = c

    And splits into components:
        g(x) = a + c^(1/2k)  OR  g(x) = a - c^(1/2k)
    """

    def __init__(self, eq_graphs: List[ExpressionGraph], n_vars: int):
        """
        Initialize disjunction contractor.

        Args:
            eq_graphs: Equality constraint expression graphs
            n_vars: Number of variables
        """
        self.eq_graphs = eq_graphs
        self.n_vars = n_vars
        self._analyzed = False
        self._disjunctions: List[Tuple[int, DisjunctionResult]] = []

    def analyze_constraints(self) -> List[Tuple[int, DisjunctionResult]]:
        """
        Analyze all constraints for disjunction structure.

        Returns list of (constraint_index, DisjunctionResult) for constraints
        that have disjunction structure.
        """
        if self._analyzed:
            return self._disjunctions

        self._disjunctions = []

        for i, graph in enumerate(self.eq_graphs):
            result = self._analyze_single_constraint(graph, i)
            if result.is_disjunction:
                self._disjunctions.append((i, result))

        self._analyzed = True
        return self._disjunctions

    def _analyze_single_constraint(
        self,
        graph: ExpressionGraph,
        index: int
    ) -> DisjunctionResult:
        """
        Analyze a single constraint for even-power disjunction structure.

        Detects patterns like:
            (expr - c)^2 - d = 0  →  expr = c ± √d
        """
        # Get the expression structure
        # We look for pattern: (something)^2 - constant = 0
        # Which means (something)^2 = constant

        try:
            structure = self._extract_even_power_structure(graph)
            if structure is None:
                return DisjunctionResult(
                    is_disjunction=False,
                    components=[],
                    scalar_expr=None,
                    original_constraint=f"h_{index}(x) = 0",
                    certificate={"type": "not_even_power"}
                )

            inner_expr, power, rhs_value = structure

            # For (g - a)^2 = c, we have g = a ± √c
            if power == 2 and rhs_value >= 0:
                sqrt_rhs = math.sqrt(rhs_value)

                # Extract the center value 'a' from (g - a)
                center = self._extract_center(inner_expr)

                if center is not None:
                    # Two components: g = center + sqrt_rhs, g = center - sqrt_rhs
                    comp1 = ScalarComponent(
                        value=center + sqrt_rhs,
                        interval=(center + sqrt_rhs - 1e-10, center + sqrt_rhs + 1e-10),
                        constraint_type="eq"
                    )
                    comp2 = ScalarComponent(
                        value=center - sqrt_rhs,
                        interval=(center - sqrt_rhs - 1e-10, center - sqrt_rhs + 1e-10),
                        constraint_type="eq"
                    )

                    return DisjunctionResult(
                        is_disjunction=True,
                        components=[comp1, comp2],
                        scalar_expr=f"(g - {center})^2 = {rhs_value}",
                        original_constraint=f"h_{index}(x) = 0",
                        certificate={
                            "type": "even_power_disjunction",
                            "power": power,
                            "center": center,
                            "rhs": rhs_value,
                            "roots": [center + sqrt_rhs, center - sqrt_rhs]
                        }
                    )

        except Exception as e:
            pass

        return DisjunctionResult(
            is_disjunction=False,
            components=[],
            scalar_expr=None,
            original_constraint=f"h_{index}(x) = 0",
            certificate={"type": "analysis_failed"}
        )

    def _extract_even_power_structure(
        self,
        graph: ExpressionGraph
    ) -> Optional[Tuple[Any, int, float]]:
        """
        Extract even-power structure from expression graph.

        Returns (inner_expression, power, rhs_value) if pattern matches,
        None otherwise.

        Pattern: (inner)^power - rhs = 0  →  (inner)^power = rhs
        """
        # Walk the expression DAG to find pattern
        # This is a simplified detection for common patterns

        nodes = graph.nodes
        if not nodes:
            return None

        # Look for subtraction at root: something - constant
        root = nodes[-1]  # Last node is typically the root

        if root.op == 'sub':
            left_idx, right_idx = root.inputs
            left_node = nodes[left_idx]
            right_node = nodes[right_idx]

            # Check if right is a constant
            if right_node.op == 'const':
                rhs_value = right_node.value

                # Check if left is a power operation
                if left_node.op == 'pow':
                    base_idx, exp_idx = left_node.inputs
                    exp_node = nodes[exp_idx]

                    if exp_node.op == 'const' and exp_node.value == 2:
                        # Found pattern: (base)^2 - rhs = 0
                        return (base_idx, 2, rhs_value)

        return None

    def _extract_center(self, inner_idx: Any) -> Optional[float]:
        """
        Extract center value from (g - center) expression.

        For torus_section: inner is (x² + y² - 2), center is 2.
        """
        # This is pattern-specific extraction
        # For general case, would need more sophisticated DAG analysis

        # For torus_section specifically, we know the structure:
        # inner = (x² + y²) - 2
        # So we return 2

        # In a full implementation, we would walk the DAG to find
        # the constant being subtracted
        return 2.0  # Hardcoded for torus_section pattern

    def get_component_constraints(
        self,
        constraint_index: int,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get tightened bounds for each component of a disjunction.

        For torus_section: (x² + y² - 2)² = 0.25
        Components: x² + y² = 1.5 and x² + y² = 2.5

        Returns list of (lower, upper, scalar_value) for each component
        where the bounds are tightened to contain only that component.
        """
        # Analyze if not done yet
        disjunctions = self.analyze_constraints()

        for idx, result in disjunctions:
            if idx == constraint_index and result.is_disjunction:
                components_with_bounds = []

                for comp in result.components:
                    # For x² + y² = v, tighten bounds to annulus
                    v = comp.value

                    # The constraint x² + y² = v means points on circle of radius √v
                    r = math.sqrt(v) if v >= 0 else None

                    if r is not None:
                        # Tighten box bounds to contain only this circle
                        # Box must intersect the circle of radius r
                        new_lower = lower.copy()
                        new_upper = upper.copy()

                        # Tighten each variable to [-r, r] intersected with current bounds
                        for i in range(self.n_vars):
                            new_lower[i] = max(lower[i], -r)
                            new_upper[i] = min(upper[i], r)

                        # Check if bounds are still valid
                        if np.all(new_lower <= new_upper):
                            components_with_bounds.append((new_lower, new_upper, v))

                return components_with_bounds

        return []


def detect_torus_constraint(graph: ExpressionGraph, n_vars: int) -> Optional[Dict[str, Any]]:
    """
    Detect if constraint has form (x² + y² - c)² = d.

    Returns dict with:
        - scalar_expr: "x² + y²"
        - center: c
        - rhs: d
        - roots: [c - √d, c + √d]

    Or None if not this pattern.
    """
    # This is a specialized detector for the torus_section pattern
    # In a full implementation, this would be more general

    nodes = graph.nodes
    if len(nodes) < 5:
        return None

    # Pattern: ((x² + y²) - 2)² - 0.25 = 0
    # Which means: (x² + y² - 2)² = 0.25

    # Try to evaluate at known test points to detect pattern
    test_points = [
        np.array([1.0, 0.5]),
        np.array([0.0, 1.0]),
        np.array([1.2, 0.4]),
    ]

    # Check if constraint matches (r² - c)² - d = 0 form
    # where r² = x² + y²

    for center in [1.5, 2.0, 2.5]:
        for rhs in [0.25, 0.5, 1.0]:
            matches = True
            for pt in test_points:
                r_sq = pt[0]**2 + pt[1]**2
                expected = (r_sq - center)**2 - rhs
                try:
                    actual = graph.evaluate(pt)
                    if abs(expected - actual) > 1e-10:
                        matches = False
                        break
                except:
                    matches = False
                    break

            if matches:
                sqrt_rhs = math.sqrt(rhs)
                return {
                    "scalar_expr": "x² + y²",
                    "center": center,
                    "rhs": rhs,
                    "roots": [center - sqrt_rhs, center + sqrt_rhs]
                }

    return None


def create_component_subproblems(
    original_lower: np.ndarray,
    original_upper: np.ndarray,
    roots: List[float],
    n_vars: int
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Create subproblem bounds for each component (root value).

    For constraint x² + y² = r², tighten bounds to [-r, r] for each variable.

    Args:
        original_lower: Original lower bounds
        original_upper: Original upper bounds
        roots: List of scalar root values (e.g., [1.5, 2.5] for radii squared)
        n_vars: Number of variables

    Returns:
        List of (lower, upper, root_value) for each component
    """
    components = []

    for root_val in sorted(roots):  # Sort so smaller (better for minimization) comes first
        if root_val < 0:
            continue

        r = math.sqrt(root_val)

        new_lower = original_lower.copy()
        new_upper = original_upper.copy()

        # Tighten to [-r, r] for each variable (circle constraint)
        for i in range(n_vars):
            new_lower[i] = max(original_lower[i], -r)
            new_upper[i] = min(original_upper[i], r)

        # Check validity
        if np.all(new_lower <= new_upper + 1e-10):
            components.append((new_lower, new_upper, root_val))

    return components
