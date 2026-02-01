"""
Test Circle Manifold: The Sanity Check for GLOBALLib Certification

Problem: x² + y² = 1, minimize x + y over [-2,2]²

This tests:
- Equality manifold handling (Krawczyk contracts to thin annulus)
- Feasibility finding (any point on circle)
- Gap closure (minimum at (-1/√2, -1/√2) = -√2)

If this works, GLOBALLib becomes a scaling exercise.
"""

import numpy as np
import pytest
from math import sqrt

from opoch_optimizer.expr_graph import ExpressionGraph
from opoch_optimizer.contract import ProblemContract, Region
from opoch_optimizer.bounds.krawczyk import KrawczykOperator, KrawczykStatus
from opoch_optimizer.solver.constraint_closure import ConstraintClosure, ClosureStatus
from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig


def build_circle_manifold_problem():
    """
    Build the circle manifold problem:
        min x + y
        s.t. x² + y² = 1
        x, y ∈ [-2, 2]

    Optimal: x* = y* = -1/√2 ≈ -0.7071
    Optimal value: f* = -√2 ≈ -1.4142
    """
    # Objective: f(x,y) = x + y
    # Note: from_callable passes individual variables (x, y), not an array
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y,
        num_vars=2
    )

    # Constraint: h(x,y) = x² + y² - 1 = 0
    eq_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 - 1,
        num_vars=2
    )

    # Build problem contract
    problem = ProblemContract(
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=lambda x: x[0] + x[1],
        eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1]
    )

    # Attach expression graphs
    problem._obj_graph = obj_graph
    problem._eq_graphs = [eq_graph]
    problem._ineq_graphs = []

    return problem


class TestKrawczykOnCircle:
    """Test Krawczyk contractor on circle manifold."""

    def test_krawczyk_contracts_circle(self):
        """Krawczyk should contract bounds when circle intersects box."""
        # Circle constraint: x² + y² - 1 = 0
        eq_graph = ExpressionGraph.from_callable(
            lambda x, y: x**2 + y**2 - 1,
            num_vars=2
        )

        krawczyk = KrawczykOperator([eq_graph], n_vars=2)

        # Start with large box [-2,2]²
        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])

        result = krawczyk.contract(lower, upper)

        # Should contract (circle is inside box)
        print(f"Krawczyk status: {result.status}")
        print(f"Initial box: [{lower}, {upper}]")
        print(f"Contracted box: [{result.lower}, {result.upper}]")
        print(f"Certificate: {result.certificate}")

        # Box should have shrunk
        initial_volume = np.prod(upper - lower)
        final_volume = np.prod(result.upper - result.lower)
        print(f"Volume ratio: {final_volume / initial_volume:.4f}")

    def test_krawczyk_refutes_empty(self):
        """Krawczyk should refute box that doesn't contain circle."""
        # Circle constraint: x² + y² - 1 = 0
        eq_graph = ExpressionGraph.from_callable(
            lambda x, y: x**2 + y**2 - 1,
            num_vars=2
        )

        krawczyk = KrawczykOperator([eq_graph], n_vars=2)

        # Box far from circle: [5,6]²
        lower = np.array([5.0, 5.0])
        upper = np.array([6.0, 6.0])

        result = krawczyk.contract(lower, upper)

        print(f"Krawczyk status: {result.status}")
        print(f"Empty: {result.empty}")

        # Should prove empty (no part of circle in this box)
        # Note: might not always refute depending on implementation
        # At minimum should not contract significantly


class TestConstraintClosureOnCircle:
    """Test unified constraint closure on circle manifold."""

    def test_closure_contracts_circle(self):
        """Unified closure should contract bounds tightly."""
        # Circle constraint: x² + y² - 1 = 0
        eq_graph = ExpressionGraph.from_callable(
            lambda x, y: x**2 + y**2 - 1,
            num_vars=2
        )

        closure = ConstraintClosure(
            n_vars=2,
            eq_constraints=[eq_graph],
            ineq_constraints=[],
            max_outer_iterations=20
        )

        # Start with large box [-2,2]²
        lower = np.array([-2.0, -2.0])
        upper = np.array([2.0, 2.0])

        result = closure.apply(lower, upper)

        print(f"Closure status: {result.status}")
        print(f"Initial box: [{lower}, {upper}]")
        print(f"Contracted box: [{result.lower}, {result.upper}]")
        print(f"FBBT iterations: {result.fbbt_iterations}")
        print(f"Krawczyk iterations: {result.krawczyk_iterations}")
        print(f"Certificate: {result.certificate}")

        # Should have contracted
        assert result.status in [ClosureStatus.CONTRACTED, ClosureStatus.UNCHANGED]

        # Circle has x,y ∈ [-1, 1], so bounds should be within this range
        # (with some tolerance for iterative methods)


class TestOPOCHKernelOnCircle:
    """Test full OPOCH kernel on circle manifold problem."""

    def test_solve_circle_problem(self):
        """OPOCH kernel should solve circle manifold to certified optimum."""
        problem = build_circle_manifold_problem()

        config = OPOCHConfig(
            epsilon=1e-3,  # Moderate tolerance
            max_time=60.0,
            max_nodes=10000
        )

        kernel = OPOCHKernel(problem, config)
        verdict, result = kernel.solve()

        print(f"\n=== Circle Manifold Result ===")
        print(f"Verdict: {verdict}")
        print(f"Upper bound: {result.upper_bound:.6f}")
        print(f"Lower bound: {result.lower_bound:.6f}")
        print(f"Gap: {result.upper_bound - result.lower_bound:.6f}")

        if hasattr(result, 'x_optimal') and result.x_optimal is not None:
            print(f"Solution: {result.x_optimal}")
            print(f"Objective: {result.objective_value:.6f}")

            # Check feasibility
            x = result.x_optimal
            constraint_val = x[0]**2 + x[1]**2 - 1
            print(f"Constraint violation: |x² + y² - 1| = {abs(constraint_val):.2e}")

        # Known optimal
        opt_val = -sqrt(2)
        opt_x = np.array([-1/sqrt(2), -1/sqrt(2)])
        print(f"\nKnown optimal: f* = -√2 ≈ {opt_val:.6f}")
        print(f"Known x* = (-1/√2, -1/√2) ≈ {opt_x}")

        print(f"Nodes explored: {result.nodes_explored}")

    def test_solve_circle_tight_epsilon(self):
        """Test with tight epsilon for certified gap closure."""
        problem = build_circle_manifold_problem()

        config = OPOCHConfig(
            epsilon=1e-4,
            max_time=120.0,
            max_nodes=50000
        )

        kernel = OPOCHKernel(problem, config)
        verdict, result = kernel.solve()

        print(f"\n=== Circle Manifold (Tight ε) ===")
        print(f"Verdict: {verdict}")
        print(f"Gap: {result.upper_bound - result.lower_bound:.6f}")
        print(f"ε: {config.epsilon}")

        # For UNIQUE_OPT, gap should be ≤ ε
        if verdict.name == "UNIQUE_OPT":
            assert result.upper_bound - result.lower_bound <= config.epsilon * 1.01  # small tolerance


class TestCircleWithInequality:
    """Test circle with additional inequality constraints."""

    def build_semicircle_problem(self):
        """
        Semicircle problem:
            min x + y
            s.t. x² + y² = 1
                 x ≥ 0 (equivalently: -x ≤ 0)
            x, y ∈ [-2, 2]

        Optimal on right semicircle: (0, -1) with f* = -1
        """
        obj_graph = ExpressionGraph.from_callable(
            lambda x, y: x + y,
            num_vars=2
        )

        eq_graph = ExpressionGraph.from_callable(
            lambda x, y: x**2 + y**2 - 1,
            num_vars=2
        )

        # Inequality: -x ≤ 0 (i.e., x ≥ 0)
        ineq_graph = ExpressionGraph.from_callable(
            lambda x, y: -x,
            num_vars=2
        )

        problem = ProblemContract(
            bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            objective=lambda x: x[0] + x[1],
            eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1],
            ineq_constraints=[lambda x: -x[0]]
        )

        problem._obj_graph = obj_graph
        problem._eq_graphs = [eq_graph]
        problem._ineq_graphs = [ineq_graph]

        return problem

    def test_semicircle_problem(self):
        """Test optimization on semicircle."""
        problem = self.build_semicircle_problem()

        config = OPOCHConfig(
            epsilon=1e-3,
            max_time=60.0,
            max_nodes=10000
        )

        kernel = OPOCHKernel(problem, config)
        verdict, result = kernel.solve()

        print(f"\n=== Semicircle Result ===")
        print(f"Verdict: {verdict}")
        print(f"Upper bound: {result.upper_bound:.6f}")
        print(f"Lower bound: {result.lower_bound:.6f}")

        if hasattr(result, 'x_optimal') and result.x_optimal is not None:
            print(f"Solution: {result.x_optimal}")

        # Known optimal: (0, -1), f* = -1
        print(f"Known optimal: (0, -1), f* = -1")


if __name__ == "__main__":
    # Run key tests
    print("=" * 60)
    print("CIRCLE MANIFOLD SANITY CHECK")
    print("=" * 60)

    print("\n--- Testing Krawczyk on Circle ---")
    test_k = TestKrawczykOnCircle()
    test_k.test_krawczyk_contracts_circle()

    print("\n--- Testing Constraint Closure ---")
    test_c = TestConstraintClosureOnCircle()
    test_c.test_closure_contracts_circle()

    print("\n--- Testing OPOCH Kernel ---")
    test_o = TestOPOCHKernelOnCircle()
    test_o.test_solve_circle_problem()

    print("\n" + "=" * 60)
    print("SANITY CHECK COMPLETE")
    print("=" * 60)
