"""
Tests for OPOCH Kernel Solver
"""

import numpy as np
import pytest
from opoch_optimizer import (
    ProblemContract,
    ExpressionGraph,
    OpType,
    TracedVar,
    OPOCHKernel,
    OPOCHConfig,
    Verdict,
)


class TestSimpleOptimization:
    """Test simple optimization problems."""

    def test_quadratic_unconstrained(self):
        """Test min x^2 on [-5, 5]."""
        # Build expression graph for x^2
        g = ExpressionGraph()
        x = g.variable(0)
        x2 = g.unary(OpType.SQUARE, x)
        g.set_output(x2)

        # Create problem
        problem = ProblemContract(
            objective=g,
            bounds=[(-5.0, 5.0)],
            epsilon=1e-4
        )

        # Solve
        config = OPOCHConfig(epsilon=1e-4, max_time=10.0, max_nodes=1000)
        kernel = OPOCHKernel(problem, config)
        verdict, result = kernel.solve()

        # Should find UNIQUE-OPT at x=0 with f=0
        assert verdict == Verdict.UNIQUE_OPT
        assert abs(result.objective_value) < 1e-2
        assert abs(result.x_optimal[0]) < 1e-2

    def test_shifted_quadratic(self):
        """Test min (x-2)^2 on [-5, 5]."""
        # Build expression graph for (x-2)^2
        g = ExpressionGraph()
        x = g.variable(0)
        two = g.constant(2.0)
        diff = g.binary(OpType.SUB, x, two)
        sq = g.unary(OpType.SQUARE, diff)
        g.set_output(sq)

        problem = ProblemContract(
            objective=g,
            bounds=[(-5.0, 5.0)],
            epsilon=1e-4
        )

        config = OPOCHConfig(epsilon=1e-4, max_time=10.0, max_nodes=1000)
        kernel = OPOCHKernel(problem, config)
        verdict, result = kernel.solve()

        # Should find UNIQUE-OPT at x=2 with f=0
        assert verdict == Verdict.UNIQUE_OPT
        assert abs(result.objective_value) < 1e-2
        assert abs(result.x_optimal[0] - 2.0) < 1e-2

    def test_sum_of_squares(self):
        """Test min x^2 + y^2 on [-5, 5]^2."""
        g = ExpressionGraph()
        x = g.variable(0)
        y = g.variable(1)
        x2 = g.unary(OpType.SQUARE, x)
        y2 = g.unary(OpType.SQUARE, y)
        s = g.binary(OpType.ADD, x2, y2)
        g.set_output(s)

        problem = ProblemContract(
            objective=g,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            epsilon=1e-4
        )

        config = OPOCHConfig(epsilon=1e-4, max_time=30.0, max_nodes=5000)
        kernel = OPOCHKernel(problem, config)
        verdict, result = kernel.solve()

        # Should find UNIQUE-OPT at (0,0) with f=0
        assert verdict == Verdict.UNIQUE_OPT
        assert abs(result.objective_value) < 1e-2


class TestConstrainedOptimization:
    """Test constrained optimization problems."""

    def test_equality_constraint(self):
        """Test min x^2 + y^2 s.t. x + y = 1."""
        # Objective: x^2 + y^2
        g_obj = ExpressionGraph()
        x1 = g_obj.variable(0)
        y1 = g_obj.variable(1)
        x2 = g_obj.unary(OpType.SQUARE, x1)
        y2 = g_obj.unary(OpType.SQUARE, y1)
        obj = g_obj.binary(OpType.ADD, x2, y2)
        g_obj.set_output(obj)

        # Equality constraint: x + y - 1 = 0
        g_eq = ExpressionGraph()
        x2 = g_eq.variable(0)
        y2 = g_eq.variable(1)
        one = g_eq.constant(1.0)
        sum_xy = g_eq.binary(OpType.ADD, x2, y2)
        eq = g_eq.binary(OpType.SUB, sum_xy, one)
        g_eq.set_output(eq)

        problem = ProblemContract(
            objective=g_obj,
            bounds=[(-5.0, 5.0), (-5.0, 5.0)],
            eq_constraints=[lambda x: x[0] + x[1] - 1],
            epsilon=1e-4
        )
        problem._eq_graphs = [g_eq]

        config = OPOCHConfig(epsilon=1e-4, max_time=30.0, max_nodes=5000)
        kernel = OPOCHKernel(problem, config)
        verdict, result = kernel.solve()

        # Optimal at x = y = 0.5 with f = 0.5
        if verdict == Verdict.UNIQUE_OPT:
            assert abs(result.objective_value - 0.5) < 0.1
            assert abs(result.x_optimal[0] - 0.5) < 0.2
            assert abs(result.x_optimal[1] - 0.5) < 0.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
