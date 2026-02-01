"""
Tests for FBBT (Feasibility-Based Bound Tightening)
"""

import numpy as np
import pytest
from opoch_optimizer.expr_graph import ExpressionGraph, OpType
from opoch_optimizer.bounds.fbbt import FBBTOperator, FBBTInequalityOperator, apply_fbbt_all_constraints


class TestFBBTEquality:
    """Test FBBT for equality constraints."""

    def test_simple_linear(self):
        """Test FBBT on x + y = 0."""
        g = ExpressionGraph()
        x = g.variable(0)
        y = g.variable(1)
        s = g.binary(OpType.ADD, x, y)
        g.set_output(s)

        op = FBBTOperator(g, 2)

        lower = np.array([-5.0, -5.0])
        upper = np.array([5.0, 5.0])

        result = op.tighten(lower, upper)

        # For x + y = 0 with x,y in [-5,5]:
        # x = -y, so x in [-5, 5] means y in [-5, 5] (already satisfied)
        # But the constraint x + y = 0 should tighten bounds
        assert not result.empty


class TestFBBTInequality:
    """Test FBBT for inequality constraints."""

    def test_simple_bound(self):
        """Test FBBT on x - 1 <= 0 (i.e., x <= 1)."""
        g = ExpressionGraph()
        x = g.variable(0)
        one = g.constant(1.0)
        diff = g.binary(OpType.SUB, x, one)
        g.set_output(diff)

        op = FBBTInequalityOperator(g, 1)

        lower = np.array([-5.0])
        upper = np.array([5.0])

        result = op.tighten(lower, upper)

        # x - 1 <= 0 means x <= 1
        # Upper bound should be tightened to 1
        assert not result.empty
        assert result.upper[0] <= 1.0 + 1e-8

    def test_infeasible_inequality(self):
        """Test FBBT on x - 10 <= 0 when x >= 11."""
        g = ExpressionGraph()
        x = g.variable(0)
        ten = g.constant(10.0)
        diff = g.binary(OpType.SUB, x, ten)
        g.set_output(diff)

        op = FBBTInequalityOperator(g, 1)

        lower = np.array([11.0])  # x >= 11
        upper = np.array([20.0])

        result = op.tighten(lower, upper)

        # x - 10 <= 0 means x <= 10, but x >= 11, so infeasible
        assert result.empty


class TestCombinedFBBT:
    """Test combined FBBT for multiple constraints."""

    def test_box_tightening(self):
        """Test FBBT with both eq and ineq constraints."""
        # Create constraint graphs
        # Equality: x + y = 2
        g_eq = ExpressionGraph()
        x1 = g_eq.variable(0)
        y1 = g_eq.variable(1)
        two = g_eq.constant(2.0)
        eq = g_eq.binary(OpType.SUB, g_eq.binary(OpType.ADD, x1, y1), two)
        g_eq.set_output(eq)

        # Inequality: x <= 3
        g_ineq = ExpressionGraph()
        x2 = g_ineq.variable(0)
        three = g_ineq.constant(3.0)
        ineq = g_ineq.binary(OpType.SUB, x2, three)
        g_ineq.set_output(ineq)

        lower = np.array([0.0, 0.0])
        upper = np.array([5.0, 5.0])

        result = apply_fbbt_all_constraints(
            eq_constraints=[g_eq],
            ineq_constraints=[g_ineq],
            n_vars=2,
            lower=lower,
            upper=upper
        )

        assert not result.empty
        # x <= 3 should be enforced
        assert result.upper[0] <= 3.0 + 1e-8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
