"""
Tests for Expression Graph
"""

import numpy as np
import pytest
from opoch_optimizer.expr_graph import ExpressionGraph, OpType, TracedVar


class TestExpressionGraph:
    """Test expression graph creation and evaluation."""

    def test_constant(self):
        """Test constant node."""
        g = ExpressionGraph()
        c = g.constant(5.0)
        g.set_output(c)
        assert g.evaluate(np.array([1.0, 2.0])) == 5.0

    def test_variable(self):
        """Test variable node."""
        g = ExpressionGraph()
        x = g.variable(0)
        g.set_output(x)
        assert g.evaluate(np.array([3.0])) == 3.0

    def test_addition(self):
        """Test addition operation."""
        g = ExpressionGraph()
        x = g.variable(0)
        y = g.variable(1)
        s = g.binary(OpType.ADD, x, y)
        g.set_output(s)
        assert g.evaluate(np.array([2.0, 3.0])) == 5.0

    def test_multiplication(self):
        """Test multiplication operation."""
        g = ExpressionGraph()
        x = g.variable(0)
        y = g.variable(1)
        p = g.binary(OpType.MUL, x, y)
        g.set_output(p)
        assert g.evaluate(np.array([2.0, 3.0])) == 6.0

    def test_square(self):
        """Test square operation."""
        g = ExpressionGraph()
        x = g.variable(0)
        sq = g.unary(OpType.SQUARE, x)
        g.set_output(sq)
        assert g.evaluate(np.array([4.0])) == 16.0

    def test_complex_expression(self):
        """Test complex expression: x^2 + 2*x*y + y^2 = (x+y)^2"""
        g = ExpressionGraph()
        x = g.variable(0)
        y = g.variable(1)

        x2 = g.unary(OpType.SQUARE, x)
        y2 = g.unary(OpType.SQUARE, y)
        xy = g.binary(OpType.MUL, x, y)
        two = g.constant(2.0)
        two_xy = g.binary(OpType.MUL, two, xy)

        sum1 = g.binary(OpType.ADD, x2, two_xy)
        result = g.binary(OpType.ADD, sum1, y2)
        g.set_output(result)

        # (2+3)^2 = 25
        assert abs(g.evaluate(np.array([2.0, 3.0])) - 25.0) < 1e-10


class TestTracedVar:
    """Test traced variable for automatic graph construction."""

    def test_traced_addition(self):
        """Test traced variable addition."""
        g = ExpressionGraph()
        x = TracedVar(g, g.variable(0))
        y = TracedVar(g, g.variable(1))
        z = x + y
        g.set_output(z.node)
        assert g.evaluate(np.array([2.0, 3.0])) == 5.0

    def test_traced_quadratic(self):
        """Test traced variable quadratic."""
        g = ExpressionGraph()
        x = TracedVar(g, g.variable(0))
        z = x * x + 2 * x + 1  # (x+1)^2
        g.set_output(z.node)
        # (3+1)^2 = 16
        assert abs(g.evaluate(np.array([3.0])) - 16.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
