"""
Tests for Interval Arithmetic (Tier 0 Bounds)
"""

import numpy as np
import pytest
from opoch_optimizer.bounds.interval import Interval, ROUND_EPS


class TestInterval:
    """Test basic interval operations."""

    def test_creation(self):
        """Test interval creation."""
        iv = Interval(1.0, 2.0)
        assert iv.lo == 1.0
        assert iv.hi == 2.0

    def test_point_interval(self):
        """Test point interval creation."""
        iv = Interval.point(3.0)
        assert iv.lo == 3.0
        assert iv.hi == 3.0

    def test_empty_interval(self):
        """Test empty interval."""
        iv = Interval.empty()
        assert iv.is_empty

    def test_contains(self):
        """Test containment."""
        iv = Interval(1.0, 3.0)
        assert iv.contains(2.0)
        assert iv.contains(1.0)
        assert iv.contains(3.0)
        assert not iv.contains(0.0)
        assert not iv.contains(4.0)

    def test_width(self):
        """Test width computation."""
        iv = Interval(1.0, 4.0)
        assert iv.width == 3.0

    def test_midpoint(self):
        """Test midpoint computation."""
        iv = Interval(1.0, 3.0)
        assert iv.midpoint == 2.0


class TestIntervalArithmetic:
    """Test interval arithmetic operations."""

    def test_addition(self):
        """Test interval addition."""
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        c = a + b
        assert c.lo <= 4.0 <= c.hi
        assert c.lo <= 6.0 <= c.hi

    def test_subtraction(self):
        """Test interval subtraction."""
        a = Interval(3.0, 5.0)
        b = Interval(1.0, 2.0)
        c = a - b
        assert c.lo <= 1.0 <= c.hi
        assert c.lo <= 4.0 <= c.hi

    def test_multiplication(self):
        """Test interval multiplication."""
        a = Interval(2.0, 3.0)
        b = Interval(4.0, 5.0)
        c = a * b
        assert c.lo <= 8.0 <= c.hi
        assert c.lo <= 15.0 <= c.hi

    def test_square(self):
        """Test interval squaring."""
        a = Interval(-2.0, 3.0)
        c = a.square()
        # x^2 on [-2,3] has range [0, 9]
        assert c.lo <= 0.0
        assert c.hi >= 9.0

    def test_sqrt(self):
        """Test interval square root."""
        a = Interval(4.0, 9.0)
        c = a.sqrt()
        assert 2.0 - ROUND_EPS <= c.lo
        assert c.hi <= 3.0 + ROUND_EPS

    def test_exp(self):
        """Test interval exponential."""
        a = Interval(0.0, 1.0)
        c = a.exp()
        assert c.lo <= 1.0 <= c.hi
        assert c.lo <= np.e <= c.hi

    def test_log(self):
        """Test interval logarithm."""
        a = Interval(1.0, np.e)
        c = a.log()
        assert c.lo <= 0.0 <= c.hi
        assert c.lo <= 1.0 <= c.hi


class TestIntervalIntersection:
    """Test interval intersection."""

    def test_overlapping(self):
        """Test overlapping intervals."""
        a = Interval(1.0, 3.0)
        b = Interval(2.0, 4.0)
        c = a.intersect(b)
        assert c.lo >= 2.0 - ROUND_EPS
        assert c.hi <= 3.0 + ROUND_EPS

    def test_disjoint(self):
        """Test disjoint intervals."""
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        c = a.intersect(b)
        assert c.is_empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
