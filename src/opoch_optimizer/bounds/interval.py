"""
Interval Arithmetic (Tier 0 Bounds)

Provides rigorous interval enclosures for function evaluation.
Each operation is computed with proper rounding to ensure
the true value is always contained in the resulting interval.

This is the foundation of certified global optimization:
- Every function value is guaranteed to be in the computed interval
- Lower bound of objective interval gives certified LB for region
- Constraint intervals enable feasibility refutation

Note: For production use, this should use directed rounding
(e.g., via MPFR or hardware rounding modes). This implementation
uses conservative outward rounding via small epsilon.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..expr_graph import (
    ExpressionGraph,
    ExprNode,
    Variable,
    Constant,
    UnaryOp,
    BinaryOp,
    OpType,
)


# Small epsilon for conservative rounding (production: use directed rounding)
ROUND_EPS = 1e-15


@dataclass
class Interval:
    """
    A closed interval [lo, hi] with arithmetic operations.

    All operations are computed with outward rounding to ensure
    the true result is always contained.
    """
    lo: float
    hi: float

    def __post_init__(self):
        # Allow empty intervals (lo > hi represents empty set)
        # This is important for FBBT and infeasibility detection
        if self.lo > self.hi + ROUND_EPS:
            # Check if this is intentionally empty (inf, -inf)
            if not (self.lo == float('inf') and self.hi == float('-inf')):
                # For small violations due to numerics, collapse to point
                if self.lo > self.hi and self.lo <= self.hi + 1e-10:
                    mid = (self.lo + self.hi) / 2
                    self.lo = mid
                    self.hi = mid
                else:
                    raise ValueError(f"Invalid interval: [{self.lo}, {self.hi}]")
        # Handle small numerical issues (near-empty)
        elif self.lo > self.hi:
            mid = (self.lo + self.hi) / 2
            self.lo = mid
            self.hi = mid

    @classmethod
    def point(cls, x: float) -> 'Interval':
        """Create a point interval [x, x]."""
        return cls(x, x)

    @classmethod
    def empty(cls) -> 'Interval':
        """Create an empty interval (for infeasibility)."""
        return cls(float('inf'), float('-inf'))

    @classmethod
    def entire(cls) -> 'Interval':
        """Create the entire real line."""
        return cls(float('-inf'), float('inf'))

    @property
    def is_empty(self) -> bool:
        return self.lo > self.hi

    @property
    def width(self) -> float:
        if self.is_empty:
            return 0.0
        return self.hi - self.lo

    @property
    def midpoint(self) -> float:
        if self.is_empty:
            return float('nan')
        return (self.lo + self.hi) / 2.0

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def contains_zero(self) -> bool:
        return self.lo <= 0 <= self.hi

    def intersect(self, other: 'Interval') -> 'Interval':
        """Intersection of two intervals."""
        new_lo = max(self.lo, other.lo)
        new_hi = min(self.hi, other.hi)
        if new_lo > new_hi + ROUND_EPS:
            return Interval.empty()
        return Interval(new_lo, new_hi)

    def union_hull(self, other: 'Interval') -> 'Interval':
        """Convex hull of two intervals."""
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    # Arithmetic operations with outward rounding

    def __neg__(self) -> 'Interval':
        return Interval(-self.hi, -self.lo)

    def __add__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, (int, float)):
            other = Interval.point(other)
        return Interval(
            self.lo + other.lo - ROUND_EPS,
            self.hi + other.hi + ROUND_EPS
        )

    def __radd__(self, other) -> 'Interval':
        return self.__add__(Interval.point(float(other)))

    def __sub__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, (int, float)):
            other = Interval.point(other)
        return Interval(
            self.lo - other.hi - ROUND_EPS,
            self.hi - other.lo + ROUND_EPS
        )

    def __rsub__(self, other) -> 'Interval':
        return Interval.point(float(other)).__sub__(self)

    def __mul__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, (int, float)):
            other = Interval.point(other)

        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi
        ]
        return Interval(
            min(products) - ROUND_EPS,
            max(products) + ROUND_EPS
        )

    def __rmul__(self, other) -> 'Interval':
        return self.__mul__(Interval.point(float(other)))

    def __truediv__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, (int, float)):
            other = Interval.point(other)

        # Check for division by zero
        if other.contains_zero():
            if other.lo == 0 and other.hi == 0:
                return Interval.entire()  # 0/0 case
            elif other.lo == 0:
                # [a,b] / [0,d] where d > 0
                recip = Interval(1.0 / other.hi - ROUND_EPS, float('inf'))
            elif other.hi == 0:
                # [a,b] / [c,0] where c < 0
                recip = Interval(float('-inf'), 1.0 / other.lo + ROUND_EPS)
            else:
                # [a,b] / [c,d] where c < 0 < d
                return Interval.entire()
        else:
            recip = Interval(
                1.0 / other.hi - ROUND_EPS,
                1.0 / other.lo + ROUND_EPS
            )

        return self * recip

    def __rtruediv__(self, other) -> 'Interval':
        return Interval.point(float(other)).__truediv__(self)

    def __pow__(self, n: Union[int, 'Interval']) -> 'Interval':
        """Power operation x^n."""
        if isinstance(n, int):
            return self._pow_int(n)
        elif isinstance(n, Interval):
            return self._pow_interval(n)
        else:
            return self._pow_interval(Interval.point(float(n)))

    def _pow_int(self, n: int) -> 'Interval':
        """Integer power with proper interval handling."""
        if n == 0:
            return Interval.point(1.0)
        elif n == 1:
            return Interval(self.lo, self.hi)
        elif n == 2:
            return self.square()
        elif n > 0 and n % 2 == 0:
            # Even power
            if self.hi <= 0:
                return Interval(
                    self.hi ** n - ROUND_EPS,
                    self.lo ** n + ROUND_EPS
                )
            elif self.lo >= 0:
                return Interval(
                    self.lo ** n - ROUND_EPS,
                    self.hi ** n + ROUND_EPS
                )
            else:
                return Interval(
                    0 - ROUND_EPS,
                    max(self.lo ** n, self.hi ** n) + ROUND_EPS
                )
        else:
            # Odd power or negative
            vals = [self.lo ** n, self.hi ** n]
            return Interval(min(vals) - ROUND_EPS, max(vals) + ROUND_EPS)

    def _pow_interval(self, n: 'Interval') -> 'Interval':
        """General power x^y via exp(y*log(x))."""
        if self.lo <= 0:
            # Cannot take log of non-positive
            if self.hi <= 0:
                return Interval.entire()  # Undefined/complex
            # Restrict to positive part for now
            self_pos = Interval(max(self.lo, ROUND_EPS), self.hi)
            return self_pos._pow_interval(n)

        return (n * self.log()).exp()

    def square(self) -> 'Interval':
        """Optimized x^2 computation."""
        if self.hi <= 0:
            return Interval(
                self.hi * self.hi - ROUND_EPS,
                self.lo * self.lo + ROUND_EPS
            )
        elif self.lo >= 0:
            return Interval(
                self.lo * self.lo - ROUND_EPS,
                self.hi * self.hi + ROUND_EPS
            )
        else:
            # Interval contains zero
            return Interval(
                -ROUND_EPS,
                max(self.lo * self.lo, self.hi * self.hi) + ROUND_EPS
            )

    def abs(self) -> 'Interval':
        """Absolute value."""
        if self.lo >= 0:
            return Interval(self.lo, self.hi)
        elif self.hi <= 0:
            return Interval(-self.hi, -self.lo)
        else:
            return Interval(0, max(-self.lo, self.hi))

    def sqrt(self) -> 'Interval':
        """Square root (defined for non-negative)."""
        if self.hi < 0:
            return Interval.empty()  # Undefined

        lo = max(0, self.lo)
        return Interval(
            np.sqrt(lo) - ROUND_EPS if lo > 0 else 0,
            np.sqrt(self.hi) + ROUND_EPS
        )

    def exp(self) -> 'Interval':
        """Exponential function."""
        return Interval(
            np.exp(self.lo) - ROUND_EPS,
            np.exp(self.hi) + ROUND_EPS
        )

    def log(self) -> 'Interval':
        """Natural logarithm (defined for positive)."""
        if self.hi <= 0:
            return Interval.empty()

        lo = max(ROUND_EPS, self.lo)
        return Interval(
            np.log(lo) - ROUND_EPS,
            np.log(self.hi) + ROUND_EPS
        )

    def sin(self) -> 'Interval':
        """Sine function with proper range handling."""
        # For wide intervals, return [-1, 1]
        if self.width >= 2 * np.pi:
            return Interval(-1, 1)

        # Reduce to [0, 2*pi]
        lo_red = self.lo % (2 * np.pi)
        hi_red = lo_red + self.width

        vals = [np.sin(lo_red), np.sin(hi_red)]

        # Check if extrema are in the interval
        # Max at pi/2 + 2k*pi
        if lo_red <= np.pi/2 <= hi_red or lo_red <= np.pi/2 + 2*np.pi <= hi_red:
            vals.append(1)
        # Min at 3*pi/2 + 2k*pi
        if lo_red <= 3*np.pi/2 <= hi_red or lo_red <= 3*np.pi/2 + 2*np.pi <= hi_red:
            vals.append(-1)

        return Interval(min(vals) - ROUND_EPS, max(vals) + ROUND_EPS)

    def cos(self) -> 'Interval':
        """Cosine function."""
        return (self + np.pi/2).sin()

    def tan(self) -> 'Interval':
        """Tangent function."""
        # Check for discontinuity
        if self.width >= np.pi:
            return Interval.entire()

        lo_red = self.lo % np.pi
        hi_red = lo_red + self.width

        # Check if pi/2 is in the interval
        if lo_red < np.pi/2 < hi_red:
            return Interval.entire()

        return Interval(
            np.tan(self.lo) - ROUND_EPS,
            np.tan(self.hi) + ROUND_EPS
        )

    def to_canonical(self) -> Dict[str, Any]:
        return {"lo": self.lo, "hi": self.hi}

    def __repr__(self) -> str:
        return f"[{self.lo:.6g}, {self.hi:.6g}]"


class IntervalEvaluator:
    """
    Evaluates an expression graph over intervals.

    Given variable bounds (as intervals), computes interval enclosures
    for all nodes in the graph.
    """

    def __init__(self, graph: ExpressionGraph):
        self.graph = graph
        self._node_intervals: Dict[int, Interval] = {}

    def evaluate(
        self,
        var_intervals: Dict[int, Interval]
    ) -> Tuple[Interval, Dict[int, Interval]]:
        """
        Evaluate the graph over given variable intervals.

        Args:
            var_intervals: Map from variable index to interval

        Returns:
            Tuple of (output interval, all node intervals)
        """
        self._node_intervals = {}

        # Process nodes in topological order
        for node in self.graph.topological_order():
            self._node_intervals[node.node_id] = self._eval_node(node, var_intervals)

        output_interval = self._node_intervals[self.graph.output_node.node_id]
        return output_interval, self._node_intervals.copy()

    def _eval_node(
        self,
        node: ExprNode,
        var_intervals: Dict[int, Interval]
    ) -> Interval:
        """Evaluate a single node."""

        if isinstance(node, Variable):
            return var_intervals[node.var_index]

        elif isinstance(node, Constant):
            return Interval.point(node.value)

        elif isinstance(node, UnaryOp):
            child_int = self._node_intervals[node.child.node_id]
            return self._eval_unary(node.op, child_int)

        elif isinstance(node, BinaryOp):
            left_int = self._node_intervals[node.left.node_id]
            right_int = self._node_intervals[node.right.node_id]
            return self._eval_binary(node.op, left_int, right_int)

        else:
            raise ValueError(f"Unknown node type: {type(node)}")

    def _eval_unary(self, op: OpType, x: Interval) -> Interval:
        """Evaluate a unary operation on an interval."""
        if op == OpType.NEG:
            return -x
        elif op == OpType.ABS:
            return x.abs()
        elif op == OpType.SQRT:
            return x.sqrt()
        elif op == OpType.EXP:
            return x.exp()
        elif op == OpType.LOG:
            return x.log()
        elif op == OpType.SIN:
            return x.sin()
        elif op == OpType.COS:
            return x.cos()
        elif op == OpType.TAN:
            return x.tan()
        elif op == OpType.SQUARE:
            return x.square()
        elif op == OpType.SINH:
            return ((x.exp() - (-x).exp()) / 2)
        elif op == OpType.COSH:
            return ((x.exp() + (-x).exp()) / 2)
        elif op == OpType.TANH:
            exp_x = x.exp()
            exp_neg_x = (-x).exp()
            return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)
        elif op == OpType.ASIN:
            # arcsin defined on [-1, 1]
            x_clamped = x.intersect(Interval(-1, 1))
            if x_clamped.is_empty:
                return Interval.empty()
            return Interval(
                np.arcsin(x_clamped.lo) - ROUND_EPS,
                np.arcsin(x_clamped.hi) + ROUND_EPS
            )
        elif op == OpType.ACOS:
            x_clamped = x.intersect(Interval(-1, 1))
            if x_clamped.is_empty:
                return Interval.empty()
            return Interval(
                np.arccos(x_clamped.hi) - ROUND_EPS,
                np.arccos(x_clamped.lo) + ROUND_EPS
            )
        elif op == OpType.ATAN:
            return Interval(
                np.arctan(x.lo) - ROUND_EPS,
                np.arctan(x.hi) + ROUND_EPS
            )
        else:
            raise ValueError(f"Unknown unary op: {op}")

    def _eval_binary(self, op: OpType, l: Interval, r: Interval) -> Interval:
        """Evaluate a binary operation on intervals."""
        if op == OpType.ADD:
            return l + r
        elif op == OpType.SUB:
            return l - r
        elif op == OpType.MUL:
            return l * r
        elif op == OpType.DIV:
            return l / r
        elif op == OpType.POW:
            # Check if exponent is constant integer
            if r.lo == r.hi and r.lo == int(r.lo):
                return l._pow_int(int(r.lo))
            else:
                return l._pow_interval(r)
        elif op == OpType.MIN:
            return Interval(min(l.lo, r.lo), min(l.hi, r.hi))
        elif op == OpType.MAX:
            return Interval(max(l.lo, r.lo), max(l.hi, r.hi))
        else:
            raise ValueError(f"Unknown binary op: {op}")


def interval_evaluate(
    func: Union[ExpressionGraph, callable],
    lower: np.ndarray,
    upper: np.ndarray
) -> Interval:
    """
    Evaluate a function over a box using interval arithmetic.

    This is the main entry point for Tier 0 bound computation.

    Args:
        func: ExpressionGraph or callable
        lower: Lower bounds of the box
        upper: Upper bounds of the box

    Returns:
        Interval enclosure of the function over the box
    """
    n = len(lower)
    var_intervals = {
        i: Interval(lower[i], upper[i])
        for i in range(n)
    }

    if isinstance(func, ExpressionGraph):
        evaluator = IntervalEvaluator(func)
        result, _ = evaluator.evaluate(var_intervals)
        return result
    else:
        # For callable, use natural interval extension via sampling
        # This is a fallback - less tight than graph-based
        # Sample corners and midpoint
        samples = []
        for corner in _box_corners(lower, upper, max_corners=32):
            try:
                samples.append(func(corner))
            except:
                pass

        center = (lower + upper) / 2
        try:
            samples.append(func(center))
        except:
            pass

        if not samples:
            return Interval.entire()

        # Conservative interval (not rigorous for general callables)
        lo = min(samples) - 0.1 * (max(samples) - min(samples)) - ROUND_EPS
        hi = max(samples) + 0.1 * (max(samples) - min(samples)) + ROUND_EPS
        return Interval(lo, hi)


def _box_corners(lower: np.ndarray, upper: np.ndarray, max_corners: int = 32) -> List[np.ndarray]:
    """Generate corner points of a box."""
    n = len(lower)
    if 2**n <= max_corners:
        # All corners
        corners = []
        for i in range(2**n):
            corner = np.array([
                lower[j] if (i >> j) & 1 == 0 else upper[j]
                for j in range(n)
            ])
            corners.append(corner)
        return corners
    else:
        # Sample subset of corners
        import random
        random.seed(42)  # Deterministic
        corners = []
        indices = random.sample(range(2**n), max_corners)
        for i in indices:
            corner = np.array([
                lower[j] if (i >> j) & 1 == 0 else upper[j]
                for j in range(n)
            ])
            corners.append(corner)
        return corners
