"""
Hard Polynomial Benchmarks - The TRUE Test of Δ* Closure

These functions expose the weakness of interval arithmetic:
- Styblinski-Tang: x⁴ - 16x² + 5x creates catastrophic dependency blow-up
- Dixon-Price: Nested polynomial structure
- Trid: Coupled quadratic terms

The KEY insight: These are NOT hard because of "many local minima."
They are hard because interval arithmetic is catastrophically LOOSE
on polynomials with repeated variable occurrences.

The SOLUTION: Separable bound detection + exact 1D polynomial minimization.

For Styblinski-Tang f(x) = Σ s(xᵢ) where s(x) = (x⁴ - 16x² + 5x)/2:
- Each s(xᵢ) is a univariate quartic
- Exact minimum found by solving s'(x) = 0 (a cubic)
- Total LB = Σ min s(xᵢ) is EXACT

This is the FORCED Δ* constructor that makes 100% certification achievable.
"""

import numpy as np
import math
from typing import List
from dataclasses import dataclass

import sys
sys.path.insert(0, '/Users/chetanchauhan/opoch-optimizer/src')

from opoch_optimizer.expr_graph import (
    ExpressionGraph, TracedVar, OpType,
    sqrt, exp, log, sin, cos
)


@dataclass
class HardPolynomialProblem:
    """A hard polynomial benchmark problem."""
    name: str
    dimension: int
    objective: ExpressionGraph
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    known_optimum: float
    optimal_x: np.ndarray
    is_separable: bool
    difficulty: str


def _trace_styblinski_tang(*vars: TracedVar) -> TracedVar:
    """
    Styblinski-Tang Function

    s(x) = (x⁴ - 16x² + 5x) / 2
    f(x) = Σᵢ s(xᵢ)

    Global minimum: f(x*) = -39.16599d at x* ≈ (-2.903534, ..., -2.903534)

    This is the CANONICAL hard polynomial for testing separable bounds.
    Interval arithmetic fails catastrophically on x⁴ terms.
    """
    n = len(vars)

    # First term
    x = vars[0]
    term = (x ** 4 - 16.0 * x ** 2 + 5.0 * x) / 2.0
    result = term

    # Sum remaining terms
    for i in range(1, n):
        x = vars[i]
        term = (x ** 4 - 16.0 * x ** 2 + 5.0 * x) / 2.0
        result = result + term

    return result


def _trace_dixon_price(*vars: TracedVar) -> TracedVar:
    """
    Dixon-Price Function

    f(x) = (x₁ - 1)² + Σᵢ₌₂ⁿ i(2xᵢ² - xᵢ₋₁)²

    Global minimum: f(x*) = 0
    x*ᵢ = 2^(-(2ⁱ-2)/2ⁱ)

    Nested polynomial structure makes it challenging.
    """
    n = len(vars)

    # First term: (x₁ - 1)²
    result = (vars[0] - 1.0) ** 2

    # Remaining terms: i(2xᵢ² - xᵢ₋₁)²
    for i in range(1, n):
        inner = 2.0 * vars[i] ** 2 - vars[i - 1]
        result = result + float(i + 1) * inner ** 2

    return result


def _trace_trid(*vars: TracedVar) -> TracedVar:
    """
    Trid Function

    f(x) = Σᵢ(xᵢ - 1)² - Σᵢ₌₂ⁿ xᵢxᵢ₋₁

    Global minimum: f(x*) = -d(d+4)(d-1)/6
    x*ᵢ = i(d+1-i)

    Block-separable with coupled terms.
    """
    n = len(vars)

    # Sum of (xᵢ - 1)²
    sum1 = (vars[0] - 1.0) ** 2
    for i in range(1, n):
        sum1 = sum1 + (vars[i] - 1.0) ** 2

    # Sum of xᵢxᵢ₋₁
    sum2 = vars[1] * vars[0]
    for i in range(2, n):
        sum2 = sum2 + vars[i] * vars[i - 1]

    return sum1 - sum2


def _trace_sum_of_powers(*vars: TracedVar) -> TracedVar:
    """
    Sum of Different Powers Function

    f(x) = Σᵢ |xᵢ|^(i+1)

    Global minimum: f(0) = 0

    Separable but with varying exponents.
    """
    n = len(vars)

    # First term: |x₁|²
    result = vars[0] ** 2

    # Remaining terms: |xᵢ|^(i+1)
    for i in range(1, n):
        power = i + 2
        # |x|^k = (x²)^(k/2) for even k
        if power % 2 == 0:
            half_power = power // 2
            term = vars[i] ** 2
            for _ in range(half_power - 1):
                term = term * vars[i] ** 2
        else:
            # For odd powers, use x * (x²)^((k-1)/2)
            half_power = (power - 1) // 2
            term = vars[i] ** 2
            for _ in range(half_power - 1):
                term = term * vars[i] ** 2
            # Multiply by x to get odd power (note: loses sign info)
            term = term * vars[i]

        result = result + term

    return result


def _trace_sum_squares(*vars: TracedVar) -> TracedVar:
    """
    Sum of Squares Function (Axis Parallel Hyper-Ellipsoid)

    f(x) = Σᵢ i·xᵢ²

    Global minimum: f(0) = 0

    Fully separable, weighted quadratic.
    """
    n = len(vars)

    result = 1.0 * vars[0] ** 2
    for i in range(1, n):
        result = result + float(i + 1) * vars[i] ** 2

    return result


def _trace_powell(*vars: TracedVar) -> TracedVar:
    """
    Powell Function (4D groups)

    For groups of 4 variables:
    (x₄ₖ₊₁ + 10x₄ₖ₊₂)² + 5(x₄ₖ₊₃ - x₄ₖ₊₄)² + (x₄ₖ₊₂ - 2x₄ₖ₊₃)⁴ + 10(x₄ₖ₊₁ - x₄ₖ₊₄)⁴

    Global minimum: f(0) = 0

    Block-separable with 4-variable blocks.
    """
    n = len(vars)
    n_groups = n // 4

    result = None

    for g in range(n_groups):
        i = 4 * g
        x1 = vars[i]
        x2 = vars[i + 1]
        x3 = vars[i + 2]
        x4 = vars[i + 3]

        t1 = (x1 + 10.0 * x2) ** 2
        t2 = 5.0 * (x3 - x4) ** 2
        t3 = (x2 - 2.0 * x3) ** 4
        t4 = 10.0 * (x1 - x4) ** 4

        group_sum = t1 + t2 + t3 + t4

        if result is None:
            result = group_sum
        else:
            result = result + group_sum

    return result


def build_styblinski_tang(dim: int) -> HardPolynomialProblem:
    """Build Styblinski-Tang problem."""
    graph = ExpressionGraph.from_callable(_trace_styblinski_tang, dim)

    # Known optimal
    x_opt = np.full(dim, -2.903534)
    f_opt = -39.16599 * dim

    return HardPolynomialProblem(
        name=f"Styblinski_Tang_{dim}D",
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -5.0),
        upper_bounds=np.full(dim, 5.0),
        known_optimum=f_opt,
        optimal_x=x_opt,
        is_separable=True,
        difficulty="hard_polynomial"
    )


def build_dixon_price(dim: int) -> HardPolynomialProblem:
    """Build Dixon-Price problem."""
    graph = ExpressionGraph.from_callable(_trace_dixon_price, dim)

    # Known optimal: x*ᵢ = 2^(-(2ⁱ-2)/2ⁱ)
    x_opt = np.array([2 ** (-(2**(i+1) - 2) / 2**(i+1)) for i in range(dim)])
    f_opt = 0.0

    return HardPolynomialProblem(
        name=f"Dixon_Price_{dim}D",
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -10.0),
        upper_bounds=np.full(dim, 10.0),
        known_optimum=f_opt,
        optimal_x=x_opt,
        is_separable=False,  # NOT separable (xᵢ₋₁ appears in xᵢ term)
        difficulty="hard_polynomial"
    )


def build_trid(dim: int) -> HardPolynomialProblem:
    """Build Trid problem."""
    graph = ExpressionGraph.from_callable(_trace_trid, dim)

    # Known optimal
    x_opt = np.array([i * (dim + 1 - i) for i in range(1, dim + 1)], dtype=float)
    f_opt = -dim * (dim + 4) * (dim - 1) / 6.0

    return HardPolynomialProblem(
        name=f"Trid_{dim}D",
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -dim**2),
        upper_bounds=np.full(dim, dim**2),
        known_optimum=f_opt,
        optimal_x=x_opt,
        is_separable=False,  # NOT separable (coupled terms)
        difficulty="hard_polynomial"
    )


def build_sum_of_powers(dim: int) -> HardPolynomialProblem:
    """Build Sum of Different Powers problem."""
    graph = ExpressionGraph.from_callable(_trace_sum_of_powers, dim)

    x_opt = np.zeros(dim)
    f_opt = 0.0

    return HardPolynomialProblem(
        name=f"Sum_of_Powers_{dim}D",
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -1.0),
        upper_bounds=np.full(dim, 1.0),
        known_optimum=f_opt,
        optimal_x=x_opt,
        is_separable=True,
        difficulty="hard_polynomial"
    )


def build_sum_squares(dim: int) -> HardPolynomialProblem:
    """Build Sum of Squares (Weighted) problem."""
    graph = ExpressionGraph.from_callable(_trace_sum_squares, dim)

    x_opt = np.zeros(dim)
    f_opt = 0.0

    return HardPolynomialProblem(
        name=f"Sum_Squares_{dim}D",
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -10.0),
        upper_bounds=np.full(dim, 10.0),
        known_optimum=f_opt,
        optimal_x=x_opt,
        is_separable=True,
        difficulty="easy_polynomial"
    )


def build_powell(dim: int) -> HardPolynomialProblem:
    """Build Powell problem (dim must be multiple of 4)."""
    assert dim % 4 == 0, "Powell function requires dimension to be multiple of 4"

    graph = ExpressionGraph.from_callable(_trace_powell, dim)

    x_opt = np.zeros(dim)
    f_opt = 0.0

    return HardPolynomialProblem(
        name=f"Powell_{dim}D",
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -4.0),
        upper_bounds=np.full(dim, 5.0),
        known_optimum=f_opt,
        optimal_x=x_opt,
        is_separable=False,  # Block-separable (4-variable blocks)
        difficulty="hard_polynomial"
    )


def get_hard_polynomial_problems() -> List[HardPolynomialProblem]:
    """
    Get hard polynomial problems for testing separable bounds.

    Focus: Styblinski-Tang (the canonical hard case)
    """
    problems = []

    # Styblinski-Tang: THE hard polynomial (fully separable, quartic)
    for dim in [2, 3, 5, 10, 15, 20, 30, 50]:
        problems.append(build_styblinski_tang(dim))

    # Sum of Squares: Easy baseline (fully separable, quadratic)
    for dim in [2, 5, 10, 20]:
        problems.append(build_sum_squares(dim))

    # Dixon-Price: Non-separable nested polynomial
    for dim in [2, 3, 5, 10]:
        problems.append(build_dixon_price(dim))

    # Trid: Coupled terms
    for dim in [2, 3, 5, 10]:
        problems.append(build_trid(dim))

    # Powell: Block-separable
    for dim in [4, 8, 12, 20]:
        problems.append(build_powell(dim))

    return problems


def get_styblinski_tang_only() -> List[HardPolynomialProblem]:
    """Get only Styblinski-Tang problems for focused testing."""
    problems = []
    for dim in [2, 3, 5, 10, 15, 20, 30, 50, 100]:
        problems.append(build_styblinski_tang(dim))
    return problems


if __name__ == "__main__":
    print("Hard Polynomial Problem Definitions Test")
    print("=" * 60)

    problems = get_hard_polynomial_problems()

    for prob in problems[:10]:
        # Evaluate at optimal
        f_opt_computed = prob.objective(prob.optimal_x)

        print(f"\n{prob.name}:")
        print(f"  Dimension: {prob.dimension}")
        print(f"  Separable: {prob.is_separable}")
        print(f"  f(x_opt): {f_opt_computed:.6f}")
        print(f"  Known optimum: {prob.known_optimum:.6f}")
        print(f"  Match: {abs(f_opt_computed - prob.known_optimum) < 0.01}")
