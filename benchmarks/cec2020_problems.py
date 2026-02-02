"""
CEC 2020 Benchmark Problems as Expression Graphs

This module defines the CEC 2020 benchmark functions as expression graphs
for use with the OPOCH mathematical certification kernel.

CEC 2020 contains 10 test functions:
F1: Bent Cigar Function (unimodal)
F2: Schwefel's Function (multimodal)
F3: Lunacek Bi-Rastrigin Function (multimodal)
F4: Expanded Rosenbrock plus Griewank (multimodal)
F5-F7: Hybrid Functions
F8-F10: Composition Functions

For mathematical certification, we express these as factorable expression
graphs and apply interval arithmetic, McCormick relaxations, and Δ* closure.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import math

import sys
sys.path.insert(0, '/Users/chetanchauhan/opoch-optimizer/src')

from opoch_optimizer.expr_graph import (
    ExpressionGraph, TracedVar, OpType,
    sqrt, exp, log, sin, cos
)


@dataclass
class CEC2020Problem:
    """A CEC 2020 benchmark problem."""
    name: str
    function_id: int
    dimension: int
    objective: ExpressionGraph
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    known_optimum: float  # f_bias for CEC functions
    difficulty: str  # unimodal, multimodal, hybrid, composition


def _sum_squares(vars: List[TracedVar]) -> TracedVar:
    """Sum of squares: Σxᵢ²"""
    result = vars[0] ** 2
    for v in vars[1:]:
        result = result + v ** 2
    return result


def _trace_bent_cigar(x0: TracedVar, *rest: TracedVar) -> TracedVar:
    """
    Bent Cigar Function (F1)
    f(x) = x₀² + 10⁶ * Σᵢ₌₁ⁿ⁻¹ xᵢ²

    Unimodal, smooth, has a narrow ridge leading to the optimum.
    """
    result = x0 ** 2
    if rest:
        sum_rest = rest[0] ** 2
        for v in rest[1:]:
            sum_rest = sum_rest + v ** 2
        result = result + 1e6 * sum_rest
    return result


def _trace_schwefel_12(x0: TracedVar, *rest: TracedVar) -> TracedVar:
    """
    Schwefel's Problem 1.2 (used in F2 base)
    f(x) = Σᵢ₌₀ⁿ⁻¹ (Σⱼ₌₀ⁱ xⱼ)²

    Non-separable, smooth.
    """
    all_vars = [x0] + list(rest)
    n = len(all_vars)

    total = all_vars[0] ** 2  # i=0: (x₀)²
    cumsum = all_vars[0]

    for i in range(1, n):
        cumsum = cumsum + all_vars[i]
        total = total + cumsum ** 2

    return total


def _trace_rosenbrock(*vars: TracedVar) -> TracedVar:
    """
    Rosenbrock Function (used in F4 base)
    f(x) = Σᵢ₌₀ⁿ⁻² [100(xᵢ₊₁ - xᵢ²)² + (xᵢ - 1)²]

    Classic non-convex function with a narrow curved valley.
    """
    n = len(vars)
    if n < 2:
        return vars[0] ** 2

    result = 100.0 * (vars[1] - vars[0] ** 2) ** 2 + (vars[0] - 1.0) ** 2

    for i in range(1, n - 1):
        term = 100.0 * (vars[i + 1] - vars[i] ** 2) ** 2 + (vars[i] - 1.0) ** 2
        result = result + term

    return result


def _trace_rastrigin(*vars: TracedVar) -> TracedVar:
    """
    Rastrigin Function
    f(x) = 10n + Σᵢ₌₀ⁿ⁻¹ [xᵢ² - 10cos(2πxᵢ)]

    Highly multimodal with 10^n local minima.
    """
    n = len(vars)
    result = vars[0] ** 2 - 10.0 * cos(2.0 * math.pi * vars[0])

    for v in vars[1:]:
        term = v ** 2 - 10.0 * cos(2.0 * math.pi * v)
        result = result + term

    # Add 10n constant
    result = result + float(10 * n)
    return result


def _trace_griewank(*vars: TracedVar) -> TracedVar:
    """
    Griewank Function
    f(x) = Σxᵢ²/4000 - Πcos(xᵢ/√(i+1)) + 1

    Multimodal with regularly distributed local minima.
    """
    n = len(vars)

    # Sum term
    sum_term = vars[0] ** 2
    for v in vars[1:]:
        sum_term = sum_term + v ** 2
    sum_term = sum_term / 4000.0

    # Product term
    prod_term = cos(vars[0] / math.sqrt(1.0))
    for i, v in enumerate(vars[1:], start=2):
        prod_term = prod_term * cos(v / math.sqrt(float(i)))

    return sum_term - prod_term + 1.0


def _trace_sphere(*vars: TracedVar) -> TracedVar:
    """
    Sphere Function
    f(x) = Σxᵢ²

    Simple unimodal, separable, smooth.
    """
    result = vars[0] ** 2
    for v in vars[1:]:
        result = result + v ** 2
    return result


def _trace_ackley(*vars: TracedVar) -> TracedVar:
    """
    Ackley Function
    f(x) = -20exp(-0.2√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e

    Multimodal with nearly flat outer region.
    Note: This requires careful handling due to exp/sqrt composition.
    """
    n = len(vars)

    # Sum of squares
    sum_sq = vars[0] ** 2
    for v in vars[1:]:
        sum_sq = sum_sq + v ** 2

    # Sum of cosines
    sum_cos = cos(2.0 * math.pi * vars[0])
    for v in vars[1:]:
        sum_cos = sum_cos + cos(2.0 * math.pi * v)

    # -20*exp(-0.2*sqrt(sum_sq/n))
    term1 = -20.0 * exp(-0.2 * sqrt(sum_sq / float(n)))

    # -exp(sum_cos/n)
    term2 = -1.0 * exp(sum_cos / float(n))

    return term1 + term2 + 20.0 + math.e


def _trace_weierstrass(*vars: TracedVar) -> TracedVar:
    """
    Weierstrass Function
    f(x) = Σᵢ₌₀ⁿ⁻¹ [Σₖ₌₀ᵏᵐᵃˣ aᵏcos(2πbᵏ(xᵢ+0.5))] - n*Σₖ₌₀ᵏᵐᵃˣ aᵏcos(πbᵏ)

    Continuous everywhere but differentiable nowhere.
    a=0.5, b=3, kmax=20
    """
    n = len(vars)
    a, b = 0.5, 3.0
    kmax = 20

    # Precompute constant term: Σₖ aᵏcos(πbᵏ)
    const_term = sum(a**k * math.cos(math.pi * b**k) for k in range(kmax + 1))

    # Build expression for first variable
    result = sum(a**k * cos(2.0 * math.pi * b**k * (vars[0] + 0.5))
                 for k in range(kmax + 1))

    # Add remaining variables
    for v in vars[1:]:
        term = sum(a**k * cos(2.0 * math.pi * b**k * (v + 0.5))
                   for k in range(kmax + 1))
        result = result + term

    return result - float(n * const_term)


def build_bent_cigar(dim: int) -> CEC2020Problem:
    """Build Bent Cigar (F1) problem."""
    graph = ExpressionGraph.from_callable(_trace_bent_cigar, dim)
    return CEC2020Problem(
        name=f"CEC2020_F1_BentCigar_{dim}D",
        function_id=1,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -100.0),
        upper_bounds=np.full(dim, 100.0),
        known_optimum=100.0,  # f_bias
        difficulty="unimodal"
    )


def build_schwefel(dim: int) -> CEC2020Problem:
    """Build Schwefel's Problem 1.2 (F2) problem."""
    graph = ExpressionGraph.from_callable(_trace_schwefel_12, dim)
    return CEC2020Problem(
        name=f"CEC2020_F2_Schwefel_{dim}D",
        function_id=2,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -100.0),
        upper_bounds=np.full(dim, 100.0),
        known_optimum=1100.0,  # f_bias
        difficulty="multimodal"
    )


def build_rosenbrock(dim: int) -> CEC2020Problem:
    """Build Rosenbrock (base for F4) problem."""
    graph = ExpressionGraph.from_callable(_trace_rosenbrock, dim)
    return CEC2020Problem(
        name=f"CEC2020_Rosenbrock_{dim}D",
        function_id=4,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -100.0),
        upper_bounds=np.full(dim, 100.0),
        known_optimum=0.0,  # True optimum at (1,1,...,1)
        difficulty="multimodal"
    )


def build_rastrigin(dim: int) -> CEC2020Problem:
    """Build Rastrigin problem."""
    graph = ExpressionGraph.from_callable(_trace_rastrigin, dim)
    return CEC2020Problem(
        name=f"CEC2020_Rastrigin_{dim}D",
        function_id=3,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -5.12),
        upper_bounds=np.full(dim, 5.12),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def build_griewank(dim: int) -> CEC2020Problem:
    """Build Griewank problem."""
    graph = ExpressionGraph.from_callable(_trace_griewank, dim)
    return CEC2020Problem(
        name=f"CEC2020_Griewank_{dim}D",
        function_id=0,  # Base function
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -600.0),
        upper_bounds=np.full(dim, 600.0),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def build_sphere(dim: int) -> CEC2020Problem:
    """Build Sphere problem (simplest baseline)."""
    graph = ExpressionGraph.from_callable(_trace_sphere, dim)
    return CEC2020Problem(
        name=f"CEC2020_Sphere_{dim}D",
        function_id=0,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -100.0),
        upper_bounds=np.full(dim, 100.0),
        known_optimum=0.0,
        difficulty="unimodal"
    )


def build_ackley(dim: int) -> CEC2020Problem:
    """Build Ackley problem."""
    graph = ExpressionGraph.from_callable(_trace_ackley, dim)
    return CEC2020Problem(
        name=f"CEC2020_Ackley_{dim}D",
        function_id=0,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -32.768),
        upper_bounds=np.full(dim, 32.768),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def get_cec2020_problems(dimensions: List[int] = [2, 5, 10]) -> List[CEC2020Problem]:
    """
    Get a collection of CEC 2020 benchmark problems.

    These are the base functions used in CEC 2020, expressed as
    factorable expression graphs for mathematical certification.
    """
    problems = []

    for dim in dimensions:
        # Unimodal
        problems.append(build_sphere(dim))
        problems.append(build_bent_cigar(dim))

        # Multimodal
        problems.append(build_schwefel(dim))
        problems.append(build_rosenbrock(dim))
        problems.append(build_rastrigin(dim))
        problems.append(build_griewank(dim))
        problems.append(build_ackley(dim))

    return problems


def get_cec2020_core_problems() -> List[CEC2020Problem]:
    """
    Get the core CEC 2020 problems for certification testing.

    These are lower-dimensional versions suitable for mathematical
    certification with gap closure.
    """
    problems = []

    # 2D problems (tractable for rigorous certification)
    problems.append(build_sphere(2))
    problems.append(build_bent_cigar(2))
    problems.append(build_schwefel(2))
    problems.append(build_rosenbrock(2))
    problems.append(build_rastrigin(2))
    problems.append(build_griewank(2))

    # 3D problems
    problems.append(build_sphere(3))
    problems.append(build_rosenbrock(3))
    problems.append(build_rastrigin(3))

    # 5D problems (challenging but still tractable)
    problems.append(build_sphere(5))
    problems.append(build_bent_cigar(5))
    problems.append(build_rosenbrock(5))

    return problems


def get_cec2020_extended_problems() -> List[CEC2020Problem]:
    """
    Get extended CEC 2020 problems for comprehensive certification testing.

    Includes higher dimensions (up to 10D) and all function types.
    """
    problems = []

    # All functions at 2D (easiest)
    for builder in [build_sphere, build_bent_cigar, build_schwefel,
                    build_rosenbrock, build_rastrigin, build_griewank, build_ackley]:
        problems.append(builder(2))

    # All functions at 3D
    for builder in [build_sphere, build_bent_cigar, build_schwefel,
                    build_rosenbrock, build_rastrigin, build_griewank, build_ackley]:
        problems.append(builder(3))

    # All functions at 5D
    for builder in [build_sphere, build_bent_cigar, build_schwefel,
                    build_rosenbrock, build_rastrigin, build_griewank]:
        problems.append(builder(5))

    # Selected functions at 10D (most challenging)
    problems.append(build_sphere(10))
    problems.append(build_bent_cigar(10))
    problems.append(build_schwefel(10))
    problems.append(build_rosenbrock(10))
    problems.append(build_rastrigin(10))

    return problems


def get_cec2020_hard_problems() -> List[CEC2020Problem]:
    """
    Get HARD CEC 2020 problems for stress testing certification.

    Includes 15D and 20D problems - pushing the limits of mathematical certification.
    """
    problems = []

    # 10D - all functions
    for builder in [build_sphere, build_bent_cigar, build_schwefel,
                    build_rosenbrock, build_rastrigin, build_griewank]:
        problems.append(builder(10))

    # 15D - selected functions (very challenging)
    problems.append(build_sphere(15))
    problems.append(build_bent_cigar(15))
    problems.append(build_schwefel(15))
    problems.append(build_rosenbrock(15))
    problems.append(build_rastrigin(15))

    # 20D - pushing limits
    problems.append(build_sphere(20))
    problems.append(build_bent_cigar(20))
    problems.append(build_schwefel(20))

    return problems


if __name__ == "__main__":
    # Test the problem definitions
    print("CEC 2020 Problem Definitions Test")
    print("=" * 50)

    problems = get_cec2020_core_problems()

    for prob in problems:
        # Test at origin
        x_zero = np.zeros(prob.dimension)
        f_zero = prob.objective(x_zero)

        # Test at known optimum location (origin for base functions)
        x_opt = np.zeros(prob.dimension)
        if "Rosenbrock" in prob.name:
            x_opt = np.ones(prob.dimension)

        f_opt = prob.objective(x_opt)

        print(f"\n{prob.name}:")
        print(f"  Dimension: {prob.dimension}")
        print(f"  Bounds: [{prob.lower_bounds[0]}, {prob.upper_bounds[0]}]")
        print(f"  f(0): {f_zero:.6f}")
        print(f"  f(x_opt): {f_opt:.6f}")
        print(f"  Known optimum: {prob.known_optimum}")
        print(f"  Difficulty: {prob.difficulty}")
