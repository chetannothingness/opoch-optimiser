"""
CEC 2022 Benchmark Problems as Expression Graphs

CEC 2022 contains 12 test functions for single-objective optimization.
We implement the BASE functions (without shifts/rotations) as expression
graphs for mathematical certification via gap closure.

Base Functions in CEC 2022:
- Zakharov (unimodal)
- Rosenbrock (multimodal)
- Schaffer's F6 (multimodal)
- Rastrigin (highly multimodal)
- Levy (multimodal)
- Plus Hybrid and Composition functions

For pure mathematical certification, we express these as factorable
expression graphs and apply the Δ* kernel.
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
class CEC2022Problem:
    """A CEC 2022 benchmark problem."""
    name: str
    function_id: int
    dimension: int
    objective: ExpressionGraph
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    known_optimum: float
    difficulty: str


def _trace_zakharov(*vars: TracedVar) -> TracedVar:
    """
    Zakharov Function (F1 base)
    f(x) = Σxᵢ² + (0.5 * Σi*xᵢ)² + (0.5 * Σi*xᵢ)⁴

    Unimodal, non-separable.
    """
    n = len(vars)

    # Σxᵢ²
    sum_sq = vars[0] ** 2
    for v in vars[1:]:
        sum_sq = sum_sq + v ** 2

    # Σi*xᵢ (1-indexed)
    sum_ix = 1.0 * vars[0]
    for i, v in enumerate(vars[1:], start=2):
        sum_ix = sum_ix + float(i) * v

    # (0.5 * Σi*xᵢ)² and (0.5 * Σi*xᵢ)⁴
    half_sum = 0.5 * sum_ix
    term2 = half_sum ** 2
    term4 = half_sum ** 2
    term4 = term4 ** 2

    return sum_sq + term2 + term4


def _trace_rosenbrock(*vars: TracedVar) -> TracedVar:
    """
    Rosenbrock Function (F2 base)
    f(x) = Σᵢ₌₀ⁿ⁻² [100(xᵢ₊₁ - xᵢ²)² + (xᵢ - 1)²]

    Multimodal with curved valley.
    """
    n = len(vars)
    if n < 2:
        return vars[0] ** 2

    result = 100.0 * (vars[1] - vars[0] ** 2) ** 2 + (vars[0] - 1.0) ** 2

    for i in range(1, n - 1):
        term = 100.0 * (vars[i + 1] - vars[i] ** 2) ** 2 + (vars[i] - 1.0) ** 2
        result = result + term

    return result


def _trace_schaffer_f6(*vars: TracedVar) -> TracedVar:
    """
    Expanded Schaffer's F6 Function (F3 base)
    g(x,y) = 0.5 + (sin²(√(x²+y²)) - 0.5) / (1 + 0.001(x²+y²))²
    f(x) = Σg(xᵢ, xᵢ₊₁) + g(xₙ, x₁)

    Highly multimodal.
    """
    n = len(vars)

    def schaffer_pair(x: TracedVar, y: TracedVar) -> TracedVar:
        sum_sq = x ** 2 + y ** 2
        sqrt_sum = sqrt(sum_sq)
        sin_term = sin(sqrt_sum)
        sin_sq = sin_term ** 2

        denom = 1.0 + 0.001 * sum_sq
        denom_sq = denom ** 2

        return 0.5 + (sin_sq - 0.5) / denom_sq

    # Sum over consecutive pairs
    result = schaffer_pair(vars[0], vars[1])
    for i in range(1, n - 1):
        result = result + schaffer_pair(vars[i], vars[i + 1])
    # Wrap-around term
    result = result + schaffer_pair(vars[n - 1], vars[0])

    return result


def _trace_rastrigin(*vars: TracedVar) -> TracedVar:
    """
    Rastrigin Function (F4 base)
    f(x) = 10n + Σᵢ₌₀ⁿ⁻¹ [xᵢ² - 10cos(2πxᵢ)]

    Highly multimodal with 10^n local minima.
    """
    n = len(vars)
    result = vars[0] ** 2 - 10.0 * cos(2.0 * math.pi * vars[0])

    for v in vars[1:]:
        term = v ** 2 - 10.0 * cos(2.0 * math.pi * v)
        result = result + term

    return result + float(10 * n)


def _trace_levy(*vars: TracedVar) -> TracedVar:
    """
    Levy Function (F5 base)
    wᵢ = 1 + (xᵢ - 1)/4
    f(x) = sin²(πw₁) + Σᵢ₌₁ⁿ⁻¹(wᵢ-1)²(1+10sin²(πwᵢ+1)) + (wₙ-1)²(1+sin²(2πwₙ))

    Multimodal.
    """
    n = len(vars)

    # w = 1 + (x - 1)/4
    ws = [1.0 + (v - 1.0) / 4.0 for v in vars]

    # sin²(πw₁)
    term1 = sin(math.pi * ws[0]) ** 2

    # Middle sum
    middle_sum = (ws[0] - 1.0) ** 2 * (1.0 + 10.0 * sin(math.pi * ws[0] + 1.0) ** 2)
    for i in range(1, n - 1):
        term = (ws[i] - 1.0) ** 2 * (1.0 + 10.0 * sin(math.pi * ws[i] + 1.0) ** 2)
        middle_sum = middle_sum + term

    # Last term
    last_term = (ws[n - 1] - 1.0) ** 2 * (1.0 + sin(2.0 * math.pi * ws[n - 1]) ** 2)

    return term1 + middle_sum + last_term


def _trace_sphere(*vars: TracedVar) -> TracedVar:
    """Sphere Function - simple baseline."""
    result = vars[0] ** 2
    for v in vars[1:]:
        result = result + v ** 2
    return result


def _trace_griewank(*vars: TracedVar) -> TracedVar:
    """
    Griewank Function
    f(x) = Σxᵢ²/4000 - Πcos(xᵢ/√(i+1)) + 1
    """
    n = len(vars)

    sum_term = vars[0] ** 2
    for v in vars[1:]:
        sum_term = sum_term + v ** 2
    sum_term = sum_term / 4000.0

    prod_term = cos(vars[0] / math.sqrt(1.0))
    for i, v in enumerate(vars[1:], start=2):
        prod_term = prod_term * cos(v / math.sqrt(float(i)))

    return sum_term - prod_term + 1.0


def _trace_ackley(*vars: TracedVar) -> TracedVar:
    """
    Ackley Function
    f(x) = -20exp(-0.2√(Σxᵢ²/n)) - exp(Σcos(2πxᵢ)/n) + 20 + e
    """
    n = len(vars)

    sum_sq = vars[0] ** 2
    for v in vars[1:]:
        sum_sq = sum_sq + v ** 2

    sum_cos = cos(2.0 * math.pi * vars[0])
    for v in vars[1:]:
        sum_cos = sum_cos + cos(2.0 * math.pi * v)

    term1 = -20.0 * exp(-0.2 * sqrt(sum_sq / float(n)))
    term2 = -1.0 * exp(sum_cos / float(n))

    return term1 + term2 + 20.0 + math.e


def build_zakharov(dim: int) -> CEC2022Problem:
    """Build Zakharov (F1 base) problem."""
    graph = ExpressionGraph.from_callable(_trace_zakharov, dim)
    return CEC2022Problem(
        name=f"CEC2022_Zakharov_{dim}D",
        function_id=1,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -5.0),
        upper_bounds=np.full(dim, 10.0),
        known_optimum=0.0,
        difficulty="unimodal"
    )


def build_rosenbrock_cec(dim: int) -> CEC2022Problem:
    """Build Rosenbrock (F2 base) problem."""
    graph = ExpressionGraph.from_callable(_trace_rosenbrock, dim)
    return CEC2022Problem(
        name=f"CEC2022_Rosenbrock_{dim}D",
        function_id=2,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -5.0),
        upper_bounds=np.full(dim, 10.0),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def build_schaffer_f6(dim: int) -> CEC2022Problem:
    """Build Expanded Schaffer's F6 (F3 base) problem."""
    graph = ExpressionGraph.from_callable(_trace_schaffer_f6, dim)
    return CEC2022Problem(
        name=f"CEC2022_SchafferF6_{dim}D",
        function_id=3,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -100.0),
        upper_bounds=np.full(dim, 100.0),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def build_rastrigin_cec(dim: int) -> CEC2022Problem:
    """Build Rastrigin (F4 base) problem."""
    graph = ExpressionGraph.from_callable(_trace_rastrigin, dim)
    return CEC2022Problem(
        name=f"CEC2022_Rastrigin_{dim}D",
        function_id=4,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -5.12),
        upper_bounds=np.full(dim, 5.12),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def build_levy(dim: int) -> CEC2022Problem:
    """Build Levy (F5 base) problem."""
    graph = ExpressionGraph.from_callable(_trace_levy, dim)
    return CEC2022Problem(
        name=f"CEC2022_Levy_{dim}D",
        function_id=5,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -10.0),
        upper_bounds=np.full(dim, 10.0),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def build_sphere_cec(dim: int) -> CEC2022Problem:
    """Build Sphere problem."""
    graph = ExpressionGraph.from_callable(_trace_sphere, dim)
    return CEC2022Problem(
        name=f"CEC2022_Sphere_{dim}D",
        function_id=0,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -100.0),
        upper_bounds=np.full(dim, 100.0),
        known_optimum=0.0,
        difficulty="unimodal"
    )


def build_griewank_cec(dim: int) -> CEC2022Problem:
    """Build Griewank problem."""
    graph = ExpressionGraph.from_callable(_trace_griewank, dim)
    return CEC2022Problem(
        name=f"CEC2022_Griewank_{dim}D",
        function_id=0,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -600.0),
        upper_bounds=np.full(dim, 600.0),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def build_ackley_cec(dim: int) -> CEC2022Problem:
    """Build Ackley problem."""
    graph = ExpressionGraph.from_callable(_trace_ackley, dim)
    return CEC2022Problem(
        name=f"CEC2022_Ackley_{dim}D",
        function_id=0,
        dimension=dim,
        objective=graph,
        lower_bounds=np.full(dim, -32.768),
        upper_bounds=np.full(dim, 32.768),
        known_optimum=0.0,
        difficulty="multimodal"
    )


def get_cec2022_core_problems() -> List[CEC2022Problem]:
    """
    Get core CEC 2022 problems for mathematical certification.

    Base functions from CEC 2022 without shifts/rotations.
    """
    problems = []

    # 2D problems
    problems.append(build_sphere_cec(2))
    problems.append(build_zakharov(2))
    problems.append(build_rosenbrock_cec(2))
    problems.append(build_rastrigin_cec(2))
    problems.append(build_levy(2))
    problems.append(build_griewank_cec(2))
    problems.append(build_ackley_cec(2))

    # 3D problems
    problems.append(build_sphere_cec(3))
    problems.append(build_zakharov(3))
    problems.append(build_rosenbrock_cec(3))
    problems.append(build_rastrigin_cec(3))
    problems.append(build_levy(3))

    # 5D problems
    problems.append(build_sphere_cec(5))
    problems.append(build_zakharov(5))
    problems.append(build_rosenbrock_cec(5))
    problems.append(build_rastrigin_cec(5))

    # 10D problems
    problems.append(build_sphere_cec(10))
    problems.append(build_zakharov(10))
    problems.append(build_rosenbrock_cec(10))
    problems.append(build_rastrigin_cec(10))

    return problems


def get_cec2022_extended_problems() -> List[CEC2022Problem]:
    """
    Get extended CEC 2022 problems including higher dimensions.
    """
    problems = get_cec2022_core_problems()

    # Add 15D and 20D
    problems.append(build_sphere_cec(15))
    problems.append(build_zakharov(15))
    problems.append(build_rosenbrock_cec(15))
    problems.append(build_rastrigin_cec(15))

    problems.append(build_sphere_cec(20))
    problems.append(build_zakharov(20))
    problems.append(build_rosenbrock_cec(20))

    return problems


if __name__ == "__main__":
    print("CEC 2022 Problem Definitions Test")
    print("=" * 50)

    problems = get_cec2022_core_problems()

    for prob in problems[:5]:  # Test first 5
        x_zero = np.zeros(prob.dimension)
        f_zero = prob.objective(x_zero)

        x_opt = np.zeros(prob.dimension)
        if "Rosenbrock" in prob.name:
            x_opt = np.ones(prob.dimension)
        elif "Levy" in prob.name:
            x_opt = np.ones(prob.dimension)

        f_opt = prob.objective(x_opt)

        print(f"\n{prob.name}:")
        print(f"  Dimension: {prob.dimension}")
        print(f"  f(0): {f_zero:.6f}")
        print(f"  f(x_opt): {f_opt:.6f}")
        print(f"  Known optimum: {prob.known_optimum}")
