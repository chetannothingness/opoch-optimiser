"""
Sphere Function - Simplest Convex Test Problem

f(x) = sum((x_i - center_i)^2)

Properties:
- Convex
- Unimodal
- Optimum at x = center with f* = 0
- Typical bounds: [-5, 5]^n
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from opoch_optimizer import (
    ExpressionGraph,
    OpType,
    ProblemContract,
    OPOCHKernel,
    OPOCHConfig,
    Verdict,
)


def build_sphere_graph(n: int, center: np.ndarray = None) -> ExpressionGraph:
    """
    Build expression graph for Sphere function.

    Args:
        n: Dimension
        center: Optional shift (default: origin)

    Returns:
        ExpressionGraph for f(x) = sum((x_i - center_i)^2)
    """
    if center is None:
        center = np.zeros(n)

    g = ExpressionGraph()
    terms = []

    for i in range(n):
        v = g.variable(i)
        c = g.constant(center[i])
        diff = g.binary(OpType.SUB, v, c)
        terms.append(g.unary(OpType.SQUARE, diff))

    result = terms[0]
    for t in terms[1:]:
        result = g.binary(OpType.ADD, result, t)

    g.set_output(result)
    return g


def sphere_numpy(x: np.ndarray, center: np.ndarray = None) -> float:
    """NumPy implementation for reference."""
    if center is None:
        center = np.zeros_like(x)
    return np.sum((x - center) ** 2)


def demo():
    """Demo: Solve shifted Sphere."""
    print("=" * 60)
    print("Sphere Function Demo")
    print("=" * 60)

    # 5D shifted sphere
    n = 5
    np.random.seed(42)
    center = np.array([1.5, -2.0, 0.5, 2.5, -1.0])

    print(f"\nProblem: min f(x) = sum((x_i - c_i)^2)")
    print(f"Dimension: {n}")
    print(f"Center: {center}")
    print(f"Expected optimum: x* = {center}, f* = 0")

    # Build problem
    graph = build_sphere_graph(n, center)
    bounds = [(-5.0, 5.0)] * n

    problem = ProblemContract(
        objective=graph,
        bounds=bounds,
        epsilon=1e-6,
        name="Sphere-5D"
    )
    problem._obj_graph = graph

    config = OPOCHConfig(
        epsilon=1e-6,
        max_time=60.0,
        max_nodes=10000
    )

    # Solve
    print("\nSolving...")
    kernel = OPOCHKernel(problem, config)
    verdict, result = kernel.solve()

    # Results
    print(f"\nVerdict: {verdict.name}")
    print(f"Solution: {kernel.best_solution}")
    print(f"Objective: {kernel.upper_bound:.6e}")
    print(f"Lower bound: {kernel.lower_bound:.6e}")
    print(f"Gap: {kernel.upper_bound - kernel.lower_bound:.6e}")
    print(f"Nodes explored: {kernel.nodes_explored}")

    if kernel.best_solution is not None:
        error = np.linalg.norm(kernel.best_solution - center)
        print(f"Error from true optimum: {error:.6e}")

    return verdict == Verdict.UNIQUE_OPT


if __name__ == "__main__":
    success = demo()
    print("\n" + ("PASSED" if success else "FAILED"))
