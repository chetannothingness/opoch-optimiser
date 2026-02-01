"""
Rosenbrock Function - Classic Valley Test Problem

f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)

Properties:
- Non-convex
- Long narrow curved valley
- Global optimum at x = (1, 1, ..., 1) with f* = 0
- Typical bounds: [-5, 10]^n
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


def build_rosenbrock_graph(n: int, shift: np.ndarray = None) -> ExpressionGraph:
    """
    Build expression graph for Rosenbrock function.

    Args:
        n: Dimension (must be >= 2)
        shift: Optional shift (default: optimum at all 1s)

    Returns:
        ExpressionGraph for Rosenbrock
    """
    if shift is None:
        shift = np.zeros(n)

    g = ExpressionGraph()
    hundred = g.constant(100.0)
    one = g.constant(1.0)

    terms = []
    for i in range(n - 1):
        vi = g.variable(i)
        vi1 = g.variable(i + 1)
        si = g.constant(shift[i])
        si1 = g.constant(shift[i + 1])

        # (x_i - s_i)
        diff_i = g.binary(OpType.SUB, vi, si)
        # (x_{i+1} - s_{i+1})
        diff_i1 = g.binary(OpType.SUB, vi1, si1)

        # (x_i - s_i)^2
        diff_i_sq = g.unary(OpType.SQUARE, diff_i)
        # x_{i+1} - s_{i+1} - (x_i - s_i)^2
        inner = g.binary(OpType.SUB, diff_i1, diff_i_sq)
        # 100 * (...)^2
        term1 = g.binary(OpType.MUL, hundred, g.unary(OpType.SQUARE, inner))

        # 1 - (x_i - s_i)
        one_minus = g.binary(OpType.SUB, one, diff_i)
        # (1 - (x_i - s_i))^2
        term2 = g.unary(OpType.SQUARE, one_minus)

        terms.append(g.binary(OpType.ADD, term1, term2))

    if not terms:
        g.set_output(g.constant(0.0))
    else:
        result = terms[0]
        for t in terms[1:]:
            result = g.binary(OpType.ADD, result, t)
        g.set_output(result)

    return g


def rosenbrock_numpy(x: np.ndarray) -> float:
    """NumPy implementation for reference."""
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def demo():
    """Demo: Solve Rosenbrock."""
    print("=" * 60)
    print("Rosenbrock Function Demo")
    print("=" * 60)

    # 5D Rosenbrock
    n = 5

    print(f"\nProblem: Rosenbrock (banana valley)")
    print(f"Dimension: {n}")
    print(f"Expected optimum: x* = (1, 1, ..., 1), f* = 0")

    # Build problem
    graph = build_rosenbrock_graph(n)
    bounds = [(-5.0, 10.0)] * n

    problem = ProblemContract(
        objective=graph,
        bounds=bounds,
        epsilon=1e-4,  # Rosenbrock is harder
        name="Rosenbrock-5D"
    )
    problem._obj_graph = graph

    config = OPOCHConfig(
        epsilon=1e-4,
        max_time=120.0,
        max_nodes=50000
    )

    # Solve
    print("\nSolving...")
    kernel = OPOCHKernel(problem, config)
    verdict, result = kernel.solve()

    # Results
    true_opt = np.ones(n)

    print(f"\nVerdict: {verdict.name}")
    print(f"Solution: {kernel.best_solution}")
    print(f"Objective: {kernel.upper_bound:.6e}")
    print(f"Lower bound: {kernel.lower_bound:.6e}")
    print(f"Gap: {kernel.upper_bound - kernel.lower_bound:.6e}")
    print(f"Nodes explored: {kernel.nodes_explored}")

    if kernel.best_solution is not None:
        error = np.linalg.norm(kernel.best_solution - true_opt)
        print(f"Error from true optimum: {error:.6e}")

    return verdict == Verdict.UNIQUE_OPT


def valley_visualization():
    """Show the valley structure."""
    print("\n" + "=" * 60)
    print("Rosenbrock Valley Visualization")
    print("=" * 60)

    print("\n2D Rosenbrock objective values:\n")

    points = [
        ([1.0, 1.0], "Global optimum"),
        ([0.0, 0.0], "In the valley"),
        ([-1.0, 1.0], "Near valley bottom"),
        ([2.0, 4.0], "On the valley"),
        ([0.0, 1.0], "Near optimum"),
    ]

    for pt, desc in points:
        x = np.array(pt)
        f = rosenbrock_numpy(x)
        print(f"  x = {pt}, f(x) = {f:.4f}  ({desc})")

    print("\n-> The valley floor follows x_2 = x_1^2, x_3 = x_2^2, etc.")
    print("-> Most optimizers find the valley but struggle to reach (1,1,...,1).")


if __name__ == "__main__":
    valley_visualization()
    print()
    success = demo()
    print("\n" + ("PASSED" if success else "NEEDS MORE ITERATIONS"))
