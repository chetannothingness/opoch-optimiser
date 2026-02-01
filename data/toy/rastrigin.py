"""
Rastrigin Function - Classic Multimodal Test Problem

f(x) = 10n + sum((x_i - s_i)^2 - 10*cos(2*pi*(x_i - s_i)))

Properties:
- Highly multimodal: ~10^n local minima
- Global optimum at x = shift with f* = 0
- Typical bounds: [-5.12, 5.12]^n
- PhaseProbe can identify shift using DFT
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
from opoch_optimizer.primal import PhaseProbe


def build_rastrigin_graph(n: int, shift: np.ndarray = None) -> ExpressionGraph:
    """
    Build expression graph for Rastrigin function.

    Args:
        n: Dimension
        shift: Optional shift (default: origin)

    Returns:
        ExpressionGraph for shifted Rastrigin
    """
    if shift is None:
        shift = np.zeros(n)

    g = ExpressionGraph()
    ten = g.constant(10.0)
    two_pi = g.constant(2.0 * np.pi)
    base = g.constant(10.0 * n)

    result = base
    for i in range(n):
        v = g.variable(i)
        s = g.constant(shift[i])
        diff = g.binary(OpType.SUB, v, s)

        # (x_i - s_i)^2
        diff_sq = g.unary(OpType.SQUARE, diff)
        # 2*pi*(x_i - s_i)
        angle = g.binary(OpType.MUL, two_pi, diff)
        # cos(...)
        cos_term = g.unary(OpType.COS, angle)
        # 10*cos(...)
        scaled_cos = g.binary(OpType.MUL, ten, cos_term)
        # (x_i-s_i)^2 - 10*cos(...)
        term = g.binary(OpType.SUB, diff_sq, scaled_cos)
        result = g.binary(OpType.ADD, result, term)

    g.set_output(result)
    return g


def rastrigin_numpy(x: np.ndarray, shift: np.ndarray = None) -> float:
    """NumPy implementation for reference."""
    if shift is None:
        shift = np.zeros_like(x)
    n = len(x)
    diff = x - shift
    return 10 * n + np.sum(diff ** 2 - 10 * np.cos(2 * np.pi * diff))


def demo_with_phaseprobe():
    """Demo: Solve shifted Rastrigin using PhaseProbe."""
    print("=" * 60)
    print("Rastrigin Function Demo with PhaseProbe")
    print("=" * 60)

    # 10D shifted Rastrigin
    n = 10
    np.random.seed(42)
    shift = np.random.uniform(-2, 2, n)

    print(f"\nProblem: Shifted Rastrigin (highly multimodal)")
    print(f"Dimension: {n}")
    print(f"Shift: {shift[:3]}... (showing first 3)")
    print(f"Expected optimum: x* = shift, f* = 0")

    # Build problem
    graph = build_rastrigin_graph(n, shift)
    bounds = [(-5.12, 5.12)] * n

    # First, use PhaseProbe to identify the shift
    print("\n--- Phase 1: PhaseProbe Identification ---")

    def objective(x):
        return graph.evaluate(x)

    probe = PhaseProbe(objective, n, bounds)
    probe_result = probe.identify_and_refine(M=32)

    print(f"Identified shift: {probe_result.candidate_x[:3]}...")
    print(f"True shift:       {shift[:3]}...")
    print(f"Identification error: {np.linalg.norm(probe_result.candidate_x - shift):.6e}")
    print(f"f(candidate): {probe_result.candidate_f:.6e}")
    print(f"Evals used: {probe_result.total_evals}")

    # Now solve with OPOCH
    print("\n--- Phase 2: OPOCH Certification ---")

    problem = ProblemContract(
        objective=graph,
        bounds=bounds,
        epsilon=1e-6,
        name="Rastrigin-10D"
    )
    problem._obj_graph = graph

    config = OPOCHConfig(
        epsilon=1e-6,
        max_time=120.0,
        max_nodes=50000
    )

    kernel = OPOCHKernel(problem, config)
    verdict, result = kernel.solve()

    print(f"\nVerdict: {verdict.name}")
    print(f"Solution: {kernel.best_solution[:3] if kernel.best_solution is not None else 'None'}...")
    print(f"Objective: {kernel.upper_bound:.6e}")
    print(f"Lower bound: {kernel.lower_bound:.6e}")
    print(f"Gap: {kernel.upper_bound - kernel.lower_bound:.6e}")
    print(f"Nodes explored: {kernel.nodes_explored}")

    if kernel.best_solution is not None:
        error = np.linalg.norm(kernel.best_solution - shift)
        print(f"Error from true optimum: {error:.6e}")

    return verdict == Verdict.UNIQUE_OPT


def demo_local_minima():
    """Demonstrate the challenge: many local minima."""
    print("\n" + "=" * 60)
    print("Local Minima Demonstration")
    print("=" * 60)

    n = 2
    shift = np.array([0.5, 0.5])

    print(f"\n2D Rastrigin with shift = {shift}")
    print("Showing objective values at various points:\n")

    # Sample points
    points = [
        ([0.5, 0.5], "True optimum"),
        ([0.0, 0.0], "Origin (local min)"),
        ([1.0, 1.0], "Local minimum"),
        ([0.5, 1.5], "Local minimum"),
        ([-0.5, 0.5], "Local minimum"),
    ]

    for pt, desc in points:
        x = np.array(pt)
        f = rastrigin_numpy(x, shift)
        print(f"  x = {pt}, f(x) = {f:.4f}  ({desc})")

    print("\n-> Without knowing the shift, search gets trapped in local minima.")
    print("-> PhaseProbe identifies the shift directly using DFT.")


if __name__ == "__main__":
    demo_local_minima()
    print()
    success = demo_with_phaseprobe()
    print("\n" + ("PASSED" if success else "NEEDS MORE ITERATIONS"))
