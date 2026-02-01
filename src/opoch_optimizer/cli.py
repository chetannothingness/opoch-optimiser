"""
OPOCH Optimizer Command-Line Interface

Provides a command-line interface for running optimization problems.
"""

import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional

from . import (
    ExpressionGraph,
    OpType,
    ProblemContract,
    OPOCHKernel,
    OPOCHConfig,
    Verdict,
)
from .primal import PhaseProbe


def build_sphere(n: int, shift: np.ndarray) -> ExpressionGraph:
    """Build Sphere function graph."""
    g = ExpressionGraph()
    terms = []
    for i in range(n):
        v = g.variable(i)
        s = g.constant(shift[i])
        diff = g.binary(OpType.SUB, v, s)
        terms.append(g.unary(OpType.SQUARE, diff))
    result = terms[0]
    for t in terms[1:]:
        result = g.binary(OpType.ADD, result, t)
    g.set_output(result)
    return g


def build_rastrigin(n: int, shift: np.ndarray) -> ExpressionGraph:
    """Build Rastrigin function graph."""
    g = ExpressionGraph()
    ten = g.constant(10.0)
    two_pi = g.constant(2.0 * np.pi)
    base = g.constant(10.0 * n)
    result = base
    for i in range(n):
        v = g.variable(i)
        s = g.constant(shift[i])
        diff = g.binary(OpType.SUB, v, s)
        diff_sq = g.unary(OpType.SQUARE, diff)
        angle = g.binary(OpType.MUL, two_pi, diff)
        cos_term = g.unary(OpType.COS, angle)
        scaled_cos = g.binary(OpType.MUL, ten, cos_term)
        term = g.binary(OpType.SUB, diff_sq, scaled_cos)
        result = g.binary(OpType.ADD, result, term)
    g.set_output(result)
    return g


FUNCTIONS = {
    'sphere': build_sphere,
    'rastrigin': build_rastrigin,
}


def cmd_solve(args):
    """Solve an optimization problem."""
    print("=" * 60)
    print(f"OPOCH Optimizer")
    print("=" * 60)

    # Set seed
    np.random.seed(args.seed)

    # Generate shift
    if args.shift:
        shift = np.array(args.shift)
        if len(shift) != args.dim:
            print(f"Error: shift length ({len(shift)}) must match dimension ({args.dim})")
            return 1
    else:
        shift = np.random.uniform(-2, 2, args.dim)

    # Build problem
    if args.function not in FUNCTIONS:
        print(f"Error: Unknown function '{args.function}'")
        print(f"Available: {list(FUNCTIONS.keys())}")
        return 1

    print(f"\nFunction: {args.function}")
    print(f"Dimension: {args.dim}")
    print(f"Shift: {shift[:min(5, len(shift))]}{'...' if len(shift) > 5 else ''}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Max time: {args.max_time}s")

    graph = FUNCTIONS[args.function](args.dim, shift)
    bounds = [(args.lower, args.upper)] * args.dim

    problem = ProblemContract(
        objective=graph,
        bounds=bounds,
        epsilon=args.epsilon,
        name=f"{args.function}-{args.dim}D"
    )
    problem._obj_graph = graph

    config = OPOCHConfig(
        epsilon=args.epsilon,
        max_time=args.max_time,
        max_nodes=args.max_nodes
    )

    # Solve
    print("\nSolving...")
    start = time.time()
    kernel = OPOCHKernel(problem, config)
    verdict, result = kernel.solve()
    elapsed = time.time() - start

    # Output
    print("\n" + "-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"Verdict: {verdict.name}")
    print(f"Time: {elapsed:.3f}s")
    print(f"Nodes explored: {kernel.nodes_explored}")
    print(f"Upper bound: {kernel.upper_bound:.6e}")
    print(f"Lower bound: {kernel.lower_bound:.6e}")
    print(f"Gap: {kernel.upper_bound - kernel.lower_bound:.6e}")

    if kernel.best_solution is not None:
        print(f"Solution: {kernel.best_solution[:min(5, len(kernel.best_solution))]}{'...' if len(kernel.best_solution) > 5 else ''}")
        error = np.linalg.norm(kernel.best_solution - shift)
        print(f"Error from true optimum: {error:.6e}")

    # Save output if requested
    if args.output:
        output_data = {
            'function': args.function,
            'dimension': args.dim,
            'shift': shift.tolist(),
            'verdict': verdict.name,
            'upper_bound': kernel.upper_bound,
            'lower_bound': kernel.lower_bound,
            'gap': kernel.upper_bound - kernel.lower_bound,
            'solution': kernel.best_solution.tolist() if kernel.best_solution is not None else None,
            'nodes_explored': kernel.nodes_explored,
            'time': elapsed
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0 if verdict == Verdict.UNIQUE_OPT else 1


def cmd_benchmark(args):
    """Run a quick benchmark."""
    print("=" * 60)
    print("OPOCH Quick Benchmark")
    print("=" * 60)

    results = []

    for func_name in ['sphere', 'rastrigin']:
        for dim in [2, 5, 10]:
            np.random.seed(42 + dim)
            shift = np.random.uniform(-2, 2, dim)
            graph = FUNCTIONS[func_name](dim, shift)
            bounds = [(-5.12, 5.12)] * dim

            problem = ProblemContract(
                objective=graph,
                bounds=bounds,
                epsilon=1e-6,
                name=f"{func_name}-{dim}D"
            )
            problem._obj_graph = graph

            config = OPOCHConfig(
                epsilon=1e-6,
                max_time=60.0,
                max_nodes=10000
            )

            start = time.time()
            kernel = OPOCHKernel(problem, config)
            verdict, _ = kernel.solve()
            elapsed = time.time() - start

            status = "CERT" if verdict == Verdict.UNIQUE_OPT else "FAIL"
            print(f"{func_name:12} {dim}D: [{status}] {elapsed:6.3f}s, "
                  f"gap={kernel.upper_bound - kernel.lower_bound:.2e}")

            results.append({
                'function': func_name,
                'dimension': dim,
                'certified': verdict == Verdict.UNIQUE_OPT,
                'time': elapsed,
                'gap': kernel.upper_bound - kernel.lower_bound
            })

    # Summary
    certified = sum(1 for r in results if r['certified'])
    print(f"\nTotal: {certified}/{len(results)} certified")

    return 0


def cmd_version(args):
    """Print version information."""
    from . import __version__
    print(f"opoch-optimizer {__version__}")
    print("Mathematical certification for global optimization")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='opoch',
        description='OPOCH Optimizer - Certified Global Optimization'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve an optimization problem')
    solve_parser.add_argument('function', choices=list(FUNCTIONS.keys()),
                              help='Function to optimize')
    solve_parser.add_argument('--dim', '-d', type=int, default=5,
                              help='Dimension (default: 5)')
    solve_parser.add_argument('--shift', type=float, nargs='+',
                              help='Shift vector (optional)')
    solve_parser.add_argument('--seed', type=int, default=42,
                              help='Random seed (default: 42)')
    solve_parser.add_argument('--epsilon', '-e', type=float, default=1e-6,
                              help='Precision (default: 1e-6)')
    solve_parser.add_argument('--max-time', '-t', type=float, default=60.0,
                              help='Max time in seconds (default: 60)')
    solve_parser.add_argument('--max-nodes', '-n', type=int, default=100000,
                              help='Max nodes (default: 100000)')
    solve_parser.add_argument('--lower', type=float, default=-5.0,
                              help='Lower bound (default: -5)')
    solve_parser.add_argument('--upper', type=float, default=5.0,
                              help='Upper bound (default: 5)')
    solve_parser.add_argument('--output', '-o', type=str,
                              help='Output JSON file')
    solve_parser.set_defaults(func=cmd_solve)

    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run quick benchmark')
    bench_parser.set_defaults(func=cmd_benchmark)

    # Version command
    ver_parser = subparsers.add_parser('version', help='Print version')
    ver_parser.set_defaults(func=cmd_version)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
