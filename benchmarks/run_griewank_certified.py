"""
Griewank Function - Complete Mathematical Certification Benchmark

The Griewank function is a classic multimodal test function:
    f(x) = Σxᵢ²/4000 - Πcos(xᵢ/√i) + 1

Properties:
- Global minimum: f(0) = 0 at x* = (0, 0, ..., 0)
- Domain: [-600, 600]^n
- Highly multimodal with regularly distributed local minima
- The product of cosines creates exponentially many local minima
- As dimension increases, local minima become shallower

This benchmark certifies Griewank via PURE MATHEMATICAL gap closure:
    gap = UB - LB ≤ ε

NO SHORTCUTS. NO reference values. PURE MATHEMATICS.
"""

import sys
import time
import numpy as np
import math

sys.path.insert(0, '/Users/chetanchauhan/opoch-optimizer/src')

from opoch_optimizer.expr_graph import (
    ExpressionGraph, TracedVar, OpType,
    sqrt, exp, log, sin, cos
)
from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig
from opoch_optimizer.core.output_gate import Verdict
from opoch_optimizer.contract import ProblemContract


def _trace_griewank(*vars: TracedVar) -> TracedVar:
    """
    Griewank Function
    f(x) = Σxᵢ²/4000 - Πcos(xᵢ/√i) + 1

    Global minimum: f(0) = 0
    """
    n = len(vars)

    # Sum term: Σxᵢ²/4000
    sum_term = vars[0] ** 2
    for v in vars[1:]:
        sum_term = sum_term + v ** 2
    sum_term = sum_term / 4000.0

    # Product term: Πcos(xᵢ/√i)
    prod_term = cos(vars[0] / math.sqrt(1.0))
    for i, v in enumerate(vars[1:], start=2):
        prod_term = prod_term * cos(v / math.sqrt(float(i)))

    return sum_term - prod_term + 1.0


def build_griewank_problem(dim: int, bounds_scale: float = 600.0):
    """Build a Griewank problem for a given dimension."""
    graph = ExpressionGraph.from_callable(_trace_griewank, dim)
    bounds = [(-bounds_scale, bounds_scale)] * dim

    return ProblemContract(
        objective=graph,
        bounds=bounds,
        epsilon=1e-4,
        name=f"Griewank_{dim}D"
    )


def run_griewank_benchmark(
    dimensions: list = None,
    epsilon: float = 1e-4,
    max_iterations: int = 100000,
    time_limit: float = 300.0,
    bounds_scale: float = 600.0
):
    """
    Run complete Griewank benchmark across all dimensions.

    Args:
        dimensions: List of dimensions to test (default: 2 to 50)
        epsilon: Certification tolerance
        max_iterations: Max branch-and-bound iterations
        time_limit: Time limit per problem in seconds
        bounds_scale: Domain bounds [-scale, scale]
    """
    if dimensions is None:
        # Test from 2D to 50D
        dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10,
                      12, 15, 18, 20, 25, 30, 35, 40, 45, 50]

    print("\n" + "=" * 80)
    print("GRIEWANK FUNCTION - COMPLETE MATHEMATICAL CERTIFICATION")
    print("=" * 80)
    print(f"""
Function: f(x) = Σxᵢ²/4000 - Πcos(xᵢ/√i) + 1

Properties:
  - Global minimum: f(0) = 0 at x* = (0, 0, ..., 0)
  - Domain: [-{bounds_scale}, {bounds_scale}]^n
  - Highly multimodal (exponentially many local minima)
  - The product term creates shallow but numerous local minima

Method: Δ* Closure + Branch-and-Reduce (PURE MATHEMATICS)
Certification: gap = UB - LB ≤ ε = {epsilon}
""")
    print("=" * 80)
    print("NO SHORTCUTS: Pure mathematical gap closure.")
    print("=" * 80 + "\n")

    print(f"{'Dimension':<12} {'Verdict':<14} {'UB':>14} {'LB':>14} {'Gap':>14} {'Nodes':>10} {'Time':>10}")
    print("-" * 100)

    results = []
    total_certified = 0
    start_total = time.time()

    for dim in dimensions:
        try:
            contract = build_griewank_problem(dim, bounds_scale)

            config = OPOCHConfig(
                epsilon=epsilon,
                max_time=time_limit,
                max_nodes=max_iterations,
                log_frequency=1000000  # Quiet
            )

            start = time.time()
            kernel = OPOCHKernel(contract, config)
            verdict, result = kernel.solve()
            elapsed = time.time() - start

            ub = getattr(result, 'upper_bound', getattr(result, 'objective_value', float('inf')))
            lb = getattr(result, 'lower_bound', float('-inf'))
            gap = ub - lb
            nodes = getattr(result, 'nodes_explored', 0)

            certified = verdict == Verdict.UNIQUE_OPT and gap <= epsilon
            if certified:
                total_certified += 1
                verdict_str = "CERTIFIED"
            else:
                verdict_str = str(verdict).split('.')[-1]

            print(f"{dim:>4}D        {verdict_str:<14} {ub:>14.6e} {lb:>14.6e} {gap:>14.2e} {nodes:>10} {elapsed:>9.2f}s")

            results.append({
                'dimension': dim,
                'certified': certified,
                'verdict': verdict_str,
                'upper_bound': ub,
                'lower_bound': lb,
                'gap': gap,
                'nodes': nodes,
                'time': elapsed
            })

        except Exception as e:
            print(f"{dim:>4}D        {'ERROR':<14} {str(e)[:40]}")
            results.append({
                'dimension': dim,
                'certified': False,
                'error': str(e)
            })

    total_time = time.time() - start_total
    rate = total_certified / len(dimensions) * 100

    # Summary
    print("\n" + "=" * 80)
    print("GRIEWANK CERTIFICATION SUMMARY")
    print("=" * 80)

    # By dimension range
    ranges = [
        ("Low (2-5D)", [r for r in results if r['dimension'] <= 5]),
        ("Medium (6-10D)", [r for r in results if 6 <= r['dimension'] <= 10]),
        ("High (11-20D)", [r for r in results if 11 <= r['dimension'] <= 20]),
        ("Very High (21-50D)", [r for r in results if r['dimension'] > 20])
    ]

    print("\nBy Dimension Range:")
    for range_name, range_results in ranges:
        if range_results:
            certified_count = sum(1 for r in range_results if r.get('certified', False))
            total_count = len(range_results)
            pct = certified_count / total_count * 100 if total_count > 0 else 0
            print(f"  {range_name:<20}: {certified_count}/{total_count} ({pct:.0f}%)")

    # Statistics
    certified_results = [r for r in results if r.get('certified', False)]
    if certified_results:
        avg_gap = np.mean([r['gap'] for r in certified_results])
        max_gap = max(r['gap'] for r in certified_results)
        avg_time = np.mean([r['time'] for r in certified_results])
        max_dim = max(r['dimension'] for r in certified_results)

        print(f"\nStatistics (certified problems):")
        print(f"  Average gap: {avg_gap:.2e}")
        print(f"  Maximum gap: {max_gap:.2e}")
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  Maximum certified dimension: {max_dim}D")

    print(f"\n{'=' * 80}")
    print(f"GRIEWANK CERTIFICATION RATE: {total_certified}/{len(dimensions)} = {rate:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 80)

    if rate == 100:
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗██████╗ ██╗███████╗██╗    ██╗ █████╗ ███╗   ██╗██╗  ██╗           ║
║   ██╔════╝██╔══██╗██║██╔════╝██║    ██║██╔══██╗████╗  ██║██║ ██╔╝           ║
║   ██║  ███╝██████╔╝██║█████╗  ██║ █╗ ██║███████║██╔██╗ ██║█████╔╝            ║
║   ██║   ██║██╔══██╗██║██╔══╝  ██║███╗██║██╔══██║██║╚██╗██║██╔═██╗            ║
║   ╚██████╔╝██║  ██║██║███████╗╚███╔███╔╝██║  ██║██║ ╚████║██║  ██╗           ║
║    ╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝           ║
║                                                                              ║
║                    100% MATHEMATICALLY CERTIFIED                             ║
║                                                                              ║
║         All dimensions certified via gap closure (UB - LB ≤ ε)              ║
║         NO shortcuts, NO reference values, PURE MATHEMATICS                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    return results


def run_stress_test():
    """Run stress test on extreme dimensions."""
    print("\n" + "=" * 80)
    print("GRIEWANK STRESS TEST - EXTREME DIMENSIONS")
    print("=" * 80)

    # Push to 100D
    dimensions = [50, 60, 70, 80, 90, 100]

    return run_griewank_benchmark(
        dimensions=dimensions,
        epsilon=1e-4,
        max_iterations=200000,
        time_limit=600.0
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Griewank Function - Complete Mathematical Certification"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test (2D-10D only)"
    )
    parser.add_argument(
        "--standard", action="store_true",
        help="Standard test (2D-30D)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full test (2D-50D)"
    )
    parser.add_argument(
        "--stress", action="store_true",
        help="Stress test (50D-100D)"
    )
    parser.add_argument(
        "--extreme", action="store_true",
        help="Extreme test (2D-100D)"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-4,
        help="Certification tolerance"
    )
    parser.add_argument(
        "--max-iter", type=int, default=100000,
        help="Maximum iterations per problem"
    )
    parser.add_argument(
        "--time-limit", type=float, default=300.0,
        help="Time limit per problem in seconds"
    )

    args = parser.parse_args()

    if args.quick:
        dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    elif args.standard:
        dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30]
    elif args.full:
        dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
    elif args.stress:
        run_stress_test()
        return
    elif args.extreme:
        dimensions = list(range(2, 101))  # 2D to 100D
    else:
        # Default: comprehensive test
        dimensions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]

    run_griewank_benchmark(
        dimensions=dimensions,
        epsilon=args.epsilon,
        max_iterations=args.max_iter,
        time_limit=args.time_limit
    )


if __name__ == "__main__":
    main()
