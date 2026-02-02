"""
CEC 2020 Mathematical Certification Benchmark Runner

This runner applies the OPOCH mathematical kernel to CEC 2020 benchmark
functions with REAL certification via gap closure (UB - LB ≤ ε).

NO SHORTCUTS:
- NO generator inversion (we treat functions as unknown)
- NO reference to known optima during optimization
- ONLY mathematical gap closure for certification

The certification contract:
- UNIQUE-OPT: gap = UB - LB ≤ ε (mathematically proven)
- Ω-GAP: budget exhausted, returns exact remaining gap
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Any, Tuple

sys.path.insert(0, '/Users/chetanchauhan/opoch-optimizer/src')

from cec2020_problems import (
    CEC2020Problem, get_cec2020_core_problems, get_cec2020_extended_problems,
    get_cec2020_hard_problems, build_sphere, build_bent_cigar, build_schwefel,
    build_rosenbrock, build_rastrigin, build_griewank, build_ackley
)

from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig
from opoch_optimizer.core.output_gate import Verdict, OptimalityResult, UnsatResult, OmegaGapResult
from opoch_optimizer.bounds.interval import interval_evaluate
from opoch_optimizer.contract import ProblemContract


def run_certified_benchmark(
    problems: List[CEC2020Problem] = None,
    epsilon: float = 1e-4,
    max_iterations: int = 10000,
    time_limit: float = 60.0,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run CEC 2020 benchmark with REAL mathematical certification.

    Certification is via gap closure:
        gap = UB - LB ≤ ε

    Where:
        - UB: Best feasible solution found
        - LB: Proven lower bound via interval arithmetic + McCormick
        - ε: Certification tolerance

    Args:
        problems: List of CEC2020Problem to solve (default: core problems)
        epsilon: Certification tolerance
        max_iterations: Maximum branch-and-bound iterations
        time_limit: Time limit per problem in seconds
        verbose: Print detailed output

    Returns:
        Summary dict with certification results
    """
    if problems is None:
        problems = get_cec2020_core_problems()

    results = []
    total_certified = 0
    total_problems = len(problems)

    print("\n" + "=" * 70)
    print("CEC 2020 MATHEMATICAL CERTIFICATION BENCHMARK")
    print("=" * 70)
    print(f"Method: Δ* Closure + Branch-and-Reduce (Pure Mathematics)")
    print(f"Certification: gap = UB - LB ≤ ε = {epsilon}")
    print(f"Problems: {total_problems}")
    print(f"Max iterations: {max_iterations}")
    print(f"Time limit: {time_limit}s per problem")
    print("=" * 70)
    print("\nNO SHORTCUTS: No generator inversion, no reference optima.")
    print("Certification comes ONLY from mathematical gap closure.")
    print("=" * 70 + "\n")

    print(f"{'Problem':<35} {'Dim':>4} {'Verdict':<12} {'UB':>12} {'LB':>12} {'Gap':>12} {'Time':>8}")
    print("-" * 100)

    start_total = time.time()

    for prob in problems:
        start_time = time.time()

        try:
            # Create ProblemContract for this problem
            bounds_list = list(zip(prob.lower_bounds.tolist(), prob.upper_bounds.tolist()))
            contract = ProblemContract(
                objective=prob.objective,
                bounds=bounds_list,
                epsilon=epsilon,
                name=prob.name
            )

            # Create OPOCH config and kernel
            config = OPOCHConfig(
                epsilon=epsilon,
                max_time=time_limit,
                max_nodes=max_iterations,
                log_frequency=1000000  # Quiet
            )

            kernel = OPOCHKernel(contract, config)

            # Solve with pure mathematics
            verdict, result = kernel.solve()

            elapsed = time.time() - start_time

            # Extract certification info based on result type
            ub = getattr(result, 'upper_bound', getattr(result, 'objective_value', float('inf')))
            lb = getattr(result, 'lower_bound', float('-inf'))
            gap = ub - lb if lb > float('-inf') else float('inf')

            # Check if certified
            certified = verdict == Verdict.UNIQUE_OPT and gap <= epsilon

            if certified:
                total_certified += 1
                verdict_str = "UNIQUE-OPT"
            elif verdict == Verdict.UNSAT:
                verdict_str = "UNSAT"
            else:
                verdict_str = "Ω-GAP"

            # Extract solution
            x_opt = getattr(result, 'x_optimal', getattr(result, 'x_best', None))

            results.append({
                'name': prob.name,
                'dimension': prob.dimension,
                'verdict': verdict_str,
                'upper_bound': ub,
                'lower_bound': lb,
                'gap': gap,
                'certified': certified,
                'solution': x_opt.tolist() if x_opt is not None else None,
                'iterations': getattr(result, 'nodes_explored', 0),
                'time': elapsed,
                'difficulty': prob.difficulty
            })

            print(f"{prob.name:<35} {prob.dimension:>4} {verdict_str:<12} {ub:>12.4f} {lb:>12.4f} {gap:>12.2e} {elapsed:>7.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            results.append({
                'name': prob.name,
                'dimension': prob.dimension,
                'verdict': 'ERROR',
                'error': str(e),
                'certified': False,
                'time': elapsed
            })
            print(f"{prob.name:<35} {prob.dimension:>4} {'ERROR':<12} {'-':>12} {'-':>12} {'-':>12} {elapsed:>7.2f}s")
            if verbose:
                import traceback
                traceback.print_exc()

    total_time = time.time() - start_total

    # Summary
    print("\n" + "=" * 70)
    print("CERTIFICATION SUMMARY")
    print("=" * 70)

    # By difficulty
    by_difficulty = {}
    for r in results:
        diff = r.get('difficulty', 'unknown')
        if diff not in by_difficulty:
            by_difficulty[diff] = {'total': 0, 'certified': 0}
        by_difficulty[diff]['total'] += 1
        if r.get('certified', False):
            by_difficulty[diff]['certified'] += 1

    print("\nBy Difficulty:")
    for diff, stats in by_difficulty.items():
        rate = stats['certified'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {diff:<12}: {stats['certified']}/{stats['total']} ({rate:.0f}%)")

    # By dimension
    by_dim = {}
    for r in results:
        dim = r.get('dimension', 0)
        if dim not in by_dim:
            by_dim[dim] = {'total': 0, 'certified': 0}
        by_dim[dim]['total'] += 1
        if r.get('certified', False):
            by_dim[dim]['certified'] += 1

    print("\nBy Dimension:")
    for dim in sorted(by_dim.keys()):
        stats = by_dim[dim]
        rate = stats['certified'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {dim}D: {stats['certified']}/{stats['total']} ({rate:.0f}%)")

    certification_rate = total_certified / total_problems * 100 if total_problems > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"TOTAL CERTIFICATION RATE: {total_certified}/{total_problems} = {certification_rate:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per problem: {total_time/total_problems:.2f}s")
    print("=" * 70)

    if certification_rate == 100:
        print("\n*** 100% MATHEMATICAL CERTIFICATION ACHIEVED ***")
        print("All problems certified via gap closure (UB - LB ≤ ε)")
    else:
        print(f"\n{total_problems - total_certified} problems need more iterations or tighter bounds")

    summary = {
        'method': 'CEC2020 Mathematical Certification',
        'epsilon': epsilon,
        'total_problems': total_problems,
        'certified': total_certified,
        'certification_rate': certification_rate,
        'by_difficulty': by_difficulty,
        'by_dimension': by_dim,
        'total_time': total_time,
        'results': results
    }

    return summary


def run_quick_test():
    """Run a quick test on simpler problems."""
    print("Running quick certification test on simpler problems...")

    problems = [
        build_sphere(2),
        build_sphere(3),
        build_bent_cigar(2),
        build_rosenbrock(2),
        build_rastrigin(2),
    ]

    return run_certified_benchmark(
        problems=problems,
        epsilon=1e-4,
        max_iterations=5000,
        time_limit=30.0,
        verbose=True
    )


def run_extended_benchmark():
    """Run extended benchmark with more problems and dimensions."""
    print("Running extended CEC 2020 benchmark (including higher dimensions)...")

    problems = get_cec2020_extended_problems()

    return run_certified_benchmark(
        problems=problems,
        epsilon=1e-4,
        max_iterations=50000,
        time_limit=120.0,
        verbose=True
    )


def run_hard_benchmark():
    """Run HARD benchmark with 10D-20D problems."""
    print("Running HARD CEC 2020 benchmark (10D-20D - stress test)...")

    problems = get_cec2020_hard_problems()

    return run_certified_benchmark(
        problems=problems,
        epsilon=1e-4,
        max_iterations=100000,
        time_limit=300.0,
        verbose=True
    )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CEC 2020 Mathematical Certification Benchmark"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick test on simpler problems"
    )
    parser.add_argument(
        "--extended", action="store_true",
        help="Run extended benchmark with higher dimensions"
    )
    parser.add_argument(
        "--hard", action="store_true",
        help="Run HARD benchmark with 10D-20D problems"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-4,
        help="Certification tolerance"
    )
    parser.add_argument(
        "--max-iter", type=int, default=10000,
        help="Maximum iterations per problem"
    )
    parser.add_argument(
        "--time-limit", type=float, default=60.0,
        help="Time limit per problem in seconds"
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    elif args.extended:
        run_extended_benchmark()
    elif args.hard:
        run_hard_benchmark()
    else:
        run_certified_benchmark(
            epsilon=args.epsilon,
            max_iterations=args.max_iter,
            time_limit=args.time_limit
        )


if __name__ == "__main__":
    main()
