"""
Hard Polynomial Benchmark Runner - Testing Separable Bounds

This benchmark tests the FORCED Δ* constructor for hard polynomials:
- Separability detection
- Exact 1D polynomial minimization
- Gap closure without branching

The KEY insight: Styblinski-Tang is NOT hard because of local minima.
It's hard because interval arithmetic fails on x⁴ - 16x² + 5x.

The SOLUTION: Exploit separability. For f(x) = Σ s(xᵢ):
    LB(X) = Σ min_{xᵢ ∈ Xᵢ} s(xᵢ)

Each 1D minimum is computed EXACTLY by solving the cubic s'(x) = 0.

EXPECTED RESULT: 100% certification with minimal branching (often 0 nodes).
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/Users/chetanchauhan/opoch-optimizer/src')

from hard_polynomials import (
    HardPolynomialProblem,
    get_hard_polynomial_problems,
    get_styblinski_tang_only,
    build_styblinski_tang
)

from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig
from opoch_optimizer.core.output_gate import Verdict
from opoch_optimizer.contract import ProblemContract


def run_hard_polynomial_benchmark(
    problems=None,
    epsilon: float = 1e-4,
    max_iterations: int = 10000,
    time_limit: float = 60.0,
    verbose: bool = False
):
    """
    Run hard polynomial benchmark with separable bounds.

    This tests the foundational fix: separability detection + exact 1D minimization.
    """
    if problems is None:
        problems = get_hard_polynomial_problems()

    print("\n" + "=" * 80)
    print("HARD POLYNOMIAL BENCHMARK - SEPARABLE BOUNDS TEST")
    print("=" * 80)
    print(f"""
The KEY insight:
  - Styblinski-Tang fails with interval arithmetic (dependency blow-up on x⁴)
  - But it's FULLY SEPARABLE: f(x) = Σ s(xᵢ)
  - Each s(xᵢ) = (xᵢ⁴ - 16xᵢ² + 5xᵢ)/2 is a univariate quartic
  - Exact minimum by solving s'(x) = 0 (cubic)
  - Total LB = Σ min s(xᵢ) is EXACT

This is the FORCED Δ* constructor that makes 100% certification achievable.
""")
    print("=" * 80)
    print(f"Method: Separable Exact Bounds + Branch-and-Reduce")
    print(f"Certification: gap = UB - LB ≤ ε = {epsilon}")
    print(f"Problems: {len(problems)}")
    print("=" * 80 + "\n")

    print(f"{'Problem':<25} {'Dim':>4} {'Sep':>4} {'Verdict':<12} {'UB':>14} {'LB':>14} {'Gap':>12} {'Nodes':>8} {'Time':>8}")
    print("-" * 115)

    results = []
    total_certified = 0
    total_separable_certified = 0
    total_separable = 0
    start_total = time.time()

    for prob in problems:
        try:
            bounds_list = list(zip(prob.lower_bounds.tolist(), prob.upper_bounds.tolist()))
            contract = ProblemContract(
                objective=prob.objective,
                bounds=bounds_list,
                epsilon=epsilon,
                name=prob.name
            )

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

            if prob.is_separable:
                total_separable += 1
                if certified:
                    total_separable_certified += 1

            sep_str = "Yes" if prob.is_separable else "No"

            print(f"{prob.name:<25} {prob.dimension:>4} {sep_str:>4} {verdict_str:<12} {ub:>14.6f} {lb:>14.6f} {gap:>12.2e} {nodes:>8} {elapsed:>7.2f}s")

            # Verbose output
            if verbose and certified:
                cert_data = getattr(result, 'certificate', {})
                print(f"  └─ LB witness tier: {cert_data.get('tier', 'unknown')}")

            results.append({
                'name': prob.name,
                'dimension': prob.dimension,
                'is_separable': prob.is_separable,
                'certified': certified,
                'gap': gap,
                'nodes': nodes,
                'time': elapsed
            })

        except Exception as e:
            print(f"{prob.name:<25} {prob.dimension:>4} {'?':>4} {'ERROR':<12} {str(e)[:30]}")
            results.append({'name': prob.name, 'certified': False, 'error': str(e)})

    total_time = time.time() - start_total

    # Summary
    print("\n" + "=" * 80)
    print("HARD POLYNOMIAL BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\nOverall: {total_certified}/{len(problems)} = {100*total_certified/len(problems):.1f}%")

    if total_separable > 0:
        print(f"Separable functions: {total_separable_certified}/{total_separable} = {100*total_separable_certified/total_separable:.1f}%")

    non_separable = [r for r in results if not r.get('is_separable', True)]
    if non_separable:
        non_sep_certified = sum(1 for r in non_separable if r.get('certified', False))
        print(f"Non-separable functions: {non_sep_certified}/{len(non_separable)} = {100*non_sep_certified/len(non_separable):.1f}%")

    # Styblinski-Tang specific
    st_results = [r for r in results if 'Styblinski' in r.get('name', '')]
    if st_results:
        st_certified = sum(1 for r in st_results if r.get('certified', False))
        print(f"\nStyblinski-Tang: {st_certified}/{len(st_results)} = {100*st_certified/len(st_results):.1f}%")

        if st_certified == len(st_results):
            print("  └─ ALL Styblinski-Tang problems CERTIFIED via separable exact bounds!")

    print(f"\nTotal time: {total_time:.2f}s")
    print("=" * 80)

    if total_certified == len(problems):
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   100% HARD POLYNOMIAL CERTIFICATION ACHIEVED                                ║
║                                                                              ║
║   The FORCED Δ* constructor (separable bounds) eliminates Ω-gap!            ║
║   This is NOT a shortcut - it's the canonical closure for additive structure.║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

    return results


def run_styblinski_tang_focus():
    """Focus test on Styblinski-Tang only."""
    print("\n" + "=" * 80)
    print("STYBLINSKI-TANG FOCUS TEST")
    print("=" * 80)
    print("""
Styblinski-Tang: s(x) = (x⁴ - 16x² + 5x)/2, f(x) = Σ s(xᵢ)

This function exposes interval arithmetic's catastrophic failure:
  - Interval of x⁴ over [-5, 5] = [0, 625]
  - Interval of 16x² over [-5, 5] = [0, 400]
  - Combined: huge overestimation due to dependency

The FIX: Exploit separability.
  - Each s(xᵢ) is a 1D quartic
  - Solve s'(x) = 2x³ - 16x + 2.5 = 0 exactly
  - LB = Σ min s(xᵢ) is EXACT

Expected: 100% certification with 0 branching nodes.
""")
    print("=" * 80)

    problems = get_styblinski_tang_only()

    return run_hard_polynomial_benchmark(
        problems=problems,
        epsilon=1e-4,
        max_iterations=1000,  # Should need almost none
        time_limit=60.0,
        verbose=True
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hard Polynomial Benchmark - Separable Bounds Test"
    )
    parser.add_argument(
        "--styblinski", action="store_true",
        help="Focus on Styblinski-Tang only"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full hard polynomial suite"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1e-4,
        help="Certification tolerance"
    )

    args = parser.parse_args()

    if args.styblinski:
        run_styblinski_tang_focus()
    elif args.full:
        run_hard_polynomial_benchmark(verbose=args.verbose, epsilon=args.epsilon)
    else:
        # Default: run focused Styblinski-Tang test
        run_styblinski_tang_focus()


if __name__ == "__main__":
    main()
