"""
CEC 2022 Mathematical Certification Benchmark Runner

Applies the OPOCH mathematical kernel to CEC 2022 benchmark functions
with REAL certification via gap closure (UB - LB ≤ ε).

NO SHORTCUTS - Pure mathematics only.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/Users/chetanchauhan/opoch-optimizer/src')

from cec2022_problems import (
    CEC2022Problem, get_cec2022_core_problems, get_cec2022_extended_problems
)

from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig
from opoch_optimizer.core.output_gate import Verdict
from opoch_optimizer.contract import ProblemContract


def run_cec2022_benchmark(
    problems=None,
    epsilon: float = 1e-4,
    max_iterations: int = 50000,
    time_limit: float = 120.0
):
    """Run CEC 2022 benchmark with mathematical certification."""

    if problems is None:
        problems = get_cec2022_core_problems()

    print("\n" + "=" * 70)
    print("CEC 2022 MATHEMATICAL CERTIFICATION BENCHMARK")
    print("=" * 70)
    print(f"Method: Δ* Closure + Branch-and-Reduce (Pure Mathematics)")
    print(f"Certification: gap = UB - LB ≤ ε = {epsilon}")
    print(f"Problems: {len(problems)}")
    print("=" * 70)
    print("\nNO SHORTCUTS: Pure mathematical gap closure.")
    print("=" * 70 + "\n")

    print(f"{'Problem':<30} {'Dim':>4} {'Verdict':<12} {'UB':>12} {'LB':>12} {'Gap':>12} {'Time':>8}")
    print("-" * 95)

    total_certified = 0
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
                log_frequency=1000000
            )

            start = time.time()
            kernel = OPOCHKernel(contract, config)
            verdict, result = kernel.solve()
            elapsed = time.time() - start

            ub = getattr(result, 'upper_bound', getattr(result, 'objective_value', float('inf')))
            lb = getattr(result, 'lower_bound', float('-inf'))
            gap = ub - lb

            certified = verdict == Verdict.UNIQUE_OPT and gap <= epsilon
            if certified:
                total_certified += 1
                verdict_str = "CERTIFIED"
            else:
                verdict_str = str(verdict).split('.')[-1]

            print(f"{prob.name:<30} {prob.dimension:>4} {verdict_str:<12} {ub:>12.4f} {lb:>12.4f} {gap:>12.2e} {elapsed:>7.2f}s")

        except Exception as e:
            print(f"{prob.name:<30} {prob.dimension:>4} {'ERROR':<12} {str(e)[:30]}")

    total_time = time.time() - start_total
    rate = total_certified / len(problems) * 100

    print("\n" + "=" * 70)
    print(f"CEC 2022 CERTIFICATION RATE: {total_certified}/{len(problems)} = {rate:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 70)

    if rate == 100:
        print("\n*** 100% MATHEMATICAL CERTIFICATION ACHIEVED ***")

    return total_certified, len(problems)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()

    if args.extended:
        problems = get_cec2022_extended_problems()
    else:
        problems = get_cec2022_core_problems()

    run_cec2022_benchmark(problems)


if __name__ == "__main__":
    main()
