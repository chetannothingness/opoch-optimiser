"""
Complete Mathematical Certification Benchmark

Runs ALL benchmarks with pure mathematics, NO shortcuts:
1. GLOBALLib HARD (38 problems)
2. CEC 2020 Extended (25 problems)
3. CEC 2020 HARD (14 problems - up to 20D)

Total: 77 problems with mathematical gap closure certification.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/Users/chetanchauhan/opoch-optimizer/src')

from opoch_optimizer.solver.opoch_kernel import OPOCHKernel, OPOCHConfig
from opoch_optimizer.core.output_gate import Verdict
from opoch_optimizer.contract import ProblemContract


def run_globallib():
    """Run GLOBALLib HARD benchmark."""
    from globallib_hard import get_all

    print("\n" + "=" * 70)
    print("GLOBALLib HARD BENCHMARK (38 problems)")
    print("=" * 70)

    problems = get_all()
    certified = 0
    results = []

    print(f"\n{'Problem':<30} {'Cat':<12} {'Verdict':<12} {'Gap':>12} {'Time':>8}")
    print("-" * 80)

    for prob in problems:
        try:
            contract = prob.to_problem_contract()
            config = OPOCHConfig(
                epsilon=1e-4,
                max_time=60.0,
                max_nodes=20000,
                log_frequency=1000000
            )

            start = time.time()
            kernel = OPOCHKernel(contract, config)
            verdict, result = kernel.solve()
            elapsed = time.time() - start

            ub = getattr(result, 'upper_bound', getattr(result, 'objective_value', float('inf')))
            lb = getattr(result, 'lower_bound', float('-inf'))
            gap = ub - lb

            is_certified = verdict == Verdict.UNIQUE_OPT and gap <= 1e-4
            if is_certified:
                certified += 1
                verdict_str = "CERTIFIED"
            else:
                verdict_str = str(verdict).split('.')[-1]

            print(f"{prob.name:<30} {prob.category:<12} {verdict_str:<12} {gap:>12.2e} {elapsed:>7.2f}s")
            results.append({
                'name': prob.name,
                'certified': is_certified,
                'gap': gap,
                'time': elapsed
            })

        except Exception as e:
            print(f"{prob.name:<30} {'ERROR':<12} {str(e)[:20]}")
            results.append({'name': prob.name, 'certified': False, 'error': str(e)})

    rate = certified / len(problems) * 100
    print(f"\nGLOBALLib HARD: {certified}/{len(problems)} = {rate:.1f}%")
    return certified, len(problems)


def run_cec2020_all():
    """Run all CEC 2020 problems."""
    from cec2020_problems import (
        get_cec2020_extended_problems, get_cec2020_hard_problems
    )

    print("\n" + "=" * 70)
    print("CEC 2020 BENCHMARK (Extended + HARD)")
    print("=" * 70)

    # Combine extended and hard (avoiding duplicates)
    extended = get_cec2020_extended_problems()
    hard = get_cec2020_hard_problems()

    # Filter hard to only include dimensions > 10
    hard_unique = [p for p in hard if p.dimension > 10 or 'Griewank_10D' in p.name]
    problems = extended + hard_unique

    certified = 0

    print(f"\n{'Problem':<35} {'Dim':>4} {'Verdict':<12} {'Gap':>12} {'Time':>8}")
    print("-" * 80)

    for prob in problems:
        try:
            bounds_list = list(zip(prob.lower_bounds.tolist(), prob.upper_bounds.tolist()))
            contract = ProblemContract(
                objective=prob.objective,
                bounds=bounds_list,
                epsilon=1e-4,
                name=prob.name
            )

            config = OPOCHConfig(
                epsilon=1e-4,
                max_time=120.0,
                max_nodes=50000,
                log_frequency=1000000
            )

            start = time.time()
            kernel = OPOCHKernel(contract, config)
            verdict, result = kernel.solve()
            elapsed = time.time() - start

            ub = getattr(result, 'upper_bound', getattr(result, 'objective_value', float('inf')))
            lb = getattr(result, 'lower_bound', float('-inf'))
            gap = ub - lb

            is_certified = verdict == Verdict.UNIQUE_OPT and gap <= 1e-4
            if is_certified:
                certified += 1
                verdict_str = "CERTIFIED"
            else:
                verdict_str = str(verdict).split('.')[-1]

            print(f"{prob.name:<35} {prob.dimension:>4} {verdict_str:<12} {gap:>12.2e} {elapsed:>7.2f}s")

        except Exception as e:
            print(f"{prob.name:<35} {prob.dimension:>4} {'ERROR':<12} {str(e)[:20]}")

    rate = certified / len(problems) * 100
    print(f"\nCEC 2020: {certified}/{len(problems)} = {rate:.1f}%")
    return certified, len(problems)


def main():
    """Run complete benchmark suite."""
    print("\n" + "=" * 70)
    print("OPOCH COMPLETE MATHEMATICAL CERTIFICATION")
    print("=" * 70)
    print("NO SHORTCUTS. Pure mathematics via gap closure (UB - LB ≤ ε)")
    print("=" * 70)

    start_total = time.time()

    # Run GLOBALLib
    gl_certified, gl_total = run_globallib()

    # Run CEC 2020
    cec_certified, cec_total = run_cec2020_all()

    total_time = time.time() - start_total

    # Final summary
    total_certified = gl_certified + cec_certified
    total_problems = gl_total + cec_total
    rate = total_certified / total_problems * 100

    print("\n" + "=" * 70)
    print("COMPLETE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"\nGLOBALLib HARD: {gl_certified}/{gl_total} ({gl_certified/gl_total*100:.1f}%)")
    print(f"CEC 2020:       {cec_certified}/{cec_total} ({cec_certified/cec_total*100:.1f}%)")
    print("-" * 40)
    print(f"TOTAL:          {total_certified}/{total_problems} = {rate:.1f}%")
    print(f"\nTotal time: {total_time:.2f}s")

    if rate == 100:
        print("\n" + "=" * 70)
        print("*** 100% MATHEMATICAL CERTIFICATION ACHIEVED ***")
        print("All problems certified via gap closure (UB - LB ≤ ε)")
        print("NO shortcuts, NO reference values, PURE MATHEMATICS")
        print("=" * 70)


if __name__ == "__main__":
    main()
