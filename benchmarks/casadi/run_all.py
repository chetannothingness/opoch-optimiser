#!/usr/bin/env python3
"""
OPOCH CasADi Verification Layer - Master Benchmark Runner

Runs all benchmark suites and generates comprehensive results.
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import casadi as ca

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from casadi_official_examples import get_official_casadi_examples
from hock_schittkowski import get_hock_schittkowski_problems
from suite_a_industrial import get_industrial_problems
from suite_b_regression import get_regression_problems
from kkt_verifier import verify_casadi_solution, certify_or_repair


def run_problem(nlp, epsilon=1e-6, max_rounds=3):
    """Run a single problem with KKT verification and repair."""
    # Build NLP dict
    nlp_dict = {'x': nlp.x_sym, 'f': nlp.f_sym}
    if hasattr(nlp, 'g_sym') and nlp.g_sym is not None:
        nlp_dict['g'] = nlp.g_sym

    # Get bounds
    lbx = np.array(nlp.lbx) if hasattr(nlp, 'lbx') else np.full(nlp.n_vars, -np.inf)
    ubx = np.array(nlp.ubx) if hasattr(nlp, 'ubx') else np.full(nlp.n_vars, np.inf)
    lbg = np.array(nlp.lbg) if hasattr(nlp, 'lbg') and len(nlp.lbg) > 0 else np.array([])
    ubg = np.array(nlp.ubg) if hasattr(nlp, 'ubg') and len(nlp.ubg) > 0 else np.array([])
    x0 = np.array(nlp.x0) if hasattr(nlp, 'x0') else np.zeros(nlp.n_vars)

    t0 = time.perf_counter()

    try:
        sol, residuals, round_passed, opts = certify_or_repair(
            nlp_dict=nlp_dict,
            x0=x0,
            lbx=lbx, ubx=ubx,
            lbg=lbg, ubg=ubg,
            epsilon=epsilon,
            max_rounds=max_rounds,
        )

        elapsed = time.perf_counter() - t0

        return {
            'name': nlp.name,
            'n_vars': nlp.n_vars,
            'n_constraints': nlp.n_constraints,
            'objective': float(sol['f']),
            'r_primal': residuals.r_primal,
            'r_dual': residuals.r_dual,
            'r_complementarity': residuals.r_complementarity,
            'r_max': residuals.r_max,
            'certified': residuals.certified,
            'round_passed': round_passed,
            'time_ms': elapsed * 1000,
            'error': None,
        }

    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            'name': nlp.name,
            'n_vars': nlp.n_vars,
            'n_constraints': nlp.n_constraints,
            'certified': False,
            'time_ms': elapsed * 1000,
            'error': str(e),
        }


def run_suite(suite_name, problems, epsilon=1e-6):
    """Run a complete test suite."""
    print(f"\n{'='*80}")
    print(f"SUITE: {suite_name}")
    print(f"{'='*80}")

    results = []
    total_time = 0

    for nlp in problems:
        result = run_problem(nlp, epsilon)
        results.append(result)
        total_time += result['time_ms']

        status = "CERTIFIED" if result['certified'] else "FAIL"
        if result['error']:
            status = f"ERROR: {result['error'][:30]}"

        r_max_str = f"{result.get('r_max', 'N/A'):.2e}" if 'r_max' in result else "N/A"
        print(f"  {nlp.name:<30} {nlp.n_vars:>4} vars  r_max={r_max_str}  {status}")

    certified = sum(1 for r in results if r['certified'])
    total = len(results)

    print(f"\n  Summary: {certified}/{total} certified ({100*certified/total:.1f}%)")
    print(f"  Total time: {total_time:.2f} ms")

    return results, certified, total


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("OPOCH CasADi Verification Layer - Complete Benchmark Suite")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_results = {}
    total_certified = 0
    total_problems = 0

    # Suite 1: Official CasADi Examples
    try:
        official = get_official_casadi_examples()
        results, certified, total = run_suite("Official CasADi Examples", official)
        all_results['official'] = results
        total_certified += certified
        total_problems += total
    except Exception as e:
        print(f"  Suite failed: {e}")

    # Suite 2: Hock-Schittkowski
    try:
        hs_problems = get_hock_schittkowski_problems()
        results, certified, total = run_suite("Hock-Schittkowski", hs_problems)
        all_results['hock_schittkowski'] = results
        total_certified += certified
        total_problems += total
    except Exception as e:
        print(f"  Suite failed: {e}")

    # Suite 3: Industrial Problems
    try:
        industrial = get_industrial_problems()
        results, certified, total = run_suite("Industrial (OCP, Robotics)", industrial)
        all_results['industrial'] = results
        total_certified += certified
        total_problems += total
    except Exception as e:
        print(f"  Suite failed: {e}")

    # Suite 4: Regression Problems
    try:
        regression = get_regression_problems()
        results, certified, total = run_suite("Regression (NIST-style)", regression)
        all_results['regression'] = results
        total_certified += certified
        total_problems += total
    except Exception as e:
        print(f"  Suite failed: {e}")

    # Final Summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\nTotal: {total_certified}/{total_problems} certified ({100*total_certified/total_problems:.1f}%)")

    print("\n" + "-" * 80)
    print(f"{'Suite':<30} {'Certified':>10} {'Total':>10} {'Rate':>10}")
    print("-" * 80)

    for suite_name, results in all_results.items():
        certified = sum(1 for r in results if r['certified'])
        total = len(results)
        rate = 100 * certified / total if total > 0 else 0
        print(f"{suite_name:<30} {certified:>10} {total:>10} {rate:>9.1f}%")

    # Detailed table
    print("\n" + "-" * 80)
    print(f"{'Problem':<35} {'Vars':>6} {'Cons':>6} {'r_max':>12} {'Status':>12}")
    print("-" * 80)

    for suite_name, results in all_results.items():
        print(f"\n[{suite_name}]")
        for r in results:
            status = "CERTIFIED" if r['certified'] else "FAIL"
            if r.get('error'):
                status = "ERROR"
            r_max = r.get('r_max', float('nan'))
            n_cons = r.get('n_constraints', 0)
            print(f"  {r['name']:<33} {r['n_vars']:>6} {n_cons:>6} {r_max:>12.2e} {status:>12}")

    # Save results
    output_dir = Path(__file__).parent / 'runs' / 'complete'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_certified': total_certified,
            'total_problems': total_problems,
            'suites': all_results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Also save latest results
    with open(output_dir / 'latest.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_certified': total_certified,
            'total_problems': total_problems,
            'suites': all_results,
        }, f, indent=2, default=str)

    return total_certified, total_problems


if __name__ == "__main__":
    certified, total = main()

    # Exit with success only if all certified
    sys.exit(0 if certified == total else 1)
