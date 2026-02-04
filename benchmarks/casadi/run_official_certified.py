#!/usr/bin/env python3
"""
Run Official CasADi Examples with Corrected KKT Verifier + Repair Loop

This implements:
1. Correct two-sided KKT decomposition (split multipliers by sign)
2. Deterministic repair loop (scaling-aware, warmstart)
3. IPOPT-consistent termination metrics
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import casadi as ca

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from casadi_official_examples import get_official_casadi_examples
from kkt_verifier import verify_casadi_solution, certify_or_repair, KKTResiduals


def run_benchmark():
    """Run all official CasADi examples with corrected verification."""
    print("=" * 80)
    print("OFFICIAL CASADI EXAMPLES - CORRECTED KKT VERIFIER + REPAIR LOOP")
    print("=" * 80)
    print()
    print("Source: https://github.com/casadi/casadi/tree/main/docs/examples/python")
    print()
    print("Fixes applied:")
    print("  1. Two-sided KKT decomposition (split multipliers by sign)")
    print("  2. Deterministic repair loop (no scaling, strict tolerances)")
    print("  3. IPOPT-consistent complementarity computation")
    print()

    examples = get_official_casadi_examples()
    results = []

    total_time = 0

    for ex in examples:
        print(f"\n{'='*60}")
        print(f"[{ex.name}]")
        print(f"  Source: {ex.source}")
        print(f"  Variables: {ex.n_vars}, Constraints: {ex.n_constraints}")

        # Build NLP
        nlp_dict = {'x': ex.x_sym, 'f': ex.f_sym}
        if ex.g_sym is not None:
            nlp_dict['g'] = ex.g_sym

        t0 = time.perf_counter()

        try:
            sol, residuals, round_passed, opts_used = certify_or_repair(
                nlp_dict=nlp_dict,
                x0=ex.x0,
                lbx=ex.lbx, ubx=ex.ubx,
                lbg=ex.lbg, ubg=ex.ubg,
                epsilon=1e-6,
                max_rounds=3,
            )

            elapsed = time.perf_counter() - t0
            total_time += elapsed

            f_opt = float(sol['f'])

            print(f"  Objective: {f_opt:.6g}")
            print(f"  Time: {elapsed*1000:.2f} ms")
            print(f"  Round passed: {round_passed}")
            print()
            print(f"  KKT Residuals (corrected):")
            print(f"    r_primal:         {residuals.r_primal:.2e}")
            print(f"    r_dual:           {residuals.r_dual:.2e}")
            print(f"    r_complementarity:{residuals.r_complementarity:.2e}")
            print(f"    r_max:            {residuals.r_max:.2e}")
            print()
            print(f"  Complementarity breakdown:")
            print(f"    compl_x_lb: {residuals.compl_x_lb:.2e}")
            print(f"    compl_x_ub: {residuals.compl_x_ub:.2e}")
            print(f"    compl_g_lb: {residuals.compl_g_lb:.2e}")
            print(f"    compl_g_ub: {residuals.compl_g_ub:.2e}")

            status = "CERTIFIED" if residuals.certified else "FAIL"
            print(f"\n  Status: {status}")

            results.append({
                'name': ex.name,
                'source': ex.source,
                'n_vars': ex.n_vars,
                'n_constraints': ex.n_constraints,
                'objective': f_opt,
                'r_primal': residuals.r_primal,
                'r_dual': residuals.r_dual,
                'r_complementarity': residuals.r_complementarity,
                'r_max': residuals.r_max,
                'certified': residuals.certified,
                'round_passed': round_passed,
                'time': elapsed,
            })

        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  ERROR: {e}")
            results.append({
                'name': ex.name,
                'source': ex.source,
                'error': str(e),
                'certified': False,
                'time': elapsed,
            })

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    certified_count = sum(1 for r in results if r.get('certified', False))
    total_count = len(results)

    print(f"\nTotal: {certified_count}/{total_count} certified ({100*certified_count/total_count:.1f}%)")
    print(f"Total time: {total_time*1000:.2f} ms")

    print("\n" + "-" * 80)
    print(f"{'Problem':<25} {'Vars':>6} {'r_max':>10} {'Round':>6} {'Status':>10}")
    print("-" * 80)

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<25} {'ERROR':>40}")
        else:
            status = "CERTIFIED" if r['certified'] else "FAIL"
            round_str = str(r['round_passed']) if r['round_passed'] >= 0 else "all"
            print(f"{r['name']:<25} {r['n_vars']:>6} {r['r_max']:>10.2e} {round_str:>6} {status:>10}")

    # Save results
    output_dir = Path(__file__).parent / 'runs' / 'official_certified'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'results.json'}")

    return results


if __name__ == "__main__":
    run_benchmark()
