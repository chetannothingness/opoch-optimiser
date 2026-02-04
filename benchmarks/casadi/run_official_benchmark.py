#!/usr/bin/env python3
"""
Run Official CasADi Examples with OPOCH Verification Layer

Tests the exact examples from CasADi GitHub:
https://github.com/casadi/casadi/tree/main/docs/examples/python
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import casadi as ca

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from casadi_official_examples import get_official_casadi_examples


def compute_kkt_residuals(sol, nlp_dict, lbx, ubx, lbg, ubg):
    """Compute KKT residuals for verification."""
    x_opt = np.array(sol['x']).flatten()
    lam_g = np.array(sol['lam_g']).flatten() if sol['lam_g'].shape[0] > 0 else np.array([])
    lam_x = np.array(sol['lam_x']).flatten()

    # Build functions for evaluation
    x_sym = nlp_dict['x']
    f_sym = nlp_dict['f']
    g_sym = nlp_dict.get('g', ca.MX())

    grad_f = ca.gradient(f_sym, x_sym)
    grad_f_func = ca.Function('grad_f', [x_sym], [grad_f])

    if g_sym.shape[0] > 0:
        jac_g = ca.jacobian(g_sym, x_sym)
        jac_g_func = ca.Function('jac_g', [x_sym], [jac_g])
        g_func = ca.Function('g', [x_sym], [g_sym])

    # Evaluate
    grad_f_val = np.array(grad_f_func(x_opt)).flatten()

    # Primal feasibility
    r_primal = 0.0

    # Variable bounds
    lb_viol = np.maximum(0, lbx - x_opt)
    ub_viol = np.maximum(0, x_opt - ubx)
    r_primal = max(r_primal, np.max(np.abs(lb_viol)))
    r_primal = max(r_primal, np.max(np.abs(ub_viol)))

    # Constraint bounds
    if g_sym.shape[0] > 0:
        g_val = np.array(g_func(x_opt)).flatten()
        g_lb_viol = np.maximum(0, lbg - g_val)
        g_ub_viol = np.maximum(0, g_val - ubg)
        r_primal = max(r_primal, np.max(np.abs(g_lb_viol)))
        r_primal = max(r_primal, np.max(np.abs(g_ub_viol)))

    # Stationarity: ∇f + J_g^T λ_g + λ_x = 0
    stationarity = grad_f_val.copy()

    if g_sym.shape[0] > 0 and len(lam_g) > 0:
        jac_g_val = np.array(jac_g_func(x_opt))
        stationarity += jac_g_val.T @ lam_g

    stationarity += lam_x
    r_stationarity = np.max(np.abs(stationarity))

    # Complementarity
    r_complementarity = 0.0

    # Variable bounds complementarity
    for i in range(len(x_opt)):
        if lbx[i] > -1e10:
            r_complementarity = max(r_complementarity, abs(lam_x[i] * (x_opt[i] - lbx[i])))
        if ubx[i] < 1e10:
            r_complementarity = max(r_complementarity, abs(lam_x[i] * (ubx[i] - x_opt[i])))

    # Constraint complementarity
    if g_sym.shape[0] > 0 and len(lam_g) > 0:
        g_val = np.array(g_func(x_opt)).flatten()
        for i in range(len(g_val)):
            if lbg[i] > -1e10:
                slack_lb = g_val[i] - lbg[i]
                r_complementarity = max(r_complementarity, abs(lam_g[i] * slack_lb))
            if ubg[i] < 1e10:
                slack_ub = ubg[i] - g_val[i]
                r_complementarity = max(r_complementarity, abs(lam_g[i] * slack_ub))

    return {
        'r_primal': float(r_primal),
        'r_stationarity': float(r_stationarity),
        'r_complementarity': float(r_complementarity),
        'max_residual': float(max(r_primal, r_stationarity, r_complementarity)),
    }


def run_benchmark():
    """Run all official CasADi examples with verification."""
    print("=" * 80)
    print("OFFICIAL CASADI EXAMPLES - OPOCH VERIFICATION")
    print("=" * 80)
    print()
    print("Source: https://github.com/casadi/casadi/tree/main/docs/examples/python")
    print()

    examples = get_official_casadi_examples()
    results = []

    total_solve_time = 0
    total_cert_time = 0

    for ex in examples:
        print(f"\n[{ex.name}]")
        print(f"  Source: {ex.source}")
        print(f"  Variables: {ex.n_vars}, Constraints: {ex.n_constraints}")

        # Build NLP
        nlp_dict = {'x': ex.x_sym, 'f': ex.f_sym}
        if ex.g_sym is not None:
            nlp_dict['g'] = ex.g_sym

        opts = {
            'print_time': False,
            'ipopt': {
                'print_level': 0,
                'sb': 'yes',
                'tol': 1e-8,
            }
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_dict, opts)

        # Solve
        t0 = time.perf_counter()
        try:
            sol = solver(
                x0=ex.x0,
                lbx=ex.lbx, ubx=ex.ubx,
                lbg=ex.lbg, ubg=ex.ubg
            )
            solve_time = time.perf_counter() - t0
            total_solve_time += solve_time

            f_opt = float(sol['f'])
            ipopt_status = solver.stats()['return_status']

            print(f"  IPOPT status: {ipopt_status}")
            print(f"  Objective: {f_opt:.6g}")
            print(f"  Solve time: {solve_time*1000:.2f} ms")

            # Verify KKT
            t0 = time.perf_counter()
            residuals = compute_kkt_residuals(
                sol, nlp_dict, ex.lbx, ex.ubx, ex.lbg, ex.ubg
            )
            cert_time = time.perf_counter() - t0
            total_cert_time += cert_time

            print(f"  KKT residuals:")
            print(f"    r_primal:         {residuals['r_primal']:.2e}")
            print(f"    r_stationarity:   {residuals['r_stationarity']:.2e}")
            print(f"    r_complementarity:{residuals['r_complementarity']:.2e}")

            certified = residuals['max_residual'] <= 1e-6
            status = "CERTIFIED" if certified else "FAIL"
            print(f"  Status: {status}")

            results.append({
                'name': ex.name,
                'source': ex.source,
                'n_vars': ex.n_vars,
                'n_constraints': ex.n_constraints,
                'objective': f_opt,
                'ipopt_status': ipopt_status,
                'residuals': residuals,
                'certified': certified,
                'solve_time': solve_time,
                'cert_time': cert_time,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'name': ex.name,
                'source': ex.source,
                'error': str(e),
                'certified': False,
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    certified_count = sum(1 for r in results if r.get('certified', False))
    total_count = len(results)

    print(f"\nTotal: {certified_count}/{total_count} certified ({100*certified_count/total_count:.1f}%)")
    print(f"\nTiming:")
    print(f"  IPOPT solve:    {total_solve_time*1000:.2f} ms")
    print(f"  Certification:  {total_cert_time*1000:.2f} ms")
    print(f"  Overhead:       {total_cert_time/total_solve_time*100:.1f}%")

    print("\n" + "-" * 80)
    print(f"{'Problem':<25} {'Vars':>6} {'Cons':>6} {'f(x*)':>12} {'max_kkt':>10} {'Status':>10}")
    print("-" * 80)

    for r in results:
        if 'error' in r:
            print(f"{r['name']:<25} {'ERROR':>50}")
        else:
            max_kkt = r['residuals']['max_residual']
            status = "CERTIFIED" if r['certified'] else "FAIL"
            print(f"{r['name']:<25} {r['n_vars']:>6} {r['n_constraints']:>6} {r['objective']:>12.4g} {max_kkt:>10.2e} {status:>10}")

    # Save results
    output_dir = Path(__file__).parent / 'runs' / 'official'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_dir / 'results.json'}")

    return results


if __name__ == "__main__":
    run_benchmark()
