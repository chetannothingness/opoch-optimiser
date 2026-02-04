#!/usr/bin/env python3
"""
OPOCH CasADi Certified Benchmark
================================

100% certified KKT verification on Hock-Schittkowski standard problems.
No shortcuts. Real numbers. Mathematical proof.

Usage:
    cd /tmp && python /path/to/run_certified_benchmark.py
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

# Import CasADi
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    print("ERROR: CasADi not installed. Install with: pip install casadi")
    sys.exit(1)

# Import OPOCH components
from opoch_optimizer.casadi.nlp_contract import CasADiNLP, create_nlp_from_casadi
from opoch_optimizer.casadi.adapter import CasADiAdapter
from opoch_optimizer.casadi.solver_wrapper import DeterministicIPOPT, SolverResult
from opoch_optimizer.casadi.kkt_certificate import KKTCertifier, KKTCertificate, KKTStatus


# ============================================================================
# KNOWN OPTIMAL SOLUTIONS (From Hock-Schittkowski Reference)
# ============================================================================
KNOWN_OPTIMA = {
    'hs035': {'x': [4/3, 7/9, 4/9], 'f': 1/9},
    'hs038': {'x': [1, 1, 1, 1], 'f': 0},
    'hs044': {'x': [0, 3, 0, 4], 'f': -15},
    'hs065': {'x': [3.650461821, 3.65046169, 4.6204170507], 'f': 0.9535288567},
    'hs071': {'x': [1.0, 4.74299963, 3.82114998, 1.37940829], 'f': 17.0140173},
    'hs076': {'x': [0.0, 0.0, 0.5, 0.0], 'f': -4.681818},
    'hs100': {'f': 680.63},  # Known from literature
    'rosenbrock_2d': {'x': [1, 1], 'f': 0},
    'rosenbrock_5d': {'x': [1, 1, 1, 1, 1], 'f': 0},
    'rosenbrock_10d': {'x': [1]*10, 'f': 0},
}


# ============================================================================
# HOCK-SCHITTKOWSKI PROBLEM DEFINITIONS
# ============================================================================

def hs035() -> CasADiNLP:
    """HS035: min 9 - 8x1 - 6x2 - 4x3 + 2x1² + 2x2² + x3² + 2x1x2 + 2x1x3, s.t. x1+x2+2x3<=3"""
    x = ca.SX.sym('x', 3)
    f = 9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[0]*x[1] + 2*x[0]*x[2]
    g = x[0] + x[1] + 2*x[2]
    return create_nlp_from_casadi(
        x=x, f=f, g=g,
        lbx=[0, 0, 0], ubx=[ca.inf, ca.inf, ca.inf],
        lbg=[-ca.inf], ubg=[3],
        x0=[0.5, 0.5, 0.5], name='hs035'
    )


def hs038() -> CasADiNLP:
    """HS038: Extended Rosenbrock (unconstrained with bounds)"""
    x = ca.SX.sym('x', 4)
    f = (100*(x[1] - x[0]**2)**2 + (1 - x[0])**2 +
         90*(x[3] - x[2]**2)**2 + (1 - x[2])**2 +
         10.1*((x[1] - 1)**2 + (x[3] - 1)**2) +
         19.8*(x[1] - 1)*(x[3] - 1))
    return create_nlp_from_casadi(
        x=x, f=f,
        lbx=[-10, -10, -10, -10], ubx=[10, 10, 10, 10],
        x0=[-3, -1, -3, -1], name='hs038'
    )


def hs044() -> CasADiNLP:
    """HS044: LP-like with bilinear objective"""
    x = ca.SX.sym('x', 4)
    f = x[0] - x[1] - x[2] - x[0]*x[2] + x[0]*x[3] + x[1]*x[2] - x[1]*x[3]
    g = ca.vertcat(
        x[0] + 2*x[1],
        4*x[0] + x[1],
        3*x[0] + 4*x[1],
        2*x[2] + x[3],
        x[2] + 2*x[3],
        x[2] + x[3]
    )
    return create_nlp_from_casadi(
        x=x, f=f, g=g,
        lbx=[0, 0, 0, 0], ubx=[ca.inf, ca.inf, ca.inf, ca.inf],
        lbg=[-ca.inf, -ca.inf, -ca.inf, -ca.inf, -ca.inf, -ca.inf],
        ubg=[8, 12, 12, 8, 8, 5],
        x0=[1, 1, 1, 1], name='hs044'
    )


def hs065() -> CasADiNLP:
    """HS065: Quadratic objective with quadratic constraint"""
    x = ca.SX.sym('x', 3)
    f = (x[0] - x[1])**2 + ((x[0] + x[1] - 10)**2)/9 + (x[2] - 5)**2
    g = x[0]**2 + x[1]**2 + x[2]**2
    return create_nlp_from_casadi(
        x=x, f=f, g=g,
        lbx=[-4.5, -4.5, -5], ubx=[4.5, 4.5, 5],
        lbg=[-ca.inf], ubg=[48],
        x0=[-5, 5, 0], name='hs065'
    )


def hs071() -> CasADiNLP:
    """HS071: THE IPOPT STANDARD TEST PROBLEM"""
    x = ca.SX.sym('x', 4)
    f = x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2]
    g = ca.vertcat(
        x[0]*x[1]*x[2]*x[3],  # >= 25
        x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2  # = 40
    )
    return create_nlp_from_casadi(
        x=x, f=f, g=g,
        lbx=[1, 1, 1, 1], ubx=[5, 5, 5, 5],
        lbg=[25, 40], ubg=[ca.inf, 40],
        x0=[1, 5, 5, 1], name='hs071'
    )


def hs076() -> CasADiNLP:
    """HS076: QP with linear constraints"""
    x = ca.SX.sym('x', 4)
    f = x[0]**2 + 0.5*x[1]**2 + x[2]**2 + 0.5*x[3]**2 - x[0]*x[2] + x[2]*x[3] - x[0] - 3*x[1] + x[2] - x[3]
    g = ca.vertcat(
        x[0] + 2*x[1] + x[2] + x[3],
        3*x[0] + x[1] + 2*x[2] - x[3],
        -x[1] + 4*x[2]
    )
    return create_nlp_from_casadi(
        x=x, f=f, g=g,
        lbx=[0, 0, 0, 0], ubx=[ca.inf, ca.inf, ca.inf, ca.inf],
        lbg=[-ca.inf, -ca.inf, -ca.inf], ubg=[5, 4, 1.5],
        x0=[0.5, 0.5, 0.5, 0.5], name='hs076'
    )


def hs100() -> CasADiNLP:
    """HS100: 7 variables, 4 constraints"""
    x = ca.SX.sym('x', 7)
    f = ((x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 +
         10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6])
    g = ca.vertcat(
        2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4],
        7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4],
        23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6],
        4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]
    )
    return create_nlp_from_casadi(
        x=x, f=f, g=g,
        lbx=[-ca.inf]*7, ubx=[ca.inf]*7,
        lbg=[-ca.inf, -ca.inf, -ca.inf, -ca.inf], ubg=[127, 282, 196, 0],
        x0=[1, 2, 0, 4, 0, 1, 1], name='hs100'
    )


def rosenbrock(n: int) -> CasADiNLP:
    """Rosenbrock function in n dimensions"""
    x = ca.SX.sym('x', n)
    f = 0
    for i in range(n-1):
        f += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return create_nlp_from_casadi(
        x=x, f=f,
        lbx=[-10]*n, ubx=[10]*n,
        x0=[-1.2 if i % 2 == 0 else 1.0 for i in range(n)],
        name=f'rosenbrock_{n}d'
    )


def get_all_problems():
    """Get all benchmark problems."""
    return [
        hs035(),
        hs038(),
        hs044(),
        hs065(),
        hs071(),  # THE STANDARD
        hs076(),
        hs100(),
        rosenbrock(2),
        rosenbrock(5),
        rosenbrock(10),
    ]


# ============================================================================
# CERTIFIED SOLVER AND VERIFIER
# ============================================================================

def solve_and_certify(nlp: CasADiNLP) -> dict:
    """
    Solve NLP and certify KKT conditions.

    Returns dict with:
        - objective: f(x*)
        - certified: bool
        - status: KKT status
        - residuals: {primal, stationarity, complementarity, dual}
        - solution: x*
        - iterations: solver iterations
        - time: solve time
        - known_f: known optimal from literature
        - gap_from_known: |f - f*|
    """
    result = {
        'name': nlp.name,
        'n_vars': nlp.n_vars,
        'n_constraints': nlp.n_constraints,
        'certified': False,
        'error': None,
    }

    try:
        # Build adapter
        adapter = CasADiAdapter(nlp)

        # Solve with deterministic IPOPT
        solver = DeterministicIPOPT(nlp, adapter=adapter)
        sol = solver.solve()

        result['objective'] = sol.f
        result['solution'] = sol.x.tolist()
        result['iterations'] = sol.iterations
        result['time'] = sol.time
        result['solver_status'] = sol.return_status

        # Check known optimum
        known = KNOWN_OPTIMA.get(nlp.name, {})
        if 'f' in known:
            result['known_f'] = known['f']
            result['gap_from_known'] = abs(sol.f - known['f'])

        # Certify KKT conditions
        certifier = KKTCertifier(adapter)
        cert = certifier.certify(
            sol.x, sol.lam_g, sol.lam_x,
            eps_feas=1e-6, eps_kkt=1e-6, eps_comp=1e-6,
            solver_info={
                'solver_name': 'IPOPT',
                'iterations': sol.iterations,
                'time': sol.time,
                'status': sol.return_status,
            }
        )

        result['certified'] = cert.is_certified
        result['status'] = cert.status.value
        result['residuals'] = {
            'primal': cert.r_primal,
            'stationarity': cert.r_stationarity,
            'complementarity': cert.r_complementarity,
            'dual': cert.r_dual,
        }
        result['max_residual'] = cert.max_residual
        result['hashes'] = {
            'input': cert.input_hash[:16],
            'solution': cert.solution_hash[:16],
            'certificate': cert.certificate_hash[:16],
        }

    except Exception as e:
        result['error'] = str(e)
        import traceback
        result['traceback'] = traceback.format_exc()

    return result


# ============================================================================
# MAIN BENCHMARK RUNNER
# ============================================================================

def run_benchmark():
    """Run complete certified benchmark suite."""
    print("=" * 80)
    print("OPOCH CERTIFIED BENCHMARK - HOCK-SCHITTKOWSKI STANDARD PROBLEMS")
    print("=" * 80)
    print()
    print("Contract L: KKT Certificate with residuals r_p, r_s, r_c, r_d")
    print("Certified iff: max(r_p, r_s, r_c, r_d) <= epsilon")
    print()
    print("NO SHORTCUTS. REAL NUMBERS. MATHEMATICAL PROOF.")
    print("=" * 80)
    print()

    problems = get_all_problems()
    results = []

    total_certified = 0
    total_problems = len(problems)

    for i, nlp in enumerate(problems, 1):
        print(f"[{i}/{total_problems}] {nlp.name}")
        print(f"    Variables: {nlp.n_vars}, Constraints: {nlp.n_constraints}")

        result = solve_and_certify(nlp)
        results.append(result)

        if result.get('error'):
            print(f"    ERROR: {result['error']}")
            continue

        # Print results
        print(f"    Objective: {result['objective']:.10g}")

        if 'known_f' in result:
            gap = result['gap_from_known']
            print(f"    Known f*:  {result['known_f']:.10g}")
            print(f"    Gap:       {gap:.2e}")

        print(f"    Iterations: {result['iterations']}, Time: {result['time']:.4f}s")

        # KKT residuals
        r = result['residuals']
        print(f"    KKT Residuals:")
        print(f"        r_primal:         {r['primal']:.2e}")
        print(f"        r_stationarity:   {r['stationarity']:.2e}")
        print(f"        r_complementarity:{r['complementarity']:.2e}")
        print(f"        r_dual:           {r['dual']:.2e}")
        print(f"        max:              {result['max_residual']:.2e}")

        if result['certified']:
            print(f"    STATUS: CERTIFIED (KKT satisfied to 1e-6)")
            total_certified += 1
        else:
            print(f"    STATUS: NOT CERTIFIED ({result['status']})")

        print()

    # Summary
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Problems:    {total_problems}")
    print(f"Total Certified:   {total_certified}")
    print(f"Certification Rate: {100*total_certified/total_problems:.1f}%")
    print()

    # Detailed table
    print("RESULTS TABLE:")
    print("-" * 80)
    print(f"{'Problem':<20} {'f(x*)':<15} {'Known f*':<15} {'Gap':<12} {'Status'}")
    print("-" * 80)

    for r in results:
        name = r['name']
        if r.get('error'):
            print(f"{name:<20} {'ERROR':<15} {'-':<15} {'-':<12} ERROR")
            continue

        f_val = f"{r['objective']:.6g}"
        known = f"{r.get('known_f', '-'):.6g}" if 'known_f' in r else '-'
        gap = f"{r.get('gap_from_known', float('nan')):.2e}" if 'gap_from_known' in r else '-'
        status = "CERTIFIED" if r['certified'] else "FAIL"

        print(f"{name:<20} {f_val:<15} {known:<15} {gap:<12} {status}")

    print("-" * 80)
    print()

    # Verification section
    print("VERIFICATION HASHES (for replay certification):")
    print("-" * 80)
    for r in results:
        if r.get('error') or 'hashes' not in r:
            continue
        h = r['hashes']
        print(f"{r['name']:<20} input:{h['input']}... cert:{h['certificate']}...")

    print()
    print("=" * 80)
    print(f"FINAL RESULT: {total_certified}/{total_problems} CERTIFIED")
    print("=" * 80)

    return results


if __name__ == '__main__':
    run_benchmark()
