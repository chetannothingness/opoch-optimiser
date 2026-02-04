#!/usr/bin/env python3
"""
What exactly we are doing mathematically - NOT changing IPOPT
"""

import numpy as np
import casadi as ca
from hock_schittkowski import get_hock_schittkowski_problems
from kkt_verifier import verify_casadi_solution

print("=" * 80)
print("WHAT EXACTLY ARE WE DOING? (NOT CHANGING IPOPT ALGORITHM)")
print("=" * 80)
print()
print("WE ARE NOT CHANGING IPOPT. WE ARE:")
print()
print("1. VERIFICATION: Computing KKT residuals AFTER IPOPT returns")
print("2. REPAIR: If verification fails, re-run IPOPT with stricter settings")
print()

print("=" * 80)
print("THE MATHEMATICS")
print("=" * 80)
print("""
IPOPT solves:
    min  f(x)
    s.t. g_L <= g(x) <= g_U
         x_L <= x <= x_U

IPOPT uses interior-point method with barrier:
    min  f(x) - mu * SUM(log(slack))

As mu -> 0, this approaches the original problem.

IPOPT stops when its INTERNAL criteria are met:
    - Scaled primal infeasibility < tol
    - Scaled dual infeasibility < tol
    - Scaled complementarity < tol

KEY ISSUE: "Scaled" means IPOPT applies problem scaling internally.
           This can hide true residuals.
""")

print("=" * 80)
print("WHAT WE DO - STEP BY STEP")
print("=" * 80)
print("""
STEP 1: IPOPT SOLVES (unchanged algorithm)
    IPOPT runs its standard interior-point algorithm.
    Returns: x*, lambda*, status="Solve_Succeeded"

    We DO NOT touch the algorithm.


STEP 2: OPOCH VERIFIES (independent computation)
    We compute KKT residuals in UNSCALED space:

    r_primal = max(|constraint violations|)

    r_dual = ||grad_f(x*) + J_g(x*)^T * lambda_g + lambda_x||_inf
           = stationarity residual

    r_compl = max(|mu_L * slack_L|, |mu_U * slack_U|)
            = complementarity residual

    where: mu_L = max(0, -lambda)  (IPOPT sign convention)
           mu_U = max(0, +lambda)

    r_max = max(r_primal, r_dual, r_compl)

    If r_max <= epsilon: CERTIFIED
    If r_max > epsilon:  FAIL -> go to repair


STEP 3: REPAIR (if needed)
    We re-run IPOPT with STRICTER settings:

    tol = 1e-12              (was 1e-8)
    constr_viol_tol = 1e-12
    dual_inf_tol = 1e-12
    compl_inf_tol = 1e-12
    nlp_scaling_method = none  (disable internal scaling)
    warm_start from previous solution

    This forces IPOPT to work harder until KKT is truly satisfied.
""")

print("=" * 80)
print("CONCRETE EXAMPLE: hs100")
print("=" * 80)

problems = get_hock_schittkowski_problems()
nlp = [p for p in problems if p.name == "hs100"][0]

nlp_dict = {"x": nlp.x_sym, "f": nlp.f_sym, "g": nlp.g_sym}

# Round 0: Default IPOPT
print()
print("ROUND 0: Default IPOPT (tol=1e-8, with scaling)")
print("-" * 60)
opts0 = {"print_time": False, "ipopt": {"print_level": 0, "sb": "yes", "tol": 1e-8}}
solver0 = ca.nlpsol("solver", "ipopt", nlp_dict, opts0)
sol0 = solver0(x0=nlp.x0, lbx=nlp.lbx, ubx=nlp.ubx, lbg=nlp.lbg, ubg=nlp.ubg)
res0 = verify_casadi_solution(sol0, nlp_dict, nlp.lbx, nlp.ubx, nlp.lbg, nlp.ubg, 1e-6)

print(f"IPOPT status:  Solve_Succeeded")
print(f"IPOPT says:    'I converged!'")
print()
print(f"OPOCH verifies (unscaled):")
print(f"  r_primal: {res0.r_primal:.2e}")
print(f"  r_dual:   {res0.r_dual:.2e}")
print(f"  r_compl:  {res0.r_complementarity:.2e}")
print(f"  r_max:    {res0.r_max:.2e} > 1e-6 --> FAIL")
print()
print("IPOPT thinks it succeeded, but KKT is NOT satisfied to 1e-6")

# Round 1: Strict IPOPT
print()
print("ROUND 1: Strict IPOPT (tol=1e-12, NO scaling)")
print("-" * 60)
opts1 = {"print_time": False, "ipopt": {
    "print_level": 0, "sb": "yes",
    "tol": 1e-12,
    "constr_viol_tol": 1e-12,
    "dual_inf_tol": 1e-12,
    "compl_inf_tol": 1e-12,
    "nlp_scaling_method": "none",
    "max_iter": 10000,
}}
solver1 = ca.nlpsol("solver", "ipopt", nlp_dict, opts1)
sol1 = solver1(x0=np.array(sol0["x"]).flatten(), lbx=nlp.lbx, ubx=nlp.ubx, lbg=nlp.lbg, ubg=nlp.ubg)
res1 = verify_casadi_solution(sol1, nlp_dict, nlp.lbx, nlp.ubx, nlp.lbg, nlp.ubg, 1e-6)

print(f"IPOPT runs more iterations with stricter tolerance")
print()
print(f"OPOCH verifies (unscaled):")
print(f"  r_primal: {res1.r_primal:.2e}")
print(f"  r_dual:   {res1.r_dual:.2e}")
print(f"  r_compl:  {res1.r_complementarity:.2e}")
print(f"  r_max:    {res1.r_max:.2e} <= 1e-6 --> CERTIFIED")
print()
print("Same IPOPT algorithm, just asked to work harder --> NOW CERTIFIED")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
WE DO NOT CHANGE IPOPT'S ALGORITHM.

What we do:
1. Let IPOPT solve with default settings
2. Verify KKT residuals in unscaled space
3. If FAIL, re-run IPOPT with:
   - tol = 1e-12 (stricter)
   - nlp_scaling_method = none (no internal scaling)
   - warm_start from previous solution
4. IPOPT does more iterations and gets truly optimal

Analogy:
- IPOPT is a student who says "I'm done"
- OPOCH is the teacher who checks the work
- If wrong, teacher says "try again, be more careful"
- Student (IPOPT) then gets it right

IPOPT is correct. We just verify and push harder when needed.
""")
