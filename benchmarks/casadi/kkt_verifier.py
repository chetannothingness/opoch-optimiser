#!/usr/bin/env python3
"""
Correct KKT Verifier for CasADi/IPOPT

Implements proper two-sided KKT decomposition matching IPOPT's formulation.

Key insight: CasADi/IPOPT uses sign convention for bound multipliers:
- Negative multiplier → active LOWER bound
- Positive multiplier → active UPPER bound

References:
- IPOPT documentation on termination criteria
- CasADi GitHub discussions on multiplier conventions
"""

import numpy as np
import casadi as ca
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional


@dataclass
class KKTResiduals:
    """Complete KKT residuals with proper decomposition."""
    r_primal: float          # Constraint violation
    r_dual: float            # Stationarity residual
    r_complementarity: float # Complementarity residual
    r_max: float             # Max of all residuals

    # Detailed breakdown
    primal_x_lb: float       # Variable lower bound violation
    primal_x_ub: float       # Variable upper bound violation
    primal_g_lb: float       # Constraint lower bound violation
    primal_g_ub: float       # Constraint upper bound violation

    compl_x_lb: float        # Variable lower complementarity
    compl_x_ub: float        # Variable upper complementarity
    compl_g_lb: float        # Constraint lower complementarity
    compl_g_ub: float        # Constraint upper complementarity

    certified: bool          # r_max <= epsilon
    epsilon: float           # Tolerance used


def split_multipliers(lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split multipliers into lower/upper parts using IPOPT sign convention.

    IPOPT convention:
    - Negative multiplier → active lower bound (mu_L = max(0, -lam))
    - Positive multiplier → active upper bound (mu_U = max(0, lam))

    Args:
        lam: Combined multiplier vector from IPOPT

    Returns:
        (mu_lower, mu_upper): Split multiplier vectors
    """
    mu_lower = np.maximum(0.0, -lam)
    mu_upper = np.maximum(0.0, lam)
    return mu_lower, mu_upper


def compute_kkt_residuals(
    x_opt: np.ndarray,
    lam_g: np.ndarray,
    lam_x: np.ndarray,
    grad_f: np.ndarray,
    g_val: np.ndarray,
    jac_g: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray,
    ubg: np.ndarray,
    epsilon: float = 1e-6,
) -> KKTResiduals:
    """
    Compute KKT residuals with correct two-sided decomposition.

    KKT conditions for:
        min  f(x)
        s.t. lbx <= x <= ubx
             lbg <= g(x) <= ubg

    Lagrangian:
        L = f(x) + mu_g_L'(lbg - g) + mu_g_U'(g - ubg)
                 + mu_x_L'(lbx - x) + mu_x_U'(x - ubx)

    Stationarity:
        ∇f + J_g'(mu_g_U - mu_g_L) + (mu_x_U - mu_x_L) = 0

    Since IPOPT returns combined multipliers with sign encoding:
        lam_g = mu_g_U - mu_g_L  (positive = upper active, negative = lower active)
        lam_x = mu_x_U - mu_x_L

    Stationarity becomes:
        ∇f + J_g' * lam_g + lam_x = 0
    """
    n_vars = len(x_opt)
    n_cons = len(g_val) if len(g_val) > 0 else 0

    # === PRIMAL FEASIBILITY ===
    # Variable bounds: lbx <= x <= ubx
    slack_x_lb = x_opt - lbx  # >= 0 if feasible
    slack_x_ub = ubx - x_opt  # >= 0 if feasible

    primal_x_lb = np.max(np.maximum(0.0, -slack_x_lb))  # violation of x >= lbx
    primal_x_ub = np.max(np.maximum(0.0, -slack_x_ub))  # violation of x <= ubx

    # Constraint bounds: lbg <= g(x) <= ubg
    if n_cons > 0:
        slack_g_lb = g_val - lbg  # >= 0 if feasible
        slack_g_ub = ubg - g_val  # >= 0 if feasible

        primal_g_lb = np.max(np.maximum(0.0, -slack_g_lb))
        primal_g_ub = np.max(np.maximum(0.0, -slack_g_ub))
    else:
        slack_g_lb = np.array([])
        slack_g_ub = np.array([])
        primal_g_lb = 0.0
        primal_g_ub = 0.0

    r_primal = max(primal_x_lb, primal_x_ub, primal_g_lb, primal_g_ub)

    # === SPLIT MULTIPLIERS ===
    mu_x_L, mu_x_U = split_multipliers(lam_x)

    if n_cons > 0:
        mu_g_L, mu_g_U = split_multipliers(lam_g)
    else:
        mu_g_L = np.array([])
        mu_g_U = np.array([])

    # === STATIONARITY ===
    # ∇f + J_g' * lam_g + lam_x = 0
    # Or equivalently: ∇f + J_g' * (mu_g_U - mu_g_L) + (mu_x_U - mu_x_L) = 0
    stationarity = grad_f.copy()

    if n_cons > 0 and jac_g is not None:
        stationarity += jac_g.T @ lam_g

    stationarity += lam_x  # This is (mu_x_U - mu_x_L)

    r_dual = np.max(np.abs(stationarity))

    # === COMPLEMENTARITY ===
    # For lower bounds: mu_L * slack_L = 0
    # For upper bounds: mu_U * slack_U = 0

    # Variable complementarity
    # Only check where bounds are finite
    compl_x_lb = 0.0
    compl_x_ub = 0.0

    for i in range(n_vars):
        if lbx[i] > -1e15:  # Finite lower bound
            compl_x_lb = max(compl_x_lb, abs(mu_x_L[i] * max(0.0, slack_x_lb[i])))
        if ubx[i] < 1e15:   # Finite upper bound
            compl_x_ub = max(compl_x_ub, abs(mu_x_U[i] * max(0.0, slack_x_ub[i])))

    # Constraint complementarity
    compl_g_lb = 0.0
    compl_g_ub = 0.0

    for i in range(n_cons):
        if lbg[i] > -1e15:  # Finite lower bound
            compl_g_lb = max(compl_g_lb, abs(mu_g_L[i] * max(0.0, slack_g_lb[i])))
        if ubg[i] < 1e15:   # Finite upper bound
            compl_g_ub = max(compl_g_ub, abs(mu_g_U[i] * max(0.0, slack_g_ub[i])))

    r_complementarity = max(compl_x_lb, compl_x_ub, compl_g_lb, compl_g_ub)

    # === OVERALL ===
    r_max = max(r_primal, r_dual, r_complementarity)
    certified = r_max <= epsilon

    return KKTResiduals(
        r_primal=r_primal,
        r_dual=r_dual,
        r_complementarity=r_complementarity,
        r_max=r_max,
        primal_x_lb=primal_x_lb,
        primal_x_ub=primal_x_ub,
        primal_g_lb=primal_g_lb,
        primal_g_ub=primal_g_ub,
        compl_x_lb=compl_x_lb,
        compl_x_ub=compl_x_ub,
        compl_g_lb=compl_g_lb,
        compl_g_ub=compl_g_ub,
        certified=certified,
        epsilon=epsilon,
    )


def verify_casadi_solution(
    sol: Dict[str, Any],
    nlp_dict: Dict[str, ca.MX],
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray,
    ubg: np.ndarray,
    epsilon: float = 1e-6,
) -> KKTResiduals:
    """
    Verify a CasADi solution with proper KKT decomposition.

    Args:
        sol: Solution dict from nlpsol
        nlp_dict: NLP dict with 'x', 'f', 'g'
        lbx, ubx: Variable bounds
        lbg, ubg: Constraint bounds
        epsilon: KKT tolerance

    Returns:
        KKTResiduals with detailed breakdown
    """
    x_opt = np.array(sol['x']).flatten()
    lam_g = np.array(sol['lam_g']).flatten() if sol['lam_g'].shape[0] > 0 else np.array([])
    lam_x = np.array(sol['lam_x']).flatten()

    x_sym = nlp_dict['x']
    f_sym = nlp_dict['f']
    g_sym = nlp_dict.get('g', ca.MX())

    # Build evaluation functions
    grad_f_sym = ca.gradient(f_sym, x_sym)
    grad_f_func = ca.Function('grad_f', [x_sym], [grad_f_sym])

    grad_f = np.array(grad_f_func(x_opt)).flatten()

    if g_sym.shape[0] > 0:
        g_func = ca.Function('g', [x_sym], [g_sym])
        jac_g_sym = ca.jacobian(g_sym, x_sym)
        jac_g_func = ca.Function('jac_g', [x_sym], [jac_g_sym])

        g_val = np.array(g_func(x_opt)).flatten()
        jac_g = np.array(jac_g_func(x_opt))
    else:
        g_val = np.array([])
        jac_g = None

    return compute_kkt_residuals(
        x_opt=x_opt,
        lam_g=lam_g,
        lam_x=lam_x,
        grad_f=grad_f,
        g_val=g_val,
        jac_g=jac_g,
        lbx=lbx,
        ubx=ubx,
        lbg=lbg,
        ubg=ubg,
        epsilon=epsilon,
    )


def certify_or_repair(
    nlp_dict: Dict[str, ca.MX],
    x0: np.ndarray,
    lbx: np.ndarray,
    ubx: np.ndarray,
    lbg: np.ndarray,
    ubg: np.ndarray,
    epsilon: float = 1e-6,
    max_rounds: int = 3,
) -> Tuple[Dict[str, Any], KKTResiduals, int, Dict[str, Any]]:
    """
    Deterministic repair loop: drive SUCCESS to CERTIFIED.

    Rounds:
    0: Default IPOPT options
    1: Disable scaling, tighten tolerances, warmstart
    2: Additional numerical stabilization
    3: Problem-side rescaling (if needed)

    Returns:
        (solution, residuals, round_passed, solver_options_used)
    """
    rounds_config = [
        # Round 0: Default
        {
            'ipopt.tol': 1e-8,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
        },
        # Round 1: No scaling, strict tolerances
        {
            'ipopt.tol': 1e-10,
            'ipopt.constr_viol_tol': 1e-10,
            'ipopt.dual_inf_tol': 1e-10,
            'ipopt.compl_inf_tol': 1e-10,
            'ipopt.acceptable_tol': 1e-10,
            'ipopt.acceptable_iter': 0,
            'ipopt.nlp_scaling_method': 'none',
            'ipopt.max_iter': 6000,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes',
        },
        # Round 2: Numerical stabilization
        {
            'ipopt.tol': 1e-10,
            'ipopt.constr_viol_tol': 1e-10,
            'ipopt.dual_inf_tol': 1e-10,
            'ipopt.compl_inf_tol': 1e-10,
            'ipopt.acceptable_tol': 1e-10,
            'ipopt.acceptable_iter': 0,
            'ipopt.nlp_scaling_method': 'none',
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.max_iter': 10000,
            'ipopt.print_level': 0,
            'ipopt.sb': 'yes',
            'ipopt.warm_start_init_point': 'yes',
        },
    ]

    sol = None
    x_warm = x0.copy()
    lam_g_warm = None
    lam_x_warm = None

    for round_idx in range(min(max_rounds, len(rounds_config))):
        opts = {'print_time': False}
        opts.update(rounds_config[round_idx])

        solver = ca.nlpsol('solver', 'ipopt', nlp_dict, opts)

        # Solve with warmstart if available
        solve_args = {
            'x0': x_warm,
            'lbx': lbx, 'ubx': ubx,
            'lbg': lbg, 'ubg': ubg,
        }

        if round_idx > 0 and lam_g_warm is not None:
            solve_args['lam_g0'] = lam_g_warm
        if round_idx > 0 and lam_x_warm is not None:
            solve_args['lam_x0'] = lam_x_warm

        try:
            sol = solver(**solve_args)

            # Verify
            residuals = verify_casadi_solution(
                sol, nlp_dict, lbx, ubx, lbg, ubg, epsilon
            )

            if residuals.certified:
                return sol, residuals, round_idx, rounds_config[round_idx]

            # Warmstart for next round
            x_warm = np.array(sol['x']).flatten()
            lam_g_warm = np.array(sol['lam_g']).flatten()
            lam_x_warm = np.array(sol['lam_x']).flatten()

        except Exception as e:
            print(f"Round {round_idx} failed: {e}")
            continue

    # Return best effort
    if sol is not None:
        residuals = verify_casadi_solution(
            sol, nlp_dict, lbx, ubx, lbg, ubg, epsilon
        )
        return sol, residuals, -1, rounds_config[-1]

    raise RuntimeError("All repair rounds failed")
