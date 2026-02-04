"""
Suite C: Mixed-Integer Nonlinear Programming (MINLP)

Problems with both continuous and discrete variables.
These require integer branching + continuous optimization.
"""

from typing import List
import numpy as np

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False
    ca = None

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from opoch_optimizer.casadi.nlp_contract import CasADiNLP, NLPBounds, create_nlp_from_casadi


def ex1223a() -> CasADiNLP:
    """
    ex1223a from MINLPLib.

    A small but challenging MINLP test problem.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # Continuous variables
    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')

    # Binary variables
    y1 = ca.SX.sym('y1')
    y2 = ca.SX.sym('y2')

    z = ca.vertcat(x1, x2, y1, y2)

    # Objective
    f = -2*x1 - 3*x2 + y1 + y2

    # Constraints
    g1 = x1 + x2  # <= 3
    g2 = x1 + 2*x2  # <= 4
    g3 = x1 - y1  # <= 0 (x1 <= y1)
    g4 = x2 - y2  # <= 0 (x2 <= y2)
    g = ca.vertcat(g1, g2, g3, g4)

    nlp = create_nlp_from_casadi(
        x=z,
        f=f,
        g=g,
        lbx=[0, 0, 0, 0],
        ubx=[3, 3, 1, 1],
        lbg=[-ca.inf, -ca.inf, -ca.inf, -ca.inf],
        ubg=[3, 4, 0, 0],
        x0=[1, 1, 1, 1],
        name='ex1223a',
        description='MINLPLib ex1223a',
        source='MINLPLib',
    )
    nlp.integer_vars = [2, 3]  # y1, y2 are binary
    return nlp


def facility_location(n_facilities: int = 3, n_customers: int = 5) -> CasADiNLP:
    """
    Facility Location Problem (MINLP formulation).

    Decide which facilities to open and how to allocate customers.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    np.random.seed(42)  # Deterministic

    # Random costs and demands
    fixed_costs = np.random.uniform(10, 50, n_facilities)
    transport_costs = np.random.uniform(1, 10, (n_facilities, n_customers))
    demands = np.random.uniform(1, 5, n_customers)
    capacities = np.random.uniform(10, 20, n_facilities)

    # Variables
    # y_i: binary, facility i open
    # x_ij: continuous, fraction of customer j served by facility i
    y = [ca.SX.sym(f'y_{i}') for i in range(n_facilities)]
    x = [[ca.SX.sym(f'x_{i}_{j}') for j in range(n_customers)] for i in range(n_facilities)]

    all_vars = y + [x_ij for x_row in x for x_ij in x_row]
    z = ca.vertcat(*all_vars)

    # Objective: minimize fixed + transport costs
    J = 0
    for i in range(n_facilities):
        J += fixed_costs[i] * y[i]
        for j in range(n_customers):
            J += transport_costs[i, j] * x[i][j] * demands[j]

    # Constraints
    g_list = []
    g_lb = []
    g_ub = []

    # Each customer fully served
    for j in range(n_customers):
        demand_met = sum(x[i][j] for i in range(n_facilities))
        g_list.append(demand_met)
        g_lb.append(1.0)
        g_ub.append(1.0)

    # Capacity constraints (only serve if open)
    for i in range(n_facilities):
        total_served = sum(x[i][j] * demands[j] for j in range(n_customers))
        g_list.append(total_served - capacities[i] * y[i])
        g_lb.append(-ca.inf)
        g_ub.append(0)

    g = ca.vertcat(*g_list)

    # Bounds
    lbx = [0] * n_facilities + [0] * (n_facilities * n_customers)
    ubx = [1] * n_facilities + [1] * (n_facilities * n_customers)
    x0 = [1] * n_facilities + [1/n_facilities] * (n_facilities * n_customers)

    nlp = create_nlp_from_casadi(
        x=z,
        f=J,
        g=g,
        lbx=lbx,
        ubx=ubx,
        lbg=g_lb,
        ubg=g_ub,
        x0=x0,
        name=f'facility_location_{n_facilities}x{n_customers}',
        description=f'Facility location ({n_facilities} facilities, {n_customers} customers)',
        source='Custom',
    )
    nlp.integer_vars = list(range(n_facilities))  # y_i are binary
    return nlp


def knapsack_nonlinear(n_items: int = 5) -> CasADiNLP:
    """
    Nonlinear Knapsack Problem.

    Binary selection with nonlinear value function.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    np.random.seed(123)  # Deterministic

    # Item properties
    weights = np.random.uniform(1, 10, n_items)
    values = np.random.uniform(5, 20, n_items)
    capacity = sum(weights) * 0.5

    # Binary selection variables
    x = [ca.SX.sym(f'x_{i}') for i in range(n_items)]
    z = ca.vertcat(*x)

    # Nonlinear objective: value with diminishing returns
    # f = -Σ v_i * sqrt(x_i) (maximizing, so negate)
    J = 0
    for i in range(n_items):
        J -= values[i] * ca.sqrt(x[i] + 0.01)  # Small constant for smoothness

    # Weight constraint
    total_weight = sum(weights[i] * x[i] for i in range(n_items))
    g = total_weight

    nlp = create_nlp_from_casadi(
        x=z,
        f=J,
        g=g,
        lbx=[0] * n_items,
        ubx=[1] * n_items,
        lbg=[-ca.inf],
        ubg=[capacity],
        x0=[0.5] * n_items,
        name=f'knapsack_nonlinear_{n_items}',
        description=f'Nonlinear knapsack ({n_items} items)',
        source='Custom',
    )
    nlp.integer_vars = list(range(n_items))
    return nlp


def pooling_simple() -> CasADiNLP:
    """
    Simplified Pooling Problem.

    A classic MINLP benchmark from process engineering.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # Flow variables (continuous)
    x12 = ca.SX.sym('x12')  # Source 1 to Pool
    x22 = ca.SX.sym('x22')  # Source 2 to Pool
    x23 = ca.SX.sym('x23')  # Pool to Product
    y1 = ca.SX.sym('y1')    # Direct flow 1
    p = ca.SX.sym('p')       # Pool quality

    z = ca.vertcat(x12, x22, x23, y1, p)

    # Costs and prices
    c1 = 2  # Source 1 cost
    c2 = 3  # Source 2 cost
    s1 = 10  # Product price

    # Objective: maximize profit
    J = -(s1 * x23 - c1 * x12 - c2 * x22)

    # Quality constraints
    q1 = 1  # Source 1 quality
    q2 = 3  # Source 2 quality
    q_max = 2  # Max product quality

    # Pool balance: inflow = outflow
    g1 = x12 + x22 - x23

    # Quality balance in pool (bilinear)
    g2 = q1 * x12 + q2 * x22 - p * (x12 + x22)

    # Product quality constraint
    g3 = p - q_max

    g = ca.vertcat(g1, g2, g3)

    return create_nlp_from_casadi(
        x=z,
        f=J,
        g=g,
        lbx=[0, 0, 0, 0, 0],
        ubx=[10, 10, 10, 10, 5],
        lbg=[0, 0, -ca.inf],
        ubg=[0, 0, 0],
        x0=[5, 5, 10, 0, 2],
        name='pooling_simple',
        description='Simplified pooling problem',
        source='Process Engineering',
    )


def batch_scheduling() -> CasADiNLP:
    """
    Simple Batch Scheduling Problem.

    Schedule batch operations with sequencing constraints.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    n_jobs = 3
    processing_times = [2, 3, 1]
    due_dates = [5, 7, 4]

    # Variables: start times and binary sequencing
    s = [ca.SX.sym(f's_{i}') for i in range(n_jobs)]  # Start times
    # Binary: y_ij = 1 if job i before job j
    y = {}
    for i in range(n_jobs):
        for j in range(i+1, n_jobs):
            y[(i,j)] = ca.SX.sym(f'y_{i}_{j}')

    all_vars = s + list(y.values())
    z = ca.vertcat(*all_vars)

    # Objective: minimize total tardiness
    J = 0
    for i in range(n_jobs):
        completion = s[i] + processing_times[i]
        tardiness = ca.fmax(0, completion - due_dates[i])
        J += tardiness

    # Constraints: no overlap (big-M formulation)
    M = 100
    g_list = []
    g_lb = []
    g_ub = []

    for i in range(n_jobs):
        for j in range(i+1, n_jobs):
            # Either i before j or j before i
            # s_j >= s_i + p_i - M*(1-y_ij)
            g_list.append(s[j] - s[i] - processing_times[i] + M*(1 - y[(i,j)]))
            g_lb.append(0)
            g_ub.append(ca.inf)

            # s_i >= s_j + p_j - M*y_ij
            g_list.append(s[i] - s[j] - processing_times[j] + M*y[(i,j)])
            g_lb.append(0)
            g_ub.append(ca.inf)

    g = ca.vertcat(*g_list)

    # Bounds
    n_y = len(y)
    lbx = [0] * n_jobs + [0] * n_y
    ubx = [20] * n_jobs + [1] * n_y
    x0 = [0, 2, 5] + [0.5] * n_y

    nlp = create_nlp_from_casadi(
        x=z,
        f=J,
        g=g,
        lbx=lbx,
        ubx=ubx,
        lbg=g_lb,
        ubg=g_ub,
        x0=x0,
        name='batch_scheduling',
        description='Batch scheduling with sequencing',
        source='Custom',
    )
    nlp.integer_vars = list(range(n_jobs, n_jobs + n_y))
    return nlp


def get_minlp_problems() -> List[CasADiNLP]:
    """Get all MINLP benchmark problems."""
    if not HAS_CASADI:
        return []

    problems = [
        ex1223a(),
        facility_location(n_facilities=3, n_customers=4),
        knapsack_nonlinear(n_items=4),
        pooling_simple(),
        batch_scheduling(),
    ]

    return problems


if __name__ == '__main__':
    if HAS_CASADI:
        problems = get_minlp_problems()
        print(f"Suite C: {len(problems)} MINLP problems")
        for p in problems:
            print(f"  - {p.name}: {p.n_vars} vars ({len(p.integer_vars)} integer), {p.n_constraints} constraints")
    else:
        print("CasADi not installed")
