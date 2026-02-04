"""
Hock-Schittkowski Test Problems

The classic Hock & Schittkowski model collection is the mandatory test scenario
for Nonlinear Programming codes. These are the REAL benchmark problems used by
IPOPT, CasADi, and all serious NLP solvers.

Reference: W. Hock, K. Schittkowski, "Test Examples for Nonlinear Programming Codes",
Lecture Notes in Economics and Mathematical Systems, Vol. 187, Springer 1981.

NO SHORTCUTS. These are the exact formulations with known optimal solutions.
"""

from typing import List, Tuple, Optional
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


def hs071() -> CasADiNLP:
    """
    HS071 - Hock-Schittkowski Problem 71.

    THE STANDARD IPOPT TEST PROBLEM.

    min  x1*x4*(x1 + x2 + x3) + x3
    s.t. x1*x2*x3*x4 >= 25
         x1² + x2² + x3² + x4² = 40
         1 <= x1, x2, x3, x4 <= 5

    Optimal: x* = (1.0, 4.74299963, 3.82114998, 1.37940829)
    f* = 17.0140173
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x4 = ca.SX.sym('x4')
    x = ca.vertcat(x1, x2, x3, x4)

    # Objective
    f = x1 * x4 * (x1 + x2 + x3) + x3

    # Constraints
    g1 = x1 * x2 * x3 * x4  # >= 25
    g2 = x1**2 + x2**2 + x3**2 + x4**2  # = 40
    g = ca.vertcat(g1, g2)

    return create_nlp_from_casadi(
        x=x,
        f=f,
        g=g,
        lbx=[1, 1, 1, 1],
        ubx=[5, 5, 5, 5],
        lbg=[25, 40],
        ubg=[ca.inf, 40],
        x0=[1, 5, 5, 1],
        name='hs071',
        description='Hock-Schittkowski 71 - IPOPT standard test',
        source='Hock-Schittkowski',
        difficulty='medium',
    )


def hs076() -> CasADiNLP:
    """
    HS076 - Hock-Schittkowski Problem 76.

    Quadratic objective with linear constraints only.
    Origin: Murtagh, Sargent.

    min  f(x) = x1² + 0.5*x2² + x3² + 0.5*x4² - x1*x3 + x3*x4 - x1 - 3*x2 + x3 - x4
    s.t. x1 + 2*x2 + x3 + x4 <= 5
         3*x1 + x2 + 2*x3 - x4 <= 4
         -x2 + 4*x3 <= 1.5
         0 <= xi

    Optimal: x* = (0.0, 0.0, 0.5, 0.0), f* = -4.681818...
    (Actually f* ≈ -4.6818181818)
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x4 = ca.SX.sym('x4')
    x = ca.vertcat(x1, x2, x3, x4)

    # Objective (quadratic)
    f = x1**2 + 0.5*x2**2 + x3**2 + 0.5*x4**2 - x1*x3 + x3*x4 - x1 - 3*x2 + x3 - x4

    # Linear constraints
    g1 = x1 + 2*x2 + x3 + x4  # <= 5
    g2 = 3*x1 + x2 + 2*x3 - x4  # <= 4
    g3 = -x2 + 4*x3  # <= 1.5
    g = ca.vertcat(g1, g2, g3)

    return create_nlp_from_casadi(
        x=x,
        f=f,
        g=g,
        lbx=[0, 0, 0, 0],
        ubx=[ca.inf, ca.inf, ca.inf, ca.inf],
        lbg=[-ca.inf, -ca.inf, -ca.inf],
        ubg=[5, 4, 1.5],
        x0=[0.5, 0.5, 0.5, 0.5],
        name='hs076',
        description='Hock-Schittkowski 76 - QP with linear constraints',
        source='Hock-Schittkowski',
        difficulty='easy',
        is_quadratic=True,
    )


def hs035() -> CasADiNLP:
    """
    HS035 - Hock-Schittkowski Problem 35.

    min  9 - 8*x1 - 6*x2 - 4*x3 + 2*x1² + 2*x2² + x3² + 2*x1*x2 + 2*x1*x3
    s.t. x1 + x2 + 2*x3 <= 3
         0 <= xi

    Optimal: x* = (4/3, 7/9, 4/9), f* = 1/9
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x = ca.vertcat(x1, x2, x3)

    f = 9 - 8*x1 - 6*x2 - 4*x3 + 2*x1**2 + 2*x2**2 + x3**2 + 2*x1*x2 + 2*x1*x3
    g = x1 + x2 + 2*x3

    return create_nlp_from_casadi(
        x=x,
        f=f,
        g=g,
        lbx=[0, 0, 0],
        ubx=[ca.inf, ca.inf, ca.inf],
        lbg=[-ca.inf],
        ubg=[3],
        x0=[0.5, 0.5, 0.5],
        name='hs035',
        description='Hock-Schittkowski 35',
        source='Hock-Schittkowski',
        difficulty='easy',
        is_quadratic=True,
    )


def hs038() -> CasADiNLP:
    """
    HS038 - Hock-Schittkowski Problem 38.

    Unconstrained with bounds only.

    min  100*(x2 - x1²)² + (1 - x1)² + 90*(x4 - x3²)² + (1 - x3)²
         + 10.1*((x2 - 1)² + (x4 - 1)²) + 19.8*(x2 - 1)*(x4 - 1)
    s.t. -10 <= xi <= 10

    Optimal: x* = (1, 1, 1, 1), f* = 0
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x4 = ca.SX.sym('x4')
    x = ca.vertcat(x1, x2, x3, x4)

    f = (100*(x2 - x1**2)**2 + (1 - x1)**2 +
         90*(x4 - x3**2)**2 + (1 - x3)**2 +
         10.1*((x2 - 1)**2 + (x4 - 1)**2) +
         19.8*(x2 - 1)*(x4 - 1))

    return create_nlp_from_casadi(
        x=x,
        f=f,
        lbx=[-10, -10, -10, -10],
        ubx=[10, 10, 10, 10],
        x0=[-3, -1, -3, -1],
        name='hs038',
        description='Hock-Schittkowski 38 - Extended Rosenbrock',
        source='Hock-Schittkowski',
        difficulty='medium',
    )


def hs044() -> CasADiNLP:
    """
    HS044 - Hock-Schittkowski Problem 44.

    min  x1 - x2 - x3 - x1*x3 + x1*x4 + x2*x3 - x2*x4
    s.t. x1 + 2*x2 <= 8
         4*x1 + x2 <= 12
         3*x1 + 4*x2 <= 12
         2*x3 + x4 <= 8
         x3 + 2*x4 <= 8
         x3 + x4 <= 5
         0 <= xi

    Optimal: x* = (0, 3, 0, 4), f* = -15
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x4 = ca.SX.sym('x4')
    x = ca.vertcat(x1, x2, x3, x4)

    f = x1 - x2 - x3 - x1*x3 + x1*x4 + x2*x3 - x2*x4

    g1 = x1 + 2*x2
    g2 = 4*x1 + x2
    g3 = 3*x1 + 4*x2
    g4 = 2*x3 + x4
    g5 = x3 + 2*x4
    g6 = x3 + x4
    g = ca.vertcat(g1, g2, g3, g4, g5, g6)

    return create_nlp_from_casadi(
        x=x,
        f=f,
        g=g,
        lbx=[0, 0, 0, 0],
        ubx=[ca.inf, ca.inf, ca.inf, ca.inf],
        lbg=[-ca.inf]*6,
        ubg=[8, 12, 12, 8, 8, 5],
        x0=[1, 1, 1, 1],
        name='hs044',
        description='Hock-Schittkowski 44',
        source='Hock-Schittkowski',
        difficulty='easy',
    )


def hs065() -> CasADiNLP:
    """
    HS065 - Hock-Schittkowski Problem 65.

    min  (x1 - x2)² + (x1 + x2 - 10)²/9 + (x3 - 5)²
    s.t. x1² + x2² + x3² <= 48
         -4.5 <= x1 <= 4.5
         -4.5 <= x2 <= 4.5
         -5 <= x3 <= 5

    Optimal: x* = (3.650461821, 3.65046169, 4.6204170507), f* = 0.9535288567
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x3 = ca.SX.sym('x3')
    x = ca.vertcat(x1, x2, x3)

    f = (x1 - x2)**2 + ((x1 + x2 - 10)**2)/9 + (x3 - 5)**2
    g = x1**2 + x2**2 + x3**2

    return create_nlp_from_casadi(
        x=x,
        f=f,
        g=g,
        lbx=[-4.5, -4.5, -5],
        ubx=[4.5, 4.5, 5],
        lbg=[-ca.inf],
        ubg=[48],
        x0=[-5, 5, 0],
        name='hs065',
        description='Hock-Schittkowski 65',
        source='Hock-Schittkowski',
        difficulty='medium',
    )


def hs100() -> CasADiNLP:
    """
    HS100 - Hock-Schittkowski Problem 100.

    A 7-variable problem with 4 constraints.

    min  (x1-10)² + 5*(x2-12)² + x3⁴ + 3*(x4-11)² + 10*x5⁶
         + 7*x6² + x7⁴ - 4*x6*x7 - 10*x6 - 8*x7

    s.t. 2*x1² + 3*x2⁴ + x3 + 4*x4² + 5*x5 <= 127
         7*x1 + 3*x2 + 10*x3² + x4 - x5 <= 282
         23*x1 + x2² + 6*x6² - 8*x7 <= 196
         4*x1² + x2² - 3*x1*x2 + 2*x3² + 5*x6 - 11*x7 <= 0
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x = [ca.SX.sym(f'x{i+1}') for i in range(7)]
    xv = ca.vertcat(*x)

    f = ((x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 +
         10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6])

    g1 = 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4]
    g2 = 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4]
    g3 = 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6]
    g4 = 4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]
    g = ca.vertcat(g1, g2, g3, g4)

    return create_nlp_from_casadi(
        x=xv,
        f=f,
        g=g,
        lbx=[-ca.inf]*7,
        ubx=[ca.inf]*7,
        lbg=[-ca.inf, -ca.inf, -ca.inf, -ca.inf],
        ubg=[127, 282, 196, 0],
        x0=[1, 2, 0, 4, 0, 1, 1],
        name='hs100',
        description='Hock-Schittkowski 100 - 7 vars, 4 constraints',
        source='Hock-Schittkowski',
        difficulty='hard',
    )


def rosenbrock_casadi(n: int = 2) -> CasADiNLP:
    """
    Rosenbrock function in n dimensions.

    min  Σ [100*(x_{i+1} - x_i²)² + (1 - x_i)²]

    Optimal: x* = (1, 1, ..., 1), f* = 0
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x = [ca.SX.sym(f'x{i}') for i in range(n)]
    xv = ca.vertcat(*x)

    f = 0
    for i in range(n-1):
        f += 100*(x[i+1] - x[i]**2)**2 + (1 - x[i])**2

    return create_nlp_from_casadi(
        x=xv,
        f=f,
        lbx=[-10]*n,
        ubx=[10]*n,
        x0=[-1.2 if i % 2 == 0 else 1.0 for i in range(n)],
        name=f'rosenbrock_{n}d',
        description=f'Rosenbrock function ({n}D)',
        source='CasADi Tutorial',
        difficulty='easy' if n <= 2 else 'medium',
    )


def get_hock_schittkowski_problems() -> List[CasADiNLP]:
    """Get all Hock-Schittkowski benchmark problems."""
    if not HAS_CASADI:
        return []

    problems = [
        hs035(),
        hs038(),
        hs044(),
        hs065(),
        hs071(),  # THE standard IPOPT test
        hs076(),
        hs100(),
        rosenbrock_casadi(2),
        rosenbrock_casadi(5),
        rosenbrock_casadi(10),
    ]

    return problems


# Known optimal solutions for verification
KNOWN_OPTIMA = {
    'hs035': {'x': [4/3, 7/9, 4/9], 'f': 1/9},
    'hs038': {'x': [1, 1, 1, 1], 'f': 0},
    'hs044': {'x': [0, 3, 0, 4], 'f': -15},
    'hs065': {'x': [3.650461821, 3.65046169, 4.6204170507], 'f': 0.9535288567},
    'hs071': {'x': [1.0, 4.74299963, 3.82114998, 1.37940829], 'f': 17.0140173},
    'hs076': {'x': [0.0, 0.0, 0.5, 0.0], 'f': -4.681818},
    'rosenbrock_2d': {'x': [1, 1], 'f': 0},
    'rosenbrock_5d': {'x': [1, 1, 1, 1, 1], 'f': 0},
    'rosenbrock_10d': {'x': [1]*10, 'f': 0},
}


if __name__ == '__main__':
    if HAS_CASADI:
        problems = get_hock_schittkowski_problems()
        print(f"Hock-Schittkowski Problems: {len(problems)}")
        print()
        for p in problems:
            opt = KNOWN_OPTIMA.get(p.name, {})
            f_opt = opt.get('f', 'unknown')
            print(f"  {p.name}:")
            print(f"    Variables: {p.n_vars}, Constraints: {p.n_constraints}")
            print(f"    Known optimal f* = {f_opt}")
    else:
        print("CasADi not installed")
