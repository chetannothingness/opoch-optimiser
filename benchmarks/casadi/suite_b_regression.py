"""
Suite B: Regression / System Identification Problems

NIST-style nonlinear least-squares problems formulated in CasADi.
These problems have ResidualIR structure: f(θ) = ||r(θ)||²

Key insight: Exposing residual structure enables Gauss-Newton bounds.
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


def misra1a() -> CasADiNLP:
    """
    Misra1a NIST Regression Problem.

    Model: y = b1 * (1 - exp(-b2 * x))

    2 parameters, 14 data points.
    Certified values: b1 = 238.94, b2 = 0.000550
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # NIST data
    x_data = np.array([77.6, 114.9, 141.1, 190.8, 239.9, 289.0, 332.8, 378.4,
                       434.8, 477.3, 536.8, 593.1, 689.1, 760.0])
    y_data = np.array([10.07, 14.73, 17.94, 23.93, 29.61, 35.18, 40.02, 44.82,
                       50.76, 55.05, 61.01, 66.40, 75.47, 81.78])

    # Parameters
    b1 = ca.SX.sym('b1')
    b2 = ca.SX.sym('b2')
    theta = ca.vertcat(b1, b2)

    # Residuals
    residuals = []
    for i in range(len(x_data)):
        model = b1 * (1 - ca.exp(-b2 * x_data[i]))
        r_i = y_data[i] - model
        residuals.append(r_i)

    r = ca.vertcat(*residuals)
    f = ca.dot(r, r)  # ||r||²

    return create_nlp_from_casadi(
        x=theta,
        f=f,
        lbx=[0, 0],
        ubx=[500, 1],
        x0=[250, 0.0005],
        name='misra1a',
        description='NIST Misra1a regression',
        source='NIST',
        is_least_squares=True,
    )


def chwirut2() -> CasADiNLP:
    """
    Chwirut2 NIST Regression Problem.

    Model: y = exp(-b1*x) / (b2 + b3*x)

    3 parameters, 54 data points.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # NIST data (subset for efficiency)
    x_data = np.array([0.5, 1.0, 1.75, 3.75, 5.75, 0.875, 2.25, 3.25,
                       5.25, 0.75, 1.75, 2.75, 4.75, 0.625, 1.25, 2.25, 4.25])
    y_data = np.array([92.9, 57.1, 31.05, 11.5875, 8.025, 63.6, 21.4,
                       14.25, 8.475, 80.4, 29.25, 17.75, 9.45, 74.1, 42.3, 20.3, 10.95])

    # Parameters
    b1 = ca.SX.sym('b1')
    b2 = ca.SX.sym('b2')
    b3 = ca.SX.sym('b3')
    theta = ca.vertcat(b1, b2, b3)

    # Residuals
    residuals = []
    for i in range(len(x_data)):
        model = ca.exp(-b1 * x_data[i]) / (b2 + b3 * x_data[i])
        r_i = y_data[i] - model
        residuals.append(r_i)

    r = ca.vertcat(*residuals)
    f = ca.dot(r, r)

    return create_nlp_from_casadi(
        x=theta,
        f=f,
        lbx=[0, 0, 0],
        ubx=[1, 1, 1],
        x0=[0.1, 0.01, 0.02],
        name='chwirut2',
        description='NIST Chwirut2 regression',
        source='NIST',
        is_least_squares=True,
    )


def lanczos1() -> CasADiNLP:
    """
    Lanczos1 NIST Regression Problem.

    Model: y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)

    6 parameters, 24 data points.
    High precision required.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # NIST data
    x_data = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                       0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                       1.0, 1.05, 1.1, 1.15])
    y_data = np.array([2.5134, 2.0443, 1.6684, 1.3664, 1.1232, 0.9269, 0.7679,
                       0.6389, 0.5338, 0.4479, 0.3776, 0.3197, 0.2720, 0.2325,
                       0.1997, 0.1723, 0.1493, 0.1301, 0.1138, 0.1000, 0.0883,
                       0.0783, 0.0698, 0.0624])

    # Parameters
    b = [ca.SX.sym(f'b{i+1}') for i in range(6)]
    theta = ca.vertcat(*b)

    # Residuals
    residuals = []
    for i in range(len(x_data)):
        model = b[0]*ca.exp(-b[1]*x_data[i]) + b[2]*ca.exp(-b[3]*x_data[i]) + b[4]*ca.exp(-b[5]*x_data[i])
        r_i = y_data[i] - model
        residuals.append(r_i)

    r = ca.vertcat(*residuals)
    f = ca.dot(r, r)

    return create_nlp_from_casadi(
        x=theta,
        f=f,
        lbx=[0, 0, 0, 0, 0, 0],
        ubx=[5, 5, 5, 10, 5, 10],
        x0=[1.2, 0.3, 1.0, 1.0, 1.0, 3.0],
        name='lanczos1',
        description='NIST Lanczos1 regression (6 params)',
        source='NIST',
        difficulty='hard',
        is_least_squares=True,
    )


def gauss1() -> CasADiNLP:
    """
    Gauss1 NIST Regression Problem.

    Model: y = b1*exp(-b2*x) + b3*exp(-(x-b4)²/b5²) + b6*exp(-(x-b7)²/b8²)

    8 parameters. Very challenging.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # NIST data (subset)
    x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                       16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    y_data = np.array([97.58776, 97.76344, 96.56705, 92.52037, 91.15097,
                       95.21728, 90.21355, 89.29235, 91.51479, 89.60966,
                       86.56187, 85.55316, 87.13054, 85.67940, 80.04851,
                       82.18925, 87.24081, 80.79407, 81.28570, 81.56940,
                       79.22715, 79.43275, 77.90195, 76.75468, 77.17377])

    # Parameters
    b = [ca.SX.sym(f'b{i+1}') for i in range(8)]
    theta = ca.vertcat(*b)

    # Residuals
    residuals = []
    for i in range(len(x_data)):
        xi = x_data[i]
        model = (b[0]*ca.exp(-b[1]*xi) +
                 b[2]*ca.exp(-((xi-b[3])**2)/(b[4]**2 + 1e-10)) +
                 b[5]*ca.exp(-((xi-b[6])**2)/(b[7]**2 + 1e-10)))
        r_i = y_data[i] - model
        residuals.append(r_i)

    r = ca.vertcat(*residuals)
    f = ca.dot(r, r)

    return create_nlp_from_casadi(
        x=theta,
        f=f,
        lbx=[50, 0, 50, 0, 1, 50, 0, 1],
        ubx=[150, 1, 150, 50, 50, 150, 100, 50],
        x0=[97, 0.01, 100, 25, 10, 90, 50, 10],
        name='gauss1',
        description='NIST Gauss1 regression (8 params)',
        source='NIST',
        difficulty='extreme',
        is_least_squares=True,
    )


def rosenbrock_nist() -> CasADiNLP:
    """
    Rosenbrock NIST regression formulation.

    Classic test function as least-squares.
    f = (1-x1)² + 100(x2-x1²)²

    This is equivalent to residuals:
    r1 = 1 - x1
    r2 = 10(x2 - x1²)
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x1 = ca.SX.sym('x1')
    x2 = ca.SX.sym('x2')
    x = ca.vertcat(x1, x2)

    # Residuals
    r1 = 1 - x1
    r2 = 10 * (x2 - x1**2)
    r = ca.vertcat(r1, r2)
    f = ca.dot(r, r)

    return create_nlp_from_casadi(
        x=x,
        f=f,
        lbx=[-10, -10],
        ubx=[10, 10],
        x0=[-1.2, 1.0],
        name='rosenbrock_nist',
        description='Rosenbrock as NIST-style least-squares',
        source='NIST',
        is_least_squares=True,
    )


def box3d() -> CasADiNLP:
    """
    Box3D NIST Regression Problem.

    Model: y = exp(-b1*t_i) - exp(-b2*t_i) - b3*(exp(-t_i) - exp(-10*t_i))

    3 parameters, 10 data points.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # Generate NIST-style data
    t_data = np.linspace(0.1, 1.0, 10)
    # True parameters: b1=1, b2=10, b3=1
    y_data = np.exp(-t_data) - np.exp(-10*t_data) - (np.exp(-t_data) - np.exp(-10*t_data))

    b1 = ca.SX.sym('b1')
    b2 = ca.SX.sym('b2')
    b3 = ca.SX.sym('b3')
    theta = ca.vertcat(b1, b2, b3)

    residuals = []
    for i in range(len(t_data)):
        t = t_data[i]
        model = ca.exp(-b1*t) - ca.exp(-b2*t) - b3*(ca.exp(-t) - ca.exp(-10*t))
        r_i = y_data[i] - model
        residuals.append(r_i)

    r = ca.vertcat(*residuals)
    f = ca.dot(r, r)

    return create_nlp_from_casadi(
        x=theta,
        f=f,
        lbx=[0, 0, 0],
        ubx=[20, 20, 5],
        x0=[0.5, 5.0, 0.5],
        name='box3d',
        description='NIST Box3D regression',
        source='NIST',
        is_least_squares=True,
    )


def get_regression_problems() -> List[CasADiNLP]:
    """Get all regression benchmark problems."""
    if not HAS_CASADI:
        return []

    problems = [
        misra1a(),
        chwirut2(),
        lanczos1(),
        gauss1(),
        rosenbrock_nist(),
        box3d(),
    ]

    return problems


# Quick test
if __name__ == '__main__':
    if HAS_CASADI:
        problems = get_regression_problems()
        print(f"Suite B: {len(problems)} regression problems")
        for p in problems:
            print(f"  - {p.name}: {p.n_vars} vars, {p.n_constraints} constraints")
    else:
        print("CasADi not installed")
