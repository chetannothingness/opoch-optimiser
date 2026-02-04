"""
Suite A: Industrial NLP Problems

Optimal control, robotics, and engineering design problems.
These represent real-world CasADi use cases.
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


def van_der_pol_ocp(N: int = 20) -> CasADiNLP:
    """
    Van der Pol Oscillator Optimal Control Problem.

    Minimize: ∫₀ᵀ (x₁² + x₂² + u²) dt
    Subject to:
        ẋ₁ = (1-x₂²)x₁ - x₂ + u
        ẋ₂ = x₁
        x(0) = [0, 1]

    Discretized via direct multiple shooting.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    T = 10.0  # Time horizon
    dt = T / N

    # Decision variables: [x1_0, x2_0, u_0, x1_1, x2_1, u_1, ...]
    n_states = 2
    n_controls = 1
    n_vars_per_step = n_states + n_controls

    # All decision variables
    w = []
    w_lb = []
    w_ub = []
    w0 = []

    # Constraints
    g = []
    g_lb = []
    g_ub = []

    # Objective
    J = 0

    # Initial state
    x1_init = ca.SX.sym('x1_0')
    x2_init = ca.SX.sym('x2_0')
    w += [x1_init, x2_init]
    w_lb += [0, 1]  # Fixed initial condition
    w_ub += [0, 1]
    w0 += [0, 1]

    x_prev = ca.vertcat(x1_init, x2_init)

    for k in range(N):
        # Control at step k
        u_k = ca.SX.sym(f'u_{k}')
        w.append(u_k)
        w_lb.append(-1.0)
        w_ub.append(1.0)
        w0.append(0.0)

        # Stage cost
        J += dt * (x_prev[0]**2 + x_prev[1]**2 + u_k**2)

        # Euler integration (simple for demo)
        x1_dot = (1 - x_prev[1]**2) * x_prev[0] - x_prev[1] + u_k
        x2_dot = x_prev[0]
        x_next_euler = x_prev + dt * ca.vertcat(x1_dot, x2_dot)

        if k < N - 1:
            # State at next step
            x1_next = ca.SX.sym(f'x1_{k+1}')
            x2_next = ca.SX.sym(f'x2_{k+1}')
            w += [x1_next, x2_next]
            w_lb += [-10, -10]
            w_ub += [10, 10]
            w0 += [0, 0]

            # Continuity constraint
            x_next = ca.vertcat(x1_next, x2_next)
            g.append(x_next - x_next_euler)
            g_lb += [0, 0]
            g_ub += [0, 0]

            x_prev = x_next

    # Build NLP
    w_vec = ca.vertcat(*w)
    g_vec = ca.vertcat(*g) if g else ca.SX([])

    return create_nlp_from_casadi(
        x=w_vec,
        f=J,
        g=g_vec,
        lbx=w_lb,
        ubx=w_ub,
        lbg=g_lb,
        ubg=g_ub,
        x0=w0,
        name='van_der_pol_ocp',
        description=f'Van der Pol OCP (N={N})',
        source='CasADi Tutorial',
        is_ocp=True,
    )


def rocket_landing(N: int = 30) -> CasADiNLP:
    """
    Simplified Rocket Landing Problem.

    Minimize: Final velocity magnitude
    Subject to:
        ẋ = v
        v̇ = -g + u/m
        x(0) = h₀, v(0) = v₀
        x(T) = 0, |u| ≤ u_max
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    T = 5.0  # Time horizon
    dt = T / N
    g = 9.81  # Gravity
    m = 1000  # Mass (constant for simplicity)
    u_max = 15000  # Max thrust

    h0 = 100  # Initial height
    v0 = -20  # Initial velocity (downward)

    # Decision variables
    w = []
    w_lb = []
    w_ub = []
    w0 = []
    g_con = []
    g_lb = []
    g_ub = []

    J = 0

    # Initial state: [h, v]
    h = ca.SX.sym('h_0')
    v = ca.SX.sym('v_0')
    w += [h, v]
    w_lb += [h0, v0]
    w_ub += [h0, v0]
    w0 += [h0, v0]

    for k in range(N):
        # Control
        u = ca.SX.sym(f'u_{k}')
        w.append(u)
        w_lb.append(0)  # Thrust is non-negative
        w_ub.append(u_max)
        w0.append(m * g)  # Hover thrust

        # Dynamics (Euler)
        h_next = h + dt * v
        v_next = v + dt * (-g + u / m)

        if k < N - 1:
            h_new = ca.SX.sym(f'h_{k+1}')
            v_new = ca.SX.sym(f'v_{k+1}')
            w += [h_new, v_new]
            w_lb += [0, -100]  # Height >= 0
            w_ub += [200, 100]
            w0 += [h0 * (1 - (k+1)/N), v0 * (1 - (k+1)/N)]

            g_con += [h_new - h_next, v_new - v_next]
            g_lb += [0, 0]
            g_ub += [0, 0]

            h = h_new
            v = v_new
        else:
            # Terminal constraints
            g_con += [h_next]  # Land at h=0
            g_lb += [-0.1]
            g_ub += [0.1]

            # Objective: minimize final velocity magnitude
            J = v_next**2

    w_vec = ca.vertcat(*w)
    g_vec = ca.vertcat(*g_con)

    return create_nlp_from_casadi(
        x=w_vec,
        f=J,
        g=g_vec,
        lbx=w_lb,
        ubx=w_ub,
        lbg=g_lb,
        ubg=g_ub,
        x0=w0,
        name='rocket_landing',
        description=f'Rocket landing OCP (N={N})',
        source='Custom',
        is_ocp=True,
    )


def parameter_estimation() -> CasADiNLP:
    """
    Constrained Parameter Estimation Problem.

    Estimate parameters of a dynamic system with bounds.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # Synthetic measurement data from true system
    t_meas = np.array([0, 1, 2, 3, 4, 5])
    y_meas = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2])  # Exponential decay

    # Model: y = A * exp(-k * t)
    A = ca.SX.sym('A')
    k = ca.SX.sym('k')
    theta = ca.vertcat(A, k)

    # Residuals
    residuals = []
    for i in range(len(t_meas)):
        model = A * ca.exp(-k * t_meas[i])
        r_i = y_meas[i] - model
        residuals.append(r_i)

    r = ca.vertcat(*residuals)
    f = ca.dot(r, r)

    # Constraint: A + k <= 2 (for testing constrained estimation)
    g = A + k

    return create_nlp_from_casadi(
        x=theta,
        f=f,
        g=g,
        lbx=[0, 0],
        ubx=[5, 5],
        lbg=[-ca.inf],
        ubg=[2.0],
        x0=[1.0, 0.5],
        name='parameter_estimation',
        description='Constrained parameter estimation',
        source='Custom',
        is_least_squares=True,
    )


def robot_arm_2d() -> CasADiNLP:
    """
    2D Robot Arm Inverse Kinematics.

    Find joint angles to reach a target position.
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    # Arm parameters
    L1 = 1.0  # Link 1 length
    L2 = 0.8  # Link 2 length

    # Target position
    x_target = 1.2
    y_target = 0.8

    # Joint angles
    theta1 = ca.SX.sym('theta1')
    theta2 = ca.SX.sym('theta2')
    q = ca.vertcat(theta1, theta2)

    # Forward kinematics
    x_end = L1 * ca.cos(theta1) + L2 * ca.cos(theta1 + theta2)
    y_end = L1 * ca.sin(theta1) + L2 * ca.sin(theta1 + theta2)

    # Objective: distance to target
    f = (x_end - x_target)**2 + (y_end - y_target)**2

    return create_nlp_from_casadi(
        x=q,
        f=f,
        lbx=[-np.pi, -np.pi],
        ubx=[np.pi, np.pi],
        x0=[0.5, 0.5],
        name='robot_arm_2d',
        description='2D robot arm inverse kinematics',
        source='Custom',
    )


def quadratic_with_constraints() -> CasADiNLP:
    """
    Simple quadratic with linear constraints.

    min  (x-1)² + (y-2)²
    s.t. x + y <= 2
         x >= 0, y >= 0
    """
    if not HAS_CASADI:
        raise ImportError("CasADi required")

    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    z = ca.vertcat(x, y)

    f = (x - 1)**2 + (y - 2)**2
    g = x + y  # x + y <= 2

    return create_nlp_from_casadi(
        x=z,
        f=f,
        g=g,
        lbx=[0, 0],
        ubx=[10, 10],
        lbg=[-ca.inf],
        ubg=[2],
        x0=[0.5, 0.5],
        name='quadratic_constrained',
        description='Quadratic with linear inequality',
        source='Custom',
        is_convex=True,
        is_quadratic=True,
    )


def get_industrial_problems() -> List[CasADiNLP]:
    """Get all industrial NLP benchmark problems."""
    if not HAS_CASADI:
        return []

    problems = [
        van_der_pol_ocp(N=20),
        rocket_landing(N=20),
        parameter_estimation(),
        robot_arm_2d(),
        quadratic_with_constraints(),
    ]

    return problems


if __name__ == '__main__':
    if HAS_CASADI:
        problems = get_industrial_problems()
        print(f"Suite A: {len(problems)} industrial problems")
        for p in problems:
            print(f"  - {p.name}: {p.n_vars} vars, {p.n_constraints} constraints")
    else:
        print("CasADi not installed")
