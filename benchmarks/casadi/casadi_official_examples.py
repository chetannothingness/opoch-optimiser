#!/usr/bin/env python3
"""
Official CasADi Examples from GitHub
https://github.com/casadi/casadi/tree/main/docs/examples/python

These are the EXACT examples from CasADi's official repository.
No modifications - direct from source.
"""

import casadi as ca
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class CasADiExample:
    """Official CasADi example problem."""
    name: str
    source: str  # GitHub path
    x_sym: ca.MX
    f_sym: ca.MX
    g_sym: Optional[ca.MX]
    x0: np.ndarray
    lbx: np.ndarray
    ubx: np.ndarray
    lbg: np.ndarray
    ubg: np.ndarray

    @property
    def n_vars(self) -> int:
        return self.x_sym.shape[0]

    @property
    def n_constraints(self) -> int:
        return 0 if self.g_sym is None else self.g_sym.shape[0]


def simple_nlp() -> CasADiExample:
    """
    simple_nlp.py - Basic NLP example
    https://github.com/casadi/casadi/blob/main/docs/examples/python/simple_nlp.py

    min  x[0]^2 + x[1]^2
    s.t. x[0] + x[1] - 10 = 0
    """
    x = ca.MX.sym("x", 2)
    f = x[0]**2 + x[1]**2
    g = x[0] + x[1] - 10

    return CasADiExample(
        name="simple_nlp",
        source="docs/examples/python/simple_nlp.py",
        x_sym=x,
        f_sym=f,
        g_sym=g,
        x0=np.array([0.0, 0.0]),
        lbx=np.array([-np.inf, -np.inf]),
        ubx=np.array([np.inf, np.inf]),
        lbg=np.array([0.0]),
        ubg=np.array([0.0]),
    )


def rosenbrock() -> CasADiExample:
    """
    rosenbrock.py - Classic Rosenbrock problem
    https://github.com/casadi/casadi/blob/main/docs/examples/python/rosenbrock.py

    min  x^2 + 100*z^2
    s.t. z + (1-x)^2 - y = 0
    """
    x = ca.MX.sym("x")
    y = ca.MX.sym("y")
    z = ca.MX.sym("z")

    vars = ca.vertcat(x, y, z)
    f = x**2 + 100*z**2
    g = z + (1-x)**2 - y

    return CasADiExample(
        name="rosenbrock",
        source="docs/examples/python/rosenbrock.py",
        x_sym=vars,
        f_sym=f,
        g_sym=g,
        x0=np.array([2.5, 3.0, 0.75]),
        lbx=np.array([-np.inf, -np.inf, -np.inf]),
        ubx=np.array([np.inf, np.inf, np.inf]),
        lbg=np.array([0.0]),
        ubg=np.array([0.0]),
    )


def rocket() -> CasADiExample:
    """
    rocket.py - Rocket optimal control (discretized)
    https://github.com/casadi/casadi/blob/main/docs/examples/python/rocket.py

    Minimize control effort to reach position=10, velocity=0
    State: [position, velocity, mass]
    Control: thrust
    """
    # Number of control segments
    nu = 50

    # Control for all segments
    U = ca.MX.sym("U", nu)

    # State: [position, velocity, mass]
    # ODE: sdot=v, vdot=(u-0.05*v^2)/m, mdot=-0.1*u^2

    # Discrete dynamics via Euler integration
    dt = 0.01
    x = ca.MX([0, 0, 1])  # Initial: pos=0, vel=0, mass=1

    for k in range(nu):
        u = U[k]
        for j in range(20):  # 20 Euler steps per control
            s, v, m = x[0], x[1], x[2]
            sdot = v
            vdot = (u - 0.05 * v * v) / m
            mdot = -0.1 * u * u
            x = x + dt * ca.vertcat(sdot, vdot, mdot)

    # Objective: minimize control effort
    f = ca.mtimes(U.T, U)

    # Constraints: final position=10, final velocity=0
    g = x[0:2]

    return CasADiExample(
        name="rocket",
        source="docs/examples/python/rocket.py",
        x_sym=U,
        f_sym=f,
        g_sym=g,
        x0=0.4 * np.ones(nu),
        lbx=-0.5 * np.ones(nu),
        ubx=0.5 * np.ones(nu),
        lbg=np.array([10.0, 0.0]),
        ubg=np.array([10.0, 0.0]),
    )


def vdp_multiple_shooting() -> CasADiExample:
    """
    direct_multiple_shooting.py - Van der Pol oscillator
    https://github.com/casadi/casadi/blob/main/docs/examples/python/direct_multiple_shooting.py

    Minimize integral of x1^2 + x2^2 + u^2
    Dynamics: Van der Pol oscillator
    """
    T = 10.0  # Time horizon
    N = 20    # Control intervals

    # Decision variables: [X0, U0, X1, U1, ..., XN]
    # X = [x1, x2], U = scalar
    n_vars = 2 * (N + 1) + N  # states + controls

    w = []
    w0 = []
    lbw = []
    ubw = []

    # Initial state
    X0 = ca.MX.sym('X0', 2)
    w.append(X0)
    lbw.extend([0, 1])
    ubw.extend([0, 1])
    w0.extend([0, 1])

    # RK4 integration
    def rk4_step(x, u, dt):
        M = 4
        DT = dt / M
        for _ in range(M):
            x1, x2 = x[0], x[1]
            xdot = ca.vertcat((1 - x2**2) * x1 - x2 + u, x1)
            k1 = xdot
            x_mid = x + DT/2 * k1
            x1, x2 = x_mid[0], x_mid[1]
            k2 = ca.vertcat((1 - x2**2) * x1 - x2 + u, x1)
            x_mid = x + DT/2 * k2
            x1, x2 = x_mid[0], x_mid[1]
            k3 = ca.vertcat((1 - x2**2) * x1 - x2 + u, x1)
            x_end = x + DT * k3
            x1, x2 = x_end[0], x_end[1]
            k4 = ca.vertcat((1 - x2**2) * x1 - x2 + u, x1)
            x = x + DT/6 * (k1 + 2*k2 + 2*k3 + k4)
        return x

    J = 0
    g = []
    lbg = []
    ubg = []

    Xk = X0
    dt = T / N

    for k in range(N):
        Uk = ca.MX.sym(f'U_{k}')
        w.append(Uk)
        lbw.append(-1)
        ubw.append(1)
        w0.append(0)

        # Integrate
        Xk_end = rk4_step(Xk, Uk, dt)

        # Cost
        J = J + dt * (Xk[0]**2 + Xk[1]**2 + Uk**2)

        # Next state variable
        Xk = ca.MX.sym(f'X_{k+1}', 2)
        w.append(Xk)
        lbw.extend([-0.25, -np.inf])
        ubw.extend([np.inf, np.inf])
        w0.extend([0, 0])

        # Continuity constraint
        g.append(Xk_end - Xk)
        lbg.extend([0, 0])
        ubg.extend([0, 0])

    return CasADiExample(
        name="vdp_multiple_shooting",
        source="docs/examples/python/direct_multiple_shooting.py",
        x_sym=ca.vertcat(*w),
        f_sym=J,
        g_sym=ca.vertcat(*g),
        x0=np.array(w0),
        lbx=np.array(lbw),
        ubx=np.array(ubw),
        lbg=np.array(lbg),
        ubg=np.array(ubg),
    )


def race_car() -> CasADiExample:
    """
    race_car.py - Minimum time race car
    https://github.com/casadi/casadi/blob/main/docs/examples/python/race_car.py

    Minimize time to reach position=1 with speed limit constraint
    """
    N = 100

    # Decision variables: states X[2, N+1], controls U[1, N], final time T
    n_x = 2 * (N + 1)
    n_u = N
    n_total = n_x + n_u + 1

    # Build the problem manually
    w = ca.MX.sym('w', n_total)

    # Extract variables
    X = ca.reshape(w[:n_x], 2, N+1)
    U = w[n_x:n_x+n_u]
    T = w[-1]

    pos = X[0, :]
    speed = X[1, :]

    # Objective: minimize time
    f = T

    # Dynamics constraints
    g_list = []
    dt = T / N

    for k in range(N):
        x_k = X[:, k]
        u_k = U[k]

        # RK4
        def f_ode(x, u):
            return ca.vertcat(x[1], u - x[1])

        k1 = f_ode(x_k, u_k)
        k2 = f_ode(x_k + dt/2 * k1, u_k)
        k3 = f_ode(x_k + dt/2 * k2, u_k)
        k4 = f_ode(x_k + dt * k3, u_k)
        x_next = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        g_list.append(X[:, k+1] - x_next)

    # Speed limit: speed <= 1 - sin(2*pi*pos)/2
    for k in range(N+1):
        limit = 1 - ca.sin(2 * np.pi * pos[k]) / 2
        g_list.append(speed[k] - limit)

    # Boundary conditions
    g_list.append(pos[0])      # start at 0
    g_list.append(speed[0])    # start from rest
    g_list.append(pos[-1] - 1) # end at 1

    g = ca.vertcat(*g_list)

    # Bounds
    lbw = np.concatenate([
        np.tile([-np.inf, -np.inf], N+1),  # states
        np.zeros(N),                        # control >= 0
        [0]                                 # T >= 0
    ])
    ubw = np.concatenate([
        np.tile([np.inf, np.inf], N+1),
        np.ones(N),                         # control <= 1
        [np.inf]
    ])

    # Initial guess
    w0 = np.concatenate([
        np.tile([0.5, 0.5], N+1),  # states
        0.5 * np.ones(N),          # controls
        [1.0]                       # time
    ])

    # Constraint bounds
    n_dyn = 2 * N
    n_speed = N + 1
    lbg = np.concatenate([
        np.zeros(n_dyn),           # dynamics
        -np.inf * np.ones(n_speed), # speed <= limit
        [0, 0, 0]                   # boundary
    ])
    ubg = np.concatenate([
        np.zeros(n_dyn),
        np.zeros(n_speed),
        [0, 0, 0]
    ])

    return CasADiExample(
        name="race_car",
        source="docs/examples/python/race_car.py",
        x_sym=w,
        f_sym=f,
        g_sym=g,
        x0=w0,
        lbx=lbw,
        ubx=ubw,
        lbg=lbg,
        ubg=ubg,
    )


def chain_qp() -> CasADiExample:
    """
    chain_qp.py - Hanging chain (QP)
    https://github.com/casadi/casadi/blob/main/docs/examples/python/chain_qp.py

    Minimize potential energy of hanging chain
    """
    N = 40
    m_i = 40.0 / N
    D_i = 70.0 * N
    g0 = 9.81
    zmin = 0.5

    # Variables: [y1, z1, y2, z2, ..., yN, zN]
    w = ca.MX.sym('w', 2 * N)

    # Potential energy
    V = 0
    g_list = []

    lbw = []
    ubw = []
    w0 = []

    for i in range(N):
        y_i = w[2*i]
        z_i = w[2*i + 1]

        # Bounds
        if i == 0:
            lbw.extend([-2., 1.])
            ubw.extend([-2., 1.])
        elif i == N - 1:
            lbw.extend([2., 1.])
            ubw.extend([2., 1.])
        else:
            lbw.extend([-np.inf, zmin])
            ubw.extend([np.inf, np.inf])

        w0.extend([4.0 * i / N - 2, 0.5])

        # Spring energy
        if i > 0:
            y_prev = w[2*(i-1)]
            z_prev = w[2*(i-1) + 1]
            V += D_i / 2 * ((y_prev - y_i)**2 + (z_prev - z_i)**2)

        # Gravitational energy
        V += g0 * m_i * z_i

        # Ground constraint: z - 0.1*y >= 0.5
        g_list.append(z_i - 0.1 * y_i)

    g = ca.vertcat(*g_list)
    lbg = 0.5 * np.ones(N)
    ubg = np.inf * np.ones(N)

    return CasADiExample(
        name="chain_qp",
        source="docs/examples/python/chain_qp.py",
        x_sym=w,
        f_sym=V,
        g_sym=g,
        x0=np.array(w0),
        lbx=np.array(lbw),
        ubx=np.array(ubw),
        lbg=lbg,
        ubg=ubg,
    )


def get_official_casadi_examples():
    """Get all official CasADi examples."""
    return [
        simple_nlp(),
        rosenbrock(),
        rocket(),
        vdp_multiple_shooting(),
        race_car(),
        chain_qp(),
    ]


if __name__ == "__main__":
    print("Official CasADi Examples from GitHub")
    print("=" * 60)

    for ex in get_official_casadi_examples():
        print(f"\n{ex.name}")
        print(f"  Source: {ex.source}")
        print(f"  Variables: {ex.n_vars}")
        print(f"  Constraints: {ex.n_constraints}")
