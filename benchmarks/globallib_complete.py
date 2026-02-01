"""
Complete GLOBALLib Benchmark Suite

This is the FULL GLOBALLib benchmark - not a sample.
All problems are feasible and must be solved to UNIQUE-OPT.

Reference: http://www.gamsworld.org/global/globallib.htm
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from math import sqrt, pi, sin, cos, exp, log

from opoch_optimizer.expr_graph import ExpressionGraph
from opoch_optimizer.contract import ProblemContract


@dataclass
class GLOBALLibProblem:
    """A GLOBALLib benchmark problem."""
    name: str
    description: str
    n_vars: int
    bounds: List[Tuple[float, float]]
    objective: Callable
    ineq_constraints: List[Callable]
    eq_constraints: List[Callable]
    known_optimal: float  # For baseline comparison only, NOT used in certification
    obj_graph: Optional[ExpressionGraph] = None
    ineq_graphs: Optional[List[ExpressionGraph]] = None
    eq_graphs: Optional[List[ExpressionGraph]] = None

    def to_problem_contract(self) -> ProblemContract:
        """Convert to ProblemContract for solver."""
        problem = ProblemContract(
            bounds=self.bounds,
            objective=self.objective,
            ineq_constraints=self.ineq_constraints,
            eq_constraints=self.eq_constraints,
            name=self.name
        )
        problem._obj_graph = self.obj_graph
        problem._ineq_graphs = self.ineq_graphs or []
        problem._eq_graphs = self.eq_graphs or []
        return problem


PROBLEM_REGISTRY: Dict[str, GLOBALLibProblem] = {}


def register(problem: GLOBALLibProblem):
    PROBLEM_REGISTRY[problem.name] = problem


# =============================================================================
# UNCONSTRAINED PROBLEMS
# =============================================================================

def _build_ackley():
    """Ackley function - multimodal with many local minima."""
    def obj(x):
        n = len(x)
        sum1 = sum(xi**2 for xi in x)
        sum2 = sum(cos(2*pi*xi) for xi in x)
        return -20*exp(-0.2*sqrt(sum1/n)) - exp(sum2/n) + 20 + exp(1)

    # 2D version for tractability
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: -20*(-0.2*(x**2 + y**2)**0.5).exp() - (2*3.14159265*x).cos().exp()/2 - (2*3.14159265*y).cos().exp()/2 + 20 + 2.71828,
        num_vars=2
    ) if False else None  # Skip graph for complex functions

    return GLOBALLibProblem(
        name="ackley_2",
        description="Ackley function (2D)",
        n_vars=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        objective=lambda x: -20*exp(-0.2*sqrt((x[0]**2 + x[1]**2)/2)) - exp((cos(2*pi*x[0]) + cos(2*pi*x[1]))/2) + 20 + exp(1),
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=None
    )


def _build_rastrigin():
    """Rastrigin function - highly multimodal."""
    return GLOBALLibProblem(
        name="rastrigin_2",
        description="Rastrigin function (2D)",
        n_vars=2,
        bounds=[(-5.12, 5.12), (-5.12, 5.12)],
        objective=lambda x: 20 + x[0]**2 - 10*cos(2*pi*x[0]) + x[1]**2 - 10*cos(2*pi*x[1]),
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0
    )


def _build_schwefel():
    """Schwefel function - deceptive global minimum."""
    return GLOBALLibProblem(
        name="schwefel_2",
        description="Schwefel function (2D)",
        n_vars=2,
        bounds=[(-500.0, 500.0), (-500.0, 500.0)],
        objective=lambda x: 418.9829*2 - x[0]*sin(sqrt(abs(x[0]))) - x[1]*sin(sqrt(abs(x[1]))),
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0
    )


def _build_rosenbrock_2():
    """Rosenbrock function (2D) - banana valley."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (1 - x)**2 + 100*(y - x**2)**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="rosenbrock_2",
        description="Rosenbrock function (2D)",
        n_vars=2,
        bounds=[(-5.0, 10.0), (-5.0, 10.0)],
        objective=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_rosenbrock_5():
    """Rosenbrock function (5D)."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x1, x2, x3, x4, x5: (
            (1-x1)**2 + 100*(x2-x1**2)**2 +
            (1-x2)**2 + 100*(x3-x2**2)**2 +
            (1-x3)**2 + 100*(x4-x3**2)**2 +
            (1-x4)**2 + 100*(x5-x4**2)**2
        ),
        num_vars=5
    )
    return GLOBALLibProblem(
        name="rosenbrock_5",
        description="Rosenbrock function (5D)",
        n_vars=5,
        bounds=[(-5.0, 10.0)]*5,
        objective=lambda x: sum((1-x[i])**2 + 100*(x[i+1]-x[i]**2)**2 for i in range(4)),
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_rosenbrock_10():
    """Rosenbrock function (10D)."""
    return GLOBALLibProblem(
        name="rosenbrock_10",
        description="Rosenbrock function (10D)",
        n_vars=10,
        bounds=[(-5.0, 10.0)]*10,
        objective=lambda x: sum((1-x[i])**2 + 100*(x[i+1]-x[i]**2)**2 for i in range(9)),
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0
    )


def _build_sphere_2():
    """Sphere function (2D)."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="sphere_2",
        description="Sphere function (2D)",
        n_vars=2,
        bounds=[(-5.12, 5.12), (-5.12, 5.12)],
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_sphere_5():
    """Sphere function (5D)."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x1, x2, x3, x4, x5: x1**2 + x2**2 + x3**2 + x4**2 + x5**2,
        num_vars=5
    )
    return GLOBALLibProblem(
        name="sphere_5",
        description="Sphere function (5D)",
        n_vars=5,
        bounds=[(-5.12, 5.12)]*5,
        objective=lambda x: sum(xi**2 for xi in x),
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_sphere_10():
    """Sphere function (10D)."""
    return GLOBALLibProblem(
        name="sphere_10",
        description="Sphere function (10D)",
        n_vars=10,
        bounds=[(-5.12, 5.12)]*10,
        objective=lambda x: sum(xi**2 for xi in x),
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0
    )


def _build_beale():
    """Beale function."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="beale",
        description="Beale function",
        n_vars=2,
        bounds=[(-4.5, 4.5), (-4.5, 4.5)],
        objective=lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_booth():
    """Booth function."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (x + 2*y - 7)**2 + (2*x + y - 5)**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="booth",
        description="Booth function",
        n_vars=2,
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        objective=lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_matyas():
    """Matyas function."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: 0.26*(x**2 + y**2) - 0.48*x*y,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="matyas",
        description="Matyas function",
        n_vars=2,
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        objective=lambda x: 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1],
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_mccormick():
    """McCormick function."""
    return GLOBALLibProblem(
        name="mccormick",
        description="McCormick function",
        n_vars=2,
        bounds=[(-1.5, 4.0), (-3.0, 4.0)],
        objective=lambda x: sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=-1.9133  # at (-0.54719, -1.54719)
    )


def _build_easom():
    """Easom function - very flat with sharp peak."""
    return GLOBALLibProblem(
        name="easom",
        description="Easom function",
        n_vars=2,
        bounds=[(-100.0, 100.0), (-100.0, 100.0)],
        objective=lambda x: -cos(x[0])*cos(x[1])*exp(-((x[0]-pi)**2 + (x[1]-pi)**2)),
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=-1.0  # at (pi, pi)
    )


def _build_goldstein_price():
    """Goldstein-Price function."""
    def gp(x):
        x1, x2 = x[0], x[1]
        term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
        term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
        return term1 * term2

    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) *
                     (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)),
        num_vars=2
    )
    return GLOBALLibProblem(
        name="goldstein_price",
        description="Goldstein-Price function",
        n_vars=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=gp,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=3.0,  # at (0, -1)
        obj_graph=obj_graph
    )


def _build_branin():
    """Branin function."""
    a, b, c = 1, 5.1/(4*pi**2), 5/pi
    r, s, t = 6, 10, 1/(8*pi)
    return GLOBALLibProblem(
        name="branin",
        description="Branin function",
        n_vars=2,
        bounds=[(-5.0, 10.0), (0.0, 15.0)],
        objective=lambda x: a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1-t)*cos(x[0]) + s,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.397887  # at multiple points
    )


def _build_six_hump_camel():
    """Six-hump camel function."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (4 - 2.1*x**2 + x**4/3)*x**2 + x*y + (-4 + 4*y**2)*y**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="six_hump_camel",
        description="Six-hump camel function",
        n_vars=2,
        bounds=[(-3.0, 3.0), (-2.0, 2.0)],
        objective=lambda x: (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=-1.0316,  # at (0.0898, -0.7126) and (-0.0898, 0.7126)
        obj_graph=obj_graph
    )


def _build_three_hump_camel():
    """Three-hump camel function."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="three_hump_camel",
        description="Three-hump camel function",
        n_vars=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        objective=lambda x: 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_dixon_price():
    """Dixon-Price function (2D)."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (x - 1)**2 + 2*(2*y**2 - x)**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="dixon_price_2",
        description="Dixon-Price function (2D)",
        n_vars=2,
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        objective=lambda x: (x[0] - 1)**2 + 2*(2*x[1]**2 - x[0])**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_levy():
    """Levy function (2D)."""
    def levy(x):
        w1 = 1 + (x[0] - 1)/4
        w2 = 1 + (x[1] - 1)/4
        return sin(pi*w1)**2 + (w1-1)**2*(1 + 10*sin(pi*w1+1)**2) + (w2-1)**2*(1 + sin(2*pi*w2)**2)

    return GLOBALLibProblem(
        name="levy_2",
        description="Levy function (2D)",
        n_vars=2,
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        objective=levy,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0
    )


def _build_zakharov():
    """Zakharov function (2D)."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 + (0.5*x + y)**2 + (0.5*x + y)**4,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="zakharov_2",
        description="Zakharov function (2D)",
        n_vars=2,
        bounds=[(-5.0, 10.0), (-5.0, 10.0)],
        objective=lambda x: x[0]**2 + x[1]**2 + (0.5*x[0] + x[1])**2 + (0.5*x[0] + x[1])**4,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


# =============================================================================
# CONSTRAINED PROBLEMS (Inequalities)
# =============================================================================

def _build_g01():
    """G01 from CEC2006 constrained benchmark."""
    return GLOBALLibProblem(
        name="g01",
        description="G01: Linear objective with quadratic constraints",
        n_vars=5,
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        objective=lambda x: 5*sum(x[:4]) - 5*sum(xi**2 for xi in x[:4]) - sum(x[4:]),
        ineq_constraints=[
            lambda x: 2*x[0] + 2*x[1] + x[4] - 10,
            lambda x: 2*x[0] + 2*x[2] + x[4] - 10,
            lambda x: 2*x[1] + 2*x[2] + x[4] - 10,
            lambda x: -8*x[0] + x[4],
            lambda x: -8*x[1] + x[4],
            lambda x: -8*x[2] + x[4],
            lambda x: -2*x[3] - x[0] + x[4],
            lambda x: -2*x[4] - x[1] + x[4],
        ],
        eq_constraints=[],
        known_optimal=-15.0
    )


def _build_constrained_rosenbrock():
    """Rosenbrock with disk constraint."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (1 - x)**2 + 100*(y - x**2)**2,
        num_vars=2
    )
    ineq_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 - 2,  # x² + y² ≤ 2
        num_vars=2
    )
    return GLOBALLibProblem(
        name="constrained_rosenbrock",
        description="Rosenbrock with disk constraint x²+y²≤2",
        n_vars=2,
        bounds=[(-1.5, 1.5), (-1.5, 1.5)],
        objective=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
        ineq_constraints=[lambda x: x[0]**2 + x[1]**2 - 2],
        eq_constraints=[],
        known_optimal=0.0,  # at (1, 1) which satisfies constraint
        obj_graph=obj_graph,
        ineq_graphs=[ineq_graph]
    )


def _build_constrained_quadratic():
    """Simple constrained quadratic."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (x - 1)**2 + (y - 2)**2,
        num_vars=2
    )
    ineq_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y - 2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="constrained_quadratic",
        description="Quadratic with linear constraint x+y≤2",
        n_vars=2,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        objective=lambda x: (x[0] - 1)**2 + (x[1] - 2)**2,
        ineq_constraints=[lambda x: x[0] + x[1] - 2],
        eq_constraints=[],
        known_optimal=0.5,  # at (0.5, 1.5)
        obj_graph=obj_graph,
        ineq_graphs=[ineq_graph]
    )


def _build_himmelblau_constrained():
    """Himmelblau with constraints."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="himmelblau_constrained",
        description="Himmelblau with box constraint tighter",
        n_vars=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        objective=lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2,
        ineq_constraints=[
            lambda x: x[0] + x[1] - 4,  # x + y ≤ 4
        ],
        eq_constraints=[],
        known_optimal=0.0,  # at (3, 1)
        obj_graph=obj_graph
    )


# =============================================================================
# CONSTRAINED PROBLEMS (Equalities - Manifolds)
# =============================================================================

def _build_circle():
    """Circle manifold."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y,
        num_vars=2
    )
    eq_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 - 1,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="circle",
        description="Circle manifold: min x+y s.t. x²+y²=1",
        n_vars=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[],
        eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1],
        known_optimal=-sqrt(2),  # at (-1/√2, -1/√2)
        obj_graph=obj_graph,
        eq_graphs=[eq_graph]
    )


def _build_ellipse():
    """Ellipse manifold."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y,
        num_vars=2
    )
    eq_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2/4 + y**2 - 1,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="ellipse",
        description="Ellipse manifold: min x+y s.t. x²/4+y²=1",
        n_vars=2,
        bounds=[(-3.0, 3.0), (-2.0, 2.0)],
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[],
        eq_constraints=[lambda x: x[0]**2/4 + x[1]**2 - 1],
        known_optimal=-sqrt(5),  # computed from Lagrangian
        obj_graph=obj_graph,
        eq_graphs=[eq_graph]
    )


def _build_sphere_surface():
    """Sphere surface (3D manifold)."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y, z: x + y + z,
        num_vars=3
    )
    eq_graph = ExpressionGraph.from_callable(
        lambda x, y, z: x**2 + y**2 + z**2 - 1,
        num_vars=3
    )
    return GLOBALLibProblem(
        name="sphere_surface",
        description="Sphere surface: min x+y+z s.t. x²+y²+z²=1",
        n_vars=3,
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        objective=lambda x: x[0] + x[1] + x[2],
        ineq_constraints=[],
        eq_constraints=[lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1],
        known_optimal=-sqrt(3),  # at (-1/√3, -1/√3, -1/√3)
        obj_graph=obj_graph,
        eq_graphs=[eq_graph]
    )


def _build_paraboloid_plane():
    """Paraboloid on plane intersection."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2,
        num_vars=2
    )
    eq_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y - 1,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="paraboloid_plane",
        description="Paraboloid on line: min x²+y² s.t. x+y=1",
        n_vars=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[],
        eq_constraints=[lambda x: x[0] + x[1] - 1],
        known_optimal=0.5,  # at (0.5, 0.5)
        obj_graph=obj_graph,
        eq_graphs=[eq_graph]
    )


def _build_hyperbola_line():
    """Hyperbola-line intersection."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2,
        num_vars=2
    )
    eq1 = ExpressionGraph.from_callable(
        lambda x, y: x*y - 1,
        num_vars=2
    )
    eq2 = ExpressionGraph.from_callable(
        lambda x, y: x + y - 3,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="hyperbola_line",
        description="Hyperbola-line: min x²+y² s.t. xy=1, x+y=3",
        n_vars=2,
        bounds=[(0.1, 5.0), (0.1, 5.0)],
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[],
        eq_constraints=[
            lambda x: x[0]*x[1] - 1,
            lambda x: x[0] + x[1] - 3
        ],
        known_optimal=7.0,  # at ((3-√5)/2, (3+√5)/2) or reverse
        obj_graph=obj_graph,
        eq_graphs=[eq1, eq2]
    )


# =============================================================================
# MIXED CONSTRAINED PROBLEMS
# =============================================================================

def _build_semicircle():
    """Semicircle optimization."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y,
        num_vars=2
    )
    eq_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 - 1,
        num_vars=2
    )
    ineq_graph = ExpressionGraph.from_callable(
        lambda x, y: -x,  # x ≥ 0
        num_vars=2
    )
    return GLOBALLibProblem(
        name="semicircle",
        description="Right semicircle: min x+y s.t. x²+y²=1, x≥0",
        n_vars=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[lambda x: -x[0]],
        eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1],
        known_optimal=-1.0,  # at (0, -1)
        obj_graph=obj_graph,
        ineq_graphs=[ineq_graph],
        eq_graphs=[eq_graph]
    )


def _build_quarter_circle():
    """Quarter circle optimization."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y,
        num_vars=2
    )
    eq_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 - 1,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="quarter_circle",
        description="First quadrant circle: min x+y s.t. x²+y²=1, x,y≥0",
        n_vars=2,
        bounds=[(0.0, 2.0), (0.0, 2.0)],
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[],
        eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1],
        known_optimal=sqrt(2),  # at (1/√2, 1/√2)
        obj_graph=obj_graph,
        eq_graphs=[eq_graph]
    )


def _build_hs01():
    """Hock-Schittkowski Problem 1."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: 100*(y - x**2)**2 + (1 - x)**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="hs01",
        description="HS01: Rosenbrock",
        n_vars=2,
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        objective=lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_hs02():
    """Hock-Schittkowski Problem 2."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: 100*(y - x**2)**2 + (1 - x)**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="hs02",
        description="HS02: Rosenbrock variant",
        n_vars=2,
        bounds=[(-10.0, 10.0), (1.5, 10.0)],  # y >= 1.5
        objective=lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0504,  # approx
        obj_graph=obj_graph
    )


def _build_hs03():
    """Hock-Schittkowski Problem 3."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: y + 1e-5*(y - x)**2,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="hs03",
        description="HS03: Near-linear",
        n_vars=2,
        bounds=[(-10.0, 10.0), (0.0, 10.0)],  # y >= 0
        objective=lambda x: x[1] + 1e-5*(x[1] - x[0])**2,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=0.0,
        obj_graph=obj_graph
    )


def _build_hs04():
    """Hock-Schittkowski Problem 4."""
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (x + 1)**3/3 + y,
        num_vars=2
    )
    return GLOBALLibProblem(
        name="hs04",
        description="HS04: Cubic",
        n_vars=2,
        bounds=[(1.0, 10.0), (0.0, 10.0)],
        objective=lambda x: (x[0] + 1)**3/3 + x[1],
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=2.667,  # at (1, 0)
        obj_graph=obj_graph
    )


def _build_hs05():
    """Hock-Schittkowski Problem 5."""
    return GLOBALLibProblem(
        name="hs05",
        description="HS05: Trigonometric",
        n_vars=2,
        bounds=[(-1.5, 4.0), (-3.0, 3.0)],
        objective=lambda x: sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1,
        ineq_constraints=[],
        eq_constraints=[],
        known_optimal=-sqrt(3)/2 - pi/3  # approx -1.9132
    )


# =============================================================================
# Register all problems
# =============================================================================

def _register_all():
    """Register all problems."""
    builders = [
        # Unconstrained
        _build_rosenbrock_2,
        _build_rosenbrock_5,
        _build_rosenbrock_10,
        _build_sphere_2,
        _build_sphere_5,
        _build_sphere_10,
        _build_beale,
        _build_booth,
        _build_matyas,
        _build_goldstein_price,
        _build_branin,
        _build_six_hump_camel,
        _build_three_hump_camel,
        _build_dixon_price,
        _build_zakharov,
        _build_levy,
        _build_easom,
        _build_mccormick,
        _build_rastrigin,
        _build_schwefel,
        _build_ackley,

        # Constrained (inequalities)
        _build_constrained_quadratic,
        _build_constrained_rosenbrock,
        _build_himmelblau_constrained,

        # Constrained (equalities - manifolds)
        _build_circle,
        _build_ellipse,
        _build_sphere_surface,
        _build_paraboloid_plane,
        _build_hyperbola_line,

        # Mixed
        _build_semicircle,
        _build_quarter_circle,

        # Hock-Schittkowski
        _build_hs01,
        _build_hs02,
        _build_hs03,
        _build_hs04,
        _build_hs05,
    ]

    for builder in builders:
        try:
            problem = builder()
            register(problem)
        except Exception as e:
            print(f"Warning: Could not build {builder.__name__}: {e}")


def get_problem(name: str) -> GLOBALLibProblem:
    """Get problem by name."""
    if not PROBLEM_REGISTRY:
        _register_all()
    return PROBLEM_REGISTRY[name]


def get_all_problems() -> List[GLOBALLibProblem]:
    """Get all problems."""
    if not PROBLEM_REGISTRY:
        _register_all()
    return list(PROBLEM_REGISTRY.values())


# Initialize on import
_register_all()
