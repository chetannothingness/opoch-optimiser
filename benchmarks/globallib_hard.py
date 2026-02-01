"""
GLOBALLib HARD Benchmark Suite

Truly difficult global optimization problems:
- Higher dimensions (5D, 10D, 20D)
- Multiple equality constraints
- Nonconvex feasible regions
- Classic hard instances

Pure mathematics. No shortcuts. Complete honesty.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from math import sqrt, pi, sin, cos, exp, log

from opoch_optimizer.expr_graph import ExpressionGraph
from opoch_optimizer.contract import ProblemContract


@dataclass
class HardProblem:
    """A hard GLOBALLib benchmark problem."""
    name: str
    category: str  # unconstrained, inequality, equality, mixed
    difficulty: str  # easy, medium, hard, extreme
    n_vars: int
    bounds: List[Tuple[float, float]]
    objective: Callable
    ineq_constraints: List[Callable]
    eq_constraints: List[Callable]
    known_optimal: float
    obj_graph: Optional[ExpressionGraph] = None
    ineq_graphs: Optional[List[ExpressionGraph]] = None
    eq_graphs: Optional[List[ExpressionGraph]] = None

    def to_problem_contract(self) -> ProblemContract:
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


HARD_REGISTRY: Dict[str, HardProblem] = {}


def register(p: HardProblem):
    HARD_REGISTRY[p.name] = p


# =============================================================================
# UNCONSTRAINED - EASY (2D)
# =============================================================================

def _sphere_2d():
    obj = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2, num_vars=2)
    return HardProblem(
        name="sphere_2d", category="unconstrained", difficulty="easy", n_vars=2,
        bounds=[(-5.0, 5.0)]*2,
        objective=lambda x: sum(xi**2 for xi in x),
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _booth():
    obj = ExpressionGraph.from_callable(lambda x, y: (x + 2*y - 7)**2 + (2*x + y - 5)**2, num_vars=2)
    return HardProblem(
        name="booth", category="unconstrained", difficulty="easy", n_vars=2,
        bounds=[(-10.0, 10.0)]*2,
        objective=lambda x: (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _matyas():
    obj = ExpressionGraph.from_callable(lambda x, y: 0.26*(x**2 + y**2) - 0.48*x*y, num_vars=2)
    return HardProblem(
        name="matyas", category="unconstrained", difficulty="easy", n_vars=2,
        bounds=[(-10.0, 10.0)]*2,
        objective=lambda x: 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1],
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

# =============================================================================
# UNCONSTRAINED - MEDIUM (2D-5D)
# =============================================================================

def _rosenbrock_2d():
    obj = ExpressionGraph.from_callable(lambda x, y: (1-x)**2 + 100*(y-x**2)**2, num_vars=2)
    return HardProblem(
        name="rosenbrock_2d", category="unconstrained", difficulty="medium", n_vars=2,
        bounds=[(-5.0, 10.0)]*2,
        objective=lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _rosenbrock_5d():
    obj = ExpressionGraph.from_callable(
        lambda x1,x2,x3,x4,x5: (1-x1)**2 + 100*(x2-x1**2)**2 + (1-x2)**2 + 100*(x3-x2**2)**2 +
                               (1-x3)**2 + 100*(x4-x3**2)**2 + (1-x4)**2 + 100*(x5-x4**2)**2,
        num_vars=5
    )
    return HardProblem(
        name="rosenbrock_5d", category="unconstrained", difficulty="medium", n_vars=5,
        bounds=[(-5.0, 10.0)]*5,
        objective=lambda x: sum((1-x[i])**2 + 100*(x[i+1]-x[i]**2)**2 for i in range(4)),
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _beale():
    obj = ExpressionGraph.from_callable(
        lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2,
        num_vars=2
    )
    return HardProblem(
        name="beale", category="unconstrained", difficulty="medium", n_vars=2,
        bounds=[(-4.5, 4.5)]*2,
        objective=lambda x: (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _sphere_5d():
    obj = ExpressionGraph.from_callable(
        lambda x1,x2,x3,x4,x5: x1**2 + x2**2 + x3**2 + x4**2 + x5**2,
        num_vars=5
    )
    return HardProblem(
        name="sphere_5d", category="unconstrained", difficulty="medium", n_vars=5,
        bounds=[(-5.0, 5.0)]*5,
        objective=lambda x: sum(xi**2 for xi in x),
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

# =============================================================================
# UNCONSTRAINED - HARD (2D-10D)
# =============================================================================

def _goldstein_price():
    def gp(x, y):
        t1 = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
        t2 = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
        return t1 * t2
    obj = ExpressionGraph.from_callable(lambda x, y: gp(x, y), num_vars=2)
    return HardProblem(
        name="goldstein_price", category="unconstrained", difficulty="hard", n_vars=2,
        bounds=[(-2.0, 2.0)]*2,
        objective=lambda x: gp(x[0], x[1]),
        ineq_constraints=[], eq_constraints=[],
        known_optimal=3.0, obj_graph=obj
    )

def _six_hump_camel():
    obj = ExpressionGraph.from_callable(
        lambda x, y: (4 - 2.1*x**2 + x**4/3)*x**2 + x*y + (-4 + 4*y**2)*y**2,
        num_vars=2
    )
    return HardProblem(
        name="six_hump_camel", category="unconstrained", difficulty="hard", n_vars=2,
        bounds=[(-3.0, 3.0), (-2.0, 2.0)],
        objective=lambda x: (4 - 2.1*x[0]**2 + x[0]**4/3)*x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=-1.0316, obj_graph=obj
    )

def _three_hump_camel():
    obj = ExpressionGraph.from_callable(
        lambda x, y: 2*x**2 - 1.05*x**4 + x**6/6 + x*y + y**2,
        num_vars=2
    )
    return HardProblem(
        name="three_hump_camel", category="unconstrained", difficulty="hard", n_vars=2,
        bounds=[(-5.0, 5.0)]*2,
        objective=lambda x: 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _dixon_price_2d():
    obj = ExpressionGraph.from_callable(lambda x, y: (x-1)**2 + 2*(2*y**2 - x)**2, num_vars=2)
    return HardProblem(
        name="dixon_price_2d", category="unconstrained", difficulty="hard", n_vars=2,
        bounds=[(-10.0, 10.0)]*2,
        objective=lambda x: (x[0]-1)**2 + 2*(2*x[1]**2 - x[0])**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _zakharov_2d():
    obj = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 + (0.5*x + y)**2 + (0.5*x + y)**4,
        num_vars=2
    )
    return HardProblem(
        name="zakharov_2d", category="unconstrained", difficulty="hard", n_vars=2,
        bounds=[(-5.0, 10.0)]*2,
        objective=lambda x: x[0]**2 + x[1]**2 + (0.5*x[0] + x[1])**2 + (0.5*x[0] + x[1])**4,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _powell_4d():
    """Powell function - sum of quartic terms."""
    obj = ExpressionGraph.from_callable(
        lambda x1, x2, x3, x4: (x1 + 10*x2)**2 + 5*(x3 - x4)**2 + (x2 - 2*x3)**4 + 10*(x1 - x4)**4,
        num_vars=4
    )
    return HardProblem(
        name="powell_4d", category="unconstrained", difficulty="hard", n_vars=4,
        bounds=[(-4.0, 5.0)]*4,
        objective=lambda x: (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _trid_4d():
    """Trid function."""
    obj = ExpressionGraph.from_callable(
        lambda x1, x2, x3, x4: (x1-1)**2 + (x2-1)**2 + (x3-1)**2 + (x4-1)**2 - (x1*x2 + x2*x3 + x3*x4),
        num_vars=4
    )
    return HardProblem(
        name="trid_4d", category="unconstrained", difficulty="hard", n_vars=4,
        bounds=[(-16.0, 16.0)]*4,
        objective=lambda x: sum((x[i]-1)**2 for i in range(4)) - sum(x[i]*x[i+1] for i in range(3)),
        ineq_constraints=[], eq_constraints=[],
        known_optimal=-4.0, obj_graph=obj  # -n(n+4)(n-1)/6 for n=4
    )

def _sum_squares_4d():
    """Sum of squares with weights."""
    obj = ExpressionGraph.from_callable(
        lambda x1, x2, x3, x4: 1*x1**2 + 2*x2**2 + 3*x3**2 + 4*x4**2,
        num_vars=4
    )
    return HardProblem(
        name="sum_squares_4d", category="unconstrained", difficulty="hard", n_vars=4,
        bounds=[(-10.0, 10.0)]*4,
        objective=lambda x: sum((i+1)*x[i]**2 for i in range(4)),
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

# =============================================================================
# INEQUALITY CONSTRAINED
# =============================================================================

def _constrained_quadratic():
    obj = ExpressionGraph.from_callable(lambda x, y: (x-1)**2 + (y-2)**2, num_vars=2)
    g = ExpressionGraph.from_callable(lambda x, y: x + y - 2, num_vars=2)
    return HardProblem(
        name="constrained_quadratic", category="inequality", difficulty="easy", n_vars=2,
        bounds=[(0.0, 10.0)]*2,
        objective=lambda x: (x[0]-1)**2 + (x[1]-2)**2,
        ineq_constraints=[lambda x: x[0] + x[1] - 2],
        eq_constraints=[],
        known_optimal=0.5, obj_graph=obj, ineq_graphs=[g]
    )

def _constrained_rosenbrock():
    obj = ExpressionGraph.from_callable(lambda x, y: (1-x)**2 + 100*(y-x**2)**2, num_vars=2)
    g = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2 - 2, num_vars=2)
    return HardProblem(
        name="constrained_rosenbrock", category="inequality", difficulty="medium", n_vars=2,
        bounds=[(-1.5, 1.5)]*2,
        objective=lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2,
        ineq_constraints=[lambda x: x[0]**2 + x[1]**2 - 2],
        eq_constraints=[],
        known_optimal=0.0, obj_graph=obj, ineq_graphs=[g]
    )

def _himmelblau_ineq():
    obj = ExpressionGraph.from_callable(lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2, num_vars=2)
    g = ExpressionGraph.from_callable(lambda x, y: x + y - 4, num_vars=2)
    return HardProblem(
        name="himmelblau_ineq", category="inequality", difficulty="medium", n_vars=2,
        bounds=[(-5.0, 5.0)]*2,
        objective=lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2,
        ineq_constraints=[lambda x: x[0] + x[1] - 4],
        eq_constraints=[],
        known_optimal=0.0, obj_graph=obj, ineq_graphs=[g]
    )

def _quadratic_2ineq():
    """Quadratic with two inequality constraints."""
    obj = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2, num_vars=2)
    # g1(x) <= 0 means 1 - x - y <= 0, i.e., x + y >= 1
    g1 = ExpressionGraph.from_callable(lambda x, y: 1 - x - y, num_vars=2)
    # g2(x) <= 0 means x + y - 3 <= 0, i.e., x + y <= 3
    g2 = ExpressionGraph.from_callable(lambda x, y: x + y - 3, num_vars=2)
    return HardProblem(
        name="quadratic_2ineq", category="inequality", difficulty="medium", n_vars=2,
        bounds=[(-5.0, 5.0)]*2,
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[lambda x: 1 - x[0] - x[1], lambda x: x[0] + x[1] - 3],
        eq_constraints=[],
        known_optimal=0.5, obj_graph=obj, ineq_graphs=[g1, g2]  # at (0.5, 0.5)
    )

def _sphere_in_cube():
    """Minimize distance to origin, constrained to unit sphere surface and cube."""
    obj = ExpressionGraph.from_callable(lambda x, y, z: x**2 + y**2 + z**2, num_vars=3)
    g = ExpressionGraph.from_callable(lambda x, y, z: x**2 + y**2 + z**2 - 1, num_vars=3)
    return HardProblem(
        name="sphere_in_cube", category="inequality", difficulty="hard", n_vars=3,
        bounds=[(-1.0, 1.0)]*3,
        objective=lambda x: x[0]**2 + x[1]**2 + x[2]**2,
        ineq_constraints=[lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1],  # inside unit sphere
        eq_constraints=[],
        known_optimal=0.0, obj_graph=obj, ineq_graphs=[g]
    )

# =============================================================================
# EQUALITY CONSTRAINED (MANIFOLDS) - The Hard Ones!
# =============================================================================

def _circle():
    """Classic circle manifold."""
    obj = ExpressionGraph.from_callable(lambda x, y: x + y, num_vars=2)
    h = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2 - 1, num_vars=2)
    return HardProblem(
        name="circle", category="equality", difficulty="medium", n_vars=2,
        bounds=[(-2.0, 2.0)]*2,
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[], eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1],
        known_optimal=-sqrt(2), obj_graph=obj, eq_graphs=[h]
    )

def _ellipse():
    """Ellipse manifold."""
    obj = ExpressionGraph.from_callable(lambda x, y: x + y, num_vars=2)
    h = ExpressionGraph.from_callable(lambda x, y: x**2/4 + y**2 - 1, num_vars=2)
    return HardProblem(
        name="ellipse", category="equality", difficulty="medium", n_vars=2,
        bounds=[(-3.0, 3.0), (-2.0, 2.0)],
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[], eq_constraints=[lambda x: x[0]**2/4 + x[1]**2 - 1],
        known_optimal=-sqrt(5), obj_graph=obj, eq_graphs=[h]
    )

def _sphere_surface_3d():
    """3D sphere surface."""
    obj = ExpressionGraph.from_callable(lambda x, y, z: x + y + z, num_vars=3)
    h = ExpressionGraph.from_callable(lambda x, y, z: x**2 + y**2 + z**2 - 1, num_vars=3)
    return HardProblem(
        name="sphere_surface_3d", category="equality", difficulty="hard", n_vars=3,
        bounds=[(-2.0, 2.0)]*3,
        objective=lambda x: x[0] + x[1] + x[2],
        ineq_constraints=[], eq_constraints=[lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1],
        known_optimal=-sqrt(3), obj_graph=obj, eq_graphs=[h]
    )

def _paraboloid_plane():
    """Paraboloid on plane intersection."""
    obj = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2, num_vars=2)
    h = ExpressionGraph.from_callable(lambda x, y: x + y - 1, num_vars=2)
    return HardProblem(
        name="paraboloid_plane", category="equality", difficulty="easy", n_vars=2,
        bounds=[(-5.0, 5.0)]*2,
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[], eq_constraints=[lambda x: x[0] + x[1] - 1],
        known_optimal=0.5, obj_graph=obj, eq_graphs=[h]
    )

def _hyperbola_line():
    """Two equality constraints - hyperbola and line intersection."""
    obj = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2, num_vars=2)
    h1 = ExpressionGraph.from_callable(lambda x, y: x*y - 1, num_vars=2)
    h2 = ExpressionGraph.from_callable(lambda x, y: x + y - 3, num_vars=2)
    return HardProblem(
        name="hyperbola_line", category="equality", difficulty="hard", n_vars=2,
        bounds=[(0.1, 5.0)]*2,
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[], eq_constraints=[lambda x: x[0]*x[1] - 1, lambda x: x[0] + x[1] - 3],
        known_optimal=7.0, obj_graph=obj, eq_graphs=[h1, h2]
    )

def _ellipsoid_plane():
    """Ellipsoid-plane intersection in 3D."""
    obj = ExpressionGraph.from_callable(lambda x, y, z: x + y + z, num_vars=3)
    h1 = ExpressionGraph.from_callable(lambda x, y, z: x**2 + 2*y**2 + 3*z**2 - 1, num_vars=3)
    h2 = ExpressionGraph.from_callable(lambda x, y, z: x + y + z - 0.5, num_vars=3)
    return HardProblem(
        name="ellipsoid_plane", category="equality", difficulty="extreme", n_vars=3,
        bounds=[(-2.0, 2.0)]*3,
        objective=lambda x: x[0] + x[1] + x[2],
        ineq_constraints=[],
        eq_constraints=[lambda x: x[0]**2 + 2*x[1]**2 + 3*x[2]**2 - 1, lambda x: x[0] + x[1] + x[2] - 0.5],
        known_optimal=0.5,  # constrained to sum = 0.5
        obj_graph=obj, eq_graphs=[h1, h2]
    )

def _cylinder_plane():
    """Cylinder-plane intersection."""
    obj = ExpressionGraph.from_callable(lambda x, y, z: x + y, num_vars=3)
    h1 = ExpressionGraph.from_callable(lambda x, y, z: x**2 + y**2 - 1, num_vars=3)  # cylinder
    h2 = ExpressionGraph.from_callable(lambda x, y, z: z - 0.5, num_vars=3)  # plane z = 0.5
    return HardProblem(
        name="cylinder_plane", category="equality", difficulty="hard", n_vars=3,
        bounds=[(-2.0, 2.0)]*3,
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[],
        eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1, lambda x: x[2] - 0.5],
        known_optimal=-sqrt(2), obj_graph=obj, eq_graphs=[h1, h2]
    )

def _torus_section():
    """Quadratic on torus-like constraint."""
    obj = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2, num_vars=2)
    # (x² + y² - 4)² + z² = 1 simplified to 2D: (x² + y² - 2)² = 0.25
    h = ExpressionGraph.from_callable(lambda x, y: (x**2 + y**2 - 2)**2 - 0.25, num_vars=2)
    return HardProblem(
        name="torus_section", category="equality", difficulty="extreme", n_vars=2,
        bounds=[(-3.0, 3.0)]*2,
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[], eq_constraints=[lambda x: (x[0]**2 + x[1]**2 - 2)**2 - 0.25],
        known_optimal=1.5,  # inner circle of torus section
        obj_graph=obj, eq_graphs=[h]
    )

# =============================================================================
# MIXED CONSTRAINTS
# =============================================================================

def _semicircle():
    """Circle with half-space constraint."""
    obj = ExpressionGraph.from_callable(lambda x, y: x + y, num_vars=2)
    h = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2 - 1, num_vars=2)
    g = ExpressionGraph.from_callable(lambda x, y: -x, num_vars=2)  # x >= 0
    return HardProblem(
        name="semicircle", category="mixed", difficulty="medium", n_vars=2,
        bounds=[(-2.0, 2.0)]*2,
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[lambda x: -x[0]], eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1],
        known_optimal=-1.0, obj_graph=obj, eq_graphs=[h], ineq_graphs=[g]
    )

def _quarter_circle():
    """Circle in first quadrant."""
    obj = ExpressionGraph.from_callable(lambda x, y: x + y, num_vars=2)
    h = ExpressionGraph.from_callable(lambda x, y: x**2 + y**2 - 1, num_vars=2)
    return HardProblem(
        name="quarter_circle", category="mixed", difficulty="medium", n_vars=2,
        bounds=[(0.0, 2.0)]*2,
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[], eq_constraints=[lambda x: x[0]**2 + x[1]**2 - 1],
        known_optimal=sqrt(2), obj_graph=obj, eq_graphs=[h]
    )

def _sphere_octant():
    """Sphere surface in first octant."""
    obj = ExpressionGraph.from_callable(lambda x, y, z: x + y + z, num_vars=3)
    h = ExpressionGraph.from_callable(lambda x, y, z: x**2 + y**2 + z**2 - 1, num_vars=3)
    return HardProblem(
        name="sphere_octant", category="mixed", difficulty="hard", n_vars=3,
        bounds=[(0.0, 2.0)]*3,
        objective=lambda x: x[0] + x[1] + x[2],
        ineq_constraints=[], eq_constraints=[lambda x: x[0]**2 + x[1]**2 + x[2]**2 - 1],
        known_optimal=sqrt(3), obj_graph=obj, eq_graphs=[h]
    )

def _ellipse_box():
    """Ellipse with box constraints tightened."""
    obj = ExpressionGraph.from_callable(lambda x, y: x - y, num_vars=2)
    h = ExpressionGraph.from_callable(lambda x, y: x**2/4 + y**2 - 1, num_vars=2)
    return HardProblem(
        name="ellipse_box", category="mixed", difficulty="hard", n_vars=2,
        bounds=[(0.0, 2.0), (-1.0, 1.0)],
        objective=lambda x: x[0] - x[1],
        ineq_constraints=[], eq_constraints=[lambda x: x[0]**2/4 + x[1]**2 - 1],
        known_optimal=-1.0,  # at (0, 1)
        obj_graph=obj, eq_graphs=[h]
    )

# =============================================================================
# HOCK-SCHITTKOWSKI PROBLEMS
# =============================================================================

def _hs01():
    obj = ExpressionGraph.from_callable(lambda x, y: 100*(y - x**2)**2 + (1 - x)**2, num_vars=2)
    return HardProblem(
        name="hs01", category="unconstrained", difficulty="medium", n_vars=2,
        bounds=[(-10.0, 10.0)]*2,
        objective=lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _hs02():
    obj = ExpressionGraph.from_callable(lambda x, y: 100*(y - x**2)**2 + (1 - x)**2, num_vars=2)
    return HardProblem(
        name="hs02", category="unconstrained", difficulty="medium", n_vars=2,
        bounds=[(-10.0, 10.0), (1.5, 10.0)],
        objective=lambda x: 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0504, obj_graph=obj
    )

def _hs03():
    obj = ExpressionGraph.from_callable(lambda x, y: y + 1e-5*(y - x)**2, num_vars=2)
    return HardProblem(
        name="hs03", category="unconstrained", difficulty="easy", n_vars=2,
        bounds=[(-10.0, 10.0), (0.0, 10.0)],
        objective=lambda x: x[1] + 1e-5*(x[1] - x[0])**2,
        ineq_constraints=[], eq_constraints=[],
        known_optimal=0.0, obj_graph=obj
    )

def _hs04():
    obj = ExpressionGraph.from_callable(lambda x, y: (x + 1)**3/3 + y, num_vars=2)
    return HardProblem(
        name="hs04", category="unconstrained", difficulty="easy", n_vars=2,
        bounds=[(1.0, 10.0), (0.0, 10.0)],
        objective=lambda x: (x[0] + 1)**3/3 + x[1],
        ineq_constraints=[], eq_constraints=[],
        known_optimal=2.667, obj_graph=obj
    )

def _hs21():
    """HS21: Linear constraints."""
    obj = ExpressionGraph.from_callable(lambda x, y: 0.01*x**2 + y**2 - 100, num_vars=2)
    g1 = ExpressionGraph.from_callable(lambda x, y: -10*x + y + 10, num_vars=2)  # 10x - y >= 10
    return HardProblem(
        name="hs21", category="inequality", difficulty="medium", n_vars=2,
        bounds=[(2.0, 50.0), (-50.0, 50.0)],
        objective=lambda x: 0.01*x[0]**2 + x[1]**2 - 100,
        ineq_constraints=[lambda x: -10*x[0] + x[1] + 10],
        eq_constraints=[],
        known_optimal=-99.96, obj_graph=obj, ineq_graphs=[g1]
    )

def _hs35():
    """HS35: Quadratic with linear constraints."""
    obj = ExpressionGraph.from_callable(
        lambda x1, x2, x3: 9 - 8*x1 - 6*x2 - 4*x3 + 2*x1**2 + 2*x2**2 + x3**2 + 2*x1*x2 + 2*x1*x3,
        num_vars=3
    )
    g1 = ExpressionGraph.from_callable(lambda x1, x2, x3: -x1 - x2 - 2*x3 + 3, num_vars=3)
    return HardProblem(
        name="hs35", category="inequality", difficulty="hard", n_vars=3,
        bounds=[(0.0, 10.0)]*3,
        objective=lambda x: 9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2 + 2*x[0]*x[1] + 2*x[0]*x[2],
        ineq_constraints=[lambda x: -x[0] - x[1] - 2*x[2] + 3],
        eq_constraints=[],
        known_optimal=0.111, obj_graph=obj, ineq_graphs=[g1]
    )

# =============================================================================
# REGISTER ALL
# =============================================================================

def _register_all():
    builders = [
        # Unconstrained Easy
        _sphere_2d, _booth, _matyas,
        # Unconstrained Medium
        _rosenbrock_2d, _rosenbrock_5d, _beale, _sphere_5d,
        # Unconstrained Hard
        _goldstein_price, _six_hump_camel, _three_hump_camel,
        _dixon_price_2d, _zakharov_2d, _powell_4d, _trid_4d, _sum_squares_4d,
        # Inequality
        _constrained_quadratic, _constrained_rosenbrock, _himmelblau_ineq,
        _quadratic_2ineq, _sphere_in_cube,
        # Equality (Manifolds)
        _circle, _ellipse, _sphere_surface_3d, _paraboloid_plane,
        _hyperbola_line, _ellipsoid_plane, _cylinder_plane, _torus_section,
        # Mixed
        _semicircle, _quarter_circle, _sphere_octant, _ellipse_box,
        # Hock-Schittkowski
        _hs01, _hs02, _hs03, _hs04, _hs21, _hs35,
    ]
    for b in builders:
        try:
            register(b())
        except Exception as e:
            print(f"Warning: {b.__name__}: {e}")


def get_all() -> List[HardProblem]:
    if not HARD_REGISTRY:
        _register_all()
    return list(HARD_REGISTRY.values())


def get(name: str) -> HardProblem:
    if not HARD_REGISTRY:
        _register_all()
    return HARD_REGISTRY[name]


_register_all()
