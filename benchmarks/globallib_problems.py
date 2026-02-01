"""
GLOBALLib Problem Definitions

Standard global optimization test problems.
Each problem is rigorously defined with:
- Objective function as expression graph
- Constraint functions as expression graphs
- Variable bounds
- Problem metadata

These are NOT validated against external optimal values.
Certification is via gap closure: UB - LB ≤ ε.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

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


# =============================================================================
# Problem Registry
# =============================================================================

PROBLEM_REGISTRY: Dict[str, GLOBALLibProblem] = {}


def register_problem(problem: GLOBALLibProblem):
    """Register a problem in the registry."""
    PROBLEM_REGISTRY[problem.name] = problem


def get_problem(name: str) -> GLOBALLibProblem:
    """Get a problem by name."""
    if name not in PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem: {name}")
    return PROBLEM_REGISTRY[name]


def get_all_problems() -> List[GLOBALLibProblem]:
    """Get all registered problems."""
    return list(PROBLEM_REGISTRY.values())


# =============================================================================
# Problem Definitions
# =============================================================================

def _build_problem_rosenbrock_2d():
    """
    Rosenbrock function (2D)
    min (1-x)² + 100(y-x²)²
    x, y ∈ [-5, 5]

    Global minimum at (1, 1) with f* = 0
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (1 - x)**2 + 100*(y - x**2)**2,
        num_vars=2
    )

    return GLOBALLibProblem(
        name="rosenbrock_2d",
        description="Rosenbrock function (2D)",
        n_vars=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        objective=lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
        ineq_constraints=[],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_circle():
    """
    Circle manifold optimization
    min x + y
    s.t. x² + y² = 1
    x, y ∈ [-2, 2]

    Global minimum at (-1/√2, -1/√2) with f* = -√2
    """
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
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[eq_graph]
    )


def _build_problem_semicircle():
    """
    Semicircle optimization (right half)
    min x + y
    s.t. x² + y² = 1
         x ≥ 0
    x, y ∈ [-2, 2]

    Global minimum at (0, -1) with f* = -1
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y,
        num_vars=2
    )
    eq_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 - 1,
        num_vars=2
    )
    ineq_graph = ExpressionGraph.from_callable(
        lambda x, y: -x,  # -x ≤ 0 means x ≥ 0
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
        obj_graph=obj_graph,
        ineq_graphs=[ineq_graph],
        eq_graphs=[eq_graph]
    )


def _build_problem_sphere_2d():
    """
    Sphere function (2D)
    min x² + y²
    x, y ∈ [-5, 5]

    Global minimum at (0, 0) with f* = 0
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2,
        num_vars=2
    )

    return GLOBALLibProblem(
        name="sphere_2d",
        description="Sphere function (2D)",
        n_vars=2,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_booth():
    """
    Booth function
    min (x + 2y - 7)² + (2x + y - 5)²
    x, y ∈ [-10, 10]

    Global minimum at (1, 3) with f* = 0
    """
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
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_three_hump_camel():
    """
    Three-hump camel function
    min 2x² - 1.05x⁴ + x⁶/6 + xy + y²
    x, y ∈ [-5, 5]

    Global minimum at (0, 0) with f* = 0
    """
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
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_beale():
    """
    Beale function
    min (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    x, y ∈ [-4.5, 4.5]

    Global minimum at (3, 0.5) with f* = 0
    """
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
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_constrained_quadratic():
    """
    Constrained quadratic
    min (x-1)² + (y-2)²
    s.t. x + y ≤ 2
         x ≥ 0, y ≥ 0
    x, y ∈ [0, 10]

    Global minimum at (0.5, 1.5) with f* = 0.5
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (x - 1)**2 + (y - 2)**2,
        num_vars=2
    )
    ineq_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y - 2,  # x + y ≤ 2
        num_vars=2
    )

    return GLOBALLibProblem(
        name="constrained_quadratic",
        description="Constrained quadratic with linear constraint",
        n_vars=2,
        bounds=[(0.0, 10.0), (0.0, 10.0)],
        objective=lambda x: (x[0] - 1)**2 + (x[1] - 2)**2,
        ineq_constraints=[lambda x: x[0] + x[1] - 2],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[ineq_graph],
        eq_graphs=[]
    )


def _build_problem_infeasible():
    """
    Infeasible problem (for testing UNSAT)
    min x + y
    s.t. x² + y² ≤ 1
         x + y ≥ 10
    x, y ∈ [-2, 2]

    No feasible solution exists (circle doesn't intersect half-plane)
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: x + y,
        num_vars=2
    )
    ineq1 = ExpressionGraph.from_callable(
        lambda x, y: x**2 + y**2 - 1,  # x² + y² ≤ 1
        num_vars=2
    )
    ineq2 = ExpressionGraph.from_callable(
        lambda x, y: -(x + y - 10),  # -(x + y - 10) ≤ 0 means x + y ≥ 10
        num_vars=2
    )

    return GLOBALLibProblem(
        name="infeasible",
        description="Infeasible problem (unit circle + x+y≥10)",
        n_vars=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[
            lambda x: x[0]**2 + x[1]**2 - 1,
            lambda x: -(x[0] + x[1] - 10)
        ],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[ineq1, ineq2],
        eq_graphs=[]
    )


def _build_problem_hs01():
    """
    Hock-Schittkowski Problem #1 (modified)
    min (x-10)² + (y-5)²
    s.t. (x-5)² + (y-5)² ≤ 25
    x, y ∈ [-10, 20]

    Feasible region is a disk centered at (5,5) with radius 5.
    Unconstrained minimum is at (10,5), which is on the boundary.
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (x - 10)**2 + (y - 5)**2,
        num_vars=2
    )
    ineq_graph = ExpressionGraph.from_callable(
        lambda x, y: (x - 5)**2 + (y - 5)**2 - 25,
        num_vars=2
    )

    return GLOBALLibProblem(
        name="hs01",
        description="HS01-like: quadratic with disk constraint",
        n_vars=2,
        bounds=[(-10.0, 20.0), (-10.0, 20.0)],
        objective=lambda x: (x[0] - 10)**2 + (x[1] - 5)**2,
        ineq_constraints=[lambda x: (x[0] - 5)**2 + (x[1] - 5)**2 - 25],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[ineq_graph],
        eq_graphs=[]
    )


def _build_problem_ellipse():
    """
    Ellipse manifold optimization
    min x + y
    s.t. x²/4 + y² = 1  (ellipse with semi-axes 2 and 1)
    x, y ∈ [-3, 3]

    Global minimum at (-2/√5 * 2, -√5/√5) ≈ (-1.789, -0.894)
    """
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
        bounds=[(-3.0, 3.0), (-3.0, 3.0)],
        objective=lambda x: x[0] + x[1],
        ineq_constraints=[],
        eq_constraints=[lambda x: x[0]**2/4 + x[1]**2 - 1],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[eq_graph]
    )


def _build_problem_hyperbola_intersection():
    """
    Intersection of hyperbola and line
    min x² + y²
    s.t. xy = 1 (hyperbola)
         x + y = 3 (line)
    x, y ∈ [0, 5]

    Solutions: From xy=1 and x+y=3, we get x² - 3x + 1 = 0
    x = (3 ± √5)/2, giving x ≈ 0.382 or 2.618
    Both solutions give f* = 7
    """
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
        name="hyperbola_intersection",
        description="Intersection of hyperbola xy=1 and line x+y=3",
        n_vars=2,
        bounds=[(0.0, 5.0), (0.0, 5.0)],
        objective=lambda x: x[0]**2 + x[1]**2,
        ineq_constraints=[],
        eq_constraints=[
            lambda x: x[0]*x[1] - 1,
            lambda x: x[0] + x[1] - 3
        ],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[eq1, eq2]
    )


def _build_problem_sphere_3d():
    """
    Sphere function (3D)
    min x² + y² + z²
    x, y, z ∈ [-5, 5]

    Global minimum at (0, 0, 0) with f* = 0
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y, z: x**2 + y**2 + z**2,
        num_vars=3
    )

    return GLOBALLibProblem(
        name="sphere_3d",
        description="Sphere function (3D)",
        n_vars=3,
        bounds=[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
        objective=lambda x: x[0]**2 + x[1]**2 + x[2]**2,
        ineq_constraints=[],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_sphere_5d():
    """
    Sphere function (5D)
    min sum(x_i²)
    x_i ∈ [-5, 5]

    Global minimum at origin with f* = 0
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x1, x2, x3, x4, x5: x1**2 + x2**2 + x3**2 + x4**2 + x5**2,
        num_vars=5
    )

    return GLOBALLibProblem(
        name="sphere_5d",
        description="Sphere function (5D)",
        n_vars=5,
        bounds=[(-5.0, 5.0)] * 5,
        objective=lambda x: sum(xi**2 for xi in x),
        ineq_constraints=[],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_dixon_price_2d():
    """
    Dixon-Price function (2D)
    min (x - 1)² + 2(2y² - x)²
    x, y ∈ [-10, 10]

    Global minimum at (1, 1/√2) with f* = 0
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y: (x - 1)**2 + 2*(2*y**2 - x)**2,
        num_vars=2
    )

    return GLOBALLibProblem(
        name="dixon_price_2d",
        description="Dixon-Price function (2D)",
        n_vars=2,
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
        objective=lambda x: (x[0] - 1)**2 + 2*(2*x[1]**2 - x[0])**2,
        ineq_constraints=[],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_goldstein_price():
    """
    Goldstein-Price function
    min [1 + (x + y + 1)²(19 - 14x + 3x² - 14y + 6xy + 3y²)]
        × [30 + (2x - 3y)²(18 - 32x + 12x² + 48y - 36xy + 27y²)]
    x, y ∈ [-2, 2]

    Global minimum at (0, -1) with f* = 3
    """
    def gp_obj(x, y):
        term1 = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
        term2 = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)
        return term1 * term2

    obj_graph = ExpressionGraph.from_callable(gp_obj, num_vars=2)

    return GLOBALLibProblem(
        name="goldstein_price",
        description="Goldstein-Price function",
        n_vars=2,
        bounds=[(-2.0, 2.0), (-2.0, 2.0)],
        objective=lambda x: gp_obj(x[0], x[1]),
        ineq_constraints=[],
        eq_constraints=[],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_matyas():
    """
    Matyas function
    min 0.26(x² + y²) - 0.48xy
    x, y ∈ [-10, 10]

    Global minimum at (0, 0) with f* = 0
    """
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
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[]
    )


def _build_problem_quadratic_cone():
    """
    Quadratic optimization on cone section
    min x² + y² + z²
    s.t. x + y + z = 1 (plane)
         x² + y² = z² (cone)
    x, y, z ∈ [-2, 2]
    """
    obj_graph = ExpressionGraph.from_callable(
        lambda x, y, z: x**2 + y**2 + z**2,
        num_vars=3
    )
    eq1 = ExpressionGraph.from_callable(
        lambda x, y, z: x + y + z - 1,
        num_vars=3
    )
    eq2 = ExpressionGraph.from_callable(
        lambda x, y, z: x**2 + y**2 - z**2,
        num_vars=3
    )

    return GLOBALLibProblem(
        name="quadratic_cone",
        description="Quadratic on cone-plane intersection",
        n_vars=3,
        bounds=[(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
        objective=lambda x: x[0]**2 + x[1]**2 + x[2]**2,
        ineq_constraints=[],
        eq_constraints=[
            lambda x: x[0] + x[1] + x[2] - 1,
            lambda x: x[0]**2 + x[1]**2 - x[2]**2
        ],
        obj_graph=obj_graph,
        ineq_graphs=[],
        eq_graphs=[eq1, eq2]
    )


# =============================================================================
# Register all problems
# =============================================================================

def _register_all():
    """Register all problems."""
    builders = [
        _build_problem_rosenbrock_2d,
        _build_problem_circle,
        _build_problem_semicircle,
        _build_problem_sphere_2d,
        _build_problem_booth,
        _build_problem_three_hump_camel,
        _build_problem_beale,
        _build_problem_constrained_quadratic,
        _build_problem_infeasible,
        _build_problem_hs01,
        _build_problem_ellipse,
        _build_problem_hyperbola_intersection,
        _build_problem_sphere_3d,
        _build_problem_sphere_5d,
        _build_problem_dixon_price_2d,
        _build_problem_goldstein_price,
        _build_problem_matyas,
        _build_problem_quadratic_cone,
    ]

    for builder in builders:
        register_problem(builder())


# Auto-register on import
_register_all()
