"""
McCormick Convex Relaxations (Tier 1 Bounds)

McCormick relaxations provide convex under/over-estimators for
factorable functions. Each elementary operation has known convex
and concave envelopes over a box.

The relaxation produces a Linear Program (LP) that can be solved
to obtain tighter lower bounds than pure interval arithmetic.

Key operations:
- Bilinear: x*y on [xl,xu] x [yl,yu] has McCormick envelopes
- Univariate convex: direct use
- Univariate concave: concave envelope
- General: piecewise linear approximation
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .interval import Interval, IntervalEvaluator
from ..expr_graph import (
    ExpressionGraph,
    ExprNode,
    Variable,
    Constant,
    UnaryOp,
    BinaryOp,
    OpType,
)


@dataclass
class McCormickBounds:
    """
    McCormick relaxation bounds for a node.

    cv: convex underestimator value at a point
    cc: concave overestimator value at a point
    cv_grad: gradient of convex underestimator (for LP)
    cc_grad: gradient of concave overestimator
    interval: interval bounds [lo, hi]
    """
    cv: float  # convex underestimator
    cc: float  # concave overestimator
    cv_grad: Optional[np.ndarray] = None
    cc_grad: Optional[np.ndarray] = None
    interval: Interval = None

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "cv": self.cv,
            "cc": self.cc,
            "cv_grad": self.cv_grad.tolist() if self.cv_grad is not None else None,
            "cc_grad": self.cc_grad.tolist() if self.cc_grad is not None else None,
            "interval": self.interval.to_canonical() if self.interval else None
        }


@dataclass
class LinearConstraint:
    """
    A linear constraint: a^T x <= b or a^T x >= b.
    """
    coeffs: np.ndarray  # Coefficient vector a
    rhs: float          # Right-hand side b
    sense: str          # "<=" or ">="

    def to_canonical(self) -> Dict[str, Any]:
        return {
            "coeffs": self.coeffs.tolist(),
            "rhs": self.rhs,
            "sense": self.sense
        }


@dataclass
class RelaxationLP:
    """
    Linear programming relaxation of the original problem.

    Variables: original x_i plus auxiliary w_j for intermediate nodes
    Objective: minimize w_output (linear in w)
    Constraints: McCormick envelopes for each operation
    """
    n_original_vars: int
    n_aux_vars: int
    objective_coeffs: np.ndarray  # Linear objective
    constraints: List[LinearConstraint] = field(default_factory=list)
    var_lower: np.ndarray = None  # Variable lower bounds
    var_upper: np.ndarray = None  # Variable upper bounds

    @property
    def n_total_vars(self) -> int:
        return self.n_original_vars + self.n_aux_vars

    def solve(self) -> Tuple[float, Optional[np.ndarray]]:
        """
        Solve the LP relaxation.

        Returns:
            Tuple of (lower_bound, dual_solution)
        """
        try:
            from scipy.optimize import linprog

            # Build constraint matrix
            n = self.n_total_vars
            n_cons = len(self.constraints)

            if n_cons == 0:
                # No constraints, just bound-based
                lb = 0.0
                for i in range(n):
                    if self.objective_coeffs[i] > 0:
                        lb += self.objective_coeffs[i] * self.var_lower[i]
                    else:
                        lb += self.objective_coeffs[i] * self.var_upper[i]
                return lb, None

            # Separate into <= and >= constraints
            A_ub = []
            b_ub = []

            for con in self.constraints:
                if con.sense == "<=":
                    A_ub.append(con.coeffs)
                    b_ub.append(con.rhs)
                else:  # >=
                    A_ub.append(-con.coeffs)
                    b_ub.append(-con.rhs)

            if A_ub:
                A_ub = np.array(A_ub)
                b_ub = np.array(b_ub)
            else:
                A_ub = None
                b_ub = None

            bounds = [(self.var_lower[i], self.var_upper[i]) for i in range(n)]

            result = linprog(
                c=self.objective_coeffs,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='highs'
            )

            if result.success:
                return result.fun, result.x
            else:
                # Fall back to interval bound
                return float('-inf'), None

        except ImportError:
            # SciPy not available, use simple bound
            return float('-inf'), None


class McCormickEvaluator:
    """
    Builds McCormick relaxation for an expression graph.

    For each node, computes:
    1. Interval bounds [lo, hi] via natural interval extension
    2. Convex underestimator cv(x)
    3. Concave overestimator cc(x)

    The final LP minimizes the convex underestimator of the objective.
    """

    def __init__(self, graph: ExpressionGraph, n_vars: int):
        self.graph = graph
        self.n_vars = n_vars
        self._intervals: Dict[int, Interval] = {}
        self._mccormick: Dict[int, McCormickBounds] = {}

    def build_relaxation(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> RelaxationLP:
        """
        Build the LP relaxation over a box.

        Args:
            lower: Lower bounds of the box
            upper: Upper bounds of the box

        Returns:
            RelaxationLP that can be solved for a lower bound
        """
        # First, compute interval bounds for all nodes
        var_intervals = {
            i: Interval(lower[i], upper[i])
            for i in range(self.n_vars)
        }

        interval_eval = IntervalEvaluator(self.graph)
        _, self._intervals = interval_eval.evaluate(var_intervals)

        # Now build McCormick relaxation
        # Each node gets an auxiliary variable in the LP
        nodes = self.graph.topological_order()
        node_to_aux = {}  # node_id -> aux var index
        aux_idx = self.n_vars

        for node in nodes:
            if isinstance(node, Variable):
                node_to_aux[node.node_id] = node.var_index
            else:
                node_to_aux[node.node_id] = aux_idx
                aux_idx += 1

        n_aux = aux_idx - self.n_vars
        n_total = self.n_vars + n_aux

        # Build constraints
        constraints = []

        # Variable bounds
        var_lower = np.full(n_total, float('-inf'))
        var_upper = np.full(n_total, float('inf'))

        # Original variables
        var_lower[:self.n_vars] = lower
        var_upper[:self.n_vars] = upper

        # Auxiliary variables from intervals
        for node in nodes:
            idx = node_to_aux[node.node_id]
            if idx >= self.n_vars:
                ivl = self._intervals[node.node_id]
                var_lower[idx] = ivl.lo
                var_upper[idx] = ivl.hi

        # Add McCormick constraints for each operation
        for node in nodes:
            if isinstance(node, (Variable, Constant)):
                continue

            w_out = node_to_aux[node.node_id]

            if isinstance(node, UnaryOp):
                w_in = node_to_aux[node.child.node_id]
                ivl_in = self._intervals[node.child.node_id]
                ivl_out = self._intervals[node.node_id]

                cons = self._unary_mccormick_constraints(
                    node.op, w_in, w_out, ivl_in, ivl_out, n_total
                )
                constraints.extend(cons)

            elif isinstance(node, BinaryOp):
                w_left = node_to_aux[node.left.node_id]
                w_right = node_to_aux[node.right.node_id]
                ivl_left = self._intervals[node.left.node_id]
                ivl_right = self._intervals[node.right.node_id]
                ivl_out = self._intervals[node.node_id]

                cons = self._binary_mccormick_constraints(
                    node.op, w_left, w_right, w_out,
                    ivl_left, ivl_right, ivl_out, n_total
                )
                constraints.extend(cons)

        # Objective: minimize the output variable
        objective = np.zeros(n_total)
        objective[node_to_aux[self.graph.output_node.node_id]] = 1.0

        return RelaxationLP(
            n_original_vars=self.n_vars,
            n_aux_vars=n_aux,
            objective_coeffs=objective,
            constraints=constraints,
            var_lower=var_lower,
            var_upper=var_upper
        )

    def _unary_mccormick_constraints(
        self,
        op: OpType,
        w_in: int,
        w_out: int,
        ivl_in: Interval,
        ivl_out: Interval,
        n_vars: int
    ) -> List[LinearConstraint]:
        """
        Generate McCormick constraints for unary operations.

        For convex functions: secant overestimator, tangent underestimator
        For concave functions: tangent overestimator, secant underestimator
        """
        cons = []
        xl, xu = ivl_in.lo, ivl_in.hi

        if xu - xl < 1e-10:
            # Near-point interval, just bound
            return cons

        if op == OpType.NEG:
            # w_out = -w_in (linear)
            c = np.zeros(n_vars)
            c[w_out] = 1.0
            c[w_in] = 1.0
            cons.append(LinearConstraint(c, 0.0, "<="))
            cons.append(LinearConstraint(-c, 0.0, "<="))

        elif op == OpType.SQUARE:
            # x^2 is convex: secant over, tangent under
            # Secant: w <= (xl + xu) * x - xl * xu
            c = np.zeros(n_vars)
            c[w_out] = 1.0
            c[w_in] = -(xl + xu)
            cons.append(LinearConstraint(c, -xl * xu, "<="))

            # Tangent at midpoint: w >= 2*xm*x - xm^2
            xm = (xl + xu) / 2
            c = np.zeros(n_vars)
            c[w_out] = -1.0
            c[w_in] = 2 * xm
            cons.append(LinearConstraint(c, xm * xm, "<="))

        elif op == OpType.EXP:
            # exp is convex: secant over, tangent under
            if xl > -700 and xu < 700:  # Avoid overflow
                fl, fu = np.exp(xl), np.exp(xu)
                # Secant: w <= fl + (fu - fl) / (xu - xl) * (x - xl)
                slope = (fu - fl) / (xu - xl)
                c = np.zeros(n_vars)
                c[w_out] = 1.0
                c[w_in] = -slope
                cons.append(LinearConstraint(c, fl - slope * xl, "<="))

                # Tangent at xm
                xm = (xl + xu) / 2
                fm = np.exp(xm)
                c = np.zeros(n_vars)
                c[w_out] = -1.0
                c[w_in] = fm
                cons.append(LinearConstraint(c, fm * xm - fm, "<="))

        elif op == OpType.LOG:
            # log is concave: secant under, tangent over
            if xl > 0:
                fl, fu = np.log(xl), np.log(xu)
                # Secant: w >= fl + (fu - fl) / (xu - xl) * (x - xl)
                slope = (fu - fl) / (xu - xl)
                c = np.zeros(n_vars)
                c[w_out] = -1.0
                c[w_in] = slope
                cons.append(LinearConstraint(c, slope * xl - fl, "<="))

                # Tangent at xm: w <= log(xm) + (1/xm) * (x - xm)
                xm = (xl + xu) / 2
                c = np.zeros(n_vars)
                c[w_out] = 1.0
                c[w_in] = -1.0 / xm
                cons.append(LinearConstraint(c, np.log(xm) - 1, "<="))

        elif op == OpType.SQRT:
            # sqrt is concave: secant under, tangent over
            if xl >= 0:
                fl, fu = np.sqrt(xl), np.sqrt(xu)
                # Secant
                slope = (fu - fl) / (xu - xl) if xu > xl else 0
                c = np.zeros(n_vars)
                c[w_out] = -1.0
                c[w_in] = slope
                cons.append(LinearConstraint(c, slope * xl - fl, "<="))

        elif op == OpType.ABS:
            # |x| is convex
            # w >= x, w >= -x
            c1 = np.zeros(n_vars)
            c1[w_out] = -1.0
            c1[w_in] = 1.0
            cons.append(LinearConstraint(c1, 0.0, "<="))

            c2 = np.zeros(n_vars)
            c2[w_out] = -1.0
            c2[w_in] = -1.0
            cons.append(LinearConstraint(c2, 0.0, "<="))

        # For trig functions, use interval bounds (complex envelopes)
        # This is a simplification; full implementation would add piecewise linear

        return cons

    def _binary_mccormick_constraints(
        self,
        op: OpType,
        w_left: int,
        w_right: int,
        w_out: int,
        ivl_left: Interval,
        ivl_right: Interval,
        ivl_out: Interval,
        n_vars: int
    ) -> List[LinearConstraint]:
        """
        Generate McCormick constraints for binary operations.

        The bilinear product x*y has the famous McCormick envelopes.
        """
        cons = []
        xl, xu = ivl_left.lo, ivl_left.hi
        yl, yu = ivl_right.lo, ivl_right.hi

        if op == OpType.ADD:
            # w = x + y (linear)
            c = np.zeros(n_vars)
            c[w_out] = 1.0
            c[w_left] = -1.0
            c[w_right] = -1.0
            cons.append(LinearConstraint(c, 0.0, "<="))
            cons.append(LinearConstraint(-c, 0.0, "<="))

        elif op == OpType.SUB:
            # w = x - y (linear)
            c = np.zeros(n_vars)
            c[w_out] = 1.0
            c[w_left] = -1.0
            c[w_right] = 1.0
            cons.append(LinearConstraint(c, 0.0, "<="))
            cons.append(LinearConstraint(-c, 0.0, "<="))

        elif op == OpType.MUL:
            # McCormick envelopes for w = x * y
            # Convex underestimator (max of two linear):
            #   w >= xl*y + yl*x - xl*yl
            #   w >= xu*y + yu*x - xu*yu
            # Concave overestimator (min of two linear):
            #   w <= xl*y + yu*x - xl*yu
            #   w <= xu*y + yl*x - xu*yl

            # Under 1: w >= xl*y + yl*x - xl*yl
            c = np.zeros(n_vars)
            c[w_out] = -1.0
            c[w_left] = yl
            c[w_right] = xl
            cons.append(LinearConstraint(c, xl * yl, "<="))

            # Under 2: w >= xu*y + yu*x - xu*yu
            c = np.zeros(n_vars)
            c[w_out] = -1.0
            c[w_left] = yu
            c[w_right] = xu
            cons.append(LinearConstraint(c, xu * yu, "<="))

            # Over 1: w <= xl*y + yu*x - xl*yu
            c = np.zeros(n_vars)
            c[w_out] = 1.0
            c[w_left] = -yu
            c[w_right] = -xl
            cons.append(LinearConstraint(c, -xl * yu, "<="))

            # Over 2: w <= xu*y + yl*x - xu*yl
            c = np.zeros(n_vars)
            c[w_out] = 1.0
            c[w_left] = -yl
            c[w_right] = -xu
            cons.append(LinearConstraint(c, -xu * yl, "<="))

        elif op == OpType.DIV:
            # w = x / y: reformulate as w * y = x
            # Use McCormick on w * y with interval of w from ivl_out
            # This is a simplification; proper handling requires more care
            if yl > 0 or yu < 0:  # y doesn't contain zero
                # Simple secant bounds
                pass  # Use interval bounds for now

        elif op == OpType.POW:
            # x^y: use interval bounds for general case
            # Special case: y is constant integer
            if yl == yu and yl == int(yl):
                n = int(yl)
                if n == 2:
                    # x^2: convex, use square constraints
                    c = np.zeros(n_vars)
                    c[w_out] = 1.0
                    c[w_left] = -(xl + xu)
                    cons.append(LinearConstraint(c, -xl * xu, "<="))

        return cons


class McCormickRelaxation:
    """
    High-level interface for McCormick relaxation lower bounds.
    """

    def __init__(
        self,
        objective: ExpressionGraph,
        n_vars: int,
        ineq_graphs: List[ExpressionGraph] = None,
        eq_graphs: List[ExpressionGraph] = None
    ):
        self.objective = objective
        self.n_vars = n_vars
        self.ineq_graphs = ineq_graphs or []
        self.eq_graphs = eq_graphs or []

    def compute_lower_bound(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute a certified lower bound via McCormick relaxation.

        Args:
            lower: Lower bounds of the region
            upper: Upper bounds of the region

        Returns:
            Tuple of (lower_bound, certificate)
        """
        evaluator = McCormickEvaluator(self.objective, self.n_vars)
        lp = evaluator.build_relaxation(lower, upper)

        lb, dual = lp.solve()

        certificate = {
            "relaxation_type": "mccormick",
            "lower_bound": lb,
            "n_constraints": len(lp.constraints),
            "n_variables": lp.n_total_vars
        }

        return lb, certificate

    def check_feasibility(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Check if the region might be feasible.

        Returns:
            Tuple of (status, certificate)
            status is "empty" or "maybe"
        """
        # Check each inequality constraint
        for i, g in enumerate(self.ineq_graphs):
            evaluator = McCormickEvaluator(g, self.n_vars)
            lp = evaluator.build_relaxation(lower, upper)
            lb, _ = lp.solve()

            if lb > 0:  # g(x) > 0 implies infeasible
                return "empty", {
                    "reason": f"inequality_{i}_violated",
                    "lower_bound": lb
                }

        # Check each equality constraint
        for j, h in enumerate(self.eq_graphs):
            evaluator = McCormickEvaluator(h, self.n_vars)
            lp = evaluator.build_relaxation(lower, upper)
            lb, _ = lp.solve()

            # For h(x) = 0, need both h(x) >= 0 and h(x) <= 0
            # If lower bound of h > 0, infeasible
            if lb > 1e-8:
                return "empty", {
                    "reason": f"equality_{j}_lb_positive",
                    "lower_bound": lb
                }

            # Check upper bound (min of -h)
            # This would require another LP, simplify for now

        return "maybe", {}

    def compute_constrained_lower_bound(
        self,
        lower: np.ndarray,
        upper: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute a certified lower bound including constraint relaxations.

        This builds a single LP that:
        1. Minimizes the McCormick relaxation of the objective
        2. Includes McCormick relaxations of g_i(x) ≤ 0
        3. Includes McCormick relaxations of h_j(x) = 0

        Args:
            lower: Lower bounds of the region
            upper: Upper bounds of the region

        Returns:
            Tuple of (lower_bound, certificate)
        """
        try:
            from scipy.optimize import linprog

            # Build the main LP for objective
            obj_eval = McCormickEvaluator(self.objective, self.n_vars)
            obj_lp = obj_eval.build_relaxation(lower, upper)

            # Start with objective LP structure
            n_total = obj_lp.n_total_vars
            all_constraints = list(obj_lp.constraints)
            var_lower = obj_lp.var_lower.copy()
            var_upper = obj_lp.var_upper.copy()

            # Track auxiliary variables offset
            aux_offset = n_total

            # Add inequality constraints g_i(x) ≤ 0
            # For each g_i, we add: g_i^cv ≤ 0 (convex underestimator ≤ 0)
            for i, g in enumerate(self.ineq_graphs):
                g_eval = McCormickEvaluator(g, self.n_vars)
                g_lp = g_eval.build_relaxation(lower, upper)

                # Add aux variables for this constraint
                n_g_aux = g_lp.n_aux_vars
                new_n_total = n_total + n_g_aux

                # Extend bounds arrays
                new_var_lower = np.full(new_n_total, float('-inf'))
                new_var_upper = np.full(new_n_total, float('inf'))
                new_var_lower[:n_total] = var_lower
                new_var_upper[:n_total] = var_upper
                new_var_lower[n_total:] = g_lp.var_lower[self.n_vars:]
                new_var_upper[n_total:] = g_lp.var_upper[self.n_vars:]

                var_lower = new_var_lower
                var_upper = new_var_upper

                # Add g constraints with offset
                for con in g_lp.constraints:
                    new_coeffs = np.zeros(new_n_total)
                    new_coeffs[:self.n_vars] = con.coeffs[:self.n_vars]
                    # Map aux vars
                    for j in range(self.n_vars, len(con.coeffs)):
                        new_coeffs[n_total + j - self.n_vars] = con.coeffs[j]
                    all_constraints.append(LinearConstraint(
                        new_coeffs, con.rhs, con.sense
                    ))

                # Add constraint: g_output ≤ 0
                # Find output variable for g
                g_output_idx = n_total + g_lp.n_aux_vars - 1  # Last aux var
                c = np.zeros(new_n_total)
                c[g_output_idx] = 1.0
                all_constraints.append(LinearConstraint(c, 0.0, "<="))

                n_total = new_n_total

            # Add equality constraints h_j(x) = 0
            # For each h_j, add: h_j^cv ≤ 0 and -h_j^cv ≤ 0
            for j, h in enumerate(self.eq_graphs):
                h_eval = McCormickEvaluator(h, self.n_vars)
                h_lp = h_eval.build_relaxation(lower, upper)

                # Add aux variables
                n_h_aux = h_lp.n_aux_vars
                new_n_total = n_total + n_h_aux

                new_var_lower = np.full(new_n_total, float('-inf'))
                new_var_upper = np.full(new_n_total, float('inf'))
                new_var_lower[:n_total] = var_lower
                new_var_upper[:n_total] = var_upper
                new_var_lower[n_total:] = h_lp.var_lower[self.n_vars:]
                new_var_upper[n_total:] = h_lp.var_upper[self.n_vars:]

                var_lower = new_var_lower
                var_upper = new_var_upper

                # Add h constraints
                for con in h_lp.constraints:
                    new_coeffs = np.zeros(new_n_total)
                    new_coeffs[:self.n_vars] = con.coeffs[:self.n_vars]
                    for k in range(self.n_vars, len(con.coeffs)):
                        new_coeffs[n_total + k - self.n_vars] = con.coeffs[k]
                    all_constraints.append(LinearConstraint(
                        new_coeffs, con.rhs, con.sense
                    ))

                # Add constraint: h_output = 0 (via h ≤ 0 and h ≥ 0)
                h_output_idx = n_total + n_h_aux - 1
                c = np.zeros(new_n_total)
                c[h_output_idx] = 1.0
                all_constraints.append(LinearConstraint(c, 0.0, "<="))
                all_constraints.append(LinearConstraint(-c, 0.0, "<="))

                n_total = new_n_total

            # Build objective coefficients (pad to new size)
            objective = np.zeros(n_total)
            objective[:len(obj_lp.objective_coeffs)] = obj_lp.objective_coeffs

            # Solve the LP
            A_ub = []
            b_ub = []

            for con in all_constraints:
                # Pad constraint to full size
                padded = np.zeros(n_total)
                padded[:len(con.coeffs)] = con.coeffs

                if con.sense == "<=":
                    A_ub.append(padded)
                    b_ub.append(con.rhs)
                else:  # >=
                    A_ub.append(-padded)
                    b_ub.append(-con.rhs)

            if A_ub:
                A_ub = np.array(A_ub)
                b_ub = np.array(b_ub)
            else:
                A_ub = None
                b_ub = None

            bounds = [(var_lower[i], var_upper[i]) for i in range(n_total)]

            result = linprog(
                c=objective,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='highs'
            )

            if result.success:
                lb = result.fun
            else:
                lb = float('-inf')

            certificate = {
                "relaxation_type": "mccormick_constrained",
                "lower_bound": lb,
                "n_constraints": len(all_constraints),
                "n_variables": n_total,
                "n_ineq_constraints": len(self.ineq_graphs),
                "n_eq_constraints": len(self.eq_graphs),
                "solver_status": result.message if hasattr(result, 'message') else 'unknown'
            }

            return lb, certificate

        except Exception as e:
            # Fallback to simple interval bound
            from .interval import IntervalEvaluator
            var_intervals = {
                i: Interval(lower[i], upper[i])
                for i in range(self.n_vars)
            }
            evaluator = IntervalEvaluator(self.objective)
            interval, _ = evaluator.evaluate(var_intervals)

            return interval.lo, {
                "relaxation_type": "interval_fallback",
                "lower_bound": interval.lo,
                "error": str(e)
            }
