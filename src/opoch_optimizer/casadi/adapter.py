"""
CasADi Adapter

Converts CasADi NLP formulations to OPOCH ObjectiveIR and builds
automatic differentiation functions for KKT certification.
"""

from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any, List, Tuple
import numpy as np

from .nlp_contract import CasADiNLP, NLPBounds


@dataclass
class ADFunctions:
    """Automatic differentiation functions from CasADi."""
    # Objective
    f_func: Callable[[np.ndarray], float]              # f(x)
    grad_f: Callable[[np.ndarray], np.ndarray]         # ∇f(x)

    # Constraints
    g_func: Callable[[np.ndarray], np.ndarray]         # g(x)
    jac_g: Callable[[np.ndarray], np.ndarray]          # J_g(x)

    # Lagrangian (optional)
    hess_L: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None  # ∇²L(x,λ)


class CasADiAdapter:
    """
    Adapter for CasADi NLP problems.

    Provides:
    1. AD functions for gradient/Jacobian/Hessian computation
    2. Conversion to OPOCH ObjectiveIR (ExprIR/ResidualIR/FactorIR)
    3. Evaluation interfaces for certificate computation
    """

    def __init__(self, nlp: CasADiNLP):
        """
        Initialize adapter from CasADiNLP.

        Args:
            nlp: The CasADi NLP problem specification
        """
        self.nlp = nlp
        self._ad_functions: Optional[ADFunctions] = None
        self._casadi_solver = None

        # Build AD functions if CasADi symbols are available
        if nlp.x_sym is not None and nlp.f_sym is not None:
            self._build_ad_functions()

    def _build_ad_functions(self):
        """Build CasADi AD functions for gradients and Jacobians."""
        try:
            import casadi as ca
        except ImportError:
            raise ImportError("CasADi is required: pip install casadi")

        x = self.nlp.x_sym
        f = self.nlp.f_sym
        g = self.nlp.g_sym
        p = self.nlp.p_sym

        # Build CasADi Functions
        inputs = [x] if p is None else [x, p]
        input_names = ['x'] if p is None else ['x', 'p']

        # Objective function f(x)
        f_func = ca.Function('f', inputs, [f], input_names, ['f'])

        # Gradient of objective: ∇f(x)
        grad_f_expr = ca.jacobian(f, x).T
        grad_f_func = ca.Function('grad_f', inputs, [grad_f_expr], input_names, ['grad_f'])

        # Constraint function g(x)
        g_func = ca.Function('g', inputs, [g], input_names, ['g'])

        # Jacobian of constraints: J_g(x)
        if g.shape[0] > 0:
            jac_g_expr = ca.jacobian(g, x)
            jac_g_func = ca.Function('jac_g', inputs, [jac_g_expr], input_names, ['jac_g'])
        else:
            jac_g_func = None

        # Hessian of Lagrangian: ∇²L(x, λ)
        # L(x, λ) = f(x) + λᵀg(x)
        if g.shape[0] > 0:
            lam = ca.SX.sym('lam', g.shape[0])
            L = f + ca.dot(lam, g)
            hess_L_expr = ca.hessian(L, x)[0]
            hess_inputs = [x, lam] if p is None else [x, lam, p]
            hess_names = ['x', 'lam'] if p is None else ['x', 'lam', 'p']
            hess_L_func = ca.Function('hess_L', hess_inputs, [hess_L_expr], hess_names, ['hess_L'])
        else:
            hess_L_expr = ca.hessian(f, x)[0]
            hess_L_func = ca.Function('hess_L', inputs, [hess_L_expr], input_names, ['hess_L'])

        # Wrap in Python callables
        def eval_f(x_val: np.ndarray) -> float:
            if p is None:
                return float(f_func(x_val))
            else:
                return float(f_func(x_val, self.nlp.p0))

        def eval_grad_f(x_val: np.ndarray) -> np.ndarray:
            if p is None:
                return np.array(grad_f_func(x_val)).flatten()
            else:
                return np.array(grad_f_func(x_val, self.nlp.p0)).flatten()

        def eval_g(x_val: np.ndarray) -> np.ndarray:
            if p is None:
                return np.array(g_func(x_val)).flatten()
            else:
                return np.array(g_func(x_val, self.nlp.p0)).flatten()

        def eval_jac_g(x_val: np.ndarray) -> np.ndarray:
            if jac_g_func is None:
                return np.zeros((0, len(x_val)))
            if p is None:
                return np.array(jac_g_func(x_val))
            else:
                return np.array(jac_g_func(x_val, self.nlp.p0))

        def eval_hess_L(x_val: np.ndarray, lam_val: np.ndarray) -> np.ndarray:
            if g.shape[0] > 0:
                if p is None:
                    return np.array(hess_L_func(x_val, lam_val))
                else:
                    return np.array(hess_L_func(x_val, lam_val, self.nlp.p0))
            else:
                if p is None:
                    return np.array(hess_L_func(x_val))
                else:
                    return np.array(hess_L_func(x_val, self.nlp.p0))

        self._ad_functions = ADFunctions(
            f_func=eval_f,
            grad_f=eval_grad_f,
            g_func=eval_g,
            jac_g=eval_jac_g,
            hess_L=eval_hess_L,
        )

        # Store raw CasADi functions for solver
        self._ca_f_func = f_func
        self._ca_g_func = g_func
        self._ca_grad_f = grad_f_func
        self._ca_jac_g = jac_g_func
        self._ca_hess_L = hess_L_func

    @property
    def ad(self) -> ADFunctions:
        """Get AD functions."""
        if self._ad_functions is None:
            raise RuntimeError("AD functions not built - CasADi symbols required")
        return self._ad_functions

    def eval_objective(self, x: np.ndarray) -> float:
        """Evaluate objective f(x)."""
        return self.ad.f_func(x)

    def eval_gradient(self, x: np.ndarray) -> np.ndarray:
        """Evaluate gradient ∇f(x)."""
        return self.ad.grad_f(x)

    def eval_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate constraints g(x)."""
        return self.ad.g_func(x)

    def eval_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Evaluate constraint Jacobian J_g(x)."""
        return self.ad.jac_g(x)

    def eval_lagrangian_hessian(self, x: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Evaluate Hessian of Lagrangian ∇²L(x, λ)."""
        return self.ad.hess_L(x, lam)

    def to_objective_ir(self):
        """
        Convert CasADi NLP to OPOCH ObjectiveIR.

        Detection order:
        1. Sum-of-squares → ResidualIR (for least-squares problems)
        2. Factor structure → FactorIR (for chain/separable problems)
        3. Default → ExprIR (general case)
        """
        from ..ir import ObjectiveIR, ExprIR, ResidualIR, FactorIR
        from ..expr_graph import ExpressionGraph

        # For now, create ExprIR from traced callable
        # TODO: Walk CasADi graph directly for better structure detection
        graph = self._trace_to_expr_graph()

        if self.nlp.is_least_squares:
            # Build ResidualIR if marked as least-squares
            return self._build_residual_ir()

        # Default to ExprIR
        return ExprIR(graph=graph, n_vars=self.nlp.n_vars)

    def _trace_to_expr_graph(self):
        """Trace objective to ExpressionGraph via callable."""
        from ..trace import trace_callable_to_graph

        def objective(x):
            return self.ad.f_func(x)

        try:
            graph = trace_callable_to_graph(objective, self.nlp.n_vars)
            return graph
        except Exception:
            return None

    def _build_residual_ir(self):
        """Build ResidualIR for least-squares problems."""
        from ..ir import ResidualIR

        # For least-squares, we need to extract the residual vector
        # This requires structure detection from the CasADi graph
        # For now, return a basic ResidualIR

        return ResidualIR(
            residual_graphs=[],  # Would extract from CasADi
            n_params=self.nlp.n_vars,
            n_residuals=0,
        )

    def _detect_sum_of_squares(self) -> bool:
        """Detect if objective is sum of squares f = ||r||²."""
        # TODO: Walk CasADi graph to detect sum-of-squares structure
        return self.nlp.is_least_squares

    def _detect_factor_structure(self) -> Tuple[bool, Optional[List]]:
        """Detect if objective has factor structure f = Σ f_α(x_{S_α})."""
        # TODO: Analyze CasADi graph for separability
        return False, None

    def get_nlp_dict(self) -> Dict[str, Any]:
        """Get NLP dictionary for CasADi nlpsol."""
        nlp_dict = {
            'x': self.nlp.x_sym,
            'f': self.nlp.f_sym,
            'g': self.nlp.g_sym,
        }
        if self.nlp.p_sym is not None:
            nlp_dict['p'] = self.nlp.p_sym
        return nlp_dict

    def to_problem_contract(self):
        """Convert to OPOCH ProblemContract for global optimization."""
        from ..contract import ProblemContract

        # Create bounds list
        bounds = list(zip(self.nlp.lbx, self.nlp.ubx))

        # Create constraint functions from AD
        eq_constraints = []
        ineq_constraints = []

        # Separate equality and inequality constraints based on bounds
        for i in range(self.nlp.n_constraints):
            lb = self.nlp.lbg[i]
            ub = self.nlp.ubg[i]

            if np.isclose(lb, ub):
                # Equality constraint: g_i(x) = lb
                def eq_func(x, idx=i, target=lb):
                    return self.ad.g_func(x)[idx] - target
                eq_constraints.append(eq_func)
            else:
                # Inequality constraint(s)
                if np.isfinite(lb):
                    def ineq_lb(x, idx=i, bound=lb):
                        return bound - self.ad.g_func(x)[idx]  # lb - g(x) <= 0
                    ineq_constraints.append(ineq_lb)
                if np.isfinite(ub):
                    def ineq_ub(x, idx=i, bound=ub):
                        return self.ad.g_func(x)[idx] - bound  # g(x) - ub <= 0
                    ineq_constraints.append(ineq_ub)

        return ProblemContract(
            objective=self.ad.f_func,
            bounds=bounds,
            eq_constraints=eq_constraints if eq_constraints else None,
            ineq_constraints=ineq_constraints if ineq_constraints else None,
            name=self.nlp.name,
        )
