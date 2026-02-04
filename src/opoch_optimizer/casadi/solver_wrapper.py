"""
Deterministic Solver Wrappers

Provides deterministic wrappers for CasADi-supported solvers (IPOPT, Bonmin)
with fixed options for reproducible optimization.

Key principle: Same input → Same output (deterministic replay)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import time


@dataclass
class SolverResult:
    """Result from solver execution."""
    x: np.ndarray                    # Primal solution
    f: float                         # Objective value
    g: np.ndarray                    # Constraint values
    lam_g: np.ndarray               # Constraint multipliers
    lam_x: np.ndarray               # Bound multipliers

    # Solver statistics
    success: bool
    status: str
    iterations: int
    time: float
    return_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x.tolist(),
            'f': self.f,
            'g': self.g.tolist(),
            'lam_g': self.lam_g.tolist(),
            'lam_x': self.lam_x.tolist(),
            'success': self.success,
            'status': self.status,
            'iterations': self.iterations,
            'time': self.time,
            'return_status': self.return_status,
        }


class DeterministicIPOPT:
    """
    Deterministic IPOPT wrapper for NLP solving.

    Fixed options ensure reproducibility across runs and machines.
    """

    # Fixed options for deterministic behavior
    DEFAULT_OPTIONS = {
        # Output
        'ipopt.print_level': 0,
        'print_time': False,

        # Convergence tolerances
        'ipopt.tol': 1e-8,
        'ipopt.acceptable_tol': 1e-6,
        'ipopt.max_iter': 3000,
        'ipopt.acceptable_iter': 15,

        # Linear solver (MUMPS is deterministic)
        'ipopt.linear_solver': 'mumps',

        # Barrier parameter strategy
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.mu_init': 0.1,

        # Step computation
        'ipopt.mehrotra_algorithm': 'no',  # Deterministic

        # Initialization
        'ipopt.warm_start_init_point': 'no',
        'ipopt.warm_start_bound_push': 1e-9,
        'ipopt.warm_start_mult_bound_push': 1e-9,

        # Bound handling
        'ipopt.honor_original_bounds': 'yes',
        'ipopt.check_derivatives_for_naninf': 'yes',

        # Hessian approximation (if needed)
        'ipopt.hessian_approximation': 'exact',

        # Fixed random seed for any stochastic components
        # (IPOPT is deterministic but this ensures any future changes don't break it)
    }

    def __init__(
        self,
        nlp: 'CasADiNLP',
        options: Optional[Dict[str, Any]] = None,
        adapter: Optional['CasADiAdapter'] = None
    ):
        """
        Initialize deterministic IPOPT solver.

        Args:
            nlp: CasADiNLP problem specification
            options: Override default options (use carefully)
            adapter: Optional CasADiAdapter (built if not provided)
        """
        self.nlp = nlp
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        self._solver = None

        # Build adapter if needed
        if adapter is not None:
            self.adapter = adapter
        else:
            from .adapter import CasADiAdapter
            self.adapter = CasADiAdapter(nlp)

        self._build_solver()

    def _build_solver(self):
        """Build CasADi nlpsol with fixed options."""
        try:
            import casadi as ca
        except ImportError:
            raise ImportError("CasADi is required: pip install casadi")

        nlp_dict = self.adapter.get_nlp_dict()

        self._solver = ca.nlpsol(
            'ipopt_solver',
            'ipopt',
            nlp_dict,
            self.options
        )

    def solve(self, x0: Optional[np.ndarray] = None) -> SolverResult:
        """
        Solve NLP with deterministic initial guess.

        Args:
            x0: Initial guess (uses nlp.x0 if None)

        Returns:
            SolverResult with solution and statistics
        """
        if x0 is None:
            x0 = self.nlp.x0
        x0 = np.asarray(x0).flatten()

        # Solve
        start_time = time.time()

        try:
            result = self._solver(
                x0=x0,
                lbx=self.nlp.lbx,
                ubx=self.nlp.ubx,
                lbg=self.nlp.lbg,
                ubg=self.nlp.ubg,
            )
            elapsed = time.time() - start_time

            # Extract solution
            x = np.array(result['x']).flatten()
            f = float(result['f'])
            g = np.array(result['g']).flatten()
            lam_g = np.array(result['lam_g']).flatten()
            lam_x = np.array(result['lam_x']).flatten()

            # Get solver stats
            stats = self._solver.stats()
            success = stats.get('success', False)
            return_status = stats.get('return_status', 'unknown')
            iterations = stats.get('iter_count', 0)

            return SolverResult(
                x=x,
                f=f,
                g=g,
                lam_g=lam_g,
                lam_x=lam_x,
                success=success,
                status='optimal' if success else 'failed',
                iterations=iterations,
                time=elapsed,
                return_status=return_status,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return SolverResult(
                x=x0,
                f=float('inf'),
                g=np.zeros(self.nlp.n_constraints),
                lam_g=np.zeros(self.nlp.n_constraints),
                lam_x=np.zeros(self.nlp.n_vars),
                success=False,
                status=f'error: {str(e)}',
                iterations=0,
                time=elapsed,
                return_status='exception',
            )

    def solve_with_warmstart(
        self,
        x0: np.ndarray,
        lam_g0: np.ndarray,
        lam_x0: np.ndarray
    ) -> SolverResult:
        """
        Solve with warmstart from previous solution.

        Args:
            x0: Initial primal point
            lam_g0: Initial constraint multipliers
            lam_x0: Initial bound multipliers

        Returns:
            SolverResult
        """
        # Enable warmstart
        warmstart_options = {
            **self.options,
            'ipopt.warm_start_init_point': 'yes',
        }

        try:
            import casadi as ca
        except ImportError:
            raise ImportError("CasADi is required: pip install casadi")

        # Rebuild solver with warmstart options
        nlp_dict = self.adapter.get_nlp_dict()
        warmstart_solver = ca.nlpsol(
            'ipopt_warmstart',
            'ipopt',
            nlp_dict,
            warmstart_options
        )

        start_time = time.time()

        try:
            result = warmstart_solver(
                x0=x0,
                lbx=self.nlp.lbx,
                ubx=self.nlp.ubx,
                lbg=self.nlp.lbg,
                ubg=self.nlp.ubg,
                lam_g0=lam_g0,
                lam_x0=lam_x0,
            )
            elapsed = time.time() - start_time

            x = np.array(result['x']).flatten()
            f = float(result['f'])
            g = np.array(result['g']).flatten()
            lam_g = np.array(result['lam_g']).flatten()
            lam_x = np.array(result['lam_x']).flatten()

            stats = warmstart_solver.stats()
            success = stats.get('success', False)
            return_status = stats.get('return_status', 'unknown')
            iterations = stats.get('iter_count', 0)

            return SolverResult(
                x=x,
                f=f,
                g=g,
                lam_g=lam_g,
                lam_x=lam_x,
                success=success,
                status='optimal' if success else 'failed',
                iterations=iterations,
                time=elapsed,
                return_status=return_status,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return SolverResult(
                x=x0,
                f=float('inf'),
                g=np.zeros(self.nlp.n_constraints),
                lam_g=np.zeros(self.nlp.n_constraints),
                lam_x=np.zeros(self.nlp.n_vars),
                success=False,
                status=f'error: {str(e)}',
                iterations=0,
                time=elapsed,
                return_status='exception',
            )


class DeterministicBonmin:
    """
    Deterministic Bonmin wrapper for MINLP solving.

    For mixed-integer nonlinear programs with integer variables.
    """

    DEFAULT_OPTIONS = {
        'print_level': 0,
        'bonmin.algorithm': 'B-BB',  # Branch-and-bound
        'bonmin.time_limit': 300,
        'bonmin.node_limit': 10000,
        'bonmin.integer_tolerance': 1e-6,
        'bonmin.allowable_gap': 1e-6,
        'bonmin.allowable_fraction_gap': 1e-6,
    }

    def __init__(
        self,
        nlp: 'CasADiNLP',
        options: Optional[Dict[str, Any]] = None,
        adapter: Optional['CasADiAdapter'] = None
    ):
        """
        Initialize deterministic Bonmin solver.

        Args:
            nlp: CasADiNLP problem specification (must have integer_vars)
            options: Override default options
            adapter: Optional CasADiAdapter
        """
        self.nlp = nlp
        self.options = {**self.DEFAULT_OPTIONS, **(options or {})}
        self._solver = None

        if adapter is not None:
            self.adapter = adapter
        else:
            from .adapter import CasADiAdapter
            self.adapter = CasADiAdapter(nlp)

        if nlp.is_minlp:
            self._build_solver()
        else:
            raise ValueError("Bonmin requires integer variables")

    def _build_solver(self):
        """Build CasADi nlpsol with Bonmin."""
        try:
            import casadi as ca
        except ImportError:
            raise ImportError("CasADi is required: pip install casadi")

        nlp_dict = self.adapter.get_nlp_dict()

        # Add discrete variables
        nlp_dict['discrete'] = self.nlp.integer_vars

        self._solver = ca.nlpsol(
            'bonmin_solver',
            'bonmin',
            nlp_dict,
            self.options
        )

    def solve(self, x0: Optional[np.ndarray] = None) -> SolverResult:
        """
        Solve MINLP with deterministic settings.

        Args:
            x0: Initial guess

        Returns:
            SolverResult
        """
        if x0 is None:
            x0 = self.nlp.x0
        x0 = np.asarray(x0).flatten()

        start_time = time.time()

        try:
            result = self._solver(
                x0=x0,
                lbx=self.nlp.lbx,
                ubx=self.nlp.ubx,
                lbg=self.nlp.lbg,
                ubg=self.nlp.ubg,
            )
            elapsed = time.time() - start_time

            x = np.array(result['x']).flatten()
            f = float(result['f'])
            g = np.array(result['g']).flatten()
            lam_g = np.array(result['lam_g']).flatten()
            lam_x = np.array(result['lam_x']).flatten()

            stats = self._solver.stats()
            success = stats.get('success', False)
            return_status = stats.get('return_status', 'unknown')
            iterations = stats.get('iter_count', 0)

            return SolverResult(
                x=x,
                f=f,
                g=g,
                lam_g=lam_g,
                lam_x=lam_x,
                success=success,
                status='optimal' if success else 'failed',
                iterations=iterations,
                time=elapsed,
                return_status=return_status,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            return SolverResult(
                x=x0,
                f=float('inf'),
                g=np.zeros(self.nlp.n_constraints),
                lam_g=np.zeros(self.nlp.n_constraints),
                lam_x=np.zeros(self.nlp.n_vars),
                success=False,
                status=f'error: {str(e)}',
                iterations=0,
                time=elapsed,
                return_status='exception',
            )
